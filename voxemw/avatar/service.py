"""Avatar 积木：数字人服务（AVTR-1 流式说话头，TensorRT）。

独立 GPU 进程。输入：参考肖像（persona 的 ref_image，16:9 横版胸像）+
16kHz int16 PCM 音频流；输出：JPEG 视频帧流（1280×720，25fps 节奏）。
引擎实现见 voxemw/avatar/avtr1_engine.py（须在 pixi env 直调启动，勿 pixi run
——会按 lock 重同步 env 覆盖 pip 降级；start_assistant.sh 已内置）。

协议（ws，默认 127.0.0.1:8767，orchestrator 是唯一客户端）：
  入（JSON 文本帧）：
    {"type": "audio", "pcm": "<base64 int16 16k mono>"}   TTS 音频（speech 轨）
    {"type": "listen", "pcm": "<base64 int16 16k mono>"}  用户麦克风音频（listen 轨，
                                                          active listening）
    {"type": "reset"}                                       打断：清音频缓冲（运动上下文保留，
                                                          对齐官方 interrupt 语义）
    {"type": "set_image", "path": "<服务器本地路径>"}       切换 persona 肖像（同机路径直传）
    {"type": "speech_active", "on": true|false}             助手说话期间禁 idle 生成
    {"type": "idle_mode", "mode": "listening"|"thinking"|"calm"}
                                                            待机模式（listening 时启用 listen 轨）
  出：
    二进制帧：tag(1B) + JPEG 图片（一帧一条）。tag：0x00=idle（待机微动/倾听反应）、
              0x01=speech（真实音频驱动）；前端合流同队列沿音频时钟连播
    JSON 文本帧：{"type": "ready"} / {"type": "error", "message": ...}
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

TGT_FPS = 25

# 下行帧 tag（见模块 docstring 协议）
FRAME_TAG_IDLE = 0x00    # 待机微动 / 倾听反应
FRAME_TAG_SPEECH = 0x01  # 真实音频驱动（含句尾淡出闭嘴帧）


def _encode_jpeg(frame_rgb, quality: int) -> bytes:
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    # 轻量锐化(unsharp mask):找回 JPEG 压缩丢掉的边缘,半径小强度低不过曝
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    bgr = cv2.addWeighted(bgr, 1.35, blur, -0.35, 0)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


async def _serve(ws, engine, jpeg_quality: int) -> None:
    """单个 orchestrator 连接：收音频/控制消息，推 JPEG 帧。"""
    import queue as _queue

    import numpy as np

    loop = asyncio.get_running_loop()
    out_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=TGT_FPS * 4)
    raw_queue: _queue.Queue = _queue.Queue(maxsize=TGT_FPS * 2)
    stat = {"t0": time.monotonic(), "produced": 0, "dropped": 0}  # 产出自检
    # raw 模式（orchestrator RTC 链路协商开启）：跳过 JPEG 编解码，
    # 直接发 tag + RGB 裸帧（loopback 带宽无压力，省两端 CPU）
    raw_mode = {"on": False}

    def on_frames(frames, is_idle: bool) -> None:
        # 推理线程只入队原始帧（满则丢最旧）；JPEG 编码在专用线程，
        # 避免编码耗时阻塞下一 chunk 生成
        tag = FRAME_TAG_IDLE if is_idle else FRAME_TAG_SPEECH
        for frame in frames:
            if raw_queue.full():
                try:
                    raw_queue.get_nowait()
                    stat["dropped"] += 1
                except _queue.Empty:
                    pass
            raw_queue.put_nowait((frame, tag))
            stat["produced"] += 1
        now = time.monotonic()
        if now - stat["t0"] >= 10:
            logger.info("帧产出: %.1f fps（10s 产 %d，挤队丢 %d）",
                        stat["produced"] / (now - stat["t0"]),
                        stat["produced"], stat["dropped"])
            stat["t0"], stat["produced"], stat["dropped"] = now, 0, 0

    def _encoder() -> None:
        while True:
            frame, tag = raw_queue.get()
            if raw_mode["on"]:
                data = bytes([tag]) + frame.tobytes()
            else:
                data = bytes([tag]) + _encode_jpeg(frame, jpeg_quality)
            loop.call_soon_threadsafe(_offer, data)

    def _offer(data: bytes) -> None:
        # 队列满则丢最旧帧（视频允许丢帧，音频不允许）
        if out_queue.full():
            try:
                out_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        out_queue.put_nowait(data)

    engine.on_frames = on_frames  # 单客户端设计：最后一个连接接管帧流
    threading.Thread(target=_encoder, daemon=True).start()

    async def sender() -> None:
        while True:
            data = await out_queue.get()
            await ws.send(data)

    send_task = asyncio.create_task(sender())
    await ws.send(json.dumps({"type": "ready"}))
    try:
        async for message in ws:
            if not isinstance(message, str):
                continue
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "audio":
                pcm = np.frombuffer(base64.b64decode(event["pcm"]), dtype=np.int16)
                engine.feed_audio(pcm.astype(np.float32) / 32768.0)
            elif etype == "raw_frames":
                raw_mode["on"] = bool(event.get("on"))
            elif etype == "listen":
                # 用户麦克风音频（active listening）
                pcm = np.frombuffer(base64.b64decode(event["pcm"]), dtype=np.int16)
                engine.feed_listen(pcm.astype(np.float32) / 32768.0)
            elif etype == "reset":
                engine.reset()
            elif etype == "set_image":
                engine.set_image(event["path"])
            elif etype == "speech_active":
                engine.set_speech_active(bool(event.get("on")))
            elif etype == "idle_mode":
                engine.set_idle_mode(str(event.get("mode", "calm")))
    finally:
        send_task.cancel()
        # 只清自己的回调：旧连接断开若误清，会把新连接刚接管的帧流杀死
        #（刷新页面顶会话的时序：新连接先接管，旧连接 finally 后跑）
        if engine.on_frames is on_frames:
            engine.on_frames = None


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 数字人服务（AVTR-1）")
    parser.add_argument("--config", default=os.environ.get("VOXEMW_CONFIG", "configs/assistant.yaml"))
    args = parser.parse_args()

    from voxemw.config import load_config, load_dotenv

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(config_path)

    avatar = config["avatar"]
    if not avatar.get("enabled", True):
        sys.exit("avatar.enabled=false，数字人服务无需启动")

    personas = config["personas"]["resolved"]
    default = personas[config["personas"]["default"]]
    image = default.get("ref_image")
    if not image:
        sys.exit(f"默认 persona 缺 ref_image，无法启动数字人服务: {config['personas']['default']}")

    jpeg_quality = int(avatar.get("jpeg_quality", 80))
    host = str(avatar.get("host", "127.0.0.1"))
    port = int(avatar.get("port", 8767))

    from voxemw.avatar.avtr1_engine import AVTR1Engine

    engine = AVTR1Engine(
        image,
        storage=avatar.get("avtr1_storage") or None,
        bg_id=str(avatar.get("avtr1_bg", "plain_white")),
        idle_motion=bool(avatar.get("idle_motion", True)),
        cfg_self_audio=float(avatar.get("avtr1_cfg_self_audio", 2.0)),
    )

    import websockets

    async def _main() -> None:
        def on_frames(frames, is_idle: bool) -> None:
            cb = engine.on_frames
            if cb:
                cb(frames, is_idle)

        thread = threading.Thread(target=engine.run_inference_loop, args=(on_frames,), daemon=True)
        thread.start()
        engine.warmup(on_frames)
        async with websockets.serve(
            lambda ws: _serve(ws, engine, jpeg_quality), host, port,
            # 裸帧 2.76MB/帧×25fps：默认开启的 permessage-deflate 压缩根本压不动
            #（实测发送端只剩 ~10fps，缓冲堆出 30-60s 延迟），必须关
            compression=None,
        ):
            logger.info("数字人服务就绪: ws://%s:%d", host, port)
            await asyncio.Future()  # 永久运行

    asyncio.run(_main())


if __name__ == "__main__":
    main()
