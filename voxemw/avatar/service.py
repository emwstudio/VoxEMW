"""Avatar 积木：数字人服务（AVTR-1 流式说话头，TensorRT）。

独立 GPU 进程。输入：参考肖像（persona 的 ref_image，16:9 横版胸像）+
16kHz int16 PCM 音频流；输出：裸 RGB 视频帧流（1280×720，25fps 节奏）。
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
    二进制帧：tag(1B) + 裸 RGB（1280×720×3 字节，一帧一条）。tag：0x00=idle
              （待机微动/倾听反应）、0x01=speech（真实音频驱动）
    JSON 文本帧：{"type": "ready"} / {"type": "error", "message": ...}

  注：裸帧直发是踩坑后的选择——JPEG 编解码双端白烧 CPU；ws 默认的
  permessage-deflate 压缩压不动 69MB/s 裸流（吞吐塌到 10fps、缓冲堆积
  出 30-60s 延迟），必须 compression=None（orchestrator 侧同样设置）。
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


async def _serve(ws, engine) -> None:
    """单个 orchestrator 连接：收音频/控制消息，推裸 RGB 帧。"""
    import queue as _queue

    import numpy as np

    loop = asyncio.get_running_loop()
    out_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=TGT_FPS * 4)
    raw_queue: _queue.Queue = _queue.Queue(maxsize=TGT_FPS * 2)
    stat = {"t0": time.monotonic(), "produced": 0, "dropped": 0}  # 产出自检

    def on_frames(frames, is_idle: bool) -> None:
        # 推理线程只入队原始帧（满则丢最旧）；转发在专用线程，
        # 避免耗时阻塞下一 chunk 生成
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
            data = bytes([tag]) + frame.tobytes()
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
            elif etype == "set_idle_motion":
                # 热切换常驻微动。启动期 GPU 共存约束（2026-08-17 实测）：
                # avatar 持续渲染时 pipeline 的 VoxCPM conv1d 初始化必炸
                # illegal memory access——故启动脚本先以 idle_motion=false 起
                # avatar，pipeline 就绪后经此消息开回 true。运行期共存无碍。
                engine.idle_motion = bool(event.get("on"))
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

    host = str(avatar.get("host", "127.0.0.1"))
    port = int(avatar.get("port", 8767))

    from voxemw.avatar.avtr1_engine import AVTR1Engine

    # AVTR_IDLE_MOTION=0 覆盖配置：启动脚本在 pipeline 初始化窗口关微动，
    # 避免 avatar 持续渲染与 VoxCPM conv1d 初始化在 GPU 上互踩（2026-08-17 实测）
    idle_motion = bool(avatar.get("idle_motion", True))
    if os.environ.get("AVTR_IDLE_MOTION") == "0":
        idle_motion = False

    engine = AVTR1Engine(
        image,
        storage=avatar.get("avtr1_storage") or None,
        bg_id=str(avatar.get("avtr1_bg", "plain_white")),
        idle_motion=idle_motion,
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
            lambda ws: _serve(ws, engine), host, port,
            # 裸帧 2.76MB/帧×25fps：默认开启的 permessage-deflate 压缩根本压不动
            #（实测发送端只剩 ~10fps，缓冲堆出 30-60s 延迟），必须关
            compression=None,
        ):
            logger.info("数字人服务就绪: ws://%s:%d", host, port)
            await asyncio.Future()  # 永久运行

    asyncio.run(_main())


if __name__ == "__main__":
    main()
