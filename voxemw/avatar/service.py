"""Avatar 积木：数字人服务（FlashHead 流式说话头）。

独立 GPU 进程。输入：参考肖像（persona 的 ref_image）+ 16kHz int16 PCM 音频流；
输出：JPEG 视频帧流（512×512，目标 25fps，随生成节奏推流）。

协议（ws，默认 127.0.0.1:8767，orchestrator 是唯一客户端）：
  入（JSON 文本帧）：
    {"type": "audio", "pcm": "<base64 int16 16k mono>"}   喂音频
    {"type": "reset"}                                       utterance 边界/打断：清缓冲、运动上下文归位
    {"type": "set_image", "path": "<服务器本地路径>"}       切换 persona 肖像（同机路径直传）
  出：
    二进制帧：JPEG 图片（一帧一条）
    JSON 文本帧：{"type": "ready"} / {"type": "error", "message": ...}

实现要点（对齐 SoulX-FlashHead gradio_app_streaming.py 的 producer 模式）：
- 推理固定消费 15360 采样（0.96s）/chunk，产出 24 帧（25fps），
  音频上下文为 8s 零填充环形缓冲，wav2vec2 每 chunk 重提一次（成本恒定）。
-  utterance 结束/打断：reset → 清输入缓冲 + pipeline.reset_person_name()
  （运动上下文归位到参考图，下一句从头起）。
- 欠载策略：没有足量真实音频时不生成（浏览器端显示静态肖像/保持末帧）；
  句尾不足一 chunk 的尾巴零填充补齐生成，让嘴型自然闭合。
- torch.compile 首次 chunk 编译很慢，启动时用静音跑 2-3 个 chunk 预热。
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
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
# FlashHead 推理代码由 scripts/autodl_setup.sh 克隆到 vendor/（gitignore）
sys.path.insert(0, str(REPO_ROOT / "vendor" / "SoulX-FlashHead"))
# flash_head 内部按 CWD 相对路径读配置（import 时 open("flash_head/configs/infer_params.yaml")），
# 与上游 app 一致切到 vendor 根目录；本服务的 config/models/assets 路径均已绝对化，不受影响
os.chdir(REPO_ROOT / "vendor" / "SoulX-FlashHead")

logger = logging.getLogger(__name__)

# ── 流式常量（来自 flash_head/configs/infer_params.yaml：frame_num=33、
#    motion_frames_latent_num=2、tgt_fps=25、sample_rate=16000、cached_audio_duration=8）──
SAMPLE_RATE = 16000
TGT_FPS = 25
CACHE_SECONDS = 8
# 单位是"视频帧":8s 上下文 × 25fps = 200 帧（wav2vec 路径每秒音频映射为 fps 帧 embedding）
AUDIO_END_IDX = CACHE_SECONDS * TGT_FPS                 # 200
FRAME_NUM = 33
AUDIO_START_IDX = AUDIO_END_IDX - FRAME_NUM             # 167
MOTION_FRAMES_NUM = 9                                   # (2-1)*8+1（Lite VAE 时间 stride 8）
CHUNK_SAMPLES = (FRAME_NUM - MOTION_FRAMES_NUM) * SAMPLE_RATE // TGT_FPS  # 15360 = 0.96s
WARMUP_CHUNKS = 2


class AvatarEngine:
    """FlashHead 推理引擎：所有 pipeline 调用都序列化在 inference 线程里。"""

    def __init__(self, model_dir: str, wav2vec_dir: str, image_path: str, seed: int = 9999):
        import numpy as np  # noqa: F401
        from flash_head.inference import (
            get_base_data,
            get_infer_params,
            get_pipeline,
        )

        logger.info("加载 FlashHead Lite: %s", model_dir)
        self.pipeline = get_pipeline(
            world_size=1,
            ckpt_dir=model_dir,
            model_type="lite",
            wav2vec_dir=wav2vec_dir,
        )
        params = get_infer_params()
        assert params["sample_rate"] == SAMPLE_RATE and params["tgt_fps"] == TGT_FPS, (
            f"infer_params 与本服务常量不一致: {params}"
        )
        self.seed = seed
        self.image_path = image_path
        get_base_data(self.pipeline, cond_image_path_or_dir=image_path,
                      base_seed=seed, use_face_crop=False)
        self._new_audio_dq()

        # 输入累积缓冲 + 条件变量（feed_audio 生产，inference 线程消费）
        import numpy as _np
        self._pending = _np.empty(0, dtype=_np.float32)
        self._cond = threading.Condition()
        self._closed = False
        self._reset_motion = False
        self._pending_image = None
        self.on_frames = None  # 单客户端设计：ws 连接建立时接管帧流

    def _new_audio_dq(self):
        self.audio_dq = deque([0.0] * (SAMPLE_RATE * CACHE_SECONDS),
                              maxlen=SAMPLE_RATE * CACHE_SECONDS)

    # ── 生产侧（ws 线程调用，只碰累积缓冲）──

    def feed_audio(self, pcm_f32) -> None:
        with self._cond:
            import numpy as np

            self._pending = np.concatenate([self._pending, pcm_f32])
            self._cond.notify()

    def reset(self) -> None:
        """utterance 边界/打断：丢弃未消费音频，标记运动上下文归位。"""
        with self._cond:
            import numpy as np

            self._pending = np.empty(0, dtype=np.float32)
            self._reset_motion = True
            self._cond.notify()

    def set_image(self, image_path: str) -> None:
        with self._cond:
            self._pending_image = image_path
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()

    # ── 消费侧（inference 线程）──

    def run_inference_loop(self, on_frames) -> None:
        """阻塞循环：凑满一 chunk 真实音频才生成；0.5s 无新音频且缓冲有残留
        → 零填充补齐最后一 chunk（句尾嘴型自然闭合）。
        on_frames(frames_uint8: np.ndarray (24,512,512,3)) 每 chunk 回调一次。"""
        import numpy as np
        from flash_head.inference import get_audio_embedding, run_pipeline

        while True:
            with self._cond:
                while not self._closed and len(self._pending) < CHUNK_SAMPLES:
                    notified = self._cond.wait(timeout=0.5)
                    if not notified and len(self._pending) > 0:
                        break  # 静默超时：句尾尾巴零填充生成
                if self._closed:
                    return
                chunk = self._pending[:CHUNK_SAMPLES]
                self._pending = self._pending[CHUNK_SAMPLES:]
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                if self._reset_motion:
                    self._new_audio_dq()
                    self.pipeline.reset_person_name()
                    self._reset_motion = False
                if self._pending_image:
                    from flash_head.inference import get_base_data

                    logger.info("切换数字人肖像: %s", self._pending_image)
                    get_base_data(self.pipeline, cond_image_path_or_dir=self._pending_image,
                                  base_seed=self.seed, use_face_crop=False)
                    self._new_audio_dq()
                    self._pending_image = None
            self.audio_dq.extend(chunk.tolist())
            emb = get_audio_embedding(
                self.pipeline, np.array(self.audio_dq), AUDIO_START_IDX, AUDIO_END_IDX
            )
            video = run_pipeline(self.pipeline, emb)
            frames = video[MOTION_FRAMES_NUM:].cpu().numpy().astype("uint8")
            on_frames(frames)

    def warmup(self, on_frames) -> None:
        import numpy as np

        logger.info("FlashHead 预热（torch.compile 首个 chunk 很慢）...")
        for _ in range(WARMUP_CHUNKS):
            self.feed_audio(np.zeros(CHUNK_SAMPLES, dtype=np.float32))
        # 直接同步跑循环直到预热 chunk 消费完
        while len(self._pending) > 0:
            threading.Event().wait(0.1)
        logger.info("FlashHead 预热完成")


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


async def _serve(ws, engine: AvatarEngine, jpeg_quality: int) -> None:
    """单个 orchestrator 连接：收音频/控制消息，推 JPEG 帧。"""
    import queue as _queue

    import numpy as np

    loop = asyncio.get_running_loop()
    out_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=TGT_FPS * 4)
    raw_queue: _queue.Queue = _queue.Queue(maxsize=TGT_FPS * 2)

    def on_frames(frames) -> None:
        # 推理线程只入队原始帧（满则丢最旧）；JPEG 编码在专用线程，
        # 避免每 chunk 24 帧的编码耗时（q92 约 0.2-0.3s）阻塞下一 chunk 生成
        for frame in frames:
            if raw_queue.full():
                try:
                    raw_queue.get_nowait()
                except _queue.Empty:
                    pass
            raw_queue.put_nowait(frame)

    def _encoder() -> None:
        while True:
            frame = raw_queue.get()
            data = _encode_jpeg(frame, jpeg_quality)
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
            elif etype == "reset":
                engine.reset()
            elif etype == "set_image":
                engine.set_image(event["path"])
    finally:
        send_task.cancel()
        engine.on_frames = None


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 数字人服务（FlashHead）")
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

    model_dir = str(REPO_ROOT / avatar.get("model_dir", "models/SoulX-FlashHead-1_3B"))
    wav2vec_dir = str(REPO_ROOT / avatar.get("wav2vec_dir", "models/wav2vec2-base-960h"))
    jpeg_quality = int(avatar.get("jpeg_quality", 80))
    host = str(avatar.get("host", "127.0.0.1"))
    port = int(avatar.get("port", 8767))

    engine = AvatarEngine(model_dir, wav2vec_dir, image)

    import websockets

    async def _main() -> None:
        def on_frames(frames) -> None:
            cb = engine.on_frames
            if cb:
                cb(frames)

        thread = threading.Thread(target=engine.run_inference_loop, args=(on_frames,), daemon=True)
        thread.start()
        engine.warmup(on_frames)
        async with websockets.serve(
            lambda ws: _serve(ws, engine, jpeg_quality), host, port
        ):
            logger.info("数字人服务就绪: ws://%s:%d", host, port)
            await asyncio.Future()  # 永久运行

    asyncio.run(_main())


if __name__ == "__main__":
    main()
