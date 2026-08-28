"""SoulX-FlashHead 数字人渲染服务：实时音频流 → JPEG 视频帧流（WebSocket）。

运行环境：flashhead conda env（python 3.10 + torch 2.7.1 + flash_attn），
与语音管线（py312 env）隔离。cwd 必须是 SoulX-FlashHead 仓库根
（inference.py 用相对路径读 flash_head/configs/infer_params.yaml）。

协议（客户端 ↔ 本服务）：
  → 文本 {"type":"audio","pcm":<base64>}  16kHz PCM16 单声道；
    由编排层按【播放时刻】paced 喂入（不是生成时刻），本服务天然按实时节奏渲染
  → 文本 {"type":"flush"}                 打断：丢弃待渲染音频，回待机
  → 文本 {"type":"ping"}
  ← 二进制帧：JPEG（512x512，生成即推，~25fps）
  ← 文本 {"type":"ready"|"flushed"|"pong"|"error", ...}

渲染节奏：每 chunk = slice_len 帧（0.96s 音频 @25fps），Lite 生成 ~0.27s/chunk，
渲染循环按 chunk 时长对齐实时（生成完睡到 chunk 边界），GPU 占用 ~28%，
音频到达滞后/打断由队列深度自适应。无音频时喂静音 → 待机呼吸/眨眼。
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import deque

SOULX_ROOT = os.environ.get("SOULX_ROOT", "/root/autodl-tmp/SoulX-FlashHead")
sys.path.insert(0, SOULX_ROOT)
os.chdir(SOULX_ROOT)  # infer_params.yaml 是相对路径

import numpy as np  # noqa: E402

logger = logging.getLogger("soulx_server")

PIPELINE_SR = 16000
JPEG_QUALITY = 80
AUDIO_Q_MAX = 8  # 待渲染 chunk 上限（≈8s），溢出说明上游没 paced，丢最旧


class RenderEngine:
    """同步渲染引擎（独立线程跑）：音频 chunk 队列 → JPEG 帧回调。"""

    def __init__(self, cond_image: str, ckpt: str, wav2vec: str,
                 on_frames, jpeg_quality: int = JPEG_QUALITY):
        from flash_head.inference import (
            get_base_data,
            get_infer_params,
            get_pipeline,
        )

        t0 = time.perf_counter()
        self.pipeline = get_pipeline(world_size=1, ckpt_dir=ckpt,
                                     model_type="lite", wav2vec_dir=wav2vec)
        get_base_data(self.pipeline, cond_image_path_or_dir=cond_image,
                      base_seed=9999, use_face_crop=False)
        params = get_infer_params()
        self.sample_rate = params["sample_rate"]
        self.fps = params["tgt_fps"]
        self.cached_audio_duration = params["cached_audio_duration"]
        self.slice_len = params["frame_num"] - params["motion_frames_num"]
        self.chunk_samples = self.slice_len * self.sample_rate // self.fps
        self.chunk_seconds = self.chunk_samples / self.sample_rate
        self.motion_frames_num = params["motion_frames_num"]
        self.audio_start_idx = self.cached_audio_duration * self.fps - params["frame_num"]
        self.audio_end_idx = self.cached_audio_duration * self.fps
        self.on_frames = on_frames  # callback(jpeg_bytes)
        self.jpeg_quality = jpeg_quality
        self.audio_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=AUDIO_Q_MAX)
        self._stop = threading.Event()
        self._flushed = threading.Event()
        logger.info("引擎加载完成 %.1fs：slice=%d 帧(%.2fs) fps=%d 上下文=%ds",
                    time.perf_counter() - t0, self.slice_len, self.chunk_seconds,
                    self.fps, self.cached_audio_duration)

    def feed(self, pcm: bytes) -> None:
        """喂 16k PCM16（任意长度，内部按 chunk 切）。"""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self.audio_q.put_nowait(x)
        except queue.Full:
            logger.warning("渲染队列溢出，丢最旧 chunk（上游喂太快？）")
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                pass
            self.audio_q.put_nowait(x)

    def flush(self) -> None:
        """打断：清空待渲染音频。正在生成中的 chunk 帧照出（0.27s 尾巴，可接受）。"""
        dropped = 0
        while True:
            try:
                self.audio_q.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        self._flushed.set()  # run 循环顺带丢弃 pending 半块
        logger.info("flush：丢弃 %d 个待渲染块", dropped)

    def _next_chunk(self, pending: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """从队列攒够一个 chunk 的样本；队列空则补静音（待机动画）。"""
        while len(pending) < self.chunk_samples:
            try:
                x = self.audio_q.get(timeout=0.2)
                pending = np.concatenate([pending, x])
            except queue.Empty:
                if self._stop.is_set():
                    return pending, np.zeros(0, dtype=np.float32)
                # 待机只补 0.2s 小片静音（而非整 chunk），真实语音到达
                # 最坏 0.2s 就能抢占待机动画，不整 chunk 陪绑
                need = min(self.chunk_samples - len(pending),
                           int(0.2 * self.sample_rate))
                pending = np.concatenate(
                    [pending, np.zeros(need, dtype=np.float32)])
        return pending[self.chunk_samples:], pending[:self.chunk_samples]

    def run(self) -> None:
        from flash_head.inference import get_audio_embedding, run_pipeline

        ctx_len = self.cached_audio_duration * self.sample_rate
        audio_ctx = deque([0.0] * ctx_len, maxlen=ctx_len)
        pending = np.zeros(0, dtype=np.float32)
        while not self._stop.is_set():
            if self._flushed.is_set():
                pending = np.zeros(0, dtype=np.float32)
                self._flushed.clear()
            cycle0 = time.perf_counter()
            pending, chunk = self._next_chunk(pending)
            if len(chunk) < self.chunk_samples:
                continue  # stop 中
            audio_ctx.extend(chunk.tolist())
            audio_array = np.array(audio_ctx)
            embedding = get_audio_embedding(
                self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
            video = run_pipeline(self.pipeline, embedding)
            video = video[self.motion_frames_num:]
            frames = video.cpu().numpy().astype(np.uint8)
            gen_s = time.perf_counter() - cycle0
            logger.debug("chunk 生成 %.2fs（%.0fx 实时）",
                         gen_s, self.chunk_seconds / max(gen_s, 1e-3))
            for i in range(frames.shape[0]):
                self.on_frames(self._to_jpeg(frames[i]))
            # 对齐实时节奏：生成快于 chunk 时长就睡到边界（防 3.8x 速灌帧+吃满 GPU）
            lag = self.chunk_seconds - (time.perf_counter() - cycle0)
            if lag > 0:
                self._stop.wait(lag)

    @staticmethod
    def _to_jpeg(frame: np.ndarray) -> bytes:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, "JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()

    def stop(self) -> None:
        self._stop.set()


class AvatarServer:
    def __init__(self, engine: RenderEngine):
        self.engine = engine
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def emit_frames_threadsafe(self, jpeg: bytes) -> None:
        """渲染线程回调：帧 → asyncio 广播队列。"""
        if self.loop is None or not self.clients:
            return
        dead = []
        for ws in list(self.clients):
            try:
                # 帧率 25，逐帧 send  futures 不管完成（UDP 式丢得起）
                asyncio.run_coroutine_threadsafe(self._send(ws, jpeg), self.loop)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    @staticmethod
    async def _send(ws, data: bytes) -> None:
        try:
            await ws.send(data)
        except Exception:
            pass

    async def handler(self, ws) -> None:
        self.clients.add(ws)
        peer = getattr(ws, "remote_address", "?")
        logger.info("客户端接入 %s（共 %d）", peer, len(self.clients))
        try:
            await ws.send(json.dumps({"type": "ready",
                                      "fps": self.engine.fps,
                                      "chunk_seconds": self.engine.chunk_seconds}))
            async for message in ws:
                if isinstance(message, str):
                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "audio":
                        self.engine.feed(base64.b64decode(event["pcm"]))
                    elif etype == "flush":
                        self.engine.flush()
                        await ws.send(json.dumps({"type": "flushed"}))
                    elif etype == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
        finally:
            self.clients.discard(ws)
            logger.info("客户端离开 %s（剩 %d）", peer, len(self.clients))

    async def serve(self, host: str, port: int) -> None:
        from websockets.asyncio.server import serve

        self.loop = asyncio.get_running_loop()
        async with serve(self.handler, host, port, max_size=16 * 1024 * 1024):
            logger.info("SoulX 渲染服务就绪: ws://%s:%d", host, port)
            await asyncio.Event().wait()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SoulX-FlashHead 数字人渲染服务")
    parser.add_argument("--cond-image",
                        default="/root/voxemw/assets/henannier/face_ref.jpg")
    parser.add_argument("--ckpt", default="/root/autodl-tmp/models/SoulX")
    parser.add_argument("--wav2vec", default="/root/autodl-tmp/models/wav2vec2-base-960h")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    server: AvatarServer | None = None

    def on_frames(jpeg: bytes) -> None:
        if server is not None:
            server.emit_frames_threadsafe(jpeg)

    engine = RenderEngine(args.cond_image, args.ckpt, args.wav2vec, on_frames)
    server = AvatarServer(engine)

    render_thread = threading.Thread(target=engine.run, daemon=True)
    render_thread.start()
    try:
        asyncio.run(server.serve(args.host, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
