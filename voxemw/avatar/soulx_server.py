"""SoulX-FlashHead 数字人渲染服务：实时音频流 → 带时间戳的 JPEG 帧流（WebSocket）。

运行环境：flashhead conda env（python 3.10 + torch 2.7.1 + flash_attn），
与语音管线（py312 env）隔离。cwd 必须是 SoulX-FlashHead 仓库根
（inference.py 用相对路径读 flash_head/configs/infer_params.yaml）。

同步架构（显式对齐，2026-08-29 定稿）：
- 编排层按【生成速度】喂音频（不 paced），首段 0.96s 音频 ~0.4s 攒齐，
  比 paced 喂法快 ~0.6s 出首帧
- 回复开始时对齐 chunk 边界（丢弃待机半段），每个 chunk 精确对应
  回复音频时间轴 [k*0.96, (k+1)*0.96]
- 每帧带 4 字节 float32 音频时间戳（秒，回复内相对时间；-1 = 待机帧即来即播），
  【浏览器】按自己的播放时钟定点放映——同步真理在播放端，不猜网络时钟
- 回复进行中音频暂时断供时【不补静音】（保持音频轴精确），嘴停在上帧

协议（客户端 ↔ 本服务）：
  → 文本 {"type":"response_start"}   新回复音频流开始（对齐 chunk 边界）
  → 文本 {"type":"audio","pcm":<base64>}  16kHz PCM16 单声道（生成速度喂入）
  → 文本 {"type":"response_end"}     回复音频流结束（尾巴补静音收嘴，回待机）
  → 文本 {"type":"flush"}            打断：三段全清，立刻回待机
  → 文本 {"type":"ping"}
  ← 二进制帧：4B float32 音频时间戳 + JPEG（512x512）
  ← 文本 {"type":"ready"|"flushed"|"pong"|"error", ...}
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import queue
import struct
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
AUDIO_Q_MAX = 64  # 快速喂法下整句可能瞬到，给足缓冲（64 chunk ≈ 2MB 音频）
IDLE_ATIME = -1.0  # 待机帧时间戳：浏览器即来即播


class RenderEngine:
    """渲染引擎：攒段/生成/输出三段流水线，帧带音频时间戳。"""

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
        self.on_frames = on_frames  # callback(bytes: 4B时间戳头 + JPEG)
        self.jpeg_quality = jpeg_quality
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=AUDIO_Q_MAX)
        self.chunk_q: queue.Queue[tuple] = queue.Queue(maxsize=2)
        self.emit_q: queue.Queue[tuple] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._flushed = threading.Event()
        self._flush_seq = 0  # 打断代际：生成完才发现代际变了就丢帧
        # 回复状态（显式对齐的核心账本）。计数器单调递增【永不归零】：
        # response_start 只记音频轴原点 _response_fed_origin——归零会和
        # 攒段线程里已读出的旧 consumed_before 打架（读出旧值→归零→
        # n_fed 算成 0→语音段被误标待机帧，实测复现过）
        self._in_response = False
        self._fed_samples = 0        # 累计已喂样本数（含语音的）
        self._consumed_samples = 0   # 累计已进 chunk 的喂入样本数
        self._response_fed_origin = 0  # 本回复音频轴原点（= response_start 时的 fed）
        logger.info("引擎加载完成 %.1fs：slice=%d 帧(%.2fs) fps=%d 上下文=%ds",
                    time.perf_counter() - t0, self.slice_len, self.chunk_seconds,
                    self.fps, self.cached_audio_duration)

    # ── 客户端消息入口 ──

    def response_start(self) -> None:
        """新回复：记音频轴原点（计数器不归零），攒段线程对齐 chunk 边界。"""
        logger.info("response_start 收到")
        self._in_response = True
        self._response_fed_origin = self._fed_samples
        # 丢弃待机积压（静音段），让回复音频立即成段
        while True:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break
        self._flushed.set()  # 攒段线程丢 pending 半块，从回复起点重开

    def response_end(self) -> None:
        """回复结束：_in_response 翻False，攒段线程自然给尾巴补静音收嘴
        （时间戳顺延，播完自然定格），随后回待机。"""
        self._in_response = False

    def feed(self, pcm: bytes) -> None:
        """喂 16k PCM16（任意长度；生成速度，不 paced）。"""
        logger.debug("feed 收到 %d 字节", len(pcm))
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self.audio_q.put_nowait(x)
            self._fed_samples += len(x)
        except queue.Full:
            logger.warning("渲染队列溢出，丢最旧（%d 段积压）", AUDIO_Q_MAX)
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                pass
            self.audio_q.put_nowait(x)
            self._fed_samples += len(x)

    def flush(self) -> None:
        """打断：三段全清——待渲染音频、待生成段、待播帧组。
        正在生成中的段靠代际戳丢弃（嘴唇立刻回待机，不播打断前的尾巴）。"""
        self._flush_seq += 1
        self._in_response = False
        dropped = 0
        for q in (self.audio_q, self.chunk_q, self.emit_q):
            while True:
                try:
                    q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
        self._flushed.set()
        logger.info("flush：丢弃 %d 个排队项", dropped)

    # ── 三段流水线 ──

    def _next_chunk(self, pending: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """攒够一个 chunk。待机：队列空补静音（呼吸/眨眼）；
        回复中：队列空【等】不补（保音频轴精确），response_end 后尾巴补静音收嘴。"""
        while len(pending) < self.chunk_samples:
            try:
                x = self.audio_q.get(timeout=0.2)
                pending = np.concatenate([pending, x])
            except queue.Empty:
                if self._stop.is_set():
                    return pending, np.zeros(0, dtype=np.float32)
                if self._in_response:
                    continue  # 回复中宁等勿补：补了音频轴就漂了
                need = min(self.chunk_samples - len(pending),
                           int(0.2 * self.sample_rate))
                pending = np.concatenate(
                    [pending, np.zeros(need, dtype=np.float32)])
        return pending[self.chunk_samples:], pending[:self.chunk_samples]

    def _accumulate_loop(self) -> None:
        """攒段线程：audio_q → 整 chunk + 每帧音频时间戳（或 None=待机）。"""
        pending = np.zeros(0, dtype=np.float32)
        while not self._stop.is_set():
            if self._flushed.is_set():
                pending = np.zeros(0, dtype=np.float32)
                self._flushed.clear()
            was_in_response = self._in_response
            consumed_before = self._consumed_samples
            pending, chunk = self._next_chunk(pending)
            if len(chunk) < self.chunk_samples:
                continue  # stop 中
            n_fed = min(len(chunk), max(0, self._fed_samples - consumed_before))
            self._consumed_samples = consumed_before + n_fed
            logger.debug("成段: n_fed=%d fed=%d consumed=%d in_resp=%s",
                        n_fed, self._fed_samples, self._consumed_samples,
                        was_in_response)
            if n_fed > 0:
                # 段内含回复音频：逐帧时间戳（回复内相对秒，原点差值）；
                # 尾巴补静音的帧顺延（收嘴动作紧跟语音尾，不排队）
                base = (consumed_before - self._response_fed_origin) / self.sample_rate
                a_times = [base + (i * self.sample_rate / self.fps) / self.sample_rate
                           for i in range(self.slice_len)]
            else:
                a_times = None  # 纯待机段
            self.chunk_q.put((chunk, a_times))  # 满则阻塞（背压）

    def _generate_loop(self) -> None:
        """生成线程：chunk → 帧组。0.25s/段 vs 0.96s/段到达，等 chunk_q 为主。"""
        from flash_head.inference import get_audio_embedding, run_pipeline

        ctx_len = self.cached_audio_duration * self.sample_rate
        audio_ctx = deque([0.0] * ctx_len, maxlen=ctx_len)
        while not self._stop.is_set():
            try:
                chunk, a_times = self.chunk_q.get(timeout=0.2)
            except queue.Empty:
                continue
            seq = self._flush_seq  # 段起点记代际：生成中途被打断则成品丢弃
            audio_ctx.extend(chunk.tolist())
            audio_array = np.array(audio_ctx)
            t0 = time.perf_counter()
            embedding = get_audio_embedding(
                self.pipeline, audio_array, self.audio_start_idx, self.audio_end_idx)
            video = run_pipeline(self.pipeline, embedding)
            video = video[self.motion_frames_num:]
            frames = video.cpu().numpy().astype(np.uint8)
            logger.debug("chunk 生成 %.2fs（%.0fx 实时）",
                         time.perf_counter() - t0,
                         self.chunk_seconds / max(time.perf_counter() - t0, 1e-3))
            if seq != self._flush_seq:
                continue  # 打断代际：这段是打断前的语音，丢
            if self.emit_q.full():
                try:
                    self.emit_q.get_nowait()  # 播不过来就丢最旧组，嘴跳新词
                except queue.Empty:
                    pass
            self.emit_q.put((frames, a_times))

    def _emit_loop(self) -> None:
        """输出线程：帧 → 4B 时间戳头 + JPEG。
        语音帧立即发（浏览器按时间戳定点放映，不需要这边匀速）；
        待机帧按 25fps 滴灌（浏览器即来即播，动画要匀）。"""
        frame_interval = 1.0 / self.fps
        while not self._stop.is_set():
            try:
                frames, a_times = self.emit_q.get(timeout=0.2)
            except queue.Empty:
                continue
            is_idle = a_times is None
            for i in range(frames.shape[0]):
                a = IDLE_ATIME if is_idle else a_times[i]
                payload = struct.pack("<f", a) + self._to_jpeg(frames[i])
                self.on_frames(payload)
                if is_idle:
                    target = time.perf_counter() + frame_interval
                    delay = target - time.perf_counter()
                    if delay > 0:
                        self._stop.wait(delay)

    def run(self) -> None:
        """三段流水线：攒段/生成/输出各吃满自己的节奏互不拖累。"""
        threads = [
            threading.Thread(target=self._accumulate_loop, daemon=True),
            threading.Thread(target=self._generate_loop, daemon=True),
            threading.Thread(target=self._emit_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

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

    def emit_frames_threadsafe(self, payload: bytes) -> None:
        """渲染线程回调：帧 → asyncio 广播。"""
        if self.loop is None or not self.clients:
            return
        for ws in list(self.clients):
            try:
                asyncio.run_coroutine_threadsafe(self._send(ws, payload), self.loop)
            except Exception:
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
                    elif etype == "response_start":
                        self.engine.response_start()
                    elif etype == "response_end":
                        self.engine.response_end()
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

    def on_frames(payload: bytes) -> None:
        if server is not None:
            server.emit_frames_threadsafe(payload)

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
