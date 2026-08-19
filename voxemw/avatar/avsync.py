"""音画同步调度器：把 TTS 音频流与 avatar 帧流对齐到同一条单调时钟（纯逻辑，可单测）。

对齐原理（对标 AVTR-1 官方 demo worklets/rendering.py 的时间戳模型）：
- avatar 引擎每消费 0.2s（3200 采样@16k）音频输出 5 帧
  → 第 k 帧（0 起）对应音频第 k*640 采样，帧序号 = 音频时间轴
- avatar 在 TTS 生成时刻就拿到音频（RTF<1，帧提前渲染好，前瞻窗口有着落）；
  展示侧由播放时钟门控：第 k 帧只在已有 k*640 采样被音频轨真正取走
  （= 正在扬声器播放）后才放行，超前则重复上一帧等音频追上
- WebRTC 音频轨/视频轨分别按 20ms / 40ms 实时节奏从本调度器取数据；
  浏览器按 RTP/RTCP 时间戳原生同步音画
- 句尾零填充闭嘴帧（is_speech 但序号超出实喂音频时长）在此丢弃——
  它对应不存在的音频，WS 时代靠前端裁剪，现收敛到服务端

打断（barge-in）：flush() 清队列、游标归零（avatar reset 后音频索引重新计数）。
两条轨的播放游标在 RTC track 侧各自连续推进，flush 不动它们——
清队列后新音频/新帧从当前时刻续播，对齐天然保持。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

SAMPLES_PER_FRAME = 640          # 16kHz / 25fps，一帧对应 640 采样
AUDIO_TICK_SAMPLES = 320         # 音频轨每次取用量：20ms @16k
FRAME_TICK_SECONDS = 1 / 25      # 视频轨节奏：25fps

DEFAULT_AUDIO_LEAD = 0.20        # 新回复音频压后秒数（见 feed_audio 注释；实测 0.20 口型最准）


class AVSyncScheduler:
    """音频字节队列 + 帧队列 + 丢尾帧规则。全部方法在 orchestrator 单 loop 内调用。

    音画对齐关键：avatar 渲染有 ~0.35s 固有滞后（0.2s 攒块 + 0.2s 前瞻 + 生成），
    音频若到达即播，每段回复开头嘴都比声音慢半拍（WS 时代前端用
    AVATAR_AUDIO_DELAY=0.35s 补偿）。这里在服务端压后音频：队列从空到非空
    （新回复/打断后）时，音频等 lead 秒再播，让渲染追上来。
    连续积压的多段回复队列不空，无额外等待。
    """

    def __init__(self, audio_lead: float = DEFAULT_AUDIO_LEAD) -> None:
        self._audio = bytearray()
        self._audio_samples_fed = 0   # 累计喂入采样（与喂 avatar 的是同一股流）
        self._frames: deque[tuple[bytes, bool]] = deque()  # (jpeg, is_speech)
        self._speech_out = 0          # 已交付显示的 speech 帧数（帧序号游标）
        self._last_jpeg: bytes | None = None
        self._frame_event = asyncio.Event()
        self._closed = False
        self._audio_lead = audio_lead
        self._audio_ready_at = 0.0    # 音频队列可播放的最早时刻（单调钟）
        self._suppress_speech = False  # flush 后封杀在途陈旧 speech 帧
        self.tail_dropped = 0         # 丢尾帧计数（排障观测）
        self.stale_dropped = 0        # 陈旧帧封杀计数（排障观测）
        # 已真实播出去的采样数（RTC 音频轨每取一拍真音频 +=320）。
        # speech 帧的展示门控用播放时钟而不是喂入时钟：TTS 生成快于播放
        #（RTF<1），按喂入量放行会让嘴型超前于扬声器在播的内容
        #（2026-08-19 良子长文本嘴快进/提前结束的根因）
        self._audio_samples_played = 0

    # ── 生产者（Session 转发协程调用）──

    def feed_audio(self, pcm: bytes) -> None:
        """TTS PCM（int16 16k mono），与喂给 avatar 的同一份字节。"""
        if not self._audio:
            # 队列从空到非空 = 新回复开始：压后 lead 秒等 avatar 渲染追赶
            self._audio_ready_at = time.monotonic() + self._audio_lead
        self._suppress_speech = False  # 新音频到 → 其后续 speech 帧合法，解除封杀
        self._audio.extend(pcm)
        self._audio_samples_fed += len(pcm) // 2

    def feed_frame(self, jpeg: bytes, is_speech: bool) -> None:
        if is_speech and self._suppress_speech:
            # 打断后 avatar 服务队列里还在路上的陈旧 speech 帧（reset 前渲染的），
            # 一律丢弃——否则会触发丢尾规则误清空队列（误杀新回复的真帧）
            self.stale_dropped += 1
            return
        # 无消费者（RTC 未建连/断开）时防无限堆积：裸帧 2.76MB/帧，
        # 不加帽就是内存泄漏。超帽丢最旧——反正没人看
        while len(self._frames) >= 125:
            self._frames.popleft()
        self._frames.append((jpeg, is_speech))
        self._frame_event.set()

    def flush(self) -> None:
        """打断：清队列、游标归零。_last_jpeg 保留——新帧到达前画面不黑屏。"""
        self._audio.clear()
        self._audio_samples_fed = 0
        self._audio_samples_played = 0
        self._frames.clear()
        self._speech_out = 0
        self._suppress_speech = True  # 封杀在途陈旧 speech 帧，直到新音频到达

    def drop_stale_frames(self) -> None:
        """RTC 视频轨启动时清积压帧，只留最新一帧——ICE 协商/建连期间
        avatar 的 idle 帧会持续积进队列，produce 从队头取会永远落后
        积压时长（实测积到 121 帧 = 画面比声音慢 ~5s，2026-08-19）。"""
        if self._frames:
            self._frames = deque([self._frames[-1]])

    def close(self) -> None:
        """会话结束：唤醒阻塞中的取帧协程退出。"""
        self._closed = True
        self._frame_event.set()

    # ── 消费者（RTC track recv 调用）──

    def next_audio_tick(self) -> bytes:
        """取 20ms（320 采样）16k PCM；不足一整拍或尚在压后等待期则补静音。"""
        need = AUDIO_TICK_SAMPLES * 2
        if len(self._audio) >= need and time.monotonic() >= self._audio_ready_at:
            out = bytes(self._audio[:need])
            del self._audio[:need]
            self._audio_samples_played += AUDIO_TICK_SAMPLES
            return out
        return b"\x00" * need

    async def next_frame_tick(self) -> bytes | None:
        """取下一显示帧（JPEG 字节）。队空重复上一帧（防定格）；close 后返回 None。

        speech 帧按播放时钟放行：第 k 帧（0 起）对应音频第 k*640 采样，
        只有当这些采样已被音频轨真正取走（= 即将/正在扬声器播放）才弹出；
        超前则重复上一帧等音频追上。avatar 在 TTS 生成时刻就拿到音频
        （RTF<1，帧提前渲染好），靠这道门把展示时刻对齐到播放时刻。
        """
        while True:
            if self._closed:
                return None
            while self._frames:
                jpeg, is_speech = self._frames[0]
                if is_speech:
                    if self._speech_out >= self._audio_samples_fed // SAMPLES_PER_FRAME:
                        # 零填充闭嘴尾帧：只丢这一帧，绝不清队列
                        #（清队列会把排在后面的新回复真帧一起误杀，实测踩过）
                        self._frames.popleft()
                        self.tail_dropped += 1
                        continue
                    if self._speech_out * SAMPLES_PER_FRAME > self._audio_samples_played:
                        break  # 帧超前于已播音频 → 等音频时钟（走下方限时等待）
                    self._speech_out += 1
                self._frames.popleft()
                self._last_jpeg = jpeg
                return jpeg
            self._frame_event.clear()
            if self._last_jpeg is not None:
                # 等一小拍看有没有新帧，没有就重复上一帧
                try:
                    await asyncio.wait_for(self._frame_event.wait(), FRAME_TICK_SECONDS)
                except asyncio.TimeoutError:
                    return self._last_jpeg
            elif self._frames:
                # 队首 speech 帧超前且还没有上一帧可重复：限时等音频追上再重查，
                # 不能裸等 frame_event（新帧未必再来，会死锁）
                try:
                    await asyncio.wait_for(self._frame_event.wait(), FRAME_TICK_SECONDS)
                except asyncio.TimeoutError:
                    pass
            else:
                await self._frame_event.wait()

    # ── 测试/调试用只读状态 ──

    @property
    def buffered_audio_seconds(self) -> float:
        return len(self._audio) / 2 / 16000

    @property
    def queued_frames(self) -> int:
        return len(self._frames)
