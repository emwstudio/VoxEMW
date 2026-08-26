"""音频播放调度器：TTS 音频流排队 + 20ms 节拍出料（纯逻辑，可单测）。

前身是云时代的 AVSyncScheduler（音画同步）；数字人/视频轨拆除后只剩音频：
- RTC 音频轨按 20ms 一拍调用 next_audio_tick() 取 16k PCM，无数据补静音，
  保持 pts 时钟连续
- audio_lead：新回复（队列空→非空）压后秒数。云时代为等 avatar 渲染追赶
  默认 0.20s；纯语音模式无需等待，默认 0（?alead= 调试参数仍可调）
- 打断（barge-in）：flush() 清队列、游标归零；RTC track 侧 pts 游标不动，
  清队列后新音频从当前时刻续播
"""

from __future__ import annotations

import time

AUDIO_TICK_SAMPLES = 320         # 音频轨每次取用量：20ms @16k

DEFAULT_AUDIO_LEAD = 0.0         # 新回复音频压后秒数（云时代 0.20 是等 avatar 渲染）

FADE_IN_SAMPLES = 128            # 新回复开头淡入长度（8ms @16k）


def fade_in(pcm: bytes, n: int = FADE_IN_SAMPLES) -> bytes:
    """起步淡入（纯函数，便于单测）：前 n 个采样乘 0→1 线性坡，
    磨平回复开头的瞬态毛刺（破音/click）。短于 n 的块整体只按首部比例衰减。"""
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return pcm
    k = min(n, samples.size)
    ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
    out = samples.astype(np.float32)
    out[:k] *= ramp
    return out.astype(np.int16).tobytes()


class AudioPacer:
    """音频字节队列 + 播放计数。全部方法在 orchestrator 单 loop 内调用。"""

    def __init__(self, audio_lead: float = DEFAULT_AUDIO_LEAD) -> None:
        self._audio = bytearray()
        self._audio_samples_fed = 0     # 累计喂入采样
        # 口型音素帧（与音频同队列推进）：入队时记绝对采样位（取分析窗中心
        # i*512+512），出队播放游标越过即弹出，由 Session 转发浏览器——
        # 口型对齐的是「播出时刻」而不是「生成时刻」（队列深时差几百 ms，
        # 生成时刻下发会全程对不上，2026-08-25 实测）
        self._lip_queue: list[tuple[int, dict]] = []
        self._lip_played: list[dict] = []
        self._audio_lead = audio_lead
        self._audio_ready_at = 0.0      # 音频队列可播放的最早时刻（单调钟）
        # 已真实播出去的采样数（RTC 音频轨每取一拍真音频 +=320）——
        # 打断回报估算「用户实际听到了多少」靠它而不是喂入时钟
        #（TTS 生成快于播放 RTF<1，按喂入量算会高估）
        self._audio_samples_played = 0
        self._played_at_reply_start = 0  # 当前回复开播时的播放游标（算本条已听时长用）

    # ── 生产者（Session 转发协程调用）──

    def feed_audio(self, pcm: bytes, lip_frames: list[dict] | None = None) -> None:
        """TTS PCM（int16 16k mono）。新回复开头做 ~8ms 淡入，磨平起步瞬态
        （编码/重采样链路在回复起点的毛刺——「第一个音破音」实测修复）。
        lip_frames：该块对应的音素权重帧（512 采样步长，见 gateway.phoneme）。"""
        if lip_frames:
            base = self._audio_samples_fed
            for i, w in enumerate(lip_frames):
                self._lip_queue.append((base + i * 512 + 512, w))
        if not self._audio:
            # 队列从空到非空 = 新回复开始：压后 lead 秒（默认 0 = 到即播）
            self._audio_ready_at = time.monotonic() + self._audio_lead
            self._played_at_reply_start = self._audio_samples_played
            pcm = fade_in(pcm)
        self._audio.extend(pcm)
        self._audio_samples_fed += len(pcm) // 2

    def flush(self) -> None:
        """打断：清队列、游标归零。"""
        self._audio.clear()
        self._audio_samples_fed = 0
        self._audio_samples_played = 0
        self._played_at_reply_start = 0
        self._lip_queue.clear()
        self._lip_played.clear()

    # ── 消费者（RTC track recv 调用）──

    def next_audio_tick(self) -> bytes:
        """取 20ms（320 采样）16k PCM；不足一整拍或尚在压后等待期则补静音。"""
        need = AUDIO_TICK_SAMPLES * 2
        if len(self._audio) >= need and time.monotonic() >= self._audio_ready_at:
            out = bytes(self._audio[:need])
            del self._audio[:need]
            self._audio_samples_played += AUDIO_TICK_SAMPLES
            while self._lip_queue and self._lip_queue[0][0] <= self._audio_samples_played:
                self._lip_played.append(self._lip_queue.pop(0)[1])
            return out
        return b"\x00" * need

    def pop_played_lip(self) -> list[dict]:
        """取走自上次以来已播出的音素帧（Session 转发浏览器用）。"""
        out, self._lip_played = self._lip_played, []
        return out

    # ── 观测用只读状态 ──

    @property
    def buffered_audio_seconds(self) -> float:
        return len(self._audio) / 2 / 16000

    @property
    def reply_played_seconds(self) -> float:
        """当前回复已被真实播放的秒数（打断回报用：用户实际听到了多少）。

        回复边界以音频队列「空→非空」判定；多条回复积压连播时只对
        最前面那条精确（后面的还没开播，归零合理）。"""
        return max(0, self._audio_samples_played - self._played_at_reply_start) / 16000
