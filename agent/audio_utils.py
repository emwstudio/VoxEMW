"""Audio conversion helpers.

Pure-logic module: only depends on numpy (no torch / transformers / livekit),
so it can be unit-tested on machines without GPU dependencies.
"""

from __future__ import annotations

import numpy as np

_INT16_FULL_SCALE = 32767  # 2**15 - 1


def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float audio samples in [-1.0, 1.0] to little-endian int16 PCM bytes.

    Values outside [-1.0, 1.0] are clipped (saturating conversion).
    Returns b"" for empty input.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * _INT16_FULL_SCALE).astype(np.int16)
    return pcm.tobytes()


def _trim_unstable_head(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_ms: int = 20,
    rms_threshold: float = 0.015,
    jump_ratio: float = 4.0,
    min_consecutive: int = 3,
    pre_roll_ms: int = 60,
    fade_in_ms: int = 16,
    loud_start_threshold: float = 0.1,
) -> np.ndarray:
    """Trim the low-energy unstable murmur at the start of a synthesis stream.

    VoxCPM2 开头有两种废料：纯近静音（RMS ≤0.001）和 ~200ms 的颤音平台
    （RMS 0.02-0.08，接近弱语音，单靠绝对阈值切不掉）。但两者之后的人声
    都有明显的能量跳变，所以起点判定用双条件：

    1. 窗口 RMS ≥ ``rms_threshold``（绝对地板），且连续 ``min_consecutive``
       窗都达标（排除孤立杂音尖峰）；
    2. 窗口 RMS ≥ ``jump_ratio`` × 此前全部窗口的中位数（相对跳变——
       颤音平台内部涨落不到 4 倍，人声起音轻松超过）。

    然后从起点向前最多回退 ``pre_roll_ms``，但只穿过低于地板的窗口：
    保住中文擦音/送气音声母的弱起音（才俩的 /c/），又不会把颤音尾端
    重新包进来。前 5 个窗口没有基线，只有 RMS ≥ ``loud_start_threshold``
    （明显是语音）才允许作起点。找不到起点（整段都安静）则原样返回。
    """
    samples = np.asarray(audio, dtype=np.float32)
    win = max(1, int(sample_rate * window_ms / 1000))
    n = samples.size // win
    if n == 0:
        return samples
    windows = samples[: n * win].reshape(n, win)
    rms = np.sqrt((windows ** 2).mean(axis=1))

    onset = -1
    for i in range(n):
        if rms[i] < rms_threshold:
            continue
        if i + min_consecutive > n or np.any(rms[i : i + min_consecutive] < rms_threshold):
            continue
        if i < 5:
            if rms[i] >= loud_start_threshold:
                onset = i
                break
            continue
        if rms[i] >= jump_ratio * float(np.median(rms[:i])):
            onset = i
            break
    if onset < 0:
        return samples

    start = onset
    max_back = pre_roll_ms // window_ms
    while start > 0 and onset - start < max_back and rms[start - 1] < rms_threshold:
        start -= 1

    out = samples[start * win :].copy()
    fade = min(out.size, int(sample_rate * fade_in_ms / 1000))
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
        out[:fade] *= ramp.astype(np.float32)
    return out


def tame_head_harshness(
    audio: np.ndarray,
    sample_rate: int,
    *,
    tame_ms: int = 120,
    cutoff_hz: float = 7000.0,
) -> np.ndarray:
    """Soften the electronic harshness VoxCPM2 sometimes emits at onset.

    合成开头常带 20-120ms 的高频毛刺（4kHz+ 能量占比 0.7-0.9、过零率
    ~0.3 的"电颤音"），与中文擦音/送气音同频段，裁剪会误伤声母——
    改为对开头 ``tame_ms`` 毫秒做一阶低通（``cutoff_hz``），并向原信号
    线性渐变回去：压住刺耳的高频，保住摩擦音的质感。
    """
    samples = np.asarray(audio, dtype=np.float32)
    n = min(samples.size, int(sample_rate * tame_ms / 1000))
    if n <= 1:
        return samples
    alpha = 1.0 - float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
    head = samples[:n]
    try:
        from scipy.signal import lfilter

        smoothed = lfilter([alpha], [1.0, alpha - 1.0], head)
    except ImportError:  # scipy 不在时的纯 numpy 兜底（一阶递归）
        smoothed = np.empty_like(head)
        acc = np.float32(0.0)
        for i, x in enumerate(head):
            acc += alpha * (x - acc)
            smoothed[i] = acc
    wet = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out = samples.copy()
    out[:n] = smoothed * wet + head * (1.0 - wet)
    return out
