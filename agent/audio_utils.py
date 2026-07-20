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
    rms_threshold: float = 0.02,
    min_consecutive: int = 3,
    pre_roll_ms: int = 60,
    fade_in_ms: int = 8,
) -> np.ndarray:
    """Trim the low-energy unstable murmur at the start of a synthesis stream.

    VoxCPM2 emits ~160 ms of near-silent, wobbly audio before the voice
    actually starts, and the murmur flickers — a single window can spike
    above the threshold without speech having begun. Onset is declared only
    when ``min_consecutive`` windows in a row all reach ``rms_threshold``
    (60 ms of sustained energy), then the audio starts a small
    ``pre_roll_ms`` earlier and gets a short ``fade_in_ms`` raised-cosine
    ramp so the cut neither keeps murmur nor clicks. If nothing qualifies
    (a genuinely quiet utterance), return the input unchanged.

    阈值和 pre-roll 刻意保守：中文音节常以低能量擦音/送气音开头
    （s/sh/f/h/q/x/c），切得太狠会吃掉首字声母，听起来像首字卡顿。
    """
    samples = np.asarray(audio, dtype=np.float32)
    win = max(1, int(sample_rate * window_ms / 1000))
    n = samples.size // win
    if n == 0:
        return samples
    windows = samples[: n * win].reshape(n, win)
    rms = np.sqrt((windows ** 2).mean(axis=1))
    above = rms >= rms_threshold
    run = np.convolve(above.astype(np.int32), np.ones(min_consecutive, dtype=np.int32), mode="valid")
    hits = np.nonzero(run >= min_consecutive)[0]
    if hits.size == 0:
        return samples
    onset_window = int(hits[0])
    start = max(0, onset_window - pre_roll_ms // window_ms)
    out = samples[start * win :].copy()
    fade = min(out.size, int(sample_rate * fade_in_ms / 1000))
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
        out[:fade] *= ramp.astype(np.float32)
    return out
