import struct

import numpy as np

from agent.audio_utils import float_to_int16_bytes


def test_amplitude_mapping():
    pcm = float_to_int16_bytes(np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32))
    values = struct.unpack("<4h", pcm)
    assert values == (0, 16383, 32767, -32767)


def test_clipping():
    pcm = float_to_int16_bytes(np.array([1.5, -2.0, 10.0], dtype=np.float32))
    values = struct.unpack("<3h", pcm)
    assert values == (32767, -32767, 32767)


def test_empty_input():
    assert float_to_int16_bytes(np.array([], dtype=np.float32)) == b""


def test_output_byte_length():
    n = 1000
    pcm = float_to_int16_bytes(np.zeros(n, dtype=np.float32))
    assert len(pcm) == 2 * n


def test_small_signal_precision():
    pcm = float_to_int16_bytes(np.array([0.1, -0.1], dtype=np.float32))
    values = struct.unpack("<2h", pcm)
    # 0.1 * 32767 = 3276.7 -> truncated toward zero
    assert values == (3276, -3276)


def test_trim_unstable_head_skips_murmur():
    import numpy as np
    from agent.audio_utils import _trim_unstable_head

    sr = 48000
    quiet = np.full(int(0.16 * sr), 0.001, dtype=np.float32)  # 160ms 弱噪
    loud = np.full(int(0.3 * sr), 0.2, dtype=np.float32)        # 300ms 正常语音
    audio = np.concatenate([quiet, loud])
    out = _trim_unstable_head(audio, sr)
    # 应从语音起点前 ~40ms pre-roll 处开始，而不是 0ms
    head_sec = (len(audio) - len(out)) / sr
    assert 0.1 <= head_sec <= 0.16, head_sec
    assert len(out) > len(loud)


def test_trim_unstable_head_all_quiet_unchanged():
    import numpy as np
    from agent.audio_utils import _trim_unstable_head

    audio = np.full(4800, 0.001, dtype=np.float32)
    out = _trim_unstable_head(audio, 48000)
    assert len(out) == len(audio)


def test_trim_unstable_head_empty():
    import numpy as np
    from agent.audio_utils import _trim_unstable_head

    out = _trim_unstable_head(np.array([], dtype=np.float32), 48000)
    assert out.size == 0


def test_trim_unstable_head_ignores_single_spike():
    import numpy as np
    from agent.audio_utils import _trim_unstable_head

    sr = 48000
    win = sr // 50  # 20ms
    quiet = np.full(8 * win, 0.001, dtype=np.float32)
    spike = np.full(win, 0.2, dtype=np.float32)            # 单个 20ms 尖峰（杂音闪烁）
    quiet2 = np.full(3 * win, 0.001, dtype=np.float32)
    loud = np.full(10 * win, 0.2, dtype=np.float32)        # 持续人声
    audio = np.concatenate([quiet, spike, quiet2, loud])
    out = _trim_unstable_head(audio, sr)
    # 起点应落在持续人声开始前 ~40ms，而不是那个尖峰
    onset_sec = (len(audio) - len(out)) / sr
    expected_onset = (8 + 1 + 3) * win / sr - 0.04
    assert abs(onset_sec - expected_onset) < 0.03, onset_sec


def test_trim_unstable_head_cuts_murmur_plateau():
    """回归：VoxCPM2 开头 ~200ms 颤音平台（RMS 0.02-0.08，接近弱语音），
    必须靠能量跳变切掉，不能被绝对阈值放行。"""
    import numpy as np
    from agent.audio_utils import _trim_unstable_head

    sr = 48000
    win = sr // 50  # 20ms
    rng = np.random.default_rng(7)
    # 10 窗颤音：0.02-0.08 起伏
    plateau = (0.02 + 0.06 * rng.random(10 * win)).astype(np.float32)
    speech = np.full(20 * win, 0.3, dtype=np.float32)
    audio = np.concatenate([plateau, speech])
    out = _trim_unstable_head(audio, sr)
    cut_windows = (len(audio) - len(out)) // win
    # 应从语音起点（第 10 窗）开始，至多向前多留 3 窗（pre-roll 只穿静音，
    # 颤音窗不低于地板所以一窗都不许回退）
    assert cut_windows == 10, cut_windows


def test_trim_unstable_head_keeps_fricative_ramp():
    """回归：中文送气音声母的弱起音斜坡（0.003→0.018 渐强进人声）要保住，
    不能把首字声母切掉。"""
    import numpy as np
    from agent.audio_utils import _trim_unstable_head

    sr = 48000
    win = sr // 50
    silence = np.full(38 * win, 0.0005, dtype=np.float32)
    ramp = np.concatenate(
        [np.full(win, v, dtype=np.float32) for v in (0.003, 0.011, 0.018, 0.047)]
    )
    speech = np.full(20 * win, 0.25, dtype=np.float32)
    audio = np.concatenate([silence, ramp, speech])
    out = _trim_unstable_head(audio, sr)
    cut_windows = (len(audio) - len(out)) // win
    # 起点判在斜坡首窗（0.003 低于地板不判，判在 0.018 窗 = 40），
    # 回退穿静音最多 3 窗 → 37，保住全部斜坡
    assert cut_windows == 37, cut_windows


def test_tame_head_harshness_attenuates_hf_only_at_head():
    """开头的 8kHz 蜂鸣被低通压掉，120ms 之后的信号原样不动。"""
    import numpy as np
    from agent.audio_utils import tame_head_harshness

    sr = 48000
    t = np.arange(sr) / sr
    buzz = 0.1 * np.sin(2 * np.pi * 8000 * t[: sr // 10])  # 100ms 8kHz 蜂鸣
    tone = 0.2 * np.sin(2 * np.pi * 200 * t[: sr // 10])   # 100ms 200Hz 人声基频
    audio = np.concatenate([buzz, tone]).astype(np.float32)
    out = tame_head_harshness(audio, sr, tame_ms=100, cutoff_hz=1000.0)
    # 最开头 20ms（wet≈1）蜂鸣应被明显压掉
    in20 = np.sqrt((audio[: sr // 50] ** 2).mean())
    out20 = np.sqrt((out[: sr // 50] ** 2).mean())
    assert out20 < in20 * 0.3, (in20, out20)
    # 100ms 之后（tame 窗口外）必须逐样本一致
    np.testing.assert_allclose(out[sr // 10 :], audio[sr // 10 :], atol=1e-6)


def test_tame_head_harshness_short_input_passthrough():
    import numpy as np
    from agent.audio_utils import tame_head_harshness

    audio = np.zeros(10, dtype=np.float32)
    out = tame_head_harshness(audio, 48000)
    assert out.shape == audio.shape
