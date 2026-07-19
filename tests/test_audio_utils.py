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
