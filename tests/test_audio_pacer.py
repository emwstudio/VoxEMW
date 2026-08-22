"""AudioPacer 单测：20ms 节拍出料、压后等待、flush、播放计数（打断回报用）。"""

import time

import pytest

from voxemw.gateway.audio_pacer import AUDIO_TICK_SAMPLES, AudioPacer


def _pcm(samples: int) -> bytes:
    return b"\x01\x00" * samples


def _play(p: AudioPacer, samples: int) -> None:
    """模拟 RTC 音频轨消费：推进播放时钟。"""
    for _ in range(samples // AUDIO_TICK_SAMPLES):
        p.next_audio_tick()


def test_audio_tick_exact_and_silence():
    p = AudioPacer(audio_lead=0)  # lead=0：立即可播
    # 空缓冲 → 静音
    assert p.next_audio_tick() == b"\x00" * AUDIO_TICK_SAMPLES * 2
    # 喂一拍半 → 第一拍取真数据，余量不足一拍 → 静音且余量保留
    p.feed_audio(_pcm(AUDIO_TICK_SAMPLES + 100))
    assert p.next_audio_tick() == _pcm(AUDIO_TICK_SAMPLES)
    assert p.next_audio_tick() == b"\x00" * AUDIO_TICK_SAMPLES * 2
    assert p.buffered_audio_seconds == pytest.approx(100 / 16000)


def test_audio_lead_holds_then_plays():
    p = AudioPacer(audio_lead=0.05)
    p.feed_audio(_pcm(AUDIO_TICK_SAMPLES))
    # 压后等待期：有数据也先出静音
    assert p.next_audio_tick() == b"\x00" * AUDIO_TICK_SAMPLES * 2
    time.sleep(0.06)
    # 队列未空时追加（同段回复的连续积压）不重置压后点，到点连播两拍
    p.feed_audio(_pcm(AUDIO_TICK_SAMPLES))
    assert p.next_audio_tick() == _pcm(AUDIO_TICK_SAMPLES)
    assert p.next_audio_tick() == _pcm(AUDIO_TICK_SAMPLES)
    # 队列走空再喂（新一段回复）→ 重新压后
    p.feed_audio(_pcm(AUDIO_TICK_SAMPLES))
    assert p.next_audio_tick() == b"\x00" * AUDIO_TICK_SAMPLES * 2


def test_default_lead_is_zero():
    # 纯语音模式无 avatar 渲染可等：默认压后 = 0，到即播
    p = AudioPacer()
    p.feed_audio(_pcm(AUDIO_TICK_SAMPLES))
    assert p.next_audio_tick() == _pcm(AUDIO_TICK_SAMPLES)


def test_flush_resets_counters():
    p = AudioPacer(audio_lead=0)
    p.feed_audio(_pcm(16000))
    _play(p, 3200)
    p.flush()
    assert p.buffered_audio_seconds == 0
    assert p.reply_played_seconds == 0
    # flush 后新一轮：新音频重新从 0 计
    p.feed_audio(_pcm(AUDIO_TICK_SAMPLES))
    assert p.next_audio_tick() == _pcm(AUDIO_TICK_SAMPLES)


def test_reply_played_seconds_tracks_current_reply():
    p = AudioPacer(audio_lead=0)
    p.feed_audio(_pcm(16000))  # 1.0s 回复
    assert p.reply_played_seconds == 0
    _play(p, 8000)  # 播了 0.5s
    assert p.reply_played_seconds == pytest.approx(0.5)
    # 回复播完、队列走空 → 新回复重起游标
    _play(p, 8000)
    p.feed_audio(_pcm(3200))  # 第二条回复 0.2s
    assert p.reply_played_seconds == 0
    _play(p, 3200)
    assert p.reply_played_seconds == pytest.approx(0.2)
    # flush（打断）后归零
    p.feed_audio(_pcm(16000))
    _play(p, 3200)
    p.flush()
    assert p.reply_played_seconds == 0
