"""AVSyncScheduler 单测：打戳对齐、丢尾帧、flush、静音补齐。

注意：asyncio.Event 绑定首个使用的 loop，每个测试只用一个 asyncio.run 跑完整场景。
"""

import asyncio
import time

import pytest

from voxemw.avatar.avsync import AUDIO_TICK_SAMPLES, SAMPLES_PER_FRAME, AVSyncScheduler


def _pcm(samples: int) -> bytes:
    return b"\x01\x00" * samples


def _jpeg(tag: int) -> bytes:
    return bytes([tag]) * 10  # 假 JPEG，用首字节区分


def test_audio_tick_exact_and_silence():
    s = AVSyncScheduler(audio_lead=0)  # lead=0：立即可播
    # 空缓冲 → 静音
    assert s.next_audio_tick() == b"\x00" * AUDIO_TICK_SAMPLES * 2
    # 喂一拍半 → 第一拍取真数据，余量不足一拍 → 静音且余量保留
    s.feed_audio(_pcm(AUDIO_TICK_SAMPLES + 100))
    assert s.next_audio_tick() == _pcm(AUDIO_TICK_SAMPLES)
    assert s.next_audio_tick() == b"\x00" * AUDIO_TICK_SAMPLES * 2
    assert s.buffered_audio_seconds == pytest.approx(100 / 16000)


def test_audio_lead_holds_then_plays():
    s = AVSyncScheduler(audio_lead=0.05)
    s.feed_audio(_pcm(AUDIO_TICK_SAMPLES))
    # 压后等待期：有数据也先出静音
    assert s.next_audio_tick() == b"\x00" * AUDIO_TICK_SAMPLES * 2
    time.sleep(0.06)
    # 队列未空时追加（同段回复的连续积压）不重置压后点，到点连播两拍
    s.feed_audio(_pcm(AUDIO_TICK_SAMPLES))
    assert s.next_audio_tick() == _pcm(AUDIO_TICK_SAMPLES)
    assert s.next_audio_tick() == _pcm(AUDIO_TICK_SAMPLES)
    # 队列走空再喂（新一段回复）→ 重新压后
    s.feed_audio(_pcm(AUDIO_TICK_SAMPLES))
    assert s.next_audio_tick() == b"\x00" * AUDIO_TICK_SAMPLES * 2


def test_speech_frames_pass_and_tail_dropped():
    async def scenario():
        s = AVSyncScheduler()
        # 喂 1.0s 音频（16000 采样 = 25 帧）+ 25 真帧 + 4 零填充尾帧
        s.feed_audio(_pcm(16000))
        for _ in range(25):
            s.feed_frame(_jpeg(1), is_speech=True)
        for _ in range(4):
            s.feed_frame(_jpeg(2), is_speech=True)
        got = [await s.next_frame_tick() for _ in range(25)]
        assert all(g == _jpeg(1) for g in got)  # 25 真帧全过
        # 尾帧已被连簇清掉 → 下一拍取到「重复上一帧」
        assert await s.next_frame_tick() == _jpeg(1)
        assert s.queued_frames == 0

    asyncio.run(scenario())


def test_partial_audio_frame_count_floors():
    async def scenario():
        s = AVSyncScheduler()
        # 16100 采样 → 25 帧有效（100 采样零头不构成一帧）
        s.feed_audio(_pcm(16100))
        for _ in range(26):
            s.feed_frame(_jpeg(1), is_speech=True)
        got = [await s.next_frame_tick() for _ in range(26)]
        # 第 26 帧被丢 → 取到的是重复上一帧（内容同第 25 帧）
        assert got == [_jpeg(1)] * 26
        assert s.queued_frames == 0

    asyncio.run(scenario())


def test_idle_frames_not_counted():
    async def scenario():
        s = AVSyncScheduler()
        s.feed_frame(_jpeg(9), is_speech=False)
        s.feed_audio(_pcm(SAMPLES_PER_FRAME))
        s.feed_frame(_jpeg(1), is_speech=True)
        assert await s.next_frame_tick() == _jpeg(9)  # idle 帧不占 speech 序号
        assert await s.next_frame_tick() == _jpeg(1)

    asyncio.run(scenario())


def test_flush_resets_counters():
    async def scenario():
        s = AVSyncScheduler()
        s.feed_audio(_pcm(16000))
        for _ in range(10):
            s.feed_frame(_jpeg(1), is_speech=True)
        s.flush()
        assert s.buffered_audio_seconds == 0
        assert s.queued_frames == 0
        # flush 后新一轮：新音频重新从序号 0 计
        s.feed_audio(_pcm(SAMPLES_PER_FRAME * 2))
        s.feed_frame(_jpeg(3), is_speech=True)
        assert await s.next_frame_tick() == _jpeg(3)

    asyncio.run(scenario())


def test_repeat_last_frame_when_starved():
    async def scenario():
        s = AVSyncScheduler()
        s.feed_audio(_pcm(SAMPLES_PER_FRAME))
        s.feed_frame(_jpeg(7), is_speech=True)
        assert await s.next_frame_tick() == _jpeg(7)
        # 队空 → 重复上一帧而不是阻塞（等一小拍超时）
        assert await s.next_frame_tick() == _jpeg(7)

    asyncio.run(scenario())


def test_close_unblocks_waiter():
    async def scenario():
        s = AVSyncScheduler()
        task = asyncio.create_task(s.next_frame_tick())
        await asyncio.sleep(0.05)
        s.close()
        assert await task is None

    asyncio.run(scenario())
