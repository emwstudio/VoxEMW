"""唱歌功能单测：singing 模块纯逻辑/客户端（mock HTTP 层）+ orchestrator vox.sing 编排。

无需 GPU/ws/网络：MusicClient 的 _post/_get_bytes 全 monkeypatch，
orchestrator 侧用假 browser/s2s/avatar/sched 直调 Session 方法。
"""

import asyncio
import base64
import json

from voxemw.avatar.orchestrator import Session
from voxemw.avatar.singing import (
    MAX_DURATION,
    MIN_DURATION,
    MusicClient,
    SongSpec,
    build_task_payload,
    iter_song_segments,
    split_durations,
)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------

def test_split_durations_even_and_remainder():
    assert split_durations(60, 20) == [20, 20, 20]
    assert split_durations(50, 20) == [20, 20, 10]
    assert split_durations(20, 20) == [20]


def test_split_durations_remainder_below_api_floor_merges():
    # 余数 <10s 不满足 API 下限，并进上一段
    assert split_durations(45, 20) == [20, 25]
    assert split_durations(21, 20) == [21]


def test_split_durations_first_segment_shorter():
    # 首段单独调短：开播快；后续段恢复 seg
    assert split_durations(60, 20, first=10) == [10, 20, 20, 10]
    assert split_durations(50, 20, first=10) == [10, 20, 20]
    # 首段余数不足下限仍并进上一段
    assert split_durations(35, 20, first=10) == [10, 25]
    # total 不足首段 → 一段
    assert split_durations(12, 20, first=10) == [12]
    # first 也会被夹到下限
    assert split_durations(40, 20, first=5) == [MIN_DURATION, 20, 10]


def test_split_durations_clamps_total_and_seg():
    assert split_durations(5, 20) == [MIN_DURATION]        # total 不足下限
    assert split_durations(9999, 20)[0] == 20              # total 超上限被夹
    assert sum(split_durations(9999, 20)) == MAX_DURATION
    assert split_durations(120, 3) == [MIN_DURATION] * 12  # seg 不足下限被夹


def test_build_task_payload():
    spec = SongSpec(prompt="民谣", lyrics="啦啦啦", seconds=30, vocal_language="zh")
    payload = build_task_payload(spec, "acestep-v15-turbo")
    assert payload["prompt"] == "民谣"
    assert payload["lyrics"] == "啦啦啦"
    assert payload["audio_duration"] == 30.0
    assert payload["model"] == "acestep-v15-turbo"
    assert payload["vocal_language"] == "zh"
    assert payload["audio_format"] == "wav"
    # 歌词直给 → thinking 关（跳过 LM，每段省 ~25s）；歌词空 → thinking 开（LM 写词）
    assert payload["thinking"] is False
    empty = build_task_payload(SongSpec(prompt="p", lyrics="", seconds=30), "m")
    assert empty["thinking"] is True


# ---------------------------------------------------------------------------
# MusicClient（monkeypatch _post/_get_bytes，不打真网络）
# ---------------------------------------------------------------------------

def _client(monkeypatch, query_replies):
    """query_replies：/query_result 依次返回的 data 列表。"""
    client = MusicClient("http://127.0.0.1:8001", poll_interval=0, task_timeout=30)
    calls = {"query": 0}

    async def fake_post(path, *, json_body=None, data=None):
        if path == "/release_task":
            assert json_body is not None or data is not None
            return {"data": {"task_id": "t1", "status": "queued"}}
        assert path == "/query_result"
        assert json_body == {"task_id_list": ["t1"]}
        reply = query_replies[min(calls["query"], len(query_replies) - 1)]
        calls["query"] += 1
        return {"data": reply}

    async def fake_get(path):
        assert path.startswith("/v1/audio?path=")
        return b"WAVBYTES"

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client, "_get_bytes", fake_get)
    return client


def test_generate_success_after_queue(monkeypatch):
    running = [{"task_id": "t1", "status": 0, "result": "[]"}]
    done = [{"task_id": "t1", "status": 1,
             "result": json.dumps([{"file": "/v1/audio?path=%2Ftmp%2Fx.wav"}])}]
    client = _client(monkeypatch, [running, done])
    audio = run(client.generate(SongSpec(prompt="p", lyrics="l", seconds=20)))
    assert audio == b"WAVBYTES"


def test_generate_failure_raises_with_server_error(monkeypatch):
    failed = [{"task_id": "t1", "status": 2,
               "result": json.dumps([{"error": "CUDA OOM"}])}]
    client = _client(monkeypatch, [failed])
    try:
        run(client.generate(SongSpec(prompt="p", lyrics="l", seconds=20)))
        raise AssertionError("应当抛 RuntimeError")
    except RuntimeError as e:
        assert "CUDA OOM" in str(e)


def test_generate_timeout(monkeypatch):
    running = [{"task_id": "t1", "status": 0, "result": "[]"}]
    client = _client(monkeypatch, [running])
    client.task_timeout = 0  # 立刻超时
    try:
        run(client.generate(SongSpec(prompt="p", lyrics="l", seconds=20)))
        raise AssertionError("应当抛 TimeoutError")
    except TimeoutError:
        pass


class _FakeClient:
    """iter_song_segments 用：按段产固定字节，记录调用。"""

    def __init__(self):
        self.generated: list[int] = []

    async def generate(self, spec, ref_audio=None, src_audio=None, cover_strength=None):
        self.generated.append(spec.seconds)
        return f"AUDIO{spec.seconds}".encode()

    async def to_pcm16k(self, audio):
        return b"PCM_" + audio


def test_iter_song_segments_yields_in_order():
    client = _FakeClient()
    spec = SongSpec(prompt="p", lyrics="l", seconds=50)

    async def collect():
        return [pcm async for pcm in iter_song_segments(client, spec, 20)]

    out = run(collect())
    assert client.generated == [20, 20, 10]
    assert out == [b"PCM_AUDIO20", b"PCM_AUDIO20", b"PCM_AUDIO10"]


def test_iter_song_segments_cancellation_stops_generation():
    client = _FakeClient()
    spec = SongSpec(prompt="p", lyrics="l", seconds=60)

    async def first_then_cancel():
        agen = iter_song_segments(client, spec, 20)
        first = await agen.__anext__()
        await agen.aclose()  # 打断：不再生成后续段
        return first

    first = run(first_then_cancel())
    assert first == b"PCM_AUDIO20"
    assert client.generated == [20]


# ---------------------------------------------------------------------------
# orchestrator 编排（假连接直调 Session 方法）
# ---------------------------------------------------------------------------

class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, data):
        self.sent.append(json.loads(data))

    def types(self):
        return [m.get("type") for m in self.sent]


class _FakeBrowser:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_str(self, data):
        self.sent.append(json.loads(data))

    def sing_events(self):
        return [m for m in self.sent if m.get("type") == "vox.sing"]


class _FakeSched:
    def __init__(self):
        self.audio: list[bytes] = []
        self.flushed = 0

    def feed_audio(self, pcm):
        self.audio.append(pcm)

    def flush(self):
        self.flushed += 1


def _session(monkeypatch, *, segments=(b"\x01\x02" * 100, b"\x03\x04" * 100),
             music_cfg=None, with_avatar=True):
    """构造带假连接的 Session；iter_song_segments 替换为产固定段。"""
    import voxemw.avatar.orchestrator as orch
    import voxemw.avatar.singing as singing_mod

    async def fake_iter(client, spec, segment_seconds, ref_audio=None, first_segment_seconds=None, src_audio=None, cover_strength=None):
        for pcm in segments:
            yield pcm

    monkeypatch.setattr(singing_mod, "iter_song_segments", fake_iter)
    monkeypatch.setattr(orch, "iter_song_segments", fake_iter, raising=False)

    cfg = {"sync_singing": True, "use_persona_ref": False,
           "segment_seconds": 20, "max_duration": 180, "vocal_language": "zh"}
    cfg.update(music_cfg or {})
    session = Session(
        _FakeBrowser(), "ws://s2s", "ws://avatar" if with_avatar else None,
        personas={"p1": {"text": "人设", "ref_wav": None}}, default_persona="p1",
        rtc_sched=_FakeSched(), music_client=MusicClient("http://x"), music_cfg=cfg,
    )
    session.s2s = _FakeWS()
    session.avatar = _FakeWS() if with_avatar else None
    return session


def test_sing_feeds_rtc_and_avatar_then_notifies(monkeypatch):
    session = _session(monkeypatch)
    run(session._sing(SongSpec(prompt="民谣", lyrics="", seconds=40)))
    # 两段 PCM 都进了 RTC 音轨
    assert len(session.sched.audio) == 2
    # avatar 收到 speech_active 开/关 + 每段 audio + 收尾 calm
    types = session.avatar.types()
    assert {"type": "speech_active", "on": True} in session.avatar.sent
    assert {"type": "speech_active", "on": False} in session.avatar.sent
    assert {"type": "idle_mode", "mode": "calm"} in session.avatar.sent
    assert types.count("audio") == 2
    # s2s 收到上下文留痕
    assert "conversation.item.create" in session.s2s.types()
    # 浏览器收到 started → finished
    statuses = [e["status"] for e in session.browser.sing_events()]
    assert statuses == ["started", "finished"]


def test_sing_sync_off_skips_avatar_audio(monkeypatch):
    session = _session(monkeypatch, music_cfg={"sync_singing": False})
    run(session._sing(SongSpec(prompt="p", lyrics="", seconds=20)))
    assert len(session.sched.audio) == 2  # RTC 照喂（两段）
    assert session.avatar.sent == [{"type": "reset"}]  # 只复位，不喂音频/不开 speech_active


def test_sing_cancel_stops_midway(monkeypatch):
    import voxemw.avatar.orchestrator as orch
    import voxemw.avatar.singing as singing_mod

    yielded = []

    async def slow_iter(client, spec, segment_seconds, ref_audio=None, first_segment_seconds=None, src_audio=None, cover_strength=None):
        for i in range(5):
            yielded.append(i)
            yield b"\x00" * 64
            await asyncio.sleep(0.05)

    monkeypatch.setattr(singing_mod, "iter_song_segments", slow_iter)
    monkeypatch.setattr(orch, "iter_song_segments", slow_iter, raising=False)

    session = _session(monkeypatch)
    # _session 里已打了快速假生成器，这里重新盖回慢速版（_sing 是调用时现取模块属性）
    monkeypatch.setattr(singing_mod, "iter_song_segments", slow_iter)

    async def go():
        task = asyncio.create_task(session._sing(SongSpec(prompt="p", lyrics="", seconds=60)))
        session._sing_task = task
        await asyncio.sleep(0.08)  # 放出 1-2 段
        session._cancel_sing()     # 打断
        try:
            await task
        except asyncio.CancelledError:
            pass
        return task

    task = run(go())
    assert task.cancelled()
    assert len(yielded) < 5  # 没唱完
    assert session._sing_task is None


def test_start_sing_disabled_replies_off():
    session = Session(_FakeBrowser(), "ws://s2s", None,
                      personas={"p1": {"text": "x"}}, default_persona="p1",
                      rtc_sched=_FakeSched(), music_client=None, music_cfg={})
    run(session._start_sing({"type": "vox.sing", "prompt": "p"}))
    events = session.browser.sing_events()
    assert events == [{"type": "vox.sing", "status": "off"}]
    assert session._sing_task is None


def test_start_sing_without_rtc_rejected():
    session = Session(_FakeBrowser(), "ws://s2s", None,
                      personas={"p1": {"text": "x"}}, default_persona="p1",
                      rtc_sched=None, music_client=MusicClient("http://x"), music_cfg={})
    run(session._start_sing({"type": "vox.sing", "prompt": "p"}))
    statuses = [e["status"] for e in session.browser.sing_events()]
    assert statuses == ["failed"]
    assert session._sing_task is None


def test_start_sing_clamps_seconds_and_creates_task(monkeypatch):
    session = _session(monkeypatch)

    async def go():
        await session._start_sing({"type": "vox.sing", "prompt": "p", "seconds": 99999})
        task = session._sing_task
        assert task is not None
        await task  # 假生成器瞬间唱完
        return task

    run(go())
    assert session._sing_task is None  # 唱完自清
    assert [e["status"] for e in session.browser.sing_events()] == ["started", "finished"]


# ---------------------------------------------------------------------------
# 语音口播点歌（LLM function calling → sing_song）
# ---------------------------------------------------------------------------

def test_build_session_update_with_tools():
    from voxemw.avatar.orchestrator import build_session_update, build_sing_tool

    tool = build_sing_tool()
    assert tool["type"] == "function" and tool["name"] == "sing_song"
    assert "prompt" in tool["parameters"]["required"]

    event = build_session_update("fengge", "你是峰哥。", tools=[tool])
    assert event["session"]["tools"] == [tool]
    # 不传 tools：session 里没有 tools 字段（不影响既有行为）
    assert "tools" not in build_session_update("fengge", "你是峰哥。")["session"]


def test_tools_registered_only_when_music_enabled():
    with_music = Session(_FakeBrowser(), "ws://s2s", None,
                         personas={"p1": {"text": "x"}}, default_persona="p1",
                         rtc_sched=_FakeSched(),
                         music_client=MusicClient("http://x"), music_cfg={})
    assert with_music._tools is not None and with_music._tools[0]["name"] == "sing_song"
    without_music = Session(_FakeBrowser(), "ws://s2s", None,
                            personas={"p1": {"text": "x"}}, default_persona="p1",
                            rtc_sched=_FakeSched(), music_client=None, music_cfg={})
    assert without_music._tools is None


def test_sing_tool_call_starts_song_and_acks(monkeypatch):
    session = _session(monkeypatch)

    async def go():
        await session._handle_sing_tool_call({
            "type": "response.function_call_arguments.done",
            "name": "sing_song", "call_id": "call_123",
            "arguments": json.dumps({"prompt": "民谣, 登山", "seconds": 60}),
        })
        task = session._sing_task
        assert task is not None
        await task

    run(go())
    # 工具结果按 call_id 回传；不触发 response.create（演唱期间模型保持安静）
    items = [m for m in session.s2s.sent
             if m.get("type") == "conversation.item.create"
             and (m.get("item") or {}).get("type") == "function_call_output"]
    assert len(items) == 1
    assert items[0]["item"]["call_id"] == "call_123"
    assert "已开始" in items[0]["item"]["output"]
    assert "response.create" not in session.s2s.types()
    # 歌真的唱了
    assert [e["status"] for e in session.browser.sing_events()] == ["started", "finished"]
    assert len(session.sched.audio) == 2


def test_sing_tool_call_bad_arguments_still_acks(monkeypatch):
    session = _session(monkeypatch)

    async def go():
        await session._handle_sing_tool_call({
            "type": "response.function_call_arguments.done",
            "name": "sing_song", "call_id": "call_bad",
            "arguments": "{不是合法 json",
        })
        if session._sing_task is not None:
            await session._sing_task

    run(go())
    items = [m for m in session.s2s.sent
             if (m.get("item") or {}).get("type") == "function_call_output"]
    assert len(items) == 1  # 参数坏了也要回传（不然上游锁住后续 response.create）


def test_sing_tool_call_without_music_reports_unavailable():
    session = Session(_FakeBrowser(), "ws://s2s", None,
                      personas={"p1": {"text": "x"}}, default_persona="p1",
                      rtc_sched=_FakeSched(), music_client=None, music_cfg={})
    session.s2s = _FakeWS()
    run(session._handle_sing_tool_call({
        "type": "response.function_call_arguments.done",
        "name": "sing_song", "call_id": "c1", "arguments": "{}"}))
    items = [m for m in session.s2s.sent
             if (m.get("item") or {}).get("type") == "function_call_output"]
    assert len(items) == 1 and "不可用" in items[0]["item"]["output"]
    assert session._sing_task is None


class _FakeS2SStream(_FakeWS):
    """可 async 迭代的假 s2s：按序吐出预设事件，驱动 _s2s_to_browser 全循环。"""

    def __init__(self, events):
        super().__init__()
        self._events = [json.dumps(e) for e in events]

    def __aiter__(self):
        async def gen():
            for raw in self._events:
                yield raw
                await asyncio.sleep(0)  # 让唱歌协程有机会推进
        return gen()


def test_sing_tool_call_survives_response_done(monkeypatch):
    """工具调用回复的 response.done 不得翻转唱歌状态、不得触发空回复追问。"""
    import voxemw.avatar.singing as singing_mod

    async def slow_iter(client, spec, segment_seconds, ref_audio=None, first_segment_seconds=None, src_audio=None, cover_strength=None):
        for _ in range(2):
            yield b"\x00" * 64
            await asyncio.sleep(0.05)

    session = _session(monkeypatch)
    # _session 里打的是瞬完版，盖回慢速版保证 response.done 到达时歌还在唱
    monkeypatch.setattr(singing_mod, "iter_song_segments", slow_iter)
    session.s2s = _FakeS2SStream([
        {"type": "response.created"},
        {"type": "response.output_item.added", "item": {"type": "function_call"}},
        {"type": "response.function_call_arguments.done", "name": "sing_song",
         "call_id": "call_1",
         "arguments": json.dumps({"prompt": "民谣", "seconds": 40})},
        {"type": "response.output_item.done", "item": {"type": "function_call"}},
        {"type": "response.done", "response": {"status": "completed"}},
    ])

    async def go():
        await session._s2s_to_browser()
        if session._sing_task is not None:
            await session._sing_task

    run(go())
    # speech_active 只有 _sing 的一对开/关——response.done 没有中途翻掉
    assert session.avatar.sent.count({"type": "speech_active", "on": True}) == 1
    assert session.avatar.sent.count({"type": "speech_active", "on": False}) == 1
    # 空回复兜底没误触发（纯 function call 的回复没有文本/音频产出）
    nudges = [m for m in session.s2s.sent if "空的" in json.dumps(m, ensure_ascii=False)]
    assert nudges == []
    assert [e["status"] for e in session.browser.sing_events()] == ["started", "finished"]


# ---------------------------------------------------------------------------
# cover 翻唱模式（源歌切片 + task_type=cover）
# ---------------------------------------------------------------------------

def test_build_task_payload_cover():
    spec = SongSpec(prompt="翻唱", lyrics="新词", seconds=20)
    payload = build_task_payload(spec, "m", cover_strength=0.4)
    assert payload["task_type"] == "cover"
    assert payload["audio_cover_strength"] == 0.4
    assert payload["thinking"] is True  # cover 强制开 CoT（歌词-旋律对齐）
    # 不给 strength = text2music（兼容旧行为）
    plain = build_task_payload(spec, "m")
    assert plain["task_type"] == "text2music"
    assert "audio_cover_strength" not in plain


def test_sing_source_path_lookup(tmp_path, monkeypatch):
    from voxemw.avatar import orchestrator as orch

    monkeypatch.setattr(orch, "SING_SOURCE_DIR", tmp_path)
    f = tmp_path / "abcd1234abcd_我的歌.wav"
    f.write_bytes(b"wavdata")
    assert orch.sing_source_path("abcd1234abcd") == f
    # 目录穿越/非法 id 一律 None
    assert orch.sing_source_path("../etc/passwd") is None
    assert orch.sing_source_path("zzzz") is None
    assert orch.sing_source_path("000000000000") is None  # 不存在


def test_start_sing_with_missing_src_rejected():
    session = Session(_FakeBrowser(), "ws://s2s", None,
                      personas={"p1": {"text": "x"}}, default_persona="p1",
                      rtc_sched=_FakeSched(), music_client=MusicClient("http://x"),
                      music_cfg={})
    run(session._start_sing({"type": "vox.sing", "prompt": "p", "src": "abcd1234abcd"}))
    statuses = [e["status"] for e in session.browser.sing_events()]
    assert statuses == ["failed"]
    assert session._sing_task is None


def test_sing_cover_passes_src_and_strength(monkeypatch):
    """cover 路径：src_audio + cover_strength 透传到分段生成器。"""
    import voxemw.avatar.singing as singing_mod

    captured = {}

    async def fake_iter(client, spec, segment_seconds, ref_audio=None,
                        first_segment_seconds=None, src_audio=None,
                        cover_strength=None):
        captured["src_audio"] = src_audio
        captured["cover_strength"] = cover_strength
        yield b"\x00" * 64

    monkeypatch.setattr(singing_mod, "iter_song_segments", fake_iter)
    session = _session(monkeypatch, music_cfg={"cover_strength": 0.35})
    # _session 里已打过瞬完版假生成器，盖回捕获版（_sing 是调用时现取模块属性）
    monkeypatch.setattr(singing_mod, "iter_song_segments", fake_iter)
    run(session._sing(SongSpec(prompt="p", lyrics="新词", seconds=20),
                      src_audio=("src.wav", b"SRCDATA")))
    assert captured["src_audio"] == ("src.wav", b"SRCDATA")
    assert captured["cover_strength"] == 0.35
    assert [e["status"] for e in session.browser.sing_events()] == ["started", "finished"]


def test_iter_song_segments_cover_slicing():
    """cover 模式：整首一次出（不切源歌不分段）、歌长夹到源歌时长。"""
    import shutil

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        import pytest
        pytest.skip("需要 ffmpeg/ffprobe")

    class _CoverClient:
        def __init__(self):
            self.calls = []

        async def probe_duration(self, b):
            return 25.0

        async def slice_audio(self, b, offset, dur):
            self.calls.append((offset, dur))  # 不应再被调用
            return b"slice"

        async def generate(self, spec, ref_audio=None, src_audio=None, cover_strength=None):
            # 整首一次生成，源歌原样透传（不切）
            assert src_audio == ("src.wav", b"whatever")
            assert spec.seconds == 25  # 99 夹到源歌 25s
            assert cover_strength == 0.8
            return f"A{spec.seconds}".encode()

        async def to_pcm16k(self, audio):
            return b"P_" + audio

    client = _CoverClient()
    spec = SongSpec(prompt="p", lyrics="l", seconds=99)  # 故意超源歌长

    async def collect():
        return [pcm async for pcm in iter_song_segments(
            client, spec, 20, first_segment_seconds=10, cover_strength=0.8,
            src_audio=("src.wav", b"whatever"))]

    out = run(collect())
    assert client.calls == []          # 不切源歌
    assert out == [b"P_A25"]           # 整首一段
