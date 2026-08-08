"""orchestrator 纯逻辑单测：session.update 构造与 s2s 事件分类（无需 GPU/ws）。"""

import base64

from voxemw.avatar.orchestrator import (
    avatar_state_transition,
    build_session_update,
    classify_s2s_event,
)


def test_build_session_update():
    event = build_session_update("fengge", "你是峰哥。")
    assert event["type"] == "session.update"
    session = event["session"]
    assert session["instructions"] == "你是峰哥。"
    assert session["audio"]["output"]["voice"] == "fengge"
    assert session["audio"]["input"]["turn_detection"]["interrupt_response"] is True


def test_classify_audio_delta_ga_name():
    pcm = b"\x01\x02" * 100
    event = {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()}
    relay, reset, tapped = classify_s2s_event(event)
    assert relay is True
    assert reset is False
    assert tapped == pcm


def test_classify_audio_delta_beta_name():
    pcm = b"\x00" * 64
    event = {"type": "response.audio.delta", "delta": base64.b64encode(pcm).decode()}
    _, _, tapped = classify_s2s_event(event)
    assert tapped == pcm


def test_classify_speech_started_resets_avatar():
    relay, reset, tapped = classify_s2s_event({"type": "input_audio_buffer.speech_started"})
    assert relay is True
    assert reset is True
    assert tapped is None


def test_classify_other_events_passthrough():
    for etype in ("response.done", "session.created", "conversation.item.input_audio_transcription.completed"):
        relay, reset, tapped = classify_s2s_event({"type": etype})
        assert relay is True
        assert reset is False
        assert tapped is None


def _delta_event():
    return {"type": "response.output_audio.delta",
            "delta": base64.b64encode(b"\x00" * 64).decode()}


def test_transition_first_delta_starts_speech():
    speaking, msgs = avatar_state_transition(_delta_event(), speaking=False)
    assert speaking is True
    assert msgs == [{"type": "speech_active", "on": True}]


def test_transition_subsequent_deltas_silent():
    speaking, msgs = avatar_state_transition(_delta_event(), speaking=True)
    assert speaking is True
    assert msgs == []


def test_transition_response_done_stops_and_calms():
    speaking, msgs = avatar_state_transition({"type": "response.done"}, speaking=True)
    assert speaking is False
    assert msgs == [{"type": "speech_active", "on": False},
                    {"type": "idle_mode", "mode": "calm"}]


def test_transition_speech_started_interrupts_to_listening():
    speaking, msgs = avatar_state_transition(
        {"type": "input_audio_buffer.speech_started"}, speaking=True)
    assert speaking is False
    assert msgs == [{"type": "speech_active", "on": False},
                    {"type": "idle_mode", "mode": "listening"}]


def test_transition_speech_stopped_thinking_only_when_not_speaking():
    _, msgs = avatar_state_transition(
        {"type": "input_audio_buffer.speech_stopped"}, speaking=False)
    assert msgs == [{"type": "idle_mode", "mode": "thinking"}]
    speaking, msgs = avatar_state_transition(
        {"type": "input_audio_buffer.speech_stopped"}, speaking=True)
    assert speaking is True and msgs == []


def test_transition_ignores_empty_delta():
    speaking, msgs = avatar_state_transition(
        {"type": "response.output_audio.delta", "delta": ""}, speaking=False)
    assert speaking is False and msgs == []


def test_pick_filler_index_no_repeat():
    import random
    from voxemw.avatar.orchestrator import pick_filler_index

    random.seed(42)
    last = -1
    for _ in range(200):
        idx = pick_filler_index(8, last)
        assert 0 <= idx < 8
        assert idx != last  # 绝不连续重复
        last = idx


def test_pick_filler_index_single_clip():
    from voxemw.avatar.orchestrator import pick_filler_index

    assert pick_filler_index(1, 0) == 0  # 只有一条时允许重复


def test_load_fillers(tmp_path):
    import json
    import wave

    from voxemw.avatar.orchestrator import build_filler_history_item, load_fillers

    fdir = tmp_path / "assets" / "demo" / "fillers"
    (fdir / "positive").mkdir(parents=True)
    with wave.open(str(fdir / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x01\x02" * 1600)
    with wave.open(str(fdir / "positive" / "b.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x03\x04" * 800)
    (fdir / "bad.wav").write_bytes(b"not a wav")
    (fdir / "texts.json").write_text(
        json.dumps({"a.wav": "嗯——", "positive/b.wav": "哦？"}), encoding="utf-8")
    groups = load_fillers({"ref_image": str(tmp_path / "assets" / "demo" / "ref.png")})
    assert groups["neutral"] == [(b"\x01\x02" * 1600, "嗯——")]  # 根目录散落 wav 归 neutral
    assert groups["positive"] == [(b"\x03\x04" * 800, "哦？")]
    assert groups["negative"] == []
    assert load_fillers({"ref_image": None}) == {"positive": [], "negative": [], "neutral": []}
    item = build_filler_history_item("嗯——")
    assert item["type"] == "conversation.item.create"
    assert item["item"]["role"] == "assistant"
    assert item["item"]["content"][0]["text"] == "嗯——"


def test_emotion_mapping_and_sidecar(tmp_path):
    from voxemw.avatar.orchestrator import EMOTION_TO_GROUP, read_emotion_sidecar

    assert EMOTION_TO_GROUP["HAPPY"] == "positive"
    assert EMOTION_TO_GROUP["SAD"] == "negative"
    assert "NEUTRAL" not in EMOTION_TO_GROUP  # 未映射情绪回退 neutral
    sidecar = tmp_path / "emotion"
    sidecar.write_text("ANGRY")
    assert read_emotion_sidecar(str(sidecar)) == "ANGRY"
    assert read_emotion_sidecar(str(tmp_path / "missing")) == "NEUTRAL"


def test_extract_emotion():
    pytest = __import__("pytest")
    pytest.importorskip("speech_to_speech")  # handler 依赖管线包，仅 GPU 实例可跑
    from voxemw.pipeline.stt_sensevoice import extract_emotion

    assert extract_emotion("<|zh|><|HAPPY|><|Speech|><|withitn|>太好了") == "HAPPY"
    assert extract_emotion("<|zh|><|ANGRY|><|Speech|><|withitn|>你干嘛") == "ANGRY"
    assert extract_emotion("没有标签的文本") == "NEUTRAL"


def test_build_memory_block():
    from voxemw.memory import build_memory_block

    assert build_memory_block([]) == ""
    block = build_memory_block(["用户叫小明", "用户在减肥"])
    assert "关于用户的记忆" in block
    assert "- 用户叫小明" in block
    # 超长截断（80 字上限）
    long_block = build_memory_block(["x" * 100])
    assert "x" * 81 not in long_block


def test_create_memory_store_disabled_by_default():
    from voxemw.memory import create_memory_store

    assert create_memory_store({}) is None
    assert create_memory_store({"memory": {"enabled": False}}) is None
    # 启用但缺 key → 降级 None
    assert create_memory_store({
        "memory": {"enabled": True},
        "llm": {"api_key_env": "DEFINITELY_MISSING_KEY"},
    }) is None
