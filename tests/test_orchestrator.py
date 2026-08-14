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


def test_split_dance_marker():
    from voxemw.avatar.orchestrator import split_dance_marker

    assert split_dance_marker("[[dance:科目三]] 看好了啊") == ("科目三", " 看好了啊")
    assert split_dance_marker("  [[dance:鬼步舞]]瞧你的")[0] == "鬼步舞"
    assert split_dance_marker("没有标记的回复") == (None, "没有标记的回复")
    assert split_dance_marker("[[dance:]][[dance: x") == (None, "[[dance:]][[dance: x")


def test_is_marker_prefix():
    from voxemw.avatar.orchestrator import is_marker_prefix

    assert is_marker_prefix("[")
    assert is_marker_prefix("[[dance:科目")
    assert not is_marker_prefix("[[dance:科目三]]")  # 已完结不是前缀
    assert not is_marker_prefix("正常说话")
    assert not is_marker_prefix("x" * 60)  # 超长不可能
