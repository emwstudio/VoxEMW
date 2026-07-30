"""orchestrator 纯逻辑单测：session.update 构造与 s2s 事件分类（无需 GPU/ws）。"""

import base64

from voxemw.avatar.orchestrator import build_session_update, classify_s2s_event


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
