"""orchestrator 纯逻辑单测：session.update 构造与 s2s 事件分类（无需 GPU/ws）。"""

import base64

from voxemw.gateway.orchestrator import (
    build_session_update,
    classify_s2s_event,
    heard_prefix,
)


def test_heard_prefix_proportional_cut():
    # 播放 1/4 音频 → 约 1/4 文本，截到最近标点
    t = "这事儿吧，得从登山说起。我跟你说，当年那个坡啊，真不是人爬的，累死个人。"
    out = heard_prefix(t, audio_seconds=8.0, played_seconds=2.0)
    assert out == "这事儿吧，"
    # 全播完 → 全文保留
    assert heard_prefix(t, 8.0, 8.0) == t
    # 还没开播 / 无音频 / 空转写 → 空（保持整条回滚）
    assert heard_prefix(t, 8.0, 0) == ""
    assert heard_prefix(t, 0, 1.0) == ""
    assert heard_prefix("", 8.0, 1.0) == ""
    # 不足 2 字不注入
    assert heard_prefix(t, 8.0, 0.1) == ""


def test_heard_prefix_no_punctuation_keeps_raw_cut():
    t = "没有一个标点的长句子在这里直接被切断掉"
    out = heard_prefix(t, audio_seconds=4.0, played_seconds=2.0)
    assert out == t[: len(t) // 2]


def test_build_session_update():
    event = build_session_update("liangzi", "你是良子。")
    assert event["type"] == "session.update"
    session = event["session"]
    assert session["instructions"] == "你是良子。"
    assert session["audio"]["output"]["voice"] == "liangzi"
    assert session["audio"]["input"]["turn_detection"]["interrupt_response"] is True


def test_classify_audio_delta_ga_name():
    pcm = b"\x01\x02" * 100
    event = {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()}
    relay, is_interrupt, tapped = classify_s2s_event(event)
    assert relay is True
    assert is_interrupt is False
    assert tapped == pcm


def test_classify_audio_delta_beta_name():
    pcm = b"\x00" * 64
    event = {"type": "response.audio.delta", "delta": base64.b64encode(pcm).decode()}
    _, _, tapped = classify_s2s_event(event)
    assert tapped == pcm


def test_classify_speech_started_is_interrupt():
    relay, is_interrupt, tapped = classify_s2s_event({"type": "input_audio_buffer.speech_started"})
    assert relay is True
    assert is_interrupt is True
    assert tapped is None


def test_classify_other_events_passthrough():
    for etype in ("response.done", "session.created", "conversation.item.input_audio_transcription.completed"):
        relay, is_interrupt, tapped = classify_s2s_event({"type": etype})
        assert relay is True
        assert is_interrupt is False
        assert tapped is None
