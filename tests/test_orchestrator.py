"""orchestrator 纯逻辑单测：session.update 构造与 s2s 事件分类（无需 GPU/ws）。"""

import base64

from voxemw.gateway.orchestrator import (
    build_session_update,
    classify_s2s_event,
    heard_prefix,
    is_echo,
    is_vocabulary_recitation,
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


def test_is_echo_catches_playback_leak():
    # 良子的回答被麦克风收回去：候选被近期助手文本包含
    recent = ["来了老弟！咋啦，这是等着看你良弟吃播来啦？"]
    assert is_echo("来了，老弟。", recent) is True
    # 反向包含：回声转写比原句长（多收了尾巴）
    assert is_echo("来了老弟咋啦这是", ["来了老弟咋啦"]) is True
    # 混入其他声音的回声（非干净包含）：覆盖率 4/7 ≈ 0.57 ≥ 0.5
    assert is_echo("来了，老弟。你好啊。", recent) is True
    # 字序碎裂的回声（她说"哎，来了老弟！"，回声成"来了老弟哎"）：覆盖率 4/5
    assert is_echo("来了，老弟。哎。", ["哎，来了老弟！"]) is True


def test_is_echo_passes_real_speech():
    recent = ["火锅整起来，羊肉片子涮上，大窑一开，味真足！"]
    assert is_echo("今晚吃啥好？给点建议", recent) is False
    # 短句撞车不判回声（「你好啊」这种真实短句必须放行）
    assert is_echo("你好啊", ["你好啊老弟"]) is False
    # 空输入/空历史不判
    assert is_echo("", recent) is False
    assert is_echo("来点吃的推荐", []) is False


def test_is_echo_passes_short_real_answers():
    # 2026-08-23 误杀实录：用户真实短答撞上助手文案的零散文案，最长块 <4 字
    her = ["你吃了没啊老弟？你良弟我这胃袋都空了。"]
    assert is_echo("嗯，我吃了。", her) is False
    assert is_echo("呃，我吃了呀，真的是。", her) is False
    assert is_echo("是吧？浑身带劲儿吧。", ["今天这身板，带劲！"]) is False


def test_is_vocabulary_recitation():
    hw = ["良子", "大胃袋", "味真足"]
    # 噪音被词表脑补的整段背诵 → 掐
    assert is_vocabulary_recitation("良子，大胃袋，味真足。", hw) is True
    assert is_vocabulary_recitation("大胃袋味真足", hw) is True
    # 单个热词（真人叫他）→ 放行
    assert is_vocabulary_recitation("大胃袋", hw) is False
    # 带残余的真实短句 → 放行
    assert is_vocabulary_recitation("味真足啊", hw) is False
    assert is_vocabulary_recitation("良子今晚吃啥", hw) is False
    # 空/超短 → 放行
    assert is_vocabulary_recitation("", hw) is False
    assert is_vocabulary_recitation("嗯", hw) is False
