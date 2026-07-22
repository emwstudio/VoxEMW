from agent.duet_logic import DEFAULT_TOPIC, build_messages, parse_beats, turn_order


def test_build_messages_maps_roles():
    history = [("liangzi", "开场"), ("fengge", "接话"), ("liangzi", "再捧")]
    msgs = build_messages(history, "fengge", "SYS")
    assert msgs[0] == {"role": "system", "content": "SYS"}
    # 对方（良子）的话是 user，自己（峰哥）的话是 assistant
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "开场"


def test_build_messages_empty_history():
    msgs = build_messages([], "liangzi", "SYS")
    assert msgs == [{"role": "system", "content": "SYS"}]


def test_build_messages_names_prefix_for_group_chat():
    history = [("liangzi", "开场"), ("fengge", "接话"), ("user", "我插一句")]
    names = {"liangzi": "良子", "fengge": "峰哥", "user": "老铁"}
    msgs = build_messages(history, "liangzi", "SYS", None, names)
    # 自己（良子）= assistant 无前缀；峰哥和老铁 = user 带名字前缀
    assert msgs[1] == {"role": "assistant", "content": "开场"}
    assert msgs[2] == {"role": "user", "content": "峰哥：接话"}
    assert msgs[3] == {"role": "user", "content": "老铁：我插一句"}


def test_turn_order_alternates():
    it = turn_order("fengge")
    assert [next(it) for _ in range(4)] == ["fengge", "liangzi", "fengge", "liangzi"]


def test_turn_order_rejects_unknown():
    try:
        next(turn_order("nobody"))
        assert False, "should raise"
    except ValueError:
        pass


def test_default_topic_not_empty():
    assert DEFAULT_TOPIC


def test_parse_beats_ok():
    beats = parse_beats("liangzi:夸他白|fengge:好事啊|liangzi:录祝福")
    assert beats == [("liangzi", "夸他白"), ("fengge", "好事啊"), ("liangzi", "录祝福")]


def test_parse_beats_rejects_bad_input():
    for bad in ["", "liangzi只有一条", "unknown:谁"]:
        try:
            parse_beats(bad)
            assert False, f"should raise: {bad!r}"
        except ValueError:
            pass


def test_default_beats_valid_and_cover_both():
    from agent.duet_logic import DEFAULT_BEATS

    keys = [k for k, _ in DEFAULT_BEATS]
    assert "liangzi" in keys and "fengge" in keys
    assert len(DEFAULT_BEATS) >= 3


def test_build_debate_steer():
    from agent.duet_logic import build_debate_steer

    s = build_debate_steer("年轻人该不该躺平", "反")
    assert "年轻人该不该躺平" in s
    assert "反方" in s
    assert "评委" not in s
    s2 = build_debate_steer("X", "正", user_interject=True)
    assert "正方" in s2 and "评委" in s2
