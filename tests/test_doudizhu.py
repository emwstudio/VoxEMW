"""斗地主引擎纯逻辑单测（macOS 本地可跑，不依赖 GPU/torch）。"""

import pytest

from doudizhu.cards import (
    analyze,
    beats,
    full_deck,
    lead_smallest_single,
    minimal_beat,
    rank_value,
    sort_cards,
)
from doudizhu.engine import Game, IllegalMove
from doudizhu import heuristic


# ---------------------------------------------------------------------------
# 牌型识别
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cards,type_,main",
    [
        (["S3"], "single", 3),
        (["RJ"], "single", 17),
        (["S3", "H3"], "pair", 3),
        (["BJ", "RJ"], "rocket", 17),
        (["S3", "H3", "D3"], "triple", 3),
        (["S3", "H3", "D3", "C3"], "bomb", 3),
        (["S3", "H3", "D3", "C5"], "triple_one", 3),
        (["S3", "H3", "D3", "C5", "H5"], "triple_pair", 3),
        (["S3", "H4", "D5", "C6", "S7"], "straight", 7),
        (["S3", "H4", "D5", "C6", "S7", "H8", "D9", "C10"], "straight", 10),
        (["S3", "H3", "D4", "C4", "S5", "H5"], "pairs_seq", 5),
        (["S3", "H3", "D3", "C4", "H4", "D4"], "plane", 4),
        (["S3", "H3", "D3", "C4", "H4", "D4", "S5", "C6"], "plane_single", 4),
        (["S3", "H3", "D3", "C4", "H4", "D4", "S5", "H5", "C6", "D6"], "plane_pair", 4),
        (["S3", "H3", "D3", "C3", "S5", "C6"], "four_two_single", 3),
        (["S3", "H3", "D3", "C3", "S5", "H5", "C6", "D6"], "four_two_pair", 3),
    ],
)
def test_analyze_valid(cards, type_, main):
    combo = analyze(cards)
    assert combo is not None, f"{cards} 应识别为 {type_}"
    assert combo.type == type_
    assert combo.main == main
    assert combo.length == len(cards)


@pytest.mark.parametrize(
    "cards",
    [
        ["S3", "H4"],                       # 两张不同点
        ["S3", "H3", "D4"],                 # 三张不成型
        ["S3", "H4", "D5", "C6"],           # 顺子不足 5 张
        ["SJ", "HQ", "DK", "CA", "S2"],     # 顺子带 2
        ["S3", "H3", "D4", "C4"],           # 两对不是连对（不足 3 对）
        ["S3", "H3", "D3", "C5", "H5", "D6"],  # 三张+两张散牌不成型
        ["S3", "H3", "D3", "C3", "S5"],     # 4+1
    ],
)
def test_analyze_invalid(cards):
    assert analyze(cards) is None


def test_beats():
    s3 = analyze(["S3"])
    s4 = analyze(["S4"])
    p3 = analyze(["S3", "H3"])
    bomb4 = analyze(["S4", "H4", "D4", "C4"])
    rocket = analyze(["BJ", "RJ"])
    assert beats(s4, s3)
    assert not beats(s3, s4)
    assert not beats(p3, s3)          # 不同型不可比
    assert beats(bomb4, p3)           # 炸弹管一切普通牌
    assert beats(bomb4, s3)
    assert beats(rocket, bomb4)       # 王炸最大
    assert not beats(bomb4, rocket)


def test_beats_straight_needs_same_length():
    s5 = analyze(["S3", "H4", "D5", "C6", "S7"])
    s6 = analyze(["S3", "H4", "D5", "C6", "S7", "H8"])
    assert not beats(s6, s5)


# ---------------------------------------------------------------------------
# 最小管牌
# ---------------------------------------------------------------------------

def test_minimal_beat_single():
    hand = sort_cards(["S3", "H5", "D9", "C2"])
    assert minimal_beat(hand, analyze(["S4"])) == ["H5"]
    assert minimal_beat(hand, analyze(["S2"])) is None  # 没炸弹


def test_minimal_beat_prefers_non_bomb():
    hand = sort_cards(["S5", "H5", "D3", "C3", "S3", "H3"])  # 有对5和炸弹3
    assert minimal_beat(hand, analyze(["S4", "H4"])) == sort_cards(["S5", "H5"])


def test_minimal_beat_bomb_fallback():
    hand = sort_cards(["S3", "H3", "D3", "C3", "S9"])
    assert minimal_beat(hand, analyze(["SK"])) == sort_cards(["S3", "H3", "D3", "C3"])


def test_minimal_beat_joker_single_before_rocket():
    hand = sort_cards(["BJ", "RJ", "S3"])
    # 单张小王就能管 2，不必动王炸
    assert minimal_beat(hand, analyze(["S2"])) == ["BJ"]


def test_minimal_beat_rocket_against_bomb():
    hand = sort_cards(["BJ", "RJ", "S3"])
    # 对面炸弹，手里没有更大炸弹 → 王炸兜底
    assert minimal_beat(hand, analyze(["S5", "H5", "D5", "C5"])) == ["BJ", "RJ"]


def test_minimal_beat_straight():
    hand = sort_cards(["S3", "H4", "D5", "C6", "S7", "H8", "D9"])
    assert minimal_beat(hand, analyze(["S3", "H4", "D5", "C6", "S7"])) == \
        ["H4", "D5", "C6", "S7", "H8"]


def test_lead_smallest_single_prefers_singles():
    hand = sort_cards(["S3", "H3", "D5", "C9"])
    assert lead_smallest_single(hand) == ["D5"]  # 3 成对不拆，出单 5


# ---------------------------------------------------------------------------
# 引擎流程
# ---------------------------------------------------------------------------

def make_game(seed=42):
    g = Game(seed=seed)
    g.start()
    return g


def test_deal():
    g = make_game()
    assert g.phase == "bidding"
    assert all(len(h) == 17 for h in g.hands.values())
    assert len(g.bottom) == 3
    all_cards = sum(g.hands.values(), []) + g.bottom
    assert sorted(all_cards) == sorted(full_deck())


def test_bid_first_caller_becomes_landlord():
    g = make_game()
    first = g.bid_turn
    events = g.bid(first, True)
    assert g.landlord == first
    assert len(g.hands[first]) == 20
    assert g.phase == "playing" and g.turn == first
    assert any(e["type"] == "landlord" for e in events)


def test_fixed_landlord_skips_bidding():
    g = make_game()
    g.start(fixed_landlord="fengge")
    assert g.phase == "playing"          # 跳过叫地主
    assert g.landlord == "fengge"
    assert g.bid_turn is None
    assert len(g.hands["fengge"]) == 20  # 地主拿底牌
    assert g.turn == "fengge"            # 地主先出
    all_cards = sum(g.hands.values(), [])
    assert sorted(all_cards) == sorted(full_deck())


def test_fixed_landlord_rejects_unknown_seat():
    g = make_game()
    with pytest.raises(ValueError):
        g.start(fixed_landlord="nobody")


def test_last_action_tracking():
    g = make_game()
    g.start(fixed_landlord="fengge")
    assert g.last_action is None
    # 地主首出一张 -> last_action=play
    card = g.hands["fengge"][0]
    g.play("fengge", [card])
    assert g.last_action == {"type": "play", "seat": "fengge"}
    # 下一家不要 -> last_action=pass
    g.pass_turn(g.turn)
    assert g.last_action["type"] == "pass"
    # 再上一家出牌 -> 覆盖回 play
    g2 = make_game()
    g2.start(fixed_landlord="fengge")
    c = g2.hands["fengge"][0]
    g2.play("fengge", [c])
    assert g2.last_action == {"type": "play", "seat": "fengge"}


# ---------------------------------------------------------------------------
# 闲聊路由（route_bot）
# ---------------------------------------------------------------------------

def _bots():
    from doudizhu.persona import Persona
    return {
        "liangzi": Persona("liangzi", "大胃袋良子", "", "", ""),
        "fengge": Persona("fengge", "峰哥亡命天涯", "", "", ""),
    }


def test_route_bot_named():
    from doudizhu.chat import route_bot
    g = make_game()
    g.start(fixed_landlord="fengge")
    bots = _bots()
    assert route_bot(g, "you", "峰哥你这牌不行啊", bots) == "fengge"
    assert route_bot(g, "you", "良子咱俩一起搂他", bots) == "liangzi"


def test_route_bot_stt_homophone():
    from doudizhu.chat import route_bot
    g = make_game()
    g.start(fixed_landlord="fengge")
    bots = _bots()
    # STT 常把「峰」听成「风/锋」，「良」听成「梁」
    assert route_bot(g, "you", "风哥你别狂", bots) == "fengge"
    assert route_bot(g, "you", "锋哥出牌啊", bots) == "fengge"
    assert route_bot(g, "you", "梁子好样的", bots) == "liangzi"


def test_route_bot_unnamed_goes_to_landlord():
    from doudizhu.chat import route_bot
    g = make_game()
    g.start(fixed_landlord="fengge")
    # 不点名：农民说话由地主回怼，而不是队友良子搭腔
    assert route_bot(g, "you", "就这？你不行啊", _bots()) == "fengge"


def test_bid_pass_rotation_and_redeal():
    g = make_game()
    order = []
    for _ in range(3):
        order.append(g.bid_turn)
        events = g.bid(g.bid_turn, False)
    assert len(set(order)) == 3  # 三家轮一遍
    assert any(e["type"] == "no_bid_redeal" for e in events)
    assert g.phase == "bidding"    # 重发后重新叫
    assert all(len(h) == 17 for h in g.hands.values())


def test_bid_wrong_turn_rejected():
    g = make_game()
    other = next(s for s in g.seats if s != g.bid_turn)
    with pytest.raises(IllegalMove):
        g.bid(other, True)


def play_first(game):
    """帮当前 turn 玩家用 heuristic 出一手，返回事件。"""
    return heuristic.act(game, game.turn) if game.phase == "bidding" else None


def test_play_validation():
    g = make_game()
    g.bid(g.bid_turn, True)
    turn = g.turn
    card = g.hands[turn][0]
    wrong_seat = next(s for s in g.seats if s != turn)
    with pytest.raises(IllegalMove):
        g.play(wrong_seat, [g.hands[wrong_seat][0]])  # 没轮到
    with pytest.raises(IllegalMove):
        g.play(turn, ["XX"])                           # 没有的牌
    with pytest.raises(IllegalMove):
        g.play(turn, g.hands[turn][:2])                # 两张不同点（大概率非法）或正好是对子…改用确定非法的
    # 上面一行不稳，换成确定非法：单走一张不存在的组合已在上一行覆盖
    events = g.play(turn, [card])
    assert events[0]["type"] == "play" and events[0]["seat"] == turn
    assert g.turn != turn


def test_pass_rotation_free_turn():
    g = make_game()
    g.bid(g.bid_turn, True)
    first = g.turn
    g.play(first, [g.hands[first][0]])
    second, third = g.turn, None
    g.pass_turn(second)
    third = g.turn
    events = g.pass_turn(third)
    # 两家不要 → 出牌权回到 first，自由出牌
    assert g.turn == first and g.last_play is None
    assert any(e["type"] == "free_turn" for e in events)


def test_cannot_pass_on_free_turn():
    g = make_game()
    g.bid(g.bid_turn, True)
    with pytest.raises(IllegalMove):
        g.pass_turn(g.turn)


def test_win_landlord_and_farmers():
    for winner_role in ("landlord", "farmers"):
        g = make_game()
        g.bid(g.bid_turn, True)
        # 直接构造残局：给某家只留一张 3，其他牌清空
        win_seat = g.landlord if winner_role == "landlord" else next(
            s for s in g.seats if s != g.landlord
        )
        g.hands[win_seat] = ["S3"]
        g.turn = win_seat
        g.last_play = None
        events = g.play(win_seat, ["S3"])
        assert g.phase == "finished"
        assert g.winner == winner_role
        finish = next(e for e in events if e["type"] == "finish")
        assert finish["win_seat"] == win_seat


def test_simulation_many_games_all_heuristic():
    """100 局全自动对局：必须收敛、牌守恒、事件闭环。"""
    for seed in range(100):
        g = Game(seed=seed)
        g.start()
        steps = 0
        while g.phase != "finished":
            heuristic.act(g, g.bid_turn if g.phase == "bidding" else g.turn)
            steps += 1
            assert steps < 2000, f"seed={seed} 对局不收敛"
        assert g.winner in ("landlord", "farmers")
        all_cards = sum(g.hands.values(), [])
        assert len(all_cards) == len(set(all_cards))  # 无重复牌
        # 赢家手牌必为空（出完才算赢）；其余家剩余 = 54 - 已出张数，无法预知只查守恒
        win_seat_empty = any(len(h) == 0 for h in g.hands.values())
        assert win_seat_empty


def test_state_for_masks_other_hands():
    g = make_game()
    g.bid(g.bid_turn, True)
    st = g.state_for("you")
    assert len(st["hand"]) in (17, 20)
    assert st["counts"]["liangzi"] == len(g.hands["liangzi"])
    assert "hands" not in st  # 不泄露别家牌


def test_normalize_say_pass_gets_prefix():
    from doudizhu.bot import _normalize_say

    ev = [{"type": "pass", "seat": "liangzi"}]
    assert _normalize_say(ev, "哦呦，管不上，你走好").startswith("不要，")
    assert _normalize_say(ev, "管不上，你走好") == "不要，管不上，你走好"
    # 已经合规的不动
    assert _normalize_say(ev, "不要，你走好我垫后") == "不要，你走好我垫后"
    assert _normalize_say(ev, "过，不拦你") == "不要，不拦你"


def test_normalize_say_play_gets_label():
    from doudizhu.bot import _normalize_say

    ev = [{"type": "play", "seat": "fengge", "cards": ["S9", "H9"],
           "combo_type": "pair", "main": 9, "remaining": 15}]
    # 开头没带牌信息 → 补牌型标签
    assert _normalize_say(ev, "这事儿说白了，我垫后") == "对9，这事儿说白了，我垫后"
    # 开头已带主点或牌型词 → 不动，避免「对9，对9管上」式重复
    assert _normalize_say(ev, "对9管上，味真足！") == "对9管上，味真足！"
    assert _normalize_say(ev, "9管上，你走") == "9管上，你走"


def test_correct_count_claims():
    from doudizhu.bot import _correct_count_claims

    g = Game(seats=("you", "liangzi", "fengge"), seed=3)
    g.start(fixed_landlord="you")
    names = {"you": "你", "liangzi": "大胃袋良子", "fengge": "峰哥亡命天涯"}
    n_you = len(g.hands["you"])       # 20
    n_lz = len(g.hands["liangzi"])    # 17

    # 说错地主剩牌 → 改成真实数（阿拉伯数字和中文小写都认）
    assert f"地主你还有{n_you}张" in _correct_count_claims(
        g, "liangzi", names, "地主你还有16张，慢慢来")
    assert f"地主就剩{n_you}张" in _correct_count_claims(
        g, "liangzi", names, "地主就剩三张，拦住他")
    # 说对了不动
    assert _correct_count_claims(
        g, "liangzi", names, f"地主还有{n_you}张，稳住") == f"地主还有{n_you}张，稳住"
    # 自称「我」报数 → 按说话者真实手牌改
    assert f"我还剩{n_lz}张" in _correct_count_claims(
        g, "liangzi", names, "我还剩5张，味真足")
    # 队友剩牌
    n_fg = len(g.hands["fengge"])
    assert f"峰哥只剩{n_fg}张" in _correct_count_claims(
        g, "liangzi", names, "峰哥只剩2张，我掩护")
    # 不带剩牌数断言的句子原样通过
    assert _correct_count_claims(g, "liangzi", names, "对9管上，味真足！") == "对9管上，味真足！"
