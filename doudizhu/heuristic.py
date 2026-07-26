"""兜底出牌策略：DeepSeek 决策不合法/失败时用它，保证游戏永远能推进。

刻意简单：首出打最小单（整手能一把出就一把出）；跟牌用最小管牌，
农民不压队友的大牌；炸弹/王炸只在残局或对手报单报双时才用。
"""

from __future__ import annotations

from typing import Optional

from .cards import analyze, lead_smallest_single, minimal_beat, rank_value, sort_cards
from .engine import Game


def should_bid(hand: list[str]) -> bool:
    """叫地主：有王炸/炸弹，或大牌（2 和王）≥3 张就叫。"""
    values = [rank_value(c) for c in hand]
    big = sum(1 for v in values if v >= 15)
    has_bomb = len(values) - len(set(values)) >= 3 and any(
        values.count(v) == 4 for v in set(values)
    )
    has_rocket = 16 in values and 17 in values
    return has_rocket or has_bomb or big >= 3


def _can_go_out(hand: list[str], last_combo) -> Optional[list[str]]:
    """整手牌合法且管得上（或自由出牌）→ 一把出完。"""
    combo = analyze(hand)
    if combo is None:
        return None
    if last_combo is not None:
        from .cards import beats

        if not beats(combo, last_combo):
            return None
    return sort_cards(hand)


def choose_play(game: Game, seat: str) -> Optional[list[str]]:
    """返回要出的牌列表；不要返回 None。调用方保证轮到 seat 且在 playing 阶段。"""
    hand = game.hands[seat]
    last = game.last_play.combo if game.last_play else None

    out = _can_go_out(hand, last)
    if out is not None:
        return out

    if last is None:
        return lead_smallest_single(hand)

    # 农民队友出了大牌（A/2/王）就不压，让给队友走
    last_seat = game.last_play.seat
    is_farmer = seat != game.landlord
    teammate_play = is_farmer and last_seat != game.landlord
    if teammate_play and game.last_play.combo.main >= 14 and not game.last_play.combo.is_bomb_like:
        return None

    # 对手报单/报双时允许动炸弹；平时只用普通牌管
    opponents = [s for s in game.seats if s != seat]
    if is_farmer:
        threat = min(len(game.hands[game.landlord]), 99)
    else:
        threat = min((len(game.hands[s]) for s in opponents), default=99)
    allow_bomb = threat <= 2 or len(hand) <= 5

    return minimal_beat(hand, last, allow_bomb=allow_bomb)


def act(game: Game, seat: str) -> list[dict]:
    """替 seat 执行一步（叫地主或出牌/不要），返回引擎事件。"""
    if game.phase == "bidding":
        return game.bid(seat, should_bid(game.hands[seat]))
    play = choose_play(game, seat)
    if play is None:
        return game.pass_turn(seat)
    return game.play(seat, play)
