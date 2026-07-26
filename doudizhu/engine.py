"""斗地主游戏引擎：状态机（发牌 → 叫地主 → 出牌轮转 → 胜负）。

座位固定 3 个，id 由调用方给（默认 you/liangzi/fengge）。
所有动作方法返回事件 dict 列表，供 server 驱动嘴炮/字幕/广播；
不合法动作抛 IllegalMove。纯逻辑，无 IO，可本地单测。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .cards import Combo, analyze, beats, full_deck, sort_cards

DEFAULT_SEATS = ("you", "liangzi", "fengge")


class IllegalMove(Exception):
    pass


@dataclass
class LastPlay:
    seat: str
    combo: Combo


@dataclass
class Game:
    seats: tuple[str, ...] = DEFAULT_SEATS
    seed: Optional[int] = None
    # ---- 运行态（start 后填充）----
    hands: dict[str, list[str]] = field(default_factory=dict)
    bottom: list[str] = field(default_factory=list)
    phase: str = "idle"           # idle/bidding/playing/finished
    landlord: Optional[str] = None
    bid_turn: Optional[str] = None
    _bid_passes: int = 0
    turn: Optional[str] = None
    last_play: Optional[LastPlay] = None
    last_action: Optional[dict] = None  # 上一家动作 {"type": play/pass/bid, "seat": ...}（嘴炮点评用）
    _passes: int = 0
    winner: Optional[str] = None   # "landlord" / "farmers"
    spring: bool = False
    bombs: int = 0
    _played_counts: dict[str, int] = field(default_factory=dict)
    _rng: random.Random = field(default=None)  # type: ignore[assignment]
    round_no: int = 0

    def __post_init__(self) -> None:
        if self._rng is None:
            self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # 流程
    # ------------------------------------------------------------------

    def start(self, fixed_landlord: Optional[str] = None) -> list[dict]:
        """发牌（或没人叫地主后的重发）。

        fixed_landlord 给定时跳过叫地主：直接定地主、地主拿底牌先出；
        否则进入叫地主阶段。
        """
        self.round_no += 1
        deck = full_deck()
        self._rng.shuffle(deck)
        self.hands = {s: sort_cards(deck[i * 17 : (i + 1) * 17]) for i, s in enumerate(self.seats)}
        self.bottom = sort_cards(deck[51:])
        self.landlord = None
        self.bid_turn = None
        self._bid_passes = 0
        self.turn = None
        self.last_play = None
        self.last_action = None
        self._passes = 0
        self.winner = None
        self.spring = False
        self.bombs = 0
        self._played_counts = {s: 0 for s in self.seats}
        if fixed_landlord is not None:
            if fixed_landlord not in self.seats:
                raise ValueError(f"fixed_landlord 不在座位里: {fixed_landlord}")
            self.phase = "playing"
            self.landlord = fixed_landlord
            self.hands[fixed_landlord] = sort_cards(self.hands[fixed_landlord] + self.bottom)
            self.turn = fixed_landlord
            return [
                {"type": "deal", "round": self.round_no, "bid_turn": None},
                {"type": "landlord", "seat": fixed_landlord, "bottom": list(self.bottom)},
            ]
        self.phase = "bidding"
        self.bid_turn = self._rng.choice(self.seats)
        return [{"type": "deal", "round": self.round_no, "bid_turn": self.bid_turn}]

    def _next_seat(self, seat: str) -> str:
        i = self.seats.index(seat)
        return self.seats[(i + 1) % len(self.seats)]

    def bid(self, seat: str, call: bool) -> list[dict]:
        """叫地主。简化规则：轮流叫/不叫，首叫即地主；全不叫则重发。"""
        if self.phase != "bidding":
            raise IllegalMove(f"当前不在叫地主阶段（{self.phase}）")
        if seat != self.bid_turn:
            raise IllegalMove(f"还没轮到 {seat} 叫地主（当前 {self.bid_turn}）")
        events: list[dict] = [{"type": "bid", "seat": seat, "call": call}]
        self.last_action = {"type": "bid", "seat": seat}
        if call:
            self.landlord = seat
            self.hands[seat] = sort_cards(self.hands[seat] + self.bottom)
            self.phase = "playing"
            self.turn = seat
            events.append({"type": "landlord", "seat": seat, "bottom": list(self.bottom)})
        else:
            self._bid_passes += 1
            if self._bid_passes >= len(self.seats):
                events.append({"type": "no_bid_redeal"})
                events += self.start()
            else:
                self.bid_turn = self._next_seat(seat)
        return events

    def play(self, seat: str, cards: list[str]) -> list[dict]:
        """出牌。cards 为牌字符串列表；必须合法且管得上上一手（首出除外）。"""
        if self.phase != "playing":
            raise IllegalMove(f"当前不在出牌阶段（{self.phase}）")
        if seat != self.turn:
            raise IllegalMove(f"还没轮到 {seat} 出牌（当前 {self.turn}）")
        if not cards:
            raise IllegalMove("出的牌不能为空（不要请用 pass_turn）")
        hand = self.hands[seat]
        missing = [c for c in cards if c not in hand]
        if missing:
            raise IllegalMove(f"手里没有这些牌: {missing}")
        combo = analyze(cards)
        if combo is None:
            raise IllegalMove(f"不是合法牌型: {cards}")
        if self.last_play is not None and not beats(combo, self.last_play.combo):
            raise IllegalMove(
                f"{combo.type}(main={combo.main}) 管不上 {self.last_play.combo.type}(main={self.last_play.combo.main})"
            )
        for c in cards:
            hand.remove(c)
        self.last_play = LastPlay(seat, combo)
        self.last_action = {"type": "play", "seat": seat}
        self._passes = 0
        self._played_counts[seat] += len(cards)
        events: list[dict] = [{
            "type": "play", "seat": seat, "cards": sort_cards(cards),
            "combo_type": combo.type, "main": combo.main,
            "remaining": len(hand),
        }]
        if combo.is_bomb_like:
            self.bombs += 1
            events.append({"type": "bomb", "seat": seat, "combo_type": combo.type})
        if len(hand) == 1:
            events.append({"type": "last_card", "seat": seat})
        if not hand:
            self.phase = "finished"
            self.winner = "landlord" if seat == self.landlord else "farmers"
            losers = [s for s in self.seats if s != seat]
            if seat == self.landlord:
                self.spring = all(self._played_counts[s] == 0 for s in losers)
            else:
                self.spring = self._played_counts[self.landlord] <= 3  # 地主只出过一手(简化)
            events.append({
                "type": "finish", "winner": self.winner, "win_seat": seat,
                "spring": self.spring, "bombs": self.bombs,
            })
        else:
            self.turn = self._next_seat(seat)
        return events

    def pass_turn(self, seat: str) -> list[dict]:
        """不要。首出（自由出牌权）时不能不叫。"""
        if self.phase != "playing":
            raise IllegalMove(f"当前不在出牌阶段（{self.phase}）")
        if seat != self.turn:
            raise IllegalMove(f"还没轮到 {seat}（当前 {self.turn}）")
        if self.last_play is None:
            raise IllegalMove("轮到你首出，不能不要")
        events: list[dict] = [{"type": "pass", "seat": seat}]
        self.last_action = {"type": "pass", "seat": seat}
        self._passes += 1
        if self._passes >= len(self.seats) - 1:
            # 其他人都不要，出牌权回到上一手玩家，自由出牌
            winner_of_trick = self.last_play.seat
            self.last_play = None
            self._passes = 0
            self.turn = winner_of_trick
            events.append({"type": "free_turn", "seat": winner_of_trick})
        else:
            self.turn = self._next_seat(seat)
        return events

    # ------------------------------------------------------------------
    # 视图
    # ------------------------------------------------------------------

    def state_for(self, seat: str) -> dict:
        """单个玩家视角的状态（别家手牌只给数量）。"""
        return {
            "phase": self.phase,
            "round": self.round_no,
            "seats": list(self.seats),
            "you": seat,
            "hand": list(self.hands.get(seat, [])),
            "counts": {s: len(self.hands.get(s, [])) for s in self.seats},
            "bottom": list(self.bottom) if self.landlord else [],
            "landlord": self.landlord,
            "bid_turn": self.bid_turn,
            "turn": self.turn,
            "last_play": (
                {
                    "seat": self.last_play.seat,
                    "cards": list(self.last_play.combo.cards),
                    "combo_type": self.last_play.combo.type,
                }
                if self.last_play
                else None
            ),
            "winner": self.winner,
            "spring": self.spring,
            "bombs": self.bombs,
        }
