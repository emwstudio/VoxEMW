"""斗地主牌型逻辑：牌表示、牌型识别、大小比较、最小管牌生成。

牌表示：字符串。普通牌 = 花色前缀 + 点数，花色 S/H/D/C（黑红梅方），
点数 3-10/J/Q/K/A/2；王 = BJ（小王）/ RJ（大王）。如 "S3"、"H10"、"DA"。

点数大小：3<4<...<10<J<Q<K<A<2<小王<大王（值 3..17）。
顺子/连对/飞机的链条不允许包含 2 和王（链条最大点数为 A=14）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

SUITS = ("S", "H", "D", "C")
RANK_LABELS = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")
RANK_VALUES = {label: i + 3 for i, label in enumerate(RANK_LABELS)}  # 3..15
RANK_VALUES["BJ"] = 16
RANK_VALUES["RJ"] = 17
MAX_CHAIN_VALUE = 14  # 顺子/连对/飞机链条顶到 A

JOKERS = ("BJ", "RJ")


def full_deck() -> list[str]:
    return [s + r for s in SUITS for r in RANK_LABELS] + list(JOKERS)


def rank_value(card: str) -> int:
    if card in JOKERS:
        return RANK_VALUES[card]
    return RANK_VALUES[card[1:]]


def sort_cards(cards: list[str]) -> list[str]:
    """按点数升序（同点按花色稳定排序）。"""
    return sorted(cards, key=lambda c: (rank_value(c), c))


# ---------------------------------------------------------------------------
# 牌型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Combo:
    type: str          # single/pair/triple/triple_one/triple_pair/straight/
                       # pairs_seq/plane/plane_single/plane_pair/
                       # four_two_single/four_two_pair/bomb/rocket
    main: int          # 比较用的主点数（三带/飞机/四带看主体；顺子看最大点）
    length: int        # 总张数（比较时同型同长才可比）
    cards: tuple[str, ...]

    @property
    def is_bomb_like(self) -> bool:
        return self.type in ("bomb", "rocket")


def _is_chain(values: list[int]) -> bool:
    """已排序去重点数列是否构成链条（连续且顶点 ≤ A）。"""
    return (
        len(values) >= 2
        and values[-1] <= MAX_CHAIN_VALUE
        and all(values[i + 1] - values[i] == 1 for i in range(len(values) - 1))
    )


def analyze(cards: list[str]) -> Optional[Combo]:
    """识别一手牌的牌型；不是合法牌型返回 None。"""
    n = len(cards)
    if n == 0:
        return None
    values = sorted(rank_value(c) for c in cards)
    counts = Counter(values)  # 点数 -> 张数
    # 按（张数降序，点数升序）整理点数组，方便取主体/翅膀
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    sorted_cards = tuple(sort_cards(list(cards)))

    def combo(t: str, main: int) -> Combo:
        return Combo(t, main, n, sorted_cards)

    if n == 1:
        return combo("single", values[0])

    if n == 2:
        if set(values) == {16, 17}:
            return combo("rocket", 17)
        if len(counts) == 1:
            return combo("pair", values[0])
        return None

    if n == 3:
        if len(counts) == 1:
            return combo("triple", values[0])
        return None

    if n == 4:
        if len(counts) == 1:
            return combo("bomb", values[0])
        if groups[0][1] == 3:
            return combo("triple_one", groups[0][0])
        return None

    if n == 5:
        if groups[0][1] == 3 and groups[1][1] == 2:
            return combo("triple_pair", groups[0][0])
        if len(counts) == 5 and _is_chain(values):
            return combo("straight", values[-1])
        return None

    # ---- n >= 6 ----
    # 顺子：全单张、连续、≥5、不含 2/王
    if n >= 5 and len(counts) == n and _is_chain(values):
        return combo("straight", values[-1])

    # 连对：全对子、连续、≥3 对
    if n >= 6 and n % 2 == 0 and all(c == 2 for c in counts.values()):
        pair_ranks = sorted(counts)
        if _is_chain(pair_ranks):
            return combo("pairs_seq", pair_ranks[-1])

    # 炸弹已在 n==4 处理；这里是四带二
    if groups[0][1] == 4:
        rest = [v for v, c in groups[1:] for _ in range(c)]
        if n == 6 and len(rest) == 2:
            return combo("four_two_single", groups[0][0])
        if n == 8 and len(rest) == 4 and groups[1][1] == 2 and groups[2][1] == 2:
            return combo("four_two_pair", groups[0][0])
        return None

    # 飞机：k 个连续三张（k>=2），可带 k 张单或 k 对翅膀
    triple_ranks = sorted(v for v, c in counts.items() if c == 3)
    if triple_ranks:
        k = len(triple_ranks)
        if k >= 2 and _is_chain(triple_ranks):
            if n == 3 * k:
                return combo("plane", triple_ranks[-1])
            if n == 4 * k:
                return combo("plane_single", triple_ranks[-1])
            if n == 5 * k:
                wings = sorted(v for v, c in counts.items() if c != 3 for _ in range(c))
                if len(wings) == 2 * k:
                    wing_counts = Counter(wings)
                    if all(c == 2 for c in wing_counts.values()):
                        return combo("plane_pair", triple_ranks[-1])
        # 飞机主体可以只是部分三张（如 333444555 只取 333444）——
        # analyze 不做子集匹配（选牌出牌场景下玩家应一次选全），子集情况判非法。
    return None


def beats(a: Combo, b: Combo) -> bool:
    """a 是否能管上 b。"""
    if a.type == "rocket":
        return True
    if b.type == "rocket":
        return False
    if a.is_bomb_like and not b.is_bomb_like:
        return True
    if a.type != b.type or a.length != b.length:
        return False
    return a.main > b.main


# ---------------------------------------------------------------------------
# 最小管牌生成（兜底策略 / 自动出牌用）
# ---------------------------------------------------------------------------

def _rank_counts(hand: list[str]) -> Counter:
    return Counter(rank_value(c) for c in hand)


def _take(cards_by_rank: dict[int, list[str]], rank: int, count: int) -> list[str]:
    return cards_by_rank[rank][:count]


def _group_by_rank(hand: list[str]) -> dict[int, list[str]]:
    by_rank: dict[int, list[str]] = {}
    for c in sort_cards(hand):
        by_rank.setdefault(rank_value(c), []).append(c)
    return by_rank


def _find_chain(ranks: list[int], length: int, above: int) -> Optional[list[int]]:
    """在可用点数 ranks 中找长度为 length、顶点 > above 的最小链条。"""
    candidates = sorted(r for r in set(ranks) if r <= MAX_CHAIN_VALUE)
    for i in range(len(candidates) - length + 1):
        window = candidates[i : i + length]
        if window[-1] <= above:
            continue
        if all(window[j + 1] - window[j] == 1 for j in range(length - 1)):
            return window
    return None


def minimal_beat(hand: list[str], target: Combo, allow_bomb: bool = True) -> Optional[list[str]]:
    """从手牌中找能管上 target 的最小一手（优先非炸弹）；找不到返回 None。

    三带/飞机的翅膀取最小可用牌（不动用更大的三张主体，简单起见不避拆对）。
    """
    by_rank = _group_by_rank(hand)
    ranks = sorted(by_rank)

    def smallest_cards(count: int, exclude: set[int]) -> Optional[list[str]]:
        picked: list[str] = []
        for r in ranks:
            if r in exclude:
                continue
            picked.extend(by_rank[r])
            if len(picked) >= count:
                return picked[:count]
        return None

    result: Optional[list[str]] = None
    t = target.type
    main = target.main

    if t == "single":
        for r in ranks:
            if r > main:
                result = _take(by_rank, r, 1)
                break
    elif t == "pair":
        for r in ranks:
            if r > main and len(by_rank[r]) >= 2:
                result = _take(by_rank, r, 2)
                break
    elif t in ("triple", "triple_one", "triple_pair"):
        for r in ranks:
            if r > main and len(by_rank[r]) >= 3:
                base = _take(by_rank, r, 3)
                if t == "triple":
                    result = base
                elif t == "triple_one":
                    kick = smallest_cards(1, {r})
                    if kick:
                        result = base + kick
                else:
                    for r2 in ranks:
                        if r2 != r and len(by_rank[r2]) >= 2:
                            result = base + _take(by_rank, r2, 2)
                            break
                break
    elif t == "straight":
        chain = _find_chain(ranks, target.length, main)
        if chain:
            result = [c for r in chain for c in _take(by_rank, r, 1)]
    elif t == "pairs_seq":
        pair_ranks = [r for r in ranks if len(by_rank[r]) >= 2]
        chain = _find_chain(pair_ranks, target.length // 2, main)
        if chain:
            result = [c for r in chain for c in _take(by_rank, r, 2)]
    elif t in ("plane", "plane_single", "plane_pair"):
        triple_ranks = [r for r in ranks if len(by_rank[r]) >= 3]
        k = {"plane": target.length // 3, "plane_single": target.length // 4,
             "plane_pair": target.length // 5}[t]
        chain = _find_chain(triple_ranks, k, main)
        if chain:
            base = [c for r in chain for c in _take(by_rank, r, 3)]
            if t == "plane":
                result = base
            elif t == "plane_single":
                kick = smallest_cards(k, set(chain))
                if kick:
                    result = base + kick
            else:
                kick_ranks = [r for r in ranks if r not in chain and len(by_rank[r]) >= 2][:k]
                if len(kick_ranks) == k:
                    result = base + [c for r in kick_ranks for c in _take(by_rank, r, 2)]
    elif t in ("four_two_single", "four_two_pair"):
        for r in ranks:
            if r > main and len(by_rank[r]) == 4:
                base = _take(by_rank, r, 4)
                if t == "four_two_single":
                    kick = smallest_cards(2, {r})
                    if kick:
                        result = base + kick
                else:
                    kick_ranks = [r2 for r2 in ranks if r2 != r and len(by_rank[r2]) >= 2][:2]
                    if len(kick_ranks) == 2:
                        result = base + [c for r2 in kick_ranks for c in _take(by_rank, r2, 2)]
                break
    # target 是 bomb：只有更大的炸弹或王炸能管；rocket 无人能管，走下面炸弹逻辑也找不到

    if result is not None:
        return sort_cards(result)

    if not allow_bomb or target.type == "rocket":
        return None
    # 炸弹兜底：比 target（若是炸弹）大的最小炸弹
    for r in ranks:
        if len(by_rank[r]) == 4 and (not target.is_bomb_like or r > main):
            return sort_cards(_take(by_rank, r, 4))
    # 王炸
    if 16 in by_rank and 17 in by_rank:
        return ["BJ", "RJ"]
    return None


def lead_smallest_single(hand: list[str]) -> list[str]:
    """首出：最小单张（优先拆单不拆对的点数）。"""
    by_rank = _group_by_rank(hand)
    for r in sorted(by_rank):
        if len(by_rank[r]) == 1:
            return [by_rank[r][0]]
    return [sort_cards(hand)[0]]
