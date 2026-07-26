"""bot 出牌大脑：DeepSeek 按人设+牌局决策（JSON），引擎校验，不合法回退兜底策略。

llm 以可调用对象注入（messages -> str），方便本地冒烟/实例共用；
生产环境即 doudizhu.deepseek.chat_complete 的偏函数。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from . import heuristic
from .cards import RANK_VALUES
from .engine import Game, IllegalMove
from .persona import Persona

logger = logging.getLogger(__name__)

_VALUE_LABELS = {v: k for k, v in RANK_VALUES.items()}
_VALUE_LABELS[16] = "小王"  # 口语报牌：BJ/RJ 代码不能念出来
_VALUE_LABELS[17] = "大王"


def _combo_label(combo_type: str, main: Optional[int]) -> str:
    """牌型+主点转口语标签（报牌用）：单张K / 对9 / 三个2 / 王炸。"""
    r = _VALUE_LABELS.get(main, "") if main else ""
    return {
        "single": f"单张{r}", "pair": f"对{r}", "triple": f"三个{r}",
        "triple_one": f"三带一（{r}）", "triple_pair": f"三带二（{r}）",
        "straight": "顺子", "pairs_seq": "连对",
        "plane": "飞机", "plane_single": "飞机带单", "plane_pair": "飞机带对",
        "four_two_single": f"四带二（{r}）", "four_two_pair": f"四带两对（{r}）",
        "bomb": f"{r}炸", "rocket": "王炸",
    }.get(combo_type, combo_type)

LLMFn = Callable[[list[dict]], str]

_RULES_HINT = """\
斗地主规则速览：3 人一副牌（54 张），地主 20 张打两家农民（农民是队友，任一家先出完农民赢）。
牌表示：花色 S/H/D/C + 点数 3-10/J/Q/K/A/2，王：BJ 小王、RJ 大王（BJ+RJ 同出是王炸，最大）。
牌型：单张、对子、三张、三带一、三带二、顺子（≥5连）、连对（≥3连对）、飞机、四带二、炸弹（4同）、王炸。
管牌必须同型同长且主点严格更大（相等也不行！）；炸弹/王炸可管一切。管不上或不想管就 pass；轮到你首出（上一手为空）不能 pass。"""

_OUTPUT_HINT = """\
严格输出一行 JSON（不要 markdown、不要解释）：
{"action": {"kind": "play", "cards": ["S3", "H3"]} 或 {"kind": "pass"} 或 {"kind": "bid", "call": true}, "say": "你的台词"}
say 要求（和出牌决策同等重要）：
- 只说一句话（≤25字），口语干脆；口头禅要自然融入牌局，不许硬塞
- 句式铁律：先报动作再接话——不要就以「不要」开头（如「不要，你走好我垫后」）；
  出牌就先报你出的牌再接话（如「对9管上，味真足！」「顺子，我看你拿啥接」）
- 报牌用口语牌名（大王/小王/2/A/K/Q/J/10），严禁说 BJ/RJ 或 S/H/D/C 花色代码
- 你是斗地主老手，说话要有牌桌内容，按局势选题，严禁每句都一个套路
  （尤其「也敢出」式不许连用）：
  · 有人只剩一两张（见⚠️警报）→优先说：对手要溜就喊「报单了！拦住他」，队友要走了就喊掩护
  · 队友出得大→捧+表态配合（「好牌，我不压你，你走」）
  · 自己管不上队友的牌→团结让路（「你走好，我垫后」）
  · 对手出大牌/炸弹→嘴上不服（「让他狂，我憋着炸呢」）
  · 局势胶着→点名剩牌数施压（「地主还有8张，稳住别浪」）
  · 平常一手→一句话点评上一家（带上他的牌）+报自己的牌
- 剩牌数是硬数据，不许自己估：报别家剩几张必须照抄「各家剩余」里的数字；
  报自己剩几张 = 你的手牌数 − 你这次要出的张数（「你的手牌」是出牌前的数量）
- 敌我铁律（最高优先级）：你是农民，农民的任务就是合伙干掉地主——
  · 每句话的矛头必须指向地主：点评地主的牌就怼（「这也敢出？」「让他狂」），
    点评队友的牌只能捧+配合（「好牌，看他拿啥接」「你走，我垫后」）
  · 严禁损队友，玩笑式互损也不行（不许说拉黑队友、不惯着队友这种话）；
    严禁夸地主。你们是两个人打地主一个，枪口一致对外"""

# 经典口头禅池：每句台词必须自然带一句，一局内不重复（server 按局记录已用的）。
# 出自 personas/*.md 的「表达DNA/实录参照」，别凭空造梗。
_CATCHPHRASES = {
    "liangzi": [
        "味真足", "搂他", "这一块", "多了不说，少了不唠",
        "活着吃，死了算", "你良弟", "我小手一抬直接给你拉黑了", "哦呦",
    ],
    "fengge": [
        "这是个好事儿啊", "恰恰相反，这并不是个好事儿", "想连接了",
        "这事儿说白了", "b友们", "国家一级登山运动员", "纪录片导演",
    ],
}


def _catchphrase_hint(seat: str, used_phrases: Optional[list[str]]) -> str:
    """口头禅要求（动态拼进 prompt）：必须带一句、一局内不重复。"""
    pool = _CATCHPHRASES.get(seat, [])
    if not pool:
        return ""
    unused = [p for p in pool if p not in (used_phrases or [])]
    if unused:
        menu = "、".join(f"「{p}」" for p in unused)
        return (
            f"经典口头禅要求：这句台词必须自然融入一句你的经典口头禅（从里面挑：{menu}；"
            "本局已用过的不许再用，也不许整句复制之前的台词，要结合牌局换个说法）。"
        )
    menu = "、".join(f"「{p}」" for p in pool)
    return (
        f"经典口头禅要求：这句台词必须自然融入一句你的经典口头禅（你的口头禅库：{menu}；"
        "本局已轮完一轮，可以重新启用，但不许和上两句用同一句）。"
    )


def find_used_phrases(seat: str, say: str) -> list[str]:
    """台词里实际用到了池里的哪些口头禅（server 据此记录，防重）。"""
    return [p for p in _CATCHPHRASES.get(seat, []) if p in say]


_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_to_int(s: str) -> Optional[int]:
    """阿拉伯数字或中文小写（一~二十）→ int，认不出返回 None。"""
    if s.isdigit():
        return int(s)
    if s in _CN_NUM:
        return _CN_NUM[s]
    if s.startswith("十") and len(s) == 2:
        return 10 + _CN_NUM.get(s[1], 0)
    if s.endswith("十") and len(s) == 2:
        return _CN_NUM.get(s[0], 0) * 10
    if "十" in s and len(s) == 3:
        a, b = s.split("十")
        return _CN_NUM.get(a, 0) * 10 + _CN_NUM.get(b, 0)
    return None


# 「地主你还有16张」「我就剩两张」「峰哥只剩1张」这类剩牌数断言
_COUNT_CLAIM_RE = re.compile(
    r"(地主|良子|峰哥|队友|你|我)(?:还|就|只)?(?:有|剩下?|剩)"
    r"(\d{1,2}|[一二两三四五六七八九十]{1,3})张"
)


def _correct_count_claims(game: Game, seat: str, names: dict[str, str], say: str) -> str:
    """剩牌数断言的确定性校正：和引擎真实数不符就改成真实数。

    prompt 里「照抄各家剩余」只能管个大概——deepseek-v4-flash 会按上一家
    出牌自己再减一次。在服务端用真值硬改，报单/施压的数才不会离谱。
    注意在动作已作用到引擎后调用：此时 game.hands 就是出牌后的真实剩余。
    """
    def true_count(subj: str) -> Optional[int]:
        if subj == "我":
            return len(game.hands[seat])
        if subj == "地主":
            return len(game.hands[game.landlord]) if game.landlord else None
        if subj == "队友":
            mate = next((s for s in game.seats if s not in (seat, game.landlord)), None)
            return len(game.hands[mate]) if mate else None
        if subj == "你":  # bot 嘴里的「你」基本指地主（真人）
            target = game.landlord if game.landlord and game.landlord != seat else None
            return len(game.hands[target]) if target else None
        for s in game.seats:  # 良子/峰哥 这类直呼其名
            if s != seat and (subj == s or subj in names.get(s, "")):
                return len(game.hands[s])
        return None

    def repl(m: re.Match) -> str:
        n = true_count(m.group(1))
        if n is None or _cn_to_int(m.group(2)) == n:
            return m.group(0)
        return m.group(0).replace(m.group(2), str(n))

    return _COUNT_CLAIM_RE.sub(repl, say)


# pass 台词必须以「不要」开头（用户要求：说不要就先把「不要」说出口），
# 弱模型八成能守规矩，剩下两成服务端硬改——「管不上…」「过…」一律归一成「不要，…」
def _normalize_say(events: list[dict], say: str) -> str:
    """句式铁律的确定性兜底：不要就「不要」开头，出牌就先报牌名。

    LLM 大部分时候遵守，但 deepseek-v4-flash 会漏（「哦呦，管不上…」），
    在服务端硬改比靠 prompt 祈祷可靠。
    """
    say = say.strip().lstrip("，,")
    for e in events:
        if e["type"] == "pass":
            if not say.startswith("不要"):
                say = f"不要，{say.lstrip('过，, ') or '你走好'}"
            return say
        if e["type"] == "play":
            label = _combo_label(e.get("combo_type", ""), e.get("main"))
            main_label = _VALUE_LABELS.get(e.get("main"), "") if e.get("main") else ""
            head = say[:8]
            # 开头已经带牌信息（牌型词或主点）就不动，避免「对9，对9管上」式重复
            if (main_label and main_label in head) or any(
                k in head for k in ("单", "对", "三", "顺", "连对", "飞机", "炸", "王", "带")
            ):
                return say
            return f"{label}，{say}"
    return say


def snapshot_text(game: Game, seat: str, names: dict[str, str]) -> str:
    """seat 视角的牌局文字描述（给 LLM 的上下文）。"""
    lines = []
    if game.phase == "bidding":
        lines.append("阶段：叫地主（你可以选择叫或不叫）")
    else:
        role = "地主" if seat == game.landlord else "农民"
        if seat != game.landlord:
            mate = next(s for s in game.seats if s not in (seat, game.landlord))
            role += f"（队友：{names.get(mate, mate)}）"
        lines.append(f"阶段：出牌，你的身份：{role}")
    hand = game.hands[seat]
    lines.append(f"你的手牌（{len(hand)} 张）：{' '.join(hand)}")
    if seat == game.landlord:
        lines.append(f"底牌：{' '.join(game.bottom)}")
    counts = "，".join(f"{names.get(s, s)} {len(game.hands[s])} 张" for s in game.seats)
    lines.append(f"各家剩余：{counts}")
    # 报单/剩牌警报：牌桌话题的重要素材（拦人/掩护），单独点出来
    low = [(names.get(s, s), len(game.hands[s])) for s in game.seats if 0 < len(game.hands[s]) <= 2]
    if low:
        lines.append("⚠️ 警报：" + "，".join(f"{w} 只剩 {n} 张！" for w, n in low))
    if game.last_play:
        lp = game.last_play
        combo = lp.combo
        main_label = _VALUE_LABELS[combo.main]
        lines.append(
            f"上一手：{names.get(lp.seat, lp.seat)} 出了 {' '.join(combo.cards)}（{combo.type}，主点 {main_label}）"
        )
        if combo.is_bomb_like:
            lines.append("→ 要管只能出更大的炸弹或王炸，管不上就 pass")
        else:
            lines.append(
                f"→ 你这一手必须：{combo.type}（{combo.length} 张）且主点严格大于 {main_label}，"
                "或者是炸弹/王炸；不符合就是非法出牌，没有能管的就 pass"
            )
    else:
        lines.append("上一手：无（轮到你首出）" if game.phase == "playing" else "")
    # 上一家动作（出牌 or 不要）：嘴炮点评的唯一对象
    if game.phase == "playing" and game.last_action:
        la = game.last_action
        who = names.get(la["seat"], la["seat"])
        if la["type"] == "pass":
            lines.append(f"上一家动作：{who} 不要")
        elif la["type"] == "play":
            lines.append(f"上一家动作：{who} 出了牌（即上面「上一手」那手）")
    if game.bombs:
        lines.append(f"本局炸弹数：{game.bombs}")
    return "\n".join(l for l in lines if l)


def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出里抠出 JSON 对象（容忍 markdown 围栏和前后废话）。"""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# LLM 决策失败回退兜底策略时的模板台词：也按人设来，不然一兜底就出戏。
# play 模板里的 {label} 会填成具体牌型（单张K/对9/王炸）
_FALLBACK_SAY = {
    "liangzi": {
        "play": "{label}搂他，味真足！",
        "pass": "不要，这把真管不上，你良弟先憋着。",
        "bid": "叫地主！这牌味真足！",
        "nobid": "不叫，这牌太拉了。",
    },
    "fengge": {
        "play": "{label}管上，这是个好事儿啊。",
        "pass": "不要，这牌没法接，你先走。",
        "bid": "叫地主，这把我稳了。",
        "nobid": "不叫，b友们这牌我不接。",
    },
}


def _fallback_say(events: list[dict], seat: str) -> str:
    """兜底策略执行后配的模板台词（带人设口头禅、带具体牌型）。"""
    t = _FALLBACK_SAY.get(seat, {})
    for e in events:
        if e["type"] == "play":
            label = _combo_label(e.get("combo_type", ""), e.get("main"))
            return t.get("play", "{label}，管上！").format(label=label)
        if e["type"] == "pass":
            return t.get("pass", "不要，过。")
        if e["type"] == "bid":
            default = "叫地主！" if e["call"] else "不叫。"
            return t.get("bid" if e["call"] else "nobid", default)
    return "嗯。"


def decide_and_act(
    game: Game,
    seat: str,
    persona: Persona,
    llm: LLMFn,
    names: dict[str, str],
    used_phrases: Optional[list[str]] = None,
) -> tuple[list[dict], str]:
    """DeepSeek 决策并作用到引擎；返回 (事件列表, 台词)。

    任何一步失败（LLM 异常/JSON 坏/动作非法）都回退 heuristic，游戏必须推进。
    used_phrases：本局已用过的口头禅（server 按局记录），用于防重。
    """
    phase_hint = "叫地主" if game.phase == "bidding" else "出牌"
    user_prompt = (
        f"{snapshot_text(game, seat, names)}\n\n"
        f"现在轮到你{phase_hint}。结合你的手牌和局势做决策。\n"
        f"{_catchphrase_hint(seat, used_phrases)}\n{_OUTPUT_HINT}"
    )
    messages = [
        {"role": "system", "content": persona.body + "\n\n" + _RULES_HINT},
        {"role": "user", "content": user_prompt},
    ]
    # 最多试 2 次：第一次动作不合法时把错误喂回去让 LLM 改（它多半第二次就对了），
    # 不然好台词会跟着非法动作一起被丢进兜底模板——台词复读机的真正来源
    raw = ""
    for attempt in range(2):
        try:
            raw = llm(messages)
            data = _extract_json(raw)
        except Exception as e:
            logger.warning("bot %s LLM 决策异常（第%d次）: %s", seat, attempt + 1, e)
            data = None
        if data is not None:
            action = data.get("action") or {}
            say = str(data.get("say") or "").strip()
            try:
                events = _apply_action(game, seat, action)
                if not say:
                    say = _fallback_say(events, seat)
                say = _normalize_say(events, say)
                # 动作已作用到引擎，hands 是出牌后真值，可校正剩牌数断言
                return events, _correct_count_claims(game, seat, names, say)
            except IllegalMove as e:
                err = str(e)
                logger.info("bot %s 第%d次动作不合法（%s）。raw=%.300s", seat, attempt + 1, err, raw)
        else:
            err = "输出不是合法 JSON"
        messages = messages + [
            {"role": "assistant", "content": raw or ""},
            {"role": "user", "content": (
                f"你刚才的输出不合法：{err}。"
                "严格按你的手牌和「上一手」的限制重新决策，输出一行 JSON。"
            )},
        ]

    events = heuristic.act(game, seat)
    return events, _fallback_say(events, seat)


def _apply_action(game: Game, seat: str, action: dict) -> list[dict]:
    kind = action.get("kind")
    if game.phase == "bidding":
        if kind != "bid":
            raise IllegalMove(f"叫地主阶段收到 {kind!r}")
        return game.bid(seat, bool(action.get("call")))
    if kind == "pass":
        return game.pass_turn(seat)
    if kind == "play":
        cards = action.get("cards")
        if not isinstance(cards, list) or not all(isinstance(c, str) for c in cards):
            raise IllegalMove(f"cards 非法: {cards!r}")
        return game.play(seat, cards)
    raise IllegalMove(f"未知 action: {kind!r}")


def react(
    game: Game,
    seat: str,
    persona: Persona,
    llm: LLMFn,
    names: dict[str, str],
    event_desc: str,
) -> Optional[str]:
    """对牌局事件（炸弹/报单/胜负等）生成一句人设反应。失败返回 None（静默跳过）。"""
    counts = "，".join(f"{names.get(s, s)} {len(game.hands[s])} 张" for s in game.seats)
    if game.landlord and seat != game.landlord:
        mate = next(s for s in game.seats if s not in (seat, game.landlord))
        role = f"农民（队友：{names.get(mate, mate)}）"
    elif game.landlord:
        role = "地主"
    else:
        role = "未定（叫地主阶段）"
    messages = [
        {"role": "system", "content": persona.body},
        {
            "role": "user",
            "content": (
                f"斗地主牌局快报：{event_desc}\n你的身份：{role}\n当前各家剩余：{counts}\n"
                "用你的口吻说一句话（不超过 20 字，带上你的经典口头禅，立场分明——"
                "农民挺队友怼地主、地主一挑二句句回怼，任何情况下都不许夸对手；"
                "只输出台词本身，不要任何解释）。"
            ),
        },
    ]
    try:
        return llm(messages).strip().strip('"「」')
    except Exception as e:
        logger.warning("bot %s 反应生成失败: %s", seat, e)
        return None
