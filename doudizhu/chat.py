"""桌上闲聊：你说的话分流成「游戏口令」或「聊天」，聊天路由给某个 bot 回复。

口令（不要/过/叫地主等）只在轮到你的对应阶段生效；聊天按点名路由
（提到「峰哥」（含 STT 同音「风哥/锋哥」）→峰哥，提到「良子/良弟」→良子），
没点名就给对立阵营（农民说话→地主接话回怼），再否则最近有动作的 bot。
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .bot import LLMFn, snapshot_text
from .engine import Game, IllegalMove
from .persona import Persona

logger = logging.getLogger(__name__)

# 口令 -> (归一化动作)；按最长匹配优先
_PASS_WORDS = ("不要", "不出", "不管", "要不起", "过", "pass")
_BID_YES_WORDS = ("叫地主", "抢地主", "叫", "抢")
_BID_NO_WORDS = ("不叫地主", "不抢", "不叫")


def _normalize(text: str) -> str:
    return re.sub(r"[，。！？!?,.\s]", "", text)


def try_command(game: Game, user_seat: str, text: str) -> Optional[dict]:
    """识别游戏口令并执行；不是口令返回 None，是口令但当前不适用返回 applied=False。"""
    t = _normalize(text)
    if not t:
        return None

    if game.phase == "bidding" and game.bid_turn == user_seat:
        for w in _BID_NO_WORDS:
            if t == w:
                return {"kind": "command", "action": "bid", "call": False,
                        "events": game.bid(user_seat, False), "applied": True}
        for w in _BID_YES_WORDS:
            if t == w:
                return {"kind": "command", "action": "bid", "call": True,
                        "events": game.bid(user_seat, True), "applied": True}
        if t in _BID_NO_WORDS + _BID_YES_WORDS:
            return {"kind": "command", "applied": False, "reason": "还没轮到你叫地主"}

    if game.phase == "playing" and game.turn == user_seat and game.last_play is not None:
        for w in _PASS_WORDS:
            if t == w:
                return {"kind": "command", "action": "pass",
                        "events": game.pass_turn(user_seat), "applied": True}
    elif t in _PASS_WORDS and game.phase == "playing":
        return {"kind": "command", "applied": False,
                "reason": "现在不需要你不要（没轮到你或轮到你首出）"}
    return None


# 常见昵称 + STT 同音误写（语音识别经常把「峰」听成「风/锋」）
_FENGGE_ALIASES = ("峰哥", "风哥", "锋哥", "峰峰")
_LIANGZI_ALIASES = ("良子", "良弟", "梁子", "粮子")


def route_bot(game: Game, user_seat: str, text: str, bots: dict[str, Persona]) -> str:
    """选接话的 bot：点名优先（含同音误写）；没点名时给对立阵营
    （农民说话→地主接话回怼），再否则最近有动作的 bot，最后第一个。"""
    for seat, persona in bots.items():
        if persona.name in text:
            return seat
    if any(a in text for a in _FENGGE_ALIASES) and "fengge" in bots:
        return "fengge"
    if any(a in text for a in _LIANGZI_ALIASES) and "liangzi" in bots:
        return "liangzi"
    # 没点名：用户在怼人时大多是冲地主去的，让地主回怼，而不是队友搭腔
    if game.landlord and game.landlord in bots and user_seat != game.landlord:
        return game.landlord
    if game.last_play and game.last_play.seat in bots:
        return game.last_play.seat
    return next(iter(bots))


def reply_chat(
    game: Game,
    user_seat: str,
    text: str,
    bot_seat: str,
    persona: Persona,
    llm: LLMFn,
    names: dict[str, str],
) -> Optional[str]:
    """bot 按人设接话（知道牌局、知道敌我）。失败返回 None。"""
    # 敌我关系：都是农民=队友；bot 是地主→用户是对手农民；反之用户是地主=对手
    if game.landlord and bot_seat != game.landlord and user_seat != game.landlord:
        relation = "你的农民队友"
        stance = "他是你队友，捧着护着，接他的话一起怼地主"
    elif game.landlord and bot_seat == game.landlord:
        relation = "你的对手（农民，跟另一家农民一伙的）"
        stance = "他跟你对着干，必须回怼，地主一挑二嘴上不能输"
    else:
        relation = "你的对手（地主）"
        stance = "他就是地主，怼他"
    messages = [
        {"role": "system", "content": persona.body},
        {
            "role": "user",
            "content": (
                f"你们正在语音连麦斗地主。{snapshot_text(game, bot_seat, names)}\n\n"
                f"{names.get(user_seat, user_seat)}（{relation}）嘴上跟你说：「{text}」\n"
                "注意：他只是在嘴上点评牌局，不是在出牌——别催他出牌，也别以为他要管牌。\n"
                "如果他在点评你刚出的牌（哪怕是贬损），就为你那手牌吹牛找补，把它说得又妙又牛"
                "（神来之笔、投石问路、你们根本看不懂），顺势回怼他不懂牌；"
                "如果他在帮腔你队友或自夸，就拆台回怼。"
                f"{stance}。用你的口吻回一句，口语，不超过 30 字，"
                "带上你的经典口头禅，只输出回复本身。"
            ),
        },
    ]
    try:
        return llm(messages).strip().strip('"「」')
    except Exception as e:
        logger.warning("bot %s 聊天回复失败: %s", bot_seat, e)
        return None


def handle_user_text(
    game: Game,
    user_seat: str,
    text: str,
    bots: dict[str, Persona],
    llm: LLMFn,
    names: dict[str, str],
) -> dict:
    """总入口：先口令后闲聊。返回 {kind: command|chat|ignored, ...}。"""
    try:
        cmd = try_command(game, user_seat, text)
    except IllegalMove as e:
        return {"kind": "command", "applied": False, "reason": str(e)}
    if cmd is not None:
        return cmd

    # 噪声过滤：VAD 对翻书/咳嗽/环境音也会切出片段，STT 出一两个字的
    # 垃圾转写——这种不该触发 bot 闲聊搭话（口令已在上面处理过，不受影响）
    if len(_normalize(text)) < 2:
        return {"kind": "ignored", "reason": "转写过短，当噪声忽略"}

    bot_seat = route_bot(game, user_seat, text, bots)
    reply = reply_chat(game, user_seat, text, bot_seat, bots[bot_seat], llm, names)
    if reply is None:
        return {"kind": "ignored", "reason": "llm failed"}
    return {"kind": "chat", "bot": bot_seat, "reply": reply}
