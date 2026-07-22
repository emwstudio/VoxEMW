"""峰哥×良子 双唠模式的纯逻辑部分（对话史映射、回合轮转、角色配置）。

零重依赖（stdlib only），可在无 GPU/livekit 的机器上单测。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 双唠默认回合上限（不含良子开场白）
DEFAULT_MAX_TURNS = 14
#: 默认开场主题（DUET_TOPIC 环境变量可覆盖）
DEFAULT_TOPIC = "良子劝峰哥使劲造，峰哥亡命天涯蹭饭"

#: 默认脚本节拍（DUET_BEATS 环境变量可覆盖，格式 "key:指令|key:指令"）。
#: 每拍一句指令，让模型用自己的话和口头禅演出来；台词务必短——一拍一句话。
DEFAULT_BEATS: list[tuple[str, str]] = [
    (
        "liangzi",
        "夸峰哥本人长得帅、皮肤白，再自嘲你自己晒黑了。只用一句话说完，别展开。",
    ),
    (
        "fengge",
        "用「我还白啊？」开头，说良子晒黑是好事——热爱运动、喜欢拥抱大自然。一两句话说完。",
    ),
    (
        "liangzi",
        "夸峰哥说话就是让人爱听，然后提议：咱们给各位b友们录个祝福吧。一句话说完。",
    ),
    (
        "fengge",
        "答应录祝福：必须先说出「身体健康」四个字，再故意学一句良子的口头禅「味真足」当包袱。一句话说完。",
    ),
    (
        "liangzi",
        "收尾：必须先说出「多挣钱少生气」六个字，再说「味真足」。一句话说完，别加别的。",
    ),
]


def parse_beats(raw: str) -> list[tuple[str, str]]:
    """解析 DUET_BEATS 环境变量：'liangzi:指令|fengge:指令' → [(key, 指令)]。"""
    beats: list[tuple[str, str]] = []
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        key, sep, directive = part.partition(":")
        if not sep or key.strip() not in ("liangzi", "fengge") or not directive.strip():
            raise ValueError(f"bad beat: {part!r}")
        beats.append((key.strip(), directive.strip()))
    if len(beats) < 2:
        raise ValueError("need at least 2 beats")
    return beats


@dataclass(frozen=True)
class Speaker:
    """一个唠嗑角色：system prompt + 音色克隆素材。"""

    key: str  # "liangzi" | "fengge"
    display: str  # 展示名："良子" | "峰哥"
    prompt: str  # system prompt 全文
    ref_wav: str  # 音色参考音频路径
    ref_text: str  # 参考音频逐字台词


def build_messages(
    history: list[tuple[str, str]],
    speaker_key: str,
    system_prompt: str,
    steer: str | None = None,
    names: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """把共享对话史映射成当前 speaker 的 LLM 消息列表。

    history 为 (speaker_key, text) 序列；对当前 speaker 而言，自己说过的
    话是 assistant，其余一律 user。names 提供时，非自己的发言带名字前缀
    （如「良子：」「老铁：」），三方以上群聊靠它分清谁在说。
    steer 为可选的回合级规则后缀（防互抄/推进话题），拼进 system 末尾。
    """
    prompt = system_prompt if not steer else f"{system_prompt}\n\n{steer}"
    messages = [{"role": "system", "content": prompt}]
    for key, text in history:
        if key == speaker_key:
            messages.append({"role": "assistant", "content": text})
        else:
            content = f"{names[key]}：{text}" if names and key in names else text
            messages.append({"role": "user", "content": content})
    return messages


def turn_order(start_key: str = "liangzi", keys: tuple[str, str] = ("liangzi", "fengge")):
    """从 start_key 开始无限轮转的发言者序列。"""
    if start_key not in keys:
        raise ValueError(f"unknown speaker: {start_key}")
    i = keys.index(start_key)
    while True:
        yield keys[i % len(keys)]
        i += 1


def build_debate_steer(topic: str, side: str, user_interject: bool = False) -> str:
    """辩论回合 steer：立场 + 对抗结构 + 评委插话提示（口头禅防重由 say_turn 统一注入）。

    side 为「正」或「反」。正方=支持命题（该），反方=反对命题（不该）。
    反方固定用「这是个好事啊」式辩证反转接正方的坏处。
    """
    position = "该（支持）" if side == "正" else "不该（反对）"
    if side == "反":
        dynamic = (
            "顺着正方刚说的坏处，用你的「这是个好事啊」式反转把它说成好处"
            "（比如他说有味，你就说那才好）；"
        )
    else:
        dynamic = "陈述你支持命题的理由，接住反方的反转再怼回去；"
    steer = (
        f"【辩论赛】今日论题「{topic}」。你是{side}方，立场是「{position}」：{dynamic}"
        f"只说一两句、别超过 40 字；每句带一个你的经典口头禅，"
        f"且不许和你之前说过的重复（口头禅和句式都换新的）；要有梗、抖包袱，不人身攻击。"
    )
    if user_interject:
        steer += "老铁（评委）刚插了话，先用半句回应他，再继续辩论。"
    return steer


def build_mentor_steer(speaker_key: str, question: str, user_interject: bool = False) -> str:
    """人生导师解惑 steer：两位导师各自的解惑风格 + 聚焦当前困惑 + 评委插话提示。"""
    if speaker_key == "liangzi":
        steer = (
            f"【人生导师·捧哏】老铁的困惑是「{question}」，峰哥刚主答完。"
            f"你是捧哏良子：先接住峰哥的话（顺着捧一句或补一刀），再聚焦这个困惑给老铁补点情绪价值，能拐到吃上就拐；"
            f"说满两三句、40 到 60 字，梗要足，别跑题；每句至少带一个你的经典口头禅，全场不许和之前用过的重复；"
            f"不许编造故事和细节，口头禅为主轴；全部用汉字说，不许出现拼音字母和英文单词。"
        )
    else:
        steer = (
            f"【人生导师·主答】老铁的困惑是「{question}」。"
            f"你是主答峰哥：聚焦这个困惑，锐评、一针见血，必要时辩证反转把坏处说成好事，良子会给你捧哏；"
            f"说满三四句、60 到 100 字，梗要足，别跑题；每句至少带两个不同的经典口头禅，全场不许和之前用过的重复；"
            f"不许编造故事和细节，口头禅为主轴；全部用汉字说，不许出现拼音字母和英文单词。"
        )
    if user_interject:
        steer += "老铁又补充了，先接住他的补充再答。"
    return steer
