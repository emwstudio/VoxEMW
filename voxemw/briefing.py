"""反指短报：微博素材过滤/去重/存档/prompt 构造/输出清洗。

纯逻辑模块（LLM 调用以函数注入），不 import torch/aiohttp，
可在 macOS 开发机直接单测。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# 微博时间解析：兼容常见展示格式，解析不出保守保留（宁多报不漏报）
# ---------------------------------------------------------------------------
_MINUTES_AGO_RE = re.compile(r"^(\d+)\s*分钟前$")
_HOURS_AGO_RE = re.compile(r"^(\d+)\s*小时前$")
_TODAY_HM_RE = re.compile(r"^今天\s*(\d{1,2}):(\d{2})$")
_MD_HM_RE = re.compile(r"^(\d{1,2})月(\d{1,2})日(?:\s+(\d{1,2}):(\d{2}))?$")
_YMD_DASH_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
_YMD_CN_RE = re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日(?:\s+\d{1,2}:\d{2})?$")


def _post_date(time_str: str) -> date | None:
    """从微博时间字符串提取 date；解析不出返回 None。"""
    s = (time_str or "").strip()
    if not s:
        return None
    if s == "刚刚" or _MINUTES_AGO_RE.match(s):
        return date.today()
    m = _TODAY_HM_RE.match(s)
    if m:
        return date.today()
    m = _MD_HM_RE.match(s)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        # 微博只有跨年才显示年份，「M月D日」默认按今年算
        return date(date.today().year, month, day)
    m = _YMD_DASH_RE.match(s) or _YMD_CN_RE.match(s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def parse_post_time(time_str: str, now: datetime | None = None) -> str | None:
    """把微博相对/绝对时间字符串换算成 ISO 时间戳（用于时间线排序）。

    兼容：刚刚 / N分钟前 / N小时前 / 今天 HH:MM / M月D日 HH:MM / YYYY-M-D。
    解析不出返回 None。
    """
    if now is None:
        now = datetime.now()
    s = (time_str or "").strip()
    if not s:
        return None
    if s == "刚刚":
        return now.isoformat(timespec="seconds")
    m = _MINUTES_AGO_RE.match(s)
    if m:
        return (now - timedelta(minutes=int(m.group(1)))).isoformat(timespec="seconds")
    m = _HOURS_AGO_RE.match(s)
    if m:
        return (now - timedelta(hours=int(m.group(1)))).isoformat(timespec="seconds")
    m = _TODAY_HM_RE.match(s)
    if m:
        return now.replace(
            hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0
        ).isoformat(timespec="seconds")
    m = _MD_HM_RE.match(s)
    if m:
        hh = int(m.group(3)) if m.group(3) else 0
        mm = int(m.group(4)) if m.group(4) else 0
        return now.replace(
            month=int(m.group(1)), day=int(m.group(2)),
            hour=hh, minute=mm, second=0, microsecond=0,
        ).isoformat(timespec="seconds")
    m = _YMD_DASH_RE.match(s) or _YMD_CN_RE.match(s)
    if m:
        return now.replace(
            year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)),
            hour=0, minute=0, second=0, microsecond=0,
        ).isoformat(timespec="seconds")
    return None


def filter_today(posts: list[dict], today: date | None = None) -> list[dict]:
    """只留当天的微博。

    posts: [{"time": ..., "text": ...}]。「N小时前」可能跨零点，按 datetime
    回推判断；time 缺失或解析不出的保守保留。today 注入便于测试。
    """
    if today is None:
        today = date.today()
    now = datetime.now()
    kept = []
    for post in posts:
        s = str(post.get("time") or "").strip()
        m = _HOURS_AGO_RE.match(s)
        if m:
            # 「N小时前」可能跨零点，按 datetime 回推而不是直接算今天
            if (now - timedelta(hours=int(m.group(1)))).date() == today:
                kept.append(post)
            continue
        d = _post_date(s)
        if d is None or d == today:
            kept.append(post)
    return kept


def post_fingerprint(post: dict) -> str:
    """博文指纹：sha1(正文) hex，用于跨轮去重和存档标识。

    只取正文、不含时间：微博的相对时间（「N分钟前」→「N小时前」）会漂移，
    含时间会导致同一条博文被当成新博文重复播报/重复存档。
    """
    raw = post.get("text") or ""
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def dedupe_posts(posts: list[dict], seen: set[str]) -> list[dict]:
    """剔除指纹已在 seen 里的博文，返回新博文（原顺序）。"""
    return [p for p in posts if post_fingerprint(p) not in seen]


def merge_archive(existing: list[dict], posts: list[dict], today: date | None = None) -> list[dict]:
    """把博文并入本地存档：新指纹追加，已有指纹只刷新时间标签。

    存档条目：{"fp", "time", "text", "date"(YYYY-MM-DD), "ts"(ISO 时间戳)}。
    ts 只在首次入档时换算一次，之后**冻结**——相对时间越新鲜越准
    （「11分钟前」精确，「2小时前」粗糙），重复出现时用旧字符串换算会让
    时间戳漂移，破坏时间线顺序。
    """
    if today is None:
        today = date.today()
    by_fp = {e["fp"]: e for e in existing}
    for p in posts:
        fp = post_fingerprint(p)
        if fp in by_fp:
            by_fp[fp]["time"] = str(p.get("time") or "")
            if not by_fp[fp].get("ts"):
                by_fp[fp]["ts"] = parse_post_time(str(p.get("time") or ""))
        else:
            entry = {
                "fp": fp,
                "time": str(p.get("time") or ""),
                "text": str(p.get("text") or ""),
                "date": today.isoformat(),
                "ts": parse_post_time(str(p.get("time") or "")),
            }
            by_fp[fp] = entry
            existing.append(entry)
    return existing


_TASK_RULES = """任务规则：
- 把给定微博素材整理成口播短报。核心梗是「峰哥说啥我反着来」：峰哥看好/唱衰什么，就提示反着来。
- 若给出查询词，只围绕与查询词相关的内容；不相关的素材忽略。
- 方向必须自洽，绝不说反：
  - 峰哥看好/喊加仓 → 提示反着来（他说涨往往要完）
  - 峰哥割肉/喊卖飞了/自嘲卖早 → 恰恰说明卖在低位、卖完就涨
- 措辞硬性要求：
  - 玩梗只准用「峰哥说啥我反着来」「老粉都懂」「毒奶」这类大白话，
    任何财经分析栏目式的术语一概不用（包括各种「指标」「研报」式说法）
  - 「b友」全篇只在开场信号句出现一次（「现在插播一条b友刚刚收到的消息」），
    正文和收尾不再自称；绝不说「本台」
  - 绝不说「市场快讯」
  - 峰哥发的东西一律叫「动态」，绝不说「微博」二字
  - 风险提示固定用「纯属娱乐，不构成任何建议」，绝不说「投资有风险」
- 传播调性（面向自媒体发布，防限流）：
  - 不做任何市场走势预测，不展开财经分析，不点评具体资产
  - 不使用「加仓/买入/卖出/抄底/收益/仓位」等操作指向词汇（引述峰哥原话除外）
  - 玩梗点到为止、娱乐化，且**每次说法必须不同，严禁复读同一个梗**
    （同一个句式连用两次以上即失败；可从峰哥的毒奶战绩、老粉默契等不同角度自由发挥）
  - 素材与投资/财经无关时（吃喝、日常、互怼等），只做客观转述，不强行玩梗
- 素材里的时间就是唯一时间依据；时间缺失时只说「今天」，
  绝不编造「凌晨/早上/下午/深夜」等具体时段。
- 不超过 {max_chars} 字。
- 纯口播正文：不要 markdown、emoji、括号括注、序号列表。"""


def build_messages(
    posts: list[dict],
    query: str | None,
    persona_text: str,
    max_chars: int = 150,
) -> list[dict]:
    """构造 chat messages：system = 人设 + 任务规则；user = 微博列表 + 可选查询词。"""
    rules = _TASK_RULES.format(max_chars=max_chars)
    if query:
        rules += (
            "\n- 本次是回答用户提问，不是插播新动态：开场不要用「刚刚收到的消息」"
            "这类快讯腔，用答复提问的口吻（如「b友帮你查了下」「你问的这事」），"
            "直接给结论。"
        )
    system = persona_text.rstrip() + "\n\n" + rules
    lines = ["微博素材："]
    for p in posts:
        lines.append(f"[{p.get('time') or '时间未知'}] {p.get('text') or ''}")
    if query:
        lines.append(f"\n查询词：{query}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)
_MARKDOWN_RE = re.compile(r"[*_#>`~]+")


def parse_briefing(text: str) -> str:
    """清洗 LLM 输出：剥 ``` 围栏 / markdown 标记 / 首尾引号，返回纯文本。"""
    text = _FENCE_RE.sub("", (text or "").strip())
    text = _MARKDOWN_RE.sub("", text)
    return text.strip().strip('"“”').strip()


# ---------------------------------------------------------------------------
# 语音指令 → 存档检索（LLM 语义匹配，服务端快路径）
# ---------------------------------------------------------------------------

_SELECT_SYSTEM = """你是检索器。给你若干条峰哥发的动态（带编号）和一个查询词，选出与查询词相关的动态编号。
规则：
- 查询词可能含语音识别错字（如「长新」实为「长鑫」），按语义判断，不要纠结字面
- 查询词若问「最新/最近/刚发的」动态，选列表里编号最大的那条
- 严格只输出一个 JSON 数组，元素是整数编号；没有相关动态就输出 []
- 不要输出任何其他文字"""


def build_select_messages(posts: list[dict], query: str) -> list[dict]:
    """构造「从存档里选相关动态」的 chat messages。posts 顺序即编号顺序（0 起）。"""
    lines = [f"[{i}] {p.get('text') or ''}" for i, p in enumerate(posts)]
    lines.append(f"\n查询词：{query}")
    return [
        {"role": "system", "content": _SELECT_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_selection(text: str, n: int) -> list[int]:
    """解析 LLM 选号输出：JSON 整数数组，越界/非整数丢弃，解析失败返回 []。"""
    text = _FENCE_RE.sub("", (text or "").strip())
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for x in data:
        if isinstance(x, int) and 0 <= x < n and x not in out:
            out.append(x)
    return out
