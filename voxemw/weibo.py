"""微博动态积木：读 SQLite 里的峰哥动态，供会话注入。

数据由 scripts/weibo_collector.py 采集（本机浏览器 + WebBridge），
orchestrator 在会话开始时把最近 N 条注入 persona instructions——
峰哥「记得」自己今天在忙啥。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def get_recent_posts(db_path: str | Path, limit: int = 8) -> list[tuple[str, str]]:
    """最近 N 条动态，[(posted_at, text)]，新的在前。库不存在返回空。"""
    path = Path(db_path)
    if not path.is_file():
        return []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT posted_at, text FROM posts ORDER BY posted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]
    except sqlite3.Error:
        return []


def build_posts_block(posts: list[tuple[str, str]]) -> str:
    """动态列表 → 注入 instructions 的文本块（纯函数，便于单测）。"""
    if not posts:
        return ""
    lines = [f"- {ts}：{text[:120]}" for ts, text in posts]
    return "# 峰哥近期动态（你自己的微博，被问起时自然引用）\n" + "\n".join(lines)
