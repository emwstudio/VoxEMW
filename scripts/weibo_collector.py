#!/usr/bin/env python3
"""微博动态采集器：经 Kimi WebBridge 驱动本机浏览器抓峰哥主页动态，入 SQLite。

用法：python3 scripts/weibo_collector.py [--db data/weibo.db]
前提：本机浏览器登录了微博 + WebBridge 扩展已连接（daemon 127.0.0.1:10086）。
页面是虚拟列表（滚动时旧条目卸载），逐屏滚动累积、按 URL 去重。

建议 cron：每小时一次  0 * * * * cd <repo> && .venv/bin/python scripts/weibo_collector.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.request
from pathlib import Path

DAEMON = "http://127.0.0.1:10086/command"
PROFILE_URL = "https://weibo.com/u/2397417584"  # 峰哥亡命天涯
SESSION = "voxemw-weibo"

COLLECT_JS = """(async () => {
  const seen = {};
  for (let round = 0; round < 60; round++) {
    document.querySelectorAll("article").forEach(a => {
      const t = a.querySelector("a[title]");
      const c = a.querySelector(".wbpro-feed-content");
      if (t && c && t.title) {
        seen[t.href] = {time: t.title, url: t.href,
          text: c.innerText.replace(/\\nLive/g, "").trim()};
      }
    });
    window.scrollBy(0, 450);
    await new Promise(r => setTimeout(r, 450));
  }
  return Object.values(seen);
})()"""


def bridge(action: str, args: dict) -> dict:
    body = json.dumps({"action": action, "args": args, "session": SESSION}).encode()
    req = urllib.request.Request(DAEMON, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    if not data.get("ok"):
        raise RuntimeError(f"webbridge {action} 失败: {data}")
    return data["data"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/weibo.db")
    args = parser.parse_args()

    bridge("navigate", {"url": PROFILE_URL})
    import time
    time.sleep(6)  # 首屏渲染
    posts = bridge("evaluate", {"code": COLLECT_JS})["value"]
    print(f"抓到 {len(posts)} 条")

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    db.execute("""CREATE TABLE IF NOT EXISTS posts(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      posted_at TEXT NOT NULL,
      text TEXT NOT NULL,
      url TEXT UNIQUE NOT NULL)""")
    new = 0
    for p in posts:
        new += db.execute(
            "INSERT OR IGNORE INTO posts(posted_at, text, url) VALUES (?,?,?)",
            (p["time"], p["text"], p["url"])).rowcount
    db.commit()
    total = db.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    print(f"新入库 {new} 条，库共 {total} 条 → {args.db}")


if __name__ == "__main__":
    main()
