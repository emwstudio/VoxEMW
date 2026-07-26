"""bot/chat 冒烟：真调 DeepSeek，验证决策 JSON、兜底、人设台词、聊天路由。

用法：.venv/bin/python scripts/smoke_doudizhu_bot.py
（读仓库根 .env.local 的 DEEPSEEK_API_KEY；需要 openai 兼容网络可达）
"""
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doudizhu import chat, heuristic
from doudizhu.bot import decide_and_act, react
from doudizhu.deepseek import chat_complete
from doudizhu.engine import Game
from doudizhu.persona import load_persona

ROOT = Path(__file__).resolve().parent.parent

for line in (ROOT / ".env.local").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get(
    "LLM_BASE_URL", "https://api.deepseek.com/v1"
)
model = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
assert api_key, "需要 DEEPSEEK_API_KEY 或 LLM_API_KEY"
print(f"llm: {base_url} model={model}")

llm = partial(chat_complete, base_url, api_key, model)

liangzi = load_persona(ROOT / "personas/liangzi.md")
fengge = load_persona(ROOT / "personas/fengge.md")
bots = {"liangzi": liangzi, "fengge": fengge}
names = {"you": "你", "liangzi": liangzi.name, "fengge": fengge.name}

game = Game(seed=7)
game.start()
print(f"== 叫地主顺序从 {game.bid_turn} 开始 ==")

# 打到第一个 bot 决策点
steps = 0
while game.phase != "finished" and steps < 60:
    steps += 1
    if game.phase == "bidding":
        seat = game.bid_turn
    else:
        seat = game.turn
    if seat == "you":
        events = heuristic.act(game, seat)
        for e in events:
            if e["type"] in ("play", "pass", "bid", "landlord"):
                print(f"[you/heuristic] {e}")
    else:
        events, say = decide_and_act(game, seat, bots[seat], llm, names)
        for e in events:
            if e["type"] in ("play", "pass", "bid", "landlord", "bomb", "finish"):
                print(f"[{seat}] {e}")
        print(f"[{seat} say] {say}")
        if steps == 4:  # 顺手验一次事件反应 + 聊天
            r = react(game, seat, bots[seat], llm, names, "对家出了个炸弹，局势紧张")
            print(f"[{seat} react] {r}")
            c = chat.handle_user_text(game, "you", "峰哥你行不行啊，这牌都打不赢", bots, llm, names)
            print(f"[chat-> {c.get('bot')}] {c.get('reply')}")
    if game.phase == "finished":
        print(f"== 完局 winner={game.winner} ==")
        break

print("SMOKE OK")
