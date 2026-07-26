"""server.py 无头集成冒烟（本地，不加载语音模型）：

- GameServer.load_models 置空（speak 因 tts=None 自动跳过）
- LLM 用本地假 brain（走 heuristic + 模板台词），不依赖 DeepSeek
- 真 ws 客户端连上，真人座位用 heuristic 自动打，断言能完局

用法：.venv/bin/python scripts/smoke_doudizhu_server.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets
import yaml

from doudizhu import heuristic, server as srv

CFG = yaml.safe_load(Path("configs/doudizhu.yaml").read_text())


class FakeServer(srv.GameServer):
    def load_models(self):
        # 不加载 GPU 模型；speak() 对 tts=None 会跳过
        self.bots_llm_calls = 0

    def _fake_llm(self, messages):
        self.bots_llm_calls += 1
        # 让 decide_and_act 的 JSON 解析失败 → 走 heuristic 兜底 + 模板台词
        return "not json"


async def main():
    server = FakeServer(CFG)
    server.load_models()
    server.llm = server._fake_llm

    async with websockets.serve(server.handler, "127.0.0.1", 18766):
        # proxy=None：websockets 默认会读系统代理（macOS 开了 Clash 之类就会被
        # 劫持到代理上握手失败），本地回环必须直连
        async with websockets.connect("ws://127.0.0.1:18766", proxy=None) as ws:
            got_hello = json.loads(await ws.recv())
            assert got_hello["type"] == "hello", got_hello
            print("hello:", got_hello["names"])

            finished = False
            steps = 0
            while not finished and steps < 500:
                steps += 1
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                t = msg["type"]
                if t == "state":
                    st = msg["state"]
                    for e in msg.get("events", []):
                        if e["type"] in ("bid", "landlord", "play", "pass", "finish"):
                            print(" event:", e)
                    if st["phase"] == "finished":
                        finished = True
                        print("FINISHED winner:", st["winner"], "spring:", st["spring"])
                    elif st["phase"] == "bidding" and st["bid_turn"] == "you":
                        call = heuristic.should_bid(st["hand"])
                        await ws.send(json.dumps({"type": "bid", "call": call}))
                    elif st["phase"] == "playing" and st["turn"] == "you":
                        # 用引擎内部状态做决策（测试进程内，直接摸 server.game 手牌）
                        play = heuristic.choose_play(server.game, "you")
                        if play is None:
                            await ws.send(json.dumps({"type": "pass"}))
                        else:
                            await ws.send(json.dumps({"type": "play", "cards": play}))
                elif t == "error":
                    print(" server error msg:", msg["message"])
            assert finished, "对局未收敛"
            print("SMOKE OK, fake llm calls:", server.bots_llm_calls)


asyncio.run(main())
