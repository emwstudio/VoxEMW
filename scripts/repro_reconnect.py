"""断线重连回归测试（实例上跑）：

复现并验证修复：第一桌在 bot 说话（TTS 入队）期间直接断开，
第二桌重连开新局，断言 bot 仍然会行动（不会因 _tts_q.join() 卡死）。

用法：python scripts/repro_reconnect.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

URL = "ws://127.0.0.1:8766"


async def first_table():
    """连上，等到有 bot 的语音在播（队列里大概率还有残留），立刻断开。"""
    async with websockets.connect(URL, max_size=50 * 1024 * 1024) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "tts_start":  # bot 开说了，队列里可能还有后续台词
                return


async def second_table() -> bool:
    """重连开新局，60 秒内看到任意 bot 动作（bid/play/pass 事件）即通过。"""
    async with websockets.connect(URL, max_size=50 * 1024 * 1024) as ws:
        async def watch():
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "state":
                    for e in msg.get("events", []):
                        if e["type"] in ("bid", "play", "pass", "landlord"):
                            print(f"second table saw bot action: {e}", flush=True)
                            return True
            return False

        return await asyncio.wait_for(watch(), timeout=60)


async def main():
    print("table 1: connect, wait for first bot voice, drop ...", flush=True)
    await asyncio.wait_for(first_table(), timeout=60)
    print("table 1 dropped. reconnecting ...", flush=True)
    ok = await second_table()
    assert ok, "第二桌 60 秒内没有任何 bot 动作——join 卡死问题仍在"
    print("RECONNECT OK")


asyncio.run(main())
