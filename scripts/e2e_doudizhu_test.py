"""斗地主 E2E（实例上跑，真模型真 LLM）：

无头客户端连 ws://127.0.0.1:8766，真人座位用内置策略自动出牌，
断言：整局能打完、两个 bot 音色都有 TTS 流式音频产出、字幕齐全。

用法：python scripts/e2e_doudizhu_test.py
"""
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

from doudizhu.cards import analyze, lead_smallest_single, minimal_beat
from doudizhu.heuristic import should_bid


def choose(hand, last_play):
    """客户端侧最小决策（镜像 heuristic 的核心逻辑，不需要 Game 对象）。"""
    if analyze(hand) is not None and last_play is None:
        return list(hand)  # 整手一把出
    if last_play is None:
        return lead_smallest_single(hand)
    combo = analyze(last_play["cards"])
    if combo is None:
        return None
    return minimal_beat(hand, combo, allow_bomb=True)


async def main():
    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    stats = {
        "finished": None,
        "tts_voices": set(),
        "tts_bytes": 0,
        "tts_streams": 0,
        "subtitles": [],
        "errors": [],
    }

    async with websockets.connect("ws://127.0.0.1:8766", max_size=50 * 1024 * 1024) as ws:
        log("connected")
        steps = 0
        while steps < 800:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
            t = msg["type"]
            if t in ("tts", "tts_start"):  # 流式音频分片不计步
                pass
            else:
                steps += 1
            if t == "hello":
                log(f"hello names={msg['names']} voices={msg['voices']}")
            elif t == "state":
                st = msg["state"]
                for e in msg.get("events", []):
                    if e["type"] in ("landlord", "finish"):
                        log(f"event: {e}")
                if st["phase"] == "finished":
                    stats["finished"] = st["winner"]
                    log(f"FINISHED winner={st['winner']} spring={st['spring']}")
                    break
                if st["phase"] == "bidding" and st["bid_turn"] == "you":
                    await ws.send(json.dumps({"type": "bid", "call": should_bid(st["hand"])}))
                elif st["phase"] == "playing" and st["turn"] == "you":
                    play = choose(st["hand"], st["last_play"])
                    if play is None:
                        await ws.send(json.dumps({"type": "pass"}))
                    else:
                        await ws.send(json.dumps({"type": "play", "cards": play}))
            elif t == "subtitle":
                stats["subtitles"].append((msg["who"], msg["text"]))
                log(f"subtitle [{msg['who']}]: {msg['text']}")
            elif t == "tts_start":
                stats["tts_streams"] += 1
                stats["tts_voices"].add(msg["voice"])
            elif t == "tts":
                stats["tts_bytes"] += len(msg["pcm"])
            elif t == "error":
                stats["errors"].append(msg["message"])
                log(f"server error: {msg['message']}")

    log(f"stats: {dict(stats, tts_voices=sorted(stats['tts_voices']), subtitles=len(stats['subtitles']))}")
    assert stats["finished"] in ("landlord", "farmers"), "对局未完成"
    assert stats["tts_streams"] > 0, "没有任何 TTS 流"
    assert stats["tts_bytes"] > 10000, "TTS 音频量异常"
    assert stats["tts_voices"] == {"liangzi", "fengge"}, f"音色不全: {stats['tts_voices']}"
    assert len(stats["subtitles"]) > 0, "没有字幕"
    print("E2E OK")


asyncio.run(main())
