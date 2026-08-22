#!/usr/bin/env python3
"""VoxEMW 唱歌功能冒烟（在 GPU 服务器上跑，本地 .venv；服务需已全部启动）。

链路验证（不经过浏览器，直连 orchestrator /ws）：
  1. vox.status 断言 music=on（orchestrator 已加载 ACE-Step 客户端）
  2. 按钮路径：发 vox.sing → 断言收到 started/finished，记录首段开播延迟
  3. 口播路径：注入用户文本「唱一首歌」→ 断言 LLM 产出 sing_song 工具调用
     （response.function_call_arguments.done）且歌声真的开播

用法：.venv/bin/python scripts/smoke_sing.py [--port 8000] [--skip-voice]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


async def _recv_until(ws, want_types: set[str], timeout: float) -> dict | None:
    """收到 want_types 之一即返回该事件；超时返回 None。其余事件打印日志。"""
    try:
        async with asyncio.timeout(timeout):
            async for raw in ws:
                event = json.loads(raw)
                etype = event.get("type", "")
                if etype == "vox.sing":
                    print(f"  [vox.sing] {event.get('status')} {event.get('error') or ''}")
                elif etype == "response.function_call_arguments.done":
                    print(f"  [tool_call] {event.get('name')} {event.get('arguments')}")
                if etype in want_types:
                    return event
    except TimeoutError:
        pass
    return None


async def smoke(port: int, skip_voice: bool) -> bool:
    import websockets

    ok = True
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws",
                                  max_size=16 * 1024 * 1024) as ws:
        status = await _recv_until(ws, {"vox.status"}, 10)
        assert status, "没收到 vox.status"
        music = status.get("music")
        print(f"[1] vox.status: music={music}")
        if music != "on":
            print("    唱歌未启用（orchestrator 配置 music.enabled？），后续必败")
            return False

        # ── 按钮路径 ──
        print("[2] 按钮路径：vox.sing（20s 民谣）...")
        t0 = time.monotonic()
        await ws.send(json.dumps({
            "type": "vox.sing",
            "prompt": "民谣, 吉他, 深夜登山, 温暖",
            "seconds": 20,
        }))
        started = await _recv_until(ws, {"vox.sing"}, 300)
        first_latency = time.monotonic() - t0
        if not started or started.get("status") != "started":
            print(f"    FAIL: 没等到 started（{started}）")
            return False
        # started 是任务挂起即回；真正的生成耗时看 finished 何时来
        finished = await _recv_until(ws, {"vox.sing"}, 600)
        total = time.monotonic() - t0
        if not finished or finished.get("status") != "finished":
            print(f"    FAIL: 没等到 finished（{finished}）")
            ok = False
        else:
            print(f"    PASS: 开唱回执 {first_latency:.1f}s，20s 歌全程 {total:.1f}s")

        # ── 口播路径（LLM function calling）──
        if not skip_voice:
            print("[3] 口播路径：注入用户文本「唱一首歌」→ 等 sing_song 工具调用...")
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{
                    "type": "input_text",
                    "text": "别说话，直接给我唱一首关于深夜登山的短歌，15秒。"}]}}))
            await ws.send(json.dumps({"type": "response.create"}))
            call = await _recv_until(
                ws, {"response.function_call_arguments.done"}, 120)
            if not call or call.get("name") != "sing_song":
                print(f"    FAIL: LLM 没有调用 sing_song（{call}）——"
                      "检查 llama-server --jinja / 模型工具调用能力")
                ok = False
            else:
                sung = await _recv_until(ws, {"vox.sing"}, 600)
                while sung and sung.get("status") == "started":
                    sung = await _recv_until(ws, {"vox.sing"}, 600)
                if sung and sung.get("status") == "finished":
                    print("    PASS: 口播触发 → 唱完")
                else:
                    print(f"    FAIL: 工具调用了但歌没唱完（{sung}）")
                    ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 唱歌冒烟")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-voice", action="store_true")
    args = parser.parse_args()
    ok = asyncio.run(smoke(args.port, args.skip_voice))
    print("SING SMOKE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
