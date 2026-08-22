#!/usr/bin/env python3
"""VoxEMW 本地 Mac 冒烟测试（对着跑起来的 localhost:8000 打，不经过浏览器）。

两种模式：
  文本注入（默认）：conversation.item.create + response.create → 断言助手
    回文本 + RTC/事件流有音频 delta，打印首文本/首音频时延
  真音频（--wav）：把 wav 重采样成 16kHz 单声道 PCM 当麦克风上行 →
    断言 STT 转写（--expect 给子串校验）+ 助手应答

用法：
    .venv-mac/bin/python scripts/smoke_mac.py
    .venv-mac/bin/python scripts/smoke_mac.py --wav /path/to/test.wav --expect 味真足
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time

WS_URL = "ws://localhost:8000/ws"
TIMEOUT_S = 120


async def _recv_loop(ws, t0, on_event):
    async def _next():
        return json.loads(await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_S - (time.time() - t0)))

    while time.time() - t0 < TIMEOUT_S:
        try:
            msg = await _next()
        except asyncio.TimeoutError:
            break
        if on_event(msg) == "done":
            return True
    return False


async def smoke_text(question: str) -> bool:
    import websockets

    state = {"first_text": None, "first_audio": None, "text": []}

    def on_event(msg):
        ty = msg.get("type", "")
        if ty in ("response.output_text.delta", "response.output_audio_transcript.delta"):
            if state["first_text"] is None:
                state["first_text"] = time.time() - t0
            state["text"].append(msg.get("delta", ""))
        elif ty in ("response.output_audio.delta", "response.audio.delta"):
            if state["first_audio"] is None:
                state["first_audio"] = time.time() - t0
        elif ty == "response.done":
            return "done"
        elif ty == "error":
            print("ERROR:", json.dumps(msg, ensure_ascii=False)[:300])
            return "done"
        return None

    async with websockets.connect(WS_URL, max_size=16 * 1024 * 1024) as ws:
        t0 = time.time()
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": question}]}}))
        await ws.send(json.dumps({"type": "response.create"}))
        await _recv_loop(ws, t0, on_event)

    reply = "".join(state["text"]).strip()
    ok = bool(reply) and state["first_audio"] is not None
    print(f"首文本 {state['first_text']:.1f}s | 首音频 {state['first_audio']:.1f}s"
          if ok else f"未拿全（text={bool(reply)}, audio={state['first_audio']}）")
    print("回复:", reply[:120])
    return ok


async def smoke_wav(wav_path: str, expect: str | None) -> bool:
    import numpy as np
    import scipy.io.wavfile as wav
    import scipy.signal as sig
    import websockets

    sr, data = wav.read(wav_path)
    audio = data.astype(np.float32)
    if data.dtype == np.int16:
        audio /= 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = sig.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
    pcm16 = (audio * 32767).astype(np.int16).tobytes()
    silence = np.zeros(16000, dtype=np.int16).tobytes()  # 1s 静音收尾触发判停

    state = {"transcript": None, "first_audio": None, "text": []}

    def on_event(msg):
        ty = msg.get("type", "")
        if ty == "conversation.item.input_audio_transcription.completed":
            state["transcript"] = msg.get("transcript", "")
            print(f"[+{time.time() - t0:.1f}s] STT: {state['transcript']}")
        elif ty in ("response.output_text.delta", "response.output_audio_transcript.delta"):
            state["text"].append(msg.get("delta", ""))
        elif ty in ("response.output_audio.delta", "response.audio.delta"):
            if state["first_audio"] is None:
                state["first_audio"] = time.time() - t0
        elif ty == "response.done":
            return "done"
        elif ty == "error":
            print("ERROR:", json.dumps(msg, ensure_ascii=False)[:300])
            return "done"
        return None

    async with websockets.connect(WS_URL, max_size=16 * 1024 * 1024) as ws:
        t0 = time.time()
        frame = 3200  # 100ms @16kHz int16
        for chunk in (pcm16, silence):
            for i in range(0, len(chunk), frame):
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": base64.b64encode(chunk[i:i + frame]).decode()}))
        await _recv_loop(ws, t0, on_event)

    reply = "".join(state["text"]).strip()
    ok = bool(state["transcript"]) and state["first_audio"] is not None and bool(reply)
    if expect:
        hit = expect in (state["transcript"] or "")
        print(f"热词校验 {expect!r}: {'命中' if hit else '未命中'}")
        ok = ok and hit
    print("回复:", reply[:120])
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="真音频模式：喂一段 wav 当用户说话")
    ap.add_argument("--expect", help="断言转写包含该子串（热词校验）")
    ap.add_argument("--question", default="良子，打个招呼。", help="文本注入模式的问题")
    args = ap.parse_args()

    if args.wav:
        ok = asyncio.run(smoke_wav(args.wav, args.expect))
    else:
        ok = asyncio.run(smoke_text(args.question))
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
