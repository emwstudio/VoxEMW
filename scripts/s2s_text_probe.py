"""直连 s2s realtime（:8765）注入文本回合，把 TTS 音频录成 wav。

用途：无浏览器/无 RTC 环境（AutoDL TCP-only 隧道）下验证全链路音质与延迟。
用法：python -m scripts.s2s_text_probe [问题文本] [输出wav路径]
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import os
os.environ.setdefault("VOXEMW_CONFIG", "configs/assistant-4090.yaml")

from voxemw.config import load_config, load_dotenv  # noqa: E402


async def main() -> None:
    import websockets

    question = sys.argv[1] if len(sys.argv) > 1 else "妮儿，用一个字证明你是河南人。"
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/probe.wav")

    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(REPO_ROOT / os.environ["VOXEMW_CONFIG"])
    server = config.get("server") or {}
    url = f"ws://{server.get('s2s_host', '127.0.0.1')}:{server.get('s2s_port', 8765)}/v1/realtime"
    personas = config["personas"]["resolved"]
    pid = config["personas"]["default"]
    persona_text = personas[pid]["text"]

    pcm = bytearray()
    t_first_audio = None
    t0 = time.perf_counter()

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        async def send(obj):
            await ws.send(json.dumps(obj))

        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type", "")
            if etype == "session.created":
                await send({"type": "session.update", "session": {
                    "type": "realtime", "instructions": persona_text,
                    "audio": {"input": {"turn_detection": {"type": "server_vad",
                                                           "interrupt_response": True}},
                              "output": {"voice": pid}}}})
                await send({"type": "conversation.item.create", "item": {
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": question}]}})
                await send({"type": "response.create"})
                print(f"[probe] 提问: {question}", flush=True)
            elif etype in ("response.output_audio.delta", "response.audio.delta"):
                if t_first_audio is None:
                    t_first_audio = time.perf_counter() - t0
                    print(f"[probe] TTFA(含连接): {t_first_audio:.2f}s", flush=True)
                pcm.extend(base64.b64decode(event["delta"]))
            elif etype in ("response.output_audio_transcript.done",
                           "response.audio_transcript.done"):
                print(f"[probe] 回答: {event.get('transcript', '')}", flush=True)
            elif etype == "response.done":
                break
            elif etype == "error":
                print(f"[probe] ERROR: {event}", flush=True)
                break

    if not pcm:
        print("[probe] 没收到音频", file=sys.stderr)
        sys.exit(1)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(pcm))
    dur = len(pcm) / 2 / 16000
    print(f"[probe] 音频 {dur:.2f}s → {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
