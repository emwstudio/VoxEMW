"""数字人全链路无头探针：模拟浏览器连 orchestrator /ws，注入文本回合，
统计音频 delta、二进制视频帧，抽样存帧验证。

用法：PYTHONPATH=. python scripts/avatar_probe.py [问题] [帧输出目录]
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "妮儿，用一个字证明你是河南人。"
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/avatar_frames")


async def main() -> None:
    import websockets

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    url = "ws://127.0.0.1:8000/ws"
    t0 = time.perf_counter()
    n_audio = 0
    n_frames = 0
    first_audio_at = None
    first_frame_at = None
    status = {}
    transcript = ""

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        async def pump():
            nonlocal n_audio, n_frames, first_audio_at, first_frame_at, transcript, status
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    n_frames += 1
                    if first_frame_at is None:
                        first_frame_at = time.perf_counter() - t0
                    # 首/中/尾各存一帧
                    if n_frames % 25 == 1:
                        (OUT_DIR / f"frame_{n_frames:04d}.jpg").write_bytes(bytes(msg))
                    continue
                event = json.loads(msg)
                etype = event.get("type", "")
                if etype == "vox.status":
                    status = event
                    print(f"[probe] vox.status: rtc={event.get('rtc')}, avatar={event.get('avatar')}", flush=True)
                    # 注入文本回合
                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": "user",
                                 "content": [{"type": "input_text", "text": QUESTION}]}}))
                    await ws.send(json.dumps({"type": "response.create"}))
                elif etype in ("response.output_audio.delta", "response.audio.delta"):
                    n_audio += 1
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter() - t0
                elif etype in ("response.output_audio_transcript.done",
                               "response.audio_transcript.done"):
                    transcript = event.get("transcript", "")
                elif etype == "response.done":
                    return
                elif etype == "error":
                    print(f"[probe] ERROR: {event}", flush=True)
                    return

        try:
            await asyncio.wait_for(pump(), timeout=60)
        except TimeoutError:
            print("[probe] 超时（60s 未 response.done）", flush=True)
        await asyncio.sleep(2)  # 帧尾巴

    print(f"[probe] 提问: {QUESTION}", flush=True)
    print(f"[probe] 回答: {transcript}", flush=True)
    print(f"[probe] 音频 delta x{n_audio} 首音频 {first_audio_at and round(first_audio_at, 2)}s | "
          f"视频帧 x{n_frames} 首帧 {first_frame_at and round(first_frame_at, 2)}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
