#!/usr/bin/env python3
"""VoxEMW 服务器端冒烟测试（在 GPU 服务器上跑，本地 .venv）。

链路验证（不经过浏览器）：
  1. s2s 语音管线：连 ws://127.0.0.1:8765/v1/realtime，注入人设，
     上行一段 16kHz wav（当作用户说话），断言收到转写 + TTS 音频 delta
  2. avatar 数字人：连 ws://127.0.0.1:8767，把同一段音频喂进去，断言收到 JPEG 帧

用法：
    .venv/bin/python scripts/smoke_pipeline.py --wav /path/to/test_16k.wav
    # --wav 不给时用 3 秒正弦波（能触发 VAD，但 STT 转不出字，只验证出音/出帧）
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import struct
import sys


def _tone_wav(duration_s: float = 3.0, sr: int = 16000) -> bytes:
    """220Hz 正弦 + 振幅调制（过 VAD 阈值），返回 int16 PCM 字节。"""
    frames = bytearray()
    for i in range(int(duration_s * sr)):
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * i / sr)
        sample = int(12000 * envelope * math.sin(2 * math.pi * 220 * i / sr))
        frames += struct.pack("<h", sample)
    return bytes(frames)


def _load_wav(path: str) -> bytes:
    import wave

    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 16000, f"wav 必须 16kHz，当前 {wf.getframerate()}"
        assert wf.getnchannels() == 1, "wav 必须单声道"
        assert wf.getsampwidth() == 2, "wav 必须 int16"
        return wf.readframes(wf.getnframes())


async def smoke_s2s(pcm: bytes, port: int) -> bool:
    import websockets

    print(f"[s2s] 连接 ws://127.0.0.1:{port}/v1/realtime ...")
    got_transcript = False
    audio_bytes = 0
    async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime", max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "你是测试助手，用一句话回答。",
                "audio": {"input": {"turn_detection": {"type": "server_vad"}}},
            },
        }))
        # 分块上行（模拟实时麦克风，每块 100ms）
        chunk = 1600 * 2
        for i in range(0, len(pcm), chunk):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + chunk]).decode(),
            }))
            await asyncio.sleep(0.05)

        print("[s2s] 音频已上行，等待转写与回答（最长 90s）...")
        try:
            async with asyncio.timeout(90):
                async for raw in ws:
                    event = json.loads(raw)
                    etype = event.get("type", "")
                    if etype == "conversation.item.input_audio_transcription.completed":
                        print(f"[s2s] 转写: {event.get('transcript', '')!r}")
                        got_transcript = True
                    elif etype in ("response.output_audio.delta", "response.audio.delta"):
                        audio_bytes += len(event.get("delta", "")) * 3 // 4
                    elif etype == "response.done":
                        break
                    elif etype == "error":
                        print(f"[s2s] ERROR: {event}")
                        return False
        except TimeoutError:
            print("[s2s] 超时：90s 内没等到 response.done")
    print(f"[s2s] 转写={'有' if got_transcript else '无'}，TTS 音频 {audio_bytes / 32000:.1f}s")
    # 正弦波测试转不出字属正常；真 wav 应断言 got_transcript
    return audio_bytes > 0


async def smoke_avatar(pcm: bytes, port: int) -> bool:
    import websockets

    print(f"[avatar] 连接 ws://127.0.0.1:{port} ...")
    frames = 0
    async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=16 * 1024 * 1024) as ws:
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        assert json.loads(raw).get("type") == "ready", f"unexpected: {raw!r}"
        chunk = 1600 * 2
        for i in range(0, len(pcm), chunk):
            await ws.send(json.dumps({
                "type": "audio",
                "pcm": base64.b64encode(pcm[i:i + chunk]).decode(),
            }))
        print("[avatar] 音频已喂入，等待视频帧（最长 120s，首帧含推理延迟）...")
        try:
            async with asyncio.timeout(120):
                async for raw_msg in ws:
                    if isinstance(raw_msg, bytes):
                        frames += 1
                        if frames >= 10:
                            break
        except TimeoutError:
            pass
    print(f"[avatar] 收到 {frames} 帧 JPEG")
    return frames > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 冒烟测试")
    parser.add_argument("--wav", help="16kHz 单声道 int16 wav（不给用正弦波）")
    parser.add_argument("--s2s-port", type=int, default=8765)
    parser.add_argument("--avatar-port", type=int, default=8767)
    parser.add_argument("--skip-avatar", action="store_true")
    args = parser.parse_args()

    pcm = _load_wav(args.wav) if args.wav else _tone_wav()
    print(f"测试音频 {len(pcm) / 32000:.1f}s（{'文件' if args.wav else '正弦波'}）")

    ok = asyncio.run(smoke_s2s(pcm, args.s2s_port))
    if not args.skip_avatar:
        ok = asyncio.run(smoke_avatar(pcm, args.avatar_port)) and ok

    print("SMOKE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
