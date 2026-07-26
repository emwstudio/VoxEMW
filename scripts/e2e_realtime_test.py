"""E2E 验证：realtime WebSocket 全链路（STT→DeepSeek→OmniVoice）。

把 assets/liangzi/ref.wav 当作用户说话推给管道，期望：
STT 转写出参考文本 → LLM 以良子口吻回复 → 收到回复音频流。

用法：python e2e_realtime_test.py [voice] [instructions]
  voice        可选，session.update 的 audio.output.voice（对应 tts.voices 的 key）
  instructions 可选，覆盖默认人设指令
"""
import asyncio, base64, json, sys, time
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

VOICE = sys.argv[1] if len(sys.argv) > 1 else None
INSTRUCTIONS = sys.argv[2] if len(sys.argv) > 2 else "你是大胃袋良子，用良子口吻回答，口语短句。"


async def main():
    import websockets
    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    audio, sr = sf.read("/root/voxemw/assets/liangzi/ref.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    silence = b"\x00" * (16000 * 2)  # 静音触发 VAD 判端

    async with websockets.connect("ws://127.0.0.1:8765/v1/realtime", max_size=50 * 1024 * 1024) as ws:
        log("connected")
        session = {
            "type": "realtime",
            "instructions": INSTRUCTIONS,
            "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}}},
        }
        if VOICE:
            session["audio"]["output"] = {"voice": VOICE}
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        got = {"stt": None, "txt": [], "audio": 0, "done": False}

        async def recv():
            async for raw in ws:
                e = json.loads(raw)
                t = e.get("type", "")
                if t == "conversation.item.input_audio_transcription.completed":
                    got["stt"] = e.get("transcript")
                    log(f"STT: {got['stt']}")
                elif t == "response.output_audio_transcript.delta":
                    got["txt"].append(e.get("delta", ""))
                elif t == "response.output_audio_transcript.done":
                    got["txt_done"] = e.get("transcript", "")
                    log(f"LLM(done): {got['txt_done']}")
                elif t == "response.output_audio.delta":
                    got["audio"] += len(e.get("delta", ""))
                elif t == "response.done":
                    got["done"] = True
                    log("response.done")
                    return
                elif t == "error":
                    log(f"ERROR: {e}")
                    return

        rt = asyncio.create_task(recv())
        chunk = 3200 * 2  # 200ms
        for i in range(0, len(pcm), chunk):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + chunk]).decode(),
            }))
            await asyncio.sleep(0.05)
        log("audio sent, sending silence")
        for _ in range(15):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(silence[:6400]).decode(),
            }))
            await asyncio.sleep(0.1)
        try:
            await asyncio.wait_for(rt, timeout=90)
        except asyncio.TimeoutError:
            log("TIMEOUT waiting response.done")
        stt = got["stt"]
        text = "".join(got["txt"])[:80]
        log(f"RESULT stt={stt!r} llm_text={text!r} audio_b64_bytes={got['audio']}")


asyncio.run(main())
