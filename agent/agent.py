"""Liangzi voice agent — LiveKit Agents 1.x entry point.

Pipeline (all models run locally on a single RTX 4090 pod):
  STT  Qwen/Qwen3-ASR-0.6B-hf (transformers, via StreamAdapter + silero VAD)
  LLM  Qwen/Qwen3-8B-AWQ    (local vLLM OpenAI-compatible server)
  TTS  openbmb/VoxCPM2        (local inference, Ultimate Cloning voice clone)

Run:  python agent/agent.py start
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, TurnHandlingOptions, stt
from livekit.plugins import openai, silero

if __package__:
    from .prompts import LIANGZI_GREETING_INSTRUCTIONS, LIANGZI_SYSTEM_PROMPT
    from .stt_qwen3asr import Qwen3ASRSTT
    from .tts_voxcpm2 import VoxCPM2TTS
else:
    from prompts import LIANGZI_GREETING_INSTRUCTIONS, LIANGZI_SYSTEM_PROMPT
    from stt_qwen3asr import Qwen3ASRSTT
    from tts_voxcpm2 import VoxCPM2TTS

load_dotenv(".env.local")

AGENT_NAME = "liangzi-agent"


class LiangziAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=LIANGZI_SYSTEM_PROMPT)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


server = AgentServer(port=8082)  # runpod base image nginx already holds 8081


@server.rtc_session(agent_name=AGENT_NAME)
async def liangzi_session(ctx: agents.JobContext) -> None:
    ref_wav = os.environ.get("LIANGZI_REF_WAV", "assets/liangzi/ref.wav")
    ref_text_path = os.environ.get("LIANGZI_REF_TEXT", "assets/liangzi/ref.txt")
    if not os.path.exists(ref_wav):
        raise FileNotFoundError(f"LIANGZI_REF_WAV not found: {ref_wav}")
    if not os.path.exists(ref_text_path):
        raise FileNotFoundError(f"LIANGZI_REF_TEXT not found: {ref_text_path}")

    vad = silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.4,
    )

    stt_instance = Qwen3ASRSTT(
        model_id=os.environ.get("STT_MODEL", "Qwen/Qwen3-ASR-0.6B-hf"),
        language=os.environ.get("STT_LANGUAGE", "Chinese"),
        device=os.environ.get("STT_DEVICE", "cuda"),
    )
    tts_instance = VoxCPM2TTS(
        model_id=os.environ.get("TTS_MODEL", "openbmb/VoxCPM2"),
        prompt_wav_path=ref_wav,
        prompt_text=_read_text_file(ref_text_path),
        device=os.environ.get("TTS_DEVICE", "cuda"),
    )

    # Preload + pre-warm both local models off the event loop. The first
    # VoxCPM2 synthesis pays ~10 s of kernel warmup; pay it before the room
    # goes live instead of on the user's first reply.
    await asyncio.to_thread(stt_instance.warmup)
    await asyncio.to_thread(tts_instance.warmup)

    session = AgentSession(
        vad=vad,
        stt=stt.StreamAdapter(
            stt=stt_instance,
            vad=vad,
        ),
        llm=openai.LLM(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
            base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
            # Disable <think> blocks at the chat-template level so reasoning
            # never reaches the TTS (hybrid-thinking models).
            # repetition_penalty keeps small-model loops out of the banter.
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "repetition_penalty": 1.15,
            },
            # Voice chat: short, lively, on-persona answers.
            temperature=0.7,
            top_p=0.8,
            max_completion_tokens=130,
        ),
        tts=tts_instance,
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
        # Start the LLM as soon as the user stops talking (VAD-end) instead of
        # waiting for full turn confirmation — shaves perceived latency.
        preemptive_generation=True,
    )

    await session.start(
        room=ctx.room,
        agent=LiangziAgent(),
    )

    await session.generate_reply(instructions=LIANGZI_GREETING_INSTRUCTIONS)


if __name__ == "__main__":
    agents.cli.run_app(server)
