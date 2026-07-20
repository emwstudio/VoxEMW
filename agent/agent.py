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
import time
from typing import Any

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


server = AgentServer(
    port=8082,  # runpod base image nginx already holds 8081
    # 24GB 卡只装得下一套 STT+TTS，只留一个预热进程待命，避免多进程重复加载 OOM
    num_idle_processes=1,
    # prewarm 要加载两套本地模型（~35s）；且会话期间孵化的替补进程会轮询等锁
    # 直到会话结束（可能很长），所以初始化上限给到 2h，覆盖长会话
    initialize_process_timeout=7200,
    # 会话结束后的清理在大模型下超过默认 10s，超时会强杀进程导致下个会话
    # 重新预热；放宽让进程干净退出、带着模型回到待命池复用
    shutdown_process_timeout=60,
)


def _build_models() -> dict[str, Any]:
    """构建并预热良子流水线的 VAD + STT + TTS（同步阻塞，约 1 分钟）。"""
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
    # The first VoxCPM2 synthesis pays ~10 s of kernel warmup; pay it here,
    # before the room goes live, instead of on the user's first reply.
    stt_instance.warmup()
    tts_instance.warmup()
    return {"vad": vad, "stt": stt_instance, "tts": tts_instance}


_MODELS_LOCK = "/tmp/voxemw-liangzi-models.lock"


def _try_acquire_models_lock() -> bool:
    """文件锁协调：保证整机只有一套 STT+TTS（含加载中的进程）。

    仅靠显存检查有竞态：兄弟进程正在加载、显存尚未涨上来时，第三个进程
    会看到“充足”的显存也加入加载，最终 OOM。锁持有者为存活进程即抢锁失败；
    持有者已死（含加载中崩溃）则回收锁。进程退出不删文件，靠 PID 存活检测回收。
    """
    try:
        with open(_MODELS_LOCK) as f:
            holder = int(f.read().strip())
        os.kill(holder, 0)  # 不发信号，仅探测存活
        return False  # 持有者活着
    except (FileNotFoundError, ValueError, ProcessLookupError):
        pass  # 无锁或持有者已死，可回收
    try:
        os.remove(_MODELS_LOCK)
    except FileNotFoundError:
        pass
    try:
        fd = os.open(_MODELS_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False  # 被别的进程抢先
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    return True


def prewarm(proc: agents.JobProcess) -> None:
    """进程级预热：模型常驻进程内存，job 来了几秒内进房。

    LiveKit Cloud 的 dispatch 进房超时只有 ~10s，每次会话现场加载模型
    （~35s）必然超时被踢。24GB 卡 = vLLM ~9.5GB + 一套 STT+TTS ~8.4GB，
    只容得下一套：显存粗检 + 文件锁双保险。

    作业进程是一次性的（会话结束即销毁），而池子会在会话进行中就孵化
    替补进程：此时锁被占，替补轮询等待——会话结束、旧进程销毁后它再加
    载模型成为热待命。保证「完成初始化的进程必有模型」，杜绝冷进程接客
    （秒接/失败交替）和并发加载 OOM。
    """
    # 与 initialize_process_timeout 对齐并留 60s 余量：长会话期间替补一直
    # 轮询等锁，旧进程销毁后立刻补位；超时（>2h 会话）退化为冷进程兜底
    deadline = time.monotonic() + 7140
    while time.monotonic() < deadline:
        try:
            import torch

            free_gb = torch.cuda.mem_get_info(0)[0] / (1 << 30)
        except Exception:
            free_gb = 0.0
        if free_gb >= 13.0 and _try_acquire_models_lock():
            try:
                proc.userdata["models"] = _build_models()
            except Exception:
                # 加载失败释放锁并退化为冷进程，避免初始化崩溃循环
                try:
                    os.remove(_MODELS_LOCK)
                except FileNotFoundError:
                    pass
                proc.userdata["models"] = None
            return
        time.sleep(5)
    proc.userdata["models"] = None


server.setup_fnc = prewarm


def _single_session_load(srv: AgentServer) -> float:
    # 同卡只跑得下一个会话：有活 job 时把负载拉满，Cloud 不再派新 job，
    # 避免冷进程再加载一套模型 OOM。
    return 1.0 if srv.active_jobs else 0.0


server.load_fnc = _single_session_load


@server.rtc_session(agent_name=AGENT_NAME)
async def liangzi_session(ctx: agents.JobContext) -> None:
    models = ctx.proc.userdata.get("models")
    if models is None:
        # 冷进程兜底：现场加载（约 1 分钟，可能错过 Cloud 的进房超时）
        models = await asyncio.to_thread(_build_models)
    vad = models["vad"]

    session = AgentSession(
        vad=vad,
        stt=stt.StreamAdapter(
            stt=models["stt"],
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
        tts=models["tts"],
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
