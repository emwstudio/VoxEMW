"""峰哥×良子 双人唠嗑引擎（dispatch metadata == "duet" 时由 agent.py 调起）。

两人共用同一 vLLM（各自 system prompt 轮流问答）和同一份 VoxCPM2 权重
（generate_streaming 按 speaker 切换音色 prompt），文本接力、不走音频
环路——所以在同一块 24GB 卡上零新增显存。用户只围观，不插话。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents.types import ATTRIBUTE_AGENT_STATE
from openai import AsyncOpenAI

if __package__:
    from .audio_utils import float_to_int16_bytes, tame_head_harshness
    from .duet_logic import DEFAULT_BEATS, Speaker, build_messages, parse_beats
    from .prompts import FENGGE_SYSTEM_PROMPT, LIANGZI_SYSTEM_PROMPT
    from .text_utils import split_text_for_tts
else:
    from audio_utils import float_to_int16_bytes, tame_head_harshness
    from duet_logic import DEFAULT_BEATS, Speaker, build_messages, parse_beats
    from prompts import FENGGE_SYSTEM_PROMPT, LIANGZI_SYSTEM_PROMPT
    from text_utils import split_text_for_tts

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_speakers() -> dict[str, Speaker]:
    liangzi_txt = os.environ.get("LIANGZI_REF_TEXT", "assets/liangzi/ref.txt")
    fengge_txt = os.environ.get("FENGGE_REF_TEXT", "assets/fengge/ref.txt")
    return {
        "liangzi": Speaker(
            key="liangzi",
            display="良子",
            prompt=LIANGZI_SYSTEM_PROMPT,
            ref_wav=os.environ.get("LIANGZI_REF_WAV", "assets/liangzi/ref.wav"),
            ref_text=_read_text(liangzi_txt),
        ),
        "fengge": Speaker(
            key="fengge",
            display="峰哥",
            prompt=FENGGE_SYSTEM_PROMPT,
            ref_wav=os.environ.get("FENGGE_REF_WAV", "assets/fengge/ref.wav"),
            ref_text=_read_text(fengge_txt),
        ),
    }


async def run_duet(ctx: Any, models: dict) -> None:
    """在 ctx 的房间里跑一场峰哥×良子相声。models 为 prewarm 的模型字典
    （只用其中的 tts——VoxCPM2TTS 实例，其 _model 已加载在卡上）。"""
    tts = models["tts"]
    tts._ensure_loaded()
    tts_model = tts._model
    speakers = _load_speakers()

    await ctx.connect()  # 绕过 AgentSession，房间连接要自己建

    async def safe(what: str, coro: Any) -> bool:
        """房间写入容错：连接已死时记日志并返回 False（不抛异常崩 job）。"""
        try:
            await coro
            return True
        except Exception as e:
            logger.info("duet: room write failed (%s): %s", what, e)
            return False

    async def set_state(state: str) -> None:
        # 前端 useAgent 有 ~20s 初始化超时：agent 不发布 lk.agent.state 属性
        # 就判「joined but did not complete initializing」→ 会话被踢。
        # AgentSession 平时由框架发，双唠绕过框架，必须自己发。连接死时静默。
        await safe(
            "set_attributes",
            ctx.room.local_participant.set_attributes({ATTRIBUTE_AGENT_STATE: state}),
        )

    await set_state("listening")

    source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("duet-voice", source)
    await ctx.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    client = AsyncOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
    )

    async def llm_say(messages: list[dict[str, str]]) -> str:
        resp = await client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=130,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "repetition_penalty": 1.15,
            },
        )
        return (resp.choices[0].message.content or "").strip()

    async def speak(speaker: Speaker, text: str) -> None:
        """通知前端谁在说话 + 用 speaker 音色把 text 推进房间。"""
        await set_state("speaking")
        if not await safe(
            "publish_data",
            ctx.room.local_participant.publish_data(
                json.dumps({"speaker": speaker.key, "text": text}, ensure_ascii=False),
                reliable=True,
            ),
        ):
            return
        for piece in split_text_for_tts(text):
            first = True
            for audio_chunk in tts_model.generate_streaming(
                text=piece,
                prompt_wav_path=speaker.ref_wav,
                prompt_text=speaker.ref_text,
                reference_wav_path=speaker.ref_wav,
                cfg_value=2.0,
                inference_timesteps=10,
            ):
                if not ctx.room.isconnected():
                    return  # 连接死了别白烧 GPU
                chunk = np.asarray(audio_chunk, dtype=np.float32)
                if first:
                    # 与 solo 同款：软化起音段高频毛刺 + 前垫 250ms 静音热身
                    chunk = tame_head_harshness(chunk, SAMPLE_RATE)
                    chunk = np.concatenate(
                        [np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32), chunk]
                    )
                    first = False
                pcm = float_to_int16_bytes(chunk)
                if not pcm:
                    continue
                if not await safe(
                    "capture_frame",
                    source.capture_frame(
                        rtc.AudioFrame(
                            data=pcm,
                            sample_rate=SAMPLE_RATE,
                            num_channels=1,
                            samples_per_channel=len(pcm) // 2,
                        )
                    ),
                ):
                    return
        # 句尾 250ms 静音后垫（同 solo）
        tail = float_to_int16_bytes(np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32))
        await safe(
            "capture_frame",
            source.capture_frame(
                rtc.AudioFrame(
                    data=tail,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(tail) // 2,
                )
            ),
        )
        await set_state("listening")

    # 节拍卡驱动：每拍一句指令，模型用自己的话和口头禅演出来。
    # 自由轮答在小模型上三回合内必互相复读，演示局不能赌。
    beats_raw = os.environ.get("DUET_BEATS", "").strip()
    beats = parse_beats(beats_raw) if beats_raw else DEFAULT_BEATS

    opener_key, opener_directive = beats[0]
    opening = await llm_say(
        [
            {"role": "system", "content": speakers[opener_key].prompt},
            {
                "role": "user",
                "content": (
                    f"{speakers['fengge' if opener_key == 'liangzi' else 'liangzi'].display}"
                    f"也在连麦。{opener_directive}像朋友打电话一样，口语短句。"
                ),
            },
        ]
    )
    history: list[tuple[str, str]] = [(opener_key, opening)]
    logger.info("duet: %s: %s", speakers[opener_key].display, opening[:80])
    await speak(speakers[opener_key], opening)
    start_ts = asyncio.get_event_loop().time()

    for key, directive in beats[1:]:
        if not ctx.room.isconnected():
            logger.info("duet: room disconnected, stop")
            break
        if not ctx.room.remote_participants:
            # 观众可能还在路上：60s 宽限，别秒判空场
            if asyncio.get_event_loop().time() - start_ts > 60:
                logger.info("duet: room empty, stop")
                break
            await asyncio.sleep(1.0)
            continue
        speaker = speakers[key]
        other = speakers["fengge" if key == "liangzi" else "liangzi"]
        steer = (
            f"【本场规则】你在和{other.display}连麦唠嗑。接他的话往下聊，"
            f"不许复述他刚说过的话。这一拍你要做的：{directive}"
        )
        try:
            reply = await llm_say(build_messages(history, key, speaker.prompt, steer))
        except Exception:
            logger.exception("duet: llm failed for %s, retry once", key)
            await asyncio.sleep(1.0)
            continue
        if not reply:
            continue
        history.append((key, reply))
        logger.info("duet: %s: %s", speaker.display, reply[:80])
        try:
            await speak(speaker, reply)
        except Exception:
            logger.exception("duet: tts failed for %s, skip line", key)
        await asyncio.sleep(0.4)

    # 收场通知（前端显示"本场结束"）；连接已死时静默，不因此崩 job
    await safe(
        "publish_data",
        ctx.room.local_participant.publish_data(
            json.dumps({"speaker": "", "text": ""}, ensure_ascii=False),
            reliable=True,
        ),
    )
