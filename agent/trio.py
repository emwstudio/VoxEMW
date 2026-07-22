"""峰哥×良子×老铁 三方连麦（dispatch metadata == "trio" 时由 agent.py 调起）。

与双唠的区别：用户（老铁）在场随时插话——VAD+STT 拾取用户语音进共享
历史，良子峰哥都会接住。两人接话用同一 vLLM，用户拾音用 prewarm 里
现成的 silero VAD + Qwen3-ASR，零新增显存。

回声控制：persona 出声期间暂停给 STT 喂帧（speaker 外放会被 STT 误当
用户发言），代价是用户不能在一句中途打断——等人说完再说。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import ATTRIBUTE_AGENT_STATE
from openai import AsyncOpenAI

if __package__:
    from .audio_utils import float_to_int16_bytes, tame_head_harshness
    from .duet import _load_speakers
    from .duet_logic import build_mentor_steer, build_messages
    from .text_utils import split_text_for_tts
else:
    from audio_utils import float_to_int16_bytes, tame_head_harshness
    from duet import _load_speakers
    from duet_logic import build_mentor_steer, build_messages
    from text_utils import split_text_for_tts

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
HUMAN_KEY = "user"
HUMAN_DISPLAY = "老铁"
#: 各自的经典口头禅（本场防重复用）
CATCHPHRASES: dict[str, tuple[str, ...]] = {
    "liangzi": ("味真足", "这一块", "胃袋", "活着吃", "使劲造", "香味少一半", "不吃蒜", "该吃吃该喝喝"),
    "fengge": ("这是个好事啊", "恰恰相反", "有枣没枣搂一杆子", "连接", "3D人士", "我考考你"),
}


async def run_trio(ctx: Any, models: dict) -> None:
    tts = models["tts"]
    tts._ensure_loaded()
    tts_model = tts._model
    vad = models["vad"]
    stt_instance = models["stt"]
    speakers = _load_speakers()
    names = {"liangzi": "良子", "fengge": "峰哥", HUMAN_KEY: HUMAN_DISPLAY}

    await ctx.connect()

    def room_alive() -> bool:
        return ctx.room.isconnected()

    async def safe(what: str, coro: Any) -> bool:
        """房间写入容错：连接已死时记日志并返回 False（不抛异常崩 job）。"""
        try:
            await coro
            return True
        except Exception as e:
            logger.info("trio: room write failed (%s): %s", what, e)
            return False

    async def set_state(state: str) -> None:
        # 前端 useAgent ~20s 初始化超时，必须自己发 agent 状态属性。
        # 连接死时静默失败（不因此崩 job）。
        await safe(
            "set_attributes",
            ctx.room.local_participant.set_attributes({ATTRIBUTE_AGENT_STATE: state}),
        )

    await set_state("listening")

    source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("trio-voice", source)
    await ctx.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    client = AsyncOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
    )

    async def llm_say(messages: list[dict[str, str]], rep_penalty: float = 1.15) -> str:
        resp = await client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=160,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "repetition_penalty": rep_penalty,
            },
        )
        return (resp.choices[0].message.content or "").strip()

    persona_speaking = asyncio.Event()  # persona 出声时置位（保留给状态机）

    # 回声过滤：外放场景下 STT 会把 persona 自己的声音当用户发言。
    # 不做硬门（会误杀用户插话），而是逐字/高重叠比对最近台词来识别回声。
    recent_lines: list[str] = []

    def is_echo(text: str) -> bool:
        t = "".join(ch for ch in text if ch.isalnum())
        if len(t) < 4:
            return False
        for line in recent_lines[-3:]:
            l = "".join(ch for ch in line if ch.isalnum())
            if not l:
                continue
            if t in l or l in t:
                return True
            a = {t[i : i + 2] for i in range(len(t) - 1)}
            b = {l[i : i + 2] for i in range(len(l) - 1)}
            if a and b and len(a & b) / min(len(a), len(b)) > 0.7:
                return True
        return False

    async def speak(speaker_key: str, text: str) -> None:
        persona_speaking.set()
        await set_state("speaking")
        try:
            recent_lines.append(text)
            del recent_lines[:-5]
            if not await safe(
                "publish_data",
                ctx.room.local_participant.publish_data(
                    json.dumps({"speaker": speaker_key, "text": text}, ensure_ascii=False),
                    reliable=True,
                ),
            ):
                return
            speaker = speakers[speaker_key]
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
                    if not room_alive():
                        return  # 连接死了别白烧 GPU
                    chunk = np.asarray(audio_chunk, dtype=np.float32)
                    if first:
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
        finally:
            persona_speaking.clear()
            await set_state("listening")

    # ---- 用户拾音：VAD 分段 + STT 整段识别，final 进队列 ----
    user_q: asyncio.Queue[str] = asyncio.Queue()

    async def listen_user() -> None:
        # 等第一个带麦克轨的非 agent 参与者
        user_track = None
        while user_track is None:
            for p in ctx.room.remote_participants.values():
                for pub in p.track_publications.values():
                    if pub.source == rtc.TrackSource.SOURCE_MICROPHONE and pub.track is not None:
                        user_track = pub.track
                        break
                if user_track is not None:
                    break
            if user_track is None:
                await asyncio.sleep(0.5)

        adapter = stt.StreamAdapter(stt=stt_instance, vad=vad)
        stream = adapter.stream()
        resampler = rtc.AudioResampler(SAMPLE_RATE, 16000, num_channels=1)

        async def feed() -> None:
            async for ev in rtc.AudioStream(user_track):
                for frame in resampler.push(ev.frame):
                    stream.push_frame(frame)
            stream.end_input()

        async def events() -> None:
            async for ev in stream:
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                    text = ev.alternatives[0].text.strip()
                    if not text:
                        continue
                    if is_echo(text):
                        logger.info("trio: drop echo: %s", text[:60])
                        continue
                    await user_q.put(text)

        await asyncio.gather(feed(), events())

    listener = asyncio.create_task(listen_user())

    # 本场已用口头禅登记（防重复）
    used_phrases: dict[str, set[str]] = {"liangzi": set(), "fengge": set()}

    async def say_turn(key: str, steer: str, rep_penalty: float = 1.15) -> None:
        speaker = speakers[key]
        used = used_phrases[key]
        if used:
            steer += f"本场已用过的口头禅不许再用：{'、'.join(sorted(used))}。"
        reply = await llm_say(
            build_messages(history, key, speaker.prompt, steer, names), rep_penalty
        )
        if not reply:
            return
        # 同句复读/封禁口头禅：最多 3 抽，拿到干净的为止
        for attempt in range(3):
            repeated = any(reply == text for k, text in history if k == key)
            banned = any(p in reply for p in CATCHPHRASES[key] if p in used)
            if not (repeated or banned):
                break
            if attempt < 2:
                logger.info(
                    "trio: %s re-roll %d (verbatim=%s, banned=%s): %s",
                    key, attempt + 1, repeated, banned, reply[:40],
                )
                reply = await llm_say(
                    build_messages(history, key, speaker.prompt, steer, names), rep_penalty
                )
                if not reply:
                    return
        for phrase in CATCHPHRASES[key]:
            if phrase in reply:
                used.add(phrase)
        history.append((key, reply))
        logger.info("trio: %s: %s", speaker.display, reply[:80])
        try:
            await speak(key, reply)
        except Exception:
            logger.exception("trio: tts failed for %s, skip line", key)
        await asyncio.sleep(0.3)

    history: list[tuple[str, str]] = []

    async def gen_directive(key: str, directive: str) -> None:
        """指令式发言（自我介绍/请出题/总结等，不走共享历史）。"""
        speaker = speakers[key]
        other = speakers["fengge" if key == "liangzi" else "liangzi"]
        reply = await llm_say(
            [
                {"role": "system", "content": speaker.prompt},
                {
                    "role": "user",
                    "content": f"{other.display}和老铁都在连麦。{directive}口语短句。",
                },
            ]
        )
        if not reply:
            return
        for phrase in CATCHPHRASES[key]:
            if phrase in reply:
                used_phrases[key].add(phrase)
        history.append((key, reply))
        logger.info("trio: %s: %s", speaker.display, reply[:80])
        await speak(key, reply)
        await asyncio.sleep(0.3)

    async def wait_user_line(timeout: float = 120.0) -> str:
        """等老铁说第一句话（今日论题）。"""
        t0 = asyncio.get_event_loop().time()
        while room_alive():
            try:
                return user_q.get_nowait()
            except asyncio.QueueEmpty:
                if asyncio.get_event_loop().time() - t0 > timeout:
                    return ""
                await asyncio.sleep(0.5)
        return ""

    async def topic_from_metadata(timeout: float = 12.0) -> str:
        """优先从参与者 metadata 拿页面输入的辩题；拿不到则空串（回退语音出题）。"""
        t0 = asyncio.get_event_loop().time()
        while room_alive() and asyncio.get_event_loop().time() - t0 < timeout:
            for p in ctx.room.remote_participants.values():
                md = (p.metadata or "").strip()
                if md:
                    return md
            await asyncio.sleep(0.5)
        return ""

    async def drain_user_lines() -> bool:
        """把评委插话吸进历史并通知前端；有插话返回 True。"""
        got = False
        while True:
            try:
                text = user_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            got = True
            history.append((HUMAN_KEY, text))
            logger.info("trio: %s: %s", HUMAN_DISPLAY, text[:80])
            await safe(
                "publish_data",
                ctx.room.local_participant.publish_data(
                    json.dumps({"speaker": HUMAN_KEY, "text": text}, ensure_ascii=False),
                    reliable=True,
                ),
            )
        return got

    try:
        # 阶段1：两位导师自我介绍（峰哥先，良子后）
        await gen_directive(
            "fengge", "用第一人称做自我介绍，你是峰哥：要点是全国各地穷游这些年啥没见过、今天给老铁当一回人生导师。只说一句话、别超过 30 字，只念台词本身，别把指令说出来，全用汉字。"
        )
        await gen_directive(
            "liangzi", "用第一人称做自我介绍，你是人生导师良子，欢迎老铁来问心事。只说一句话、别超过 30 字，只念台词本身，别把指令说出来，全用汉字。"
        )

        # 阶段2：取第一个困惑——优先页面输入（参与者 metadata），否则语音提问
        question = await topic_from_metadata()
        if not question:
            await gen_directive("liangzi", "用一句话问老铁：今儿有啥心事/困惑，跟哥俩说说。")
            question = await wait_user_line()
        if not question:
            logger.info("trio: no question given, stop")
            return
        history.append((HUMAN_KEY, question))
        logger.info("trio: %s: %s", HUMAN_DISPLAY, question[:80])
        await safe(
            "publish_data",
            ctx.room.local_participant.publish_data(
                json.dumps({"speaker": HUMAN_KEY, "text": question}, ensure_ascii=False),
                reliable=True,
            ),
        )

        # 仪式感：亮出今日解惑（前端弹全屏大标题）。先等 1s 让最后一句的
        # 音频缓冲放完，别截了良子最后一个字
        await asyncio.sleep(1.0)
        await safe(
            "publish_data",
            ctx.room.local_participant.publish_data(
                json.dumps({"phase": "topic_reveal", "topic": question}, ensure_ascii=False),
                reliable=True,
            ),
        )
        await asyncio.sleep(3.2)

        # 阶段3：接力解惑——峰哥主答先上，良子捧哏接住，每个困惑一回合
        rounds = int(os.environ.get("MENTOR_ROUNDS", "1"))
        current_q = question
        while room_alive():
            for _ in range(rounds):
                interject = await drain_user_lines()
                await say_turn("fengge", build_mentor_steer("fengge", current_q, interject))
                interject = await drain_user_lines()
                await say_turn("liangzi", build_mentor_steer("liangzi", current_q, interject))
            # 等老铁的下一个困惑（2 分钟没人问就收摊）
            nxt = await wait_user_line()
            if not nxt:
                logger.info("trio: no more questions, stop")
                return
            current_q = nxt
            history.append((HUMAN_KEY, nxt))
            logger.info("trio: %s: %s", HUMAN_DISPLAY, nxt[:80])
            await safe(
                "publish_data",
                ctx.room.local_participant.publish_data(
                    json.dumps({"speaker": HUMAN_KEY, "text": nxt}, ensure_ascii=False),
                    reliable=True,
                ),
            )
    finally:
        listener.cancel()
        # 收场通知（前端显示"本场结束"）；连接已死时静默，不因此崩 job
        await safe(
            "publish_data",
            ctx.room.local_participant.publish_data(
                json.dumps({"speaker": "", "text": ""}, ensure_ascii=False),
                reliable=True,
            ),
        )
