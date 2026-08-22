"""Orchestrator：浏览器唯一入口，编排语音管线（s2s）与数字人服务（avatar）。

架构：
    浏览器 ←→ 本进程（aiohttp，:8000）
              ├→ s2s realtime ws（:8765，voxemw.pipeline.launch 起的语音管线）
              └→ avatar ws（:8767，voxemw.avatar.service 起的数字人服务，可缺席）

职责：
- 下行：s2s 的 TTS 音频 delta 双写 → AVSyncScheduler（RTC 音轨）+ avatar（驱动口型）；
  音画走 WebRTC 音画轨（调度器打戳对齐，浏览器 RTP 原生同步，见 voxemw/avatar/rtc.py），
  WS 只留控制/转写
- 上行：浏览器麦克风音频/控制消息 → 转发 s2s
- persona：浏览器发 {"type": "vox.persona", "id": ...} 切换人设，
  本进程把人设正文/音色/肖像注入三路（s2s instructions、TTS voice、avatar 肖像）
- 唱歌：两个触发入口——浏览器发 {"type": "vox.sing", "prompt": ...}（点歌按钮），
  或 LLM 口播点歌（session 注册 sing_song 工具，拦
  response.function_call_arguments.done 执行）。歌声经 ACE-Step
  （voxemw.avatar.singing）分段生成，转 16k PCM 后复用 TTS 下行路径
  （sched.feed_audio + avatar 口型）；用户开口打断与说话同语义
  （取消生成任务 + reset/flush）
- 打断：s2s 报 speech_started → 通知 avatar 丢弃未消费音频、运动上下文归位；
  同步 flush 音画调度器
- 对话状态下发：由 s2s 事件推导 speech_active（说话期间 avatar 禁 idle 生成，
  防句间停顿插入 idle 帧卡画面）与 idle_mode（listening/thinking/calm，
  决定待机驱动音频），见 avatar_state_transition
- listen 双流：用户说话段的麦克风音频 tee 给 avatar 做 active listening
- 降级：avatar 缺席时纯语音模式，前端显示静态肖像
- 单用户单会话：新浏览器连接顶掉旧会话（s2s 只有 1 个管线槽位，
  换网络产生的僵尸会话被新连接立即踢掉，无需等超时/刷新两次）

浏览器侧协议：
  /ws 文本帧（JSON）：
    → {"type": "vox.persona", "id": "<persona_id>"}   切换人设
    → {"type": "vox.sing", "prompt": "<风格描述>", "lyrics"?: "...", "seconds"?: 120}
      点歌（新歌顶掉进行中的歌）
    → OpenAI Realtime 事件原样透传（input_audio_buffer.append / response.cancel 等）
    ← OpenAI Realtime 事件透传（transcription / response.done 等；
      音频 delta 事件剥掉 base64 音频体——音频走 RTC 音轨，只留事件）
    ← {"type": "vox.status", "avatar": "on"|"off", "persona": "<id>",
       "rtc": {"enabled": bool, "ice_servers": [...]},
       "music": "on"|"off"}
    ← {"type": "vox.sing", "status": "started"|"finished"|"failed"|"off", ...}
  POST /rtc/offer：WebRTC 信令，body {"sdp", "type", "vbr"?} → answer
  GET  /rtc/ice ：下发 TURN ICE 配置（本地 coturn）
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

FRAME_TAG_SPEECH = 0x01  # avatar service 下行帧 tag：0x00=idle / 0x01=speech

# cover 源歌上传的落盘目录（/tmp 下，1 小时懒清理）
SING_SOURCE_DIR = Path(tempfile.gettempdir()) / "voxemw_sing_sources"


def sing_source_path(src_id: str) -> Path | None:
    """按 id 找上传的源歌音频（防目录穿越：id 只认 12 位 hex）。"""
    if not re.fullmatch(r"[a-f0-9]{12}", src_id):
        return None
    matches = sorted(SING_SOURCE_DIR.glob(f"{src_id}_*"))
    return matches[0] if matches else None

# s2s 事件 → 编排动作（纯函数分类，便于单测）
AUDIO_DELTA_EVENTS = {"response.output_audio.delta", "response.audio.delta"}  # GA / beta 名都收
AVATAR_RESET_EVENTS = {
    "input_audio_buffer.speech_started",  # 用户开口（打断）：avatar 停嘴
}


def build_session_update(persona_id: str, persona_text: str,
                         tools: list | None = None) -> dict:
    """注入人设的 session.update：instructions = 人设正文，voice = persona id
    （TTS voices 表 key，见 voxemw.pipeline.args.tts_setup_kwargs）。
    tools：Realtime 扁平格式的 function 列表（如唱歌工具），None 不注入。"""
    session = {
        "type": "realtime",
        "instructions": persona_text,
        "audio": {
            "input": {
                "turn_detection": {
                    "type": "server_vad",
                    "interrupt_response": True,
                }
            },
            "output": {"voice": persona_id},
        },
    }
    if tools:
        session["tools"] = tools
    return {
        "type": "session.update",
        "session": session,
    }


def build_sing_tool() -> dict:
    """sing_song 工具 schema（Realtime 扁平格式，经 session.update.tools 注册）。

    语音口播点歌全靠它：LLM 听到「唱首歌」类请求时产出 function call，
    orchestrator 拦 response.function_call_arguments.done 执行（见 _handle_tool_call）。
    """
    return {
        "type": "function",
        "name": "sing_song",
        "description": (
            "为用户唱一首歌（调用后歌声立即开始生成并播放，不要再用语音报幕/复述）。"
            "当用户要求唱歌、点歌、想听歌，或氛围适合主动献唱时调用。"
            "注意：口播通道只能从零创作；若用户想翻唱某首具体的歌、或用她自己"
            "录的小段换歌词，告诉她点页面底部的「点歌」按钮上传源音频。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "歌曲风格与主题描述，逗号分隔的标签最佳"
                                   "（如 '民谣, 吉他, 深夜, 登山'）",
                },
                "lyrics": {
                    "type": "string",
                    "description": "歌词。建议你来写一小段（两三句即可，更有心意）；"
                                   "留空则由歌声模型自动创作，但生成会慢约 25 秒",
                },
                "seconds": {
                    "type": "integer",
                    "description": "歌曲时长（秒），默认 30",
                },
            },
            "required": ["prompt"],
        },
    }


def classify_s2s_event(event: dict) -> tuple[bool, bool, bytes | None]:
    """分类 s2s 下行事件。返回 (relay_to_browser, reset_avatar, audio_pcm|None)。"""
    etype = event.get("type", "")
    pcm = None
    if etype in AUDIO_DELTA_EVENTS:
        delta = event.get("delta")
        if delta:
            pcm = base64.b64decode(delta)
    return True, etype in AVATAR_RESET_EVENTS, pcm


def avatar_state_transition(event: dict, speaking: bool) -> tuple[bool, list[dict]]:
    """s2s 事件 → avatar 状态控制消息（纯函数，便于单测）。

    返回 (new_speaking, 控制消息列表)。两个状态：
    - speech_active：首个音频 delta 开、response.done/打断关。说话期间 avatar
      禁 idle 生成——句间停顿 pending 排空时插入 idle 帧会被前端直画，卡画面
    - idle_mode：listening（用户开口）/ thinking（用户说完）/ calm（助手说完），
      决定待机驱动音频（listen 轨转发或纯静音）
    """
    etype = event.get("type", "")
    msgs: list[dict] = []
    if etype in AUDIO_DELTA_EVENTS and event.get("delta"):
        if not speaking:
            speaking = True
            msgs.append({"type": "speech_active", "on": True})
    elif etype == "response.done":
        if speaking:
            speaking = False
            msgs.append({"type": "speech_active", "on": False})
        msgs.append({"type": "idle_mode", "mode": "calm"})
    elif etype == "input_audio_buffer.speech_started":
        if speaking:
            speaking = False
            msgs.append({"type": "speech_active", "on": False})
        msgs.append({"type": "idle_mode", "mode": "listening"})
    elif etype == "input_audio_buffer.speech_stopped":
        if not speaking:
            msgs.append({"type": "idle_mode", "mode": "thinking"})
    return speaking, msgs


_PUNCT = "。！？；，、…—.!?;,"


def heard_prefix(transcript: str, audio_seconds: float, played_seconds: float) -> str:
    """打断时估算用户实际听到的文本前缀（纯函数，便于单测）。

    思路对齐 AVTR-1 官方 SpeechScheduler 的 played_duration 语义：播放进度
    占已生成音频的比例 ≈ 听到的文本比例（中文语速在一条回复内足够均匀）。
    不足 2 字不值得注入（上游会把整条回复从上下文回滚，零前缀=保持回滚）。
    截断处回退到最近的标点，避免半个词留在上下文里。
    """
    if not transcript or audio_seconds <= 0 or played_seconds <= 0:
        return ""
    n = int(len(transcript) * min(1.0, played_seconds / audio_seconds))
    if n < 2:
        return ""
    cut = transcript[:n]
    if n < len(transcript):
        for i in range(len(cut) - 1, 0, -1):
            if cut[i] in _PUNCT:
                cut = cut[: i + 1]
                break
    return cut


class Session:
    """一个浏览器连接 ↔ 一路 s2s + 一路 avatar 的编排。"""

    def __init__(self, browser_ws, s2s_url: str, avatar_url: str | None,
                 personas: dict, default_persona: str,
                 avatar_backend: str = "avtr1",
                 rtc_sched=None, rtc_ice_servers: list | None = None,
                 music_client=None, music_cfg: dict | None = None):
        self.browser = browser_ws
        self.s2s_url = s2s_url
        self.avatar_url = avatar_url
        self.avatar_backend = avatar_backend
        # WebRTC 音画轨：下行音画走 RTC，WS 只留控制/转写
        self.sched = rtc_sched
        self._rtc_ice_servers = rtc_ice_servers or []
        # 唱歌（ACE-Step）：client=None 即未启用；cfg 是配置里的 music 段
        self.music_client = music_client
        self._music_cfg = music_cfg or {}
        self._sing_task = None  # 进行中的唱歌协程（asyncio.Task）
        # 语音口播点歌：music 启用才给 LLM 注册 sing_song 工具
        self._tools = [build_sing_tool()] if music_client is not None else None
        self.personas = personas
        self.persona_id = default_persona
        self._user_speaking = False  # server VAD 判定的用户说话段（listen 轨转发门控）
        self.s2s = None
        self.avatar = None
        self._avatar_speaking = False  # 是否已向 avatar 下发 speech_active=on
        self._resp_had_content = False  # 本轮回复是否有任何文本/音频产出（空回复兜底用）
        self._empty_nudged = False      # 本轮是否已追问过（防追问死循环）
        self._reply_transcript = ""     # 本轮回复的转写文本（打断回报估算用）
        self._reply_audio_samples = 0   # 本轮回复已生成音频采样数（同上）

    async def run(self) -> None:
        import websockets

        async with websockets.connect(self.s2s_url, max_size=16 * 1024 * 1024) as s2s:
            self.s2s = s2s
            if self.avatar_url:
                try:
                    self.avatar = await websockets.connect(
                        self.avatar_url, max_size=16 * 1024 * 1024,
                        compression=None,  # 裸帧大消息，压缩是吞吐杀手（见 service 侧注释）
                    )
                except OSError as e:
                    logger.warning("avatar 服务不可达，降级纯语音: %s", e)
                    self.avatar = None
            await self._apply_persona(self.persona_id)
            await self._send_status()
            tasks = [
                asyncio.create_task(self._browser_to_s2s()),
                asyncio.create_task(self._s2s_to_browser()),
            ]
            if self.avatar is not None:
                tasks.append(asyncio.create_task(self._avatar_to_rtc()))
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            self._cancel_sing()  # 会话结束：停掉进行中的歌声生成
            if self.sched is not None:
                self.sched.close()  # 唤醒 RTC 取帧协程退出
            for task in pending:
                task.cancel()
            for task in done:
                logger.info("session 退出: task=%s exc=%r",
                            task.get_coro().__qualname__, task.exception())

    async def close(self) -> None:
        """关闭本会话（新浏览器连接顶掉旧连接时调用）。
        断开 s2s 释放管线槽位；转发协程随连接关闭自行退出。"""
        logger.info("close() 被调用: s2s=%s", self.s2s)
        self._cancel_sing()
        if self.s2s is not None:
            try:
                await self.s2s.close()
            except Exception:
                pass

    async def _send_status(self) -> None:
        await self.browser.send_str(json.dumps({
            "type": "vox.status",
            "avatar": "on" if self.avatar is not None else "off",
            "avatar_backend": self.avatar_backend if self.avatar is not None else "off",
            "persona": self.persona_id,
            "rtc": {"enabled": self.sched is not None, "ice_servers": self._rtc_ice_servers},
            "music": "on" if self.music_client is not None else "off",
        }))

    async def _apply_persona(self, persona_id: str) -> None:
        persona = self.personas[persona_id]
        self.persona_id = persona_id
        instructions = persona["text"]
        await self.s2s.send(json.dumps(build_session_update(persona_id, instructions, tools=self._tools)))
        if self.avatar is not None:
            image = persona.get("ref_image")
            if image:
                await self.avatar.send(json.dumps({"type": "set_image", "path": image}))
            await self.avatar.send(json.dumps({"type": "reset"}))

    # ── 唱歌（ACE-Step 分段伪流式，复用 TTS 下行路径）──

    def _cancel_sing(self) -> None:
        """取消进行中的唱歌任务（打断/新歌顶旧歌/会话关闭共用）。"""
        task = self._sing_task
        if task is not None and not task.done():
            logger.info("取消唱歌任务")
            task.cancel()
        self._sing_task = None

    async def _start_sing(self, event: dict) -> None:
        """处理浏览器 vox.sing：校验后挂后台唱歌协程。"""
        if self.music_client is None:
            await self.browser.send_str(json.dumps({"type": "vox.sing", "status": "off"}))
            return
        if self.sched is None:
            # 歌声只走 RTC 音轨，无调度器（rtc.enabled: false）时唱了也听不到
            await self.browser.send_str(json.dumps({
                "type": "vox.sing", "status": "failed", "error": "RTC 未启用，歌声无输出通道"}))
            return
        self._cancel_sing()  # 新歌顶掉进行中的歌
        from voxemw.avatar.singing import MAX_DURATION, MIN_DURATION, SongSpec

        try:
            seconds = int(event.get("seconds") or 30)
        except (TypeError, ValueError):
            seconds = 30
        seconds = max(MIN_DURATION, min(int(self._music_cfg.get("max_duration", MAX_DURATION)), seconds))
        prompt = str(event.get("prompt") or "").strip() or "pop ballad, emotional female vocal"
        src_id = str(event.get("src") or "").strip()
        # 清唱：text2music 时给 prompt 追加 a cappella 标签（cover 的伴奏跟随源歌，不管）
        if self._music_cfg.get("acappella", False) and not src_id:
            prompt += ", a cappella, solo voice, no instruments"
        spec = SongSpec(
            prompt=prompt,
            lyrics=str(event.get("lyrics") or ""),
            seconds=seconds,
            vocal_language=str(self._music_cfg.get("vocal_language", "zh")),
        )
        # cover 翻唱：src = POST /api/sing/source 上传后下发的源歌 id
        src_audio = None
        if src_id:
            src_path = sing_source_path(src_id)
            if src_path is None or not src_path.is_file():
                await self.browser.send_str(json.dumps({
                    "type": "vox.sing", "status": "failed",
                    "error": "源音频不存在或已过期，请重新上传"}))
                return
            src_audio = (src_path.name, src_path.read_bytes())
            logger.info("cover 模式：源歌 %s（%d 字节）", src_path.name, len(src_audio[1]))
        self._sing_task = asyncio.create_task(self._sing(spec, src_audio=src_audio))

    async def _sing(self, spec, src_audio: tuple[str, bytes] | None = None) -> None:
        """唱歌协程：停当前回复 → 逐段生成喂 RTC 音轨（+avatar 口型）→ 复位。

        打断复用对话语义：用户开口时 _s2s_to_browser 的 speech_started 路径
        取消本任务并做 reset/flush；唱歌期间 _avatar_speaking=True 让
        avatar_state_transition 自然发出 speech_active off + listening。
        src_audio 非空 = cover 翻唱模式（旋律照源歌，歌长先夹到源歌时长）。"""
        from voxemw.avatar.singing import iter_song_segments

        sync = bool(self._music_cfg.get("sync_singing", True)) and self.avatar is not None
        cover_strength = (float(self._music_cfg.get("cover_strength", 0.4))
                          if src_audio is not None else None)
        try:
            await self.browser.send_str(json.dumps({
                "type": "vox.sing", "status": "started", "seconds": spec.seconds}))
            # 停当前回复（若正在说话）：与打断同一套动作
            if self._avatar_speaking:
                await self.s2s.send(json.dumps({"type": "response.cancel"}))
            if self.sched is not None:
                self.sched.flush()
            if self.avatar is not None:
                await self.avatar.send(json.dumps({"type": "reset"}))
            if sync:
                await self.avatar.send(json.dumps({"type": "speech_active", "on": True}))
                self._avatar_speaking = True
            # 人设参考音：歌声贴近当前人设音色（API 只收 multipart 文件体）
            ref_audio = None
            if self._music_cfg.get("use_persona_ref", True):
                wav = (self.personas.get(self.persona_id) or {}).get("ref_wav")
                if wav:
                    try:
                        ref_audio = (Path(wav).name, Path(wav).read_bytes())
                    except OSError as e:
                        logger.warning("人设参考音读取失败，按无参考生成: %s", e)
            t_start = time.monotonic()
            first_seg = self._music_cfg.get("first_segment_seconds")
            async for pcm in iter_song_segments(
                    self.music_client, spec,
                    int(self._music_cfg.get("segment_seconds", 20)),
                    ref_audio=ref_audio,
                    first_segment_seconds=int(first_seg) if first_seg else None,
                    src_audio=src_audio,
                    cover_strength=cover_strength):
                if t_start:
                    logger.info("首段音频就位：%.1fs（%d 采样）",
                                time.monotonic() - t_start, len(pcm) // 2)
                    t_start = None
                if self.sched is not None:
                    self.sched.feed_audio(pcm)  # RTC 音频轨（与 TTS 同一股流）
                if sync:
                    await self.avatar.send(json.dumps({
                        "type": "audio", "pcm": base64.b64encode(pcm).decode()}))
            if sync:
                self._avatar_speaking = False
                await self.avatar.send(json.dumps({"type": "speech_active", "on": False}))
                await self.avatar.send(json.dumps({"type": "idle_mode", "mode": "calm"}))
            # 上下文留痕（不带歌词，防模型下文复读歌词）
            await self.s2s.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "assistant",
                         "content": [{"type": "output_text",
                                      "text": f"（刚为用户唱了一首歌：{spec.prompt}）"}]}}))
            await self.browser.send_str(json.dumps({"type": "vox.sing", "status": "finished"}))
            logger.info("唱歌完成: %ds, prompt=%r", spec.seconds, spec.prompt[:50])
        except asyncio.CancelledError:
            raise  # 打断/顶歌/关会话：avatar 复位由 speech_started 路径负责
        except Exception as e:
            logger.exception("唱歌失败")
            if sync and self._avatar_speaking:
                self._avatar_speaking = False
                try:
                    await self.avatar.send(json.dumps({"type": "speech_active", "on": False}))
                    await self.avatar.send(json.dumps({"type": "idle_mode", "mode": "calm"}))
                except Exception:
                    pass
            try:
                await self.browser.send_str(json.dumps({
                    "type": "vox.sing", "status": "failed", "error": str(e)[:200]}))
            except Exception:
                pass
        finally:
            # 只清自己——被新歌顶掉时 _sing_task 已指向新任务
            if self._sing_task is asyncio.current_task():
                self._sing_task = None

    async def _handle_sing_tool_call(self, event: dict) -> None:
        """LLM 口播点歌（function calling）：执行 sing_song 并回传工具结果。

        不发 response.create——演唱期间模型保持安静，唱完等用户开口再自然回应
        （工具结果已落库，下一轮生成自动带上）。"""
        call_id = event.get("call_id", "")
        try:
            args = json.loads(event.get("arguments") or "{}")
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {}
        logger.info("LLM 点歌（%s）: %s", call_id, str(args)[:200])
        if self.music_client is None or self.sched is None:
            output = "唱歌功能当前不可用，用一句话跟用户说明情况即可。"
        else:
            await self._start_sing({"type": "vox.sing", **args})
            output = ("歌声已开始生成并播放。演唱期间不要生成任何语音回复；"
                      "唱完后等用户开口再自然回应。")
        if call_id:
            await self.s2s.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output",
                         "call_id": call_id, "output": output}}))

    # ── 三条转发协程 ──

    async def _browser_to_s2s(self) -> None:
        async for message in self.browser:
            if message.type.name != "TEXT":
                continue  # 二进制帧（历史截帧协议）已废弃，直接忽略
            try:
                event = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "vox.persona":
                pid = event.get("id")
                if pid in self.personas:
                    await self._apply_persona(pid)
                    await self._send_status()
                continue
            if event.get("type") == "vox.sing":
                await self._start_sing(event)
                continue
            if event.get("type") == "vox.drained":
                continue  # 帧合流后该信号仅作时序参考，无需动作
            # listen 轨 tee：用户说话段（server VAD 门控，防环境噪音/回声引起多余反应）
            # 的麦克风音频转发给 avatar 做 active listening（官方 listen 轨常开，
            # 这里按段转发是 deliberate 的门控收敛）
            if (event.get("type") == "input_audio_buffer.append"
                    and self.avatar is not None and self._user_speaking):
                await self.avatar.send(json.dumps({
                    "type": "listen", "pcm": event.get("audio", "")}))
            await self.s2s.send(message.data)

    async def _s2s_to_browser(self) -> None:
        async for raw in self.s2s:
            if not isinstance(raw, str):
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await self.browser.send_str(raw)
                continue
            self._track_dialog_state(event)
            relay, reset_avatar, pcm = classify_s2s_event(event)
            # LLM 口播点歌：sing_song 工具调用（参数攒齐一次性到，无 delta 流）
            if (event.get("type") == "response.function_call_arguments.done"
                    and event.get("name") == "sing_song"):
                # 工具调用即「本轮有内容」：纯 function call 的回复没有文本/音频，
                # 不标记的话下方空回复兜底会误判追问，在演唱期间插话
                self._resp_had_content = True
                await self._handle_sing_tool_call(event)
            # 空回复兜底追踪：本轮有任何文本/音频产出即视为有内容
            etype = event.get("type", "")
            if etype == "response.created":
                self._resp_had_content = False
                self._reply_transcript = ""     # 本轮回复转写（打断回报用）
                self._reply_audio_samples = 0   # 本轮回复已生成音频采样
            if etype == "response.output_audio_transcript.delta":
                self._reply_transcript += event.get("delta", "")
            if pcm is not None:
                self._reply_audio_samples += len(pcm) // 2
            if pcm is not None or (
                etype in ("response.output_audio_transcript.delta",
                          "response.output_text.delta",
                          "response.output_audio_transcript.done")
                and (event.get("delta") or event.get("transcript"))
            ):
                self._resp_had_content = True
            elif etype == "response.done":
                status = (event.get("response") or {}).get("status")
                if (not self._resp_had_content and not self._empty_nudged
                        and status in (None, "completed")):
                    # LLM 偶发只吐 1 个 token（Qwen 本地版实测两次）→ 清理后无声。
                    # 追问一次让模型重答，把抽风变成一句话的事
                    self._empty_nudged = True
                    logger.warning("空回复兜底：本轮无文本/音频产出，追问重答")
                    await self.s2s.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": "user", "content": [{
                            "type": "input_text",
                            "text": "（你刚才的回复是空的，用一句符合你人设的话接上——比如假装清了清嗓子——然后正常回答我刚才的问题。别提这条提示）"}]}}))
                    await self.s2s.send(json.dumps({"type": "response.create"}))
            if pcm is not None and self.sched is not None:
                self.sched.feed_audio(pcm)  # RTC 音频轨（与喂 avatar 同一股流）
            was_speaking = self._avatar_speaking  # 打断回报判定要在状态翻转前取
            # 唱歌任务存活期间，avatar 说话状态归 _sing 管：工具调用回复的
            # response.done 不能把 speech_active 翻掉（否则唱歌中被插 idle 帧）
            sing_active = self._sing_task is not None and not self._sing_task.done()
            if self.avatar is not None and not (etype == "response.done" and sing_active):
                self._avatar_speaking, ctrl_msgs = avatar_state_transition(
                    event, self._avatar_speaking
                )
                for msg in ctrl_msgs:
                    await self.avatar.send(json.dumps(msg))
            if pcm is not None and self.avatar is not None:
                # 生成时刻即喂 avatar：TTS 比播放快（RTF<1），帧提前渲染好，
                # 超前问题由 sched 的播放时钟门控在展示侧收敛（见 avsync）
                await self.avatar.send(json.dumps({
                    "type": "audio",
                    "pcm": base64.b64encode(pcm).decode(),
                }))
            if reset_avatar and self.avatar is not None:
                await self.avatar.send(json.dumps({"type": "reset"}))
            if reset_avatar:
                self._cancel_sing()  # 用户开口：歌声即停（flush 在下方统一做）
            if reset_avatar and was_speaking and self.sched is not None:
                # 打断回报（对齐 AVTR-1 官方 played_duration 语义）：上游会把
                # 整条回复从上下文回滚，但用户实际已经听到了一段——把已听前缀
                # 作为 assistant 消息补写回上下文（须在下方 flush 清计数前读）
                played_s = self.sched.reply_played_seconds
                prefix = heard_prefix(self._reply_transcript,
                                      self._reply_audio_samples / 16000, played_s)
                if prefix:
                    logger.info("打断回报：已播 %.1fs，已听前缀 %d 字补写回上下文",
                                played_s, len(prefix))
                    await self.s2s.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": "assistant",
                                 "content": [{"type": "output_text", "text": prefix}]}}))
            if reset_avatar and self.sched is not None:
                self.sched.flush()  # 打断：清 RTC 音画队列
            if relay:
                if pcm is not None:
                    # 音频走 RTC 音轨，WS 只留事件本身（剥掉 base64 音频体省带宽）
                    event = {k: v for k, v in event.items() if k != "delta"}
                    await self.browser.send_str(json.dumps(event))
                else:
                    await self.browser.send_str(raw)

    def _track_dialog_state(self, event: dict) -> None:
        etype = event.get("type", "")
        if etype == "input_audio_buffer.speech_started":
            self._user_speaking = True
            self._empty_nudged = False  # 新一轮对话，重置追问名额
        elif etype == "input_audio_buffer.speech_stopped":
            self._user_speaking = False

    async def _avatar_to_rtc(self) -> None:
        """avatar 帧 → AVSyncScheduler（RTC 模式）：tag 0x01=speech / 0x00=idle。
        无需 WS 时代的中转丢帧队列——调度器/浏览器 jitter buffer 各自消化抖动。"""
        assert self.sched is not None
        stat = {"t0": time.monotonic(), "speech": 0, "idle": 0}
        async for raw in self.avatar:
            if isinstance(raw, bytes) and len(raw) > 1:
                is_speech = raw[0] == FRAME_TAG_SPEECH
                self.sched.feed_frame(bytes(raw[1:]), is_speech)
                stat["speech" if is_speech else "idle"] += 1
                now = time.monotonic()
                if now - stat["t0"] >= 10:
                    logger.info("帧接收: speech=%d idle=%d（10s）队列=%d 丢尾帧=%d 杀陈旧=%d",
                                stat["speech"], stat["idle"],
                                self.sched.queued_frames, self.sched.tail_dropped,
                                self.sched.stale_dropped)
                    stat["t0"], stat["speech"], stat["idle"] = now, 0, 0


def create_app(config: dict):
    from aiohttp import web

    server = config.get("server") or {}
    avatar_cfg = config.get("avatar") or {}
    personas = config["personas"]["resolved"]
    default_persona = config["personas"]["default"]

    s2s_url = f"ws://{server.get('s2s_host', '127.0.0.1')}:{server.get('s2s_port', 8765)}/v1/realtime"
    avatar_available = bool(avatar_cfg.get("enabled", True)) and any(
        p.get("ref_image") for p in personas.values()
    )
    avatar_url = (
        f"ws://{avatar_cfg.get('host', '127.0.0.1')}:{avatar_cfg.get('port', 8767)}"
        if avatar_available
        else None
    )
    avatar_backend = str(avatar_cfg.get("backend", "avtr1")) if avatar_available else "off"

    # 唱歌（ACE-Step 1.5）：enabled 时构造客户端注入 Session；服务缺席只在
    # 点歌时才报错（acestep-api 独立进程，主链路不依赖它就绪）
    music_cfg = config.get("music") or {}
    music_client = None
    if music_cfg.get("enabled", False):
        from voxemw.avatar.singing import MusicClient

        music_client = MusicClient(
            str(music_cfg.get("base_url", "http://127.0.0.1:8001")),
            checkpoint=str(music_cfg.get("checkpoint", "acestep-v15-turbo")),
        )
        logger.info("唱歌功能启用: %s (%s)", music_client.base_url, music_client.checkpoint)

    async def index(_request):
        return web.FileResponse(REPO_ROOT / "web" / "index.html")

    async def api_personas(_request):
        return web.json_response({
            "default": default_persona,
            "avatar": "on" if avatar_url else "off",
            "avatar_backend": avatar_backend,
            "list": [
                {
                    "id": pid,
                    "name": p["name"],
                    "label": p.get("label") or p["name"],
                    "has_image": bool(p.get("ref_image")),
                }
                for pid, p in personas.items()
            ],
        })

    async def api_persona_image(request):
        pid = request.match_info["pid"]
        persona = personas.get(pid)
        image = (persona or {}).get("ref_image")
        if not image:
            return web.Response(status=404)
        # 肖像可能被用户换图,禁缓存避免浏览器一直显示旧照片
        return web.FileResponse(image, headers={"Cache-Control": "no-cache, must-revalidate"})

    async def api_persona_image_upload(request):
        """换图免重启：覆盖 persona 肖像文件，并热推到当前会话的 avatar 服务
        （引擎 set_image 本就支持运行时换肖像——人设切换走的就是它）。"""
        pid = request.match_info["pid"]
        persona = personas.get(pid)
        image = (persona or {}).get("ref_image")
        if not image:
            return web.json_response({"error": "persona 不存在或无肖像"}, status=404)
        form = await request.post()
        field = form.get("file")
        if field is None or not getattr(field, "filename", ""):
            return web.json_response({"error": "缺少文件（字段名 file）"}, status=400)
        data = field.file.read()
        if len(data) > 20 * 1024 * 1024:
            return web.json_response({"error": "图片过大（>20MB）"}, status=400)
        with open(image, "wb") as f:
            f.write(data)
        # 热推到当前会话（无活跃会话也行——下次连接 _apply_persona 会发）
        session = current_session["session"]
        hot = False
        if session is not None and session.avatar is not None:
            await session.avatar.send(json.dumps({"type": "set_image", "path": image}))
            await session.avatar.send(json.dumps({"type": "reset"}))
            hot = True
        logger.info("persona %s 换图: %s（热推=%s）", pid, image, hot)
        return web.json_response({"ok": True, "hot": hot})

    # ── cover 源歌上传（翻唱模式的旋律源）──
    async def api_sing_source(request):
        """POST /api/sing/source：上传源歌音频 → 返回短期 id（vox.sing 的 src 引用）。"""
        form = await request.post()
        field = form.get("file")
        if field is None or not getattr(field, "filename", ""):
            return web.json_response({"error": "缺少文件（字段名 file）"}, status=400)
        data = field.file.read()
        if len(data) > 50 * 1024 * 1024:
            return web.json_response({"error": "音频过大（>50MB）"}, status=400)
        SING_SOURCE_DIR.mkdir(exist_ok=True)
        # 懒清理超 1 小时的旧源（音轨语义上只服务当次点歌）
        now = time.time()
        for old in SING_SOURCE_DIR.iterdir():
            try:
                if now - old.stat().st_mtime > 3600:
                    old.unlink()
            except OSError:
                pass
        sid = uuid.uuid4().hex[:12]
        safe_name = re.sub(r"[^\w.\-]", "_", Path(field.filename).name)[:60]
        path = SING_SOURCE_DIR / f"{sid}_{safe_name}"
        with open(path, "wb") as f:
            f.write(data)
        logger.info("cover 源歌上传: %s（%d 字节）", path.name, len(data))
        return web.json_response({"ok": True, "id": sid})

    # ── WebRTC 音画轨（对标 AVTR-1 官方 demo：RTP 时间戳原生音画同步）──
    rtc_cfg = config.get("rtc") or {}
    rtc_manager = None
    rtc_ice_servers: list = []
    if rtc_cfg.get("enabled", False):
        from voxemw.avatar.avsync import AVSyncScheduler
        from voxemw.avatar.rtc import RTCManager

        rtc_manager = RTCManager(rtc_cfg)
        rtc_ice_servers = rtc_manager.browser_ice_servers
        logger.info("WebRTC 音画轨启用（VP8 + Opus）")

    # 单用户产品：新浏览器连接顶掉旧会话（换网络/僵尸会话不再需要刷新两次）
    current_session: dict = {"session": None}

    async def api_rtc_offer(request):
        if rtc_manager is None:
            return web.json_response({"error": "rtc 未启用"}, status=404)
        session = current_session["session"]
        if session is None or session.sched is None:
            return web.json_response({"error": "无活跃会话，先连 /ws"}, status=409)
        session.sched.flush()  # 新 RTC 连接从干净的队列起步（重连不播陈年积压帧）
        offer = await request.json()
        try:
            answer = await rtc_manager.handle_offer(offer, session.sched)
        except Exception:
            # 排障：把失败的 offer SDP 落日志（Safari/老 WebKit 的 m-section 顺序差异）
            logger.exception("RTC offer 处理失败，SDP 前 800 字: %s",
                             str(offer.get("sdp"))[:800])
            raise
        return web.json_response(answer)

    async def api_rtc_ice(request):
        # 前端每次建连现取 ICE 配置（本地 coturn）
        if rtc_manager is None:
            return web.json_response({"ice_servers": []})
        return web.json_response({"ice_servers": rtc_manager.browser_ice_servers})

    async def api_rtc_debug(request):
        # 前端 12s 后回传的 ICE 诊断（候选/连接状态），排查隧道 TURN 链路
        body = await request.json()
        logger.info("RTC 前端诊断: %s", json.dumps(body, ensure_ascii=False))
        return web.json_response({"ok": True})

    async def api_sched_debug(_request):
        """排障观测点（2026-08-23 本地版无声排查）：当前会话调度器的
        音频喂入/播放计数。fed>0 而 played 不动 = RTC 音频轨没在消费；
        fed=0 = 音频根本没喂进这条会话（会话/sched 错绑）。"""
        session = current_session["session"]
        s = session.sched if session is not None else None
        if s is None:
            return web.json_response({"sched": None})
        return web.json_response({
            "buffered_seconds": round(s.buffered_audio_seconds, 2),
            "samples_fed": s._audio_samples_fed,
            "samples_played": s._audio_samples_played,
            "queued_frames": s.queued_frames,
        })

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        old = current_session["session"]
        if old is not None:
            logger.info("新连接到达，顶掉旧会话（释放管线槽位）")
            await old.close()
        # ?alead=毫秒：音频压后量（等 avatar 渲染追赶，音画对齐的关键补偿，可调）
        try:
            lead = float(request.query.get("alead", "200")) / 1000.0
        except ValueError:
            lead = 0.25
        sched = AVSyncScheduler(audio_lead=lead) if rtc_manager is not None else None
        session = Session(ws, s2s_url, avatar_url, personas, default_persona,
                          avatar_backend=avatar_backend,
                          rtc_sched=sched, rtc_ice_servers=rtc_ice_servers,
                          music_client=music_client, music_cfg=music_cfg)
        current_session["session"] = session
        try:
            await session.run()
        finally:
            if current_session["session"] is session:
                current_session["session"] = None
            if session.avatar is not None:
                await session.avatar.close()
        return ws

    app = web.Application(client_max_size=64 * 1024 * 1024)  # 换图上传可达几十 MB

    @web.middleware
    async def _no_cache(request, handler):
        # 前端 JS/HTML 迭代频繁，禁缓存防浏览器跑旧版（新旧协议不匹配会静默失声）
        resp = await handler(request)
        if request.path == "/" or request.path.startswith("/static"):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp

    app.middlewares.append(_no_cache)
    app.router.add_get("/", index)
    app.router.add_get("/api/personas", api_personas)
    app.router.add_get("/api/personas/{pid}/image", api_persona_image)
    app.router.add_post("/api/personas/{pid}/image", api_persona_image_upload)
    app.router.add_post("/api/sing/source", api_sing_source)
    app.router.add_post("/rtc/offer", api_rtc_offer)
    app.router.add_get("/rtc/ice", api_rtc_ice)
    app.router.add_post("/rtc/debug", api_rtc_debug)
    app.router.add_get("/rtc/sched", api_sched_debug)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", REPO_ROOT / "web")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 编排入口（浏览器 ↔ s2s + avatar）")
    parser.add_argument("--config", default=os.environ.get("VOXEMW_CONFIG", "configs/assistant.yaml"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    from aiohttp import web

    from voxemw.config import load_config, load_dotenv

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(config_path)

    server = config.get("server") or {}
    host = str(server.get("host", "0.0.0.0"))
    port = int(server.get("port", 8000))
    logger.info("orchestrator 就绪: http://%s:%d", host, port)
    web.run_app(create_app(config), host=host, port=port, print=None)


if __name__ == "__main__":
    main()
