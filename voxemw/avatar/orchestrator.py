"""Orchestrator：浏览器唯一入口，编排语音管线（s2s）与数字人服务（avatar）。

架构：
    浏览器 ←→ 本进程（aiohttp，:8000）
              ├→ s2s realtime ws（:8765，voxemw.pipeline.launch 起的语音管线）
              └→ avatar ws（:8767，voxemw.avatar.service 起的数字人服务，可缺席）

职责：
- 下行：s2s 的 TTS 音频 delta 双写 → 浏览器（播放）+ avatar（驱动口型）
- 上行：浏览器麦克风音频/控制消息 → 转发 s2s
- persona：浏览器发 {"type": "vox.persona", "id": ...} 切换人设，
  本进程把人设正文/音色/肖像注入三路（s2s instructions、TTS voice、avatar 肖像）
- 打断：s2s 报 speech_started → 通知 avatar 丢弃未消费音频、运动上下文归位
- 对话状态下发：由 s2s 事件推导 speech_active（说话期间 avatar 禁 idle 生成，
  防句间停顿插入 idle 帧卡画面）与 idle_mode（listening/thinking/calm，
  决定待机驱动音频），见 avatar_state_transition
- 截帧打分两阶段：收到用户截帧 → 先注入「垫场」指令让 persona 自由发挥，
  Kimi 描述返回后再注入打分指令，填掉等待空白。注入是乐观发送+被拒重试
  （VAD 回复不发 response.created，无法预判回复是否在播；response.create 被
  conversation_already_has_active_response 拒绝时，等当前回复 response.done 重发）。
  从截帧到打分回复说完期间，上行麦克风音频直接丢弃（用户插话不打断打分）
- 垫音（filler）：用户转写完成的瞬间，LLM 首句还要 ~1.4s——先把一条预渲染的
  persona 口头禅（assets/<id>/fillers/*.wav，随机轮换不重复）伪造成
  response.output_audio.delta 推给浏览器+avatar，感知延迟从 ~2.9s 压到 ~1.5s。
  与截帧「垫场→打分」同一套连续回复结构，前端唇同步无需改动；
  用户插话（speech_started）立即取消垫音任务，前端照常 flush
- 降级：avatar 缺席时纯语音模式，前端显示静态肖像
- 单用户单会话：新浏览器连接顶掉旧会话（s2s 只有 1 个管线槽位，
  换网络产生的僵尸会话被新连接立即踢掉，无需等超时/刷新两次）

浏览器侧协议（/ws）：
  文本帧（JSON）：
    → {"type": "vox.persona", "id": "<persona_id>"}   切换人设
    → OpenAI Realtime 事件原样透传（input_audio_buffer.append / response.cancel 等）
    ← OpenAI Realtime 事件原样透传（transcription / response.done 等）
    ← {"type": "vox.status", "avatar": "on"|"off", "persona": "<id>"}
    ← {"type": "vox.vision", "state": ...}  截帧处理状态（见 _send_vision_state 调用点）
  二进制帧：
    → 0x02 + JPEG：用户摄像头截帧（persona 说暗号后前端发来，走 vision 描述注入）
    ← 0x01 + JPEG：数字人视频帧
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import sys
import tempfile
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from voxemw import vision

logger = logging.getLogger(__name__)

FRAME_TYPE_JPEG = 0x01
SAMPLE_RATE_16K = 16000  # 管线全程 16kHz（垫音/音频 delta 均为 int16 mono）

# s2s 事件 → 编排动作（纯函数分类，便于单测）
AUDIO_DELTA_EVENTS = {"response.output_audio.delta", "response.audio.delta"}  # GA / beta 名都收
AVATAR_RESET_EVENTS = {
    "input_audio_buffer.speech_started",  # 用户开口（打断）：avatar 停嘴
}


def build_session_update(persona_id: str, persona_text: str) -> dict:
    """注入人设的 session.update：instructions = 人设正文，voice = persona id
    （TTS voices 表 key，见 voxemw.pipeline.args.tts_setup_kwargs）。"""
    return {
        "type": "session.update",
        "session": {
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
      决定待机驱动音频（persona 嘟囔循环或纯静音）
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


FILLER_GROUPS = ("positive", "negative", "neutral")

# SenseVoice 情绪标签 → 垫音分组（stt_sensevoice 写 /tmp 侧信道，零额外推理成本）
EMOTION_TO_GROUP = {
    "HAPPY": "positive",
    "SURPRISED": "positive",
    "SAD": "negative",
    "ANGRY": "negative",
    "FEARFUL": "negative",
    "DISGUSTED": "negative",
}

EMOTION_SIDECAR_PATH = os.path.join(tempfile.gettempdir(), "voxemw_stt_emotion")


def read_emotion_sidecar(path: str = EMOTION_SIDECAR_PATH) -> str:
    """读 STT 写的情绪侧信道（缺失/异常回退 NEUTRAL）。"""
    try:
        return Path(path).read_text().strip() or "NEUTRAL"
    except OSError:
        return "NEUTRAL"


def _read_filler_wav(wav_path: Path) -> bytes | None:
    try:
        with wave.open(str(wav_path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                logger.warning("垫音格式不符（需 16k mono s16），跳过: %s", wav_path)
                return None
            return w.readframes(w.getnframes())
    except (wave.Error, OSError) as e:
        logger.warning("垫音读取失败 %s: %s", wav_path, e)
        return None


def load_fillers(persona: dict) -> dict[str, list[tuple[bytes, str]]]:
    """persona 素材目录 fillers/<group>/*.wav → 分组 (PCM, 台词) 列表。
    根目录散落的 wav 归入 neutral；台词读 fillers/texts.json（相对路径→文本），
    缺条目台词为空串（跳过历史注入）。台词供注入 LLM 历史，让模型知道自己「说」过。"""
    groups: dict[str, list[tuple[bytes, str]]] = {g: [] for g in FILLER_GROUPS}
    image = persona.get("ref_image")
    if not image:
        return groups
    fdir = Path(image).parent / "fillers"
    texts: dict = {}
    texts_path = fdir / "texts.json"
    if texts_path.is_file():
        try:
            texts = json.loads(texts_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("垫音台词表读取失败 %s: %s", texts_path, e)
    for wav_path in sorted(fdir.rglob("*.wav")):
        pcm = _read_filler_wav(wav_path)
        if pcm is None:
            continue
        group = wav_path.parent.name if wav_path.parent != fdir else "neutral"
        if group not in groups:
            group = "neutral"
        text = texts.get(str(wav_path.relative_to(fdir)), "")
        groups[group].append((pcm, text))
    return groups


def build_filler_history_item(text: str) -> dict:
    """垫音台词 → 注入 LLM 历史的 assistant 消息（只入历史，不触发响应）。
    让模型知道自己刚「说」过这句垫音，后续轮次保持连贯。"""
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def pick_filler_index(n: int, last: int) -> int:
    """随机选垫音下标，避免与上一条重复（纯函数，便于单测）。"""
    idx = random.randrange(n)
    if n > 1 and idx == last:
        idx = (idx + 1) % n
    return idx


class Session:
    """一个浏览器连接 ↔ 一路 s2s + 一路 avatar 的编排。"""

    def __init__(self, browser_ws, s2s_url: str, avatar_url: str | None,
                 personas: dict, default_persona: str, vision_cfg: dict | None = None,
                 filler_enabled: bool = True):
        self.browser = browser_ws
        self.s2s_url = s2s_url
        self.avatar_url = avatar_url
        self.personas = personas
        self.persona_id = default_persona
        self.vision_cfg = vision_cfg
        self._vision_busy = False
        self._mic_muted = False  # 垫场→打分说完期间：上行麦克风音频丢弃，插话不打断打分
        self.s2s = None
        self.avatar = None
        self._avatar_speaking = False  # 是否已向 avatar 下发 speech_active=on
        self._filler_enabled = filler_enabled
        self._fillers = load_fillers(personas[default_persona])
        self._filler_last: dict[str, int] = {}  # 每个分组各自记上一条，避免连续重复
        self._filler_task: asyncio.Task | None = None
        # 注入状态机：response.create 被「有进行中回复」拒绝时重试。
        # 注意 VAD 驱动的回复 s2s 不发 response.created（只有注入的回复才发），
        # 所以不能靠事件推算「是否有回复在播」，只能乐观发、被拒再等 response.done 重发
        self._responses_done = 0  # response.done 计数（等回复结束/打分说完都靠它）
        self._inject_lock = asyncio.Lock()  # 串联所有注入，保证 response.created 归属唯一
        self._inject_event = asyncio.Event()  # response.created(accepted) / active_response 错误(rejected)
        self._inject_outcome: str | None = None

    async def run(self) -> None:
        import websockets

        async with websockets.connect(self.s2s_url, max_size=16 * 1024 * 1024) as s2s:
            self.s2s = s2s
            if self.avatar_url:
                try:
                    self.avatar = await websockets.connect(
                        self.avatar_url, max_size=16 * 1024 * 1024
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
                tasks.append(asyncio.create_task(self._avatar_to_browser()))
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            self._cancel_filler()
            for task in pending:
                task.cancel()
            for task in done:
                logger.info("session 退出: task=%s exc=%r",
                            task.get_coro().__qualname__, task.exception())

    async def close(self) -> None:
        """关闭本会话（新浏览器连接顶掉旧连接时调用）。
        断开 s2s 释放管线槽位；转发协程随连接关闭自行退出。"""
        logger.info("close() 被调用: s2s=%s", self.s2s)
        if self.s2s is not None:
            try:
                await self.s2s.close()
            except Exception:
                pass

    async def _send_status(self) -> None:
        await self.browser.send_str(json.dumps({
            "type": "vox.status",
            "avatar": "on" if self.avatar is not None else "off",
            "persona": self.persona_id,
        }))

    async def _apply_persona(self, persona_id: str) -> None:
        persona = self.personas[persona_id]
        self.persona_id = persona_id
        self._fillers = load_fillers(persona)
        await self.s2s.send(json.dumps(build_session_update(persona_id, persona["text"])))
        if self.avatar is not None:
            image = persona.get("ref_image")
            if image:
                await self.avatar.send(json.dumps({"type": "set_image", "path": image}))
            await self.avatar.send(json.dumps({"type": "reset"}))

    # ── 三条转发协程 ──

    async def _browser_to_s2s(self) -> None:
        async for message in self.browser:
            if message.type.name == "BINARY":
                data = bytes(message.data)
                if data and data[0] == vision.FRAME_TYPE_USER_JPEG:
                    asyncio.create_task(self._handle_user_frame(data[1:]))
                continue
            if message.type.name != "TEXT":
                continue
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
            # 垫场→打分说完期间麦克风静音：上行音频直接丢弃，插话不打断打分
            if self._mic_muted and event.get("type") == "input_audio_buffer.append":
                continue
            await self.s2s.send(message.data)

    # ── 截帧打分：vision 描述 → 注入 s2s 让 persona 锐评 ──

    # ── 垫音：转写完成 → 立即播一条预渲染口头禅，填 LLM 首句的 ~1.4s 空白 ──

    def _cancel_filler(self) -> None:
        if self._filler_task is not None and not self._filler_task.done():
            self._filler_task.cancel()
        self._filler_task = None

    def _start_filler(self) -> None:
        self._cancel_filler()
        if not self._filler_enabled:
            return
        # 按用户情绪选垫音分组（SenseVoice 同次推理副产品，侧信道读取零成本）
        emotion = read_emotion_sidecar()
        group = EMOTION_TO_GROUP.get(emotion, "neutral")
        clips = self._fillers.get(group) or self._fillers.get("neutral") or []
        if not clips:
            return
        last = self._filler_last.get(group, -1)
        idx = pick_filler_index(len(clips), last)
        self._filler_last[group] = idx
        logger.info("垫音: 情绪=%s 分组=%s 第%d/%d条", emotion, group, idx + 1, len(clips))
        pcm, text = clips[idx]
        self._filler_task = asyncio.create_task(self._play_filler(pcm, text))

    async def _play_filler(self, clip: bytes, text: str = "") -> None:
        """把垫音 PCM 伪造成 response.output_audio.delta 推给浏览器+avatar，
        台词同步注入 LLM 历史（assistant 消息，只入历史不触发响应）。
        与真实回复构成「连续两段回复」（同截帧垫场→打分结构），前端唇同步原生支持。
        播完后把 avatar 切回待机（speech_active=off + idle_mode=thinking）：
        垫音帧与真回复帧之间的 LLM 等待空隙由待机微动桥接，画面不定格。"""
        try:
            if text:
                await self.s2s.send(json.dumps(build_filler_history_item(text), ensure_ascii=False))
            if self.avatar is not None and not self._avatar_speaking:
                self._avatar_speaking = True
                await self.avatar.send(json.dumps({"type": "speech_active", "on": True}))
            chunk = SAMPLE_RATE_16K * 2 * 2 // 5  # 0.4s int16 一块
            for i in range(0, len(clip), chunk):
                b64 = base64.b64encode(clip[i:i + chunk]).decode()
                await self.browser.send_str(
                    json.dumps({"type": "response.output_audio.delta", "delta": b64}))
                if self.avatar is not None:
                    await self.avatar.send(json.dumps({"type": "audio", "pcm": b64}))
            # 伪造 response.done 关闭垫音「回复」：前端下一个 delta（真回复）会重锚
            # 视频基准并清掉垫音的零填充闭嘴尾帧——否则 ~0.96s 尾帧占着帧序号，
            # 整条回复口型落后音频 ~1s（音频结束嘴还在动）
            await self.browser.send_str(json.dumps({"type": "response.done"}))
            if self.avatar is not None and self._avatar_speaking:
                self._avatar_speaking = False
                await self.avatar.send(json.dumps({"type": "speech_active", "on": False}))
                await self.avatar.send(json.dumps({"type": "idle_mode", "mode": "thinking"}))
        except asyncio.CancelledError:
            raise  # 用户插话取消：前端已被 speech_started flush，直接退出
        except Exception as e:
            logger.info("垫音播放中断（连接关闭？）: %r", e)

    async def _send_vision_state(self, state: str) -> None:
        try:
            await self.browser.send_str(json.dumps({"type": "vox.vision", "state": state}))
        except Exception:
            pass

    async def _handle_user_frame(self, jpeg: bytes) -> None:
        if self.vision_cfg is None:
            await self._send_vision_state("off")
            return
        if self._vision_busy:
            await self._send_vision_state("busy")
            return
        self._vision_busy = True
        self._mic_muted = True  # 从此刻到打分说完：用户插话不打断流程
        await self._send_vision_state("scoring")
        try:
            async with self._inject_lock:
                # 阶段一：注入「垫场」让 persona 自由发挥，填上 Kimi 描述的等待空白
                stall_item, _ = vision.build_stall_messages()
                if not await self._inject(stall_item):
                    await self._abort_scoring()
                    return
                # 阶段二：Kimi 描述回来后注入打分指令（垫场还在播也没关系，
                # _inject 被拒会等它说完自动重发）
                description = await vision.describe(self.vision_cfg, jpeg)
                if not description:
                    await self._abort_scoring()
                    return
                item_create, _ = vision.build_inject_messages(description)
                if not await self._inject(item_create):
                    await self._abort_scoring()
                    return
                logger.info("截帧描述已注入: %s", description[:60])
                # 打分回复说完再恢复麦克风（按 response.done 计数等，无竞态）
                asyncio.create_task(self._unmute_after_scoring(self._responses_done))
        except Exception as e:
            # 浏览器中途断开（刷新/关闭）时 s2s 连接随之关闭，注入失败属正常收尾
            logger.info("截帧流程中断（连接关闭？）: %r", e)
            self._mic_muted = False
        finally:
            self._vision_busy = False

    async def _inject(self, item_create: dict, max_attempts: int = 4) -> bool:
        """发 conversation.item.create + response.create，返回是否被接受。

        回复在播时 response.create 会被 s2s 拒绝（conversation_already_has_active_response），
        此时 item 已由 s2s 的 deferred 队列保管，只需等当前回复 response.done 后
        重发 response.create。response.created 只有注入的回复才发，配合
        _inject_lock 可安全归属。
        """
        await self.s2s.send(json.dumps(item_create, ensure_ascii=False))
        for attempt in range(1, max_attempts + 1):
            self._inject_outcome = None
            self._inject_event.clear()
            await self.s2s.send(json.dumps({"type": "response.create"}))
            try:
                await asyncio.wait_for(self._inject_event.wait(), timeout=15)
            except asyncio.TimeoutError:
                logger.warning("注入第 %d 次：等 response.created 超时", attempt)
                return False
            if self._inject_outcome == "accepted":
                return True
            # rejected：等当前回复说完再重发 response.create
            done_before = self._responses_done
            waited = 0.0
            while self._responses_done <= done_before and waited < 60:
                await asyncio.sleep(0.2)
                waited += 0.2
            if waited >= 60:
                logger.warning("注入第 %d 次：等进行中回复结束超时", attempt)
                return False
            logger.info("注入第 %d 次被拒（回复在播），已等其结束，重发", attempt)
        return False

    async def _abort_scoring(self) -> None:
        """打分流程中途失败：不会有打分回复了，立刻恢复麦克风。"""
        self._mic_muted = False
        await self._send_vision_state("error")

    async def _unmute_after_scoring(self, done_at_inject: int) -> None:
        # 等注入点之后的那个 response.done（打分回复说完），兜底 120s 强制恢复
        waited = 0.0
        while self._responses_done <= done_at_inject and waited < 120:
            await asyncio.sleep(0.2)
            waited += 0.2
        if waited >= 120:
            logger.warning("截帧：打分回复 120s 未结束，强制恢复麦克风")
        self._mic_muted = False
        await self._send_vision_state("scored")

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
            etype = event.get("type", "")
            # 转写完成 → 立即垫音；用户再开口（打断）→ 取消垫音
            if etype == "conversation.item.input_audio_transcription.completed":
                if (event.get("transcript") or "").strip():
                    self._start_filler()
            elif etype == "input_audio_buffer.speech_started":
                self._cancel_filler()
            relay, reset_avatar, pcm = classify_s2s_event(event)
            if pcm is not None:
                self._cancel_filler()  # 真音频抢先到达：停发垫音余量，防结尾误切待机
            if self.avatar is not None:
                self._avatar_speaking, ctrl_msgs = avatar_state_transition(
                    event, self._avatar_speaking
                )
                for msg in ctrl_msgs:
                    await self.avatar.send(json.dumps(msg))
            # 注入重试机制的预期错误，不 relay 给浏览器（用户看到 ⚠ 会困惑）
            if event.get("type") == "error" and \
                    (event.get("error") or {}).get("type") == "conversation_already_has_active_response":
                relay = False
            if pcm is not None and self.avatar is not None:
                await self.avatar.send(json.dumps({
                    "type": "audio",
                    "pcm": base64.b64encode(pcm).decode(),
                }))
            if reset_avatar and self.avatar is not None:
                await self.avatar.send(json.dumps({"type": "reset"}))
            if relay:
                await self.browser.send_str(raw)

    def _track_dialog_state(self, event: dict) -> None:
        etype = event.get("type", "")
        if etype == "response.done":
            self._responses_done += 1
        elif etype == "response.created":
            # 只有注入的回复才发 response.created（VAD 回复不发）
            self._inject_outcome = "accepted"
            self._inject_event.set()
        elif etype == "error":
            err_type = (event.get("error") or {}).get("type", "")
            if err_type == "conversation_already_has_active_response":
                self._inject_outcome = "rejected"
                self._inject_event.set()

    async def _avatar_to_browser(self) -> None:
        # 中转队列 + 独立发送任务：浏览器/隧道抖动时丢最旧帧，
        # 而不是 await 阻塞 avatar 读取、把背压传回数字人服务
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=25)

        async def sender() -> None:
            while True:
                await self.browser.send_bytes(await q.get())

        task = asyncio.create_task(sender())
        try:
            async for raw in self.avatar:
                if isinstance(raw, bytes):
                    if q.full():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    q.put_nowait(bytes([FRAME_TYPE_JPEG]) + raw)
        finally:
            task.cancel()


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
    vision_cfg = vision.vision_config(config)
    filler_enabled = bool((config.get("filler") or {}).get("enabled", True))

    async def index(_request):
        return web.FileResponse(REPO_ROOT / "web" / "index.html")

    async def api_personas(_request):
        return web.json_response({
            "default": default_persona,
            "avatar": "on" if avatar_url else "off",
            "vision": "on" if vision_cfg else "off",
            "trigger": (vision_cfg or {}).get("trigger", "让我好好看看你"),
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

    # 单用户产品：新浏览器连接顶掉旧会话（换网络/僵尸会话不再需要刷新两次）
    current_session: dict = {"session": None}

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        old = current_session["session"]
        if old is not None:
            logger.info("新连接到达，顶掉旧会话（释放管线槽位）")
            await old.close()
        session = Session(ws, s2s_url, avatar_url, personas, default_persona,
                          vision_cfg=vision_cfg, filler_enabled=filler_enabled)
        current_session["session"] = session
        try:
            await session.run()
        finally:
            if current_session["session"] is session:
                current_session["session"] = None
            if session.avatar is not None:
                await session.avatar.close()
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/personas", api_personas)
    app.router.add_get("/api/personas/{pid}/image", api_persona_image)
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
