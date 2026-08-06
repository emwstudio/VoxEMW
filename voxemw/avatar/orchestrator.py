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
- listen 双流：用户说话段的麦克风音频 tee 给 avatar 做 active listening
- 记忆：会话开始把 persona 记忆注入 instructions；response.done 后异步写入
- 垫音（filler，默认关）：转写完成即播预渲染口头禅盖 LLM 首句空白
- 降级：avatar 缺席时纯语音模式，前端显示静态肖像
- 单用户单会话：新浏览器连接顶掉旧会话（s2s 只有 1 个管线槽位，
  换网络产生的僵尸会话被新连接立即踢掉，无需等超时/刷新两次）

浏览器侧协议（/ws）：
  文本帧（JSON）：
    → {"type": "vox.persona", "id": "<persona_id>"}   切换人设
    → {"type": "vox.drained"}                          播放排空信号（帧合流时序用）
    → OpenAI Realtime 事件原样透传（input_audio_buffer.append / response.cancel 等）
    ← OpenAI Realtime 事件原样透传（transcription / response.done 等）
    ← {"type": "vox.status", "avatar": "on"|"off", "persona": "<id>", ...}
  二进制帧：
    ← 0x01 + tag(1B) + JPEG：数字人视频帧（tag 0x00=idle 直画 / 0x01=speech 进队列）
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
                 personas: dict, default_persona: str,
                 filler_enabled: bool = True, avatar_backend: str = "avtr1",
                 memory_store=None, weibo_cfg: dict | None = None):
        self.browser = browser_ws
        self.s2s_url = s2s_url
        self.avatar_url = avatar_url
        self.avatar_backend = avatar_backend  # 前端据此选口型延迟默认值（adelay）
        self.personas = personas
        self.persona_id = default_persona
        self.memory = memory_store  # 记忆积木（None = 未启用/降级）
        weibo_cfg = weibo_cfg or {}
        self.weibo_db = weibo_cfg.get("db_path") if weibo_cfg.get("enabled") else None
        self.weibo_top_n = int(weibo_cfg.get("top_n", 8))
        self._turn_user_text = ""       # 本轮用户转写（记忆写入用）
        self._turn_assistant_text = ""  # 本轮峰哥回复（记忆写入用）
        self._user_speaking = False  # server VAD 判定的用户说话段（listen 轨转发门控）
        self.s2s = None
        self.avatar = None
        self._avatar_speaking = False  # 是否已向 avatar 下发 speech_active=on
        self._filler_enabled = filler_enabled
        self._fillers = load_fillers(personas[default_persona])
        self._filler_last: dict[str, int] = {}  # 每个分组各自记上一条，避免连续重复
        self._filler_task: asyncio.Task | None = None

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
            "avatar_backend": self.avatar_backend if self.avatar is not None else "off",
            "persona": self.persona_id,
        }))

    async def _apply_persona(self, persona_id: str) -> None:
        persona = self.personas[persona_id]
        self.persona_id = persona_id
        self._fillers = load_fillers(persona)
        instructions = persona["text"]
        if self.memory is not None:
            try:
                from voxemw.memory import build_memory_block

                memories = await asyncio.to_thread(self.memory.search, persona_id)
                block = build_memory_block(memories)
                if block:
                    instructions = instructions + "\n\n" + block
                    logger.info("记忆注入 %d 条", len(memories))
            except Exception as e:
                logger.warning("记忆召回失败（跳过）: %s", e)
        if self.weibo_db:
            from voxemw.weibo import build_posts_block, get_recent_posts

            posts = await asyncio.to_thread(
                get_recent_posts, self.weibo_db, self.weibo_top_n
            )
            block = build_posts_block(posts)
            if block:
                instructions = instructions + "\n\n" + block
                logger.info("动态注入 %d 条", len(posts))
        await self.s2s.send(json.dumps(build_session_update(persona_id, instructions)))
        if self.avatar is not None:
            image = persona.get("ref_image")
            if image:
                await self.avatar.send(json.dumps({"type": "set_image", "path": image}))
            await self.avatar.send(json.dumps({"type": "reset"}))

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

    def _maybe_write_memory(self) -> None:
        """response.done → 异步写入本轮对话到记忆（Mem0 抽取，不占语音延迟）。"""
        user_text, assistant_text = self._turn_user_text, self._turn_assistant_text
        self._turn_user_text = ""
        self._turn_assistant_text = ""
        if self.memory is None or not user_text:
            return

        async def _write():
            try:
                await asyncio.to_thread(
                    self.memory.add_turn, user_text, assistant_text, self.persona_id
                )
            except Exception as e:
                logger.info("记忆写入失败（忽略）: %s", e)

        asyncio.create_task(_write())

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
            # 记忆：跟踪本轮转写文本（写入发生在 response.done）
            if etype == "conversation.item.input_audio_transcription.completed":
                self._turn_user_text = (event.get("transcript") or "").strip()
            elif etype == "response.output_audio_transcript.done":
                self._turn_assistant_text = (event.get("transcript") or "").strip()
            elif etype == "response.done":
                self._maybe_write_memory()
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
        if etype == "input_audio_buffer.speech_started":
            self._user_speaking = True
        elif etype == "input_audio_buffer.speech_stopped":
            self._user_speaking = False

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
    avatar_backend = str(avatar_cfg.get("backend", "avtr1")) if avatar_available else "off"
    filler_enabled = bool((config.get("filler") or {}).get("enabled", True))

    from voxemw.memory import create_memory_store

    memory_store = create_memory_store(config)

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
                          filler_enabled=filler_enabled,
                          avatar_backend=avatar_backend, memory_store=memory_store,
                          weibo_cfg=config.get("weibo"))
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
