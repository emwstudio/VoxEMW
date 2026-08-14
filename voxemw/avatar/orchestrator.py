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
    → OpenAI Realtime 事件原样透传（input_audio_buffer.append / response.cancel 等）
    ← OpenAI Realtime 事件透传（transcription / response.done 等；
      音频 delta 事件剥掉 base64 音频体——音频走 RTC 音轨，只留事件）
    ← {"type": "vox.status", "avatar": "on"|"off", "persona": "<id>",
       "rtc": {"enabled": bool, "ice_servers": [...]}}
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

FRAME_TAG_SPEECH = 0x01  # avatar service 下行帧 tag：0x00=idle / 0x01=speech

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


DANCE_MARKER_RE = re.compile(r"\s*\[\[dance:([^\]]+)\]\]")


def split_dance_marker(text: str) -> tuple[str | None, str]:
    """回复开头的 [[dance:舞名]] 标记 → (舞名|None, 剥除后的文本)。纯函数便于单测。"""
    m = DANCE_MARKER_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():]
    return None, text


def is_marker_prefix(text: str) -> bool:
    """text 是否仍可能是 [[dance:...]] 标记的未完结前缀（流式攒批用）。"""
    t = text.lstrip()
    if "]]" in t or len(t) > 48:
        return False
    prefix = "[[dance:"
    return prefix.startswith(t) if len(t) <= len(prefix) else t.startswith(prefix)


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


class Session:
    """一个浏览器连接 ↔ 一路 s2s + 一路 avatar 的编排。"""

    def __init__(self, browser_ws, s2s_url: str, avatar_url: str | None,
                 personas: dict, default_persona: str,
                 avatar_backend: str = "avtr1",
                 rtc_sched=None, rtc_ice_servers: list | None = None,
                 get_dances=None):
        self.browser = browser_ws
        self.s2s_url = s2s_url
        self.avatar_url = avatar_url
        self.avatar_backend = avatar_backend
        # WebRTC 音画轨：下行音画走 RTC，WS 只留控制/转写
        self.sched = rtc_sched
        self._rtc_ice_servers = rtc_ice_servers or []
        self.personas = personas
        self.persona_id = default_persona
        self._user_speaking = False  # server VAD 判定的用户说话段（listen 轨转发门控）
        self.s2s = None
        self.avatar = None
        self._avatar_speaking = False  # 是否已向 avatar 下发 speech_active=on
        self._get_dances = get_dances or (lambda: [])  # 素材库舞名（点舞用）
        self._dance_buf = ""          # 回复开头攒批（点舞标记探测）
        self._dance_checked = False   # 本回复标记探测是否已结论
        self._pending_dance = None    # 捕获到的舞名，等他说完（response.done）再起舞台
        self._dance_epoch = 0         # 点舞代际：用户开口即+1，作废等待中的起播任务

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
        }))

    async def _apply_persona(self, persona_id: str) -> None:
        persona = self.personas[persona_id]
        self.persona_id = persona_id
        instructions = persona["text"]
        dances = self._get_dances()
        if dances:
            instructions += (
                "\n\n# 隐藏技能：跳舞（系统指令，优先级最高）\n"
                "你是会跳舞的——素材库里已经排好了你的跳舞视频：" + "、".join(dances)
                + "。用户点舞且舞名在列表中时，回复必须以 [[dance:舞名]] 开头"
                  "（精确用列表名，该标记会被系统吃掉、不会播出），然后正常说开场白；"
                  "你说完后系统会自动全屏播放你的跳舞视频。不要说自己不会跳。"
                  "列表没有的舞就按你的风格调侃回绝，让用户去素材库排新舞。"
            )
            logger.info("点舞列表已注入人设: %s", dances)
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
            if event.get("type") == "vox.dance_done":
                # 舞台播完静默回通话（试过自动收场白，用户反馈多余，2026-08-13 去掉）
                logger.info("舞蹈播完: %s", event.get("name", ""))
                continue
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
            etype = event.get("type", "")
            # 回复播完：有 pending 点舞则等音频播干再通知前端起舞台（先说后跳）
            if etype == "response.done" and self._pending_dance:
                name = self._pending_dance
                self._pending_dance = None
                asyncio.create_task(self._fire_dance_when_drained(name, self._dance_epoch))
            # 点舞标记捕获：[[dance:舞名]] 在回复开头的转写流里——剥掉不上字幕，
            # 转成 vox.dance 事件给前端（舞台播放）
            if etype in ("response.output_audio_transcript.delta",
                         "response.output_text.delta",
                         "response.output_audio_transcript.done"):
                if await self._handle_transcript_event(event):
                    continue
            relay, reset_avatar, pcm = classify_s2s_event(event)
            if pcm is not None and self.sched is not None:
                self.sched.feed_audio(pcm)  # RTC 音频轨（与喂 avatar 同一股流）
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
            if reset_avatar and self.sched is not None:
                self.sched.flush()  # 打断：清 RTC 音画队列
            if relay:
                if pcm is not None:
                    # 音频走 RTC 音轨，WS 只留事件本身（剥掉 base64 音频体省带宽）
                    event = {k: v for k, v in event.items() if k != "delta"}
                    await self.browser.send_str(json.dumps(event))
                else:
                    await self.browser.send_str(raw)

    async def _fire_dance_when_drained(self, name: str, epoch: int) -> None:
        """等回复音频在 RTC 轨道播干（buffered≈0）再发 vox.dance——response.done
        时末句音频还在缓冲里，直接起舞台会打断他说话。用户开口（epoch 变）则作废。"""
        try:
            for _ in range(150):  # 0.1s × 150 = 15s 兜底
                if epoch != self._dance_epoch:
                    return
                if self.sched is None or self.sched.buffered_audio_seconds < 0.15:
                    break
                await asyncio.sleep(0.1)
            if epoch != self._dance_epoch:
                return
            await self.browser.send_str(json.dumps({"type": "vox.dance", "name": name}))
        except Exception:
            logger.debug("舞台通知失败（连接已断？）", exc_info=True)

    def _track_dialog_state(self, event: dict) -> None:
        etype = event.get("type", "")
        if etype == "input_audio_buffer.speech_started":
            self._user_speaking = True
            # 新一轮对话开始：重置点舞标记探测（防打断残留状态泄漏到下轮）
            self._dance_checked = False
            self._dance_buf = ""
            self._pending_dance = None  # 打断时舞台未起，作废
            self._dance_epoch += 1      # 作废等待中的起播任务
        elif etype == "input_audio_buffer.speech_stopped":
            self._user_speaking = False

    async def _handle_transcript_event(self, event: dict) -> bool:
        """转写事件拦截：探测/剥除回复开头的 [[dance:舞名]] 点舞标记。
        返回 True = 已处理（调用方 continue）。标记在开头 48 字内，
        未完结前缀先攒批不转发（防标记碎在多个 delta 里漏到字幕）。
        注意：本管线按整句发 transcript.done（无 delta），检测必须在 done 路径生效。"""
        etype = event.get("type", "")
        is_done = etype.endswith(".done")
        key = "transcript" if is_done else "delta"
        text = event.get(key) or ""

        if not self._dance_checked:
            self._dance_buf += text
            if is_marker_prefix(self._dance_buf):
                return True  # 还在标记窗口，攒着
            name, rest = split_dance_marker(self._dance_buf)
            self._dance_checked = True
            self._dance_buf = ""
            if name and name in set(self._get_dances()):
                logger.info("点舞: %s（等回复播完再起舞台）", name)
                self._pending_dance = name
            text = rest

        if not text:
            return True  # 标记剥完后无内容，不再转发
        event = dict(event)
        event[key] = text
        await self.browser.send_str(json.dumps(event, ensure_ascii=False))
        return True

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
        answer = await rtc_manager.handle_offer(offer, session.sched)
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

    # ── 跳舞素材库（Wan-Animate-2 离线生成 + 语音点舞）──
    import queue as _queue
    import subprocess
    import threading

    dance_dir = REPO_ROOT / "data" / "dance"
    dance_dir.mkdir(parents=True, exist_ok=True)
    fullbody_path = dance_dir / "_fullbody.png"  # 全身照（素材库级资产）
    dance_jobs: dict[str, str] = {}              # 舞名 → 状态文案（页面轮询）
    dance_queue: _queue.Queue = _queue.Queue()

    def _dance_names() -> list[str]:
        return sorted(p.stem for p in dance_dir.glob("*.mp4")
                      if not p.stem.endswith(".driving"))

    def _driving_map() -> dict[str, str]:
        # 保留的驱动视频（预览用）：舞名 → 文件名（排除同名的 .driving.jpg 封面）
        return {p.name.split(".driving")[0]: p.name
                for p in dance_dir.glob("*.driving.*")
                if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")}

    def _dance_worker_loop() -> None:
        """串行生成：14B 模型要整张卡，先停 avatar + pipeline，完成后拉起。"""
        while True:
            job = dance_queue.get()
            name, ref_image, driving, mode, prompt, seed = job
            try:
                dance_jobs[name] = "停数字人服务…"
                subprocess.run(["pkill", "-f", "voxemw.avatar.service"], check=False)
                subprocess.run(["pkill", "-f", "voxemw.pipeline.launch"], check=False)
                time.sleep(6)
                dance_jobs[name] = "生成中（数分钟，勿通话）…"
                out = str(dance_dir / f"{name}.mp4")
                # 实例内存限流偶发 SIGKILL，失败自动重试一次
                for attempt in (1, 2):
                    try:
                        subprocess.run([
                            str(REPO_ROOT / ".venv/bin/python"), "-m", "voxemw.dance_worker",
                            "--ref-image", ref_image, "--driving-video", driving,
                            "--name", name, "--mode", mode, "--prompt", prompt,
                            "--seed", seed,
                            "--out", out,
                        ], cwd=str(REPO_ROOT), check=True, timeout=7200)
                        break
                    except subprocess.CalledProcessError:
                        if attempt == 2:
                            raise
                        logger.warning("生成被中断，自动重试: %s", name)
                        dance_jobs[name] = "被系统中断，自动重试中…"
                        time.sleep(10)
                # 自动超分（Real-ESRGAN anime6B，~3min）：原生 416x736 → 720p 档成片
                dance_jobs[name] = "超分出高清版…"
                hd = str(dance_dir / f"{name}.hd_tmp.mp4")
                try:
                    subprocess.run([
                        str(REPO_ROOT / ".venv/bin/python"), "scripts/upscale_video.py",
                        out, hd,
                    ], cwd=str(REPO_ROOT), check=True, timeout=1200)
                    os.replace(hd, out)  # 原子替换：HD 覆盖原生分辨率
                except Exception:
                    logger.exception("超分失败，保留原生分辨率版: %s", name)
                    try:
                        os.unlink(hd)
                    except OSError:
                        pass
                # 生成封面图（页面卡片 poster）
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out,
                                "-frames:v", "1", "-vf", "scale=480:-1",
                                str(dance_dir / f"{name}.jpg")], check=False)
                dance_jobs[name] = "完成"
            except Exception as e:
                logger.exception("跳舞素材生成失败: %s", name)
                dance_jobs[name] = f"失败：{e}"
            finally:
                # 拉起 pipeline（STT/LLM/TTS），再拉起 avatar 服务（无论成败）
                config = os.environ.get("VOXEMW_CONFIG", "configs/assistant.yaml")
                subprocess.Popen(
                    [str(REPO_ROOT / ".venv/bin/python"), "-m", "voxemw.pipeline.launch",
                     "--config", config],
                    cwd=str(REPO_ROOT), env={**os.environ},
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.Popen(["bash", "scripts/restart_avatar.sh"],
                                 cwd=str(REPO_ROOT),
                                 env={**os.environ},
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # 驱动视频保留在素材目录（name.driving.mp4），供页面预览对照

    threading.Thread(target=_dance_worker_loop, daemon=True).start()

    async def dance_page(_request):
        return web.FileResponse(REPO_ROOT / "web" / "dance.html")

    async def api_dance_list(_request):
        return web.json_response({
            "dances": _dance_names(),
            "jobs": dict(dance_jobs),
            "driving": _driving_map(),
            "seeds": {p.stem: p.read_text().strip()
                      for p in dance_dir.glob("*.seed")},
            "has_fullbody": fullbody_path.is_file(),
        })

    async def api_dance_upload(request):
        form = await request.post()
        video = form.get("video")
        name = (form.get("name") or "").strip()
        mode = str(form.get("mode", "move"))
        prompt = str(form.get("prompt", "")).strip()
        if video is None or not getattr(video, "filename", ""):
            return web.json_response({"error": "缺少驱动视频（字段名 video）"}, status=400)
        if not name:
            return web.json_response({"error": "缺舞蹈名（字段名 name）"}, status=400)
        if mode not in ("move", "mix"):
            return web.json_response({"error": "mode 只能是 move/mix"}, status=400)
        # 全身照：随单上传则更新库级资产；没传用库存
        photo = form.get("photo")
        if photo is not None and getattr(photo, "filename", ""):
            fullbody_path.write_bytes(photo.file.read())
        if not fullbody_path.is_file():
            return web.json_response({"error": "缺少全身照（首次必须上传）"}, status=400)
        # 驱动视频存进素材目录（生成完保留，供页面预览对照）+ 封面图
        driving_path = dance_dir / f"{name}.driving{Path(video.filename).suffix or '.mp4'}"
        driving_path.write_bytes(video.file.read())
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(driving_path),
                        "-frames:v", "1", "-vf", "scale=480:-1",
                        str(dance_dir / f"{name}.driving.jpg")], check=False)
        dance_jobs[name] = "排队中…"
        seed = str(form.get("seed", "")).strip() or "-1"  # 空=随机抽卡
        if seed == "-1":
            # 主动重抽：清掉旧 seed 卡，worker 会抽新的并重新固化
            # （.seed 复用只为同一任务内的断点续跑/被杀重试服务）
            (dance_dir / f"{name}.seed").unlink(missing_ok=True)
        dance_queue.put((name, str(fullbody_path), str(driving_path), mode, prompt, seed))
        return web.json_response({"name": name, "status": "queued"})

    async def api_dance_delete(request):
        body = await request.json()
        name = (body or {}).get("name", "")
        target = dance_dir / f"{name}.mp4"
        ok = target.is_file()
        if ok:
            target.unlink()
        thumb = dance_dir / f"{name}.jpg"
        if thumb.is_file():
            thumb.unlink()
        seed_f = dance_dir / f"{name}.seed"
        if seed_f.is_file():
            seed_f.unlink()
        for p in dance_dir.glob(f"{name}.driving.*"):
            p.unlink()
        import shutil
        shutil.rmtree(dance_dir / ".segments" / name, ignore_errors=True)
        dance_jobs.pop(name, None)
        return web.json_response({"deleted": ok})

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        old = current_session["session"]
        if old is not None:
            logger.info("新连接到达，顶掉旧会话（释放管线槽位）")
            await old.close()
        # ?alead=毫秒：音频压后量（等 avatar 渲染追赶，音画对齐的关键补偿，可调）
        try:
            lead = float(request.query.get("alead", "250")) / 1000.0
        except ValueError:
            lead = 0.25
        sched = AVSyncScheduler(audio_lead=lead) if rtc_manager is not None else None
        session = Session(ws, s2s_url, avatar_url, personas, default_persona,
                          avatar_backend=avatar_backend,
                          rtc_sched=sched, rtc_ice_servers=rtc_ice_servers,
                          get_dances=_dance_names)
        current_session["session"] = session
        try:
            await session.run()
        finally:
            if current_session["session"] is session:
                current_session["session"] = None
            if session.avatar is not None:
                await session.avatar.close()
        return ws

    app = web.Application(client_max_size=256 * 1024 * 1024)  # 驱动视频/换图上传可达几十 MB

    @web.middleware
    async def _no_cache(request, handler):
        # 前端 JS/HTML 迭代频繁，禁缓存防浏览器跑旧版（新旧协议不匹配会静默失声）
        resp = await handler(request)
        if request.path in ("/", "/dance") or request.path.startswith("/static"):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp

    app.middlewares.append(_no_cache)
    app.router.add_get("/", index)
    app.router.add_get("/dance", dance_page)
    app.router.add_get("/api/dance/list", api_dance_list)
    app.router.add_post("/api/dance/upload", api_dance_upload)
    app.router.add_post("/api/dance/delete", api_dance_delete)
    app.router.add_static("/dance_media", dance_dir)  # 成片播放
    app.router.add_get("/api/personas", api_personas)
    app.router.add_get("/api/personas/{pid}/image", api_persona_image)
    app.router.add_post("/api/personas/{pid}/image", api_persona_image_upload)
    app.router.add_post("/rtc/offer", api_rtc_offer)
    app.router.add_get("/rtc/ice", api_rtc_ice)
    app.router.add_post("/rtc/debug", api_rtc_debug)
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
