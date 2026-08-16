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
                 rtc_sched=None, rtc_ice_servers: list | None = None):
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
        self._resp_had_content = False  # 本轮回复是否有任何文本/音频产出（空回复兜底用）
        self._empty_nudged = False      # 本轮是否已追问过（防追问死循环）

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
            # 空回复兜底追踪：本轮有任何文本/音频产出即视为有内容
            etype = event.get("type", "")
            if etype == "response.created":
                self._resp_had_content = False
            elif pcm is not None or (
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
                          rtc_sched=sched, rtc_ice_servers=rtc_ice_servers)
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
