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
- 截帧打分两阶段：收到用户截帧 → 先注入「垫场」指令让 persona 自由发挥，
  Kimi 描述返回后再注入打分指令，填掉等待空白。注入是乐观发送+被拒重试
  （VAD 回复不发 response.created，无法预判回复是否在播；response.create 被
  conversation_already_has_active_response 拒绝时，等当前回复 response.done 重发）。
  从截帧到打分回复说完期间，上行麦克风音频直接丢弃（用户插话不打断打分）
- 降级：avatar 缺席时纯语音模式，前端显示静态肖像

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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from voxemw import vision

logger = logging.getLogger(__name__)

FRAME_TYPE_JPEG = 0x01

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


class Session:
    """一个浏览器连接 ↔ 一路 s2s + 一路 avatar 的编排。"""

    def __init__(self, browser_ws, s2s_url: str, avatar_url: str | None,
                 personas: dict, default_persona: str, vision_cfg: dict | None = None):
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
            for task in pending:
                task.cancel()
            for task in done:
                if task.exception():
                    logger.info("session 结束: %s", task.exception())

    async def _send_status(self) -> None:
        await self.browser.send_str(json.dumps({
            "type": "vox.status",
            "avatar": "on" if self.avatar is not None else "off",
            "persona": self.persona_id,
        }))

    async def _apply_persona(self, persona_id: str) -> None:
        persona = self.personas[persona_id]
        self.persona_id = persona_id
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
            relay, reset_avatar, pcm = classify_s2s_event(event)
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

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        session = Session(ws, s2s_url, avatar_url, personas, default_persona,
                          vision_cfg=vision_cfg)
        try:
            await session.run()
        finally:
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
