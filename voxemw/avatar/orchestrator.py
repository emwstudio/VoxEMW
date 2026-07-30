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
- 降级：avatar 缺席时纯语音模式，前端显示静态肖像

浏览器侧协议（/ws）：
  文本帧（JSON）：
    → {"type": "vox.persona", "id": "<persona_id>"}   切换人设
    → OpenAI Realtime 事件原样透传（input_audio_buffer.append / response.cancel 等）
    ← OpenAI Realtime 事件原样透传（transcription / response.done 等）
    ← {"type": "vox.status", "avatar": "on"|"off", "persona": "<id>"}
  二进制帧：
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
                 personas: dict, default_persona: str):
        self.browser = browser_ws
        self.s2s_url = s2s_url
        self.avatar_url = avatar_url
        self.personas = personas
        self.persona_id = default_persona
        self.s2s = None
        self.avatar = None

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
            relay, reset_avatar, pcm = classify_s2s_event(event)
            if pcm is not None and self.avatar is not None:
                await self.avatar.send(json.dumps({
                    "type": "audio",
                    "pcm": base64.b64encode(pcm).decode(),
                }))
            if reset_avatar and self.avatar is not None:
                await self.avatar.send(json.dumps({"type": "reset"}))
            if relay:
                await self.browser.send_str(raw)

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

    async def index(_request):
        return web.FileResponse(REPO_ROOT / "web" / "index.html")

    async def api_personas(_request):
        return web.json_response({
            "default": default_persona,
            "avatar": "on" if avatar_url else "off",
            "list": [
                {
                    "id": pid,
                    "name": p["name"],
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
        session = Session(ws, s2s_url, avatar_url, personas, default_persona)
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
