"""Orchestrator：浏览器唯一入口，编排语音管线（s2s）与 RTC 音频轨。

架构：
    浏览器 ←→ 本进程（aiohttp，:8000）
              └→ s2s realtime ws（:8765，voxemw.pipeline.launch 起的语音管线）

职责：
- 下行：s2s 的 TTS 音频 delta → AudioPacer（RTC 音频轨，Opus 48k）；
  WS 只留控制/转写事件（音频 delta 剥掉 base64 音频体省带宽）
- 上行：浏览器麦克风音频/控制消息 → 转发 s2s
- persona：浏览器发 {"type": "vox.persona", "id": ...} 切换人设，
  本进程把人设正文/音色经 session.update 注入 s2s（instructions + voice）
- 打断：s2s 报 speech_started → flush 音频队列；若回复播了一半，
  把已听前缀补写回上下文（heard_prefix，防模型忘记自己说到哪）
- 单用户单会话：新浏览器连接顶掉旧会话（s2s 只有 1 个管线槽位，
  换网络产生的僵尸会话被新连接立即踢掉，无需等超时/刷新两次）

浏览器侧协议：
  /ws 文本帧（JSON）：
    → {"type": "vox.persona", "id": "<persona_id>"}   切换人设
    → OpenAI Realtime 事件原样透传（input_audio_buffer.append / response.cancel 等）
    ← OpenAI Realtime 事件透传（transcription / response.done 等）
    ← {"type": "vox.status", "persona": "<id>",
       "rtc": {"enabled": bool, "ice_servers": [...]}}
  POST /rtc/offer：WebRTC 信令，body {"sdp", "type"} → answer
  GET  /rtc/ice ：下发 ICE 配置
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

# s2s 事件 → 编排动作（纯函数分类，便于单测）
AUDIO_DELTA_EVENTS = {"response.output_audio.delta", "response.audio.delta"}  # GA / beta 名都收
INTERRUPT_EVENTS = {
    "input_audio_buffer.speech_started",  # 用户开口（打断）：清音频队列
}


def build_session_update(persona_id: str, persona_text: str) -> dict:
    """注入人设的 session.update：instructions = 人设正文，voice = persona id
    （TTS voices 表 key）。"""
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
    """分类 s2s 下行事件。返回 (relay_to_browser, is_interrupt, audio_pcm|None)。"""
    etype = event.get("type", "")
    pcm = None
    if etype in AUDIO_DELTA_EVENTS:
        delta = event.get("delta")
        if delta:
            pcm = base64.b64decode(delta)
    return True, etype in INTERRUPT_EVENTS, pcm


_PUNCT = "。！？；，、…—.!?;,"


def heard_prefix(transcript: str, audio_seconds: float, played_seconds: float) -> str:
    """打断时估算用户实际听到的文本前缀（纯函数，便于单测）。

    播放进度占已生成音频的比例 ≈ 听到的文本比例（中文语速在一条回复内
    足够均匀）。不足 2 字不值得注入（上游会把整条回复从上下文回滚，
    零前缀=保持回滚）。截断处回退到最近的标点，避免半个词留在上下文里。
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


_NORM_RE = re.compile(r"[\s，。！？、,.!?…~—「」『』\"'：:；;（）()【】\[\]]+")


def is_echo(user_transcript: str, recent_assistant: list[str]) -> bool:
    """回声回合判定（纯函数，便于单测）：转写出的「用户话」其实是助手
    自己的声音被麦克风收回去（外放泄漏）——特征是与近期助手文本互相包含。

    规则：去标点空白后，候选 ≥4 字 且 与任一近期助手文本存在包含关系
    （候选 ⊆ 助手 或 助手 ⊆ 候选）。短句（<4 字）永不判回声——「你好啊」
    这种真实短句撞车概率太高。助手历史由调用方限制在近 2 轮，口癖复读
    （用户故意学说良子的话）长度够也会被误杀——接受这个代价，外放自激
    更烦。"""
    candidate = _NORM_RE.sub("", user_transcript or "")
    if len(candidate) < 4:
        return False
    for past in recent_assistant:
        p = _NORM_RE.sub("", past or "")
        if p and (candidate in p or p in candidate):
            return True
    return False


class Session:
    """一个浏览器连接 ↔ 一路 s2s 的编排。"""

    def __init__(self, browser_ws, s2s_url: str,
                 personas: dict, default_persona: str,
                 rtc_pacer=None, rtc_ice_servers: list | None = None):
        self.browser = browser_ws
        self.s2s_url = s2s_url
        # WebRTC 音频轨：下行音频走 RTC，WS 只留控制/转写
        self.pacer = rtc_pacer
        self._rtc_ice_servers = rtc_ice_servers or []
        self.personas = personas
        self.persona_id = default_persona
        self.s2s = None
        self._assistant_speaking = False  # 本轮回复有音频在播（打断回报判定用）
        self._resp_had_content = False  # 本轮回复是否有任何文本/音频产出（空回复兜底用）
        self._empty_nudged = False      # 本轮是否已追问过（防追问死循环）
        self._reply_transcript = ""     # 本轮回复的转写文本（打断回报估算用）
        self._reply_audio_samples = 0   # 本轮回复已生成音频采样数（同上）
        self._assistant_history: list[str] = []  # 近 2 轮完整回复（回声判定用）
        self._suppress_ghost = False    # 回声回合压制中：丢弃该回合全部 response 事件

    async def run(self) -> None:
        import websockets

        async with websockets.connect(self.s2s_url, max_size=16 * 1024 * 1024) as s2s:
            self.s2s = s2s
            await self._apply_persona(self.persona_id)
            await self._send_status()
            tasks = [
                asyncio.create_task(self._browser_to_s2s()),
                asyncio.create_task(self._s2s_to_browser()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
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
            "persona": self.persona_id,
            "rtc": {"enabled": self.pacer is not None, "ice_servers": self._rtc_ice_servers},
        }))

    async def _apply_persona(self, persona_id: str) -> None:
        persona = self.personas[persona_id]
        self.persona_id = persona_id
        await self.s2s.send(json.dumps(build_session_update(persona_id, persona["text"])))

    # ── 两条转发协程 ──

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
            relay, is_interrupt, pcm = classify_s2s_event(event)
            etype = event.get("type", "")
            # ── 回声回合压制：外放泄漏把助手自己的话收成「用户说」──
            if self._suppress_ghost:
                if etype == "response.done" or is_interrupt:
                    self._suppress_ghost = False  # 幽灵回合结束/真人新开口：解除
                elif etype.startswith("response."):
                    continue  # 幽灵回合事件全丢：不上屏、不出声、不计数
            if etype == "conversation.item.input_audio_transcription.completed":
                heard = event.get("transcript", "")
                if is_echo(heard, self._assistant_history + [self._reply_transcript]):
                    logger.info("回声回合压制：%r 与近期助手文本重合，掐掉", heard[:30])
                    self._suppress_ghost = True
                    await self.s2s.send(json.dumps({"type": "response.cancel"}))
                    continue  # 转写不上屏
            # 空回复兜底追踪：本轮有任何文本/音频产出即视为有内容
            if etype == "response.created":
                if self._reply_transcript:
                    self._assistant_history = (
                        self._assistant_history + [self._reply_transcript])[-2:]
                self._resp_had_content = False
                self._reply_transcript = ""     # 本轮回复转写（打断回报用）
                self._reply_audio_samples = 0   # 本轮回复已生成音频采样
            if etype == "response.output_audio_transcript.delta":
                self._reply_transcript += event.get("delta", "")
            if pcm is not None:
                self._reply_audio_samples += len(pcm) // 2
                self._assistant_speaking = True
            if pcm is not None or (
                etype in ("response.output_audio_transcript.delta",
                          "response.output_text.delta",
                          "response.output_audio_transcript.done")
                and (event.get("delta") or event.get("transcript"))
            ):
                self._resp_had_content = True
            elif etype == "response.done":
                self._assistant_speaking = False
                status = (event.get("response") or {}).get("status")
                if (not self._resp_had_content and not self._empty_nudged
                        and status in (None, "completed")):
                    # LLM 偶发只吐 1 个 token → 清理后无声。
                    # 追问一次让模型重答，把抽风变成一句话的事
                    self._empty_nudged = True
                    logger.warning("空回复兜底：本轮无文本/音频产出，追问重答")
                    await self.s2s.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": "user", "content": [{
                            "type": "input_text",
                            "text": "（你刚才的回复是空的，用一句符合你人设的话接上——比如假装清了清嗓子——然后正常回答我刚才的问题。别提这条提示）"}]}}))
                    await self.s2s.send(json.dumps({"type": "response.create"}))
            if is_interrupt:
                was_speaking = self._assistant_speaking  # 判定要在状态翻转前取
                self._assistant_speaking = False
                self._empty_nudged = False  # 新一轮对话，重置追问名额
                if was_speaking and self.pacer is not None:
                    # 打断回报：上游会把整条回复从上下文回滚，但用户实际已经
                    # 听到了一段——把已听前缀作为 assistant 消息补写回上下文
                    #（须在下方 flush 清计数前读）
                    played_s = self.pacer.reply_played_seconds
                    prefix = heard_prefix(self._reply_transcript,
                                          self._reply_audio_samples / 16000, played_s)
                    if prefix:
                        logger.info("打断回报：已播 %.1fs，已听前缀 %d 字补写回上下文",
                                    played_s, len(prefix))
                        await self.s2s.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "message", "role": "assistant",
                                     "content": [{"type": "output_text", "text": prefix}]}}))
                if self.pacer is not None:
                    self.pacer.flush()  # 打断：清 RTC 音频队列
            if pcm is not None and self.pacer is not None:
                self.pacer.feed_audio(pcm)  # RTC 音频轨
            if relay:
                if pcm is not None:
                    # 音频走 RTC 音轨，WS 只留事件本身（剥掉 base64 音频体省带宽）
                    event = {k: v for k, v in event.items() if k != "delta"}
                    await self.browser.send_str(json.dumps(event))
                else:
                    await self.browser.send_str(raw)


def create_app(config: dict):
    from aiohttp import web

    server = config.get("server") or {}
    personas = config["personas"]["resolved"]
    default_persona = config["personas"]["default"]

    s2s_url = f"ws://{server.get('s2s_host', '127.0.0.1')}:{server.get('s2s_port', 8765)}/v1/realtime"

    async def index(_request):
        return web.FileResponse(REPO_ROOT / "web" / "index.html")

    async def api_personas(_request):
        return web.json_response({
            "default": default_persona,
            "list": [
                {
                    "id": pid,
                    "name": p["name"],
                    "label": p.get("label") or p["name"],
                }
                for pid, p in personas.items()
            ],
        })

    # ── WebRTC 音频轨 ──
    rtc_cfg = config.get("rtc") or {}
    rtc_manager = None
    rtc_ice_servers: list = []
    if rtc_cfg.get("enabled", False):
        from voxemw.gateway.rtc import RTCManager

        rtc_manager = RTCManager(rtc_cfg)
        rtc_ice_servers = rtc_manager.browser_ice_servers
        logger.info("WebRTC 音频轨启用（Opus 48k）")

    # 单用户产品：新浏览器连接顶掉旧会话（换网络/僵尸会话不再需要刷新两次）
    current_session: dict = {"session": None}

    async def api_rtc_offer(request):
        if rtc_manager is None:
            return web.json_response({"error": "rtc 未启用"}, status=404)
        session = current_session["session"]
        if session is None or session.pacer is None:
            return web.json_response({"error": "无活跃会话，先连 /ws"}, status=409)
        session.pacer.flush()  # 新 RTC 连接从干净的队列起步（重连不播陈年积压）
        offer = await request.json()
        try:
            answer = await rtc_manager.handle_offer(offer, session.pacer)
        except Exception:
            # 排障：把失败的 offer SDP 落日志（Safari/老 WebKit 的 m-section 顺序差异）
            logger.exception("RTC offer 处理失败，SDP 前 800 字: %s",
                             str(offer.get("sdp"))[:800])
            raise
        return web.json_response(answer)

    async def api_rtc_ice(request):
        # 前端每次建连现取 ICE 配置
        if rtc_manager is None:
            return web.json_response({"ice_servers": []})
        return web.json_response({"ice_servers": rtc_manager.browser_ice_servers})

    async def api_rtc_debug(request):
        # 前端 12s 后回传的 ICE 诊断（候选/连接状态）
        body = await request.json()
        logger.info("RTC 前端诊断: %s", json.dumps(body, ensure_ascii=False))
        return web.json_response({"ok": True})

    async def api_pacer_debug(_request):
        """排障观测点：当前会话音频调度器的喂入/播放计数。
        fed>0 而 played 不动 = RTC 音频轨没在消费；fed=0 = 音频没喂进这条会话。"""
        session = current_session["session"]
        p = session.pacer if session is not None else None
        if p is None:
            return web.json_response({"pacer": None})
        return web.json_response({
            "buffered_seconds": round(p.buffered_audio_seconds, 2),
            "samples_fed": p._audio_samples_fed,
            "samples_played": p._audio_samples_played,
        })

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        old = current_session["session"]
        if old is not None:
            logger.info("新连接到达，顶掉旧会话（释放管线槽位）")
            await old.close()
        # ?alead=毫秒：新回复音频压后量（云时代等 avatar 渲染用，纯语音默认 0，可调）
        try:
            lead = float(request.query.get("alead", "0")) / 1000.0
        except ValueError:
            lead = 0.0
        pacer = None
        if rtc_manager is not None:
            from voxemw.gateway.audio_pacer import AudioPacer
            pacer = AudioPacer(audio_lead=lead)
        session = Session(ws, s2s_url, personas, default_persona,
                          rtc_pacer=pacer, rtc_ice_servers=rtc_ice_servers)
        current_session["session"] = session
        try:
            await session.run()
        finally:
            if current_session["session"] is session:
                current_session["session"] = None
        return ws

    app = web.Application()

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
    app.router.add_post("/rtc/offer", api_rtc_offer)
    app.router.add_get("/rtc/ice", api_rtc_ice)
    app.router.add_post("/rtc/debug", api_rtc_debug)
    app.router.add_get("/rtc/pacer", api_pacer_debug)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", REPO_ROOT / "web")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 编排入口（浏览器 ↔ s2s + RTC 音频轨）")
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

    # 可选 LAN TLS 入口（iPhone/iPad 用：iOS 的 getUserMedia 只在 https 下可用）。
    # 证书由 scripts/make_lan_tls.sh 生成，环境变量缺省/文件不存在则只开 http。
    tls_cert = os.environ.get("VOX_TLS_CERT", "")
    tls_key = os.environ.get("VOX_TLS_KEY", "")
    tls_port = int(os.environ.get("VOX_TLS_PORT", "9443"))
    ssl_ctx = None
    if tls_cert and tls_key and Path(tls_cert).is_file() and Path(tls_key).is_file():
        import ssl

        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(tls_cert, tls_key)

    async def serve() -> None:
        runner = web.AppRunner(create_app(config))
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        logger.info("orchestrator 就绪: http://%s:%d", host, port)
        if ssl_ctx is not None:
            # LAN 入口必须绑 0.0.0.0（http 入口可以只绑回环）
            await web.TCPSite(runner, "0.0.0.0", tls_port, ssl_context=ssl_ctx).start()
            logger.info("LAN TLS 入口就绪: https://0.0.0.0:%d（iPhone 用这个）", tls_port)
        await asyncio.Event().wait()  # 常驻

    asyncio.run(serve())


if __name__ == "__main__":
    main()
