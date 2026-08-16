"""VoxEMW 局域网 HTTPS 反代：让同 LAN 的 iPad/手机安全上下文访问（麦克风可用）。

Mac 上跑：.venv/bin/python scripts/lan_https/proxy.py
  https://<Mac局域网IP>:8443  --(TLS)-->  http://127.0.0.1:8000(SSH 隧道 → AutoDL)
iPad 需先安装并信任 scripts/lan_https/ca.crt（AirDrop 后 设置→通用→描述文件,
再在 设置→通用→关于本机→证书信任设置 里开启完全信任）。

WebRTC 音画（2026-08 起下行走 RTC）：/rtc/ice 响应里的 turn:localhost 会被改写成
Mac 局域网地址，同时本脚本在 :3479 起 TCP 转发 → 127.0.0.1:3478（SSH 隧道里的
coturn）。iPad 上 RTC 走 turn:<MacIP>:3479?transport=tcp。
"""

import asyncio
import json
import socket
import ssl
from pathlib import Path

import aiohttp
from aiohttp import web

HERE = Path(__file__).resolve().parent
BACKEND = "http://127.0.0.1:8000"
LISTEN_PORT = 8443
TURN_LISTEN_PORT = 3479      # LAN 侧 TURN 监听（转发到隧道的 3478）
TURN_BACKEND = 3478


def _lan_ip() -> str:
    """取 Mac 的局域网 IP（UDP 外联测源地址，不真正发包）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.255.255", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


LAN_IP = _lan_ip()


async def proxy_http(request: web.Request) -> web.StreamResponse:
    """普通 HTTP 请求透传（静态页/API）；/rtc/ice 的 TURN 地址改写为 Mac LAN。"""
    url = BACKEND + str(request.rel_url)
    async with aiohttp.ClientSession() as sess:
        async with sess.request(
            request.method, url, params=request.query,
            data=await request.read(), headers={
                k: v for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length", "connection")
            },
        ) as resp:
            body = await resp.read()
            if request.path == "/rtc/ice":
                try:
                    data = json.loads(body)
                    for srv in data.get("ice_servers", []):
                        srv["urls"] = [
                            u.replace("localhost:3478", f"{LAN_IP}:{TURN_LISTEN_PORT}")
                            for u in srv.get("urls", [])
                        ]
                    body = json.dumps(data).encode()
                except (json.JSONDecodeError, KeyError):
                    pass
            return web.Response(
                status=resp.status, body=body,
                headers={k: v for k, v in resp.headers.items()
                         if k.lower() not in ("content-length", "transfer-encoding", "connection")},
            )


async def proxy_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket 双向透传（/ws 会话：文本事件 + 二进制视频帧）。"""
    client_ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await client_ws.prepare(request)
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(BACKEND + "/ws", max_msg_size=16 * 1024 * 1024) as backend_ws:
            async def client_to_backend():
                async for msg in client_ws:
                    if msg.type == web.WSMsgType.TEXT:
                        await backend_ws.send_str(msg.data)
                    elif msg.type == web.WSMsgType.BINARY:
                        await backend_ws.send_bytes(msg.data)

            async def backend_to_client():
                async for msg in backend_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await client_ws.send_str(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await client_ws.send_bytes(msg.data)

            import asyncio
            c2b = asyncio.create_task(client_to_backend())
            b2c = asyncio.create_task(backend_to_client())
            done, pending = await asyncio.wait({c2b, b2c}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
    return client_ws


WSTEST_HTML = """<!doctype html><meta charset=utf-8><title>ws test</title>
<body style="font:16px monospace;padding:20px">
<div id=o>connecting...</div>
<script>
const o = document.getElementById('o');
try {
  const ws = new WebSocket('wss://' + location.host + '/ws');
  ws.onopen = () => { o.textContent = 'OPEN ✓'; };
  ws.onmessage = (m) => { o.textContent = 'MSG: ' + m.data.slice(0, 120); };
  ws.onerror = (e) => { o.textContent += ' | ERROR'; };
  ws.onclose = (e) => { o.textContent += ' | CLOSE code=' + e.code + ' reason=' + e.reason; };
} catch (e) {
  o.textContent = 'THROW: ' + e.message;
}
</script>
"""


async def wstest(_request: web.Request) -> web.Response:
    return web.Response(text=WSTEST_HTML, content_type="text/html")


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


async def _turn_forward(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
    """TCP 转发：iPad → :3479 → 127.0.0.1:3478（隧道里的 coturn）。"""
    try:
        back_r, back_w = await asyncio.open_connection("127.0.0.1", TURN_BACKEND)
    except OSError:
        client_w.close()
        return
    await asyncio.gather(_pipe(client_r, back_w), _pipe(back_r, client_w))


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)  # aiohttp.access 默认 WARNING 级,不开看不到请求日志
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(HERE / "server.crt", HERE / "server.key")
    # iPadOS 14 的 WebSocket 协议栈在 TLS 1.3 下握手必败(老 WebKit bug,
    # 页面 HTTPS 加载正常但 wss 1006);限制 TLS 1.2 兼容老 iPad
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2

    async def _start_turn_forward(_app):
        await asyncio.start_server(_turn_forward, "0.0.0.0", TURN_LISTEN_PORT)

    app = web.Application()
    app.on_startup.append(_start_turn_forward)
    app.router.add_get("/ws", proxy_ws)
    app.router.add_get("/wstest", wstest)
    app.router.add_route("*", "/{tail:.*}", proxy_http)
    print(f"LAN HTTPS 反代就绪: https://{LAN_IP}:{LISTEN_PORT} → {BACKEND}")
    print(f"TURN 转发: :{TURN_LISTEN_PORT} → 127.0.0.1:{TURN_BACKEND}")
    web.run_app(app, host="0.0.0.0", port=LISTEN_PORT, ssl_context=ctx,
                print=None, access_log_format="%r %s %{User-Agent}i")


if __name__ == "__main__":
    main()
