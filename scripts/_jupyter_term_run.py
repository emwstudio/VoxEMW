"""一次性脚本：经 JupyterLab 终端 websocket 在 AutoDL 实例上执行命令并流式回显。

用法: python _jupyter_term_run.py "<命令>" <超时秒>
终端会回放滚动缓冲，所以用随机起始/结束标记：看到起始标记前的输出一律忽略，
看到结束标记退出 0；起始标记后出现 ERROR/Traceback 退出 1；超时退出 2。
"""
import asyncio
import json
import sys
import uuid

import websockets

WS = (
    "wss://a1092349-784903c858d1.bjb2.seetacloud.com:8443"
    "/jupyter/terminals/websocket/1"
    "?token=jupyter-autodl-pro-784903c858d1-c346ea0723d76490592652e7a9bbdc41437891caccaa14a0f8b3ca63f40d730de"
)
CMD = sys.argv[1]
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 600
BAD_MARKERS = ["ERROR", "Traceback", "Segmentation fault", "core dumped"]

MARK = uuid.uuid4().hex[:8]
START = f"S_{MARK}"
END = f"E_{MARK}"
WRAPPED = f"echo {START}; {CMD}; echo {END}\n"


async def _pump(ws) -> int:
    started = False
    buf: list[str] = []
    async for raw in ws:
        msg = json.loads(raw)
        if msg[0] != "stdout":
            continue
        text = msg[1]
        if not started:
            # 终端回放的滚动缓冲里可能有旧输出，见到本次随机起始标记才算数
            if START in text:
                started = True
            continue
        buf.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()
        joined = "".join(buf[-30:])
        if END in joined:
            return 0
        if any(m in joined for m in BAD_MARKERS):
            return 1
    print("\n[driver] websocket 关闭，未见到结束标记")
    return 1


async def main() -> int:
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(json.dumps(["stdin", WRAPPED]))
        try:
            return await asyncio.wait_for(_pump(ws), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            print(f"\n[driver] 超时（{TIMEOUT}s）")
            return 2


sys.exit(asyncio.run(main()))
