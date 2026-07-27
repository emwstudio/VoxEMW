"""DeepSeek chat-completions 最小客户端（stdlib urllib，无第三方依赖）。

同步阻塞；asyncio 服务里用 asyncio.to_thread 包着调。
DeepSeek 关思考走官方 thinking.type=disabled。
"""

from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


def chat_complete(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 300,
    timeout: float = 30.0,
) -> str:
    """调 /chat/completions 非流式，返回 content 文本。失败抛异常。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # DeepSeek 官方关思考参数（chat_template_kwargs 它不认）
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]
