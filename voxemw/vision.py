"""Vision：用户摄像头截帧 → Kimi K3 多模态 API → 外貌客观描述。

使用场景：语音链路里的 LLM（DeepSeek）是纯文本，看不见用户。打分环节由 persona
说出暗号短语（vision.trigger，如「让我好好看看你」），前端检测暗号后截一帧摄像头
画面经 orchestrator 转发到本模块；拿到的客观描述再注入 s2s 对话（conversation.item.create
+ response.create），让 persona 按自己的风格锐评。

纯逻辑函数（build_*）不 import aiohttp，可在 macOS 开发机直接单测。
"""

from __future__ import annotations

import base64
import logging
import os

logger = logging.getLogger(__name__)

# 浏览器上行二进制帧首字节：用户摄像头截帧（0x01 是下行的数字人 JPEG 帧）
FRAME_TYPE_USER_JPEG = 0x02

DESCRIBE_SYSTEM = (
    "你是视频通话里的观察员。客观描述画面正中这个人的外表：年龄感、五官轮廓、"
    "发型、穿着、神态表情、镜头感。100 字以内，只描述事实，不评价、不打分、不猜测身份。"
)


def vision_config(config: dict) -> dict | None:
    """从全局配置取 vision 段并补齐 api_key；未启用/缺 key 返回 None。"""
    cfg = (config or {}).get("vision")
    if not cfg or not cfg.get("enabled", True):
        return None
    api_key = os.environ.get(cfg.get("api_key_env", "KIMI_API_KEY"), "")
    if not api_key:
        logger.warning("vision 已启用但环境变量 %s 未设置，截帧功能关闭",
                       cfg.get("api_key_env", "KIMI_API_KEY"))
        return None
    merged = dict(cfg)
    merged["api_key"] = api_key
    return merged


def build_describe_request(cfg: dict, jpeg_bytes: bytes) -> dict:
    """构造 OpenAI 兼容 chat-completions 请求体（Kimi 视觉只收 base64，不收公网 URL）。"""
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()
    return {
        "model": cfg.get("model_name", "kimi-k3"),
        "max_tokens": int(cfg.get("max_tokens", 200)),
        "messages": [
            {"role": "system", "content": DESCRIBE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "描述一下画面里这个人。"},
                ],
            },
        ],
    }


def build_stall_messages() -> tuple[dict, dict]:
    """截帧已收到、Kimi 描述还没回来：注入「垫场」指令让 persona 自由发挥，
    填上等待空白（先垫场 response，描述回来后再注入 build_inject_messages 打分）。"""
    item_create = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    "（系统旁白：你正凑近屏幕仔细看镜头画面，看得仔细但还没看完。"
                    "用一两句口语垫个场——让对方别动、保持姿势，或者调侃两句。"
                    "先别打分，看完再说。）"
                ),
            }],
        },
    }
    return item_create, {"type": "response.create"}


def build_inject_messages(description: str) -> tuple[dict, dict]:
    """把外貌描述注入 s2s 对话的两个 Realtime 事件（先加用户消息，再触发回复）。
    要求 persona 锐评时先描绘外表细节（证明「真看见了」），再给分。"""
    item_create = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    f"（镜头画面：{description}）"
                    "你正看着我了。按你的风格给我打分，十分制：先把你看到的我的外表"
                    "描绘一遍（五官、发型、穿着、神态，要具体），再给出分数和锐评。"
                ),
            }],
        },
    }
    response_create = {"type": "response.create"}
    return item_create, response_create


async def describe(cfg: dict, jpeg_bytes: bytes) -> str | None:
    """调 Kimi K3 描述截帧。失败（网络/鉴权/格式）返回 None，由调用方静默降级。"""
    import aiohttp

    url = cfg.get("base_url", "https://api.moonshot.cn/v1").rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    payload = build_describe_request(cfg, jpeg_bytes)
    # kimi-k3 是 thinking 模型，看图推理常要 20-40s，超时给足；失败重试一次
    for attempt in (1, 2):
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning("vision API 返回 %s: %s", resp.status,
                                       (await resp.text())[:300])
                        return None  # 4xx/5xx 重试无意义（鉴权/限额/参数问题）
                    data = await resp.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if text:
                return text
            logger.warning("vision 返回空描述（第 %d 次）", attempt)
        except Exception as e:
            logger.warning("vision 描述失败（第 %d 次）: %r", attempt, e)
    return None
