"""视觉模块：摄像头抓帧 → MiniCPM-V-4.6（llama-server）→ 场景描述。

形态：llama-server 作为视觉边车（OpenAI 兼容接口），本模块负责
抓帧（Swift 采集器，零依赖）和描述调用。失败一律静默返回 None——
视觉是增强，挂了不能拖累对话主链路。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_BIN = REPO_ROOT / "scripts" / "bin" / "camera_capture"

# 「让妮儿看看」触发词（宁缺毋滥，别把「你看/我觉得」误触发）
TRIGGER_PHRASES = (
    "看看", "看一下", "看一看", "瞅瞅", "瞧瞧",
    "你看这", "你瞅这", "你瞧这", "你看我", "你瞅我", "你瞧我",
    "这是啥", "那是啥", "这是什么", "那是什么",
    "看得到", "看得见", "看见没", "认识这", "帮我看", "给我看看",
)


def is_vision_trigger(text: str) -> bool:
    """用户这句话是不是在让助手「看」。"""
    return any(p in text for p in TRIGGER_PHRASES)


class VisionService:
    """llama-server 视觉边车客户端 + 摄像头采集。"""

    def __init__(self, base_url: str = "http://127.0.0.1:18099",
                 capture_bin: Path = CAPTURE_BIN,
                 prompt: str = "用三句以内的中文口语描述这张图片的主要内容（人物、物体、场景），"
                                "只描述看到的，不评价") -> None:
        self.base_url = base_url.rstrip("/")
        self.capture_bin = capture_bin
        self.prompt = prompt
        self._client = None

    def _http(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=30,
                # 本机地址绕过系统代理（系统代理会把 localhost 劫走 502）
                trust_env=False)
        return self._client

    async def available(self) -> bool:
        try:
            r = await self._http().get("/health")
            return r.status_code == 200
        except Exception:
            return False

    def _grab_frame(self, out_path: str) -> bool:
        """同步抓一帧（Swift 采集器）。在 executor 里跑，别阻塞 loop。"""
        if not self.capture_bin.exists():
            logger.warning("视觉：采集器不存在 %s", self.capture_bin)
            return False
        try:
            r = subprocess.run([str(self.capture_bin), out_path],
                               capture_output=True, timeout=15)
            if r.returncode != 0:
                logger.warning("视觉：拍照失败 %s", r.stderr.decode(errors="ignore")[:200])
                return False
            return Path(out_path).exists()
        except Exception as e:
            logger.warning("视觉：拍照异常 %r", e)
            return False

    async def describe(self, image_path: str) -> str | None:
        """一张图片 → 中文场景描述。失败返回 None。"""
        try:
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            r = await self._http().post("/v1/chat/completions", json={
                "model": "MiniCPM-V-4.6",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": self.prompt},
                ]}],
                "max_tokens": 120,
            })
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            return text or None
        except Exception as e:
            logger.warning("视觉：描述失败 %r", e)
            return None

    async def look(self) -> str | None:
        """抓一帧 + 描述，一条龙。失败返回 None。"""
        frame = "/tmp/vox_vision_frame.jpg"
        ok = await asyncio.get_running_loop().run_in_executor(None, self._grab_frame, frame)
        if not ok:
            return None
        return await self.describe(frame)
