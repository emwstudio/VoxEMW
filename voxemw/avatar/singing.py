"""歌声生成客户端：ACE-Step 1.5 官方 REST 服务（acestep-api，默认 :8001）。

流程：POST /release_task（multipart，可带参考音频）→ 轮询 POST /query_result
（task_id_list，status: 0=排队/生成中 1=成功 2=失败）→ GET {file}
（result 里每项的 file 字段即 /v1/audio?path=... 相对 URL）→ ffmpeg 转
16kHz mono s16 PCM，复用 TTS 下行路径（RTC 音轨 + avatar 口型）。

实时性靠分段伪流式：整首歌拆成 segment_seconds 片段，iter_song_segments
逐段生成逐段 yield，首段到位即开播，播放期间后台生成后续片段；
asyncio 取消即停止后续生成（打断复用对话的 cancel/flush 语义）。

API 细节以 ace-step/ACE-Step-1.5 仓库源码为准（acestep/api/http/*）：
- 参考音频只能 multipart 上传（字段 ref_audio）——路径字段禁绝对路径
- audio_format 默认 mp3，这里直接要 wav（ffmpeg 解码更省）
- 结果里 result 字段是 JSON 字符串，需二次解析
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)

MIN_DURATION = 10   # API 下限
MAX_DURATION = 600  # API 上限


@dataclass
class SongSpec:
    """一首歌的生成参数。prompt = 风格/标签描述，lyrics = 歌词文本。"""

    prompt: str
    lyrics: str
    seconds: int
    vocal_language: str = "zh"


def split_durations(total: int, seg: int, first: int | None = None) -> list[int]:
    """把 total 秒切成段长序列（最后一段是余数），夹在 API 时限内。

    first：首段时长（首段决定开播延迟，可单独调短——首段 10s 比 20s 段
    早 ~5s 开播；后续段恢复 seg 摊薄固定开销）。total/seg/first 各自夹到
    [MIN_DURATION, MAX_DURATION]；余数不足 API 下限时并进上一段。
    """
    total = max(MIN_DURATION, min(MAX_DURATION, int(total)))
    seg = max(MIN_DURATION, min(MAX_DURATION, int(seg)))
    first = max(MIN_DURATION, min(MAX_DURATION, int(first))) if first else seg
    if total <= first:
        return [total]
    out = [first]
    left = total - first
    while left > 0:
        take = min(seg, left)
        if take < MIN_DURATION:
            out[-1] += take
        else:
            out.append(take)
        left -= take
    return out


def build_task_payload(spec: SongSpec, checkpoint: str,
                       audio_format: str = "wav",
                       cover_strength: float | None = None) -> dict:
    """release_task 的请求字段（纯函数，便于单测）。

    thinking：从零创作且歌词直给时关掉省时间（LM 只是写词/推元数据的）；
    歌词留空需要 LM 创作；**cover 翻唱一律开**——官方默认 True，LM 思维链
    负责歌词与源曲旋律的对齐/元数据推理，换词翻唱的质量就靠它
    （质量优先于省那几秒，2026-08-22 按官方 INFERENCE 文档修正）。
    cover_strength 非 None 即 cover 翻唱模式（源音频由调用方 multipart
    上传，字段 src_audio）：旋律照源歌，歌词可换；strength 语义按官方：
    默认 1.0 = 忠实翻唱，0.2 小值是风格迁移（别往小调）。"""
    thinking = True if cover_strength is not None else not spec.lyrics.strip()
    payload = {
        "prompt": spec.prompt,
        "lyrics": spec.lyrics,
        "audio_duration": float(spec.seconds),
        "model": checkpoint,
        "vocal_language": spec.vocal_language,
        "audio_format": audio_format,
        "task_type": "text2music",
        "thinking": thinking,
    }
    if cover_strength is not None:
        payload["task_type"] = "cover"
        payload["audio_cover_strength"] = float(cover_strength)
    return payload


class MusicClient:
    """acestep-api 的薄 async 客户端（aiohttp）。无 GPU 依赖，跑在 orchestrator 进程。"""

    def __init__(self, base_url: str, checkpoint: str = "acestep-v15-turbo",
                 poll_interval: float = 1.0, task_timeout: float = 600.0,
                 audio_format: str = "wav"):
        self.base_url = base_url.rstrip("/")
        self.checkpoint = checkpoint
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout
        self.audio_format = audio_format

    # ── HTTP 原语（单测时 monkeypatch 这三个）──

    async def _post(self, path: str, *, json_body: dict | None = None,
                    data=None) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url + path,
                                    json=json_body, data=data) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _get_bytes(self, path: str) -> bytes:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url + path) as resp:
                resp.raise_for_status()
                return await resp.read()

    # ── 生成 ──

    async def generate(self, spec: SongSpec,
                       ref_audio: tuple[str, bytes] | None = None,
                       src_audio: tuple[str, bytes] | None = None,
                       cover_strength: float | None = None) -> bytes:
        """生成一段歌，返回音频文件字节（格式 = self.audio_format）。

        ref_audio = (文件名, 字节)：人设 ref_wav 等参考音频，multipart 上传
        （API 的路径字段禁绝对路径，跨进程只能传文件体）。
        src_audio + cover_strength：cover 翻唱模式——src 是旋律源（良子唱的
        小段），multipart 字段 src_audio；strength 越小越贴源曲。
        任务失败抛 RuntimeError（带服务端错误信息）；超时抛 TimeoutError。
        """
        payload = build_task_payload(spec, self.checkpoint, self.audio_format,
                                     cover_strength=cover_strength)
        if ref_audio is not None or src_audio is not None:
            import aiohttp

            form = aiohttp.FormData()
            for key, value in payload.items():
                form.add_field(key, str(value))
            if ref_audio is not None:
                form.add_field("ref_audio", ref_audio[1], filename=ref_audio[0])
            if src_audio is not None:
                form.add_field("src_audio", src_audio[1], filename=src_audio[0])
            reply = await self._post("/release_task", data=form)
        else:
            reply = await self._post("/release_task", json_body=payload)
        task_id = (reply.get("data") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"release_task 未返回 task_id: {reply}")

        deadline = asyncio.get_event_loop().time() + self.task_timeout
        while True:
            await asyncio.sleep(self.poll_interval)
            reply = await self._post("/query_result",
                                     json_body={"task_id_list": [task_id]})
            items = reply.get("data") or []
            item = next((i for i in items if i.get("task_id") == task_id), None)
            status = item.get("status") if item else 0
            if status == 1:
                results = json.loads(item.get("result") or "[]")
                if not results or not results[0].get("file"):
                    raise RuntimeError(f"任务成功但无音频结果: {item}")
                return await self._get_bytes(results[0]["file"])
            if status == 2:
                error = ""
                try:
                    results = json.loads(item.get("result") or "[]")
                    error = (results[0] or {}).get("error", "") if results else ""
                except (json.JSONDecodeError, IndexError):
                    pass
                raise RuntimeError(f"歌声生成失败: {error or item.get('progress_text') or item}")
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"歌声生成超时（{self.task_timeout:.0f}s）: {task_id}")

    async def to_pcm16k(self, audio_bytes: bytes) -> bytes:
        """任意音频字节 → 16kHz mono s16le PCM（ffmpeg 管道，不落临时文件）。"""
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(audio_bytes)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 转码失败: {err.decode(errors='replace')[:400]}")
        return out

    async def probe_duration(self, audio_bytes: bytes) -> float:
        """音频字节 → 时长秒（ffprobe 管道；流式 wav 头读不出时解码数采样兜底）。"""
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", "pipe:0",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate(audio_bytes)
        try:
            return float(out.decode().strip())
        except ValueError:
            # 流式写法的 wav（如 ACE-Step 产物）管道里报 N/A：解码数采样
            pcm = await self.to_pcm16k(audio_bytes)
            return len(pcm) / 2 / 16000


async def iter_song_segments(client: MusicClient, spec: SongSpec,
                             segment_seconds: int,
                             ref_audio: tuple[str, bytes] | None = None,
                             first_segment_seconds: int | None = None,
                             src_audio: tuple[str, bytes] | None = None,
                             cover_strength: float | None = None,
                             ) -> AsyncIterator[bytes]:
    """分段伪流式：逐段生成、逐段 yield 16k PCM；消费端被取消即停止后续生成。

    first_segment_seconds：首段单独调短（开播快），后续段用 segment_seconds。
    src_audio（cover 模式）：整首一次出——不分段不切源歌（分段会把旋律切断，
    实测效果差）；歌长先夹到源歌时长（cover 不能比源歌长）。"""
    if src_audio is not None:
        src_secs = await client.probe_duration(src_audio[1])
        spec = replace(spec, seconds=min(spec.seconds, max(MIN_DURATION, int(src_secs))))
        durations = [spec.seconds]  # cover 不分段
        logger.info("cover 翻唱：%ds 整首一次生成（源歌 %ds）", spec.seconds, int(src_secs))
    else:
        durations = split_durations(spec.seconds, segment_seconds, first_segment_seconds)
        logger.info("歌声分段：总 %ds → %s（cover=False）", spec.seconds, durations)
    offset = 0.0
    for seconds in durations:
        audio = await client.generate(replace(spec, seconds=seconds),
                                      ref_audio=ref_audio,
                                      src_audio=src_audio,
                                      cover_strength=cover_strength)
        offset += seconds
        yield await client.to_pcm16k(audio)
