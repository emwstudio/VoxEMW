"""TTS 层：VoxCPM2 多音色合成（Ultimate Cloning prompt cache）。

模型加载写法复自已验证的 extensions/tts_voxcpm_handler.py
（build_prompt_cache 预建 + _generate_with_prompt_cache 流式）。
仅在 GPU 实例上可用；本地开发不 import 本模块的 TTS 类
（clean_for_tts / to_wav_bytes 是纯逻辑，可单测）。
"""

from __future__ import annotations

import io
import logging
import re
import wave
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# 文案/LLM 输出里可能出现的朗读噪音：括号括注（全角半角）、markdown 标记、emoji
_PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")
_MARKDOWN_RE = re.compile(r"[*_#>`~]+")
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff☀-➿\U0001f1e6-\U0001f1ff]", flags=re.UNICODE
)
_WS_RE = re.compile(r"\s+")


def clean_for_tts(text: str) -> str:
    """TTS 前清洗：剥括号括注 / markdown / emoji，折叠空白（换行变空格）。"""
    text = _PAREN_RE.sub("", text or "")
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1,1] 单声道 -> 16bit PCM wav bytes。"""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class TTS:
    """VoxCPM2：每音色一个预建 prompt cache（Ultimate Cloning）。

    输出保留模型原生采样率（VoxCPM2 为 48kHz），不重采样——成品广告音频要音质。

    线程约束：optimize 用 torch.compile mode="reduce-overhead"（cudagraph
    trees），容器挂在首次编译所在线程的 TLS 上，换线程调用编译产物会
    AssertionError(_is_key_in_tls)。加载（含 warmup 编译）和每次合成必须
    在同一个线程——由调用方用单线程池保证（见 voxemw/server.py）。
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        optimize: bool = True,
    ):
        from voxcpm import VoxCPM

        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        logger.info("Loading TTS: %s on %s (optimize=%s)", model_name, device, optimize)
        self.model = VoxCPM.from_pretrained(
            model_name,
            load_denoiser=False,
            optimize=optimize,
            device=device,  # 注意必须传 "cuda"：官方 optimize() 只认这个字符串
        )
        self.sample_rate = int(self.model.tts_model.sample_rate)
        self._caches: dict[str, dict] = {}

    def add_voice(self, name: str, ref_wav: str, ref_text: str) -> None:
        logger.info("Building TTS prompt cache: %s <- %s", name, ref_wav)
        self._caches[name] = self.model.tts_model.build_prompt_cache(
            prompt_text=ref_text,
            prompt_wav_path=ref_wav,
            reference_wav_path=ref_wav,
        )

    @property
    def voices(self) -> list[str]:
        return list(self._caches)

    def stream_chunks(self, text: str, voice: str) -> Iterator[np.ndarray]:
        """流式合成，逐 chunk 产出模型原生采样率 float32（写法对齐 voxcpm core.py）。"""
        text = clean_for_tts(text)
        if not text:
            return
        cache = self._caches.get(voice) or next(iter(self._caches.values()))
        gen = self.model.tts_model._generate_with_prompt_cache(
            target_text=text,
            prompt_cache=cache,
            min_len=2,
            max_len=2000,
            inference_timesteps=self.inference_timesteps,
            cfg_value=self.cfg_value,
            retry_badcase=False,
            streaming=True,
        )
        try:
            for wav, _, _ in gen:
                audio = np.atleast_1d(
                    np.asarray(wav.squeeze(0).cpu().numpy(), dtype=np.float32).squeeze()
                )
                if audio.size:
                    yield audio
        finally:
            gen.close()

    def synthesize(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        """合成完整音频，返回 (float32 单声道, 采样率)。空文本返回空数组。"""
        chunks = list(self.stream_chunks(text, voice))
        if not chunks:
            return np.empty(0, dtype=np.float32), self.sample_rate
        return np.concatenate(chunks), self.sample_rate
