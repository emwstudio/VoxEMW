"""语音层：STT（Qwen3-ASR）/ TTS（VoxCPM2 双音色）/ VAD（silero 切句）。

模型加载写法分别复自已验证的两处：
- extensions/stt_qwen3asr_handler.py（apply_transcription_request，绕开 torchcodec）
- extensions/tts_voxcpm_handler.py（build_prompt_cache 预建 + _generate_with_prompt_cache 流式）
仅在 GPU 实例上可用；本地开发不 import 本模块。
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

PIPELINE_SR = 16000


class STT:
    """Qwen3-ASR 转写（16kHz 单声道 float32 -> 文本）。"""

    def __init__(self, model_name: str, device: str, torch_dtype: str, language: str = "Chinese"):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.language = language
        logger.info("Loading STT: %s on %s", model_name, device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=getattr(torch, torch_dtype),
            device_map=device,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor.apply_transcription_request(
            audio=audio,
            language=self.language,
            sampling_rate=PIPELINE_SR,
        ).to(self.model.device, self.model.dtype)
        output_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated_ids, return_format="transcription_only")[0].strip()


class TTS:
    """VoxCPM2：每音色一个预建 prompt cache（Ultimate Cloning），流式输出 16kHz float32。"""

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
        self._model_sr = int(self.model.tts_model.sample_rate)
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
        """流式合成，逐 chunk 产出 16kHz float32（写法对齐 voxcpm core.py）。"""
        import re as _re

        from scipy.signal import resample_poly

        text = _re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
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
                if audio.size == 0:
                    continue
                if self._model_sr != PIPELINE_SR:
                    # 48k -> 16k = 1/3
                    audio = resample_poly(audio, up=PIPELINE_SR, down=self._model_sr)
                yield audio.astype(np.float32)
        finally:
            gen.close()


class VADSegmenter:
    """silero VAD 切句：喂 16kHz int16 PCM 字节流，吐出完整语音段（float32）。"""

    FRAME = 512  # 32ms @16kHz，silero 要求

    def __init__(self, min_silence_ms: int = 600, speech_pad_ms: int = 200):
        import torch

        self._torch = torch
        model, utils = torch.hub.load(
            # 与管线 patch 同款：指定 :master 避免查默认分支卡死，缓存已预置
            "snakers4/silero-vad:master",
            "silero_vad",
            trust_repo=True,
            skip_validation=True,
        )
        vad_iterator_cls = utils[3]
        self._vad = vad_iterator_cls(
            model,
            sampling_rate=PIPELINE_SR,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._speaking = False
        self._buf = np.empty(0, dtype=np.float32)
        self._leftover = np.empty(0, dtype=np.float32)

    def feed(self, pcm_bytes: bytes) -> Optional[np.ndarray]:
        """喂入 PCM 字节；检出完整语音段时返回 float32，否则 None。"""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = np.concatenate([self._leftover, audio])
        n_frames = len(audio) // self.FRAME
        self._leftover = audio[n_frames * self.FRAME :]

        utterance: Optional[np.ndarray] = None
        for i in range(n_frames):
            frame = audio[i * self.FRAME : (i + 1) * self.FRAME]
            event = self._vad(self._torch.from_numpy(frame), return_seconds=False)
            if event and "start" in event:
                self._speaking = True
                self._buf = frame.copy()
                continue
            if self._speaking:
                self._buf = np.concatenate([self._buf, frame])
                if event and "end" in event:
                    utterance = self._buf
                    self._speaking = False
                    self._buf = np.empty(0, dtype=np.float32)
        return utterance

    def reset(self) -> None:
        self._vad.reset_states()
        self._speaking = False
        self._buf = np.empty(0, dtype=np.float32)
        self._leftover = np.empty(0, dtype=np.float32)
