# STT 积木：Qwen3-ASR（transformers 本地推理，架构 qwen3_asr）。
# 作为自定义 handler 由 voxemw.pipeline.launch 在运行时注册进
# huggingface/speech-to-speech 管线（module_kwargs.stt == "qwen3asr"）。
# 模型加载/推理写法继承自 v0.3 线上跑通的 extensions/stt_qwen3asr_handler.py。

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

# Qwen3-ASR 要求 16 kHz 单声道输入，与管线 VAD 输出采样率一致
TARGET_SAMPLE_RATE = 16000


class Qwen3ASRSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using Qwen3-ASR via transformers.
    """

    def setup(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B-hf",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        language: Optional[str] = "Chinese",
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        # 语言提示（如 "Chinese" / "English"）；跳过自动语种检测
        self.language = language
        self.gen_kwargs = gen_kwargs

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            device_map=device,
        )
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_audio = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
        self._transcribe(dummy_audio)
        logger.info(f"{self.__class__.__name__} warmed up")

    def _transcribe(self, audio: np.ndarray) -> str:
        # 直接给 processor 喂 numpy，绕开 torchcodec 音频后端
        #（其预编译二进制常滞后于 torch 版本）
        inputs = self.processor.apply_transcription_request(
            audio=audio,
            language=self.language,
            sampling_rate=TARGET_SAMPLE_RATE,
        ).to(self.model.device, self.model.dtype)

        output_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated_ids, return_format="transcription_only")[0].strip()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering qwen3-asr...")

        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        pred_text = self._transcribe(audio)

        logger.debug("finished qwen3-asr inference")

        if getattr(vad_audio, "mode", None) == "progressive":
            # 说话过程中的中间块（enable_live_transcription）：只作实时预览，
            # 由 TranscriptionNotifier 消费、不进 LLM；绝不能当最终转写 yield，
            # 否则整轮会被前几百毫秒的片段提前定稿（完整音频反被当 stale 丢弃）
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        console.print(f"[yellow]USER: {pred_text}")

        yield Transcription(
            text=pred_text,
            language_code=None,  # qwen3_asr 不返回语种代码；语言由启动参数固定
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
