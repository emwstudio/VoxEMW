# STT 积木：SenseVoiceSmall（FunASR，非自回归，整段一次前向出全文）。
# 作为自定义 handler 由 voxemw.pipeline.launch 在运行时注册进
# huggingface/speech-to-speech 管线（module_kwargs.stt == "sensevoice"）。
#
# 选型依据（2026-08-02，4090D 实测）：4s 中文语音 Qwen3-ASR-1.7B 转写 0.66s，
# SenseVoiceSmall ~0.1s（GPU 170x 实时，官方中文 CER 7.81% 优于 Whisper-large-v3），
# 且无需流式 ASR 的复杂度——说完即出字，等效流式。
# 输出带 FunASR 元标签（<|zh|><|HAPPY|><|Speech|> 等），文本统一走
# rich_transcription_postprocess 剥掉。

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

# SenseVoiceSmall 要求 16 kHz 单声道输入，与管线 VAD 输出采样率一致
TARGET_SAMPLE_RATE = 16000


class SenseVoiceSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using SenseVoiceSmall via FunASR.
    """

    def setup(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        language: str = "zh",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from funasr import AutoModel

        self.language = language
        self.gen_kwargs = gen_kwargs or {}
        self.model = AutoModel(model=model_name, device=device, disable_update=True)
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_audio = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
        self._transcribe(dummy_audio)
        logger.info(f"{self.__class__.__name__} warmed up")

    def _transcribe(self, audio: np.ndarray) -> str:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        res = self.model.generate(
            input=audio,
            cache={},
            language=self.language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            **self.gen_kwargs,
        )
        if not res:
            return ""
        raw = res[0]["text"]
        # 剥掉 <|zh|><|HAPPY|><|Speech|> 等元标签，只留文本
        return rich_transcription_postprocess(raw).strip()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering sensevoice...")

        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        pred_text = self._transcribe(audio)

        logger.debug("finished sensevoice inference")

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
            language_code=None,  # 语言由启动参数固定，不返回语种代码
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
