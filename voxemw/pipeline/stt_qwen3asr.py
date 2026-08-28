# STT 积木：Qwen3-ASR-0.6B-hf（transformers 原生，bf16 MPS，自回归逐字生成）。
# 作为自定义 handler 由 voxemw.pipeline.launch 在运行时注册进
# huggingface/speech-to-speech 管线（module_kwargs.stt == "qwen3asr"）。
#
# 选型依据（2026-08-23，M5 16G 实测，五段良子口吻样本 A/B）：
# - 字准：SenseVoice 会在人设黑话上翻车（大胃袋→大卫在梁无人组，CER 9-24%）；
#   0.6B+热词四段满分，真人 ref 段「大胃袋」也能识别对
# - 代价：自回归，2-5s 短句 0.3-1.0s（SenseVoice 是 0.03-0.18s）——用户拍板
#   用每轮 ~0.5s 换准确率
# - 热词走 chat-template：system 塞 "Vocabulary: ..."，assistant 预填
#   "language Chinese<asr_text>"（模型卡写的 prompt= 参数 transformers 5.14
#   不认，会被忽略）

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

# 管线 VAD 输出 16 kHz 单声道，与 Qwen3-ASR 特征提取器原生采样率一致
TARGET_SAMPLE_RATE = 16000


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    """转写后确定性校正（纯函数，便于单测）：人设专名的同音误写直接替换。

    例：本产品语境里用户说的「鸟儿」更可能是「妮儿」——热词偏置是概率性的，
    这层是兜底的确定性修复，零延迟。"""
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text


class Qwen3ASRSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using Qwen3-ASR (transformers -hf).
    """

    def setup(
        self,
        model_name: str = "Qwen/Qwen3-ASR-0.6B-hf",
        device: str = "auto",
        language: str = "Chinese",
        hotwords: str = "",
        max_new_tokens: int = 256,
        corrections: dict[str, str] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.language = language
        # 词表原文（如 "河南妮儿, 恁, 中"）；空串则不注 system，裸跑
        self.hotwords = hotwords.strip()
        self.max_new_tokens = max_new_tokens
        # 转写后确定性校正（同音误写替换表，如 鸟儿→妮儿）
        self.corrections = corrections or {}
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto",
        )
        logger.info(
            "Qwen3-ASR loaded: %s | device=%s dtype=%s | hotwords=%s | corrections=%s",
            model_name, self.model.device, self.model.dtype,
            self.hotwords or "(无)", self.corrections or "(无)",
        )
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_audio = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
        self._transcribe(dummy_audio)
        logger.info(f"{self.__class__.__name__} warmed up")

    def _transcribe(self, audio: np.ndarray) -> str:
        import torch

        if self.hotwords:
            conv = [[
                {"role": "system", "content": [
                    {"type": "text", "text": f"Vocabulary: {self.hotwords}"}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio,
                     "sampling_rate": TARGET_SAMPLE_RATE}]},
            ]]
            # 手工拼 assistant 预填：transformers 5.16 的 continue_final_message
            # 校验与该模板不兼容（渲染结果不含末条消息 → ValueError）。
            # add_generation_prompt 出的前缀以 <|im_start|>assistant\n 结尾，
            # 直接拼预填串等价于 continue_final_message 的效果。
            prompt = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True,
            )[0]
            prompt += f"language {self.language}<asr_text>"
            inputs = self.processor(
                text=prompt, audio=[audio],
                sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt",
            )
        else:
            inputs = self.processor.apply_transcription_request(
                audio=audio, language=self.language,
            )
        inputs = inputs.to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(gen, return_format="transcription_only")[0].strip()
        return apply_corrections(text, self.corrections)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering qwen3asr...")

        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        pred_text = self._transcribe(audio)

        logger.debug("finished qwen3asr inference")

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
