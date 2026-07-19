"""LiveKit Agents STT plugin for Qwen3-ASR (local transformers inference).

Non-streaming STT: meant to be wrapped with ``stt.StreamAdapter`` + a VAD so
utterances are segmented and each segment is transcribed in one batch call.

The transformers model is loaded lazily on first use so this module can be
imported on machines without GPU / torch (e.g. for local development).
"""

from __future__ import annotations

import asyncio
import io

import soundfile as sf
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

# Qwen3-ASR expects 16 kHz mono audio.
TARGET_SAMPLE_RATE = 16000


class Qwen3ASRSTT(stt.STT):
    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-ASR-0.6B-hf",
        language: str = "Chinese",
        device: str = "cuda",
    ) -> None:
        """
        Args:
            model_id: Hugging Face model id or local path.
            language: Language hint passed to the ASR model (code or full name,
                e.g. "zh" / "Chinese"). Skips auto language detection.
            device: Device for the transformers model (e.g. "cuda", "cpu").
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._model_id = model_id
        self._language = language
        self._device = device

        self._processor = None
        self._model = None

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "Qwen3-ASR (local transformers)"

    def _ensure_loaded(self) -> None:
        """Load processor + model on first use (blocking, run off the event loop)."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self._model_id,
            dtype=torch.bfloat16,
            device_map=self._device,
        )

    def warmup(self) -> None:
        """Load the model (blocking; run off the event loop) so the first
        utterance of a session doesn't pay the load time."""
        self._ensure_loaded()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._language
        wav_bytes = _buffer_to_wav_bytes(buffer)
        return await asyncio.to_thread(self._transcribe, wav_bytes, lang)

    def _transcribe(self, wav_bytes: bytes, language: str) -> stt.SpeechEvent:
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        # Decode the WAV bytes ourselves (soundfile) and hand numpy to the
        # processor. transformers 5.x's default audio backend is torchcodec,
        # whose prebuilt binaries lag behind torch releases — bypassing it
        # removes that dependency (and the temp file) entirely.
        audio, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        inputs = self._processor.apply_transcription_request(
            audio=audio,
            language=language,
            sampling_rate=TARGET_SAMPLE_RATE,
        ).to(self._model.device, self._model.dtype)

        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        text = self._processor.decode(
            generated_ids, return_format="transcription_only"
        )[0].strip()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language="zh",
                    text=text,
                    confidence=1.0,
                )
            ],
        )


def _buffer_to_wav_bytes(buffer: utils.AudioBuffer) -> bytes:
    """Merge the AudioBuffer into a single 16 kHz mono WAV byte string."""
    frame = rtc.combine_audio_frames(buffer)

    if frame.sample_rate != TARGET_SAMPLE_RATE or frame.num_channels != 1:
        resampler = rtc.AudioResampler(
            input_rate=frame.sample_rate,
            output_rate=TARGET_SAMPLE_RATE,
            num_channels=1,
            quality=rtc.AudioResamplerQuality.HIGH,
        )
        frames = resampler.push(frame)
        frames.extend(resampler.flush())
        frame = rtc.combine_audio_frames(frames)

    return frame.to_wav_bytes()
