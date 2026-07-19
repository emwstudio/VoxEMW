"""LiveKit Agents TTS plugin for VoxCPM2 (local inference, voice cloning + streaming).

Uses ``voxcpm.VoxCPM.generate_streaming`` to stream 48 kHz PCM chunks produced
by the local model into LiveKit's chunked-synthesis pipeline.

The voxcpm model is loaded lazily on first synthesis so this module can be
imported on machines without GPU / torch (e.g. for local development).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)

if __package__:
    from .audio_utils import _trim_unstable_head, float_to_int16_bytes
    from .text_utils import DEFAULT_MAX_CHARS, split_text_for_tts
else:
    from audio_utils import _trim_unstable_head, float_to_int16_bytes
    from text_utils import DEFAULT_MAX_CHARS, split_text_for_tts

logger = logging.getLogger(__name__)

# VoxCPM2 outputs 48 kHz mono audio (AudioVAE V2 with built-in super-resolution).
VOXCPM_SAMPLE_RATE = 48000
NUM_CHANNELS = 1


class VoxCPM2TTS(tts.TTS):
    def __init__(
        self,
        *,
        model_id: str = "openbmb/VoxCPM2",
        prompt_wav_path: str,
        prompt_text: str,
        reference_wav_path: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        device: str = "cuda",
        max_chars: int = DEFAULT_MAX_CHARS,
        sample_rate: int = VOXCPM_SAMPLE_RATE,
    ) -> None:
        """
        Args:
            model_id: Hugging Face model id or local path.
            prompt_wav_path: Reference audio for cloning (VoxCPM "prompt").
            prompt_text: Exact transcript of the prompt audio.
            reference_wav_path: Reference audio for timbre cloning. Defaults to
                ``prompt_wav_path`` — passing the same clip to both plus the
                transcript is VoxCPM2's "Ultimate Cloning" mode.
            cfg_value: CFG scale for the diffusion decoder.
            inference_timesteps: Number of diffusion steps per audio chunk.
            device: Device for the model (e.g. "cuda", "cpu").
            max_chars: Long texts are split into <= max_chars pieces on
                punctuation and synthesized piece by piece.
            sample_rate: Expected model output sample rate (48 kHz for VoxCPM2);
                verified against the model after loading.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._model_id = model_id
        self._prompt_wav_path = prompt_wav_path
        self._prompt_text = prompt_text
        self._reference_wav_path = reference_wav_path or prompt_wav_path
        self._cfg_value = cfg_value
        self._inference_timesteps = inference_timesteps
        self._device = device
        self._max_chars = max_chars

        self._model: Any = None

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "VoxCPM2 (local)"

    def _ensure_loaded(self) -> None:
        """Load the voxcpm model on first use (blocking, run off the event loop)."""
        if self._model is not None:
            return

        from voxcpm import VoxCPM

        self._model = VoxCPM.from_pretrained(self._model_id, load_denoiser=False)

        model_sr = self._model.tts_model.sample_rate
        if model_sr != self.sample_rate:
            logger.warning(
                "VoxCPM2 sample rate mismatch: model=%s, tts configured=%s",
                model_sr,
                self.sample_rate,
            )

    def warmup(self) -> None:
        """Load the model and run one throwaway synthesis (blocking; run off
        the event loop). The first ``generate_streaming`` call after load pays
        ~10 s of CUDA/kernel warmup — pay it here instead of mid-conversation.
        """
        self._ensure_loaded()
        for _ in self._model.generate_streaming(
            text="味真足。",
            prompt_wav_path=self._prompt_wav_path,
            prompt_text=self._prompt_text,
            reference_wav_path=self._reference_wav_path,
            cfg_value=self._cfg_value,
            inference_timesteps=self._inference_timesteps,
        ):
            pass

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: VoxCPM2TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: VoxCPM2TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import numpy as np

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()
        # Set when the consumer exits early (interruption/cancellation), so the
        # producer thread stops at the next chunk boundary instead of running
        # to completion and racing the next synthesis on the same GPU model.
        stop = threading.Event()

        def _produce() -> None:
            """Iterate the blocking generator; bridge chunks into the event loop."""
            try:
                self._tts._ensure_loaded()
                model = self._tts._model
                first = True
                head: list = []
                head_len = 0
                for text_chunk in split_text_for_tts(
                    self._input_text, max_chars=self._tts._max_chars
                ):
                    if stop.is_set():
                        break
                    for audio_chunk in model.generate_streaming(
                        text=text_chunk,
                        prompt_wav_path=self._tts._prompt_wav_path,
                        prompt_text=self._tts._prompt_text,
                        reference_wav_path=self._tts._reference_wav_path,
                        cfg_value=self._tts._cfg_value,
                        inference_timesteps=self._tts._inference_timesteps,
                    ):
                        if stop.is_set():
                            break
                        chunk = np.asarray(audio_chunk, dtype=np.float32)
                        if first:
                            # VoxCPM2 streams ~160 ms of low-energy unstable
                            # murmur before the voice starts (the "tremor" on
                            # the first syllables). Buffer the head of the
                            # utterance, then skip to the onset of real speech.
                            head.append(chunk)
                            head_len += chunk.size
                            if head_len < int(0.4 * self._tts.sample_rate) and len(head) < 4:
                                continue
                            chunk = _trim_unstable_head(
                                np.concatenate(head), self._tts.sample_rate
                            )
                            first = False
                            if chunk.size == 0:
                                continue
                        pcm = float_to_int16_bytes(chunk)
                        if pcm:
                            loop.call_soon_threadsafe(queue.put_nowait, pcm)
            except Exception as e:  # propagate to the async consumer
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        producer = loop.run_in_executor(None, _produce)
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise APIConnectionError(f"VoxCPM2 synthesis failed: {item}") from item
                output_emitter.push(item)

            output_emitter.flush()
        finally:
            stop.set()
            # Make sure the producer thread has exited before returning.
            await producer
