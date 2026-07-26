# TTS handler：k2-fsa/OmniVoice（非流式 API，按句合成）。
# 供 huggingface/speech-to-speech 管线使用（--tts omnivoice）。
# 本文件与 patches/register-handlers.patch 在 vendor/speech-to-speech 内创建的
# src/speech_to_speech/TTS/omnivoice_handler.py 内容一致（extensions/ 为人类可读副本，
# patch 是唯一事实源，改动需两边同步）。
#
# 要点：
# - OmniVoice.generate() 不是流式 API，按上游其他非流式 TTS 的模式：
#   上游 LMOutputProcessor 已按句切分，这里合成一句、输出一句的音频 chunk。
# - OmniVoice 输出 24kHz 单声道 float，重采样到管线要求的 16kHz
#   （重采样方式照抄上游 qwen3/pocket TTS handler 的 resample_poly 写法）。
# - 启动时预编码 voice-clone prompt：默认音色（ref_audio/ref_text）+ voices
#   （JSON）里的每个命名音色；.pt 统一缓存到可配置目录，二次启动直接
#   VoiceClonePrompt.load()。
# - 多音色热切换：realtime 客户端 session.update 的 session.audio.output.voice
#   按名字选 voices 里预编码的 prompt（读法仿上游 qwen3_tts_handler 的
#   _apply_session_voice_override，只取 session 级），未匹配回退默认音色。

from __future__ import annotations

import hashlib
import json
import logging
from math import gcd
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

MODEL_SAMPLE_RATE = 24000  # OmniVoice 输出采样率（官方文档：24kHz 单声道）
PIPELINE_SR = 16000  # 管线音频采样率（与上游其他 TTS handler 一致）

DEFAULT_VOICE = "default"  # ref_audio/ref_text 启动参数对应的音色名（voices 未命中时的回退）


class OmniVoiceTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech using k2-fsa/OmniVoice (non-streaming, per-sentence).
    """

    def setup(
        self,
        should_listen: Event,
        model_name: str = "k2-fsa/OmniVoice",
        device: str = "cuda:0",
        dtype: str = "float16",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        voices: Optional[str] = None,
        prompt_cache_dir: str = "./omnivoice_prompts",
        sample_rate: int = PIPELINE_SR,
        blocksize: int = 512,
        language: Optional[str] = None,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        import torch
        from omnivoice import OmniVoice

        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.gen_kwargs = gen_kwargs or {}
        # 语言提示（如 "Chinese"）：官方说明指定语言比语种无关模式质量略好
        self.language = language

        logger.info("Loading OmniVoice model: %s on %s", model_name, device)
        self.model = OmniVoice.from_pretrained(
            model_name,
            device_map=device,
            dtype=getattr(torch, dtype),
        )

        # 预编码 voice-clone prompt（带磁盘缓存，避免每次启动重算/重跑 Whisper ASR）：
        # 默认音色（ref_audio/ref_text）+ voices JSON 里的每个命名音色
        self._model_name = model_name
        self._prompt_cache_dir = Path(prompt_cache_dir).expanduser()
        self._prompt_cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice_prompts: dict[str, Any] = {}
        self.default_voice: Optional[str] = None
        if ref_audio:
            self.voice_prompts[DEFAULT_VOICE] = self._load_or_create_prompt(ref_audio, ref_text)
            self.default_voice = DEFAULT_VOICE
        if voices:
            try:
                voice_map = json.loads(voices)
            except json.JSONDecodeError as e:
                raise ValueError(f"omnivoice_tts_voices is not valid JSON: {e}") from e
            if not isinstance(voice_map, dict):
                raise ValueError(
                    "omnivoice_tts_voices must be a JSON object: {name: {ref_audio, ref_text}}"
                )
            for name, spec in voice_map.items():
                v_ref_audio = (spec or {}).get("ref_audio")
                if not v_ref_audio:
                    logger.warning("Skipping OmniVoice voice %r: missing ref_audio", name)
                    continue
                self.voice_prompts[name] = self._load_or_create_prompt(
                    v_ref_audio, (spec or {}).get("ref_text")
                )
        self._active_voice = self.default_voice

        # 24kHz -> 16kHz: 16000/24000 = 2/3（与上游 pocket/qwen3 handler 相同的有理重采样）
        g = gcd(self.sample_rate, MODEL_SAMPLE_RATE)
        self._resample_up = self.sample_rate // g
        self._resample_down = MODEL_SAMPLE_RATE // g
        self._needs_resampling = self.sample_rate != MODEL_SAMPLE_RATE

        self.warmup()

    def _load_or_create_prompt(self, ref_audio: str, ref_text: Optional[str]):
        """编码单个音色的 VoiceClonePrompt；sha256(model|ref_audio路径|ref_text)
        命中磁盘缓存则直接加载，否则编码后存缓存。"""
        from omnivoice import VoiceClonePrompt

        key = hashlib.sha256(
            f"{self._model_name}|{Path(ref_audio).expanduser().resolve()}|{ref_text or ''}".encode()
        ).hexdigest()[:16]
        cache_path = self._prompt_cache_dir / f"{Path(ref_audio).stem}-{key}.pt"
        if cache_path.exists():
            logger.info("Loading cached OmniVoice voice-clone prompt: %s", cache_path)
            return VoiceClonePrompt.load(str(cache_path))
        logger.info("Creating OmniVoice voice-clone prompt from %s", ref_audio)
        prompt_kwargs: dict[str, Any] = {"ref_audio": ref_audio}
        if ref_text:  # ref_text 可省略（OmniVoice 自动 Whisper ASR），有就给
            prompt_kwargs["ref_text"] = ref_text
        prompt = self.model.create_voice_clone_prompt(**prompt_kwargs)
        prompt.save(str(cache_path))
        logger.info("Saved OmniVoice voice-clone prompt cache: %s", cache_path)
        return prompt

    def _select_prompt(self, tts_input: TTSIn):
        """按 session.audio.output.voice 选音色 prompt；缺省/未命中回退默认音色。

        读法仿上游 qwen3_tts_handler._apply_session_voice_override（只取 session 级，
        不管 response 级覆盖）。
        """
        voice: Optional[str] = None
        runtime_config = getattr(tts_input, "runtime_config", None)
        if runtime_config is not None:
            session = getattr(runtime_config, "session", None)
            audio = getattr(session, "audio", None) if session is not None else None
            output = getattr(audio, "output", None) if audio is not None else None
            sess_voice = getattr(output, "voice", None) if output is not None else None
            voice = str(sess_voice) if sess_voice else None

        name = self.default_voice
        if voice:
            if voice in self.voice_prompts:
                name = voice
            else:
                logger.warning("Unknown OmniVoice voice %r; falling back to %r", voice, name)
        if name != self._active_voice:
            logger.info("OmniVoice voice: %s -> %s", self._active_voice, name)
            self._active_voice = name
        return self.voice_prompts.get(name) if name else None

    @property
    def min_time_to_debug(self) -> float:
        # 音频 chunk 很多，避免刷屏；只记录异常慢的 chunk
        return 0.1

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        try:
            for _ in self._synthesize("Hello, this is a warmup."):
                pass
            logger.info(f"{self.__class__.__name__} warmed up")
        except Exception as e:
            logger.warning(f"Warmup generation failed: {e}")

    def _to_int16(self, audio: np.ndarray) -> np.ndarray:
        return np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    def _synthesize(
        self,
        text: str,
        cancel_gen: int | None = None,
        voice_clone_prompt: Any = None,
    ) -> Iterator[np.ndarray]:
        """合成一句文本，重采样后按 blocksize 产出 int16 音频块。

        voice_clone_prompt 缺省用默认音色（warmup 即走这条）。
        """
        if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
            logger.info("TTS generation cancelled (interruption)")
            return

        start = perf_counter()
        if voice_clone_prompt is None and self.default_voice is not None:
            voice_clone_prompt = self.voice_prompts.get(self.default_voice)
        generate_kwargs: dict[str, Any] = {"text": text, **self.gen_kwargs}
        if self.language:
            generate_kwargs["language"] = self.language
        if voice_clone_prompt is not None:
            generate_kwargs["voice_clone_prompt"] = voice_clone_prompt

        # 非流式 API：一次 generate 返回 list of np.ndarray（24kHz 单声道 float）
        audio_list = self.model.generate(**generate_kwargs)

        if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
            logger.info("TTS generation cancelled (interruption)")
            return

        generation_time = perf_counter() - start
        logger.info(f"OmniVoice TTFA: {generation_time:.2f}s (non-streaming, full utterance)")

        audio = np.concatenate(
            [np.atleast_1d(np.asarray(a, dtype=np.float32).squeeze()) for a in audio_list]
        )
        if audio.size == 0:
            logger.warning("OmniVoice returned empty audio for text: %r", text[:50])
            return

        if self._needs_resampling:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, up=self._resample_up, down=self._resample_down)

        audio_int16 = self._to_int16(audio)
        total_samples = len(audio_int16)
        for i in range(0, total_samples, self.blocksize):
            chunk = audio_int16[i : i + self.blocksize]
            if len(chunk) < self.blocksize:
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield chunk

        audio_duration = total_samples / self.sample_rate
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            f"OmniVoice generated {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f})"
        )

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        gen = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text
        if not text or not text.strip():
            return

        console.print(f"[green]ASSISTANT: {text}")

        voice_clone_prompt = self._select_prompt(tts_input)
        try:
            yield from self._synthesize(text, gen, voice_clone_prompt)
        except Exception as e:
            logger.error(f"Error during OmniVoice generation: {e}", exc_info=True)

    def cleanup(self) -> None:
        try:
            del self.model
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("OmniVoice handler cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
