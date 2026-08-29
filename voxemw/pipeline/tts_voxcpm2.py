"""VoxCPM2 TTS 积木（4090 满血版）：官方 PyTorch + CUDA。

两种出声模式（按人设 frontmatter 决定）：
- 音色设计：persona 带 voice_control 描述词 → 描述词拼进 text（VoxCPM2 的
  (control)text 约定），不给参考音，模型按描述凭空造嗓音
- 零样本克隆：persona 只有 ref_wav/ref_text → 传统克隆路径

与上游 qwen3 handler 同契约：process(TTSInput) → 产出 int16 16k mono 音频块，
结束补 AUDIO_RESPONSE_DONE。VoxCPM2 出 48kHz，resample_poly 降到管线 16kHz。

为什么满血版用它：4090 上官方 RTF ~0.3（流式），不用量化不用缓存；
Mac 上实时不可行的实测结论见 docs/（MLX 4bit/8bit 音质被否、GGUF/PyTorch 太慢）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Event

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE

from voxemw.pipeline.launch import strip_stage_directions

logger = logging.getLogger(__name__)

VCPM_SAMPLE_RATE = 48000
PIPELINE_SAMPLE_RATE = 16000


class VoxCPM2TTSHandler(BaseHandler):
    """VoxCPM2 零样本克隆 TTS handler。"""

    def setup(
        self,
        should_listen: Event,
        model_name: str = "openbmb/VoxCPM2",
        ref_audio: str | Path | None = None,
        ref_text: str = "",
        voice_control: str = "",
        voice_seed: int = 42,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        device: str = "cuda",
        blocksize: int = 512,
        cancel_scope=None,
        speculative_turns=None,
        **_unused,
    ) -> None:
        from voxcpm import VoxCPM

        self.should_listen = should_listen
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.voice_control = (voice_control or "").strip()
        self.voice_seed = int(voice_seed)  # 设计模式钉种子：每句同一噪声起点，音色逐句一致
        # 设计模式不需要参考音；voice_control 优先于 ref_*（两个 ref 参数必须同 None）
        self.ref_audio = None if self.voice_control else (str(ref_audio) if ref_audio else None)
        self.ref_text = None if self.voice_control else ref_text
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps

        logger.info("加载 VoxCPM2: %s (device=%s, 模式=%s)", model_name, device,
                    "音色设计" if self.voice_control else "零样本克隆")
        t0 = time.perf_counter()
        self.model = VoxCPM.from_pretrained(model_name, load_denoiser=False)
        logger.info("VoxCPM2 加载完成 %.1fs", time.perf_counter() - t0)

        # 预热（首次调用编译开销大）
        t0 = time.perf_counter()
        list(self.model.generate_streaming(
            text=self._design_text("你好。"), prompt_wav_path=self.ref_audio,
            prompt_text=self.ref_text,
            cfg_value=self.cfg_value, inference_timesteps=self.inference_timesteps,
        ))
        logger.info("VoxCPM2 预热完成 %.1fs", time.perf_counter() - t0)

    def _design_text(self, text: str) -> str:
        """音色设计模式：按 VoxCPM2 的 (control)text 约定拼描述词。"""
        return f"({self.voice_control}){text}" if self.voice_control else text

    def _to_pipeline_pcm(self, chunk: np.ndarray) -> bytes:
        """48k float32 → 16k int16（resample_poly 3:1 抽取）。"""
        from scipy.signal import resample_poly

        x = np.asarray(chunk, dtype=np.float32)
        if x.size < 8:
            return b""
        # 去掉 3 的倍数余量，边界样本留给下一块（防分块伪影）
        usable = (x.size // 3) * 3
        out = resample_poly(x[:usable], 1, 3)
        int16 = np.clip(out * 32768.0, -32768, 32767).astype(np.int16)
        return int16.tobytes()

    def process(self, tts_input):
        # 句子级入口（上游按 sentence batch 下发，括号对完整）：
        # 剥掉 LLM 偶尔冒出的舞台指示（（乐）（拍大腿）），防照字面念出。
        text = strip_stage_directions(getattr(tts_input, "text", "") or "").strip()
        if not text:
            yield AUDIO_RESPONSE_DONE
            return
        # 设计模式的描述词在剥括号之后拼（剥括号只针对 LLM 产出的舞台指示）
        text = self._design_text(text)
        if self.voice_control:
            # 钉种子：设计模式每次生成默认从随机噪声采样（同描述词音色逐句漂移），
            # 固定种子让每句从同一噪声起点出发，音色确定且一致
            import torch
            torch.manual_seed(self.voice_seed)
            torch.cuda.manual_seed_all(self.voice_seed)
        t0 = time.perf_counter()
        n_samples = 0
        first = True
        for chunk in self.model.generate_streaming(
            text=text,
            prompt_wav_path=self.ref_audio,
            prompt_text=self.ref_text,
            cfg_value=self.cfg_value,
            inference_timesteps=self.inference_timesteps,
        ):
            if self.cancel_scope is not None and self.cancel_scope.discarding:
                break  # 打断：本轮输出已被废弃
            pcm = self._to_pipeline_pcm(chunk)
            if not pcm:
                continue
            n_samples += len(pcm) // 2
            if first:
                logger.info("VoxCPM2 TTFA: %.2fs", time.perf_counter() - t0)
                first = False
            for i in range(0, len(pcm), self.blocksize):
                yield pcm[i:i + self.blocksize]
        dur = n_samples / PIPELINE_SAMPLE_RATE
        logger.info("VoxCPM2 生成 %.2fs 音频，耗时 %.2fs（RTF %.2f）",
                    dur, time.perf_counter() - t0,
                    (time.perf_counter() - t0) / max(dur, 1e-6))
        yield AUDIO_RESPONSE_DONE
