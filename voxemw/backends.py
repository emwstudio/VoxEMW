"""后端注册表：YAML 积木配置 -> speech-to-speech CLI argv。

每个环节（vad/stt/llm/tts）一张表：backend 名 -> 参数名 -> CLI flag 映射。
flag 名逐一核对自 vendor/speech-to-speech/src/speech_to_speech/arguments_classes/
（自定义后端 qwen3asr / omnivoice 的 flag 见 patches/register-handlers.patch）。
纯逻辑模块：不 import torch / transformers，可在 macOS 开发机直接单测。
"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# 公共参数（llm 各后端共享，来自 LanguageModelBaseArguments）
# ---------------------------------------------------------------------------

_LLM_BASE_PARAMS = {
    "model_name": "--model_name",
    "user_role": "--user_role",
    "init_chat_role": "--init_chat_role",
    "init_chat_prompt": "--init_chat_prompt",
    "chat_size": "--chat_size",
    "stream_batch_sentences": "--stream_batch_sentences",
    "enable_lang_prompt": "--enable_lang_prompt",
    "compact_history": "--compact_history",
}

# OpenAI 兼容连接参数（responses-api / chat-completions 共享，
# chat-completions 的 ResponsesApiLanguageModelHandlerArguments 子类沿用
# responses_api_* 前缀的 flag）
_API_PARAMS = {
    "api_key": "--responses_api_api_key",
    "base_url": "--responses_api_base_url",
    "stream": "--responses_api_stream",
    "disable_thinking": "--responses_api_disable_thinking",
}

# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
# 每个 backend 条目：
#   selector_flag: 选择该后端的 CLI flag（vad 只有一个实现，无 selector）
#   params:        YAML 参数名 -> CLI flag
# YAML 的 `backend` 字段用于选择条目，不直接渲染。

BACKENDS: Dict[str, Dict[str, dict]] = {
    "vad": {
        "silero": {
            "selector_flag": None,
            "params": {
                "thresh": "--thresh",
                "sample_rate": "--sample_rate",
                "min_silence_ms": "--min_silence_ms",
                "min_speech_ms": "--min_speech_ms",
                "min_speech_continuation_ms": "--min_speech_continuation_ms",
                "max_speech_ms": "--max_speech_ms",
                "speech_pad_ms": "--speech_pad_ms",
                "audio_enhancement": "--audio_enhancement",
                "enable_realtime_transcription": "--enable_realtime_transcription",
                "realtime_processing_pause": "--realtime_processing_pause",
                "speculative_reopen_ms": "--speculative_reopen_ms",
                "unanswered_reopen_ms": "--unanswered_reopen_ms",
                "short_segment_merge_ms": "--short_segment_merge_ms",
            },
        },
    },
    "stt": {
        # ---- 自定义（patches/register-handlers.patch 注册）----
        "qwen3asr": {
            "selector_flag": "--stt",
            "params": {
                "model_name": "--qwen3asr_stt_model_name",
                "device": "--qwen3asr_stt_device",
                "torch_dtype": "--qwen3asr_stt_torch_dtype",
                "language": "--qwen3asr_stt_language",
                "gen_max_new_tokens": "--qwen3asr_stt_gen_max_new_tokens",
            },
        },
        # ---- 上游内置 ----
        "whisper": {
            "selector_flag": "--stt",
            "params": {
                "model_name": "--stt_model_name",
                "device": "--stt_device",
                "torch_dtype": "--stt_torch_dtype",
                "compile_mode": "--stt_compile_mode",
                "gen_max_new_tokens": "--stt_gen_max_new_tokens",
                "gen_num_beams": "--stt_gen_num_beams",
                "gen_return_timestamps": "--stt_gen_return_timestamps",
                "gen_task": "--stt_gen_task",
                "language": "--language",
            },
        },
        "faster-whisper": {
            "selector_flag": "--stt",
            "params": {
                "model_name": "--faster_whisper_stt_model_name",
                "device": "--faster_whisper_stt_device",
                "compute_type": "--faster_whisper_stt_compute_type",
                "gen_max_new_tokens": "--faster_whisper_stt_gen_max_new_tokens",
                "gen_beam_size": "--faster_whisper_stt_gen_beam_size",
                "gen_return_timestamps": "--faster_whisper_stt_gen_return_timestamps",
                "gen_task": "--faster_whisper_stt_gen_task",
                "gen_language": "--faster_whisper_stt_gen_language",
            },
        },
        "parakeet-tdt": {
            "selector_flag": "--stt",
            "params": {
                "model_name": "--parakeet_tdt_model_name",
                "device": "--parakeet_tdt_device",
                "compute_type": "--parakeet_tdt_compute_type",
                "language": "--parakeet_tdt_language",
            },
        },
        "paraformer": {
            "selector_flag": "--stt",
            "params": {
                "model_name": "--paraformer_stt_model_name",
                "device": "--paraformer_stt_device",
            },
        },
    },
    "llm": {
        "transformers": {
            "selector_flag": "--llm_backend",
            "params": {
                **_LLM_BASE_PARAMS,
                "device": "--llm_device",
                "torch_dtype": "--llm_torch_dtype",
                "gen_max_new_tokens": "--llm_gen_max_new_tokens",
                "gen_min_new_tokens": "--llm_gen_min_new_tokens",
                "gen_temperature": "--llm_gen_temperature",
                "gen_do_sample": "--llm_gen_do_sample",
                "is_vlm": "--llm_is_vlm",
            },
        },
        "responses-api": {
            "selector_flag": "--llm_backend",
            "params": {**_LLM_BASE_PARAMS, **_API_PARAMS},
        },
        "chat-completions": {
            "selector_flag": "--llm_backend",
            "params": {
                **_LLM_BASE_PARAMS,
                **_API_PARAMS,
                "reasoning_effort": "--responses_api_reasoning_effort",
            },
        },
    },
    "tts": {
        # ---- 自定义（patches/register-handlers.patch 注册）----
        "voxcpm": {
            "selector_flag": "--tts",
            "params": {
                "model_name": "--voxcpm_tts_model_name",
                "device": "--voxcpm_tts_device",
                "ref_audio": "--voxcpm_tts_ref_audio",
                "ref_text": "--voxcpm_tts_ref_text",
                "voices": "--voxcpm_tts_voices",
                "sample_rate": "--voxcpm_tts_sample_rate",
                "blocksize": "--voxcpm_tts_blocksize",
                "cfg_value": "--voxcpm_tts_cfg_value",
                "inference_timesteps": "--voxcpm_tts_inference_timesteps",
                "optimize": "--voxcpm_tts_optimize",
                "load_denoiser": "--voxcpm_tts_load_denoiser",
            },
        },
        "omnivoice": {
            "selector_flag": "--tts",
            "params": {
                "model_name": "--omnivoice_tts_model_name",
                "device": "--omnivoice_tts_device",
                "dtype": "--omnivoice_tts_dtype",
                "ref_audio": "--omnivoice_tts_ref_audio",
                "ref_text": "--omnivoice_tts_ref_text",
                "voices": "--omnivoice_tts_voices",
                "prompt_cache_dir": "--omnivoice_tts_prompt_cache_dir",
                "sample_rate": "--omnivoice_tts_sample_rate",
                "blocksize": "--omnivoice_tts_blocksize",
                "language": "--omnivoice_tts_language",
            },
        },
        # ---- 上游内置 ----
        "qwen3": {
            "selector_flag": "--tts",
            "params": {
                "model_name": "--qwen3_tts_model_name",
                "device": "--qwen3_tts_device",
                "dtype": "--qwen3_tts_dtype",
                "attn_implementation": "--qwen3_tts_attn_implementation",
                "backend": "--qwen3_tts_backend",
                "ref_audio": "--qwen3_tts_ref_audio",
                "ref_text": "--qwen3_tts_ref_text",
                "speaker": "--qwen3_tts_speaker",
                "instruct": "--qwen3_tts_instruct",
                "xvec_only": "--qwen3_tts_xvec_only",
                "parity_mode": "--qwen3_tts_parity_mode",
                "non_streaming_mode": "--qwen3_tts_non_streaming_mode",
                "mlx_quantization": "--qwen3_tts_mlx_quantization",
                "language": "--qwen3_tts_language",
                "streaming_chunk_size": "--qwen3_tts_streaming_chunk_size",
                "max_new_tokens": "--qwen3_tts_max_new_tokens",
                "blocksize": "--qwen3_tts_blocksize",
            },
        },
        "kokoro": {
            "selector_flag": "--tts",
            "params": {
                "model_name": "--kokoro_model_name",
                "device": "--kokoro_device",
                "voice": "--kokoro_voice",
                "lang_code": "--kokoro_lang_code",
                "speed": "--kokoro_speed",
                "blocksize": "--kokoro_blocksize",
            },
        },
        "pocket": {
            "selector_flag": "--tts",
            "params": {
                "device": "--pocket_tts_device",
                "voice": "--pocket_tts_voice",
                "sample_rate": "--pocket_tts_sample_rate",
                "blocksize": "--pocket_tts_blocksize",
                "max_tokens": "--pocket_tts_max_tokens",
            },
        },
        "chatTTS": {
            "selector_flag": "--tts",
            "params": {
                "stream": "--chat_tts_stream",
                "device": "--chat_tts_device",
                "chunk_size": "--chat_tts_chunk_size",
            },
        },
        "facebookMMS": {
            "selector_flag": "--tts",
            "params": {
                "model_name": "--facebook_mms_model_name",
                "language": "--tts_language",
                "device": "--facebook_mms_device",
                "torch_dtype": "--facebook_mms_torch_dtype",
            },
        },
    },
}

# server 段不是「可换后端」的积木，只是管线自身的 CLI 参数
SERVER_PARAMS = {
    "mode": "--mode",
    "device": "--device",
    "ws_host": "--ws_host",
    "ws_port": "--ws_port",
    "log_level": "--log_level",
    "num_pipelines": "--num_pipelines",
    "enable_live_transcription": "--enable_live_transcription",
    "local_mac_optimal_settings": "--local_mac_optimal_settings",
}

# YAML 里有、但不渲染成 CLI flag 的键（launch.py / 前端消费）
NON_CLI_KEYS = {"backend", "python", "api_key_env"}


class UnknownBackendError(ValueError):
    """配置里的 backend 不在注册表。"""


class UnknownParamError(ValueError):
    """配置里的参数在该 backend 的注册表中没有对应 flag。"""


def _format_value(value) -> str:
    if isinstance(value, bool):
        # transformers HfArgumentParser 的 bool 用 string_to_bool 解析，
        # 显式传 "true"/"false" 比裸 flag 更稳（默认 True 的 flag 也能关）
        return "true" if value else "false"
    return str(value)


def _render_section(
    section: str,
    section_cfg: Mapping,
    env: Mapping[str, str],
) -> List[str]:
    """渲染一个积木段（vad/stt/llm/tts）为 argv 片段。"""
    backends = BACKENDS[section]
    backend = section_cfg.get("backend")
    if backend not in backends:
        raise UnknownBackendError(
            f"{section}.backend={backend!r} 未注册，可选：{sorted(backends)}"
        )
    entry = backends[backend]

    argv: List[str] = []
    if entry["selector_flag"] is not None:
        argv += [entry["selector_flag"], str(backend)]

    for key, value in section_cfg.items():
        if key in NON_CLI_KEYS:
            continue
        flag = entry["params"].get(key)
        if flag is None:
            raise UnknownParamError(
                f"{section}.{key} 在 backend={backend!r} 下没有对应 CLI flag，"
                f"可用参数：{sorted(entry['params'])}"
            )
        argv += [flag, _format_value(value)]

    # llm.api_key_env：从环境变量取真实 key，渲染成 --responses_api_api_key
    api_key_env = section_cfg.get("api_key_env")
    if api_key_env is not None:
        if "api_key" not in entry["params"]:
            raise UnknownParamError(
                f"{section}.backend={backend!r} 不支持 api_key_env（无 API key 参数）"
            )
        api_key = env.get(str(api_key_env))
        if not api_key:
            raise KeyError(
                f"环境变量 {api_key_env} 未设置（llm.api_key_env 需要它提供 API key）"
            )
        argv += [entry["params"]["api_key"], api_key]

    return argv


def render_argv(
    config: Mapping,
    env: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """把整份 YAML 配置渲染成 speech-to-speech 的完整 argv（不含程序名）。

    config: 六段 dict（server/vad/stt/llm/tts/persona），persona 段不渲染。
    env:    环境变量表（默认 os.environ），供 llm.api_key_env 解析。
    """
    if env is None:
        env = os.environ

    argv: List[str] = []

    server_cfg = config.get("server") or {}
    for key, value in server_cfg.items():
        if key in NON_CLI_KEYS:
            continue
        flag = SERVER_PARAMS.get(key)
        if flag is None:
            raise UnknownParamError(
                f"server.{key} 没有对应 CLI flag，可用参数：{sorted(SERVER_PARAMS)}"
            )
        argv += [flag, _format_value(value)]

    for section in ("vad", "stt", "llm", "tts"):
        section_cfg = config.get(section) or {}
        if section_cfg:
            argv += _render_section(section, section_cfg, env)

    return argv


def list_backends() -> Dict[str, List[str]]:
    """各环节可选 backend 清单（供 README / 测试核对）。"""
    return {section: sorted(entries) for section, entries in BACKENDS.items()}
