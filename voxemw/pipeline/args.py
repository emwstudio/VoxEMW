"""把 VoxEMW YAML 积木配置渲染成 speech-to-speech CLI argv（纯逻辑，可单测）。

上游 2026-08 重构后：--stt/--tts 的 choices 由 backend_registry 动态生成，
launch.py 先调 register_custom_backends() 把我们的积木插进注册表，
这里直接渲染 --stt qwen3asr --tts qwen3 即可（不再靠 parse 后改写）。
"""

from __future__ import annotations

from voxemw.config import resolve_api_key

# VAD 段允许透传给上游 CLI 的参数（VADHandlerArguments）
_VAD_PASSTHROUGH = (
    "thresh",
    "min_silence_ms",
    "min_speech_ms",
    "min_speech_continuation_ms",
    "speech_pad_ms",
    "max_speech_ms",
    "smart_turn_model_path",
)

# server 段允许透传的模块级参数（ModuleArguments）
_SERVER_PASSTHROUGH = (
    "enable_live_transcription",
    "num_pipelines",
)


def _bool(value) -> str:
    return "true" if value else "false"


def render_s2s_argv(config: dict, env: dict | None = None) -> list[str]:
    """渲染 speech-to-speech 的 CLI 参数列表（不含程序名）。"""
    vad = config["vad"]
    llm = config["llm"]
    stt = config["stt"]
    tts = config["tts"]
    server = config.get("server") or {}

    argv = [
        "--host", str(server.get("s2s_host", "127.0.0.1")),
        "--port", str(server.get("s2s_port", 8765)),
        "--log_level", str(server.get("log_level", "info")),
        "--stt", str(stt.get("backend", "qwen3asr")),
        "--tts", str(tts.get("backend", "qwen3")),
    ]

    # 上游内置 qwen3 TTS 后端（Mac/MLX 路线）：CLI 直传模型与克隆参考
    if tts.get("backend") == "qwen3":
        persona = config["personas"]["resolved"][config["personas"]["default"]]
        argv += [
            "--qwen3_tts_model_name",
            str(tts.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")),
            "--qwen3_tts_device", str(tts.get("device", "auto")),
            "--qwen3_tts_ref_audio", str(persona["ref_wav"]),
            "--qwen3_tts_ref_text", persona["ref_text"],
        ]

    for key in _VAD_PASSTHROUGH:
        if key in vad:
            argv += [f"--{key}", str(vad[key])]

    backend = llm.get("backend", "chat-completions")
    argv += ["--llm_backend", backend, "--model_name", str(llm["model_name"])]
    if backend in ("chat-completions", "responses-api"):
        argv += [
            "--responses_api_base_url", str(llm["base_url"]),
            "--responses_api_api_key", resolve_api_key(llm, env),
            "--responses_api_stream", _bool(llm.get("stream", False)),
        ]
        if "chat_size" in llm:
            argv += ["--chat_size", str(llm["chat_size"])]
        if "stream_batch_sentences" in llm:
            argv += ["--stream_batch_sentences", str(llm["stream_batch_sentences"])]
        if "reasoning_effort" in llm:
            argv += ["--responses_api_reasoning_effort", str(llm["reasoning_effort"])]
    elif backend == "mlx-lm":
        # Mac 本地 LLM（mlx-lm）：模型经 --model_name 传入（上行已统一加），
        # 只需设备参数；采样参数上游有默认值
        argv += ["--llm_device", str(llm.get("device", "mps"))]
    elif backend == "transformers":
        argv += [
            "--llm_device", str(llm.get("device", "cuda")),
            "--llm_torch_dtype", str(llm.get("torch_dtype", "bfloat16")),
        ]
    else:
        raise ValueError(f"不支持的 llm.backend: {backend!r}")

    for key in _SERVER_PASSTHROUGH:
        if key in server:
            value = server[key]
            argv += [f"--{key}", _bool(value) if isinstance(value, bool) else str(value)]

    return argv


def stt_setup_kwargs(config: dict) -> dict:
    """STT handler 的 setup_kwargs（由 launch 的自定义工厂使用）。"""
    stt = config["stt"]
    backend = stt.get("backend")
    if backend == "qwen3asr":
        hotwords = stt.get("hotwords", [])
        if isinstance(hotwords, list):
            hotwords = ", ".join(hotwords)
        return {
            "model_name": stt.get("model_name", "Qwen/Qwen3-ASR-0.6B-hf"),
            "device": stt.get("device", "auto"),
            "language": stt.get("language", "Chinese"),
            "hotwords": hotwords,
            "max_new_tokens": int(stt.get("max_new_tokens", 256)),
        }
    raise ValueError(f"不支持的 stt.backend: {backend!r}")

