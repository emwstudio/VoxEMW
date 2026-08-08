"""把 VoxEMW YAML 积木配置渲染成 speech-to-speech CLI argv（纯逻辑，可单测）。

stt/tts 积木是自定义 handler，不在上游 CLI 的 Literal 选项里，所以这里
不渲染 --stt/--tts；launch.py 在 parse_arguments() 之后直接改
module_kwargs.stt/tts，并 monkeypatch 工厂函数完成注册。
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
)

# server 段允许透传的模块级参数（ModuleArguments / WebSocketStreamerArguments）
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
    server = config.get("server") or {}

    argv = [
        "--mode", "realtime",
        "--ws_host", str(server.get("s2s_host", "127.0.0.1")),
        "--ws_port", str(server.get("s2s_port", 8765)),
        "--log_level", str(server.get("log_level", "info")),
    ]

    for key in _VAD_PASSTHROUGH:
        if key in vad:
            argv += [f"--{key}", str(vad[key])]

    backend = llm.get("backend", "chat-completions")
    # chat-completions-rag 是我们的自定义后端（launch 运行时注册），对上游 CLI 渲染为 chat-completions
    cli_backend = "chat-completions" if backend == "chat-completions-rag" else backend
    argv += ["--llm_backend", cli_backend, "--model_name", str(llm["model_name"])]
    if backend in ("chat-completions", "chat-completions-rag", "responses-api"):
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
    gen_kwargs = {
        key[len("gen_"):]: value
        for key, value in stt.items()
        if key.startswith("gen_")
    }
    if backend == "qwen3asr":
        return {
            "model_name": stt.get("model_name", "Qwen/Qwen3-ASR-1.7B-hf"),
            "device": stt.get("device", "cuda"),
            "torch_dtype": stt.get("torch_dtype", "bfloat16"),
            "language": stt.get("language", "Chinese"),
            "gen_kwargs": gen_kwargs,
        }
    if backend == "sensevoice":
        return {
            "model_name": stt.get("model_name", "iic/SenseVoiceSmall"),
            "device": stt.get("device", "cuda"),
            "language": stt.get("language", "zh"),
            "gen_kwargs": gen_kwargs,
        }
    raise ValueError(f"不支持的 stt.backend: {backend!r}")


def tts_setup_kwargs(config: dict) -> dict:
    """voxcpm TTS handler 的 setup_kwargs（由 launch 的自定义工厂使用）。

    voices：personas 注册表全员（key = persona id），启动时预编码 prompt cache，
    session.update 的 audio.output.voice 热切换；默认音色 = personas.default。
    """
    tts = config["tts"]
    if tts.get("backend") != "voxcpm":
        raise ValueError(f"不支持的 tts.backend: {tts.get('backend')!r}")
    personas = config["personas"]["resolved"]
    default_id = config["personas"]["default"]
    default = personas[default_id]

    import json

    voices = {
        pid: {"ref_audio": p["ref_wav"], "ref_text": p["ref_text"]}
        for pid, p in personas.items()
    }
    return {
        "model_name": tts.get("model_name", "openbmb/VoxCPM2"),
        "device": tts.get("device", "cuda"),
        "ref_audio": default["ref_wav"],
        "ref_text": default["ref_text"],
        "voices": json.dumps(voices, ensure_ascii=False),
        "sample_rate": int(tts.get("sample_rate", 16000)),
        "blocksize": int(tts.get("blocksize", 512)),
        "cfg_value": float(tts.get("cfg_value", 2.0)),
        "inference_timesteps": int(tts.get("inference_timesteps", 10)),
        "optimize": bool(tts.get("optimize", True)),
        "load_denoiser": bool(tts.get("load_denoiser", False)),
        "rate": float(tts.get("rate", 1.0)),
    }
