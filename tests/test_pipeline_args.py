"""s2s argv 渲染与自定义积木 setup_kwargs 的纯逻辑单测（无需 GPU/依赖）。"""

import pytest

from voxemw.pipeline.args import render_s2s_argv, stt_setup_kwargs


def _config():
    return {
        "vad": {"backend": "silero", "min_silence_ms": 600},
        "stt": {
            "backend": "qwen3asr",
            "model_name": "Qwen/Qwen3-ASR-0.6B-hf",
            "hotwords": ["大胃袋", "味真足"],
        },
        "llm": {
            "backend": "chat-completions",
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "TEST_LLM_KEY",
            "stream": False,
            "chat_size": 30,
        },
        "tts": {
            "backend": "qwen3",
            "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "device": "auto",
        },
        "avatar": {"enabled": True},
        "personas": {
            "default": "demo",
            "resolved": {
                "demo": {
                    "name": "演示",
                    "text": "人设正文",
                    "ref_wav": "/abs/demo/ref.wav",
                    "ref_text": "逐字台词",
                },
                "other": {
                    "name": "另一个",
                    "text": "另一个正文",
                    "ref_wav": "/abs/other/ref.wav",
                    "ref_text": "另一段台词",
                },
            },
        },
        "server": {
            "s2s_host": "127.0.0.1",
            "s2s_port": 8765,
            "enable_live_transcription": False,
            "num_pipelines": 1,
        },
    }


def _pairs(argv):
    return {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1, 2)}


def test_render_s2s_argv():
    argv = render_s2s_argv(_config(), env={"TEST_LLM_KEY": "sk-test"})
    pairs = _pairs(argv)
    assert pairs["--host"] == "127.0.0.1"
    assert pairs["--port"] == "8765"
    assert pairs["--min_silence_ms"] == "600"
    assert pairs["--llm_backend"] == "chat-completions"
    assert pairs["--model_name"] == "deepseek-v4-flash"
    assert pairs["--responses_api_base_url"] == "https://api.deepseek.com/v1"
    assert pairs["--responses_api_api_key"] == "sk-test"
    assert pairs["--responses_api_stream"] == "false"
    assert pairs["--chat_size"] == "30"
    assert pairs["--enable_live_transcription"] == "false"
    assert pairs["--num_pipelines"] == "1"
    # 上游 2026-08 重构后 CLI choices 由注册表生成——自定义积木先注册即合法
    assert pairs["--stt"] == "qwen3asr"
    assert pairs["--tts"] == "qwen3"
    # qwen3 TTS：CLI 直传模型与克隆参考（默认人设的 ref_wav/ref_text）
    assert pairs["--qwen3_tts_model_name"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert pairs["--qwen3_tts_ref_audio"] == "/abs/demo/ref.wav"
    assert pairs["--qwen3_tts_ref_text"] == "逐字台词"


def test_render_s2s_argv_streaming_llm():
    config = _config()
    config["llm"]["stream"] = True
    config["llm"]["stream_batch_sentences"] = 1
    pairs = _pairs(render_s2s_argv(config, env={"TEST_LLM_KEY": "sk"}))
    assert pairs["--responses_api_stream"] == "true"
    # 基类字段（LanguageModelBaseArguments）CLI 名不带前缀，与 --chat_size 同理
    assert pairs["--stream_batch_sentences"] == "1"


def test_render_s2s_argv_transformers_llm():
    config = _config()
    config["llm"] = {
        "backend": "transformers",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "device": "cuda",
    }
    pairs = _pairs(render_s2s_argv(config, env={}))
    assert pairs["--llm_backend"] == "transformers"
    assert pairs["--llm_device"] == "cuda"
    assert "--responses_api_api_key" not in pairs


def test_stt_setup_kwargs():
    kwargs = stt_setup_kwargs(_config())
    assert kwargs["model_name"] == "Qwen/Qwen3-ASR-0.6B-hf"
    assert kwargs["device"] == "auto"
    assert kwargs["language"] == "Chinese"
    # 列表词表拼成逗号串（chat-template system 注入用）
    assert kwargs["hotwords"] == "大胃袋, 味真足"
    assert kwargs["max_new_tokens"] == 256


def test_stt_setup_kwargs_qwen3asr_defaults():
    config = _config()
    # 字符串词表原样透传；缺省 model_name 兜底
    config["stt"] = {"backend": "qwen3asr", "hotwords": "大胃袋"}
    kwargs = stt_setup_kwargs(config)
    assert kwargs["model_name"] == "Qwen/Qwen3-ASR-0.6B-hf"
    assert kwargs["hotwords"] == "大胃袋"


def test_setup_kwargs_reject_unknown_backend():
    config = _config()
    config["stt"]["backend"] = "whisper"
    with pytest.raises(ValueError):
        stt_setup_kwargs(config)
