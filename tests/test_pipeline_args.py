"""s2s argv 渲染与自定义积木 setup_kwargs 的纯逻辑单测（无需 GPU/依赖）。"""

import json

import pytest

from voxemw.pipeline.args import render_s2s_argv, stt_setup_kwargs, tts_setup_kwargs


def _config():
    return {
        "vad": {"backend": "silero", "min_silence_ms": 600},
        "stt": {
            "backend": "qwen3asr",
            "model_name": "Qwen/Qwen3-ASR-1.7B-hf",
            "device": "cuda",
            "torch_dtype": "bfloat16",
            "language": "Chinese",
            "gen_max_new_tokens": 256,
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
            "backend": "voxcpm",
            "model_name": "openbmb/VoxCPM2",
            "device": "cuda",
            "optimize": True,
        },
        "avatar": {"backend": "flashhead", "enabled": True},
        "personas": {
            "default": "demo",
            "resolved": {
                "demo": {
                    "name": "演示",
                    "text": "人设正文",
                    "ref_wav": "/abs/demo/ref.wav",
                    "ref_text": "逐字台词",
                    "ref_image": "/abs/demo/ref.png",
                },
                "other": {
                    "name": "另一个",
                    "text": "另一个正文",
                    "ref_wav": "/abs/other/ref.wav",
                    "ref_text": "另一段台词",
                    "ref_image": None,
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
    assert pairs["--mode"] == "realtime"
    assert pairs["--ws_port"] == "8765"
    assert pairs["--min_silence_ms"] == "600"
    assert pairs["--llm_backend"] == "chat-completions"
    assert pairs["--model_name"] == "deepseek-v4-flash"
    assert pairs["--responses_api_base_url"] == "https://api.deepseek.com/v1"
    assert pairs["--responses_api_api_key"] == "sk-test"
    assert pairs["--responses_api_stream"] == "false"
    assert pairs["--chat_size"] == "30"
    assert pairs["--enable_live_transcription"] == "false"
    assert pairs["--num_pipelines"] == "1"
    # 自定义积木不进 CLI（Literal 校验过不了），由 launch 运行时注册
    assert "--stt" not in pairs
    assert "--tts" not in pairs


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
    assert kwargs["model_name"] == "Qwen/Qwen3-ASR-1.7B-hf"
    assert kwargs["language"] == "Chinese"
    assert kwargs["gen_kwargs"] == {"max_new_tokens": 256}


def test_stt_setup_kwargs_sensevoice():
    config = _config()
    config["stt"] = {"backend": "sensevoice", "model_name": "iic/SenseVoiceSmall"}
    kwargs = stt_setup_kwargs(config)
    assert kwargs["model_name"] == "iic/SenseVoiceSmall"
    assert kwargs["device"] == "cuda"
    assert kwargs["language"] == "zh"
    assert kwargs["gen_kwargs"] == {}


def test_tts_setup_kwargs_voices_from_personas():
    kwargs = tts_setup_kwargs(_config())
    assert kwargs["model_name"] == "openbmb/VoxCPM2"
    assert kwargs["optimize"] is True
    # 默认音色 = personas.default
    assert kwargs["ref_audio"] == "/abs/demo/ref.wav"
    assert kwargs["ref_text"] == "逐字台词"
    # voices 表 = personas 全员（key = persona id，供 session.update 热切换）
    voices = json.loads(kwargs["voices"])
    assert set(voices) == {"demo", "other"}
    assert voices["other"] == {"ref_audio": "/abs/other/ref.wav", "ref_text": "另一段台词"}


def test_setup_kwargs_reject_unknown_backend():
    config = _config()
    config["stt"]["backend"] = "whisper"
    with pytest.raises(ValueError):
        stt_setup_kwargs(config)
    config = _config()
    config["tts"]["backend"] = "kokoro"
    with pytest.raises(ValueError):
        tts_setup_kwargs(config)
