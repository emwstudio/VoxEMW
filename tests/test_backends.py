"""后端注册表完整性 + YAML -> CLI 渲染测试（纯逻辑，不 import torch/transformers）。"""

import copy

import pytest
import yaml

from voxemw.backends import (
    BACKENDS,
    SERVER_PARAMS,
    UnknownBackendError,
    UnknownParamError,
    render_argv,
)

EXPECTED_BACKENDS = {
    "vad": {"silero"},
    "stt": {"qwen3asr", "whisper", "faster-whisper", "parakeet-tdt", "paraformer"},
    "llm": {"transformers", "responses-api", "chat-completions"},
    "tts": {"voxcpm", "omnivoice", "qwen3", "kokoro", "pocket", "chatTTS", "facebookMMS"},
}


def _load_example_config(repo_root):
    with open(repo_root / "configs" / "autodl-4090.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pairs(argv):
    """argv -> {flag: value}（本注册表渲染的全是 flag value 对）。"""
    assert len(argv) % 2 == 0
    return {argv[i]: argv[i + 1] for i in range(0, len(argv), 2)}


# ---------------------------------------------------------------------------
# 注册表完整性
# ---------------------------------------------------------------------------

def test_all_expected_backends_registered():
    for section, expected in EXPECTED_BACKENDS.items():
        assert expected <= set(BACKENDS[section]), (
            f"{section} 缺 backend：{expected - set(BACKENDS[section])}"
        )


def test_every_backend_has_nonempty_flag_mapping():
    for section, entries in BACKENDS.items():
        for name, entry in entries.items():
            assert entry["params"], f"{section}/{name} 的 flag 映射为空"
            for param, flag in entry["params"].items():
                assert flag.startswith("--"), f"{section}/{name}.{param} 的 flag 非法: {flag}"


def test_custom_backends_flags_match_patch():
    """自定义后端的 flag 名与 patches/register-handlers.patch 的参数类一致。"""
    qwen3asr = BACKENDS["stt"]["qwen3asr"]["params"]
    assert qwen3asr["model_name"] == "--qwen3asr_stt_model_name"
    assert qwen3asr["language"] == "--qwen3asr_stt_language"
    assert qwen3asr["gen_max_new_tokens"] == "--qwen3asr_stt_gen_max_new_tokens"

    omnivoice = BACKENDS["tts"]["omnivoice"]["params"]
    assert omnivoice["model_name"] == "--omnivoice_tts_model_name"
    assert omnivoice["ref_audio"] == "--omnivoice_tts_ref_audio"
    assert omnivoice["ref_text"] == "--omnivoice_tts_ref_text"
    assert omnivoice["voices"] == "--omnivoice_tts_voices"
    assert omnivoice["sample_rate"] == "--omnivoice_tts_sample_rate"
    assert omnivoice["blocksize"] == "--omnivoice_tts_blocksize"

    voxcpm = BACKENDS["tts"]["voxcpm"]["params"]
    assert voxcpm["model_name"] == "--voxcpm_tts_model_name"
    assert voxcpm["ref_audio"] == "--voxcpm_tts_ref_audio"
    assert voxcpm["ref_text"] == "--voxcpm_tts_ref_text"
    assert voxcpm["voices"] == "--voxcpm_tts_voices"
    assert voxcpm["sample_rate"] == "--voxcpm_tts_sample_rate"
    assert voxcpm["blocksize"] == "--voxcpm_tts_blocksize"
    assert voxcpm["cfg_value"] == "--voxcpm_tts_cfg_value"
    assert voxcpm["inference_timesteps"] == "--voxcpm_tts_inference_timesteps"


def test_builtin_flag_names_match_vendor():
    """抽查上游内置后端的 flag 名（核对自 arguments_classes/）。"""
    assert BACKENDS["stt"]["whisper"]["params"]["model_name"] == "--stt_model_name"
    assert BACKENDS["stt"]["faster-whisper"]["params"]["model_name"] == "--faster_whisper_stt_model_name"
    assert BACKENDS["stt"]["parakeet-tdt"]["params"]["model_name"] == "--parakeet_tdt_model_name"
    assert BACKENDS["stt"]["paraformer"]["params"]["model_name"] == "--paraformer_stt_model_name"
    assert BACKENDS["tts"]["qwen3"]["params"]["model_name"] == "--qwen3_tts_model_name"
    assert BACKENDS["tts"]["kokoro"]["params"]["voice"] == "--kokoro_voice"
    assert BACKENDS["tts"]["pocket"]["params"]["voice"] == "--pocket_tts_voice"
    assert BACKENDS["tts"]["chatTTS"]["params"]["stream"] == "--chat_tts_stream"
    assert BACKENDS["tts"]["facebookMMS"]["params"]["model_name"] == "--facebook_mms_model_name"
    # llm 通用 + API 连接参数
    for backend in ("responses-api", "chat-completions"):
        params = BACKENDS["llm"][backend]["params"]
        assert params["base_url"] == "--responses_api_base_url"
        assert params["api_key"] == "--responses_api_api_key"
        assert params["stream"] == "--responses_api_stream"
        assert params["chat_size"] == "--chat_size"
    assert BACKENDS["llm"]["chat-completions"]["params"]["reasoning_effort"] == "--responses_api_reasoning_effort"
    assert BACKENDS["llm"]["transformers"]["params"]["device"] == "--llm_device"


# ---------------------------------------------------------------------------
# YAML -> CLI 渲染
# ---------------------------------------------------------------------------

def test_render_example_config(repo_root):
    config = _load_example_config(repo_root)
    argv = render_argv(config, env={"DEEPSEEK_API_KEY": "sk-test"})
    flags = _pairs(argv)

    # server 段
    assert flags["--mode"] == "realtime"
    assert flags["--ws_host"] == "0.0.0.0"
    assert flags["--ws_port"] == "8765"
    # 积木选择器
    assert flags["--stt"] == "qwen3asr"
    assert flags["--llm_backend"] == "chat-completions"
    assert flags["--tts"] == "voxcpm"
    # 各段关键参数
    assert flags["--qwen3asr_stt_model_name"] == "Qwen/Qwen3-ASR-1.7B-hf"
    assert flags["--model_name"] == "deepseek-v4-flash"
    assert flags["--responses_api_base_url"] == "https://api.deepseek.com/v1"
    assert flags["--responses_api_stream"] == "false"  # 非流式：整段出完送 TTS（见 YAML 注释）
    # DeepSeek 关 thinking 走 patch 的 extra_body，不再传 reasoning_effort
    assert "--responses_api_reasoning_effort" not in flags
    assert flags["--responses_api_api_key"] == "sk-test"
    assert flags["--chat_size"] == "30"
    assert flags["--voxcpm_tts_model_name"] == "openbmb/VoxCPM2"
    assert flags["--voxcpm_tts_sample_rate"] == "16000"
    assert flags["--voxcpm_tts_cfg_value"] == "2.0"
    assert flags["--voxcpm_tts_inference_timesteps"] == "10"


def test_render_tts_voices(repo_root):
    """tts.voices 多音色映射经 launch 预处理后渲染成单个 --voxcpm_tts_voices JSON flag。"""
    import json
    from pathlib import Path

    import launch

    config = _load_example_config(repo_root)
    launch.resolve_file_valued_params(config)
    flags = _pairs(render_argv(config, env={"DEEPSEEK_API_KEY": "x"}))

    voices = json.loads(flags["--voxcpm_tts_voices"])
    assert set(voices) == {"liangzi", "fengge"}
    for spec in voices.values():
        # ref_audio 已解析成存在的绝对路径；ref_text 已读入文本内容（不再是 txt 路径）
        assert Path(spec["ref_audio"]).is_file()
        assert spec["ref_text"]
        assert not spec["ref_text"].endswith(".txt")
    # 默认音色（ref_audio）仍是良子
    assert flags["--voxcpm_tts_ref_audio"].endswith("assets/liangzi/ref.wav")


def test_render_missing_api_key_env_raises(repo_root):
    config = _load_example_config(repo_root)
    with pytest.raises(KeyError, match="DEEPSEEK_API_KEY"):
        render_argv(config, env={})


def test_render_unknown_backend_raises(repo_root):
    config = _load_example_config(repo_root)
    config["stt"]["backend"] = "not-a-thing"
    with pytest.raises(UnknownBackendError):
        render_argv(config, env={"DEEPSEEK_API_KEY": "x"})


def test_render_unknown_param_raises(repo_root):
    config = _load_example_config(repo_root)
    config["stt"]["no_such_param"] = 1
    with pytest.raises(UnknownParamError):
        render_argv(config, env={"DEEPSEEK_API_KEY": "x"})


def test_render_swap_backend(repo_root):
    """换积木：stt 换 paraformer、tts 换 kokoro、llm 换本地 transformers。"""
    config = _load_example_config(repo_root)
    config["stt"] = {"backend": "paraformer", "model_name": "paraformer-zh"}
    config["tts"] = {"backend": "kokoro", "voice": "zf_xiaoxiao", "lang_code": "z"}
    config["llm"] = {
        "backend": "transformers",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "device": "cuda",
        "torch_dtype": "bfloat16",
    }
    flags = _pairs(render_argv(config, env={}))
    assert flags["--stt"] == "paraformer"
    assert flags["--paraformer_stt_model_name"] == "paraformer-zh"
    assert flags["--tts"] == "kokoro"
    assert flags["--kokoro_voice"] == "zf_xiaoxiao"
    assert flags["--llm_backend"] == "transformers"
    assert flags["--llm_torch_dtype"] == "bfloat16"
    # transformers 后端不需要 API key
    assert "--responses_api_api_key" not in flags


def test_bool_rendered_explicitly(repo_root):
    """bool 渲染成显式 true/false（HfArgumentParser 的 string_to_bool 解析）。"""
    config = _load_example_config(repo_root)
    config["llm"]["stream"] = False
    flags = _pairs(render_argv(config, env={"DEEPSEEK_API_KEY": "x"}))
    assert flags["--responses_api_stream"] == "false"


def test_persona_and_python_not_rendered(repo_root):
    """persona 段与 server.python 是启动器/前端消费的，不应出现在 argv。"""
    config = _load_example_config(repo_root)
    config["server"]["python"] = ".venv/bin/python"
    argv = render_argv(config, env={"DEEPSEEK_API_KEY": "x"})
    assert not any("personas/" in a or "venv" in a for a in argv)
    assert copy.deepcopy(SERVER_PARAMS)  # server 映射非空
