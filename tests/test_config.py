"""config 纯逻辑单测：YAML 积木配置加载、persona frontmatter/素材解析、api key 解析。"""

import pytest

from voxemw.config import load_config, load_dotenv, parse_persona_file, resolve_api_key


@pytest.fixture()
def assets(tmp_path):
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_bytes(b"RIFF-fake")
    ref_txt = tmp_path / "ref.txt"
    ref_txt.write_text("  逐字台词内容  \n", encoding="utf-8")
    ref_png = tmp_path / "ref.png"
    ref_png.write_bytes(b"\x89PNG-fake")
    persona = tmp_path / "demo.md"
    persona.write_text(
        "---\n"
        "name: 演示人设\n"
        f"ref_wav: {ref_wav}\n"
        f"ref_text: {ref_txt}\n"
        f"ref_image: {ref_png}\n"
        "---\n"
        "  你是演示人设。  \n",
        encoding="utf-8",
    )
    return {"wav": ref_wav, "txt": ref_txt, "png": ref_png, "persona": persona}


def _write_config(tmp_path, assets, extra_personas=""):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
vad: {{ backend: silero, min_silence_ms: 600 }}
stt: {{ backend: qwen3asr, model_name: Qwen/Qwen3-ASR-1.7B-hf }}
llm:
  backend: chat-completions
  model_name: deepseek-v4-flash
  base_url: https://api.deepseek.com/v1
  api_key_env: TEST_LLM_KEY
tts: {{ backend: voxcpm, model_name: openbmb/VoxCPM2, device: cuda }}
avatar: {{ enabled: true }}
personas:
  default: demo
  list:
    demo: {assets['persona']}
{extra_personas}
server:
  port: 8000
""",
        encoding="utf-8",
    )
    return cfg


def test_load_config_resolves_persona(tmp_path, assets):
    config = load_config(_write_config(tmp_path, assets))
    demo = config["personas"]["resolved"]["demo"]
    assert config["personas"]["default"] == "demo"
    assert demo["name"] == "演示人设"
    assert demo["text"] == "你是演示人设。"
    assert demo["ref_wav"] == str(assets["wav"])
    assert demo["ref_text"] == "逐字台词内容"
    assert demo["ref_image"] == str(assets["png"])


def test_load_config_missing_ref_image_degrades(tmp_path, assets):
    assets["png"].unlink()
    config = load_config(_write_config(tmp_path, assets))
    # 缺肖像不阻塞启动：ref_image 置 None，avatar 降级纯语音
    assert config["personas"]["resolved"]["demo"]["ref_image"] is None


def test_load_config_missing_block_exits(tmp_path, assets):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("vad: {}\nstt: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_config(cfg)


def test_load_config_missing_ref_wav_exits(tmp_path, assets):
    assets["wav"].unlink()
    with pytest.raises(SystemExit):
        load_config(_write_config(tmp_path, assets))


def test_load_config_default_not_in_list_exits(tmp_path, assets):
    cfg = tmp_path / "bad2.yaml"
    cfg.write_text(
        f"""
vad: {{}}
stt: {{}}
llm: {{}}
tts: {{}}
avatar: {{}}
personas:
  default: ghost
  list:
    demo: {assets['persona']}
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_config(cfg)


def test_parse_persona_file_without_frontmatter(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("  你是纯文本人设。  \n", encoding="utf-8")
    persona = parse_persona_file(p)
    assert persona["name"] == "plain"
    assert persona["text"] == "你是纯文本人设。"
    assert persona["ref_wav"] is None
    assert persona["ref_image"] is None


def test_resolve_api_key(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sk-test")
    assert resolve_api_key({"api_key_env": "TEST_LLM_KEY"}) == "sk-test"


def test_resolve_api_key_fallback(monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-fallback")
    assert resolve_api_key({"api_key_env": "TEST_LLM_KEY"}) == "sk-fallback"


def test_resolve_api_key_missing_exits(monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        resolve_api_key({"api_key_env": "TEST_LLM_KEY"})


def test_load_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# 注释\nTEST_A=foo\nTEST_B="bar baz"\nTEST_C=\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_A", raising=False)
    monkeypatch.delenv("TEST_B", raising=False)
    load_dotenv(env_file)
    import os

    assert os.environ["TEST_A"] == "foo"
    assert os.environ["TEST_B"] == "bar baz"
