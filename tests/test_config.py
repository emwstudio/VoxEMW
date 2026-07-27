"""config 纯逻辑单测：新 YAML（blocks 五积木）加载、人设/音色解析、api key 解析。"""

import pytest

from voxemw.config import load_config, load_dotenv, resolve_api_key


@pytest.fixture()
def assets(tmp_path):
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_bytes(b"RIFF-fake")
    ref_txt = tmp_path / "ref.txt"
    ref_txt.write_text("  逐字台词内容  \n", encoding="utf-8")
    persona = tmp_path / "persona.md"
    persona.write_text("  你是突发主播。  \n", encoding="utf-8")
    return {"wav": ref_wav, "txt": ref_txt, "persona": persona}


def _write_config(tmp_path, assets):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
blocks:
  vad: {{ impl: webspeech }}
  stt: {{ impl: webspeech, lang: zh-CN }}
  llm:
    impl: deepseek
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com/v1
    api_key_env: TEST_LLM_KEY
  tts: {{ impl: voxcpm2, model_name: openbmb/VoxCPM2, device: cuda }}
  persona: {{ impl: file, path: {assets['persona']} }}
alerter: {{ weibo_name: 峰哥亡命天涯, max_briefing_chars: 150 }}
voices:
  demo:
    name: 演示音色
    ref_audio: {assets['wav']}
    ref_text: {assets['txt']}
server:
  port: 8000
""",
        encoding="utf-8",
    )
    return cfg


class TestLoadConfig:
    def test_resolves_blocks_persona_and_voices(self, tmp_path, assets):
        config = load_config(_write_config(tmp_path, assets))
        assert config["blocks"]["stt"]["lang"] == "zh-CN"
        assert config["persona_text"] == "你是突发主播。"  # 读入文本并 strip
        spec = config["voices"]["demo"]
        assert spec["name"] == "演示音色"
        assert spec["ref_audio"] == str(assets["wav"])
        assert spec["ref_text"] == "逐字台词内容"  # 读入文本并 strip

    def test_missing_block_exits(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("blocks:\n  vad: {impl: webspeech}\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            load_config(cfg)

    def test_missing_persona_file_exits(self, tmp_path, assets):
        assets["persona"].unlink()
        with pytest.raises(SystemExit):
            load_config(_write_config(tmp_path, assets))

    def test_unsupported_persona_impl_exits(self, tmp_path, assets):
        cfg = _write_config(tmp_path, assets)
        cfg.write_text(
            cfg.read_text(encoding="utf-8").replace("impl: file", "impl: inline"),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            load_config(cfg)

    def test_missing_voices_section_exits(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(
            "blocks:\n  vad: {}\n  stt: {}\n  llm: {}\n  tts: {}\n"
            "  persona: {impl: file, path: /dev/null}\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            load_config(cfg)

    def test_missing_ref_audio_exits(self, tmp_path, assets):
        cfg = _write_config(tmp_path, assets)
        cfg.write_text(
            cfg.read_text(encoding="utf-8").replace(str(assets["wav"]), "/nonexistent/x.wav"),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            load_config(cfg)

    def test_repo_config_loads(self, repo_root):
        """仓库自带的 configs/alerter.yaml 必须能过加载（音色素材在位）。"""
        config = load_config(repo_root / "configs" / "alerter.yaml")
        assert set(config["blocks"]) == {"vad", "stt", "llm", "tts", "persona"}
        assert config["persona_text"]
        assert set(config["voices"]) == {"fengge"}
        assert config["blocks"]["llm"]["model"]


class TestResolveApiKey:
    def test_reads_env(self):
        assert resolve_api_key({"api_key_env": "K"}, {"K": "sk-1"}) == "sk-1"

    def test_fallback_llm_api_key(self):
        assert resolve_api_key({"api_key_env": "K"}, {"LLM_API_KEY": "sk-2"}) == "sk-2"

    def test_default_env_name(self):
        with pytest.raises(SystemExit):
            resolve_api_key({}, {})

    def test_missing_key_exits(self):
        with pytest.raises(SystemExit):
            resolve_api_key({"api_key_env": "NOPE"}, {})


class TestLoadDotenv:
    def test_basic_and_no_override(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text(
            '# 注释\nA=1\nB="two"\nC=\'three\'\n\nbad-line\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("A", "preset")
        for k in ("B", "C"):
            monkeypatch.delenv(k, raising=False)
        import os

        load_dotenv(env)
        assert os.environ["A"] == "preset"  # 已存在不覆盖
        assert os.environ["B"] == "two"
        assert os.environ["C"] == "three"

    def test_missing_file_noop(self, tmp_path):
        load_dotenv(tmp_path / "nope")  # 不抛异常
