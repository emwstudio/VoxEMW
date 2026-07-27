"""配置加载：YAML（blocks 五积木）+ .env.local + 音色素材/人设文件解析。

纯逻辑模块：不 import torch / aiohttp，可在 macOS 开发机直接单测。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """最小 .env 加载（KEY=VALUE，# 注释，可选引号）。已存在的环境变量不覆盖。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_config(path: Path) -> dict:
    """读 YAML 配置，解析 blocks.persona 人设文件与 voices 音色素材。

    - blocks：五积木声明（vad/stt/llm/tts/persona），原样保留
    - persona.impl=file：path 相对仓库根解析，读入文本存 config["persona_text"]
    - voices.ref_audio：相对仓库根解析，必须存在
    - voices.ref_text：指向 txt 文件路径，读入内容替换为文本本身（VoxCPM 要的是文本）
    """
    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: 需要 pyyaml（pip install pyyaml）")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        sys.exit(f"ERROR: 配置文件为空或格式不对: {path}")

    blocks = config.get("blocks") or {}
    for name in ("vad", "stt", "llm", "tts", "persona"):
        if name not in blocks:
            sys.exit(f"ERROR: 配置缺少 blocks.{name} 积木声明: {path}")
    config["blocks"] = blocks

    persona = blocks.get("persona") or {}
    if persona.get("impl") == "file":
        persona_path = persona.get("path")
        if not persona_path:
            sys.exit("ERROR: blocks.persona.impl=file 但缺少 path")
        p = _resolve_path(persona_path)
        if not p.is_file():
            sys.exit(f"ERROR: blocks.persona.path 人设文件不存在: {p}")
        config["persona_text"] = p.read_text(encoding="utf-8").strip()
    else:
        sys.exit(f"ERROR: 不支持的 blocks.persona.impl: {persona.get('impl')!r}（目前只支持 file）")

    voices = config.get("voices") or {}
    if not voices:
        sys.exit(f"ERROR: 配置缺少 voices 音色映射: {path}")
    for name, spec in voices.items():
        spec = spec or {}
        ref_audio = spec.get("ref_audio")
        if not ref_audio:
            sys.exit(f"ERROR: voices.{name}.ref_audio 缺失")
        audio_path = _resolve_path(ref_audio)
        if not audio_path.is_file():
            sys.exit(f"ERROR: voices.{name}.ref_audio 文件不存在: {audio_path}")
        spec["ref_audio"] = str(audio_path)
        ref_text = spec.get("ref_text")
        if not ref_text:
            sys.exit(f"ERROR: voices.{name}.ref_text 缺失（Ultimate Cloning 需要逐字台词）")
        text_path = _resolve_path(ref_text)
        if not text_path.is_file():
            sys.exit(f"ERROR: voices.{name}.ref_text 文件不存在: {text_path}")
        spec["ref_text"] = text_path.read_text(encoding="utf-8").strip()
        voices[name] = spec
    config["voices"] = voices
    return config


def resolve_api_key(llm_cfg: dict, env: dict | None = None) -> str:
    """按 llm.api_key_env 从环境变量取 API key，兜底 LLM_API_KEY，缺失即退出。"""
    if env is None:
        env = os.environ
    key_env = llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY")
    api_key = env.get(key_env) or env.get("LLM_API_KEY")
    if not api_key:
        sys.exit(f"ERROR: 环境变量 {key_env}（或兜底 LLM_API_KEY）未设置，无法提供 LLM API key")
    return api_key
