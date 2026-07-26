#!/usr/bin/env python3
"""VoxEMW 启动器：读 YAML 积木配置 -> 渲染 CLI argv -> exec speech-to-speech。

用法：
    python launch.py [--config configs/autodl-4090.yaml] [--dry-run]

- 配置路径优先级：--config > 环境变量 S2S_CONFIG > configs/autodl-4090.yaml
- 启动前自动加载仓库根 .env.local（已存在的环境变量优先），供 llm.api_key_env 解析
- tts.ref_text 若指向存在的 txt 文件，读文件内容作为 CLI 参数值
- 解释器优先级：server.python > 仓库根 .venv/bin/python > vendor venv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from voxemw.backends import render_argv  # noqa: E402

DEFAULT_CONFIG = "configs/autodl-4090.yaml"
VENDOR_VENV_PYTHON = "vendor/speech-to-speech/.venv/bin/python"


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


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: 需要 pyyaml（pip install pyyaml）")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        sys.exit(f"ERROR: 配置文件为空或格式不对: {path}")
    return config


def resolve_file_valued_params(config: dict) -> None:
    """tts.ref_text / tts.ref_audio：相对仓库根解析；ref_text 若是存在的
    txt 文件则读入内容（OmniVoice/Qwen3-TTS 的 CLI 要的是文本本身）。
    tts.voices（可热切换音色映射）每个音色同样处理后 json.dumps 成单个
    CLI 参数值（--omnivoice_tts_voices）。"""
    tts = config.get("tts") or {}
    for key in ("ref_audio", "ref_text"):
        value = tts.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if key == "ref_text" and path.is_file():
            tts[key] = path.read_text(encoding="utf-8").strip()
        elif key == "ref_audio":
            if not path.is_file():
                sys.exit(f"ERROR: tts.ref_audio 文件不存在: {path}")
            tts[key] = str(path)

    voices = tts.get("voices")
    if voices:
        resolved = {}
        for name, spec in voices.items():
            spec = dict(spec or {})
            ref_audio = spec.get("ref_audio")
            if not ref_audio:
                sys.exit(f"ERROR: tts.voices.{name}.ref_audio 缺失")
            audio_path = Path(ref_audio)
            if not audio_path.is_absolute():
                audio_path = REPO_ROOT / audio_path
            if not audio_path.is_file():
                sys.exit(f"ERROR: tts.voices.{name}.ref_audio 文件不存在: {audio_path}")
            spec["ref_audio"] = str(audio_path)
            ref_text = spec.get("ref_text")
            if ref_text:
                text_path = Path(ref_text)
                if not text_path.is_absolute():
                    text_path = REPO_ROOT / text_path
                if text_path.is_file():
                    spec["ref_text"] = text_path.read_text(encoding="utf-8").strip()
            resolved[name] = spec
        tts["voices"] = json.dumps(resolved, ensure_ascii=False)


def check_persona(config: dict) -> None:
    persona = config.get("persona") or {}
    persona_file = persona.get("file")
    if persona_file and not (REPO_ROOT / persona_file).is_file():
        sys.exit(f"ERROR: persona.file 不存在: {persona_file}（人设源在 personas/ 目录）")


def pick_python(config: dict) -> str:
    server = config.get("server") or {}
    candidates = [
        server.get("python"),
        ".venv/bin/python",
        VENDOR_VENV_PYTHON,
    ]
    for cand in candidates:
        if cand and (REPO_ROOT / cand).is_file():
            return str(REPO_ROOT / cand)
    sys.exit(
        "ERROR: 找不到运行管线的 python。请先在 GPU 机器上执行 scripts/autodl_setup.sh，"
        "或在配置 server.python 里指定解释器路径。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 搭积木语音助手启动器")
    parser.add_argument(
        "--config",
        default=os.environ.get("S2S_CONFIG", DEFAULT_CONFIG),
        help="YAML 配置路径（默认 %(default)s，可用 S2S_CONFIG 覆盖）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印渲染出的 argv，不启动管线",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        sys.exit(f"ERROR: 配置不存在: {config_path}")

    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(config_path)
    resolve_file_valued_params(config)
    check_persona(config)

    try:
        argv = render_argv(config, env=os.environ)
    except (KeyError, ValueError) as e:
        sys.exit(f"ERROR: 配置渲染失败: {e}")

    if args.dry_run:
        pairs = [f"{argv[i]} {argv[i + 1]}" for i in range(0, len(argv) - 1, 2)]
        print("speech-to-speech \\")
        print("  " + " \\\n  ".join(pairs))
        return

    python = pick_python(config)
    cmd = [python, "-m", "speech_to_speech.s2s_pipeline"] + argv
    print(f"==> 启动 speech-to-speech（{len(argv)} 个参数）")
    print(f"    python: {python}")
    print("    argv: " + " ".join(argv))
    os.execvp(python, cmd)


if __name__ == "__main__":
    main()
