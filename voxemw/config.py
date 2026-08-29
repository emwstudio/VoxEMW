"""配置加载：YAML（积木声明 + personas）+ .env.local + 人设/音色素材解析。

纯逻辑模块：不 import torch / aiohttp / speech_to_speech，可在 macOS 开发机直接单测。

配置结构（configs/assistant.yaml）：
- vad / stt / llm / tts：四块硬积木，原样保留给 launch/orchestrator 渲染
- personas：人设注册表（default + list: id -> personas/<id>.md）
  每个 persona 文件用 frontmatter 声明 name/ref_wav/ref_text/ref_image，
  正文为 LLM system prompt（女娲蒸馏产物）。
- server：s2s 内部端口、orchestrator 对外端口等
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

REQUIRED_BLOCKS = ("vad", "stt", "llm", "tts")


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


def parse_persona_file(path: Path) -> dict:
    """解析 personas/<id>.md：frontmatter（name/ref_wav/ref_text/voice_control）+ 正文。

    返回 {name, label, text, ref_wav, ref_text, voice_control}；无 frontmatter 时
    name 用文件名、素材字段为 None（由调用方决定缺素材是否报错）。label 为界面
    短名（角标用），缺省回退 name。voice_control 为 VoxCPM2 音色设计描述词
    （设计模式不需要 ref 素材）。ref_text 此处仍是文件路径，读文件内容替换
    发生在 load_config 里。
    """
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = raw
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            import yaml

            meta = yaml.safe_load(raw[3:end]) or {}
            if not isinstance(meta, dict):
                sys.exit(f"ERROR: {path} frontmatter 不是键值对")
            body = raw[end + 4 :]
    name = str(meta.get("name") or path.stem)
    return {
        "name": name,
        "label": str(meta.get("label") or name),
        "text": body.strip(),
        "ref_wav": meta.get("ref_wav"),
        "ref_text": meta.get("ref_text"),
        "voice_control": meta.get("voice_control"),
        "voice_seed": meta.get("voice_seed"),
    }


def load_config(path: Path) -> dict:
    """读 YAML 配置，解析 personas 人设文件与音色素材。

    - vad/stt/llm/tts 积木段原样保留
    - personas.list 每个 id 指向 persona md 文件：解析 frontmatter，
      ref_wav/ref_text（克隆素材，有则校验存在并读入文本）或
      voice_control（VoxCPM2 音色设计描述词）二者至少其一
    """
    try:
        import yaml
    except ImportError:
        sys.exit("ERROR: 需要 pyyaml（pip install pyyaml）")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        sys.exit(f"ERROR: 配置文件为空或格式不对: {path}")

    for name in REQUIRED_BLOCKS:
        if name not in config or not isinstance(config[name], dict):
            sys.exit(f"ERROR: 配置缺少 {name} 积木声明: {path}")

    personas_cfg = config.get("personas") or {}
    persona_files = personas_cfg.get("list") or {}
    if not persona_files:
        sys.exit(f"ERROR: 配置缺少 personas.list 人设注册表: {path}")
    default_persona = personas_cfg.get("default") or next(iter(persona_files))
    if default_persona not in persona_files:
        sys.exit(f"ERROR: personas.default={default_persona!r} 不在 personas.list 里")

    resolved: dict[str, dict] = {}
    for pid, file in persona_files.items():
        p = _resolve_path(str(file))
        if not p.is_file():
            sys.exit(f"ERROR: personas.{pid} 人设文件不存在: {p}")
        persona = parse_persona_file(p)

        ref_wav = persona.get("ref_wav")
        if ref_wav:
            wav_path = _resolve_path(str(ref_wav))
            if not wav_path.is_file():
                sys.exit(f"ERROR: personas.{pid}.ref_wav 文件不存在: {wav_path}")
            persona["ref_wav"] = str(wav_path)

            ref_text = persona.get("ref_text")
            if not ref_text:
                sys.exit(f"ERROR: personas.{pid}（{p}）frontmatter 缺少 ref_text（TTS 克隆需要逐字台词）")
            text_path = _resolve_path(str(ref_text))
            if not text_path.is_file():
                sys.exit(f"ERROR: personas.{pid}.ref_text 文件不存在: {text_path}")
            persona["ref_text"] = text_path.read_text(encoding="utf-8").strip()
        elif not persona.get("voice_control"):
            # 无克隆素材时必须有 voice_control 描述词（VoxCPM2 音色设计模式）
            sys.exit(f"ERROR: personas.{pid}（{p}）frontmatter 需要 ref_wav（克隆）或 voice_control（音色设计）")

        resolved[pid] = persona

    config["personas"]["default"] = default_persona
    config["personas"]["resolved"] = resolved
    return config


def resolve_api_key(llm_cfg: dict, env: dict | None = None) -> str:
    """按 llm.api_key_env 从环境变量取 API key，兜底 LLM_API_KEY，缺失即退出。"""
    if env is None:
        env = os.environ
    key_env = llm_cfg.get("api_key_env", "LLM_API_KEY")
    api_key = env.get(key_env) or env.get("LLM_API_KEY")
    if not api_key:
        sys.exit(f"ERROR: 环境变量 {key_env}（或兜底 LLM_API_KEY）未设置，无法提供 LLM API key")
    return api_key
