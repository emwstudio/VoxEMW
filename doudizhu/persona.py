"""personas/<id>.md 加载：frontmatter（name/ref_wav/ref_text）+ 人设正文。

与 scripts/build_personas.py 同一份素材，游戏服务直接读 md，不走 web/personas.json。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Persona:
    id: str          # 文件名（= 座位 id = 音色 id）
    name: str        # 显示名（大胃袋良子 / 峰哥亡命天涯）
    body: str        # 人设正文（作 system prompt）
    ref_wav: str
    ref_text: str


def load_persona(path: str | Path) -> Persona:
    text = Path(path).read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return Persona(
        id=Path(path).stem,
        name=meta.get("name", Path(path).stem),
        body=body.strip(),
        ref_wav=meta.get("ref_wav", ""),
        ref_text=meta.get("ref_text", ""),
    )
