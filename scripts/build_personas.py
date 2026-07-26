#!/usr/bin/env python3
"""扫描 personas/*.md，生成 web/personas.json（前端角色列表）。

每个 persona 文件 = YAML frontmatter（name/ref_wav/ref_text）+ Markdown 正文。
正文全文 + 末尾追加的「语音闲聊硬规则」一起作为 realtime session.update 的
instructions 注入（说话方式）；硬规则放在最后是因为 LLM 对末尾指令最敏感，
用来压住人设长文带来的长篇大论倾向。
音色跟随人设热切换：前端 session.update 的 audio.output.voice 用 persona id，
对应配置 tts.voices 里同名的 key（ref_wav/ref_text frontmatter 只是元信息，
实际音色配置在 configs/*.yaml 的 tts.voices）。

用法：
    python scripts/build_personas.py            # 写 web/personas.json
    python scripts/build_personas.py --check    # 只校验不写文件
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = REPO_ROOT / "personas"
OUTPUT_PATH = REPO_ROOT / "web" / "personas.json"

# 追加在每个人设末尾的硬规则：语音闲聊要短，长人设文档容易把模型带成长篇输出
VOICE_CHAT_HARD_RULES = """

# 语音闲聊硬规则（优先级高于以上所有设定）

- 这是实时语音聊天，不是写文章：每次回复最多两句口语短句，加起来不超过40个字
- 严禁长篇大论、分点论述、解释背景；有话留到对方追问再往下说
- 标点是语音停顿的依据：按正常说话习惯用逗号、句号即可，专有名词和固定词组中间绝不能加标点，不许为了凑停顿乱加逗号
- 用角色的口头禅或梗自然收尾，不说教、不总结陈词
"""


def parse_persona(path: Path) -> Dict[str, str]:
    """解析单个 persona 文件：frontmatter + 正文。

    frontmatter 只支持 `key: value` 单行（人设文件不需要更复杂的 YAML），
    有意不引 pyyaml，保持这个脚本零依赖、可在任何环境直接跑。
    """
    text = path.read_text(encoding="utf-8")
    meta: Dict[str, str] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end == -1:
            raise ValueError(f"{path.name}: frontmatter 未闭合（缺第二个 ---）")
        for line in text[3:end].strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"{path.name}: frontmatter 行格式错误: {line!r}")
            meta[key.strip()] = value.strip().strip("'\"")
        body = text[end + len("\n---"):].strip()

    if not meta.get("name"):
        raise ValueError(f"{path.name}: frontmatter 缺 name")
    if not body:
        raise ValueError(f"{path.name}: 正文（instructions）为空")

    return {
        "id": path.stem,
        "name": meta["name"],
        "instructions": body + VOICE_CHAT_HARD_RULES,
        "ref_wav": meta.get("ref_wav", ""),
        "ref_text": meta.get("ref_text", ""),
    }


def build(personas_dir: Path = PERSONAS_DIR) -> List[Dict[str, str]]:
    """扫描目录，返回 persona 列表（按文件名排序）。"""
    files = sorted(personas_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"persona 目录为空: {personas_dir}")
    return [parse_persona(f) for f in files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验 persona 文件，不写 personas.json")
    args = parser.parse_args()

    try:
        personas = build()
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"ERROR: {e}")

    names = "、".join(p["name"] for p in personas)
    if args.check:
        print(f"OK: {len(personas)} 个角色（{names}）")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(personas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {OUTPUT_PATH.relative_to(REPO_ROOT)}：{len(personas)} 个角色（{names}）")


if __name__ == "__main__":
    main()
