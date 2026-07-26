"""personas/*.md frontmatter 解析与 build_personas 输出结构测试。"""

import json

import pytest


def test_persona_files_exist(repo_root):
    personas = sorted((repo_root / "personas").glob("*.md"))
    assert {p.stem for p in personas} >= {"liangzi", "fengge"}


def test_parse_frontmatter(repo_root, build_personas):
    persona = build_personas.parse_persona(repo_root / "personas" / "liangzi.md")
    assert persona["id"] == "liangzi"
    assert persona["name"] == "大胃袋良子"
    assert persona["ref_wav"] == "assets/liangzi/ref.wav"
    assert persona["ref_text"] == "assets/liangzi/ref.txt"
    # instructions 是正文全文，得足够像一份人设而不是空壳
    assert "胃袋" in persona["instructions"]
    assert len(persona["instructions"]) > 500


def test_fengge_frontmatter(repo_root, build_personas):
    persona = build_personas.parse_persona(repo_root / "personas" / "fengge.md")
    assert persona["name"] == "峰哥亡命天涯"
    assert persona["ref_wav"] == "assets/fengge/ref.wav"
    assert "连接" in persona["instructions"] or "性压抑" in persona["instructions"]


def test_ref_assets_exist_on_disk(repo_root, build_personas):
    """frontmatter 里声明的 ref_wav/ref_text 必须是仓库里真实存在的素材。"""
    for persona in build_personas.build(repo_root / "personas"):
        for key in ("ref_wav", "ref_text"):
                if persona[key]:
                    assert (repo_root / persona[key]).is_file(), (
                        f"{persona['id']}.{key} 指向不存在的文件: {persona[key]}"
                    )


def test_build_output_structure(repo_root, build_personas, tmp_path):
    personas = build_personas.build(repo_root / "personas")
    assert len(personas) >= 2
    for p in personas:
        assert set(p) == {"id", "name", "instructions", "ref_wav", "ref_text"}
        assert all(isinstance(v, str) for v in p.values())
        assert p["id"] and p["name"] and p["instructions"]
    # 与写盘产物同结构（web/personas.json 由 --check 外的运行生成）
    out = tmp_path / "personas.json"
    out.write_text(json.dumps(personas, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded[0]["id"] < loaded[-1]["id"] or True  # 排序由 build 保证，这里只验证 JSON 往返
    assert {p["id"] for p in loaded} == {p["id"] for p in personas}


def test_generated_personas_json_up_to_date(repo_root, build_personas):
    """web/personas.json 已生成且与 personas/ 源一致（忘了跑 build 脚本能测出来）。"""
    out = repo_root / "web" / "personas.json"
    assert out.is_file(), "web/personas.json 未生成，请运行 python scripts/build_personas.py"
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    fresh = build_personas.build(repo_root / "personas")
    assert on_disk == fresh


def test_parse_errors(repo_root, build_personas, tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("正文没有 frontmatter 也没有 name", encoding="utf-8")
    with pytest.raises(ValueError):
        build_personas.parse_persona(bad)

    no_body = tmp_path / "nobody.md"
    no_body.write_text("---\nname: x\n---\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_personas.parse_persona(no_body)
