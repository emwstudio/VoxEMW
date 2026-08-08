"""knowledge 积木纯逻辑单测：切块 / 存储检索 / 注入格式 / argv 渲染（无需 GPU/模型）。"""

import numpy as np
import pytest

from voxemw.knowledge import KnowledgeStore, build_rag_message, chunk_text
from voxemw.pipeline.args import render_s2s_argv


def _unit_vecs(n: int, seed: int = 0) -> np.ndarray:
    v = np.random.RandomState(seed).randn(n, 1024).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_chunk_text_respects_size_and_covers_all():
    text = "峰哥说车轱辘话。" * 200  # 1400 字，无标点长句混合
    chunks = chunk_text(text, size=300, overlap=50)
    assert all(len(c) <= 350 for c in chunks)  # size + overlap 前缀
    # 覆盖性：拼接后的文本应包含原文所有内容（重叠导致更长）
    assert sum(len(c) for c in chunks) >= len(text.replace(" ", ""))
    assert len(chunks) >= 4


def test_chunk_text_sentence_boundaries():
    text = "第一句在这里。第二句稍微长一点点，包含逗号。第三句。"
    chunks = chunk_text(text, size=20, overlap=5)
    assert chunks[0].startswith("第一句")
    assert any("第三句" in c for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_store_crud_and_search(tmp_path):
    ks = KnowledgeStore(tmp_path / "k.db")
    texts = ["X9 续航 702 公里", "今天天气晴朗", "峰哥喜欢爬山"]
    vecs = _unit_vecs(3)
    ks.upsert_document("a.pdf", texts, vecs)
    docs = ks.list_documents()
    assert len(docs) == 1 and docs[0]["doc_name"] == "a.pdf" and docs[0]["chunks"] == 3

    hits = ks.search(vecs[0], top_k=3, threshold=0.35)
    assert hits and hits[0][0] == "X9 续航 702 公里" and hits[0][1] > 0.99

    # 整组低于阈值 → 不命中（闲聊守门）
    junk = _unit_vecs(1, seed=99)[0]
    far = junk - vecs[0] * (junk @ vecs[0])  # 与库内向量正交化 → 低相似
    far /= np.linalg.norm(far)
    assert ks.search(far.astype(np.float32), top_k=3, threshold=0.35) == []

    assert ks.delete_document("a.pdf") is True
    assert ks.list_documents() == []
    assert ks.search(vecs[0]) == []
    assert ks.delete_document("a.pdf") is False


def test_store_upsert_replaces(tmp_path):
    ks = KnowledgeStore(tmp_path / "k.db")
    vecs = _unit_vecs(2)
    ks.upsert_document("a.pdf", ["旧块一", "旧块二"], vecs)
    ks.upsert_document("a.pdf", ["新块"], _unit_vecs(1, seed=7))
    docs = ks.list_documents()
    assert docs[0]["chunks"] == 1
    texts = [t for _, t in ks._meta]
    assert texts == ["新块"]


def test_build_rag_message_format():
    msg = build_rag_message(["资料甲", "资料乙"])
    assert "资料甲" in msg and "- 资料乙" in msg
    assert "参考资料" in msg and "口语" in msg


def test_rag_backend_renders_as_chat_completions():
    config = {
        "vad": {},
        "llm": {
            "backend": "chat-completions-rag",
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "TEST_LLM_KEY",
        },
        "server": {},
    }
    import os

    env = dict(os.environ, TEST_LLM_KEY="sk-test")
    argv = render_s2s_argv(config, env=env)
    i = argv.index("--llm_backend")
    assert argv[i + 1] == "chat-completions"  # 上游 CLI 只认原名，rag 由 launch 运行时注册
