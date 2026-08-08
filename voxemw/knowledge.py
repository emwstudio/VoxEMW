"""知识库积木：PDF 文档切块 + bge-m3 embedding + SQLite 存储 + 余弦检索。

第八块积木。两个进程共用 data/knowledge.db：
- orchestrator（CPU）写入侧：管理页面上传 PDF → 解析/切块/嵌入/入库
- pipeline（GPU）查询侧：llm_rag handler 每轮用户转写 → 检索 → 注入

选型（2026-08-05 定案）：SQLite + numpy 暴力余弦。不用 qdrant——其本地文件模式
有单进程锁，写读双进程打架；文档块量级几千条，numpy 全扫毫秒级。
查询侧按 db mtime 懒重载内存矩阵（写入侧更新后下一轮自动生效）。

嵌入：bge-m3（sentence-transformers，1024 维，normalize_embeddings=True →
点积即余弦）。模型加载较重（~2GB），两进程各自懒加载单例。
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDER_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

_CHUNK_SIZE = 300    # 每块目标字数
_CHUNK_OVERLAP = 50  # 相邻块重叠字数

_model = None
_model_lock = threading.Lock()


def get_embedder():
    """懒加载 bge-m3 单例（进程内共享）。离线环境需 HF_HUB_OFFLINE=1 + HF_HOME。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("加载 embedding 模型 %s（%s）", EMBEDDER_MODEL, device)
                _model = SentenceTransformer(EMBEDDER_MODEL, device=device)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """文本列表 → (N, 1024) float32 归一化向量。"""
    vecs = get_embedder().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype=np.float32)


def chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """全文 → 重叠滑窗切块。优先在段落/句号边界断开，超长段硬切。"""
    text = " ".join(text.split())
    if not text:
        return []
    # 先按段落/句末标点粗分
    import re

    pieces = re.split(r"(?<=[。！？!?])\s*", text)
    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if not piece:
            continue
        if len(cur) + len(piece) <= size:
            cur += piece
            continue
        if cur:
            chunks.append(cur)
        # 单句超长：硬切成 size 的段
        while len(piece) > size:
            chunks.append(piece[:size])
            piece = piece[size - overlap :]
        cur = piece
    if cur:
        chunks.append(cur)
    # 重叠：给非首块前缀上一块尾部
    out = []
    for i, c in enumerate(chunks):
        if i > 0 and overlap > 0:
            c = chunks[i - 1][-overlap:] + c
        out.append(c)
    return out


def build_rag_message(contexts: list[str]) -> str:
    """检索命中 → 注入 active_chat 的追加用户消息（纯函数，便于单测）。
    追加而非改写用户原话：不污染 chat 历史（original_chat 只回写 assistant 项）。"""
    body = "\n".join(f"- {c}" for c in contexts)
    return (
        f"（参考资料：\n{body}\n）"
        "请基于上面的资料用口语化回答我的问题；资料里没说的就照你平常的样子回答，别硬编。"
    )


class KnowledgeStore:
    """SQLite 存储 + 内存向量矩阵检索。读侧按 mtime 懒重载。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS docs ("
                " doc_name TEXT PRIMARY KEY, n_chunks INTEGER, created_at TEXT)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS chunks ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, doc_name TEXT, chunk_idx INTEGER,"
                " text TEXT, embedding BLOB)"
            )
        self._matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
        self._meta: list[tuple[str, str]] = []  # (doc_name, text) 与 matrix 行对齐
        self._mtime = 0.0
        self.reload()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10)

    # ── 写侧（orchestrator 入库线程）──

    def upsert_document(self, doc_name: str, texts: list[str], vectors: np.ndarray) -> int:
        """整篇重灌（同名先删后插）。vectors 须 (N, 1024) float32 已归一。"""
        assert len(texts) == len(vectors), "文本块数与向量数不一致"
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
            c.execute(
                "INSERT OR REPLACE INTO docs (doc_name, n_chunks, created_at) VALUES (?,?,?)",
                (doc_name, len(texts), now),
            )
            c.executemany(
                "INSERT INTO chunks (doc_name, chunk_idx, text, embedding) VALUES (?,?,?,?)",
                [
                    (doc_name, i, t, vectors[i].astype(np.float32).tobytes())
                    for i, t in enumerate(texts)
                ],
            )
        logger.info("知识库入库: %s（%d 块）", doc_name, len(texts))
        self.reload()  # 写后刷新内存矩阵（本进程查询立即可见）
        return len(texts)

    def delete_document(self, doc_name: str) -> bool:
        with self._conn() as c:
            n = c.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,)).rowcount
            c.execute("DELETE FROM docs WHERE doc_name = ?", (doc_name,))
        self.reload()
        return n > 0

    def list_documents(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT doc_name, n_chunks, created_at FROM docs ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"doc_name": r[0], "chunks": r[1], "created_at": r[2]} for r in rows
        ]

    # ── 读侧（pipeline 查询）──

    def reload(self) -> None:
        """全量读入内存矩阵（几千块毫秒级）。"""
        with self._conn() as c:
            rows = c.execute("SELECT doc_name, text, embedding FROM chunks ORDER BY id").fetchall()
        if rows:
            self._matrix = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32).reshape(
                len(rows), EMBED_DIM
            ).copy()
            self._meta = [(r[0], r[1]) for r in rows]
        else:
            self._matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
            self._meta = []
        try:
            self._mtime = os.path.getmtime(self.db_path)
        except OSError:
            self._mtime = 0.0

    def maybe_reload(self) -> None:
        try:
            m = os.path.getmtime(self.db_path)
        except OSError:
            return
        if m != self._mtime:
            self.reload()

    def search(self, query_vec: np.ndarray, top_k: int = 3,
               threshold: float = 0.35) -> list[tuple[str, float, str]]:
        """余弦检索 → [(text, score, doc_name)]，低于阈值整组丢弃（闲聊不插知识）。"""
        if len(self._matrix) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        scores = self._matrix @ q  # 已归一化 → 点积即余弦
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        if scores[idx[0]] < threshold:
            return []
        return [
            (self._meta[i][1], float(scores[i]), self._meta[i][0])
            for i in idx
            if scores[i] >= threshold
        ]

    def search_text(self, query: str, top_k: int = 3,
                    threshold: float = 0.35) -> list[tuple[str, float, str]]:
        self.maybe_reload()
        return self.search(embed_texts([query])[0], top_k=top_k, threshold=threshold)


def parse_pdf(fileobj_or_path) -> str:
    """PDF → 纯文本。优先 PyMuPDF（对不规范 PDF 健壮得多），回退 pypdf。"""
    try:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            import pymupdf as fitz
        if isinstance(fileobj_or_path, (str, Path)):
            doc = fitz.open(fileobj_or_path)
        else:
            doc = fitz.open(stream=fileobj_or_path.read(), filetype="pdf")
        text = "\n\n".join(
            t for t in ((page.get_text() or "").strip() for page in doc) if t
        )
        if text:
            return text
    except Exception as e:
        logger.info("PyMuPDF 解析失败，回退 pypdf: %s", e)

    from pypdf import PdfReader

    reader = PdfReader(fileobj_or_path)
    parts = []
    for page in reader.pages:
        t = (page.extract_text() or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _kimi_transcribe(jpeg_bytes: bytes, api_key: str, model: str, base_url: str) -> str:
    """单页图片 → Kimi 多模态转写文字（kimi-k3 是 thinking 模型，max_tokens 给足）。"""
    import base64
    import json
    import urllib.request

    b64 = base64.b64encode(jpeg_bytes).decode()
    payload = {
        "model": model,
        "max_tokens": 2000,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",
                 "text": "这是扫描文档的一页。请完整转写其中的文字内容，保持原有顺序和结构，"
                         "不要遗漏，不要评价，不要输出任何额外说明。"},
            ],
        }],
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return (data["choices"][0]["message"]["content"] or "").strip()


def parse_pdf_smart(fileobj_or_path, api_key: str = "", vision_model: str = "kimi-k3",
                    vision_base_url: str = "https://api.moonshot.cn/v1",
                    on_progress=None) -> str:
    """PDF → 纯文本。优先文字层抽取（PyMuPDF/pypdf）；失败或文字层缺失（扫描件）
    逐页转图喂 Kimi 视觉转写。on_progress(msg) 可选进度回调。
    无 api_key 时扫描件返回空。"""
    try:
        text = parse_pdf(fileobj_or_path)
    except Exception as e:
        logger.info("文字层解析失败（%s），尝试视觉转写", e)
        text = ""
    if len(text) >= 50 or not api_key:
        return text

    # 扫描件兜底：PyMuPDF 逐页渲染 → Kimi 视觉转写
    try:
        import fitz  # PyMuPDF（新版也叫 pymupdf）
    except ImportError:
        import pymupdf as fitz

    logger.info("PDF 无文字层，走 Kimi 视觉转写: %s", fileobj_or_path)
    doc = (fitz.open(fileobj_or_path) if isinstance(fileobj_or_path, (str, Path))
           else fitz.open(stream=fileobj_or_path.read(), filetype="pdf"))
    parts = []
    for i, page in enumerate(doc):
        if on_progress:
            on_progress(f"视觉转写 {i + 1}/{len(doc)} 页…")
        pix = page.get_pixmap(dpi=150)
        try:
            parts.append(_kimi_transcribe(pix.tobytes("jpeg"), api_key,
                                          vision_model, vision_base_url))
        except Exception as e:
            logger.warning("第 %d 页转写失败（跳过）: %s", i + 1, e)
    return "\n\n".join(p for p in parts if p)
