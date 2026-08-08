"""RAG LLM 积木：chat-completions-rag 后端（knowledge 积木的查询侧）。

包在上游 ChatCompletionsApiModelHandler 外面：每轮在 active_chat 末尾追加一条
「参考资料」用户消息，再照常走 DeepSeek。放在 pipeline 进程（LLM handler）而不是
orchestrator 注入的原因：VAD 自动响应的竞态——orchestrator 看到转写完成事件时
VAD 触发的 LLM 请求往往已经发出，注入赶不上。

注入语义（对齐官方/上游 Chat 写回机制，已逐行对账）：
- 注入到 active_chat（Chat.copy() 的浅拷贝）——只有 assistant 项会回写
  original_chat（base._generate 的 state.pending 循环），追加的用户消息
  不进长期历史、不进转写显示
- 相似度低于阈值整组不注入（闲聊不带资料）
- 嵌入/检索在 LLM 线程内同步执行：bge-m3 GPU 上单句 ~10-20ms，可忽略
"""

from __future__ import annotations

import logging

from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
)

logger = logging.getLogger(__name__)


def last_user_text(active_chat) -> str | None:
    """active_chat.buffer 里最后一条用户消息的文本（input_text 部分拼接）。"""
    for item in reversed(active_chat.buffer):
        if getattr(item, "role", None) != "user":
            continue
        parts = [
            getattr(c, "text", "") for c in (item.content or [])
            if getattr(c, "type", "") == "input_text"
        ]
        text = "".join(parts).strip()
        if text:
            return text
    return None


class RagChatCompletionsHandler(ChatCompletionsApiModelHandler):
    """chat-completions + 知识库检索注入。knowledge=None 时等同上游行为。"""

    def setup(self, *args, knowledge: dict | None = None, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self._ks = None
        self._top_k = 3
        self._threshold = 0.35
        if knowledge and knowledge.get("enabled"):
            from voxemw.knowledge import KnowledgeStore

            self._ks = KnowledgeStore(knowledge.get("db_path", "data/knowledge.db"))
            self._top_k = int(knowledge.get("top_k", 3))
            self._threshold = float(knowledge.get("threshold", 0.35))
            logger.info(
                "RAG 知识库启用: %s（top_k=%d threshold=%.2f）",
                knowledge.get("db_path"), self._top_k, self._threshold,
            )

    def _generate(self, active_chat, original_chat, turn, optional_kwargs):
        if self._ks is not None:
            try:
                query = last_user_text(active_chat)
                if query:
                    hits = self._ks.search_text(
                        query, top_k=self._top_k, threshold=self._threshold
                    )
                    if hits:
                        from voxemw.knowledge import build_rag_message

                        active_chat.add_item(
                            make_user_message(build_rag_message([h[0] for h in hits]))
                        )
                        logger.info(
                            "RAG 命中 %d 条（最高 %.3f，来自 %s）: %s",
                            len(hits), hits[0][1], hits[0][2], query[:30],
                        )
                    else:
                        logger.info("RAG 未命中（低于阈值 %.2f）: %s", self._threshold, query[:30])
            except Exception:
                # 检索/嵌入故障不阻塞对话：记日志，按无知识继续
                logger.exception("RAG 检索失败，按无知识继续")
        return super()._generate(active_chat, original_chat, turn, optional_kwargs)
