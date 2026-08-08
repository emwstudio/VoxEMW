"""记忆积木：Mem0 封装（第七块积木，orchestrator 侧 CPU 进程）。

- 召回：会话建立时把相关记忆追加到 persona instructions（一次性，不进语音回合）
- 写入：response.done 后异步抽取（Mem0 内部调 LLM 做事实抽取/去重，不占语音延迟）
- 降级：enabled=false / 依赖缺失 / 初始化失败 → 静默跳过，对话链路零影响

选型（2026-08 调研）：Mem0 Python SDK 内嵌模式——LLM 抽取走 DeepSeek
（OpenAI 兼容），embedding 走本地 bge-m3（DeepSeek 无 embedding API），
向量库用内嵌 qdrant（文件落盘）。无需新服务/数据库进程。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_MEMORY_CHARS = 80  # 单条注入截断（防人设正文被稀释）

# 自定义抽取规则：只记用户（非助手）的事实/偏好/计划。
# 默认 prompt 不区分说话人，会把助手侃大山的内容也记进去（实测踩过）。
# 注意：mem0 把它作为 custom_instructions 拼进系统 prompt（规则层最高优先级），
# 输出格式由系统 prompt 定为 {"memory": [{"text": ...}]} ——这里绝不能另写输出格式，
# 否则模型按我们的格式输出、mem0 解析 get("memory") 为空，静默不落库（实测踩过）。
_EXTRACT_PROMPT = """抽取规则（最高优先级，与上面默认规则冲突时以本规则为准）：
- 只记录用户（user 角色）透露的事实、偏好、习惯、计划、个人信息
- 绝不记录助手（assistant 角色）的任何内容：它的故事、观点、推荐、玩笑都不是用户记忆
- 只记有长期价值的信息；寒暄、一次性话题、纯提问本身不记
- 没有可记的内容就输出空的 memory 列表
- 事实用中文，简短具体（如「用户在减脂」「用户做短视频」）"""


def build_memory_block(memories: list[str]) -> str:
    """记忆条目 → 注入 instructions 的文本块（纯函数，便于单测）。"""
    if not memories:
        return ""
    lines = [f"- {m[:MAX_MEMORY_CHARS]}" for m in memories]
    return "# 关于用户的记忆（历史对话提取，自然引用，不要逐条复述）\n" + "\n".join(lines)


class MemoryStore:
    """Mem0 懒加载封装。所有方法同步阻塞，调用方用 asyncio.to_thread。"""

    def __init__(self, cfg: dict, llm_cfg: dict, api_key: str):
        from mem0 import Memory  # 依赖重，仅启用时加载

        store_dir = Path(cfg.get("store_dir", "data/memory"))
        store_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = cfg.get("user_id", "default_user")
        self.top_k = int(cfg.get("top_k", 5))
        self._m = Memory.from_config({
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_cfg.get("model_name", "deepseek-v4-flash"),
                    "openai_base_url": llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
                    "api_key": api_key,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": cfg.get("embedder_model", "BAAI/bge-m3")},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "voxemw_memory",
                    "path": str(store_dir),
                    "embedding_model_dims": 1024,  # bge-m3
                },
            },
        })

    def search(self, agent_id: str) -> list[str]:
        """召回该 persona 的记忆条目（user_id 固定单用户，agent_id=persona 隔离）。
        mem0 2.x：实体参数必须走 filters，条数参数名 top_k。"""
        results = self._m.search(
            "用户的重要事实、偏好、约定与近期事件",
            top_k=self.top_k,
            filters={"user_id": self.user_id, "agent_id": agent_id},
        )
        return [r["memory"] for r in (results.get("results") or []) if r.get("memory")]

    def add_turn(self, user_text: str, assistant_text: str, agent_id: str) -> None:
        """一轮对话 → Mem0 抽取/去重/更新（内部调 LLM，阻塞）。
        自定义抽取 prompt：只记用户侧事实，忽略助手输出。"""
        messages = []
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        if messages:
            self._m.add(messages, user_id=self.user_id, agent_id=agent_id,
                        prompt=_EXTRACT_PROMPT)


def create_memory_store(config: dict) -> MemoryStore | None:
    """从全局配置构建记忆仓库；未启用/依赖缺失/初始化失败 → None（静默降级）。"""
    cfg = config.get("memory") or {}
    if not cfg.get("enabled", False):
        return None
    import os

    llm_cfg = config.get("llm") or {}
    api_key = os.environ.get(llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not api_key:
        logger.warning("memory 启用但缺 LLM api_key，记忆关闭")
        return None
    try:
        store = MemoryStore(cfg, llm_cfg, api_key)
        logger.info("记忆积木就绪（Mem0，store=%s）", cfg.get("store_dir", "data/memory"))
        return store
    except Exception as e:
        logger.warning("记忆积木初始化失败，降级关闭: %s", e)
        return None
