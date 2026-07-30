"""VoxEMW —— 数字人实时语音聊天助手。

基于 huggingface/speech-to-speech 的 VAD → STT → LLM → TTS 实时语音管线
（pip 依赖 + 运行时注册自定义积木），外加第五块积木 avatar
（SoulX-FlashHead 数字人形象，参考肖像 + 音频驱动）。

- voxemw.config：YAML 积木配置与人设/素材解析（纯逻辑）
- voxemw.pipeline：s2s 自定义积木（qwen3asr STT / voxcpm TTS）与启动器
- voxemw.avatar：数字人服务（FlashHead）与浏览器编排入口（orchestrator）
"""

__version__ = "0.5.0"
