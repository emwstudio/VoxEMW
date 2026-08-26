"""VoxEMW —— 实时语音聊天助手（语音 + VRM 数字人，河南妮儿人设）。

基于 huggingface/speech-to-speech 的 VAD → STT → LLM → TTS 实时语音管线
（pip 依赖 + 运行时注册自定义积木），外加浏览器网关
（orchestrator + WebRTC 音频轨 + 静态肖像）。

- voxemw.config：YAML 积木配置与人设/素材解析（纯逻辑）
- voxemw.pipeline：s2s 自定义积木（qwen3asr STT）与启动器
- voxemw.gateway：浏览器编排入口（orchestrator + WebRTC 音频轨）
"""

__version__ = "1.3.0"
