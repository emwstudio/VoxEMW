"""VoxEMW —— 基于 huggingface/speech-to-speech 的搭积木语音助手。

本包提供后端注册表（backends.py）：把 configs/*.yaml 里的积木配置
渲染成 speech-to-speech 管线的 CLI argv，由仓库根的 launch.py 启动。
"""

__version__ = "0.2.0"
