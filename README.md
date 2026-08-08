# VoxEMW · 数字人实时语音聊天助手

对着浏览器说话，屏幕里的数字人开口回答你。人设、音色、形象、记忆四位一体，
单卡 RTX 4090D 即可运行，**你说完到听到第一声 ≈ 2.4s**。

链路基于 [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
（VAD → STT → LLM → TTS 实时管线）+ [AVTR-1](https://github.com/avaturn-live/avtr-1)
数字人形象（TensorRT，0.2s 生成粒度）。音画下行走 **WebRTC**（VP8 + Opus），
浏览器按 RTP 时间戳原生对齐口型。人设来自女娲蒸馏（`personas/<id>.md`）。

## 架构

```
浏览器
  │  WebRTC 音画（VP8 720p25 + Opus 48k，TURN 走 coturn :3478）
  │  WS 控制/麦克风/转写（:8000）
  ▼
orchestrator（CPU，:8000）
  ├─→ s2s 语音管线（:8765）   VAD → STT SenseVoice → LLM DeepSeek → TTS VoxCPM2
  └─→ avatar 数字人服务（:8767） AVTR-1（TensorRT，说话 + 倾听双流，裸帧无压缩）
```

音画对齐：TTS 音频双写——喂 avatar 驱动口型的同一股流进 `AVSyncScheduler`
打时间戳（音频 20ms/帧 40ms 同一条单调钟，句尾零填充帧丢弃，打断 flush），
两路 RTP 轨到浏览器后原生同步。avatar 缺席自动降级纯语音模式。

## 延迟分解（4090D 实测）

| 环节 | 耗时 |
|---|---|
| VAD 判停 | ~0.5s |
| STT（SenseVoiceSmall） | ~0.1s |
| LLM 首句（DeepSeek v4-flash 流式） | ~1.4s |
| TTS 首音（VoxCPM2 流式） | ~0.1s |
| 唇同步（AVTR-1 起播 + 音频压后补偿） | ~0.45s |

## 八块积木（configs/assistant.yaml）

| 积木 | 选型 |
|---|---|
| vad | silero-vad（s2s 内置，判停 500ms） |
| stt | `iic/SenseVoiceSmall`（FunASR，非自回归；`qwen3asr` 备选） |
| llm | DeepSeek `deepseek-v4-flash`（流式逐句送 TTS；`chat-completions-rag` 变体带知识库检索） |
| tts | `openbmb/VoxCPM2`（克隆音色，`tts.rate` 变速补偿） |
| avatar | `avaturn-live/avtr-1`（TensorRT，0.2s/chunk） |
| persona | `personas/<id>.md`（女娲蒸馏，frontmatter 绑定音色三件套） |
| memory | Mem0 内嵌（LLM 抽取走 DeepSeek，只记用户侧事实；embedding 本地 bge-m3，qdrant 文件落盘） |
| knowledge | PDF 知识库 RAG：SQLite + numpy 暴力余弦，embedding 本地 bge-m3；管理页 `/knowledge` |

## 部署（AutoDL RTX 4090D）

```bash
cp .env.example .env.local   # 填 DEEPSEEK_API_KEY
bash scripts/autodl_setup.sh # 幂等：环境 + 模型
bash scripts/start_assistant.sh   # 起四进程：TURN + avatar + 管线 + orchestrator
```

本地（Mac）开隧道并打开页面：

```bash
bash scripts/tunnel.sh        # 转发 8000（网页）+ 3478（WebRTC 媒体），缺一不可
# 打开 http://localhost:8000（?solo=1 为 16:9 全屏录制模式）
```

启停：`bash scripts/start_assistant.sh [stop]`。TTS 与 avatar 首次启动各有数分钟预热。

**参考图规范（重要）**：16:9 横版胸像（1920×1080 为佳），脸宽占图宽 ~20%、
头顶留白 ~19%，微张嘴/露齿者口型最佳——构图不对脸型会失真。

## URL 调参

| 参数 | 作用 |
|---|---|
| `?debug=1` | 右下角 RTC 统计角标（到帧率/抖动/丢包） |
| `?alead=250` | 音频压后毫秒数（音画对齐补偿：嘴慢调大，嘴快调小） |
| `?vbr=2000` | 视频码率 kbps（默认 2000 最佳画质，弱网降到 800） |
| `?rtc=0` | 回退旧 WS 推流模式（丢帧保活，弱网兜底） |
| `?solo=1` | 单栏全屏，录制用 |

## 特性

- **长期记忆**：人设记得你说过的事（Mem0 抽取 + 会话开始注入，`memory.enabled` 开关）
- **知识库**：管理页 `/knowledge` 上传 PDF，问相关内容自动参考回答（`knowledge.enabled`
  + `llm.backend: chat-completions-rag`；相似度低于 `threshold` 的闲聊不注入资料）
- **倾听反应**：你说话时数字人对你的声音实时做出注视/微表情（AVTR-1 双流）
- **待机微动**：无语音时静音驱动眨眼/轻摇头，画面永远活着
- **人设切换**：前端点 chip，LLM 人设 / TTS 音色 / 数字人肖像三路同切
- **局域网设备**：`scripts/lan_https/proxy.py`（iPhone/iPad 用，需装 CA）

## 本地开发（macOS，无 GPU）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v   # 纯逻辑单测（含音画调度器）
```

## 目录

- `configs/assistant.yaml` — 唯一配置（八积木 + server + rtc）
- `configs/turnserver.conf` — coturn 配置（WebRTC 媒体中继）
- `voxemw/pipeline/` — s2s 集成（STT/TTS/RAG-LLM 积木 + 启动器）
- `voxemw/avatar/` — 数字人服务（`service.py` + `avtr1_engine.py`）+ orchestrator
  + WebRTC 轨（`rtc.py`）+ 音画调度器（`avsync.py`）
- `voxemw/memory.py` — 记忆积木（Mem0 封装）
- `voxemw/knowledge.py` — 知识库积木（SQLite 存储 + 切块 + bge-m3 嵌入 + 余弦检索）
- `web/` — 通话页 + 知识库管理页（无构建）
- `personas/` / `skills/` — 人设与女娲造人 skill 包
- `docs/upgrade-regression.md` — 上游升级回归方案
- `tests/` — 纯逻辑单测

## 合规

音色与肖像素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈。

## 相关链接

- speech-to-speech: https://github.com/huggingface/speech-to-speech
- AVTR-1: https://github.com/avaturn-live/avtr-1
- VoxCPM2: https://huggingface.co/openbmb/VoxCPM2
- FunASR（SenseVoice）: https://github.com/modelscope/FunASR
- Mem0: https://github.com/mem0ai/mem0
- aiortc: https://github.com/aiortc/aiortc
- DeepSeek API: https://platform.deepseek.com
