# VoxEMW · 数字人实时语音聊天助手

[![version](https://img.shields.io/badge/version-v1.3.1-blue)](https://github.com/emwstudio/VoxEMW/tags)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![B站](https://img.shields.io/badge/B站-电磁波Studio-00a1d6)](https://space.bilibili.com/492428186)
[![YouTube](https://img.shields.io/badge/YouTube-@emw__studio-ff0000)](https://www.youtube.com/@emw_studio)
[![X](https://img.shields.io/badge/X-@emwstudio-000000)](https://x.com/emwstudio)

对着浏览器说话，屏幕里的数字人开口回答你。人设、音色、形象三位一体，
单卡 RTX 4090D 即可运行，**你说完到听到第一声 ≈ 2.4s**。

链路基于 [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
（VAD → STT → LLM → TTS 实时管线，pip 依赖零改动，自定义积木运行时注册）
+ [AVTR-1](https://github.com/avaturn-live/avtr-1) 数字人形象（TensorRT，0.2s 生成粒度）。
音画下行走 **WebRTC**（VP8 + Opus），浏览器按 RTP 时间戳原生对齐口型。
人设来自女娲蒸馏（`personas/<id>.md`）。

## 开发日记（视频）

本项目的开发过程记录在 **电磁波Studio**——延迟优化、模型选型、踩坑全程都有视频，
看不懂代码可以先看视频：

- B站：https://space.bilibili.com/492428186
- YouTube：https://www.youtube.com/@emw_studio
- X：https://x.com/emwstudio

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

## 积木（configs/assistant.yaml）

| 积木 | 选型 |
|---|---|
| vad | silero-vad（s2s 内置，判停 500ms） |
| stt | `iic/SenseVoiceSmall`（FunASR，非自回归一次前向出全文） |
| llm | DeepSeek `deepseek-v4-flash`（流式逐句送 TTS） |
| tts | `openbmb/VoxCPM2`（克隆音色，`tts.rate` 变速补偿） |
| avatar | `avaturn-live/avtr-1`（TensorRT，0.2s/chunk） |
| persona | `personas/<id>.md`（女娲蒸馏，frontmatter 绑定音色三件套） |

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
| `?solo=1` | 单栏全屏，录制用 |

## 特性

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

- `configs/assistant.yaml` — 唯一配置（积木 + server + rtc）
- `configs/turnserver.conf` — coturn 配置（WebRTC 媒体中继）
- `voxemw/pipeline/` — s2s 集成（STT/TTS 积木 + 启动器）
- `voxemw/avatar/` — 数字人服务（`service.py` + `avtr1_engine.py`）+ orchestrator
  + WebRTC 轨（`rtc.py`）+ 音画调度器（`avsync.py`）
- `web/` — 通话页（无构建）
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
- aiortc: https://github.com/aiortc/aiortc
- DeepSeek API: https://platform.deepseek.com
