# VoxEMW · 数字人实时语音聊天助手

[![version](https://img.shields.io/badge/version-v1.6.1-blue)](https://github.com/emwstudio/VoxEMW/tags)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![B站](https://img.shields.io/badge/B站-电磁波Studio-00a1d6?logo=bilibili&logoColor=white)](https://space.bilibili.com/492428186)
[![YouTube](https://img.shields.io/badge/YouTube-@emw__studio-ff0000?logo=youtube&logoColor=white)](https://www.youtube.com/@emw_studio)
[![X](https://img.shields.io/badge/X-@emwstudio-000000?logo=x&logoColor=white)](https://x.com/emwstudio)
[![抖音](https://img.shields.io/badge/抖音-电磁波Studio-000000?logo=tiktok&logoColor=white)](https://v.douyin.com/PlI1sZaaboA)
[![小红书](https://img.shields.io/badge/小红书-电磁波Studio-ff2442?logo=xiaohongshu&logoColor=white)](https://xhslink.cn/m/2B0XSJKDWBg)
![视频号&公众号](https://img.shields.io/badge/视频号%26公众号-微信搜「电磁波Studio」-07c160?logo=wechat&logoColor=white)
[![微博](https://img.shields.io/badge/微博-电磁波Studio-e6162d?logo=sinaweibo&logoColor=white)](https://weibo.com/u/1765053862)
[![快手](https://img.shields.io/badge/快手-电磁波Studio-ff4906?logo=kuaishou&logoColor=white)](https://v.kuaishou.com/JZ1GQ7G8)
[![TikTok](https://img.shields.io/badge/TikTok-@emw.studio-fe2c55?logo=tiktok&logoColor=white)](https://www.tiktok.com/@emw.studio)
[![Instagram](https://img.shields.io/badge/Instagram-@emwstudio.ai-e4405f?logo=instagram&logoColor=white)](https://www.instagram.com/emwstudio.ai)

对着浏览器说话，屏幕里的数字人开口回答你。**完全离线**（零 API 调用），
单卡 RTX 4090 48G 全本地运行，**你说完到听到第一声 ≈ 1.5s**。

链路：[huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
实时语音管线 + [AVTR-1](https://github.com/avaturn-live/avtr-1) 数字人形象，
音画下行走 **WebRTC**，浏览器原生对齐口型。

## 开发日记（视频）

本项目的开发过程记录在 **电磁波Studio**，看不懂代码可以先看视频：

- B站：https://space.bilibili.com/492428186
- YouTube：https://www.youtube.com/@emw_studio
- X：https://x.com/emwstudio
- 抖音：https://v.douyin.com/PlI1sZaaboA
- 小红书：https://xhslink.cn/m/2B0XSJKDWBg
- 视频号&公众号：微信搜索「电磁波Studio」
- 微博：https://weibo.com/u/1765053862
- 快手：https://v.kuaishou.com/JZ1GQ7G8
- TikTok：https://www.tiktok.com/@emw.studio
- Instagram：https://www.instagram.com/emwstudio.ai

## 架构（五块积木）

```
浏览器
  │  WebRTC 音画（VP8 + Opus，TURN 走 coturn :3478）
  │  WS 控制/麦克风/转写（:8000）
  ▼
orchestrator（CPU，:8000）
  ├─→ s2s 语音管线（:8765）   ①VAD → ②STT → ③LLM → ④TTS
  ├─→ avatar 数字人服务（:8767） ⑤AVTR-1（TensorRT，说话 + 倾听双流）
  └─→ llama-server（:8081）   本地 LLM（Qwen3.8-27B + MTP 投机解码）
```

| 积木 | 模型 | 说明 |
|---|---|---|
| ① VAD 判停 | Silero + SmartTurn v3.2 | 64ms 软判停 + 语义复核 + 800ms 重开宽限（停顿不抢答） |
| ② STT 语音转写 | SenseVoiceSmall（FunASR） | ~0.1s |
| ③ LLM 大脑 | Qwen3.8-27B（UD-Q6_K_XL + MTP） | llama.cpp 本地服务，首句 ~0.9s |
| ④ TTS 语音合成 | VoxCPM2（音色克隆，流式） | 首音 ~0.1s |
| ⑤ Avatar 数字人 | AVTR-1（TensorRT） | 唇同步 + 倾听/思考表情 |

## 延迟分解（4090 48G 全离线实测）

| 环节 | 耗时 |
|---|---|
| VAD 软判停（64ms）+ SmartTurn 复核（~0.08s） | ~0.15s |
| STT（SenseVoiceSmall） | ~0.1s |
| LLM 首句（Qwen3.8-27B 本地流式） | ~0.9s |
| TTS 首音（VoxCPM2 流式） | ~0.1s |
| 唇同步缓冲（AVTR-1） | ~0.4s |
| **合计** | **≈ 1.5s** |

## 部署（AutoDL 单卡 4090 48G）

```bash
cp .env.example .env.local        # 在线回退时填 DEEPSEEK_API_KEY（离线版不需要）
bash scripts/autodl_setup.sh      # 装环境 + 下模型（幂等）
bash scripts/start_assistant.sh   # 一键起全栈（含本地 LLM llama-server）
```

本地开隧道并打开页面：

```bash
bash scripts/tunnel.sh        # 转发 8000（网页）+ 3478（WebRTC 媒体）
# 打开 http://localhost:8000
```

启停：`bash scripts/start_assistant.sh [stop]`。

**参考图规范**：16:9 横版胸像（1920×1080），脸宽占图宽 ~20%、头顶留白 ~19%，
微张嘴/露齿者口型最佳。页面上可直接点「换图」热替换，无需重启。

## 特性

- **完全离线**：本地 Qwen3.8-27B 当大脑（MTP 投机解码 41.7 tok/s），零 API 费用零外网依赖
- **语义判停**：SmartTurn 复核——你停顿他沉住气，你说完他秒接，中途停顿不抢答不丢字
- **流式字幕**：回答逐字上屏，不整句砸脸
- **空回复兜底**：模型偶发失声自动追问重答，对话不断流
- **长对话不漂移**：超 30 轮后旧对话后台摘要压缩
- **倾听反应**：你说话时数字人实时注视/微表情回应（AVTR-1 双流）
- **待机微动**：无语音时眨眼/轻摇头，画面永远活着
- **人设切换**：前端点 chip，人设 / 音色 / 肖像三路同切
- **换图免重启**：页面右下角「换图」按钮，即传即换

## 本地开发（macOS，无 GPU）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v
```

## 合规与许可

- 代码以 [MIT](LICENSE) 发布；第三方模型遵循各自协议（speech-to-speech Apache-2.0、
  VoxCPM2 Apache-2.0、Qwen3.8 Apache-2.0、AVTR-1 Community License 等），
  商用前请自行确认各模型协议
- 音色与肖像素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈

## 相关链接

- speech-to-speech: https://github.com/huggingface/speech-to-speech
- AVTR-1: https://github.com/avaturn-live/avtr-1
- VoxCPM2: https://huggingface.co/openbmb/VoxCPM2
- Qwen3.8（GGUF 量化）: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- llama.cpp: https://github.com/ggml-org/llama.cpp
- FunASR（SenseVoice）: https://github.com/modelscope/FunASR
- DeepSeek API（在线回退）: https://platform.deepseek.com
