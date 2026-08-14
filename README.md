# VoxEMW · 数字人实时语音聊天助手

[![version](https://img.shields.io/badge/version-v1.4.0-blue)](https://github.com/emwstudio/VoxEMW/tags)
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

对着浏览器说话，屏幕里的数字人开口回答你。单卡 RTX 4090 即可运行，
**你说完到听到第一声 ≈ 2.4s**。还能点舞——说「给我跳个冰摇」，
他会回你一句开场白，然后灯光亮起、全屏开跳。

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

## 架构

```
浏览器
  │  WebRTC 音画（VP8 + Opus，TURN 走 coturn :3478）
  │  WS 控制/麦克风/转写（:8000）
  ▼
orchestrator（CPU，:8000）
  ├─→ s2s 语音管线（:8765）   VAD → STT SenseVoice → LLM DeepSeek → TTS VoxCPM2
  ├─→ avatar 数字人服务（:8767） AVTR-1（TensorRT，说话 + 倾听双流）
  └─→ 跳舞生成工人（离线子进程） Wan-Animate-2 蒸馏版（DiffSynth 管线，分段 + 超分）
```

## 跳舞积木（Wan-Animate-2）

素材页 `/dance` 上传驱动视频 + 全身照 → 离线排队生成 → 通话中点舞即播：

- **先排舞、后点舞**：生成是重活（14B 模型独占显卡，期间通话暂停，完成自动恢复）
- **语音点舞**：说「给我跳个 XX」，他说完开场白 → 全屏舞台开播 → 播完自动回通话
- **自动高清**：原生 416×736 生成后 Real-ESRGAN 超分到 720p 档
- **抽卡**：Seed 留空随机（卡片上固化显示，填回即可复现）
- **4090 实测**：81 帧/段 ≈ 6 分钟，15 秒成片 ≈ 45 分钟 + 超分 2.5 分钟

## 延迟分解（4090 实测）

| 环节 | 耗时 |
|---|---|
| VAD 判停 | ~0.5s |
| STT（SenseVoiceSmall） | ~0.1s |
| LLM 首句（DeepSeek v4-flash 流式） | ~1.4s |
| TTS 首音（VoxCPM2 流式） | ~0.1s |
| 唇同步（AVTR-1） | ~0.45s |

## 部署（AutoDL 单卡 4090）

```bash
cp .env.example .env.local        # 填 DEEPSEEK_API_KEY
bash scripts/autodl_setup.sh      # 装环境 + 下模型（幂等）
bash scripts/start_assistant.sh   # 起服务
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

- **倾听反应**：你说话时数字人实时注视/微表情回应（AVTR-1 双流）
- **待机微动**：无语音时眨眼/轻摇头，画面永远活着
- **语音点舞**：跳舞积木接入通话链路，开场白 → 灯光舞台 → 播完回通话
- **人设切换**：前端点 chip，人设 / 音色 / 肖像三路同切
- **换图免重启**：页面右下角「换图」按钮，即传即换

## 本地开发（macOS，无 GPU）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v
```

## 合规与许可

- 代码以 [MIT](LICENSE) 发布；第三方模型遵循各自协议（speech-to-speech Apache-2.0、
  VoxCPM2 Apache-2.0、AVTR-1 Community License、Wan-Animate-2 Apache-2.0 等），
  商用前请自行确认各模型协议
- 音色与肖像素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈

## 相关链接

- speech-to-speech: https://github.com/huggingface/speech-to-speech
- AVTR-1: https://github.com/avaturn-live/avtr-1
- Wan-Animate-2: https://github.com/Wan-Video/Wan-Animate-2
- DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio
- VoxCPM2: https://huggingface.co/openbmb/VoxCPM2
- FunASR（SenseVoice）: https://github.com/modelscope/FunASR
- DeepSeek API: https://platform.deepseek.com
