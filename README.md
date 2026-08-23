# VoxEMW · 良子语音助手（本地 Mac 版）

[![version](https://img.shields.io/badge/version-v1.8.0-blue)](https://github.com/emwstudio/VoxEMW/tags)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![B站](https://img.shields.io/badge/B站-电磁波Studio-00a1d6?logo=bilibili&logoColor=white)](https://space.bilibili.com/492428186)
[![YouTube](https://img.shields.io/badge/YouTube-@emw__studio-ff0000?logo=youtube&logoColor=white)](https://www.youtube.com/@emw_studio)
[![X](https://img.shields.io/badge/X-@emwstudio-000000?logo=x&logoColor=white)](https://x.com/emwstudio)

对着浏览器说话，良子用他自己的声音回答你——**你说完，他 ≈1.3s 就开口**（实测最快 1.26s，中位 ~1.6s）。

**一台 Mac 就能本地跑起来**——不需要 GPU 服务器、不需要 Docker、不需要大内存：
VAD 判停 / 语音转写 / 音色克隆 / 对话调度全部在本机完成（Apple Silicon，实测 M5 16GB 流畅），
只有大脑走 DeepSeek API（按 token 计费，日常闲聊一天几毛）。两条命令部署，浏览器开箱即聊。

> 形态说明：早期版本（云端 4090 + AVTR-1 数字人视频形象 + 本地 27B 全离线）
> 见 git tag v1.7.x。当前主线为本地 Mac 纯语音版（星空交互，无数字人画面）。

链路：浏览器（麦克风/扬声器）↔ 本机 orchestrator ↔ 本机语音管线
（[huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)）
+ DeepSeek API。音频下行走 WebRTC（本机直连，无需 TURN）。

## 开发日记（视频）

本项目的开发过程记录在 **电磁波Studio**，看不懂代码可以先看视频：

- B站：https://space.bilibili.com/492428186
- YouTube：https://www.youtube.com/@emw_studio
- X：https://x.com/emwstudio
- 抖音：https://v.douyin.com/PlI1sZaaboA
- 小红书：https://xhslink.cn/m/2B0XSJKDWBg
- 视频号&公众号：微信搜索「电磁波Studio」

## 架构（五块积木）

```
浏览器（localhost:8000）
  │  WebRTC 音频轨（Opus）+ WS 控制/转写
  ▼
orchestrator（CPU，:8000）
  └─→ s2s 语音管线（:8765）  ①VAD → ②STT → ③LLM → ④TTS
                                    │
                                    └─ DeepSeek API（deepseek-v4-flash，非思考模式）
```

| 积木 | 模型 | 说明 |
|---|---|---|
| ① VAD 判停 | Silero + SmartTurn（CPU ONNX） | 64ms 软判停 + 语义复核（停顿不抢答） |
| ② STT 语音转写 | Qwen3-ASR-0.6B-hf（transformers，MPS） | 人设热词注入，黑话近满分 |
| ③ LLM 大脑 | DeepSeek v4-flash（API） | 关思考模式保时延；工具调用支持 |
| ④ TTS 语音合成 | Qwen3-TTS-1.7B-Base（MLX 6bit） | **零样本音色克隆**：良子 ref.wav + 逐字台词注入，首音 ~0.5s |
| Persona 人设 | 女娲 · Skill 造人术（nuwa-skill） | `personas/*.md`（含克隆参考音 + 逐字台词，注册即用） |

## 部署（macOS Apple Silicon，实测 M5 16GB）

```bash
cp .env.example .env.local        # 写入 DEEPSEEK_API_KEY=sk-...（DeepSeek 控制台创建）
bash scripts/mac_setup.sh         # 装环境 + 下模型（幂等，~4G）
bash scripts/start_mac.sh         # 起 pipeline + orchestrator
open http://localhost:8000        # 开聊
```

启停：`bash scripts/start_mac.sh [stop]`。

## iPhone 壳（可选）

Mac 当服务器，iPhone 走局域网 https 访问（iOS 的麦克风只在 https 页面里可用）：

```bash
bash scripts/make_lan_tls.sh   # 生成局域网证书（一次性，幂等）
```

1. 把 `scripts/lan_tls/rootCA-for-iphone.pem` AirDrop 到 iPhone，
   设置里安装描述文件，并在「关于本机 → 证书信任设置」中启用
2. Xcode 打开 `ios/LiangziVoice/LiangziVoice.xcodeproj`，
   Signing & Capabilities 选你自己的 Team，真机运行（免费账号可侧载）
3. 日常使用：Mac 上 `bash scripts/start_mac.sh`，iPhone 点开 App 即可

Mac 换网络后 IP 变了：重跑 `make_lan_tls.sh` 并更新
`ios/LiangziVoice/Sources/LiangziVoiceApp.swift` 顶部的 `serverURL`。

## 特性

- **音色克隆**：Qwen3-TTS Base 零样本克隆，人设参考音 + 逐字台词注册即用
- **语义判停**：SmartTurn 复核——你停顿他沉住气，你说完他秒接
- **打断记忆**：中途打断，良子记得刚才说到哪（已播时长回报，已听内容写回上下文）
- **流式字幕**：回答逐字上屏，不整句砸脸
- **空回复兜底**：模型偶发失声自动追问重答，对话不断流
- **星空交互**：全屏星空随对话呼吸——你说话时向中心收拢，良子说话时随声波动
- **时延实测（M5）**：说完到首音 ≈ 1.3s（短句闲聊实测最快 1.26s，长句 ~2s）

## 本地开发（跑测试）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v
```

## 合规与许可

- 代码以 [MIT](LICENSE) 发布；第三方模型遵循各自协议（speech-to-speech Apache-2.0、
  Qwen3-TTS / Qwen3-ASR Apache-2.0 等），商用前请自行确认各模型协议
- 音色素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈
