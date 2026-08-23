# VoxEMW · 良子语音助手

[![version](https://img.shields.io/badge/version-v1.8.1-blue)](https://github.com/emwstudio/VoxEMW/tags)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![B站](https://img.shields.io/badge/B站-电磁波Studio-00a1d6?logo=bilibili&logoColor=white)](https://space.bilibili.com/492428186)
[![YouTube](https://img.shields.io/badge/YouTube-@emw__studio-ff0000?logo=youtube&logoColor=white)](https://www.youtube.com/@emw_studio)
[![X](https://img.shields.io/badge/X-@emwstudio-000000?logo=x&logoColor=white)](https://x.com/emwstudio)

对着手机（或浏览器）说话，良子用他自己的声音回答你——**你说完，他 ≈1.3s 就开口**（实测最快 1.26s，中位 ~1.6s）。

**定位：Mac 做本地服务器，iOS 做客户端。** 全部语音积木（判停 / 转写 / 克隆合成 /
对话调度）跑在你家 Mac 上（Apple Silicon，实测 M5 16GB 流畅），只有大脑上云
（DeepSeek API，按 token 计费，日常闲聊一天几毛）。iPhone/iPad 装个壳 App
走局域网 https 连进来；不装 App 用浏览器直连也一样玩。

> 形态说明：早期版本（云端 4090 + AVTR-1 数字人视频形象 + 本地 27B 全离线）
> 见 git tag v1.7.x。当前主线：Mac 本地服务器 + iOS 客户端（纯语音 + 星空交互）。

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
iPhone 壳 App / 浏览器
  │  https/wss（局域网）+ WebRTC 音频轨（Opus）
  ▼
Mac 本地服务器：orchestrator（:8000，LAN TLS :9443）
  └─→ s2s 语音管线（:8765）  ①VAD → ②STT → ③LLM → ④TTS
                                    │
                                    └─ DeepSeek API（deepseek-v4-flash，非思考模式）
```

| 积木 | 模型 | 说明 |
|---|---|---|
| ① VAD 判停 | Silero + SmartTurn（CPU ONNX） | 64ms 软判停 + 语义复核（停顿不抢答） |
| ② STT 语音转写 | Qwen3-ASR-0.6B-hf（transformers，MPS） | 人设热词注入 + 专名后校正，黑话近满分 |
| ③ LLM 大脑 | DeepSeek v4-flash（API） | 关思考模式保时延；工具调用支持 |
| ④ TTS 语音合成 | Qwen3-TTS-1.7B-Base（MLX 6bit） | **零样本音色克隆**：良子 ref.wav + 逐字台词注入，首音 ~0.5s |
| Persona 人设 | 女娲 · Skill 造人术（nuwa-skill） | `personas/*.md`（含克隆参考音 + 逐字台词，注册即用） |

## 部署

### ① Mac 服务器（必需）

```bash
cp .env.example .env.local        # 写入 DEEPSEEK_API_KEY=sk-...（DeepSeek 控制台创建）
bash scripts/mac_setup.sh         # 装环境 + 下模型（幂等，~4G）
bash scripts/start_mac.sh         # 起管线 + orchestrator
```

只用浏览器的话到这步就够了：`open http://localhost:8000` 开聊。
启停：`bash scripts/start_mac.sh [stop]`。

### ② iPhone 客户端（壳 App）

iOS 的麦克风只在 https 页面里可用，所以要先给 Mac 发一张局域网证书：

```bash
bash scripts/make_lan_tls.sh   # 生成局域网证书（一次性，幂等）
```

1. 把 `scripts/lan_tls/rootCA-for-iphone.pem` AirDrop 到 iPhone，
   设置里安装描述文件，并在「关于本机 → 证书信任设置」中启用
2. Xcode 打开 `ios/LiangziVoice/LiangziVoice.xcodeproj`，
   Signing & Capabilities 选你自己的 Team，真机运行（免费账号可侧载，7 天一签）
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
- **iOS 客户端**：壳 App 局域网直连；voiceChat 音频会话（系统级回声消除，
  外放不自激）；通话防锁屏；单按钮 + 状态灯交互
- **时延实测（M5）**：说完到首音 ≈ 1.3s（短句闲聊实测最快 1.26s，长句 ~2s）

## 更新日志

### v1.8.1（补丁版）

- **音质修复**：Opus 编码 voip→audio 模式——voip 的 SILK 偏向会抹平瞬态，
  每轮开头第一个音发闷；已逐段验证（源码 PCM 干净 → 编码器模式是元凶）
- **iOS 壳加固**：voiceChat 音频会话（系统级 AEC，外放不再自激触发下一轮）、
  通话防锁屏、刘海/Home 条安全区适配、状态栏浅色
- **幽灵回合防线（服务端）**：回声压制（外放泄漏把良子自己的话收成用户输入，
  整轮掐掉）+ 热词背诵压制（噪音被热词先验脑补成词表，整轮掐掉）——
  均带单测锁定
- **前端「星空电台」改版**：主角光环按钮（仅等待态跳波动）、悬浮气泡 +
  微信风对话头像、双状态灯合一、断线后按钮即重连入口
- **星空波纹全程在线**：服务端随音频事件下发响度（lvl），
  播放完毕信号（playback_done）让「说话中」跟播放走而不是跟生成走

## 本地开发（跑测试）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v
```

## 合规与许可

- 代码以 [MIT](LICENSE) 发布；第三方模型遵循各自协议（speech-to-speech Apache-2.0、
  Qwen3-TTS / Qwen3-ASR Apache-2.0 等），商用前请自行确认各模型协议
- 音色素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈
