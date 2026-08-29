# VoxEMW · 河南妮儿数字人

> 对着浏览器说话，一个说河南话的写实数字人秒回你——**说完 ~3s 开口，声画齐出**。
> 判停 / 转写 / 克隆合成 / 数字人渲染 / 视觉，全部跑在一张 4090 上，只有大脑上云（DeepSeek API）。

[![version](https://img.shields.io/badge/version-v1.9.1-blue)](https://github.com/emwstudio/VoxEMW/tags)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![B站](https://img.shields.io/badge/B站-电磁波Studio-00a1d6?logo=bilibili&logoColor=white)](https://space.bilibili.com/492428186)
[![YouTube](https://img.shields.io/badge/YouTube-@emw__studio-ff0000?logo=youtube&logoColor=white)](https://www.youtube.com/@emw_studio)
[![X](https://img.shields.io/badge/X-@emwstudio-000000?logo=x&logoColor=white)](https://x.com/emwstudio)

**一张 4090（24GB）全搞定。** AutoDL 单卡同时驻留四个模型：Qwen3-ASR-1.7B（耳朵）+ VoxCPM2（克隆嗓音）+ SoulX-FlashHead（写实数字人）+ MiniCPM-V-4.6（眼睛），显存 21.7G/24G。Mac + VRM 二次元轻量档存档于 **git tag v1.9.0**。

> 形态说明：早期版本（云端 AVTR-1 + 本地 27B 全离线）见 git tag v1.7.x；纯语音星空版见 v1.8.x；
> Mac 本地 + VRM 二次元版见 v1.9.0。当前主线：**4090 满血写实数字人版**。

## 开发日记（视频）

本项目的开发过程记录在 **电磁波Studio**，看不懂代码可以先看视频：

- B站：https://space.bilibili.com/492428186
- YouTube：https://www.youtube.com/@emw_studio
- X：https://x.com/emwstudio
- 抖音：https://v.douyin.com/PlI1sZaaboA
- 小红书：https://xhslink.cn/m/2B0XSJKDWBg
- 视频号&公众号：微信搜索「电磁波Studio」

## 架构（六块积木）

```
浏览器（麦克风上行 + 摄像头抓帧 + 视频画布）
  │  ws/http（TCP 隧道友好；LAN 可切 WebRTC Opus 轨）
  ▼
4090 服务器（AutoDL，24GB）
  orchestrator（:8000）——会话调度 / persona 注入 / 打断编排
    ├─ s2s 语音管线（:8765）  ①VAD → ②STT → ③LLM → ④TTS
    │                                    │
    │                                    └─ DeepSeek API（deepseek-v4-flash）
    ├─ SoulX 渲染服务（:8791，独立 flashhead env）⑤数字人： paced 喂入 → 25fps 滴灌
    └─ VLM 边车（:18099，独立 vlm env）         ⑥眼睛：MiniCPM-V-4.6 看图/OCR
```

| 积木 | 模型/方案 | 说明 |
|---|---|---|
| ① VAD 判停 | Silero + SmartTurn v3.2（CPU ONNX） | 语义复核，停顿不抢答 |
| ② STT 语音转写 | Qwen3-ASR-1.7B-hf（transformers，CUDA bf16） | 人设热词注入 + 专名后校正 |
| ③ LLM 大脑 | DeepSeek v4-flash（API） | 关思考模式保时延 |
| ④ TTS 语音合成 | VoxCPM2（官方 PyTorch bf16） | **零样本河南话克隆**，TTFA 0.2s，RTF ~0.5 |
| ⑤ 数字人 | Soul-AILab/SoulX-FlashHead-1_3B（Lite） | 音频驱动写实 talking-head，96 FPS 级，待机呼吸常活 |
| ⑥ 眼睛 | MiniCPM-V-4.6（1.3B bf16） | 「妮儿看看」触发：抓帧 → 描述 → 人设口吻重答，36 切片 OCR |
| ⑤+ Persona 人设 | 女娲 · Skill 造人术 | `personas/*.md`（含克隆参考音 + 逐字台词） |

## 部署（AutoDL 4090）

```bash
# 实例上（环境已按 docs/plan-4090.md 装好）：
bash scripts/start_4090.sh          # 一键起：管线(py312) → SoulX(flashhead) → VLM(vlm) → orchestrator
bash scripts/start_4090.sh stop     # 全停

# 本地（Mac）架隧道后开页面：
ssh -NL 8000:127.0.0.1:8000 -p <port> root@<autodl-host>
open http://localhost:8000
```

要点：TCP-only 隧道环境下行音频走 WS + WebAudio 播放（WebRTC 的 UDP 过不去，已内置切换）；局域网/公网直连可回 `rtc.enabled: true` 走低延迟 Opus 轨。

## 特性

- **写实数字人**：SoulX-FlashHead 音频驱动——唇形跟音频、眨眼、微表情、呼吸常活；待机↔说话状态机，收嘴自然
- **单时钟唇形同步（V1 架构）**：TTS 音频按播放时刻 paced 喂入渲染端，服务端 25fps 滴灌出帧，声音压后 1.3s 对齐渲染滞后——零时间戳零排程，简单才稳
- **河南话克隆**：VoxCPM2 零样本克隆，河南妮儿人设（中/得劲儿/大信球/俺不中嘞 剧本全对）
- **眼睛**：说「妮儿看看」，她看你的摄像头画面并用人设口吻描述；OCR 逐字全对
- **语义判停 + 打断**：SmartTurn 复核不抢话；插嘴即停（音频/帧队列毫秒级双清）
- **热词注入**：STT 人设热词（河南妮儿/恁/中/得劲）+ 转写后专名确定性校正
- **断线自愈**：avatar 断线 2s 自动重连；会话槽位秒拒自动重试；回声/热词背诵压制
- **四模型同卡**：12.3G 管线 + 6.2G SoulX + 3.2G VLM = 21.7G/24G 稳定共存

## 更新日志

### v1.9.1（4090 满血版）

- **主线迁移 4090/24GB**：Qwen3-ASR-1.7B（CUDA bf16）+ VoxCPM2 官方全精度 + DeepSeek API，语音链路实测：短句首音 ~1.9s、VoxCPM2 TTFA 0.16-0.24s、RTF 0.4-0.6
- **第六块积木换代：SoulX-FlashHead-1_3B 写实数字人**（独立 flashhead env 渲染服务，ws 音频流 → 25fps JPEG 帧流；Lite 档 3.6x 实时；待机呼吸/眨眼常活）
- **WS 音频下行模式**：TCP-only 隧道（AutoDL/SSH）下 WebRTC UDP 过不去——`rtc.enabled=false` 时音频 delta 走 WebSocket + 前端 WebAudio 队列无缝续播，打断本地清队
- **V1 唇形同步架构定稿**：paced 喂入（32000B/s）+ 服务端 25fps 滴灌 + 声音 lead 1.3s——曾尝试显式时间戳同步（生成速度喂入 + 前端游标排程），延迟未省反增 10+ bug，已回滚并记录决策（docs/plan-4090.md）
- **眼睛上线**：MiniCPM-V-4.6（transformers 原生边车，独立 vlm env 钉 5.7.0 绕开 5.16 切片回归；36 切片 OCR 逐字全对）；浏览器定时抓帧上传，「看看」触发视觉回合
- **工程加固**：avatar 断线 2s 自动重连 + 会话收尸防僵尸连接；渲染 flush 攒段循环内即丢防污染；收尾补静音帧只留 6 帧（0.24s 收嘴）；磁盘死缓存清理 45G
- **speech-to-speech 升 main @3986f5**（最新），Qwen3-ASR 用 -hf 变体（非 -hf 在 transformers 5.x 静默丢权重）

### v1.9.0（Mac 数字人版，存档 tag）

- **第六块积木：VRM 数字人**（three.js + @pixiv/three-vrm，免构建），星空保留为背景
- **服务端音素口型**：wLipSync 复刻，音素帧与音频同队进 pacer 对齐播出时刻
- **情绪表情**：DeepSeek 按句情绪分类驱动 VRM 表情
- **人物蒸馏**：河南妮儿（personas/henannier.md）+ Qwen3-TTS-1.7B 克隆

### v1.8.1（补丁版）

- **音质修复**：Opus 编码 voip→audio 模式（voip SILK 抹平瞬态致首音发闷）
- **iOS 壳加固**：voiceChat 音频会话（系统级 AEC）、通话防锁屏
- **幽灵回合防线**：回声压制 + 热词背诵压制（均带单测）
- **前端「星空电台」改版**：主角光环按钮、悬浮气泡、断线即重连入口

## 本地开发（跑测试）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v
```

## 合规与许可

- 代码以 [MIT](LICENSE) 发布；第三方模型遵循各自协议（speech-to-speech Apache-2.0、
  Qwen3-ASR / VoxCPM2 / SoulX-FlashHead Apache-2.0、MiniCPM-V Apache-2.0 等），商用前请自行确认
- 音色素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈
