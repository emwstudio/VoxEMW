# VoxEMW · 河南妮儿语音助手

[![version](https://img.shields.io/badge/version-v1.9.0-blue)](https://github.com/emwstudio/VoxEMW/tags)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![B站](https://img.shields.io/badge/B站-电磁波Studio-00a1d6?logo=bilibili&logoColor=white)](https://space.bilibili.com/492428186)
[![YouTube](https://img.shields.io/badge/YouTube-@emw__studio-ff0000?logo=youtube&logoColor=white)](https://www.youtube.com/@emw_studio)
[![X](https://img.shields.io/badge/X-@emwstudio-000000?logo=x&logoColor=white)](https://x.com/emwstudio)

对着浏览器说话，一个二次元数字人用你自己的角色声音回答你——**你说完，她 ≈1.3s 就开口**（实测最快 1.26s，中位 ~1.6s）。

**一台 MacBook 全搞定。** 判停 / 转写 / 克隆合成 / 对话调度 / 数字人渲染，全部跑在你家 Mac 上（Apple Silicon，实测 M5 16GB 流畅），只有大脑上云（DeepSeek API，按 token 计费，日常闲聊一天几毛）。打开浏览器就能聊；想躺床上玩，还有个可选的 iPhone 壳 App（局域网 https 连进来当客户端）。

> 形态说明：早期版本（云端 4090 + AVTR-1 真人数字人 + 本地 27B 全离线）见 git tag v1.7.x；
> 纯语音星空版见 v1.8.x。当前主线：**Mac 本地语音 + VRM 数字人**（iOS 壳可选）。

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
iPhone 壳 App / 浏览器
  │  https/wss（局域网）+ WebRTC 音频轨（Opus）
  ▼
Mac 本地服务器：orchestrator（:8000，LAN TLS :9443）
  ├─→ s2s 语音管线（:8765）  ①VAD → ②STT → ③LLM → ④TTS
  │                                 │
  │                                 └─ DeepSeek API（deepseek-v4-flash，非思考模式）
  └─→ ⑥数字人驱动：音素口型（服务端逐帧）+ 情绪分类（按句）──随音频事件下发
```

| 积木 | 模型/方案 | 说明 |
|---|---|---|
| ① VAD 判停 | Silero + SmartTurn（CPU ONNX） | 64ms 软判停 + 语义复核（停顿不抢答） |
| ② STT 语音转写 | Qwen3-ASR-0.6B-hf（transformers，MPS） | 人设热词注入 + 专名后校正 |
| ③ LLM 大脑 | DeepSeek v4-flash（API） | 关思考模式保时延；工具调用支持 |
| ④ TTS 语音合成 | Qwen3-TTS-1.7B-Base（MLX 6bit） | **零样本音色克隆**：人设 ref.wav + 逐字台词注入，首音 ~0.9s |
| ⑤ Persona 人设 | 女娲 · Skill 造人术（nuwa-skill） | `personas/*.md`（含克隆参考音 + 逐字台词，注册即用） |
| ⑥ 数字人 | VRM（VRoid）+ three.js / @pixiv/three-vrm | 浏览器渲染免构建；**服务端音素口型** + 情绪表情 + 微动作 |

## 部署

### ① Mac 服务器（必需）

```bash
cp .env.example .env.local        # 写入 DEEPSEEK_API_KEY=sk-...（DeepSeek 控制台创建）
bash scripts/mac_setup.sh         # 装环境 + 下模型（幂等，~4G）
bash scripts/start_mac.sh         # 起管线 + orchestrator
```

只用浏览器的话到这步就够了：`open http://localhost:8000` 开聊。
启停：`bash scripts/start_mac.sh [stop]`。

### ② 换人设 / 换形象

- **人设**：`personas/<id>.md` 写人设正文 + frontmatter 声明 `ref_wav`（克隆参考音，10~30s 干净单人声）和 `ref_text`（逐字稿），注册进 `configs/assistant.yaml` 的 `personas.list` 即可
- **数字人形象**：VRoid Studio 捏人导出 `.vrm` 放进 `web/avatars/`，改 `web/avatar-vrm.js` 顶部的 `MODEL_URL` 即换脸；口型 morph 兼容 VRoid `Fcl_MTH_*` / VRM 1.0 `aa` 系 / 0.x 单字母三种命名

### ③ iPhone 客户端（壳 App，可选）

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
- **VRM 数字人**：VRoid 形象立于星空——胸上取景、T-pose 校正、淡入加载；模型加载失败自动退回纯星空，对话不断流
- **音素口型（服务端）**：wLipSync 算法的服务端复刻——每帧 PCM 算 MFCC，余弦匹配 A/E/I/O/U，32ms 一帧随音频事件下发；**与 pacer 播出时刻对齐**（绕开 Chrome「RTC 音频进不了 WebAudio」的坑，浏览器/iOS 表现一致）
- **情绪表情**：DeepSeek 按句情绪分类（开心/生气/委屈/惊讶），VRM 表情 + 头部微姿态跟随台词，4.5s 自然回落
- **微动作**：随机眨眼、呼吸起伏、视线注视游移、倾听点头、思考垂眼、说话随能量晃脑；头部姿态阻尼平滑不跳变
- **语义判停**：SmartTurn 复核——你停顿她沉住气，你说完她秒接
- **打断记忆**：中途打断，她记得刚才说到哪（已播时长回报，已听内容写回上下文）
- **流式字幕**：回答逐字上屏，不整句砸脸（默认纯数字人界面，字幕仍可恢复显示）
- **空回复兜底**：模型偶发失声自动追问重答，对话不断流
- **星空交互**：全屏星空随对话呼吸——你说话时向中心收拢，她说话时随声波动
- **iOS 客户端**：壳 App 局域网直连；voiceChat 音频会话（系统级回声消除，外放不自激）；通话防锁屏；单按钮 + 状态灯交互
- **时延实测（M5）**：说完到首音 ≈ 1.3s（短句闲聊实测最快 1.26s，长句 ~2s）

## 更新日志

### v1.9.0（数字人版）

- **第六块积木：VRM 数字人**。VRoid/VRM 形象（three.js + @pixiv/three-vrm，本地 vendor 免构建），星空保留为背景，聊天框默认隐藏
- **服务端音素口型**：wLipSync 算法逐行复刻为服务端实现（pre-emphasis → Hamming → 峰值归一 → FFT 幅度谱 → Slaney mel → log10 → DCT 取 1..12 → 余弦 ^100 归一），与浏览器端输出对照验证（argmax 一致率 79%、轨迹相关 0.82）；音素帧与音频同队进 pacer，**播出才下发**，对齐误差一个监听周期（30ms）
- **情绪表情**：orchestrator 按句切分流式转写，DeepSeek 独立小调用判情绪（happy/angry/sad/surprised，失败静默、seq 保序），`vox.emotion` 事件驱动表情与头部微姿态
- **口型幅度补偿**：样例模型 blendshape 偏小（无 jaw 骨），morphTargetInfluences 超系数外推 ×1.7（封顶 1.8），兼容 VRoid / VRM 1.0 / 0.x 三种 morph 命名
- **人物蒸馏**：河南妮儿（`personas/henannier.md`——河南话说法表、话题工作流、招牌段子固定配合）；参考音 + 逐字台词注册即克隆
- **连接加固**：orchestrator 连管线带判活重试——上个会话槽位释放需等 handler 链排空（TTS 生成中可达 ~10s），期间新连接被秒拒导致「刷新/首点必连不上」；重试透明吸收
- **可观测**：打断清队记日志（response.done 后抢话丢弃的未播音频时长）；`?debug=1` 角标显示口型源/权重；`?mouth=` 固定口型权重调试位
- **启动脚本**：`no_proxy` 修复——系统全局代理会把本机 ws 握手送进代理导致 EOF

### v1.8.1（补丁版）

- **音质修复**：Opus 编码 voip→audio 模式——voip 的 SILK 偏向会抹平瞬态，
  每轮开头第一个音发闷；已逐段验证（源码 PCM 干净 → 编码器模式是元凶）
- **iOS 壳加固**：voiceChat 音频会话（系统级 AEC，外放不再自激触发下一轮）、
  通话防锁屏、刘海/Home 条安全区适配、状态栏浅色
- **幽灵回合防线（服务端）**：回声压制（外放泄漏把她自己的话收成用户输入，
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
  Qwen3-TTS / Qwen3-ASR Apache-2.0 等），商用前请自行确认各模型协议；
  VRM 模型遵循作者在 VRoid Hub 上声明的使用许可
- 音色素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈
