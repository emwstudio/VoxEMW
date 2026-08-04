# VoxEMW · 数字人实时语音聊天助手

对着浏览器说话，屏幕里的数字人开口回答你——人设、音色、形象三位一体，
全部由「积木」声明式组装。

语音链路基于 [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
（VAD → STT → LLM → TTS 实时管线 + OpenAI Realtime ws 协议，pip 依赖、运行时注册自定义
积木，不 vendor 不打补丁）；第五块积木是数字人形象
[SoulX-FlashHead](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)
（一张参考肖像 + TTS 音频流 → 实时说话头视频）。人设来自女娲蒸馏
（`skills/*-perspective` → `personas/<id>.md`）。

## 架构

```
浏览器（web/，麦克风 + 摄像头画面 + 数字人画面）
   │  ws :8000（Realtime 事件 + 自定义 vox.persona + 二进制 JPEG 帧）
   ▼
orchestrator（voxemw/avatar/orchestrator.py，CPU）
   ├─→ s2s 语音管线（voxemw/pipeline/launch.py，:8765 内部）
   │     VAD silero → STT SenseVoiceSmall → LLM DeepSeek API → TTS VoxCPM2（克隆音色）
   ├─→ avatar 数字人服务（voxemw/avatar/service.py，:8767 内部，可缺席）
   │     AVTR-1（默认，TensorRT，0.2s/chunk）或 FlashHead Lite（回退，0.96s/chunk）
   │     TTS 音频流 → 口型同步 JPEG 帧流
   └─→ vision 外貌描述（voxemw/vision.py → Kimi K3 多模态 API，可缺席）
         摄像头截帧 → 客观描述 → 注入 s2s 对话让 persona 锐评
```

三进程同卡（RTX 4090D 24GB）：orchestrator 把 TTS 音频 delta 双写——一路回浏览器
播放，一路喂数字人驱动口型。数字人缺席（未启动/缺肖像）自动降级纯语音模式；
vision 缺席（缺 KIMI_API_KEY）只关截帧打分，语音对话不受影响。
单用户单会话：新浏览器连接自动顶掉旧会话（换网络产生的僵尸会话即时释放，
无需等超时）。

## 延迟（单卡 4090D 实测，你说完 → 听到第一声 ≈ 2.4s）

| 环节 | 耗时 | 说明 |
|---|---|---|
| VAD 判停 | ~0.5s | `min_silence_ms: 500` |
| STT | ~0.1s | SenseVoiceSmall 非自回归 |
| LLM 首句 | ~1.4s | DeepSeek v4-flash 流式逐句（首 token 0.65 + 解码 0.3 + 管线 0.45） |
| TTS 首音 | ~0.1s | VoxCPM 流式 TTFA |
| 唇同步缓冲 | 0.35s | AVTR-1 0.2s 窗口（首帧实测 0.32-0.42s）；FlashHead 后端需 0.8s |

## 数字人常驻微动

没有语音时（你说话中/峰哥思考中/说完待机），avatar 服务以静音持续驱动
25fps 画面——模型原生 idle 语义产出自然的眨眼、视线游移、头部微动（嘴闭合）。
下行帧带 1 字节 tag（idle/speech）：speech 帧进前端唇同步队列（从属于音频
播放时钟），idle 帧直接上屏。说话期间 orchestrator 下发 speech_active 禁止
idle 生成（防句间停顿插入 idle 帧卡画面）；idle 生成按实时节奏节流。
`avatar.idle_motion: false` 可回退闲时定格。

## 数字人后端：AVTR-1 vs FlashHead

默认 **AVTR-1**（[avaturn-live/avtr-1](https://github.com/avaturn-live/avtr-1)，
TensorRT 加速，5 帧/0.2s 生成粒度，4090D 实测 ~80ms/chunk = 2.5x 实时）：
唇同步缓冲 0.35s、原生静音 idle、支持双流倾听（listen 轨，接入中）。
**FlashHead** 保留为回退后端（`avatar.backend: flashhead`，扩散重绘画质更锐，
但 0.96s 窗口决定了 0.8s 缓冲地板）。`start_assistant.sh` 按 backend 自动选
python 环境（avtr1 = pixi env，flashhead = .venv-avatar）。

**AVTR-1 参考图规范（重要）**：官方 loader 假定参考帧为 16:9 横版胸像
（官方帧均 1920×1080，脸宽占图宽 ~20%、头顶留白 ~19%），对输入**非等比**
resize 到 1280×720——方图/竖图会把脸型拉变形。换人务必按此构图准备
`ref.png`。AVTR-1 部署：pixi env + TRT 引擎按显卡编译（sm89 已编好，
缓存在 avtr1_storage），注意 pixi 勿用 `pixi run`（会重同步 env 覆盖 pip 降级）。

## 截帧打分（视频通话玩法）

页面左栏是用户摄像头（随「开始对话」开启，可拒绝降级纯语音），右栏数字人。
对峰哥说「给我的脸打几分」，他会说暗号「摆好姿势，让我好好看看你」——前端
检测暗号后自动截一帧发给 orchestrator，Kimi K3 生成外貌客观描述，注入对话后
峰哥先描绘他「看到」的外表细节，再按人设十分制锐评。链路：语音 LLM（DeepSeek）
纯文本无视觉，视觉只走这一跳。

等待 Kimi 描述的几秒空白由「垫场」填掉：orchestrator 收到截帧后先注入一条
系统旁白让峰哥自由发挥（别动/调侃两句），描述返回后再注入打分指令。注入是
乐观发送 + 被拒重试（VAD 回复不发 response.created，无法预判回复是否在播；
response.create 被拒时等当前回复 response.done 自动重发）。从截帧到打分回复
说完期间，orchestrator 直接丢弃上行麦克风音频——用户插话不会打断峰哥打分
（前端有「麦克风暂闭」提示）。

## 六块积木（configs/assistant.yaml）

| 积木 | 选型 | 说明 |
|---|---|---|
| vad | silero-vad | s2s 内置，判停 500ms（speculative reopen 兜误断） |
| stt | `iic/SenseVoiceSmall` | 自定义积木 `voxemw.pipeline.stt_sensevoice`，FunASR 非自回归本地推理（4s 音频 0.06s）；`qwen3asr` 备选（多语种） |
| llm | DeepSeek `deepseek-v4-flash` | s2s 内置 chat-completions；流式逐句送 TTS（长回复首音 ~2s vs 整段 3-5s）；关 thinking 由 launch 注入 |
| tts | `openbmb/VoxCPM2` | 自定义积木 `voxemw.pipeline.tts_voxcpm`，Ultimate Cloning + 流式 |
| avatar | `avaturn-live/avtr-1`（TensorRT）| 默认后端，0.2s/chunk，4090D ~80ms/chunk；`SoulX-FlashHead-1_3B` Lite 保留回退 |
| persona | `personas/<id>.md` | 女娲蒸馏产物；frontmatter 绑定音色三件套（见下） |

另有一块可选的 `vision` 积木（`voxemw/vision.py`）：Kimi K3 多模态 API，
给「截帧打分」提供外貌描述；不配 key 自动关闭，见上文「截帧打分」。
另有 `filler` 积木（默认关）：转写完成即播预渲染口头禅垫音盖 LLM 首句空白，
按 SenseVoice 情绪标签分组选词（代码与机制保留，实测衔接自然度不足已停用，
`filler.enabled: true` 可复活）。

换积木改 yaml 对应段；每个 persona 的三件套：

```
personas/fengge.md          # 人设正文（system prompt）+ frontmatter:
                            #   name / ref_wav / ref_text / ref_image（label 界面短名，可选）
assets/fengge/ref.wav       # 音色参考音（10-30s 清晰单人声）
assets/fengge/ref.txt       # 参考音逐字台词（Ultimate Cloning 必需）
assets/fengge/ref.png       # 数字人肖像（512×512 附近最佳）
```

人设切换（前端点 chip）三路同时切换：LLM instructions（session.update）、
TTS 音色（audio.output.voice，启动时全员预编码 prompt cache）、数字人肖像
（avatar set_image）。

## 部署（AutoDL RTX 4090D）

```bash
# 1. rsync 仓库到实例后
cp .env.example .env.local   # 填入 DEEPSEEK_API_KEY（必填）；KIMI_API_KEY（可选，截帧打分用）
bash scripts/autodl_setup.sh # 幂等：双 venv + 模型下载 + 启动三进程

# 2. 本机 SSH 隧道（AutoDL 默认不开公网）
ssh -CNg -L 8000:127.0.0.1:8000 root@<实例主机> -p <SSH端口>
# 浏览器打开 http://localhost:8000
```

双 venv 隔离（依赖互相冲突，不可合装）：

- `.venv`（py312 + torch 2.8）：s2s 管线（transformers 5.13 / Qwen3-ASR / VoxCPM2）+ orchestrator
- `.venv-avatar`（py310 + torch 2.7.1）：FlashHead（transformers==4.57.3 / xformers / flash_attn）
  flash_attn 说明：AutoDL 老镜像（Ubuntu 20.04，glibc 2.31）装不了官方预编译 wheel
  （需 glibc≥2.32），已成功源码编译 `flash_attn==2.8.0.post2`（FlashHead 官方钉版；
  注意 xformers 只接受 ≤2.8.0，2.8.3 会导致 xformers import 失败）。编译路径：
  NVIDIA runfile 装 CUDA 12.8 到 /root/cuda-12.8（系统自带 11.6 太老），
  `FLASH_ATTENTION_FORCE_BUILD=TRUE` 强制源码（setup.py 默认会偷下预编译 wheel）。
  嫌编译麻烦可直接用仓库 Releases 里的编译产物（artifacts-flashattn，
  glibc 2.31 专用，解包到 .venv-avatar/lib/python3.10/site-packages/ 即可）。
  无 flash_attn 时自动回退 SDPA（也够 2× 实时，编译仅为锦上添花）

部署脚本已内置国内网络与兼容性处理（幂等，可重复执行）：阿里云 PyPI 镜像、
ModelScope 模型源（~13MB/s）、GitHub 学术加速、两个空 stub wheel
（绕过 faster-qwen3-tts 的 transformers<5 约束与 qwentts-cpp 的 glibc≥2.39 要求）、
pip 钉版防解析回溯、`HF_HUB_OFFLINE=1` 离线加载（模型全缓存后启动不走网络）。

```bash
bash scripts/start_assistant.sh         # 重启三进程（改配置/加 persona 后）
bash scripts/start_assistant.sh stop    # 全停
.venv/bin/python scripts/smoke_pipeline.py --wav test_16k.wav   # 服务器冒烟
tail -f logs/{pipeline,avatar,orchestrator}.log                 # 排障
```

注意：TTS 与 FlashHead 都用 torch.compile，首次启动/首句合成各有数分钟编译预热，
看日志耐心等。实测显存：pipeline ~11.5GB（STT+VoxCPM2+VAD）+ avatar ~6.1GB，
合计 ~18/24GB——`server.num_pipelines` 务必保持 1，否则对话峰值会 OOM。

音画对齐（web/assistant.js）：数字人攒满 0.96s 音频才出帧，画面天然晚 ~0.9s；
前端把音频延迟 0.8s 播放（SDPA 时代 1.0s，flash_attn 后实测最优）、视频帧按序
从属于音频播放时钟（vlag=0），实测偏移 <0.1s。`?debug=1` 显示同步角标；
`?vlag=N` 调视频滞后帧；`?adelay=N` 调音频基础延迟；
`?solo=1` 单栏模式（隐藏摄像头画面、数字人居中，demo 录制用）。

## 女娲蒸馏（造新人设）

人设不是配置出来的，是蒸馏出来的。对 Agent 说「用女娲蒸馏 XX」
（`huashu-nuwa` skill），产出 `skills/<id>-perspective/`（完整思维框架）后，
把语音场景人设写进 `personas/<id>.md`（带 frontmatter 三件套），
在 `configs/assistant.yaml` 的 `personas.list` 注册，重启即可。
参考现成的：`fengge`（峰哥亡命天涯）。

## 局域网设备访问（可选）

Mac 本机走 SSH 隧道（`http://localhost:8000`）。同 LAN 的 iPhone/iPad 想用的话：

```bash
.venv/bin/python scripts/lan_https/proxy.py   # https://<Mac局域网IP>:8443 → 127.0.0.1:8000
```

iOS 安装并完全信任 `scripts/lan_https/ca.crt` 后即可用（安全上下文，麦克风可用；
需 iOS 15+——更老的 WebKit 的 wss 不认用户 CA，证书与密钥可用 openssl 重新生成，
均不入库）。注意管线是单会话（24GB 显存），同一时间只能一端对话。

## 本地开发（macOS，无 GPU）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v                    # 纯逻辑单测（不 import torch）
DEEPSEEK_API_KEY=sk-x .venv/bin/python -m voxemw.pipeline.launch --dry-run  # 验证配置渲染
```

GPU 进程（pipeline/avatar）只能在服务器跑；orchestrator 逻辑可本机单测。

## 目录

- `configs/assistant.yaml` — 唯一配置（六积木 + vision + filler + personas 注册表 + server）
- `voxemw/config.py` — YAML + .env + persona frontmatter/素材解析（纯逻辑）
- `voxemw/pipeline/` — s2s 集成：`args.py`（配置→argv 纯逻辑）、
  `stt_sensevoice.py` / `stt_qwen3asr.py` / `tts_voxcpm.py`（自定义积木）、`launch.py`（启动器）
- `voxemw/avatar/` — `service.py`（数字人服务，双后端）、
  `avtr1_engine.py`（AVTR-1 引擎，官方 scheduler 语义）、
  `orchestrator.py`（浏览器入口 + 双写编排 + 截帧注入 + 单会话顶掉）
- `voxemw/vision.py` — Kimi K3 多模态外貌描述（截帧打分，可缺席）
- `docs/upgrade-regression.md` — 上游 speech-to-speech 升级五阶段回归方案
- `personas/` — 女娲蒸馏的语音人设（frontmatter 绑定音色/形象）
- `skills/` — 女娲造人 skill 包（思维框架 + 研究笔记 + 选音色指南）
- `web/` — 视频通话聊天页（无构建：左栏摄像头 + 右栏数字人，AudioWorklet + canvas）
- `scripts/autodl_setup.sh` / `start_assistant.sh` / `smoke_pipeline.py` /
  `lan_https/`（局域网 HTTPS 反代）
- `assets/` — 音色/形象素材（个人材料，不入库）
- `tests/` — 纯逻辑单测（本机可跑）

## 合规

音色与肖像素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈。
与名人形象/音色相关的使用请遵守平台与法律规定。

## 相关链接

- speech-to-speech：https://github.com/huggingface/speech-to-speech
- SoulX-FlashHead：https://github.com/Soul-AILab/SoulX-FlashHead
- VoxCPM2：https://huggingface.co/openbmb/VoxCPM2
- SenseVoice（STT）：https://github.com/modelscope/FunASR
- DeepSeek API：https://platform.deepseek.com
- Kimi API（截帧打分的视觉模型）：https://platform.moonshot.cn
