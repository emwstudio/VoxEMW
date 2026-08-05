# VoxEMW · 数字人实时语音聊天助手

对着浏览器说话，屏幕里的数字人开口回答你。人设、音色、形象三位一体，
单卡 RTX 4090D 即可运行，**你说完到听到第一声 ≈ 2.4s**。

链路基于 [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
（VAD → STT → LLM → TTS 实时管线）+ [AVTR-1](https://github.com/avaturn-live/avtr-1)
数字人形象（TensorRT，0.2s 生成粒度）。人设来自女娲蒸馏（`personas/<id>.md`）。

## 架构

```
浏览器（web/，摄像头 + 数字人画面）
   │  ws :8000
   ▼
orchestrator（CPU）
   ├─→ s2s 语音管线（:8765）  VAD → STT SenseVoice → LLM DeepSeek → TTS VoxCPM2
   ├─→ avatar 数字人服务（:8767）  AVTR-1（默认）/ FlashHead（回退）
   └─→ vision（Kimi K3，可选）  截帧 → 外貌描述 → 注入对话锐评
```

orchestrator 把 TTS 音频双写：一路回浏览器播放，一路喂数字人驱动口型。
avatar/vision 缺席时自动降级纯语音模式。单用户单会话，新连接自动顶掉旧会话。

## 延迟分解（4090D 实测）

| 环节 | 耗时 |
|---|---|
| VAD 判停 | ~0.5s |
| STT（SenseVoiceSmall）| ~0.1s |
| LLM 首句（DeepSeek v4-flash 流式）| ~1.4s |
| TTS 首音（VoxCPM2 流式）| ~0.1s |
| 唇同步缓冲（AVTR-1）| 0.35s |

## 积木（configs/assistant.yaml）

| 积木 | 选型 |
|---|---|
| vad | silero-vad（s2s 内置，判停 500ms） |
| stt | `iic/SenseVoiceSmall`（`qwen3asr` 备选） |
| llm | DeepSeek `deepseek-v4-flash`（流式逐句送 TTS） |
| tts | `openbmb/VoxCPM2`（克隆音色，`tts.rate` 变速补偿） |
| avatar | `avaturn-live/avtr-1`（默认）/ `SoulX-FlashHead-1_3B`（回退） |
| persona | `personas/<id>.md`（女娲蒸馏，frontmatter 绑定音色三件套） |
| vision | Kimi K3（可选，截帧打分） |

## 部署（AutoDL RTX 4090D）

```bash
cp .env.example .env.local   # 填 DEEPSEEK_API_KEY（必填）、KIMI_API_KEY（可选）
bash scripts/autodl_setup.sh # 幂等：环境 + 模型 + 启动三进程

ssh -CNg -L 8000:127.0.0.1:8000 root@<实例> -p <端口>
# 打开 http://localhost:8000（?solo=1 为 16:9 全屏录制模式）
```

启停：`bash scripts/start_assistant.sh [stop]`（按 `avatar.backend` 自动选环境）。
TTS 与 avatar 首次启动各有数分钟编译预热，看日志耐心等。

**数字人后端**：默认 AVTR-1（0.2s/chunk，唇同步缓冲 0.35s，双流倾听、
静音微动）；FlashHead 回退（画质更锐但缓冲 0.8s）。
**参考图规范**：16:9 横版胸像（1920×1080 为佳），脸宽占图宽 ~20%、
头顶留白 ~19%，微张嘴/露齿者口型最佳——构图不对脸型会失真。

## 玩法

- **截帧打分**：对数字人说「让我好好看看你」，自动截一帧摄像头画面让
  persona 锐评你的外表（需 KIMI_API_KEY）
- **倾听反应**：你说话时数字人对你的声音实时做出注视/微表情（AVTR-1 双流）
- **人设切换**：前端点 chip，LLM 人设 / TTS 音色 / 数字人肖像三路同切
- **局域网设备**：`scripts/lan_https/proxy.py`（iPhone/iPad 用，需装 CA）

## 本地开发（macOS，无 GPU）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v   # 纯逻辑单测
```

## 目录

- `configs/assistant.yaml` — 唯一配置
- `voxemw/pipeline/` — s2s 集成（STT/TTS 积木 + 启动器）
- `voxemw/avatar/` — 数字人服务（`service.py` 双后端、`avtr1_engine.py`）+ orchestrator
- `voxemw/vision.py` — Kimi 多模态外貌描述
- `web/` — 通话页（无构建）
- `personas/` / `skills/` — 人设与女娲造人 skill 包
- `docs/upgrade-regression.md` — 上游升级回归方案
- `tests/` — 纯逻辑单测

## 合规

音色与肖像素材由使用者本人提供/授权；AI 生成内容需标注，不得用于冒充、欺诈。

## 相关链接

- speech-to-speech: https://github.com/huggingface/speech-to-speech
- AVTR-1: https://github.com/avaturn-live/avtr-1
- SoulX-FlashHead: https://github.com/Soul-AILab/SoulX-FlashHead
- VoxCPM2: https://huggingface.co/openbmb/VoxCPM2
- FunASR（SenseVoice）: https://github.com/modelscope/FunASR
- DeepSeek API: https://platform.deepseek.com
- Kimi API: https://platform.moonshot.cn
