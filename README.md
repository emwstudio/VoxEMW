# VoxEMW — 良子语音机器人

单卡 24GB GPU 全链路本地推理，与「良子」实时语音唠嗑：
STT + LLM + TTS（音色克隆）三个开源模型同卡共存，LiveKit Cloud 做实时传输。

## 相关链接

- 3090 显卡镜像（Runpod）：https://runpod.io?ref=r82lgade
- Kimi-K3 大模型：https://kimi-bot.com/activities/zh-cn/viral-referral/share?scenario=invite&from=share_poster&invitation_code=JJ5FSQ

## 架构

```
浏览器 ── frontend/（Next.js 对话页，Aura 星云 + 转写/打字）
  │ ① POST /api/token（LIVEKIT_* 凭证 mint token，指定 liangzi-agent）
  ▼
LiveKit Cloud（WebRTC 传输 + 派单）
  │ ② dispatch
  ▼
GPU Pod（RTX 3090 24GB）
  ├─ vLLM :8000 ── Qwen3-8B-AWQ（OpenAI 兼容）              ~9.5GB
  └─ agent worker（num_idle_processes=1）
       └─ 热进程（prewarm 常驻，job 来了 ~5s 进房）           ~10GB
            mic ─► silero VAD ─► Qwen3-ASR-1.7B（STT, transformers）
            text ─► vLLM（LLM, temperature 0.7, max 130 tokens）
            良子声音 ◄── VoxCPM2（TTS, 音色克隆 + 流式 48kHz）
```

- **STT**：Qwen3-ASR-1.7B（transformers 本地推理，silero VAD 分段）
- **LLM**：Qwen3-8B-AWQ（vLLM 独立 venv，与 agent 的 transformers 5.x 隔离）
- **TTS**：openbmb/VoxCPM2（Ultimate Cloning 音色克隆，流式输出）
- **会话模型**：prewarm 常驻热进程 + 文件锁保证全机一套 STT+TTS + 负载闸门，单会话串行；挂断后进程回收，~30s 重建待命
- **双唠模式**：峰哥×良子同卡相声——同一 vLLM 两份 system prompt 轮流问答、同一 VoxCPM2 按角色切音色，文本接力不走音频环路，**零新增显存**；dispatch metadata `duet` 分流，详见 `agent/duet.py`
- **实测**（RTX 3090）：进房 ~5s，端到端接话 ~1.5–2.5s，显存 ~20G / 24G。详见 `docs/设计方案与性能数据.md`

## 部署（Runpod）

1. 建 pod：模板 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`，24GB 显卡，磁盘 ≥ 50GB
2. 同步代码；放入素材 `assets/liangzi/ref.wav` + `ref.txt`（见该目录说明）
3. `cp .env.example .env.local`，填入 LiveKit Cloud 凭证
4. `bash scripts/runpod_setup.sh`（装依赖 → 下模型 → 起 vLLM → 起 agent）

Docker 方式（可选）：

```bash
docker build -t voxemw .
docker run --gpus all -v $PWD/.env.local:/app/.env.local voxemw
```

## Web 对话页面

`frontend/` 是基于 LiveKit 官方 [agent-starter-react](https://github.com/livekit-examples/agent-starter-react)
（Next.js + Agents UI）的对话页面，说话时呈现 Aura 星云可视化。

```bash
cd frontend
cp .env.example .env.local   # 填同一份 LiveKit 凭证，并加 AGENT_NAME=liangzi-agent
pnpm install && pnpm dev     # 生产：pnpm build && pnpm start
```

浏览器开 `localhost:3000`：绿键「开始唠嗑」跟良子打电话；琥珀键「听良子和峰哥唠嗑」围观两人说相声（需要 agent 已在 pod 上运行）。

## 本地开发（macOS，无 GPU）

模型全部延迟加载，纯逻辑单测可直接跑：

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy
.venv/bin/python -m pytest tests/ -v
```

## 目录

- `agent/` — LiveKit Agents 入口 + STT/TTS 插件 + 纯逻辑工具 + `duet.py`（峰哥×良子双唠引擎）
- `frontend/` — Web 对话页面（通话式 UI + Aura 星云可视化，单唠/双唠双模式）
- `tests/` — 纯逻辑单测（不 import torch/transformers/livekit）
- `scripts/` — pod 一键部署 / 容器入口
- `assets/liangzi/`、`assets/fengge/` — 音色克隆素材（自行提供，不入库）
- `skills/liangzi-perspective/`、`skills/fengge-perspective/` — 人设调研蒸馏（huashu-nuwa）
- `docs/` — 设计方案与实测数据

## 合规

音色克隆仅限本人授权或娱乐用途，禁止用于冒充、诈骗或误导性内容；AI 生成内容需明确标注。
「良子」「峰哥亡命天涯」均为真实网红，商业化使用前请确认肖像/声音授权。
