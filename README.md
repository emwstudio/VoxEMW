# VoxEMW — 大胃袋良子语音机器人

单卡 24GB GPU 全链路本地推理，与「大胃袋良子」实时语音唠嗑：
STT + LLM + TTS（音色克隆）三个开源模型同卡共存，LiveKit Cloud 做实时传输。

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
       └─ 热进程（prewarm 常驻，job 来了 ~5s 进房）            ~8.4GB
            mic ─► silero VAD ─► Qwen3-ASR-0.6B（STT, transformers）
            text ─► vLLM（LLM, temperature 0.7, max 130 tokens）
            良子声音 ◄── VoxCPM2（TTS, 音色克隆 + 流式 48kHz, 头部裁剪）
```

- **STT**：Qwen3-ASR-0.6B（transformers 本地推理，silero VAD 分段；1.7B 更准但常驻设计下显存攀顶 OOM，见文档）
- **LLM**：Qwen3-8B-AWQ（vLLM 独立 venv，与 agent 的 transformers 5.x 隔离）
- **TTS**：openbmb/VoxCPM2（Ultimate Cloning 音色克隆，流式输出）
- **会话模型**：prewarm 常驻热进程 + 文件锁保证全机一套 STT+TTS + 负载闸门，单会话串行；挂断后进程回收，~30s 重建待命
- **实测**（RTX 3090）：进房 ~5s（热进程），端到端接话 ~2-3s，显存稳态 ~17.5GB / 24GB。详见 `docs/设计方案与性能数据.md`

## 部署（Runpod）

1. 建 pod：模板 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`，24GB 显卡，磁盘 ≥ 50GB
2. 同步代码；放入素材 `assets/liangzi/ref.wav` + `ref.txt`（见该目录说明）
3. `cp .env.example .env.local`，填入 LiveKit Cloud 凭证
4. `bash scripts/runpod_setup.sh`（装依赖 → 下模型 → 起 vLLM → 起 agent）
5. 打开对话页面（见下「Web 对话页面」）开始唠嗑；也可用 [Agents Playground](https://agents-playground.livekit.io/) 连你的项目，Agent name 填 `liangzi-agent`

## Web 对话页面

`frontend/` 是基于 LiveKit 官方 [agent-starter-react](https://github.com/livekit-examples/agent-starter-react)
（Next.js + Agents UI）的对话页面，说话时呈现 Aura 星云可视化；品牌定制集中在 `frontend/app-config.ts`。

```bash
cd frontend
cp .env.example .env.local   # 填同一份 LiveKit 凭证，并加 AGENT_NAME=liangzi-agent
pnpm install && pnpm dev     # 生产：pnpm build && pnpm start
```

浏览器开 http://localhost:3000，点「开始唠嗑」即可（需要 agent 已在 pod 上运行）。

Docker 方式（可选）：

```bash
docker build -t voxemw .
docker run --gpus all -v $PWD/.env.local:/app/.env.local voxemw
```

## 本地开发（macOS，无 GPU）

模型全部延迟加载，纯逻辑单测可直接跑：

```bash
python3 -m venv .venv && .venv/bin/pip install pytest numpy
.venv/bin/python -m pytest tests/ -v
```

## 目录

- `agent/` — LiveKit Agents 入口 + STT/TTS 插件 + 纯逻辑工具
- `frontend/` — Web 对话页面（LiveKit agent-starter-react + Aura 星云可视化）
- `tests/` — 纯逻辑单测（不 import torch/transformers/livekit）
- `scripts/` — pod 一键部署 / 容器入口
- `assets/liangzi/` — 音色克隆素材（自行提供，不入库）
- `skills/liangzi-perspective/` — 良子人设调研蒸馏（huashu-nuwa）
- `docs/` — 设计方案与实测数据

## 合规

音色克隆仅限本人授权或娱乐用途，禁止用于冒充、诈骗或误导性内容；AI 生成内容需明确标注。
「大胃袋良子」为真实网红，商业化使用前请确认肖像/声音授权。
