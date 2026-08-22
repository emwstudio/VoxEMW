#!/usr/bin/env bash
# VoxEMW 本地 Mac 极简版 —— 一键部署（Apple Silicon，无数字人/无唱歌）
#
# 目标环境：macOS arm64（M 系，实测目标 M5 16GB）
# 用法：仓库根目录执行  bash scripts/mac_setup.sh
# 幂等：重复执行不会重复装依赖/下模型。
#
# 与云端（scripts/autodl_setup.sh）的差异：
# - 走 .venv-mac（与跑测试的 .venv 隔离）；上游 s2s 的 Darwin 分支自动选
#   mlx 系依赖，不装 Linux 的 faster-qwen3-tts（stub wheel 那套不需要）
# - 无 llama（LLM 走 DeepSeek API）/ coturn（localhost 不需要 TURN）/
#   avatar（AVTR-1 是 TensorRT/NVIDIA 专用）
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/4] python3.12 + .venv-mac"
UV="${UV:-$HOME/.local/bin/uv}"
command -v "$UV" > /dev/null 2>&1 || { echo "ERROR: 需要 uv（curl -LsSf https://astral.sh/uv/install.sh | sh）" >&2; exit 1; }
PY312="$("$UV" python find 3.12 2>/dev/null || true)"
if [ -z "$PY312" ]; then
    echo "    本机无 python3.12，用 uv 装一个到 uv 托管目录..."
    "$UV" python install 3.12
    PY312="$("$UV" python find 3.12)"
fi
[ -x .venv-mac/bin/python ] || "$UV" venv --python "$PY312" .venv-mac
echo "    .venv-mac: $(.venv-mac/bin/python --version)"

echo "==> [2/4] 依赖（torch mac 版 + s2s 钉死版 + 自定义积木）"
# 注意：本机有代理时直连 huggingface.co 反而比 hf-mirror 稳（2026-08-22 实测
# hf-mirror 在该网络下报 Local entry not found，直连秒下）。要强制镜像：
# HF_ENDPOINT=https://hf-mirror.com bash scripts/mac_setup.sh
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# uv venv 不带 pip，一律 uv pip（装得快）
UVPIP=("$UV" pip install --python .venv-mac/bin/python)
# torch/torchaudio 走 PyPI 默认源即可（mac wheel 无 CUDA 后缀问题）
if .venv-mac/bin/python -c "import speech_to_speech, aiortc, av" > /dev/null 2>&1; then
    echo "    依赖已装齐，跳过"
else
    "${UVPIP[@]}" -q torch torchaudio   # Mac 版（MPS 支持内置）
    # s2s 钉死上游 commit（2026-08-21 main：含转写失败保留 + reasoning_effort 透传修复）；
    # Darwin 分支自动装 mlx 系
    "${UVPIP[@]}" -q "speech-to-speech @ git+https://github.com/huggingface/speech-to-speech.git@9f59bc72f66ee84b006e2682b9547144e7f74827"
    "${UVPIP[@]}" -q aiortc av Pillow aiohttp websockets numpy scipy pyyaml accelerate
    # 本机有 SOCKS 代理环境变量时，hf 的 httpx 需要 socksio，
    # 否则下载直接 ImportError（2026-08-22 实测踩坑）
    "${UVPIP[@]}" -q "httpx[socks]"
fi
# torchcodec 在 Mac 没有 CUDA 问题，但与 transformers 5.x 共存仍有坑的话卸载它
"$UV" pip uninstall --python .venv-mac/bin/python -q torchcodec 2>/dev/null || true

echo "==> [3/4] 模型预下载（~4G，HF 直连）"
.venv-mac/bin/hf download pipecat-ai/smart-turn-v3 > /dev/null && echo "    SmartTurn ✓"
.venv-mac/bin/hf download Qwen/Qwen3-ASR-0.6B-hf > /dev/null && echo "    Qwen3-ASR-0.6B-hf ✓"
.venv-mac/bin/hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit > /dev/null && echo "    Qwen3-TTS-1.7B-Base-6bit ✓"

echo "==> [4/4] 配置检查"
if [ ! -f .env.local ] || ! grep -q "^DEEPSEEK_API_KEY=sk-" .env.local; then
    echo "ERROR: .env.local 缺少 DEEPSEEK_API_KEY（DeepSeek 控制台创建后写入）" >&2
    exit 1
fi
[ -f configs/assistant.yaml ] || { echo "ERROR: 缺 configs/assistant.yaml" >&2; exit 1; }

cat <<'EOF'

==> 部署完成。启动：

    bash scripts/start_mac.sh       # 起 pipeline + orchestrator
    open http://localhost:8000      # 浏览器打开（无数字人，纯语音）

    排障：tail -f logs/pipeline.log logs/orchestrator.log
EOF
