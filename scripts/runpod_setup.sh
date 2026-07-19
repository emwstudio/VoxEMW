#!/usr/bin/env bash
# 良子语音机器人 —— Runpod pod 一键部署脚本
# 在 pod 上（单卡 RTX 4090 24GB）从项目根目录执行: bash scripts/runpod_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [0/6] 基础环境引导（裸 runpod/base 镜像无 pip/torch 时）"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq python3-pip python3-venv git ffmpeg curl
fi
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip

echo "==> [1/6] 安装 Python 依赖"
pip install --no-cache-dir -r agent/requirements.txt
# hf CLI 随 huggingface_hub 提供（vllm/transformers 会带入），确保最新
pip install -U "huggingface_hub[cli]"
# 驱动适配：默认 torch 轮子按最新 CUDA 构建，老驱动宿主机会报
# "NVIDIA driver is too old" —— CUDA 不可用时按驱动版本换装匹配轮子
if ! python -c "import torch; torch.cuda.init()" 2>/dev/null; then
    echo "    torch CUDA 不可用，换装 cu129 轮子（适配 CUDA 12.9 驱动）..."
    pip install --no-cache-dir --force-reinstall torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu129
    pip install --no-cache-dir "numpy<2.4"
fi

echo "==> [2/6] 预下载模型（约 15GB，首次较慢）"
# HF_TOKEN 在 .env.local / 环境变量中提供时自动生效
hf download Qwen/Qwen3-ASR-0.6B-hf
hf download Qwen/Qwen3-8B-AWQ
hf download openbmb/VoxCPM2

echo "==> [3/6] 检查配置"
if [ ! -f .env.local ]; then
    echo "ERROR: .env.local 不存在。请先 cp .env.example .env.local 并填入 LiveKit Cloud 凭证。" >&2
    exit 1
fi
# 载入 .env.local 供后续进程使用
set -a; source .env.local; set +a
: "${LIANGZI_REF_WAV:=assets/liangzi/ref.wav}"
: "${LIANGZI_REF_TEXT:=assets/liangzi/ref.txt}"
[ -f "$LIANGZI_REF_WAV" ]  || { echo "ERROR: 音色参考音频不存在: $LIANGZI_REF_WAV" >&2; exit 1; }
[ -f "$LIANGZI_REF_TEXT" ] || { echo "ERROR: 参考音频台词不存在: $LIANGZI_REF_TEXT" >&2; exit 1; }

echo "==> [4/6] 创建 vLLM 独立 venv 并启动（Qwen3-8B-AWQ, 端口 8000）"
# 为什么独立 venv：agent 需要 transformers git 新版（Qwen3-ASR，5.x dev），
# 而 vllm 0.11.x 钉死 transformers<5；且 PyPI 新版 vllm 只出 cu13 轮子
# （CUDA 12.9 驱动宿主机跑不了），vllm 0.11.2 是最后一个 cu128 时代版本。
# 24GB 显存三模型共用: vLLM 限 0.45, 给 VoxCPM2 (~8GB) 和 Qwen3-ASR (~2GB) 留余量
# VLLM_USE_FLASHINFER_SAMPLER=0: 无 CUDA toolkit 的镜像上 flashinfer sampler 会崩
VLLM_VENV=/root/venv-vllm
if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
    python3 -m venv "$VLLM_VENV"
    "$VLLM_VENV/bin/pip" install --no-cache-dir "vllm==${VLLM_VERSION:-0.11.2}"
fi
if ! curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "$VLLM_VENV/bin/vllm" serve "${LLM_MODEL:-Qwen/Qwen3-8B-AWQ}" \
        --port 8000 \
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.35}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
        > vllm.log 2>&1 &
    echo "    vLLM PID=$!，日志: vllm.log"
else
    echo "    vLLM 已在运行，跳过启动"
fi

echo "    等待 vLLM /health 就绪（模型加载需几分钟）..."
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "    vLLM 就绪"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: vLLM 启动超时，请查看 vllm.log" >&2
        exit 1
    fi
    sleep 5
done

echo "==> [5/6] 启动 LiveKit agent"
exec python agent/agent.py start
