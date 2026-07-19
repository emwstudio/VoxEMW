#!/usr/bin/env bash
# 容器入口: 启动 vLLM（后台）+ LiveKit agent（前台）。供 Dockerfile CMD 使用。
set -euo pipefail
cd /app

# 载入 .env.local（若挂载/拷贝了该文件）
if [ -f .env.local ]; then
    set -a; source .env.local; set +a
fi

echo "==> 启动 vLLM（${LLM_MODEL:-Qwen/Qwen3-8B-AWQ}, 端口 8000）"
# vLLM 用独立 venv（构建期创建，vllm 0.11.2 = 最后 cu128 时代版本，
# 避免与 agent venv 的 transformers 5.x 冲突）
# VLLM_USE_FLASHINFER_SAMPLER=0: 无 CUDA toolkit 的镜像上 flashinfer sampler 会崩
VLLM_USE_FLASHINFER_SAMPLER=0 \
nohup /root/venv-vllm/bin/vllm serve "${LLM_MODEL:-Qwen/Qwen3-8B-AWQ}" \
    --port 8000 \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.35}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
    > vllm.log 2>&1 &

echo "==> 等待 vLLM /health 就绪..."
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

echo "==> 启动 LiveKit agent"
exec python agent/agent.py start
