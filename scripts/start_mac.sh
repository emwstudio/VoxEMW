#!/usr/bin/env bash
# VoxEMW 本地 Mac 极简版 —— 一键启停（llama → pipeline → orchestrator，三进程）
#
# 用法：bash scripts/start_mac.sh [stop]
set -euo pipefail
cd "$(dirname "$0")/.."

VOXEMW_CONFIG="${VOXEMW_CONFIG:-configs/assistant.yaml}"

if [ "${1:-}" = "stop" ]; then
    pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
    pkill -f "voxemw.avatar.orchestrator" 2>/dev/null || true
    pkill -f "llama-server" 2>/dev/null || true
    echo "已停止"
    exit 0
fi

[ -x .venv-mac/bin/python ] || { echo "ERROR: 先跑 bash scripts/mac_setup.sh" >&2; exit 1; }

# 模型已全部预下载，离线加载；首次 silero 需要联网（torch.hub），有缓存后离线
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p logs
pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
pkill -f "voxemw.avatar.orchestrator" 2>/dev/null || true
sleep 1

# 本地 LLM（llama-server :8081，MiniCPM-V-4.6 Q4_K_M + 视觉投影）：
# --reasoning off（思考链会吃光 max_tokens 且语音场景要直答）；
# --jinja（chat template，人设/工具调用的前提）
LLAMA_MODEL="$HOME/models/minicpmv46/MiniCPM-V-4_6-Q4_K_M.gguf"
LLAMA_MMPROJ="$HOME/models/minicpmv46/mmproj-model-f16.gguf"
if [ -f "$LLAMA_MODEL" ]; then
    if pgrep -f "llama-server" > /dev/null 2>&1; then
        echo "==> 本地 LLM 已在跑（:8081），跳过"
    else
        echo "==> 启动本地 LLM（MiniCPM-V-4.6，:8081）"
        nohup llama-server -m "$LLAMA_MODEL" --mmproj "$LLAMA_MMPROJ" \
            --alias minicpmv46-local --host 127.0.0.1 --port 8081 \
            -ngl 99 -c 8192 --jinja --reasoning off \
            > logs/llama_server.log 2>&1 &
        echo "    PID=$!，日志 logs/llama_server.log"
        echo "==> 等待本地 LLM 就绪..."
        for i in $(seq 1 30); do
            grep -q "listening on" logs/llama_server.log 2>/dev/null && break
            sleep 3
        done
    fi
else
    echo "!! 本地 LLM 模型不存在（$LLAMA_MODEL），跳过（管线会因连不上大脑报错）"
fi

echo "==> 启动语音管线（.venv-mac，:8765）"
nohup .venv-mac/bin/python -m voxemw.pipeline.launch --config "$VOXEMW_CONFIG" \
    > logs/pipeline.log 2>&1 &
echo "    PID=$!，日志 logs/pipeline.log"

echo "==> 等待语音管线 ws 就绪..."
for i in $(seq 1 60); do
    if grep -q "Uvicorn running" logs/pipeline.log 2>/dev/null; then
        break
    fi
    sleep 5
done

echo "==> 启动 orchestrator（.venv-mac，:8000）"
nohup .venv-mac/bin/python -m voxemw.avatar.orchestrator --config "$VOXEMW_CONFIG" \
    > logs/orchestrator.log 2>&1 &
echo "    PID=$!，日志 logs/orchestrator.log"

echo ""
echo "全部启动。浏览器打开 http://localhost:8000（纯语音模式，无数字人）"
echo "排障：tail -f logs/{llama_server,pipeline,orchestrator}.log"
