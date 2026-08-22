#!/usr/bin/env bash
# VoxEMW 本地 Mac 极简版 —— 一键启停（pipeline → orchestrator，两进程；
# 大脑走 DeepSeek API，本地不跑 LLM）
#
# 用法：bash scripts/start_mac.sh [stop]
set -euo pipefail
cd "$(dirname "$0")/.."

VOXEMW_CONFIG="${VOXEMW_CONFIG:-configs/assistant.yaml}"

if [ "${1:-}" = "stop" ]; then
    pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
    pkill -f "voxemw.avatar.orchestrator" 2>/dev/null || true
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
echo "排障：tail -f logs/{pipeline,orchestrator}.log"
