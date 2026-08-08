#!/usr/bin/env bash
# VoxEMW 数字人语音助手 —— 一键启停（三进程同卡：avatar → pipeline → orchestrator）
#
# 用法：bash scripts/start_assistant.sh [stop]
# 顺序：数字人服务（GPU）→ s2s 语音管线（GPU）→ orchestrator（CPU，:8000 对外）。
set -euo pipefail
cd "$(dirname "$0")/.."

VOXEMW_CONFIG="${VOXEMW_CONFIG:-configs/assistant.yaml}"

# torch/OpenMP 线程默认按核数起且忙等自旋:三个 torch 进程同时推理时空转占满核、
# load 爆 40+、实时流抖动卡顿。限 4 线程 + 被动等待
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# 模型已全部预下载到数据盘,离线加载:避免每次启动都向 hf-mirror 发校验请求
# (网络抖动时 AutoProcessor.from_pretrained 会直接挂);
# HF_HOME 必须显式指到数据盘——非 setup 上下文启动时默认 ~/.cache/huggingface 是空的
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[ -d /root/autodl-tmp ] && export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
# nltk 等启动期检查会连 GitHub raw（直连被墙会卡死管线启动），开学术加速兜底
source /etc/network_turbo >/dev/null 2>&1 || true

if [ "${1:-}" = "stop" ]; then
    pkill -f "voxemw.avatar.service" 2>/dev/null || true
    pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
    pkill -f "voxemw.avatar.orchestrator" 2>/dev/null || true
    pkill -f "turnserver.*turnserver.conf" 2>/dev/null || true
    echo "已停止全部服务"
    exit 0
fi

mkdir -p logs
# 先停旧进程（避免显存/端口占用冲突）
pkill -f "voxemw.avatar.service" 2>/dev/null || true
pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
pkill -f "voxemw.avatar.orchestrator" 2>/dev/null || true
pkill -f "turnserver.*turnserver.conf" 2>/dev/null || true
sleep 2

# TURN（coturn）：WebRTC 媒体中继，SSH 隧道用户必经（UDP 过不了隧道，
# 浏览器走 turn:localhost:3478?transport=tcp，隧道需加 -L 3478:localhost:3478）
echo "==> 启动 TURN（coturn，loopback :3478）"
nohup turnserver -c configs/turnserver.conf > logs/turn.log 2>&1 &
echo "    PID=$!，日志 logs/turn.log"

# AVTR-1：pixi env python 直调（勿 pixi run——会按 lock 重同步 env，
# 把 pip 降级的 onnxruntime-gpu 1.22 还原成 1.28）
AVTR_ENV=/root/autodl-tmp/avtr-1/.pixi/envs/renderer
SP=$AVTR_ENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(echo $SP/nvidia/*/lib | tr " " ":"):${LD_LIBRARY_PATH:-}"
export AVTR1_LOCAL_STORAGE="${AVTR1_LOCAL_STORAGE:-/root/autodl-tmp/avtr1_storage}"
echo "==> 启动数字人服务（AVTR-1，pixi env，:8767）"
nohup "$AVTR_ENV/bin/python" -m voxemw.avatar.service --config "$VOXEMW_CONFIG" \
    > logs/avatar.log 2>&1 &
echo "    PID=$!，日志 logs/avatar.log（预热要数分钟，耐心等）"

echo "==> 启动语音管线（.venv，:8765）"
nohup .venv/bin/python -m voxemw.pipeline.launch --config "$VOXEMW_CONFIG" \
    > logs/pipeline.log 2>&1 &
echo "    PID=$!，日志 logs/pipeline.log（TTS torch.compile 也要一两分钟）"

echo "==> 等待语音管线 ws 就绪..."
for i in $(seq 1 120); do
    if grep -q "Uvicorn running" logs/pipeline.log 2>/dev/null; then
        break
    fi
    sleep 5
done

echo "==> 启动 orchestrator（.venv，:8000 对外）"
nohup .venv/bin/python -m voxemw.avatar.orchestrator --config "$VOXEMW_CONFIG" \
    > logs/orchestrator.log 2>&1 &
echo "    PID=$!，日志 logs/orchestrator.log"

echo ""
echo "全部启动。排障：tail -f logs/{avatar,pipeline,orchestrator}.log"
