#!/usr/bin/env bash
# 斗地主语音局启动脚本（AutoDL 实例，幂等）。
# 与聊天管线互斥：先停 s2s_pipeline，再起 doudizhu.server（8766）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> 停掉聊天管线（如果在跑）"
pkill -f "s2s_pipelin[e]" 2>/dev/null || true
sleep 2

echo "==> 停掉旧的斗地主服务（如果在跑）"
pkill -f "doudizhu.serve[r]" 2>/dev/null || true
sleep 1

source .venv/bin/activate
set -a; source .env.local; set +a
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}

mkdir -p logs
echo "==> 起斗地主服务（模型加载+编译约 3-5 分钟，tail -f logs/game.log 观察）"
nohup python -u -m doudizhu.server --config configs/doudizhu.yaml > logs/game.log 2>&1 &
echo "STARTED=$!"
