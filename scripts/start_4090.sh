#!/bin/bash
# VoxEMW 4090 满血版一键启动（AutoDL 实例）
# 用法：bash scripts/start_4090.sh [stop]
# 四个进程：语音管线(py312) → SoulX 渲染(flashhead) → VLM 边车(py312) → orchestrator(py312)
set -u
cd /root/voxemw
source /root/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/root/autodl-tmp/hf HF_HUB_DISABLE_XET=1 VOXEMW_CONFIG=configs/assistant-4090.yaml
mkdir -p logs

if [ "${1:-}" = "stop" ]; then
  pkill -f 'voxemw.pipeline.launc[h]' ; pkill -f 'voxemw.gateway.orchestrato[r]' ; pkill -f 'soulx_serve[r]' ; pkill -f 'vlm_serve[r]'
  echo "已全部停止"; exit 0
fi

source /etc/network_turbo >/dev/null 2>&1 || true  # HF 连通性检查走学术加速

echo "[1/3] 语音管线（py312）..."
conda activate py312
nohup python -m voxemw.pipeline.launch > logs/pipeline.log 2>&1 &
echo "  日志 logs/pipeline.log（warmup 约 3-4 分钟）"

echo "[2/3] SoulX 数字人渲染（flashhead）..."
conda activate flashhead
PYTHONPATH=/root/voxemw nohup python voxemw/avatar/soulx_server.py > logs/soulx.log 2>&1 &
echo "  日志 logs/soulx.log（引擎加载 ~1 分钟）"

echo "[3/4] MiniCPM-V 视觉边车（vlm env，transformers 5.7.0 钉版，36 切片 OCR 档）..."
conda activate vlm
PYTHONPATH=/root/voxemw PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   nohup python voxemw/gateway/vlm_server.py --port 18099 > logs/vlm.log 2>&1 &
echo "  日志 logs/vlm.log（加载 ~30s）"

echo "[4/4] orchestrator（py312）..."
conda activate py312
nohup python -m voxemw.gateway.orchestrator --config configs/assistant-4090.yaml > logs/orchestrator.log 2>&1 &
sleep 5
tail -2 logs/orchestrator.log
echo
echo "完成。本地架隧道：ssh -NL 8000:127.0.0.1:8000 -p 46729 root@connect.bjb2.seetacloud.com"
echo "然后浏览器开 http://localhost:8000"
