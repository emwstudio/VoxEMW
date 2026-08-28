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
    pkill -f "voxemw.gateway.orchestrator" 2>/dev/null || true
    echo "已停止"
    exit 0
fi

[ -x .venv-mac/bin/python ] || { echo "ERROR: 先跑 bash scripts/mac_setup.sh" >&2; exit 1; }

# 模型已全部预下载，离线加载；首次 silero 需要联网（torch.hub），有缓存后离线
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# 本机有全局代理（http_proxy 等）时，orchestrator→管线（127.0.0.1:8765）的 ws
# 会被 websockets 库按环境变量走路由代理导致握手 EOF，本机地址必须直连
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,::1"
export NO_PROXY="$no_proxy"

mkdir -p logs
pkill -f "voxemw.pipeline.launch" 2>/dev/null || true
pkill -f "voxemw.gateway.orchestrator" 2>/dev/null || true
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

# 视觉边车（妮儿的眼睛）：llama-server 跑 MiniCPM-V-4.6，GGUF 齐了才启动
VLM_DIR="$HOME/.cache/models/minicpm-v-4.6-gguf"
if [ -f "$VLM_DIR/MiniCPM-V-4_6-Q4_K_M.gguf" ] && [ -f "$VLM_DIR/mmproj-model-f16.gguf" ] && \
   command -v llama-server > /dev/null; then
    pkill -f "llama-server.*18099" 2>/dev/null || true
    echo "==> 启动视觉边车 llama-server（:18099）"
    nohup llama-server -m "$VLM_DIR/MiniCPM-V-4_6-Q4_K_M.gguf" \
        --mmproj "$VLM_DIR/mmproj-model-f16.gguf" \
        --host 127.0.0.1 --port 18099 > logs/vlm.log 2>&1 &
    echo "    PID=$!，日志 logs/vlm.log"
else
    echo "==> 跳过视觉边车（GGUF 或 llama-server 未就绪，视觉功能关闭）"
fi

echo "==> 启动 orchestrator（.venv-mac，:8000）"
# LAN TLS 入口（iPhone 用，https://<局域网IP>:9443）：证书在 scripts/lan_tls/，
# 由 scripts/make_lan_tls.sh 生成；没有证书就只开本机 http
if [ -f scripts/lan_tls/cert.pem ] && [ -f scripts/lan_tls/key.pem ]; then
    export VOX_TLS_CERT="$PWD/scripts/lan_tls/cert.pem"
    export VOX_TLS_KEY="$PWD/scripts/lan_tls/key.pem"
    echo "    （检测到 LAN 证书，将同时开 https :9443）"
fi
nohup .venv-mac/bin/python -m voxemw.gateway.orchestrator --config "$VOXEMW_CONFIG" \
    > logs/orchestrator.log 2>&1 &
echo "    PID=$!，日志 logs/orchestrator.log"

echo ""
echo "全部启动。浏览器打开 http://localhost:8000（星空 + VRM 数字人）"
echo "排障：tail -f logs/{pipeline,orchestrator}.log"
