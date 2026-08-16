#!/usr/bin/env bash
# Qwen3.8-27B 本地 LLM 服务（离线版大脑，llama-server :8081）
# 采样参数按 Unsloth 官方非思考模式推荐；--spec-type draft-mtp 开 MTP 投机解码
# （模型权重自带草稿层，实测 22.4→41.7 tok/s）
set -euo pipefail

LLAMA_BIN=/root/autodl-tmp/llama.cpp/build/bin/llama-server
LLAMA_MODEL="${LLAMA_MODEL:-/root/autodl-tmp/models/Qwen3.8-27B-UD-Q6_K_XL.gguf}"

if pgrep -f "llama-server" > /dev/null 2>&1; then
    echo "llama-server 已在跑，跳过"
    exit 0
fi

nohup "$LLAMA_BIN" -m "$LLAMA_MODEL" \
    --alias qwen38-local --host 127.0.0.1 --port 8081 \
    -ngl 99 -c 8192 -fa on --spec-type draft-mtp \
    --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5 \
    > /root/llama_server.log 2>&1 &
echo "llama-server started, pid $!"
