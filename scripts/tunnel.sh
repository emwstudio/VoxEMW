#!/usr/bin/env bash
# VoxEMW 本地隧道（在 Mac 上跑）：
#   8000  → 网页/WS/信令
#   3478  → TURN 媒体中继（WebRTC 音视频走这条，缺了会失声/无画面）
# 实例换了就改环境变量：VOX_HOST=connect.xxx.seetacloud.com VOX_PORT=12345 bash scripts/tunnel.sh
set -euo pipefail

VOX_HOST="${VOX_HOST:-connect.bjb2.seetacloud.com}"
VOX_PORT="${VOX_PORT:-34567}"

echo "隧道 → ${VOX_HOST}:${VOX_PORT}（本地 8000=网页，3478=媒体）"
exec ssh -CNg \
  -L 8000:127.0.0.1:8000 \
  -L 3478:127.0.0.1:3478 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  "root@${VOX_HOST}" -p "${VOX_PORT}"
