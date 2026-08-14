#!/usr/bin/env bash
# 重启数字人服务（AVTR-1，pixi env 直调——勿 pixi run，会按 lock 重同步 env
# 覆盖 pip 降级）。生成任务/换图/日常重启共用。
cd "$(dirname "$0")/.."

pkill -f "voxemw.avatar.service" 2>/dev/null || true
sleep 2
pkill -9 -f "voxemw.avatar.service" 2>/dev/null || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[ -d /root/autodl-tmp ] && export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"

AVTR_ENV=/root/autodl-tmp/avtr-1/.pixi/envs/renderer
SP=$AVTR_ENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$(echo $SP/nvidia/*/lib | tr " " ":"):${LD_LIBRARY_PATH:-}"
export AVTR1_LOCAL_STORAGE="${AVTR1_LOCAL_STORAGE:-/root/autodl-tmp/avtr1_storage}"

VOXEMW_CONFIG="${VOXEMW_CONFIG:-configs/assistant.yaml}"
nohup "$AVTR_ENV/bin/python" -m voxemw.avatar.service --config "$VOXEMW_CONFIG" \
    > logs/avatar.log 2>&1 &
echo "avatar restarted pid $!"
