#!/usr/bin/env bash
# VoxEMW（峰哥反指提示器）—— AutoDL 实例一键部署脚本
#
# 目标环境：AutoDL Miniconda 镜像 + 单卡 RTX 4090D（Linux, CUDA 12.8 驱动）
# 用法：rsync 仓库到实例后，在仓库根目录执行  bash scripts/autodl_setup.sh
# 幂等：重复执行不会重复装依赖/下模型，已在跑的服务不重启。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [0/5] 基础环境（conda python3.12 + venv）"
# 非交互 SSH 下 conda 可能不在 PATH
if ! command -v conda > /dev/null 2>&1 && [ -x /root/miniconda3/bin/conda ]; then
    export PATH="/root/miniconda3/bin:$PATH"
fi
command -v conda > /dev/null 2>&1 || { echo "ERROR: 无 conda，请换 AutoDL Miniconda 镜像" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q '^py312 '; then
    echo "    py312 已存在"
else
    echo "    conda 新建 py312（python 3.12）..."
    # 默认源在国内拉 metadata 极慢，先配清华镜像
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 2>/dev/null || true
    conda config --set show_channel_urls no 2>/dev/null || true
    conda create -y -n py312 python=3.12
fi
CONDA_PY312="$(conda info --base)/envs/py312/bin/python"

# 系统依赖：ffmpeg（音频）、gcc（部分包编译）
MISSING_PKGS=""
command -v gcc    > /dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS build-essential"
command -v ffmpeg > /dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS ffmpeg"
command -v curl   > /dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS curl"
if [ -n "$MISSING_PKGS" ]; then
    apt-get update -qq
    # shellcheck disable=SC2086
    apt-get install -y -qq $MISSING_PKGS
fi

[ -x .venv/bin/python ] || "$CONDA_PY312" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip

# 境内网络：pip 走清华镜像（可用 PIP_INDEX_URL 覆盖），HF 走 hf-mirror
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# hf-xet 会绕开镜像直连 HF 的 CAS 服务器（国内 401/超时），禁掉走普通 HTTP 下载
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
pip config set global.index-url "$PIP_INDEX_URL" > /dev/null 2>&1 || true
# 模型缓存放数据盘（关机保留，不占系统盘）
if [ -d /root/autodl-tmp ]; then
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
fi

echo "==> [1/5] 安装 torch 2.8 + torchaudio 2.8（cu128 轮子）"
# torchaudio 必须同版本同 index 钉死：否则 pip 会从默认源拉最新版（cu13），
# 报 libcudart.so.13 缺失
if python -c "import torch, torchaudio; assert torch.__version__.startswith('2.8') and torchaudio.__version__.startswith('2.8')" > /dev/null 2>&1; then
    echo "    torch 已安装：$(python -c 'import torch; print(torch.__version__)')"
else
    # pytorch 官方 index；被墙可换清华镜像：
    #   pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128
    pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
fi

echo "==> [2/5] 安装服务依赖（voxcpm / aiohttp / numpy / scipy / pyyaml）"
pip install --no-cache-dir voxcpm aiohttp numpy scipy pyyaml "huggingface_hub[cli]"

echo "==> [3/5] 预下载模型到数据盘（HF_HOME=${HF_HOME:-默认}）"
# hf download 幂等（已下载会校验后跳过）；hf-mirror 下 openbmb 可用
hf download openbmb/VoxCPM2

echo "==> [4/5] 检查配置"
if [ ! -f .env.local ]; then
    echo "ERROR: .env.local 不存在。请 cp .env.example .env.local 并填入 DEEPSEEK_API_KEY。" >&2
    exit 1
fi
set -a; source .env.local; set +a
if [ -z "${DEEPSEEK_API_KEY:-${LLM_API_KEY:-}}" ]; then
    echo "ERROR: .env.local 里 DEEPSEEK_API_KEY（或 LLM_API_KEY）为空。" >&2
    exit 1
fi
VOXEMW_CONFIG="${VOXEMW_CONFIG:-configs/alerter.yaml}"
[ -f "$VOXEMW_CONFIG" ] || { echo "ERROR: 配置不存在: $VOXEMW_CONFIG" >&2; exit 1; }

echo "==> [5/5] 启动服务"
mkdir -p logs
if pgrep -f "voxemw.server" > /dev/null 2>&1; then
    echo "    服务已在运行，跳过（改配置/加音色需先 pkill -f voxemw.server 再重跑本脚本）"
else
    nohup python -m voxemw.server --config "$VOXEMW_CONFIG" > logs/alerter.log 2>&1 &
    echo "    voxemw.server PID=$!，日志 logs/alerter.log（http :8000，静态页 + /api/*）"
fi

cat <<'EOF'

==> 部署完成。本机访问方式（SSH 隧道，AutoDL 默认不开公网端口）：

    ssh -CNg -L 8000:127.0.0.1:8000 root@<实例主机> -p <SSH端口>

    浏览器打开  http://localhost:8000  → 警报页：语音查询 + 反指短报轮询播报。

    排障：tail -f logs/alerter.log
EOF
