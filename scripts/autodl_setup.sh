#!/usr/bin/env bash
# VoxEMW（搭积木语音助手）—— AutoDL 实例一键部署脚本
#
# 目标环境：AutoDL Miniconda 镜像 + 单卡 RTX 4090D（Linux, CUDA 12.8 驱动）
# 用法：rsync 仓库到实例后，在仓库根目录执行  bash scripts/autodl_setup.sh
# 幂等：重复执行不会重复装依赖/下模型/重复打 patch，已在跑的服务不重启。
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

echo "==> [0/7] 基础环境（conda python3.12 + venv）"
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

# 系统依赖：git（打 patch）、ffmpeg（音频）、gcc（部分包编译）
MISSING_PKGS=""
command -v gcc    > /dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS build-essential"
command -v ffmpeg > /dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS ffmpeg"
command -v git    > /dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS git"
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

echo "==> [1/7] 安装 torch 2.8 + torchaudio 2.8（cu128 轮子）"
# torchaudio 必须同版本同 index 钉死：否则 pip 会从默认源拉最新版（cu13），
# 报 libcudart.so.13 缺失
if python -c "import torch, torchaudio; assert torch.__version__.startswith('2.8') and torchaudio.__version__.startswith('2.8')" > /dev/null 2>&1; then
    echo "    torch 已安装：$(python -c 'import torch; print(torch.__version__)')"
else
    # pytorch 官方 index；被墙可换清华镜像：
    #   pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128
    pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
fi

echo "==> [2/7] 给 vendor/speech-to-speech 打自定义后端 patch"
PATCH="$REPO_ROOT/patches/register-handlers.patch"
cd "$REPO_ROOT/vendor/speech-to-speech"
if git apply --check "$PATCH" > /dev/null 2>&1; then
    git apply "$PATCH"
    echo "    patch 已应用"
elif git apply -R --check "$PATCH" > /dev/null 2>&1; then
    echo "    patch 已存在，跳过"
else
    echo "ERROR: patch 既打不上也不是已应用状态，vendor 目录可能被改过" >&2
    exit 1
fi

echo "==> [3/7] 安装 speech-to-speech（-e）与 voxcpm"
# Ubuntu 20.04（glibc 2.31）装不上 qwentts-cpp-python（wheelhouse 轮子都要 glibc≥2.35），
# 把 ggml extra 从依赖里去掉；qwen3 TTS 兜底走 --qwen3_tts_backend torch（我们主用 voxcpm）
sed -i 's/faster-qwen3-tts\[ggml\]/faster-qwen3-tts/' pyproject.toml
pip install --no-cache-dir -e .
pip install --no-cache-dir voxcpm
pip install -U "huggingface_hub[cli]" pyyaml

echo "==> [4/7] 生成 web/personas.json"
cd "$REPO_ROOT"
python scripts/build_personas.py

echo "==> [5/7] 预下载模型到数据盘（HF_HOME=$HF_HOME）"
# hf download 幂等（已下载会校验后跳过）；hf-mirror 下 Qwen/openbmb 均可用
hf download Qwen/Qwen3-ASR-1.7B-hf
hf download openbmb/VoxCPM2

# silero VAD 预置：torch.hub 会从 GitHub 拉 master.zip，国内基本拉不动；
# 走学术加速（部分机房有）预下载并放到 torch.hub 缓存目录，VAD handler 直接复用
SILERO_DIR="$HOME/.cache/torch/hub/snakers4_silero-vad_master"
if [ -d "$SILERO_DIR" ]; then
    echo "    silero-vad 已预置"
else
    echo "    预下载 silero-vad（torch.hub 缓存）..."
    TMP_ZIP="$(mktemp --suffix=.zip)"
    if [ -f /etc/network_turbo ]; then
        # shellcheck disable=SC1091
        (source /etc/network_turbo > /dev/null 2>&1; curl -sL -m 300 -o "$TMP_ZIP" https://codeload.github.com/snakers4/silero-vad/zip/refs/heads/master)
    else
        curl -sL -m 600 -o "$TMP_ZIP" https://codeload.github.com/snakers4/silero-vad/zip/refs/heads/master
    fi
    mkdir -p "$HOME/.cache/torch/hub"
    (cd "$HOME/.cache/torch/hub" && unzip -q "$TMP_ZIP" && mv silero-vad-master snakers4_silero-vad_master)
    rm -f "$TMP_ZIP"
fi

echo "==> [6/7] 检查配置"
if [ ! -f .env.local ]; then
    echo "ERROR: .env.local 不存在。请 cp .env.example .env.local 并填入 DEEPSEEK_API_KEY。" >&2
    exit 1
fi
set -a; source .env.local; set +a
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: .env.local 里 DEEPSEEK_API_KEY 为空（configs/autodl-4090.yaml 的 llm.api_key_env 需要它）。" >&2
    exit 1
fi
S2S_CONFIG="${S2S_CONFIG:-configs/autodl-4090.yaml}"
[ -f "$S2S_CONFIG" ] || { echo "ERROR: 配置不存在: $S2S_CONFIG" >&2; exit 1; }
# 干跑一遍渲染，配置有问题当场报，不必等进程起来再挂
python launch.py --config "$S2S_CONFIG" --dry-run > /dev/null

echo "==> [7/7] 启动服务"
mkdir -p logs
if pgrep -f "speech_to_speech.s2s_pipeline" > /dev/null 2>&1; then
    echo "    speech-to-speech 已在运行，跳过（改配置/换音色需先 pkill -f speech_to_speech 再重跑本脚本）"
else
    nohup python launch.py --config "$S2S_CONFIG" > logs/s2s.log 2>&1 &
    echo "    speech-to-speech PID=$!，日志 logs/s2s.log（ws :8765/v1/realtime）"
fi
if pgrep -f "http.server 8000" > /dev/null 2>&1; then
    echo "    web 静态服务已在运行，跳过"
else
    nohup python3 -m http.server 8000 --directory web > logs/web.log 2>&1 &
    echo "    web 静态服务 PID=$!，日志 logs/web.log（:8000）"
fi

cat <<'EOF'

==> 部署完成。本机访问方式（SSH 隧道，AutoDL 默认不开公网端口）：

    ssh -CNg -L 8000:127.0.0.1:8000 -L 8765:127.0.0.1:8765 root@<实例主机> -p <SSH端口>

    浏览器打开  http://localhost:8000  → 选角色 → 连接 → 说话。

    排障：tail -f logs/s2s.log
EOF
