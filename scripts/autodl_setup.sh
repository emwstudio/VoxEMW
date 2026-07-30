#!/usr/bin/env bash
# VoxEMW 数字人实时语音助手 —— AutoDL 实例一键部署脚本
#
# 目标环境：AutoDL Miniconda 镜像 + 单卡 RTX 4090D（Linux, CUDA 12.8 驱动）
# 用法：rsync 仓库到实例后，在仓库根目录执行  bash scripts/autodl_setup.sh
# 幂等：重复执行不会重复装依赖/下模型，已在跑的服务不重启。
#
# 双 venv 隔离（依赖互相冲突，不可合装）：
#   .venv         py312 + torch 2.8：s2s 语音管线（transformers 5.13 / Qwen3-ASR / VoxCPM2）+ orchestrator
#   .venv-avatar  py310 + torch 2.7.1：FlashHead 数字人（transformers==4.57.3 / xformers / flash_attn）
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [0/7] 基础环境（conda + 系统包）"
# 非交互 SSH 下 conda 可能不在 PATH
if ! command -v conda > /dev/null 2>&1 && [ -x /root/miniconda3/bin/conda ]; then
    export PATH="/root/miniconda3/bin:$PATH"
fi
command -v conda > /dev/null 2>&1 || { echo "ERROR: 无 conda，请换 AutoDL Miniconda 镜像" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 2>/dev/null || true
conda config --set show_channel_urls no 2>/dev/null || true
for spec in py312:3.12 py310:3.10; do
    env="${spec%%:*}"
    ver="${spec##*:}"
    if conda env list | grep -q "^$env "; then
        echo "    $env 已存在"
    else
        conda create -y -n "$env" "python=$ver"
    fi
done
CONDA_BASE="$(conda info --base)"

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

# 境内网络：pip 走阿里云镜像（实测 ~14MB/s,远快于清华 ~1.8MB/s;可用 PIP_INDEX_URL 覆盖）,
# HF 走 hf-mirror;env 方式导出,覆盖一切 pip 配置文件
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# hf-xet 会绕开镜像直连 HF 的 CAS 服务器（国内 401/超时），禁掉走普通 HTTP 下载
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
pip config set global.index-url "$PIP_INDEX_URL" > /dev/null 2>&1 || true
# 模型缓存放数据盘（关机保留，不占系统盘）
if [ -d /root/autodl-tmp ]; then
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
fi

echo "==> [1/7] 语音管线 venv（py312 + torch 2.8 cu128）"
[ -x .venv/bin/python ] || "$CONDA_BASE/envs/py312/bin/python" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
# torchaudio 必须同版本同 index 钉死：否则 pip 会从默认源拉最新版（cu13），报 libcudart.so.13 缺失
if python -c "import torch, torchaudio; assert torch.__version__.startswith('2.8') and torchaudio.__version__.startswith('2.8')" > /dev/null 2>&1; then
    echo "    torch 已安装：$(python -c 'import torch; print(torch.__version__)')"
else
    pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
fi
# --- 两个空 stub wheel ------------------------------------------------------------
# faster-qwen3-tts 是 speech-to-speech 的依赖,但其 transformers<5 约束与 Qwen3-ASR
# 官方要求的 transformers>=5.13 硬冲突;上游 TTS handler 全部懒加载,本项目 TTS 走
# voxcpm,faster_qwen3_tts 代码永远不会被导入 → 用 stub 顶替(带上 ggml extra,
# 因为 speech-to-speech 的依赖声明点了 faster-qwen3-tts[ggml])。
# qwentts-cpp-python 是真实 faster-qwen3-tts[ggml] 的依赖,只发 manylinux_2_39 wheel
# (需 glibc≥2.39),本机 glibc 2.31 装不上;stub 版 faster-qwen3-tts 的 ggml extra 为空,
# 正常不会引用它,留着兜底。
make_stub_wheel() {  # $1=import 名 $2=PyPI 名 $3=版本 $4=Provides-Extra(可空)
    python - "$1" "$2" "$3" "$4" <<'PYEOF'
import base64, csv, hashlib, io, sys, zipfile

mod, dist_name, ver, extra = sys.argv[1:5]
nm = dist_name.replace("-", "_")
dist = f"{nm}-{ver}.dist-info"
meta = (f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {ver}\n"
        "Summary: empty stub, see scripts/autodl_setup.sh\n")
if extra:
    meta += f"Provides-Extra: {extra}\n"
files = {
    f"{mod}/__init__.py": "# stub: never imported in this project, see scripts/autodl_setup.sh\n",
    f"{dist}/METADATA": meta,
    f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: autodl_setup.sh\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
}
records = []
out = f"/tmp/{nm}-{ver}-py3-none-any.whl"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for path, data in files.items():
        z.writestr(path, data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data.encode()).digest()).rstrip(b"=").decode()
        records.append((path, f"sha256={digest}", str(len(data.encode()))))
    records.append((f"{dist}/RECORD", "", ""))
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(records)
    z.writestr(f"{dist}/RECORD", buf.getvalue())
print(out)
PYEOF
}
# 之前可能装过真包,先卸再装 stub,保证元数据里没有 transformers<5 约束
pip uninstall -y faster-qwen3-tts > /dev/null 2>&1 || true
pip install "$(make_stub_wheel faster_qwen3_tts faster-qwen3-tts 0.3.2 ggml)"
pip show qwentts-cpp-python > /dev/null 2>&1 || \
    pip install "$(make_stub_wheel qwentts_cpp qwentts-cpp-python 0.3.1 '')"
# 钉死关键包版本防 pip 回溯（回溯会逐个下载几十个几十 MB 的 wheel,卡死数小时）:
# Qwen3-ASR 官方要求 transformers>=5.13.0;voxcpm 2.0.3 要求 >=4.36.2 + gradio>=6,<7
pip install --no-cache-dir -r requirements.txt "huggingface_hub[cli]" "voxcpm==2.0.3" "transformers==5.13.0"
deactivate

echo "==> [2/7] 数字人 venv（py310 + torch 2.7.1 cu128）"
[ -x .venv-avatar/bin/python ] || "$CONDA_BASE/envs/py310/bin/python" -m venv .venv-avatar
# shellcheck disable=SC1091
source .venv-avatar/bin/activate
pip install -U pip
if python -c "import torch; assert torch.__version__.startswith('2.7.1')" > /dev/null 2>&1; then
    echo "    torch 已安装：$(python -c 'import torch; print(torch.__version__)')"
else
    pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
fi
deactivate

echo "==> [3/7] FlashHead 推理代码（vendor/SoulX-FlashHead）"
if [ -d vendor/SoulX-FlashHead/flash_head ]; then
    echo "    已存在，跳过"
else
    mkdir -p vendor
    # GitHub 直连超时：优先 AutoDL 学术加速（只在子 shell 生效，避免拖慢 pip），失败回退代理
    (source /etc/network_turbo > /dev/null 2>&1 || true; \
     git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git vendor/SoulX-FlashHead) || \
    git clone --depth 1 https://gh-proxy.com/https://github.com/Soul-AILab/SoulX-FlashHead.git vendor/SoulX-FlashHead
fi
# shellcheck disable=SC1091
source .venv-avatar/bin/activate
# gradio/flask 是官方 demo 的依赖，推理路径用不到，不装;
# nvidia-* 钉版与 torch 2.7.1 自带的 nvidia-nccl-cu12==2.26.2 冲突（单卡根本不用 nccl），剔除
grep -vE '^(gradio|flask|nvidia-)' vendor/SoulX-FlashHead/requirements.txt > /tmp/flashhead-req.txt
pip install --no-cache-dir -r /tmp/flashhead-req.txt
pip install --no-cache-dir websockets pyyaml
# flash_attn：显著提速（官方推荐）；SKIP_FLASH_ATTN=1 可跳过（SDPA 兜底，慢）
# 注意必须真 import 验证:预编译 wheel 可能装上却因 glibc 版本不足无法加载
if [ "${SKIP_FLASH_ATTN:-0}" = "1" ]; then
    echo "    SKIP_FLASH_ATTN=1，跳过 flash_attn（SDPA 兜底，帧率会降）"
elif python -c "import flash_attn" > /dev/null 2>&1; then
    echo "    flash_attn 已安装"
else
    pip uninstall -y flash-attn > /dev/null 2>&1 || true  # 清掉装上了但加载不了的残件
    FA_OK=1
    # 官方预编译 wheel(torch2.7 cu12 abiTRUE cp310)需 glibc≥2.32,够新才试(走学术加速)
    if python -c "import platform,sys;p=platform.libc_ver()[1].split('.');sys.exit(0 if (int(p[0]),int(p[1]))>=(2,32) else 1)"; then
        FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
        if (source /etc/network_turbo > /dev/null 2>&1 || true; pip install "$FA_WHEEL") \
            && python -c "import flash_attn" > /dev/null 2>&1; then
            echo "    flash_attn 预编译 wheel 安装成功"
            FA_OK=0
        fi
    else
        echo "    glibc<2.32,预编译 wheel 不可用,本地编译（30-60 分钟）"
    fi
    if [ "$FA_OK" = "1" ]; then
        # 系统 nvcc 是 CUDA 11.6 太老;从 conda 官方 nvidia channel 装 cuda-nvcc 12.8
        # (+cudart-dev/cccl 头文件)进 py310 环境当编译工具链(pip 的 nvcc 包只有
        # ptxas 没有编译器驱动,不可用);只编 4090D 的 sm89 内核省时间
        if ! "$CONDA_BASE/envs/py310/bin/nvcc" -V 2>/dev/null | grep -q "release 12\.8"; then
            "$CONDA_BASE/bin/conda" install -y -n py310 -c nvidia \
                cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8
        fi
        # PATH 追加而非前置:conda 环境自带 pip,前置会抢掉 venv 的 pip(报 No module named 'torch')
        export CUDA_HOME="$CONDA_BASE/envs/py310"
        export PATH="$PATH:$CUDA_HOME/bin"
        export TORCH_CUDA_ARCH_LIST="8.9"
        # MAX_JOBS 限 4:16 核全并行曾把内存(60G)干到 56G+、sshd 挤死 3 小时无法登录
        pip install ninja
        MAX_JOBS=4 pip install flash_attn==2.8.0.post2 --no-build-isolation || \
            echo "    WARN: flash_attn 编译失败，将用 SDPA 兜底（可下预编译 wheel 重试）"
    fi
fi
deactivate

echo "==> [4/7] 预下载模型（HF_HOME=${HF_HOME:-默认}）"
# hf download 幂等（已下载会校验后跳过）;VoxCPM2 / Qwen3-ASR 走 HF 缓存（管线按 repo id 加载）
.venv/bin/hf download openbmb/VoxCPM2
.venv/bin/hf download Qwen/Qwen3-ASR-1.7B-hf
# 本地目录加载的模型走 ModelScope（实例连阿里 CDN ~13MB/s,hf-mirror 仅 ~0.7MB/s）
.venv/bin/pip install -q modelscope
# 数字人只要 Lite 权重 + LTX VAE（~8GB；Pro 权重不下）
.venv/bin/modelscope download --model Soul-AILab/SoulX-FlashHead-1_3B \
    --include "Model_Lite/*" "VAE_LTX/*" --local_dir models/SoulX-FlashHead-1_3B
.venv/bin/modelscope download --model AI-ModelScope/wav2vec2-base-960h \
    --local_dir models/wav2vec2-base-960h

echo "==> [5/7] 检查配置"
if [ ! -f .env.local ]; then
    echo "ERROR: .env.local 不存在。请 cp .env.example .env.local 并填入 DEEPSEEK_API_KEY。" >&2
    exit 1
fi
set -a; source .env.local; set +a
if [ -z "${DEEPSEEK_API_KEY:-${LLM_API_KEY:-}}" ]; then
    echo "ERROR: .env.local 里 DEEPSEEK_API_KEY（或 LLM_API_KEY）为空。" >&2
    exit 1
fi
VOXEMW_CONFIG="${VOXEMW_CONFIG:-configs/assistant.yaml}"
[ -f "$VOXEMW_CONFIG" ] || { echo "ERROR: 配置不存在: $VOXEMW_CONFIG" >&2; exit 1; }

echo "==> [6/7] 数字人肖像素材检查"
MISSING_IMG=0
for img in assets/fengge/ref.png assets/liangzi/ref.png; do
    [ -f "$img" ] || { echo "    缺 $img（对应 persona 将降级纯语音）"; MISSING_IMG=1; }
done
[ "$MISSING_IMG" = "0" ] || echo "    提示：缺肖像不阻塞语音对话，补齐后重启数字人服务即可"

echo "==> [7/7] 启动服务"
bash scripts/start_assistant.sh

cat <<'EOF'

==> 部署完成。本机访问方式（SSH 隧道，AutoDL 默认不开公网端口）：

    ssh -CNg -L 8000:127.0.0.1:8000 root@<实例主机> -p <SSH端口>

    浏览器打开  http://localhost:8000  → 数字人语音对话。

    排障：tail -f logs/pipeline.log logs/avatar.log logs/orchestrator.log
EOF
