#!/usr/bin/env bash
# VoxEMW 数字人实时语音助手 —— AutoDL 实例一键部署脚本
#
# 目标环境：AutoDL Miniconda 镜像 + 单卡 RTX 4090D（Linux, CUDA 12.8 驱动）
# 用法：rsync 仓库到实例后，在仓库根目录执行  bash scripts/autodl_setup.sh
# 幂等：重复执行不会重复装依赖/下模型，已在跑的服务不重启。
#
# 环境：.venv（py312 + torch 2.8）：s2s 语音管线（SenseVoice / VoxCPM2）+ orchestrator + 记忆
# 数字人（AVTR-1）运行在独立的 pixi env（/root/autodl-tmp/avtr-1/.pixi/envs/renderer），
# 安装过程含 TRT 引擎编译等一次性步骤，见本脚本 [2/7] 段说明。
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

echo "==> [2/7] 数字人（AVTR-1）环境检查"
# AVTR-1 运行在独立 pixi env（与主 venv 依赖冲突不可合装）。一次性部署步骤：
#   git clone https://github.com/avaturn-live/avtr-1 /root/autodl-tmp/avtr-1
#   cd 后 pixi install（国内镜像调整见仓库部署笔记）→ pixi run download（HF gated，
#   需先在 HF 页面接受协议）→ pixi run build-trt-engines（按显卡编译，~20 分钟）
# 已知坑：onnxruntime-gpu 需降 1.22（pixi run 会重同步，用 env python 直调）；
# glibc 2.31 需重编 libgrid_sample_3d_plugin。完成标志：pixi env python 可 import avtr1_renderer。
AVTR_ENV=/root/autodl-tmp/avtr-1/.pixi/envs/renderer
if [ -x "$AVTR_ENV/bin/python" ] && "$AVTR_ENV/bin/python" -c "import avtr1_renderer" 2>/dev/null; then
    echo "    AVTR-1 环境就绪"
else
    echo "ERROR: AVTR-1 环境未就绪（$AVTR_ENV）。请按上方步骤先完成一次性部署。" >&2
    exit 1
fi

echo "==> [4/7] 预下载模型（HF_HOME=${HF_HOME:-默认}）"
# hf download 幂等（已下载会校验后跳过）;VoxCPM2 / Qwen3-ASR 走 HF 缓存（管线按 repo id 加载）
.venv/bin/hf download openbmb/VoxCPM2
# STT（SenseVoiceSmall）与记忆 embedder（bge-m3）：ModelScope/HF 缓存（首次启动自动下载亦可）
.venv/bin/pip install -q modelscope
.venv/bin/modelscope download --model iic/SenseVoiceSmall
.venv/bin/hf download BAAI/bge-m3 --exclude "imgs/*"

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
