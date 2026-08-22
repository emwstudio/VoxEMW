#!/usr/bin/env bash
# VoxEMW 数字人实时语音助手 —— AutoDL 实例一键部署脚本
#
# 目标环境：AutoDL Miniconda 镜像 + 单卡 RTX 4090D（Linux, CUDA 12.8 驱动）
# 用法：rsync 仓库到实例后，在仓库根目录执行  bash scripts/autodl_setup.sh
# 幂等：重复执行不会重复装依赖/下模型，已在跑的服务不重启。
#
# 环境：.venv（py312 + torch 2.8）：s2s 语音管线（SenseVoice / VoxCPM2）+ orchestrator
# 数字人（AVTR-1）运行在独立的 pixi env（/root/autodl-tmp/avtr-1/.pixi/envs/renderer），
# 安装过程含 TRT 引擎编译等一次性步骤，见本脚本 [2/7] 段说明。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [0/7] 基础环境（conda + 系统包）"
# GitHub 访问（speech-to-speech git 依赖、ACE-Step clone）全程要学术加速，
# 提到最前面——放后面的话 [1/7] 的 pip git clone 会直连超时（2026-08-21 踩坑）
# shellcheck disable=SC1091
source /etc/network_turbo > /dev/null 2>&1 || true
# 但 pip 镜像（aliyun）和 pytorch 官方源必须直连——走代理会 503/超时
# （2026-08-21 北京区实例实测 setuptools 都拉不下来）
export no_proxy="${no_proxy:-localhost,127.0.0.1},mirrors.aliyun.com,download.pytorch.org"
# 非交互 SSH 下 conda 可能不在 PATH
if ! command -v conda > /dev/null 2>&1 && [ -x /root/miniconda3/bin/conda ]; then
    export PATH="/root/miniconda3/bin:$PATH"
fi
command -v conda > /dev/null 2>&1 || { echo "ERROR: 无 conda，请换 AutoDL Miniconda 镜像" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 2>/dev/null || true
# 老镜像 .condarc 里残留 tuna 的 pkgs/free（已 404 下架），不清掉 conda create 直接炸
# （2026-08-22 北京区实例踩坑）
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 2>/dev/null || true
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
# coturn：WebRTC 音画下行过 SSH 隧道的 TURN 服务（缺了页面只剩文字，无音画）
command -v turnserver > /dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS coturn"
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
# faster-qwen3-tts 是 speech-to-speech 的依赖,但其 transformers<5 约束与上游
# 当前要求的 transformers>=5.13 硬冲突;上游 TTS handler 全部懒加载,本项目 TTS 走
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
# 上游 speech-to-speech 当前要求 transformers>=5.13.0;voxcpm 2.0.3 要求 >=4.36.2 + gradio>=6,<7
# git 依赖每次都会重克隆（pip 对 git URL 无缓存），学术加速代理抖动会让整步白跑
# ——已装齐就跳过（2026-08-21 北京区实例连续三次倒在 git clone/checkout）
if python -c "import speech_to_speech, voxcpm, transformers; assert transformers.__version__.startswith('5.13')" > /dev/null 2>&1; then
    echo "    管线依赖已装齐，跳过大安装"
else
    pip install --no-cache-dir -r requirements.txt "huggingface_hub[cli]" "voxcpm==2.0.3" "transformers==5.13.0"
fi
# SmartTurn 复核走 GPU：speech-to-speech 拉的是 CPU 版 onnxruntime，
# 换 onnxruntime-gpu 让 voxemw/pipeline/launch.py 的 _patch_smart_turn_gpu 生效
# （复核 ~80ms → ~2ms）。1.28+ 要 CUDA 13 不匹配，1.22.1 已从 PyPI 撤轮子，钉 1.24.4。
pip install --no-cache-dir onnxruntime-gpu==1.24.4
# H264 编码器调优（aiortc 无官方入口，直接改 site-packages；幂等 sed）：
# gop_size 加 75（3s 一关键帧）——x264 默认 250 帧（10s），切 persona 换肖像时
# 新画面被当旧画面的 P 帧增量编码，要等关键帧才干净（2026-08-18 切换重影修复）
# + vbv-maxrate/bufsize=3000（峰值码率钳到平均码率，防运动峰值爆带宽丢帧，
# iPad 链路马赛克修复）
H264=.venv/lib/python3.12/site-packages/aiortc/codecs/h264.py
if [ -f "$H264" ]; then
    grep -q "gop_size" "$H264" || \
        sed -i 's|            self.codec.options = {|            self.codec.gop_size = 75  # VoxEMW：3s 一关键帧\n            self.codec.options = {|' "$H264"
    grep -q "vbv-maxrate" "$H264" || \
        sed -i 's|                "tune": "zerolatency",|                "tune": "zerolatency",\n                "vbv-maxrate": "4000",  # VoxEMW：峰值码率钳制（=video_bitrate 4M）\n                "vbv-bufsize": "4000",|' "$H264"
fi
deactivate

echo "==> [2/7] 数字人（AVTR-1）环境检查"
# AVTR-1 运行在独立 pixi env（与主 venv 依赖冲突不可合装）。一次性部署步骤：
#   git clone https://github.com/avaturn-live/avtr-1 /root/autodl-tmp/avtr-1
#   cd 后 pixi install && pixi install -e renderer（镜像已调进 pixi.toml）→
#   权重/TRT 引擎来自 gated 下载或旧实例接力（avtr1_storage/）
# 2026-08-22 换机重建实录（pixi env 无 pip，一律 uv 补）：
#   - tensorrt 不在 manifest，引擎要求 10.11.*（读引擎头 0a0b 得出）
#   - onnxruntime-gpu 锁的 1.28 与 CUDA 12.8 不合，降 1.22.0
#   - websockets 漏装（avatar service 直接崩 ModuleNotFoundError）
#   - glibc 2.31 的重编插件在 avtr1_storage/renderer_runtime_artifacts/ 里，
#     随存储接力即恢复，无需重编
AVTR_ENV=/root/autodl-tmp/avtr-1/.pixi/envs/renderer
if [ -x "$AVTR_ENV/bin/python" ]; then
    UV4AVTR="$(command -v uv || ls /root/miniconda3/envs/py312/bin/uv 2>/dev/null || echo "$PWD/.venv/bin/uv")"
    "$AVTR_ENV/bin/python" -c "import tensorrt" 2>/dev/null || \
        "$UV4AVTR" pip install --python "$AVTR_ENV/bin/python" "tensorrt==10.11.*"
    "$AVTR_ENV/bin/python" -c "import onnxruntime; assert onnxruntime.__version__ == '1.22.0'" 2>/dev/null || \
        "$UV4AVTR" pip install --python "$AVTR_ENV/bin/python" onnxruntime-gpu==1.22.0
    "$AVTR_ENV/bin/python" -c "import websockets" 2>/dev/null || \
        "$UV4AVTR" pip install --python "$AVTR_ENV/bin/python" websockets
fi
if [ -x "$AVTR_ENV/bin/python" ] && "$AVTR_ENV/bin/python" -c "import avtr1_renderer" 2>/dev/null; then
    echo "    AVTR-1 环境就绪"
else
    echo "ERROR: AVTR-1 环境未就绪（$AVTR_ENV）。请按上方步骤先完成一次性部署。" >&2
    exit 1
fi

echo "==> [3/7] 预下载模型（HF_HOME=${HF_HOME:-默认}）"
# hf download 幂等（已下载会校验后跳过）;VoxCPM2 走 HF 缓存（管线按 repo id 加载）
.venv/bin/hf download openbmb/VoxCPM2
# STT（SenseVoiceSmall）：ModelScope 缓存（首次启动自动下载亦可）
.venv/bin/pip install -q modelscope
.venv/bin/modelscope download --model iic/SenseVoiceSmall
# SmartTurn v3.2 GPU 版模型（配置里 smart_turn_model_path 指到 *-gpu.onnx）
# 注意要整仓下：不配 smart_turn_model_path 时上游离线加载默认 CPU 版文件，
# 只下 gpu.onnx 会 LocalEntryNotFound（2026-08-22 镜像缓存机踩坑）
.venv/bin/hf download pipecat-ai/smart-turn-v3

echo "==> [4/7] 检查配置"
if [ ! -f .env.local ]; then
    echo "ERROR: .env.local 不存在。请 cp .env.example .env.local（LLM_API_KEY 填任意非空值）。" >&2
    exit 1
fi
set -a; source .env.local; set +a
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -z "${LLM_API_KEY:-}" ]; then
    echo "ERROR: .env.local 缺少 DEEPSEEK_API_KEY（或回退 LLM_API_KEY）。" >&2
    exit 1
fi
VOXEMW_CONFIG="${VOXEMW_CONFIG:-configs/assistant.yaml}"
[ -f "$VOXEMW_CONFIG" ] || { echo "ERROR: 配置不存在: $VOXEMW_CONFIG" >&2; exit 1; }

echo "==> [5/7] 数字人肖像素材检查"
MISSING_IMG=0
for img in assets/liangzi/ref.png; do
    [ -f "$img" ] || { echo "    缺 $img（对应 persona 将降级纯语音）"; MISSING_IMG=1; }
done
[ "$MISSING_IMG" = "0" ] || echo "    提示：缺肖像不阻塞语音对话，补齐后重启数字人服务即可"

echo "==> [6/7] 歌声生成（ACE-Step 1.5，独立 uv 环境）"
# 与主 venv 隔离（torch 钉死版本不同：ACE-Step 要 2.10.0+cu128，主 venv 是 2.8），
# 同 AVTR-1 独立 pixi env 一个思路。仓库/checkpoint 都放数据盘；
# checkpoint 默认 acestep-v15-turbo（2B, ~4.7GB 显存），下到 <仓库>/checkpoints/
ACESTEP_DIR="${ACESTEP_DIR:-/root/autodl-tmp/ACE-Step-1.5}"
if [ -d "$ACESTEP_DIR" ]; then
    echo "    仓库已存在，跳过 clone"
else
    git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5 "$ACESTEP_DIR"
fi
# uv 只是环境管理器，装进主 venv 用即可
[ -x .venv/bin/uv ] || .venv/bin/pip install -q uv
UV_BIN="$PWD/.venv/bin/uv"
# uv 网络策略（2026-08-21 踩坑固化）：pypi 走阿里云镜像直连、pytorch 官方源
# 直连（学术加速代理对这两者极慢，~10MB/min）；GitHub releases（flash_attn
# 预编译轮）直连超时，必须走学术加速代理（[0/7] 已 source，取其 proxy 变量）；
# 该代理是 MITM，uv 的 rustls 不认其 CA，需 --system-certs 用系统证书
UV_ENV=(env "http_proxy=${http_proxy:-}" "https_proxy=${https_proxy:-}"
      "no_proxy=localhost,127.0.0.1,mirrors.aliyun.com,download.pytorch.org,modelscope.com"
      "UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple"
      "UV_PYTHON=$CONDA_BASE/envs/py312/bin/python")
# uv sync 按 pyproject 钉死版本装依赖；强制复用 conda py312 解释器
# （项目要求 Python >=3.11,<3.13），不让 uv 自己再下一个 Python
if (cd "$ACESTEP_DIR" && "${UV_ENV[@]}" "$UV_BIN" run --no-sync --system-certs python -c "import acestep" > /dev/null 2>&1); then
    echo "    ACE-Step 环境已就绪"
else
    echo "    uv sync 安装依赖（首次含 torch 2.10 cu128，下载量大，耐心等）..."
    (cd "$ACESTEP_DIR" && "${UV_ENV[@]}" "$UV_BIN" sync --system-certs)
fi
# flash_attn：uv.lock 只给了 cp311 预编译轮，py312 环境装不上会静默退回 SDPA
# （注意力变慢 + nano-vllm 无法开 CUDA graph，2026-08-21 实测日志确认）。
# 手动补装 cp312 预编译轮（GitHub releases，需走代理）
if ! (cd "$ACESTEP_DIR" && .venv/bin/python -c "import flash_attn" > /dev/null 2>&1); then
    (cd "$ACESTEP_DIR" && "${UV_ENV[@]}" "$UV_BIN" pip install --system-certs --python .venv/bin/python \
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl") || \
        echo "    警告：flash_attn 安装失败（退回 SDPA，生成略慢）"
fi
# 预下载 checkpoint（幂等；境内 auto 源自动走 ModelScope，也可
# ACESTEP_DOWNLOAD_SOURCE=modelscope 显式指定）
(cd "$ACESTEP_DIR" && "${UV_ENV[@]}" "$UV_BIN" run --no-sync acestep-download) || \
    echo "    警告：checkpoint 下载失败，首次点歌时会自动重试下载"
# 下载器会把 5Hz-LM 全档位都拉下来；4B 我们用不到（显存 +9GB、磁盘 ~10GB，
# 2026-08-21 实测直接把 89G 系统盘塞满），删掉——运行时用 1.7B
rm -rf "$ACESTEP_DIR/checkpoints/acestep-5Hz-lm-4B"
# 冒烟：起服务 → 生成 10s 样本 → 停服务（正式启停归 start_assistant.sh）。
# 首次含模型加载，可能要几分钟；失败只告警不阻塞（点歌时会再暴露）
echo "    冒烟：生成 10s 样本（首次含模型加载，耐心等）..."
mkdir -p logs
pushd "$ACESTEP_DIR" > /dev/null
ACESTEP_NO_INIT=false ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B \
    ACESTEP_LM_DEVICE=cpu ACESTEP_OFFLOAD_DIT_TO_CPU=true \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    nohup "$UV_BIN" run --no-sync acestep-api --host 127.0.0.1 --port 8001 \
    > "$OLDPWD/logs/acestep_smoke.log" 2>&1 &
SMOKE_PID=$!
popd > /dev/null
.venv/bin/python - <<'PYEOF' || echo "    警告：冒烟未通过（看 logs/acestep_smoke.log），唱歌功能上线前先排障"
import json, time, urllib.request

BASE = "http://127.0.0.1:8001"

def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

# 等服务起来（模型加载中 /docs 也可能未响应，给足 5 分钟）
for _ in range(60):
    try:
        urllib.request.urlopen(BASE + "/docs", timeout=5)
        break
    except Exception:
        time.sleep(5)
else:
    raise SystemExit("冒烟失败：acestep-api 5 分钟未就绪")
reply = post("/release_task", {"prompt": "pop, short test", "lyrics": "[inst]",
                               "audio_duration": 10.0, "audio_format": "wav"})
task_id = (reply.get("data") or {}).get("task_id")
assert task_id, f"release_task 无 task_id: {reply}"
for _ in range(60):
    time.sleep(5)
    items = post("/query_result", {"task_id_list": [task_id]}).get("data") or []
    item = next((i for i in items if i.get("task_id") == task_id), None)
    status = item.get("status") if item else 0
    if status == 1:
        print("    冒烟通过：10s 样本生成成功")
        break
    if status == 2:
        raise SystemExit(f"冒烟失败：任务报错 {item.get('progress_text')}")
else:
    raise SystemExit("冒烟失败：生成超时（5 分钟）")
PYEOF
kill "$SMOKE_PID" 2> /dev/null || true
pkill -f "acestep-api" 2> /dev/null || true

echo "==> [7/7] 启动服务"
bash scripts/start_assistant.sh

cat <<'EOF'

==> 部署完成。本机访问方式（SSH 隧道，AutoDL 默认不开公网端口）：

    ssh -CNg -L 8000:127.0.0.1:8000 root@<实例主机> -p <SSH端口>

    浏览器打开  http://localhost:8000  → 数字人语音对话。

    排障：tail -f logs/pipeline.log logs/avatar.log logs/orchestrator.log
EOF
