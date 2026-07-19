FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Python 依赖（transformers 从 GitHub main 安装，Qwen3-ASR 需要）
COPY agent/requirements.txt agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt \
    && pip install --no-cache-dir -U "huggingface_hub[cli]"

# vLLM 独立 venv（vllm 0.11.2：最后 cu128 时代版本；新版只出 cu13 轮子，
# 且 vllm 0.11.x 与 agent venv 的 transformers 5.x 不兼容，必须隔离）
RUN python3 -m venv /root/venv-vllm \
    && /root/venv-vllm/bin/pip install --no-cache-dir vllm==0.11.2

# 预下载三个模型到镜像内 HF cache（构建时需约 15GB 磁盘；
# 私有/受限模型可在构建时通过 --build-arg HF_TOKEN=xxx 传入）
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN hf download Qwen/Qwen3-ASR-0.6B-hf \
    && hf download Qwen/Qwen3-8B-AWQ \
    && hf download openbmb/VoxCPM2

# 拷贝代码与素材
COPY agent/ agent/
COPY scripts/ scripts/
COPY assets/ assets/
RUN chmod +x scripts/*.sh

# .env.local 含 LiveKit 凭证，不打入镜像；运行时挂载或拷入
CMD ["bash", "scripts/start_all.sh"]
