# Use DaoCloud mirrored vllm image for China region for gpu with Ampere architecture and above (Compute Capability>=8.0)
# Compute Capability version query (https://developer.nvidia.com/cuda-gpus)
FROM docker.m.daocloud.io/vllm/vllm-openai:v0.10.1.1

# Use the official vllm image
# FROM vllm/vllm-openai:v0.10.1.1

# Use DaoCloud mirrored vllm image for China region for gpu with Turing architecture and below (Compute Capability<8.0)
# FROM docker.m.daocloud.io/vllm/vllm-openai:v0.10.2

# Use the official vllm image
# FROM vllm/vllm-openai:v0.10.2

# Install libgl for opencv support & Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy necessary files for local code installation
# Copy project root files first
COPY pyproject.toml /app/
COPY README.md /app/
# Copy mineru package directory
COPY mineru/ /app/mineru/

# Install mineru from local code (editable mode)
RUN cd /app && \
    python3 -m pip install -e '.[core]' -i https://mirrors.aliyun.com/pypi/simple --break-system-packages && \
    python3 -m pip cache purge

# 设置 ModelScope 缓存路径
# ENV MODELSCOPE_CACHE=/root/.cache/modelscope

# # Download models using BuildKit cache mount (模型不会进入镜像，但会在构建间共享)
# RUN --mount=type=cache,target=/root/.cache/modelscope \
#     /bin/bash -c "mineru-models-download -s modelscope -m all"

# Set the entry point to activate the virtual environment and run the command line tool
ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]