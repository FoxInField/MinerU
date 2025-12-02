# Use the official vllm image for gpu with Ampere architecture and above (Compute Capability>=8.0)
# Compute Capability version query (https://developer.nvidia.com/cuda-gpus)
FROM vllm/vllm-openai:v0.10.1.1

# Use the official vllm image for gpu with Turing architecture and below (Compute Capability<8.0)
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
COPY parse_to_json_prompt.txt /app/parse_to_json_prompt.txt
# Copy mineru package directory
COPY mineru/ /app/mineru/

# Install mineru from local code (editable mode)
RUN cd /app && \
    python3 -m pip install -e '.[core]' --break-system-packages && \
    python3 -m pip cache purge

# ENV HF_HOME=/root/.cache/huggingface
# # Download models using BuildKit cache mount (模型不会进入镜像，但会在构建间共享)
# RUN --mount=type=cache,target=/root/.cache/huggingface \
#     /bin/bash -c "mineru-models-download -s huggingface -m all"


# Set the entry point to activate the virtual environment and run the command line tool
# ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
ENTRYPOINT ["/bin/bash", "-c", "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]