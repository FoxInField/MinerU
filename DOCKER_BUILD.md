# Docker 构建说明

## 概述

MinerU 的 Docker 镜像构建需要先在本机下载所需的模型文件，然后通过 `docker-compose.yaml` 挂载模型目录进行构建和运行。

## 构建流程

### 1. 下载模型到本机

在构建 Docker 镜像之前，需要先在本机下载 MinerU 所需的模型文件。由于模型文件较大（通常几十GB），不建议将模型打包进 Docker 镜像中，而是通过 volumes 挂载本机的模型目录。

#### 下载模型

只需安装必要的库即可下载模型，无需安装完整的 MinerU：

```bash
# 下载 HuggingFace 模型
pip install huggingface-hub
huggingface-cli download opendatalab/PDF-Extract-Kit-1.0
huggingface-cli download opendatalab/MinerU2.5-2509-1.2B

# 或下载 ModelScope 模型（中国大陆用户推荐）
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
python -c "from modelscope import snapshot_download; snapshot_download('OpenDataLab/PDF-Extract-Kit-1.0'); snapshot_download('OpenDataLab/MinerU2.5-2509-1.2B')"
```

#### 模型存储位置

- **HuggingFace 模型**：默认存储在 `~/.cache/huggingface/hub/` 目录下
  - Pipeline 模型：`opendatalab--PDF-Extract-Kit-1.0`
  - VLM 模型：`opendatalab--MinerU2.5-2509-1.2B`
- **ModelScope 模型**：默认存储在 `~/.cache/modelscope/hub/` 目录下
  - Pipeline 模型：`OpenDataLab/PDF-Extract-Kit-1.0`
  - VLM 模型：`OpenDataLab/MinerU2.5-2509-1.2B`

> [!NOTE]
> 下载模型后，模型会自动存储在上述位置，无需手动配置 `mineru.json`。Docker 容器会通过 volumes 挂载这些目录来访问模型。

### 2. 使用 Docker Compose 构建和运行

由于模型文件较大，不建议将模型打包进 Docker 镜像中。正确的做法是使用 `docker-compose.yaml` 通过 volumes 挂载本机的模型目录。

#### 构建 Docker 镜像

```bash
# 在项目根目录下（包含 docker-compose.yaml 的目录）
docker-compose build mineru-api
```

#### 运行服务

```bash
# 启动服务（会自动挂载模型目录）
docker-compose up mineru-api
```

#### docker-compose.yaml 配置说明

确保 `docker-compose.yaml` 中包含模型目录的挂载配置：

```yaml
volumes:
  - ${HOME}/.cache/huggingface:/root/.cache/huggingface  # HuggingFace 模型
  # 或
  - ${HOME}/.cache/modelscope:/root/.cache/modelscope    # ModelScope 模型
```

### 3. 使用不同的 Dockerfile

MinerU 提供了两个 Dockerfile：

- **`docker/global/Dockerfile`**：使用官方 vllm 镜像（适用于全球用户）
- **`docker/china/Dockerfile`**：使用 DaoCloud 镜像源（适用于中国大陆用户）

#### 使用 global Dockerfile

```yaml
# docker-compose.yaml
services:
  mineru-api:
    build:
      context: ./MinerU
      dockerfile: docker/global/Dockerfile
```

#### 使用 china Dockerfile

```yaml
# docker-compose.yaml
services:
  mineru-api:
    build:
      context: ./MinerU
      dockerfile: docker/china/Dockerfile
```

## 注意事项

1. **模型必须预先下载**：Dockerfile 中已注释掉模型下载步骤，模型必须在构建前下载到本机。

2. **必须使用 docker-compose 挂载**：模型文件较大（通常几十GB），直接打包进镜像会导致镜像过大，且不利于模型更新。必须通过 volumes 挂载。

3. **模型路径一致性**：确保容器内的模型路径与下载时的路径一致。默认情况下：
   - HuggingFace: `/root/.cache/huggingface`
   - ModelScope: `/root/.cache/modelscope`

4. **环境变量设置**：在 `docker-compose.yaml` 中设置 `MINERU_MODEL_SOURCE` 环境变量：
   ```yaml
   environment:
     MINERU_MODEL_SOURCE: huggingface  # 或 modelscope
   ```

5. **GPU 支持**：确保 Docker 已配置 NVIDIA Container Toolkit，并在 `docker-compose.yaml` 中正确配置 GPU 资源。

## 完整示例

```bash
# 1. 下载模型（HuggingFace）
pip install huggingface-hub
huggingface-cli download opendatalab/PDF-Extract-Kit-1.0
huggingface-cli download opendatalab/MinerU2.5-2509-1.2B

# 或下载模型（ModelScope，中国大陆用户推荐）
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
python -c "from modelscope import snapshot_download; snapshot_download('OpenDataLab/PDF-Extract-Kit-1.0'); snapshot_download('OpenDataLab/MinerU2.5-2509-1.2B')"

# 2. 构建镜像
docker-compose build mineru-api

# 3. 运行服务
docker-compose up mineru-api
```

## 故障排查

### 问题：容器启动后找不到模型

**解决方案**：
- 检查 volumes 挂载是否正确
- 确认模型已下载到指定目录
- 检查 `MINERU_MODEL_SOURCE` 环境变量设置是否正确

### 问题：模型下载失败

**解决方案**：
- 中国大陆用户建议使用 ModelScope：
  ```bash
  pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
  python -c "from modelscope import snapshot_download; snapshot_download('OpenDataLab/PDF-Extract-Kit-1.0'); snapshot_download('OpenDataLab/MinerU2.5-2509-1.2B')"
  ```
- 检查网络连接
- 确保有足够的磁盘空间

### 问题：Docker 构建失败

**解决方案**：
- 确保已安装 Docker 和 docker-compose
- 检查 Dockerfile 路径是否正确
- 查看构建日志定位具体错误

## 参考文档

- [MinerU 模型源说明](../docs/zh/usage/model_source.md)
- [Docker 部署文档](../docs/zh/quick_start/docker_deployment.md)

