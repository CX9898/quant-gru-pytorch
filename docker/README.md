# Docker Build Environment

这个目录提供一个可复用的 Linux CUDA/PyTorch 构建环境，适合：

- CUDA/C++ 项目的 CMake 构建
- PyTorch CUDAExtension / pybind11 扩展编译
- 需要 `torch` / `torchaudio` / `torchcodec` 的 Python 项目

## 默认环境

当前默认参数面向本仓库已验证的工具链：

- Base image: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- PyTorch index: `https://download.pytorch.org/whl/nightly/cu128`
- Torch packages: `torch torchvision torchaudio`
- `torchcodec`: 默认安装
- `TORCH_CUDA_ARCH_LIST`: `6.0;7.0;7.5;8.0;8.6;8.9;9.0;12.0`

## 预装工具

当前镜像已预装这些常用工具/依赖：

- Python: `python3`, `pip`, `python3-dev`
- Native build/debug: `build-essential`, `cmake`, `ninja-build`, `pkg-config`, `clang`, `gdb`
- Common CLI: `git`, `wget`, `curl`, `vim`, `less`, `unzip`, `zip`, `file`, `jq`, `rsync`, `tree`, `tmux`, `htop`, `ripgrep`
- Multimedia/audio: `ffmpeg`, `libsndfile1`, `libsox-dev`, `sox`
- FFmpeg dev libs: `libavcodec-dev`, `libavformat-dev`, `libavutil-dev`, `libswresample-dev`, `libswscale-dev`
- Python packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `tensorboard`, `pyyaml`, `soundfile`, `librosa`, `onnxscript`, `safetensors`, `pybind11`

这套预装内容已经比较接近通用 CUDA/PyTorch 开发箱，足够支撑本仓库当前的编译、调试和日常排查；如果后续还有跨项目通用需求，再按需继续补充。

## 可选镜像参数

为了兼顾通用性和国内下载速度，Dockerfile 默认使用官方源，同时支持可选 build args。

### 可选参数

- `BASE_IMAGE`: CUDA 基础镜像
- `APT_MIRROR`: Ubuntu APT 镜像前缀，例如 `https://mirrors.tuna.tsinghua.edu.cn`
- `PIP_INDEX_URL`: PyPI 镜像，例如 `https://pypi.tuna.tsinghua.edu.cn/simple`
- `PIP_TRUSTED_HOST`: 配合自定义 PyPI 镜像使用
- `PYTORCH_INDEX_URL`: PyTorch wheel 源
- `TORCH_PACKAGES`: 要安装的 torch 系列包
- `INSTALL_TORCHCODEC`: 是否安装 `torchcodec`，`1` 为安装，`0` 为跳过
- `EXTRA_PYTHON_PACKAGES`: 额外 Python 包列表
- `TORCH_CUDA_ARCH_LIST`: 默认 CUDA arch 列表

## 构建示例

### 1. 默认官方源

```bash
docker build -f docker/Dockerfile -t cuda-pytorch-ci:cu128 .
```

### 2. 国内环境使用清华镜像

```bash
docker build \
  -f docker/Dockerfile \
  --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
  -t cuda-pytorch-ci:cu128 .
```

### 3. 跳过 torchcodec

```bash
docker build \
  -f docker/Dockerfile \
  --build-arg INSTALL_TORCHCODEC=0 \
  -t cuda-pytorch-ci:no-codec .
```

## 使用示例

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  cuda-pytorch-ci:cu128
```

## Docker Compose

仓库同时提供了 `docker/docker-compose.yml`，服务名和镜像名也已改成通用环境语义：

- service: `cuda-pytorch`
- image: `cuda-pytorch-ci:cu128`

启动示例：

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec cuda-pytorch bash
```
