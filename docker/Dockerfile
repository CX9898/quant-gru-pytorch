# CUDA 12.8 + PyTorch + Python3 环境
# 支持 RTX 5090 (sm_120 Blackwell 架构)

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 配置清华 Ubuntu APT 源
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    vim \
    # FFmpeg 及开发库 (torchcodec 依赖)
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    # 音频处理依赖
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

# 创建 python 软链接
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 配置清华 PyPI 源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 升级 pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 安装 PyTorch nightly (支持 CUDA 12.8 和 sm_120)
RUN pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装 torchcodec (必须使用 PyTorch 官方源)
RUN pip3 install --pre torchcodec \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装其他常用 Python 包 (使用清华源)
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    tqdm \
    tensorboard \
    pyyaml \
    soundfile \
    librosa \
    onnxscript \
    safetensors \
    pybind11

# 设置工作目录
WORKDIR /workspace

# 设置 TORCH_CUDA_ARCH_LIST 以支持编译
ENV TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;8.9;9.0;12.0"

# 验证安装
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python3 -c "import torchaudio; print(f'torchaudio: {torchaudio.__version__}')" && \
    python3 -c "from torchcodec.decoders import AudioDecoder; print('torchcodec: OK')"

CMD ["bash"]
