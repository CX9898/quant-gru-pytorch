# Quant-GRU-PyTorch

一个高性能的量化 GRU（门控循环单元）实现，基于 CUDA 和 PyTorch，支持训练和推理的量化感知计算。

## 📋 项目简介

本项目实现了一个支持量化的 GRU 神经网络模块，核心使用 CUDA 编写以实现高性能计算，并通过 PyBind11 提供 PyTorch 接口。项目支持：

- **浮点和量化两种模式**：可在训练和推理时自由切换
- **灵活的量化配置**：支持 8/16/32 位量化，可配置对称/非对称量化
- **两种校准方法**：MinMax（快速）和 Histogram（AIMET 风格，高精度）
- **双向 GRU**：完整支持 bidirectional 模式
- **与 PyTorch 兼容**：接口与 `nn.GRU` 一致，可无缝替换

## 🏗️ 项目结构

```
quant-gru-pytorch/
├── include/                    # C++/CUDA 头文件
│   ├── gru.h                   # 浮点 GRU 前向/反向传播类
│   ├── gru_quant.h             # 量化 GRU 前向传播类
│   ├── gru_interface.hpp       # 统一接口层（校准、量化、前向传播）
│   ├── quantize_bitwidth_config.hpp  # 量化位宽配置
│   ├── quantize_ops.cuh        # 量化操作 CUDA 内核
│   ├── histogram_collector.hpp # 直方图收集器（AIMET 风格校准）
│   ├── pot_sqnr_calibrator.hpp # SQNR 校准器
│   └── ...
├── src/                        # C++/CUDA 源文件
│   ├── gru_forward_gpu.cu      # 浮点前向传播 GPU 实现
│   ├── gru_forward_gpu_quant.cu # 量化前向传播 GPU 实现
│   ├── gru_backward_gpu.cu     # 反向传播 GPU 实现
│   ├── gru_interface.cpp       # 接口实现
│   └── quantize_ops.cu         # 量化操作实现
├── pytorch/                    # PyTorch 绑定和 Python 接口
│   ├── custom_gru.py           # 自定义 GRU 类（支持量化）
│   ├── setup.py                # Python 扩展编译配置
│   ├── lib/                    # 编译生成的库文件
│   ├── config/                 # 配置文件
│   │   └── gru_quant_bitwidth_config.json  # 量化位宽配置
│   └── test_*.py               # 测试脚本
├── example/                    # C++ 使用示例
│   └── gru.cc                  # 浮点/量化 GRU 对比示例
├── CMakeLists.txt              # CMake 构建配置
├── gru_train.py                # PyTorch 训练示例（语音识别）
├── Dockerfile                  # Docker 构建文件
└── docker-compose.yml          # Docker Compose 配置
```

## 🔧 环境要求

- **CUDA Toolkit** >= 11.0
- **cuBLAS**
- **C++17** 编译器
- **CMake** >= 3.18
- **Python** >= 3.8
- **PyTorch** >= 1.10（支持 CUDA）
- **OpenMP**

## 🚀 快速开始

### 1. 编译 C++ 库

```bash
# 创建构建目录
mkdir build && cd build

# 配置 CMake
cmake ..

# 编译
make -j$(nproc)
```

编译完成后会生成：
- `pytorch/lib/libgru_quant_static.a` - 静态库
- `pytorch/lib/libgru_quant_shared.so` - 动态库
- `gru_example` - C++ 示例程序

### 2. 编译 Python 扩展

```bash
cd pytorch

# 安装 Python 扩展
pip install -e .
```

### 3. 使用示例

#### Python 使用（非量化模式）

```python
from pytorch.custom_gru import CustomGRU
import torch

# 创建模型（与 nn.GRU 接口一致）
gru = CustomGRU(
    input_size=64,
    hidden_size=128,
    batch_first=True,
    bidirectional=False
).cuda()

# 前向传播
input_data = torch.randn(32, 50, 64).cuda()  # [batch, seq_len, input_size]
output, h_n = gru(input_data)
# output: [32, 50, 128], h_n: [1, 32, 128]
```

#### Python 使用（量化模式）

```python
from pytorch.custom_gru import CustomGRU
import torch

# 1. 创建模型
gru = CustomGRU(
    input_size=64,
    hidden_size=128,
    batch_first=True
).cuda()

# 2. (可选) 加载自定义位宽配置
gru.load_bitwidth_config("config/gru_quant_bitwidth_config.json", verbose=True)

# 3. 使用校准数据进行量化校准
for batch in calibration_loader:
    gru.calibrate(batch.cuda())

# 4. 完成校准，计算量化参数
gru.finalize_calibration(verbose=True)

# 5. 开启量化模式进行推理
gru.use_quantization = True
output, h_n = gru(input_data)
```

#### 设置校准方法

```python
# MinMax 校准（默认，速度快）
gru.calibration_method = 'minmax'

# AIMET 风格直方图校准（精度高，推荐）
gru.calibration_method = 'histogram'
```

## ⚙️ 量化配置

### 量化位宽配置文件格式

配置文件 `pytorch/config/gru_quant_bitwidth_config.json`：

```json
{
  "GRU_config": {
    "default_config": {
      "disable_quantization": false
    },
    "operator_config": {
      "input.x": { "bitwidth": 8, "is_symmetric": false },
      "input.h": { "bitwidth": 8, "is_symmetric": false },
      "weight.W": { "bitwidth": 8, "is_symmetric": true },
      "weight.R": { "bitwidth": 8, "is_symmetric": true },
      "gate.z_out": { "bitwidth": 8, "is_symmetric": false },
      ...
    }
  }
}
```

### 可配置的算子

| 类别 | 算子名 | 说明 |
|------|--------|------|
| 输入 | `input.x`, `input.h` | 输入序列和隐藏状态 |
| 权重 | `weight.W`, `weight.R`, `weight.bx`, `weight.br` | 权重矩阵和偏置 |
| 矩阵乘法 | `matmul.Wx`, `matmul.Rh` | 矩阵乘法中间结果 |
| 门控 | `gate.z_pre/out`, `gate.r_pre/out`, `gate.g_pre/out` | 门控激活前后 |
| 运算 | `op.Rh_add_br`, `op.rRh`, `op.old_contrib`, `op.new_contrib` | 中间运算 |

### 快速设置所有位宽

```python
# 设置所有算子使用 8bit 对称量化
gru.set_all_bitwidth(8, is_symmetric=True)

# 设置所有算子使用 16bit 非对称量化
gru.set_all_bitwidth(16, is_symmetric=False)
```

## 📐 GRU 公式

本项目实现的 GRU 遵循以下计算公式：

```
z_t = σ(W_z · x_t + R_z · h_{t-1} + b_z)        # 更新门
r_t = σ(W_r · x_t + R_r · h_{t-1} + b_r)        # 重置门
g_t = tanh(W_g · x_t + r_t ⊙ (R_g · h_{t-1}) + b_g)  # 候选隐藏状态
h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ g_t          # 新隐藏状态
```

其中：
- `σ` 表示 Sigmoid 激活函数
- `⊙` 表示逐元素乘法

## 🧪 运行测试

### C++ 测试

```bash
./build/gru_example
```

### Python 测试

```bash
cd pytorch
python test_custom_gru_quantization.py
```

## 🔬 校准方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **MinMax** | 速度快，实现简单 | 对异常值敏感 | 快速原型验证 |
| **Histogram (AIMET)** | 精度高，SQNR 优化 | 计算开销稍大 | 生产部署 |

## 📊 性能

量化后的 GRU 相比浮点版本：
- **内存占用**：减少约 75%（8-bit 量化）
- **计算速度**：提升约 2-4x（取决于硬件）
- **精度损失**：< 1%（使用 Histogram 校准）

## 🐳 Docker 使用

```bash
# 构建镜像
docker-compose build

# 运行容器
docker-compose up -d

# 进入容器
docker-compose exec quant-gru bash
```

## 📝 API 参考

### CustomGRU 类

```python
class CustomGRU(nn.Module):
    def __init__(
        self,
        input_size: int,           # 输入特征维度
        hidden_size: int,          # 隐藏状态维度
        num_layers: int = 1,       # 层数（目前仅支持 1）
        bias: bool = True,         # 是否使用偏置
        batch_first: bool = False, # 输入格式
        bidirectional: bool = False,  # 是否双向
        use_quantization: bool = False  # 是否启用量化
    )
```

### 主要方法

| 方法 | 说明 |
|------|------|
| `forward(input, hx=None)` | 前向传播 |
| `calibrate(data)` | 累积校准数据 |
| `finalize_calibration(verbose=False)` | 完成校准，计算量化参数 |
| `reset_calibration()` | 重置校准状态 |
| `load_bitwidth_config(path, verbose=False)` | 加载位宽配置 |
| `set_all_bitwidth(bitwidth, is_symmetric=True)` | 设置统一位宽 |
| `is_calibrated()` | 检查是否已校准 |
| `print_quant_params()` | 打印量化参数 |
| `print_quant_ranges()` | 打印量化范围 |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📚 参考

- [AIMET (AI Model Efficiency Toolkit)](https://github.com/quic/aimet)
- [Haste: Fast RNN Library](https://github.com/lmnt-com/haste)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

