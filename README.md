# Quant-GRU-PyTorch

一个高性能的量化 GRU（门控循环单元）实现，基于 CUDA 和 PyTorch，支持训练和推理的量化感知计算。

## 📋 项目简介

本项目实现了一个支持量化的 GRU 神经网络模块，核心使用 CUDA 编写以实现高性能计算，并通过 PyBind11 提供 PyTorch 接口。项目支持：

- **浮点和量化两种模式**：可在训练和推理时自由切换
- **灵活的量化配置**：支持 8/16 位量化，可配置对称/非对称量化
- **两种校准方法**：Histogram（AIMET 风格，默认，高精度）和 MinMax（快速）
- **双向 GRU**：完整支持 bidirectional 模式
- **与 PyTorch 兼容**：`QuantGRU` 接口与 `nn.GRU` 一致，可无缝替换
- **ONNX 导出**：支持 QDQ 格式导出，便于部署到各类推理引擎

## 🔧 环境要求

- **Python** >= 3.9
- **PyTorch** >= 2.0（支持 CUDA）
- **CUDA Toolkit** >= 11.0（含 cuBLAS）
- **C++17** 编译器（GCC 7+ 或 Clang 5+）
- **CMake** >= 3.18
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

# 安装 Python 扩展（开发模式）
pip install -e .
```

### 3. 验证安装

```bash
# C++ 测试
./build/gru_example

# Python 测试
cd pytorch
python test_quant_gru.py
```

---

## 📖 使用示例

### 基础使用（浮点模式）

```python
from quant_gru import QuantGRU
import torch

# 创建模型（与 nn.GRU 接口一致）
gru = QuantGRU(
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

### 量化推理

```python
from quant_gru import QuantGRU
import torch

# 1. 创建模型
gru = QuantGRU(
    input_size=64,
    hidden_size=128,
    batch_first=True
).cuda()

# 2. 加载位宽配置（二选一）
# 方式一：从配置文件加载
gru.load_bitwidth_config("pytorch/config/gru_quant_bitwidth_config.json", verbose=True)
# 方式二：直接设置统一位宽（8/16位，is_symmetric控制对称量化）
# gru.set_all_bitwidth(bitwidth=8, is_symmetric=True, verbose=True)
# gru.use_quantization = True  # 启用量化

# 3. 使用校准数据进行量化校准
for batch in calibration_loader:
    gru.calibrate(batch.cuda())

# 4. 推理（首次前向时会自动完成校准）
output, h_n = gru(input_data)
```

> 💡 **量化开关**：配置文件中的 `disable_quantization` 控制是否启用量化：
> - `false`（默认）：启用量化推理
> - `true`：使用浮点推理

### 量化感知训练 (QAT)

```python
from quant_gru import QuantGRU
import torch

gru = QuantGRU(input_size=64, hidden_size=128, batch_first=True).cuda()
gru.load_bitwidth_config("pytorch/config/gru_quant_bitwidth_config.json")

# 校准
for batch in calibration_loader:
    gru.calibrate(batch.cuda())

# 训练循环（前向使用量化，反向使用浮点）
optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
gru.train()

for epoch in range(num_epochs):
    for x, target in train_loader:
        optimizer.zero_grad()
        output, _ = gru(x.cuda())
        loss = criterion(output, target.cuda())
        loss.backward()
        optimizer.step()
```

### 校准方法选择

```python
# AIMET 风格直方图校准（默认，精度高，推荐用于生产部署）
gru.calibration_method = 'histogram'

# MinMax 校准（速度快，适合快速原型验证）
gru.calibration_method = 'minmax'
```

### ONNX 导出

#### 浮点模型导出（默认）

```python
from quant_gru import QuantGRU
import torch

gru = QuantGRU(input_size=64, hidden_size=128, batch_first=True).cuda()
gru.eval()

# 启用导出模式（默认使用浮点格式）
gru.export_mode = True

# 导出 ONNX
dummy_input = torch.randn(1, 50, 64).cuda()
torch.onnx.export(
    gru, dummy_input, "gru_float.onnx",
    input_names=['input'],
    output_names=['output', 'hidden'],
    dynamic_axes={'input': {0: 'batch', 1: 'seq_len'},
                  'output': {0: 'batch', 1: 'seq_len'}},
    dynamo=False  # PyTorch 2.x 需要此参数使用传统导出
)

gru.export_mode = False  # 恢复 CUDA 模式
```

#### 量化模型导出（QDQ 格式）

```python
# 1. 创建并校准模型
gru = QuantGRU(input_size=64, hidden_size=128, batch_first=True).cuda()
gru.load_bitwidth_config("pytorch/config/gru_quant_bitwidth_config.json")

for batch in calibration_loader:
    gru.calibrate(batch.cuda())
gru.finalize_calibration()

# 2. 启用导出模式，指定 QDQ 格式
gru.export_mode = True
gru.export_format = 'qdq'  # 使用 QDQ 伪量化格式
gru.eval()

# 3. 导出 ONNX
torch.onnx.export(
    gru, dummy_input, "gru_quantized.onnx",
    input_names=['input'],
    output_names=['output', 'hidden'],
    dynamic_axes={'input': {0: 'batch', 1: 'seq_len'},
                  'output': {0: 'batch', 1: 'seq_len'}},
    dynamo=False
)

gru.export_mode = False  # 恢复 CUDA 模式
```

> 💡 **导出格式说明**：
> | `export_format` | 说明 | 适用场景 |
> |-----------------|------|----------|
> | `'float'` | 浮点格式（默认） | 非量化模型部署 |
> | `'qdq'` | QDQ 伪量化格式 | 量化模型部署，推理引擎自动优化 |

> 💡 **提示**：更多详细示例请参阅 `pytorch/example/example_usage.py`

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

## 🔬 校准方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Histogram (AIMET)** ⭐ 默认 | 精度高，SQNR 优化 | 计算开销稍大 | 生产部署 |
| **MinMax** | 速度快，实现简单 | 对异常值敏感 | 快速原型验证 |

## 📦 ONNX 导出

### 导出模式

`QuantGRU` 支持通过 `export_mode` 属性切换到纯 PyTorch 实现，以便 ONNX 追踪：

| 属性 | 说明 |
|------|------|
| `export_mode=False` | **默认**，使用 CUDA C++ 实现（高性能推理） |
| `export_mode=True` | 使用纯 PyTorch 实现（可被 ONNX 追踪） |

### 导出格式选择

通过 `export_format` 属性设置具体的导出格式：

| 格式 | 说明 | 适用场景 |
|------|------|----------|
| `'float'` | **默认**，浮点格式 | 非量化模型部署、与 Haste GRU 行为一致 |
| `'qdq'` | QDQ（Quantize-Dequantize）格式 | 量化模型部署（TensorRT、ONNX Runtime） |

```python
# 设置导出格式
gru.export_format = 'float'      # 默认，浮点
gru.export_format = 'qdq'        # 量化模型推荐
```

## 📝 API 参考

### QuantGRU 类

```python
class QuantGRU(nn.Module):
    def __init__(
        self,
        input_size: int,              # 输入特征维度
        hidden_size: int,             # 隐藏状态维度
        num_layers: int = 1,          # 层数（目前仅支持 1）
        bias: bool = True,            # 是否使用偏置
        batch_first: bool = False,    # 输入格式
        bidirectional: bool = False,  # 是否双向
        use_quantization: bool = False  # 是否启用量化
    )
    
    # 重要属性（可在创建后设置）
    gru.export_mode = True           # ONNX 导出模式
    gru.calibration_method = 'histogram'  # 校准方法（默认）
```

### 主要属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_quantization` | bool | False | 是否启用量化推理 |
| `export_mode` | bool | False | ONNX 导出模式（True 时使用纯 PyTorch 实现） |
| `export_format` | str | 'float' | 导出格式：'float'、'qdq' |
| `calibration_method` | str | 'histogram' | 校准方法：'histogram'（高精度）或 'minmax'（快速） |

### 主要方法

| 方法 | 说明 |
|------|------|
| `forward(input, hx=None)` | 前向传播（量化模式下会自动完成校准） |
| `calibrate(data)` | 累积校准数据 |
| `finalize_calibration(verbose=False)` | 手动完成校准（通常无需调用，forward 会自动处理） |
| `reset_calibration()` | 重置校准状态 |
| `load_bitwidth_config(path, verbose=False)` | 加载位宽配置 |
| `set_all_bitwidth(bitwidth, is_symmetric=True)` | 设置统一位宽 |
| `is_calibrated()` | 检查是否已校准 |

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
│   ├── quant_gru.py            # 量化 GRU 类（含 CUDA 和纯 PyTorch 双实现）
│   ├── setup.py                # Python 扩展编译配置
│   ├── lib/                    # 编译生成的库文件
│   ├── config/                 # 配置文件
│   │   └── gru_quant_bitwidth_config.json  # 量化位宽配置
│   └── test_*.py               # 测试脚本
├── example/                    # C++ 使用示例
│   └── gru.cc                  # 浮点/量化 GRU 对比示例
├── CMakeLists.txt              # CMake 构建配置
```

## 📚 参考

- [AIMET (AI Model Efficiency Toolkit)](https://github.com/quic/aimet)
- [Haste: Fast RNN Library](https://github.com/lmnt-com/haste)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

