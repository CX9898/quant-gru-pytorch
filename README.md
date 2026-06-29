# Quant-GRU-PyTorch

一个高性能的量化 GRU（门控循环单元）实现，基于 CUDA 和 PyTorch，支持训练和推理的量化感知计算。

## 📋 项目简介

本项目实现了一个支持量化的 GRU 神经网络模块，核心使用 CUDA 编写以实现高性能计算，并提供与 PyTorch 兼容的 Python 接口。项目支持：

- **浮点和量化两种模式**：可在训练和推理时自由切换
- **灵活的细粒度量化配置**：支持按算子设置位宽、对称/非对称量化和量化粒度（`PER_TENSOR` / `PER_GATE` / `PER_CHANNEL`）
- **任意 bit 混合精度**：每个算子可独立选择 1~32bit 量化
- **三种校准方法**：MinMax（默认，快速）、SQNR（高精度，推荐生产部署）和 Percentile（百分位裁剪）
- **量化参数复用**：支持量化参数导入导出，避免重复校准
- **AIMET 兼容**：支持 AIMET encodings 格式的导入导出
- **双向 GRU**：完整支持 bidirectional 模式
- **与 PyTorch 兼容**：`QuantGRU` 接口与 `nn.GRU` 一致，可无缝替换
- **ONNX 导出**：支持标准 GRU 单节点导出，便于硬件部署

具体更新内容请查看 [CHANGELOG.md](CHANGELOG.md)。

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
mkdir build && cd build
cmake ..
make -j$(nproc)
```

编译完成后会生成：
- `pytorch/lib/libgru_quant_shared.so` - 动态库
- `build/gru_example` - C++ 示例程序

### 2. 安装 Python 扩展

```bash
cd pytorch

# 开发模式安装（推荐，修改代码后立即生效）
pip install -e . --no-deps --no-build-isolation

# 或普通安装（修改代码后需重新安装）
# pip install . --no-deps --no-build-isolation
```

### 3. 验证安装

```bash
# C++ 测试
./build/gru_example

# Python 测试
python pytorch/test_quant_gru.py
```

---

## 📐 GRU 公式

本项目实现的标准 GRU 递推（与 PyTorch `nn.GRU` 一致）：

```
z_t = σ(W_z · x_t + R_z · h_{t-1} + b_z)              # 更新门
r_t = σ(W_r · x_t + R_r · h_{t-1} + b_r)              # 重置门
g_t = tanh(W_g · x_t + r_t ⊙ (R_g · h_{t-1}) + b_g)   # 候选隐藏状态
h_t = z_t ⊙ h_{t-1} + (1 - z_t) ⊙ g_t                # 新隐藏状态
```

其中 `σ` 为 Sigmoid，`⊙` 为逐元素乘。量化配置中的算子名（如 `update_gate_output`、`weight_ih_linear`）对应该公式各步骤的中间结果；逐步展开与命名对照见 [pytorch/config/README.md](pytorch/config/README.md)。

---

## 📖 典型工作流

`QuantGRU` 与 `nn.GRU` 接口一致。浮点推理直接 `forward`；量化推理遵循 **配置 → 校准 → 启用量化** 三步。

```python
from quant_gru import QuantGRU
import torch

gru = QuantGRU(input_size=64, hidden_size=128, batch_first=True).cuda()

# --- 浮点推理 ---
x = torch.randn(32, 50, 64).cuda()
output, h_n = gru(x)

# --- 量化推理 (PTQ) ---
# 1) 加载位宽配置（或 gru.set_all_bitwidth(8, is_symmetric=True)）
gru.load_bitwidth_config("pytorch/config/gru_quant_bitwidth_config.json")
# 可选: gru.calibration_method = 'sqnr'  # 默认 'minmax'，见下方对比表

# 2) 用代表性数据校准（forward 同时收集统计量）
gru.calibrating = True
for batch in calibration_loader:
    gru(batch.cuda())
gru.calibrating = False

# 3) 启用量化推理（首次 forward 会自动 finalize 校准参数）
gru.use_quantization = True
output, h_n = gru(x)

# --- 复用量化参数（跳过重新校准）---
gru.export_quant_params("quant_params.json")
gru2 = QuantGRU(input_size=64, hidden_size=128, batch_first=True).cuda()
gru2.load_state_dict(gru.state_dict(), strict=False)
gru2.load_quant_params("quant_params.json")
gru2.use_quantization = True
```

> 配置文件中的 `disable_quantization: true` 可强制该配置走浮点路径；默认 `false` 表示启用量化。

### 量化感知训练 (QAT)

完成上述校准并设置 `gru.use_quantization = True` 后，按常规 PyTorch 训练循环 `loss.backward()` 即可（前向量化、反向浮点）。完整示例见 `pytorch/example/example_usage.py` 中的 `example_training()`。

### 校准方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **MinMax** ⭐ 默认 | 速度快，实现简单 | 对异常值敏感 | 快速原型验证 |
| **SQNR** | 精度最高，自动搜索最优 scale | 计算开销稍大 | 生产部署（推荐） |
| **Percentile** | 可配置裁剪比例，抗异常值 | 需调参 | 数据有异常值时 |

设置方式：`gru.calibration_method = 'minmax' | 'sqnr' | 'percentile'`；Percentile 可额外设置 `gru.percentile_value`（默认 `100.0`）。

---

## ⚙️ 量化配置

位宽 JSON 格式、算子列表、字段语义与 GRU 计算流程见 **[pytorch/config/README.md](pytorch/config/README.md)**。

常用快捷接口：

```python
gru.set_all_bitwidth(8, is_symmetric=True)          # 全部算子 8bit 对称
gru.adjust_quant_config("update_gate_output", bitwidth=16)
config = gru.get_quant_config("update_gate_output")  # 不传参数则返回全部算子
```

---

## 📦 ONNX 导出

`export_mode=True` 时走 ONNX 单节点 `GRU` 导出路径；默认 `False` 使用 CUDA 高性能推理。

**要点：**

1. 导出前设 `gru.export_mode = True`，导出后恢复 `False`
2. 须调用 `ensure_quant_gru_onnx_registered(opset=...)`，且与 `torch.onnx.export(opset_version=...)` **一致**（支持 `opset>=13`）
3. 使用 legacy exporter：`torch.onnx.export(..., dynamo=False)`
4. 传入 `custom_opsets=get_quant_gru_custom_opsets()`

**与 `aimet_rx` 分工：** 本项目提供标准 ONNX `GRU` 节点与量化参数接口；整模型导出、encodings 合并由 `aimet_rx/export_onnx_and_encodings` 负责。多 GRU 模型建议导出前调用 `set_quant_gru_module_names(model)`。

完整代码见 `example_onnx_export()`。

---

## 📚 常见场景索引

完整可运行示例见 [`pytorch/example/example_usage.py`](pytorch/example/example_usage.py)。

| 场景 | 示例函数 |
|------|----------|
| 基础浮点使用 | `example_basic_usage()` |
| JSON 配置 + PTQ | `example_quantization_with_json()` |
| 手动统一位宽 | `example_quantization_manual()` |
| 浮点 vs 量化精度对比 | `example_compare_precision()` |
| QAT 训练 | `example_training()` |
| 校准方法对比 | `example_calibration_method()` |
| 双向 GRU | `example_bidirectional()` |
| ONNX 单节点导出 | `example_onnx_export()` |
| 量化参数导入导出 | `example_quant_params_export_import()` |
| 单算子配置调整 | `example_adjust_quant_config()` |
| per-tensor / per-gate 权重 | `example_weight_bias_granularity()` |

---

## 📝 API 速查

### 常用属性

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `use_quantization` | `False` | 量化推理开关 |
| `calibrating` | `False` | `True` 时 forward 收集校准数据 |
| `calibration_method` | `'minmax'` | `'minmax'` / `'sqnr'` / `'percentile'` |
| `percentile_value` | `100.0` | 仅 percentile 校准使用 |
| `export_mode` | `False` | `True` 启用 ONNX 单节点导出路径 |

### 常用方法

| 方法 | 说明 |
|------|------|
| `forward(input, hx=None)` | 前向传播 |
| `load_bitwidth_config(path)` | 加载位宽 JSON |
| `set_all_bitwidth(bitwidth, is_symmetric=True)` | 统一位宽 |
| `export_quant_params(path)` / `load_quant_params(path)` | 量化参数导入导出 |
| `adjust_quant_config(operator, ...)` / `get_quant_config(operator=None)` | 调整/查询算子配置 |
| `set_quant_params_locked(locked)` | 锁定量化参数，防止校准覆盖 |
| `export_quant_params_to_aimet_format(...)` / `load_quant_params_from_aimet_format(...)` | AIMET encodings 互转 |

完整签名与 docstring 见 `pytorch/quant_gru.py`。

---

## 🏗️ 项目结构

```
quant-gru-pytorch/
├── include/ / src/          # C++/CUDA 核心实现
├── pytorch/                 # Python 绑定、配置、测试与示例
│   ├── quant_gru.py
│   ├── config/              # 位宽 JSON 与配置说明
│   └── example/example_usage.py
├── example/                 # C++ 示例
├── quant-gru-cpu-only/      # CPU 定点 Reference Model
└── CMakeLists.txt
```

算法与量化流程深度文档见 `docs/`。

---

## 📚 参考

- [AIMET (AI Model Efficiency Toolkit)](https://github.com/quic/aimet)
- [Haste: Fast RNN Library](https://github.com/lmnt-com/haste)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
