# Quant GRU CPU Only

纯 CPU 实现的定点量化 GRU 前向传播，不依赖 CUDA 或 PyTorch。

## 特性

- **纯 C++11 实现，无任何外部依赖**
- 支持 8-bit 和 16-bit 量化
- 与 GPU 版本数值完全一致
- 可在任意平台编译运行

## 编译

```bash
mkdir build && cd build
cmake ..
make -j
```

## 使用

### 运行示例

```bash
# 默认参数
./gru_cpu_example

# 自定义参数
./gru_cpu_example --input-size 64 --hidden-size 128 --batch-size 16 --seq-len 20

# 16-bit 量化
./gru_cpu_example --bitwidth 16

# 帮助
./gru_cpu_example --help
```

### 集成到项目

```cpp
#include "gru_quant_cpu.h"

// 创建前向传播对象
cpu::ForwardPassQuantCPU<int8_t, int8_t, int8_t, int8_t> forward_pass(
    false, batch_size, input_size, hidden_size);

// 设置量化参数
GRUQuantitativeParameters quant_params;
// ... 设置参数 ...
forward_pass.setRescaleParam(quant_params);

// 运行前向传播
forward_pass.Run(seq_len, W_quant, R_quant, bx_quant, br_quant,
                 x_quant, h_quant, nullptr, 0.0f, nullptr);
```

## 文件结构

```
quant-gru-cpu-only/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── gru_quant_cpu.h             # CPU GRU 类声明
│   ├── quantize_bitwidth_config.h  # 位宽配置
│   ├── quantize_lut_types.h        # LUT 类型定义
│   └── quantize_ops_helper.h       # 量化辅助函数
├── src/
│   ├── gru_forward_cpu_quant.cc    # CPU GRU 实现
│   └── quantize_lut.cc             # LUT 生成
└── example/
    └── main.cc                     # 示例程序
```

## 量化公式

```
量化:   q = round(x * 2^exp2_inv) + zp
反量化: x = (q - zp) * 2^(-exp2_inv)
```

## 支持的模板实例化

- `ForwardPassQuantCPU<int8_t, int8_t, int8_t, int8_t>` - 全 8-bit
- `ForwardPassQuantCPU<int16_t, int16_t, int8_t, int8_t>` - 16-bit 激活 + 8-bit 权重
- `ForwardPassQuantCPU<int8_t, int8_t, int16_t, int16_t>` - 8-bit 激活 + 16-bit 权重
- `ForwardPassQuantCPU<int16_t, int16_t, int16_t, int16_t>` - 全 16-bit

