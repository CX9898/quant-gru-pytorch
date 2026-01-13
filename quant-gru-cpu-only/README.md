# Fixed-Point GRU CPU Reference Model

纯 C++ 定点 GRU 前向传播参考实现。

## 编译

```bash
mkdir build && cd build
cmake ..
make -j
```

## 文件

```
include/
├── gru_quant_cpu.h            # GRU 类
├── quantize_bitwidth_config.h # 位宽配置
├── quantize_lut_types.h       # LUT 类型
├── quantize_ops_helper.h      # 核心计算
└── quantize_param_types.h     # 参数结构
src/
├── gru_forward_cpu_quant.cc   # GRU 实现
└── quantize_lut.cc            # LUT 生成
```
