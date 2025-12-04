# GRU PyTorch - 量化GRU实现

一个高性能的GRU（门控循环单元）PyTorch实现，支持量化和非量化两种模式，基于CUDA/C++后端优化。

## 📋 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [项目结构](#项目结构)
- [安装与编译](#安装与编译)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
  - [训练示例](#训练示例)
  - [量化推理示例](#量化推理示例)
- [API文档](#api文档)
- [技术细节](#技术细节)
- [性能优化](#性能优化)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🎯 项目简介

本项目实现了一个高性能的GRU模块，具有以下特点：

- **完全兼容PyTorch接口**：继承自`nn.GRU`，可直接替换标准GRU
- **支持量化推理**：支持int8和int16量化，大幅降低内存占用和计算开销
- **CUDA加速**：基于HASTE GRU的C++/CUDA实现，充分利用GPU并行计算
- **支持训练和推理**：完整的反向传播支持，可用于端到端训练
- **灵活的量化策略**：支持分段线性量化、二次多项式量化等多种量化方案

## ✨ 主要特性

### 1. 量化支持
- ✅ **int8量化**：4倍内存压缩，适合移动端和边缘设备
- ✅ **int16量化**：更高精度，适合对精度要求较高的场景
- ✅ **量化感知训练（QAT）**：支持训练时量化，提升量化后精度
- ✅ **动态量化**：前向传播时实时量化权重，支持训练时权重更新

### 2. 性能优化
- ✅ **CUDA加速**：使用cuBLAS进行矩阵运算优化
- ✅ **内存优化**：量化权重减少内存占用
- ✅ **计算优化**：整数运算加速推理速度

### 3. 功能完整性
- ✅ **初始隐藏状态支持**：支持自定义初始隐藏状态
- ✅ **批量处理**：支持batch_first和标准序列格式
- ✅ **梯度计算**：完整的反向传播支持
- ✅ **训练模式**：支持训练和推理两种模式

## 📁 项目结构

```
gru-pytorch/
├── pytouch/                  # Python实现和接口
│   ├── custom_gru.py        # CustomGRU主实现
│   ├── gru_train.py         # 训练示例
│   ├── example_custom_gru.py # 使用示例
│   ├── setup.py             # Python扩展编译配置
│   └── lib/                 # 编译后的库文件
├── include/                  # C++头文件
│   ├── gru_interface.hpp   # GRU接口定义
│   ├── gru.h                # GRU核心实现
│   ├── gru_quant.h          # 量化相关定义
│   └── ...
├── src/                      # C++/CUDA源文件
│   ├── gru_interface.cpp    # 统一接口
│   ├── gru_forward_gpu.cu   # 前向传播CUDA实现
│   ├── gru_backward_gpu.cu  # 反向传播CUDA实现
│   ├── gru_forward_gpu_quant.cu # 量化前向传播CUDA实现
│   └── quantize_ops.cu      # 量化操作实现
├── example/                  # C++示例代码
│   ├── gru.cc               # GRU测试和基准测试程序
├── CMakeLists.txt           # CMake构建配置
└── README.md                # 本文件
```

## 🔧 安装与编译

### 前置要求

- **Python**: >= 3.7
- **PyTorch**: >= 1.8.0 (支持CUDA)
- **CUDA**: >= 10.0
- **CMake**: >= 3.18
- **C++编译器**: 支持C++17
- **cuBLAS**: CUDA工具包的一部分

### 编译步骤

#### 1. 编译C++/CUDA库

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置CMake
cmake ..

# 编译
make -j$(nproc)

# 库文件将输出到 pytorch/lib/ 目录
```

#### 2. 编译Python扩展

```bash
cd pytorch

# 编译Python扩展（开发模式）
python setup.py build_ext --inplace

# 或者安装为包
python setup.py install
```

### 验证安装

```python
import torch
from custom_gru import CustomGRU

# 检查CUDA是否可用
assert torch.cuda.is_available(), "需要CUDA支持"

# 创建模型测试
gru = CustomGRU(input_size=128, hidden_size=256, use_quantization=False).cuda()
print("✅ 安装成功！")
```

## 🚀 快速开始

### 非量化模式

```python
import torch
from custom_gru import CustomGRU

# 创建模型
gru = CustomGRU(
    input_size=128,
    hidden_size=256,
    batch_first=True,
    use_quantization=False
).cuda()

# 前向传播
x = torch.randn(4, 100, 128).cuda()  # [batch, seq_len, input_size]
output, h_n = gru(x)

print(f"输出形状: {output.shape}")  # [4, 100, 256]
print(f"隐藏状态形状: {h_n.shape}")  # [1, 4, 256]
```

### 量化模式（int8）

```python
import torch
from custom_gru import CustomGRU

# 准备校准数据（用于量化参数校准）
calibration_data = torch.randn(4, 100, 128).cuda()

# 创建量化模型
gru = CustomGRU(
    input_size=128,
    hidden_size=256,
    batch_first=True,
    use_quantization=True,
    quant_type='int8',
    calibration_data=calibration_data
).cuda()

# 前向传播
x = torch.randn(4, 100, 128).cuda()
output, h_n = gru(x)
```

### 从PyTorch GRU迁移

```python
import torch
import torch.nn as nn
from custom_gru import CustomGRU

# 原始PyTorch GRU
pytorch_gru = nn.GRU(input_size=128, hidden_size=256, batch_first=True).cuda()

# 创建CustomGRU
custom_gru = CustomGRU(
    input_size=128,
    hidden_size=256,
    batch_first=True,
    use_quantization=False
).cuda()

# 复制权重
with torch.no_grad():
    custom_gru.weight_ih_l0.copy_(pytorch_gru.weight_ih_l0)
    custom_gru.weight_hh_l0.copy_(pytorch_gru.weight_hh_l0)
    custom_gru.bias_ih_l0.copy_(pytorch_gru.bias_ih_l0)
    custom_gru.bias_hh_l0.copy_(pytorch_gru.bias_hh_l0)

# 验证输出一致性
x = torch.randn(4, 100, 128).cuda()
with torch.no_grad():
    out1, h1 = pytorch_gru(x)
    out2, h2 = custom_gru(x)
    print(f"输出差异: {torch.max(torch.abs(out1 - out2)).item():.6f}")
```

## 📖 使用示例

### 训练示例

```python
import torch
import torch.nn as nn
from custom_gru import CustomGRU

# 定义模型
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gru = CustomGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            use_quantization=False  # 训练时通常不使用量化
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 取最后一个时间步
        return self.fc(out)

# 创建模型和优化器
model = GRUNet(input_size=128, hidden_size=256, num_classes=10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
```

### 量化推理示例

```python
# 1. 训练浮点模型（使用非量化模式）
model = train_float_model()

# 2. 准备校准数据
calibration_data = get_calibration_samples()

# 3. 创建量化模型并加载权重
quant_model = CustomGRU(
    input_size=128,
    hidden_size=256,
    batch_first=True,
    use_quantization=True,
    quant_type='int8',
    calibration_data=calibration_data
).cuda()

# 加载训练好的权重
quant_model.load_state_dict(model.state_dict())

# 4. 推理
quant_model.eval()
with torch.no_grad():
    output, h_n = quant_model(test_input)
```

## 📚 API文档

### CustomGRU

继承自`torch.nn.GRU`的自定义GRU类。

#### 参数

- `input_size` (int): 输入特征维度
- `hidden_size` (int): 隐藏状态维度
- `num_layers` (int, default=1): GRU层数（目前仅支持1层）
- `bias` (bool, default=True): 是否使用偏置
- `batch_first` (bool, default=False): 如果为True，输入形状为[batch, seq, feature]
- `dropout` (float, default=0.0): 层间dropout概率（目前不支持）
- `bidirectional` (bool, default=False): 是否双向（目前不支持）
- `use_quantization` (bool, default=False): 是否使用量化
- `quant_type` (str, default='int8'): 量化类型，'int8' 或 'int16'
- `calibration_data` (torch.Tensor, optional): 用于校准量化参数的输入数据
  - 形状: `[seq_len, batch, input_size]` 或 `[batch, seq_len, input_size]`（取决于batch_first）

#### 方法

- `forward(input, hx=None)`: 前向传播
  - `input`: 输入张量
  - `hx`: 初始隐藏状态，形状为`[num_layers, batch, hidden_size]`
  - 返回: `(output, h_n)`
    - `output`: 输出序列，形状与input相同但最后一维为hidden_size
    - `h_n`: 最终隐藏状态，形状为`[num_layers, batch, hidden_size]`

## 🔬 技术细节

### 量化实现

本项目实现了多种量化策略：

1. **权重量化**：将浮点权重量化为int8或int16整数
2. **激活量化**：使用分段线性量化或二次多项式量化
3. **动态量化**：前向传播时实时量化，支持训练时权重更新

### 格式转换

- **PyTorch格式**: 权重顺序为 (r, z, n) - 重置门、更新门、新门
- **HASTE格式**: 权重顺序为 (z, r, n) - 更新门、重置门、新门
- 自动处理两种格式之间的转换

### 反向传播

- 使用`torch.autograd.Function`实现自定义反向传播
- 支持完整的梯度计算，可用于端到端训练
- 反向传播统一使用float32权重，保证精度

## ⚡ 性能优化

### 内存优化

- 量化权重：int8量化可减少75%内存占用
- 共享中间结果：避免重复分配内存

### 计算优化

- CUDA并行计算：充分利用GPU并行能力
- cuBLAS优化：使用优化的矩阵运算库
- 整数运算：量化后使用整数运算加速

## ❓ 常见问题

### Q1: 编译时出现CUDA相关错误？

**A**: 确保：
1. CUDA工具包已正确安装
2. CMake能够找到CUDA（检查`CMAKE_CUDA_COMPILER`）
3. GPU架构匹配（在`setup.py`中调整`-arch=sm_XX`）

### Q2: 量化后精度下降明显？

**A**: 尝试：
1. 使用int16量化（更高精度）
2. 增加校准数据量
3. 使用量化感知训练（QAT）

### Q3: 训练时梯度为None？

**A**: 确保：
1. 模型处于训练模式（`model.train()`）
2. 输入张量设置了`requires_grad=True`
3. 使用支持量化的版本（已实现反向传播）

### Q4: 如何选择量化类型？

**A**: 
- **int8**: 适合内存受限场景，精度损失较大
- **int16**: 平衡精度和性能，推荐用于大多数场景

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范

- 遵循PEP 8（Python代码）
- 使用clang-format格式化C++代码
- 添加适当的注释和文档字符串

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 🙏 致谢

- 基于[HASTE GRU](https://github.com/lmnt-com/haste)实现
- 使用PyTorch和CUDA进行加速

## 📞 联系方式

如有问题或建议，请提交Issue或Pull Request。

---

**注意**: 本项目仍在积极开发中，API可能会有变化。建议查看最新文档和示例代码。
