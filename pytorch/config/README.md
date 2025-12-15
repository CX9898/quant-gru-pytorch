# GRU 量化位宽配置文件说明

本文档介绍如何配置 `gru_quant_bitwidth_config.json` 文件来自定义 GRU 量化的位宽设置。

## GRU 公式

### 标准 GRU 公式

```
输入: x[t] (当前输入), h[t-1] (上一时刻隐藏状态)
输出: h[t] (当前隐藏状态)

z[t] = σ(W_z @ x[t] + R_z @ h[t-1] + bx_z + br_z)       # 更新门 (update gate)
r[t] = σ(W_r @ x[t] + R_r @ h[t-1] + bx_r + br_r)       # 重置门 (reset gate)
g[t] = tanh(W_g @ x[t] + bx_g + r[t] * (R_g @ h[t-1] + br_g))  # 候选状态 (candidate)
h[t] = z[t] * h[t-1] + (1 - z[t]) * g[t]                # 最终输出
```

### 变量说明

| 变量 | 维度 | 说明 |
|------|------|------|
| `x[t]` | [batch, input_size] | 当前时刻输入 |
| `h[t-1]` | [batch, hidden_size] | 上一时刻隐藏状态 |
| `W` | [input_size, hidden_size×3] | 输入权重 (W_z, W_r, W_g) |
| `R` | [hidden_size, hidden_size×3] | 循环权重 (R_z, R_r, R_g) |
| `bx` | [hidden_size×3] | 输入偏置 |
| `br` | [hidden_size×3] | 循环偏置 |
| `z[t]` | [batch, hidden_size] | 更新门输出，范围 [0,1] |
| `r[t]` | [batch, hidden_size] | 重置门输出，范围 [0,1] |
| `g[t]` | [batch, hidden_size] | 候选状态，范围 [-1,1] |
| `h[t]` | [batch, hidden_size] | 当前隐藏状态 |

### 计算流程与配置变量对应

```
步骤1: 矩阵乘法
├─ Wx = W @ x         → matmul.Wx (Wx_)
└─ Rh = R @ h         → matmul.Rh (Rh_)

步骤2: 更新门 (update gate)
├─ z_pre = Wx_z + Rh_z + bx_z + br_z   → gate.z_pre (z_pre_)
└─ z = sigmoid(z_pre)                   → gate.z_out (z_out_)

步骤3: 重置门 (reset gate)
├─ r_pre = Wx_r + Rh_r + bx_r + br_r   → gate.r_pre (r_pre_)
└─ r = sigmoid(r_pre)                   → gate.r_out (r_out_)

步骤4: 候选状态 (candidate)
├─ Rh_br = Rh_g + br_g                 → op.Rh_add_br (Rh_add_br_)
├─ rRh = r * Rh_br                     → op.rRh (rRh_)
├─ g_pre = Wx_g + bx_g + rRh           → gate.g_pre (g_pre_)
└─ g = tanh(g_pre)                     → gate.g_out (g_out_)

步骤5: 最终输出
├─ one_minus_z = 1 - z                 → op.one_minus_update (one_minus_update_)
├─ old = z * h[t-1]                    → op.old_contrib (old_contrib_)
├─ new = one_minus_z * g               → op.new_contrib (new_contrib_)
└─ h[t] = old + new
```

---

## 概述

该配置文件用于控制 GRU 各算子的量化位宽，对应 C++ 端的 `OperatorQuantConfig` 结构体。通过修改此配置，可以实现混合精度量化。

## 文件结构

```json
{
  "description": "配置文件描述",
  "comment": "注释说明",
  
  "default_bitwidth": { ... },
  "operator_config": { ... },
  "default_config": { ... }
}
```

## 字段说明

### 1. default_bitwidth（默认位宽）

```json
"default_bitwidth": {
  "weight": 8,
  "activation": 8
}
```

| 字段 | 说明 |
|------|------|
| `weight` | 权重默认位宽 |
| `activation` | 激活值默认位宽 |

### 2. operator_config（算子配置）

每个算子的配置格式：

```json
"算子名称": {
  "input_bitwidth": 8,      // 输入位宽（可选）
  "output_bitwidth": 8,     // 输出位宽
  "weight_bitwidth": 8,     // 权重位宽（权重算子使用）
  "is_symmetric": true,     // 是否对称量化
  "comment": "说明"         // 注释（可选）
}
```

#### 位宽选项

| 位宽值 | 数据类型 |
|--------|----------|
| 8 | INT8 (-128 ~ 127) 或 UINT8 (0 ~ 255) |
| 16 | INT16 (-32768 ~ 32767) 或 UINT16 (0 ~ 65535) |
| 32 | INT32 |

#### is_symmetric 说明

`is_symmetric` 决定量化时 **zero point (zp)** 的计算方式：

| 值 | 说明 | zero point | 适用场景 |
|----|------|------------|----------|
| `true` | 对称量化 | `zp = 0`（固定） | 数据分布关于 0 对称，如权重、tanh 输出 [-1,1] |
| `false` | 非对称量化 | `zp ≠ 0`（根据数据范围计算） | 数据分布不对称，如 sigmoid 输出 [0,1]、ReLU 输出 |

**量化公式：**
```
量化：   int_value = round(float_value / scale) + zero_point
反量化： float_value = scale × (int_value - zero_point)
```

**对称 vs 非对称：**
- **对称量化**：`zp = 0`，计算更简单高效，但对非对称分布的数据会浪费量化范围
- **非对称量化**：`zp ≠ 0`，能更充分利用量化范围，但计算需要额外处理零点偏移

### 3. 算子列表

#### 输入类

| 算子名 | 说明 |
|--------|------|
| `input.x` | 输入序列 x |
| `input.h` | 隐藏状态 h |

#### 权重类

| 算子名 | 说明 |
|--------|------|
| `weight.W` | 输入权重 W (input → gates) |
| `weight.R` | 循环权重 R (hidden → gates) |
| `weight.bx` | 输入偏置 bx |
| `weight.br` | 循环偏置 br |

#### 矩阵乘法类

| 算子名 | 说明 |
|--------|------|
| `matmul.Wx` | W @ x 的输出 |
| `matmul.Rh` | R @ h 的输出 |

#### 门控类

| 算子名 | 说明 | 推荐 is_symmetric |
|--------|------|-------------------|
| `gate.z_pre` | 更新门 sigmoid 激活前 | `true` |
| `gate.z_out` | 更新门 sigmoid 激活后 [0,1] | `false` |
| `gate.r_pre` | 重置门 sigmoid 激活前 | `true` |
| `gate.r_out` | 重置门 sigmoid 激活后 [0,1] | `false` |
| `gate.g_pre` | 候选门 tanh 激活前 | `true` |
| `gate.g_out` | 候选门 tanh 激活后 [-1,1] | `true` |

#### 运算类

| 算子名 | 说明 |
|--------|------|
| `op.Rh_add_br` | Rh + br 加法 |
| `op.rRh` | r * Rh 元素乘法 |
| `op.one_minus_update` | 1 - z 减法 |
| `op.old_contrib` | z * h 旧状态贡献 |
| `op.new_contrib` | (1-z) * g 新状态贡献 |

### 4. default_config（默认配置）

```json
"default_config": {
  "disable_quantization": false
}
```

| 字段 | 说明 |
|------|------|
| `disable_quantization` | 是否禁用量化（`true` 禁用，`false` 启用）|

## 使用方法

### Python 端

```python
from custom_gru import CustomGRU, load_bitwidth_config, apply_bitwidth_config

# 方式 1: 创建 GRU 后加载配置
gru = CustomGRU(input_size=64, hidden_size=128, use_quantization=True)
gru.load_bitwidth_config("config/gru_quant_bitwidth_config.json", verbose=True)

# 方式 2: 直接加载配置对象
import gru_interface_binding as gru_ops
config = gru_ops.OperatorQuantConfig()
apply_bitwidth_config(config, "config/gru_quant_bitwidth_config.json", verbose=True)

# 校准流程
for batch in calibration_loader:
    gru.calibrate(batch)
gru.finalize_calibration()  # 使用已加载的位宽配置

# 正常推理
output, h_n = gru(input_data)
```

## 配置示例

### 示例 1: 全 INT8 配置（默认）

```json
{
  "operator_config": {
    "input.x": { "output_bitwidth": 8, "is_symmetric": true },
    "gate.z_out": { "output_bitwidth": 8, "is_symmetric": false },
    "gate.r_out": { "output_bitwidth": 8, "is_symmetric": false },
    ...
  }
}
```

### 示例 2: 混合精度配置

```json
{
  "operator_config": {
    "weight.W": { "weight_bitwidth": 16, "is_symmetric": true, "comment": "权重使用 INT16" },
    "weight.R": { "weight_bitwidth": 16, "is_symmetric": true },
    "matmul.Wx": { "output_bitwidth": 16, "is_symmetric": true, "comment": "矩阵乘结果使用 INT16" },
    "matmul.Rh": { "output_bitwidth": 16, "is_symmetric": true },
    "gate.z_out": { "output_bitwidth": 8, "is_symmetric": false },
    ...
  }
}
```

## GRU 计算流程参考

```
输入: x[t], h[t-1]

1. 矩阵乘法
   Wx = W @ x          # matmul.Wx
   Rh = R @ h          # matmul.Rh

2. 门控计算
   z_pre = Wx_z + Rh_z + bx_z + br_z    # gate.z_pre
   z = sigmoid(z_pre)                    # gate.z_out (更新门)
   
   r_pre = Wx_r + Rh_r + bx_r + br_r    # gate.r_pre
   r = sigmoid(r_pre)                    # gate.r_out (重置门)
   
   Rh_br = Rh_g + br_g                  # op.Rh_add_br
   rRh = r * Rh_br                      # op.rRh
   g_pre = Wx_g + bx_g + rRh            # gate.g_pre
   g = tanh(g_pre)                      # gate.g_out (候选门)

3. 输出计算
   one_minus_z = 1 - z                  # op.one_minus_update
   old = z * h[t-1]                     # op.old_contrib
   new = one_minus_z * g                # op.new_contrib
   h[t] = old + new                     # 最终输出
```

## 注意事项

1. **sigmoid 输出**：`gate.z_out` 和 `gate.r_out` 建议设置 `is_symmetric: false`
   - sigmoid 输出范围是 [0,1]，分布不对称
   - 非对称量化能更充分利用量化范围

2. **tanh 输出**：`gate.g_out` 建议设置 `is_symmetric: true`
   - tanh 输出范围是 [-1,1]，分布关于 0 对称
   - 对称量化 `zp=0` 计算更高效

3. **权重**：一般使用 `is_symmetric: true`
   - 权重分布通常接近对称
   - 对称量化无需存储和处理零点

4. **混合精度**：可以对关键算子（如权重、矩阵乘法）使用更高精度来提升模型准确率

5. **位宽配置需在 `finalize_calibration()` 之前加载**

