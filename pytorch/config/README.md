# GRU 量化位宽配置文件说明

本文档介绍如何配置 `gru_quant_bitwidth_config.json` 文件来自定义 GRU 量化的位宽设置。

---

## 设计原则

### 1. 位宽与类型分离

| 层级 | 职责 | 说明 |
|------|------|------|
| **Python/JSON** | 只配置位宽数量 | 8, 16, 32 |
| **C++ 内部** | 决定实际类型 | INT8, INT16, UINT8 等 |

### 2. 类型自动选择规则

```
大多数操作 → INT 类型（有符号）
sigmoid 输出（z_out, r_out）→ UINT 类型（无符号）
```

**原因**：sigmoid 输出范围是 [0, 1]，使用 UINT 可以充分利用量化范围。

### 3. is_symmetric 与位宽类型解耦

`is_symmetric` **只影响 zero_point (zp) 计算**，不影响位宽类型：

| 值 | zero_point | 适用场景 |
|----|------------|----------|
| `true` | `zp = 0`（固定） | 对称分布：权重、tanh 输出 [-1,1] |
| `false` | `zp ≠ 0`（计算） | 非对称分布：sigmoid 输出 [0,1]、ReLU 输出 |

---

## GRU 公式

### 标准 GRU 计算流程

```
输入: x[t] (当前输入), h[t-1] (上一时刻隐藏状态)
输出: h[t] (当前隐藏状态)

z[t] = σ(W_z @ x[t] + R_z @ h[t-1] + bx_z + br_z)       # 更新门 (update gate)
r[t] = σ(W_r @ x[t] + R_r @ h[t-1] + bx_r + br_r)       # 重置门 (reset gate)
g[t] = tanh(W_g @ x[t] + bx_g + r[t] * (R_g @ h[t-1] + br_g))  # 候选状态 (candidate)
h[t] = z[t] * h[t-1] + (1 - z[t]) * g[t]                # 最终输出
```

### 计算流程与配置变量对应

```
步骤1: 矩阵乘法
├─ Wx = W @ x         → matmul.Wx (Wx_)
└─ Rh = R @ h         → matmul.Rh (Rh_)

步骤2: 更新门 (update gate)
├─ z_pre = Wx_z + Rh_z + bx_z + br_z   → gate.z_pre (z_pre_)
└─ z = sigmoid(z_pre)                   → gate.z_out (z_out_) [UINT]

步骤3: 重置门 (reset gate)
├─ r_pre = Wx_r + Rh_r + bx_r + br_r   → gate.r_pre (r_pre_)
└─ r = sigmoid(r_pre)                   → gate.r_out (r_out_) [UINT]

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

## 配置文件结构

```json
{
  "description": "配置文件描述",
  "comment": "注释说明",
  
  "default_bitwidth": { ... },
  "operator_config": { ... },
  "default_config": { ... }
}
```

### 1. default_bitwidth（默认位宽）

```json
"default_bitwidth": {
  "weight": 8,
  "activation": 8
}
```

### 2. operator_config（算子配置）

每个算子的配置格式：

```json
"算子名称": {
  "bitwidth": 8,            // 位宽（8, 16, 32）
  "is_symmetric": true,     // 是否对称量化
  "comment": "说明"         // 注释（可选）
}
```

### 3. 算子列表

#### 输入类

| 算子名 | 说明 | 推荐 is_symmetric |
|--------|------|-------------------|
| `input.x` | 输入序列 x | `false` |
| `input.h` | 隐藏状态 h | `false` |

#### 权重类

| 算子名 | 说明 | 推荐 is_symmetric |
|--------|------|-------------------|
| `weight.W` | 输入权重 W | `true` |
| `weight.R` | 循环权重 R | `true` |
| `weight.bx` | 输入偏置 bx | `true` |
| `weight.br` | 循环偏置 br | `true` |

#### 矩阵乘法类

| 算子名 | 说明 | 推荐 is_symmetric |
|--------|------|-------------------|
| `matmul.Wx` | W @ x 结果 | `false` |
| `matmul.Rh` | R @ h 结果 | `false` |

#### 门控类

| 算子名 | 说明 | C++ 类型 | 推荐 is_symmetric |
|--------|------|----------|-------------------|
| `gate.z_pre` | 更新门 sigmoid 前 | INT | `false` |
| `gate.z_out` | 更新门 sigmoid 后 [0,1] | **UINT** | `false` |
| `gate.r_pre` | 重置门 sigmoid 前 | INT | `false` |
| `gate.r_out` | 重置门 sigmoid 后 [0,1] | **UINT** | `false` |
| `gate.g_pre` | 候选门 tanh 前 | INT | `false` |
| `gate.g_out` | 候选门 tanh 后 [-1,1] | INT | `false` |

#### 运算类

| 算子名 | 说明 | 推荐 is_symmetric |
|--------|------|-------------------|
| `op.Rh_add_br` | Rh + br 加法 | `false` |
| `op.rRh` | r * Rh 元素乘法 | `false` |
| `op.one_minus_update` | 1 - z 减法 | `false` |
| `op.old_contrib` | z * h 旧状态贡献 | `false` |
| `op.new_contrib` | (1-z) * g 新状态贡献 | `false` |

---

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

---

## 配置示例

### 示例 1: 全 8 位配置（默认）

```json
{
  "operator_config": {
    "input.x": { "bitwidth": 8, "is_symmetric": false },
    "weight.W": { "bitwidth": 8, "is_symmetric": true },
    "gate.z_out": { "bitwidth": 8, "is_symmetric": false },
    "..."
  }
}
```

### 示例 2: 混合精度配置

```json
{
  "operator_config": {
    "weight.W": { "bitwidth": 16, "is_symmetric": true },
    "weight.R": { "bitwidth": 16, "is_symmetric": true },
    "matmul.Wx": { "bitwidth": 16, "is_symmetric": false },
    "matmul.Rh": { "bitwidth": 16, "is_symmetric": false },
    "gate.z_out": { "bitwidth": 8, "is_symmetric": false },
    "..."
  }
}
```

---

## 量化公式参考

### 量化与反量化

```
量化：   int_value = round(float_value / scale) + zero_point
反量化： float_value = scale × (int_value - zero_point)
```

### 对称 vs 非对称

| 类型 | zero_point | 优点 | 缺点 |
|------|------------|------|------|
| 对称 | `zp = 0` | 计算简单高效 | 对非对称分布浪费量化范围 |
| 非对称 | `zp ≠ 0` | 充分利用量化范围 | 需要额外处理零点偏移 |

---

## 注意事项

1. **sigmoid 输出**（`gate.z_out`, `gate.r_out`）：
   - C++ 自动使用 UINT 类型
   - 建议 `is_symmetric: false`（范围 [0,1] 不对称）

2. **tanh 输出**（`gate.g_out`）：
   - 范围 [-1,1] 理论上对称
   - 但实际分布可能有偏移，可根据实际情况选择

3. **权重**：
   - 建议 `is_symmetric: true`
   - 权重分布通常接近对称，无需存储零点

4. **位宽配置需在 `finalize_calibration()` 之前加载**

---

## 代码结构

```
量化配置相关文件：
├── include/quantize_bitwidth_config.hpp  # C++ 枚举和配置结构定义
├── pytorch/lib/gru_interface_binding.cpp # Python-C++ 绑定（类型转换逻辑）
├── pytorch/custom_gru.py                 # Python 配置加载函数
└── pytorch/config/
    ├── gru_quant_bitwidth_config.json    # JSON 配置文件
    └── README.md                         # 本文档
```
