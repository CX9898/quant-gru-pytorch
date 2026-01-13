# GRU 量化位宽配置文件说明

本文档介绍如何配置 `gru_quant_bitwidth_config.json` 文件来自定义 GRU 量化的位宽设置。

---

## 设计原则

### 1. 配置字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bitwidth` | int | 8 | 位宽 (1-32) |
| `is_symmetric` | bool | true | 是否对称量化 (true: zp=0, false: zp≠0) |
| `is_unsigned` | bool | false | 是否无符号量化 (false: INT, true: UINT) |

### 2. is_unsigned 设计原则

**只标记例外情况**：默认所有算子使用 INT（有符号），只有特殊算子需要显式标记 `is_unsigned: true`

```json
// 常见情况（INT）：无需配置 is_unsigned
"gate.g_out": { "bitwidth": 8, "is_symmetric": false }

// 特殊情况（UINT）：显式标记
"gate.z_out": { "bitwidth": 8, "is_symmetric": false, "is_unsigned": true }
```

**UINT 适用场景**：
- Sigmoid 输出 (`z_out`, `r_out`)：范围 [0, 1]，使用 UINT 可充分利用量化范围

### 3. is_symmetric 与 is_unsigned 解耦

这两个配置**独立控制**不同方面：

| 配置 | 控制内容 | 说明 |
|------|----------|------|
| `is_symmetric` | zero_point 计算 | true: zp=0, false: zp≠0 |
| `is_unsigned` | 量化范围类型 | false: INT, true: UINT |

**组合示例**（8bit）：

| is_symmetric | is_unsigned | 量化范围 | 典型场景 |
|--------------|-------------|----------|----------|
| true | false | INT8: [-128, 127], zp=0 | 权重 |
| false | false | INT8: [-128, 127], zp≠0 | 一般激活值 |
| false | true | UINT8: [0, 255], zp≠0 | Sigmoid 输出 |
| true | true | UINT8: [0, 255], zp=0 | 特殊场景 |

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
├─ Rh_br = Rh_g + br_g                 → 使用 matmul.Rh 的量化参数（已融合）
├─ rRh = r * Rh_br                     → op.rRh (rRh_)
├─ g_pre = Wx_g + bx_g + rRh           → gate.g_pre (g_pre_)
└─ g = tanh(g_pre)                     → gate.g_out (g_out_)

步骤5: 最终输出
├─ one_minus_z = 1 - z                 → 复用 gate.z_out 的 scale
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
  
  "GRU_config": {
    "default_config": { ... },
    "operator_config": { ... }
  }
}
```

### 1. default_config（全局配置）

```json
"default_config": {
  "disable_quantization": false
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `disable_quantization` | bool | `false` | 是否禁用量化。设为 `true` 时忽略所有量化配置，使用浮点计算 |

### 2. operator_config（算子配置）

每个算子的配置格式：

```json
"算子名称": {
  "bitwidth": 8,            // 位宽（1-32）
  "is_symmetric": true,     // 是否对称量化
  "is_unsigned": false,     // 是否无符号量化（可选，默认 false）
  "comment": "说明"         // 注释（可选）
}
```

### 3. 算子列表

#### 输入类

| 算子名 | 说明 | 推荐配置 |
|--------|------|----------|
| `input.x` | 输入序列 x | `is_symmetric: false` |

#### 输出类

| 算子名 | 说明 | 推荐配置 |
|--------|------|----------|
| `output.h` | 隐藏状态输出 h | `is_symmetric: false` |

#### 权重类（per-channel 量化）

| 算子名 | 说明 | 推荐配置 |
|--------|------|----------|
| `weight.W` | 输入权重 W | `is_symmetric: true` |
| `weight.R` | 循环权重 R | `is_symmetric: true` |
| `weight.bw` | 输入偏置 bw | `is_symmetric: true` |
| `weight.br` | 循环偏置 br | `is_symmetric: true` |

#### 矩阵乘法类

| 算子名 | 说明 | 推荐配置 |
|--------|------|----------|
| `matmul.Wx` | W @ x 结果 | `is_symmetric: false` |
| `matmul.Rh` | R @ h 结果 | `is_symmetric: false` |

#### 门控类

| 算子名 | 说明 | 量化类型 | 推荐配置 |
|--------|------|----------|----------|
| `gate.z_pre` | 更新门 sigmoid 前 | INT | `is_symmetric: false` |
| `gate.z_out` | 更新门 sigmoid 后 [0,1] | **UINT** | `is_symmetric: false, is_unsigned: true` |
| `gate.r_pre` | 重置门 sigmoid 前 | INT | `is_symmetric: false` |
| `gate.r_out` | 重置门 sigmoid 后 [0,1] | **UINT** | `is_symmetric: false, is_unsigned: true` |
| `gate.g_pre` | 候选门 tanh 前 | INT | `is_symmetric: false` |
| `gate.g_out` | 候选门 tanh 后 [-1,1] | INT | `is_symmetric: false` |

#### 运算类

| 算子名 | 说明 | 推荐配置 |
|--------|------|----------|
| `op.rRh` | r * Rh 元素乘法 | `is_symmetric: false` |
| `op.old_contrib` | z * h 旧状态贡献 | `is_symmetric: false` |
| `op.new_contrib` | (1-z) * g 新状态贡献 | `is_symmetric: false` |

---

## 使用方法

### Python 端

```python
from quant_gru import QuantGRU

# 1. 创建 GRU 并加载配置
gru = QuantGRU(input_size=64, hidden_size=128)
gru.load_bitwidth_config("config/gru_quant_bitwidth_config.json")

# 2. 校准
gru.calibrating = True
for batch in calibration_loader:
    gru(batch)
gru.calibrating = False
gru.finalize_calibration()

# 3. 推理
gru.use_quantization = True
output, h_n = gru(input_data)
```

**不使用 JSON 配置时**：

```python
gru = QuantGRU(input_size=64, hidden_size=128)

# 设置位宽（可选，默认全部 8bit 对称量化）
gru.set_all_bitwidth(8)                # 全部 8bit 对称量化
# gru.set_all_bitwidth(16, False)      # 全部 16bit 非对称量化

# 校准
gru.calibrating = True
for batch in calibration_loader:
    gru(batch)
gru.calibrating = False
gru.finalize_calibration()

# 开启量化并推理
gru.use_quantization = True
output, h_n = gru(input_data)
```

### 导出/导入量化参数

```python
# 导出量化参数
gru.export_quant_params("quant_params.json")

# 导入量化参数（无需重新校准）
gru2 = QuantGRU(input_size=64, hidden_size=128)
gru2.load_state_dict(gru.state_dict(), strict=False)  # 加载权重
gru2.load_quant_params("quant_params.json")           # 加载量化参数

# 调整单个算子配置
gru.adjust_quant_config("z_out", bitwidth=14)  # 修改位宽，自动调整 scale

# 获取算子配置
config = gru.get_quant_config("z_out")
print(config)  # {'bitwidth': 14, 'is_symmetric': False, 'exp2_inv': ..., ...}
```

---

## 配置示例

### 示例 1: 完整配置（所有算子）

```json
{
  "GRU_config": {
    "default_config": { "disable_quantization": false },
    "operator_config": {
      "input.x":       { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "output.h":      { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "weight.W":      { "bitwidth": 8, "is_symmetric": true,  "is_unsigned": false },
      "weight.R":      { "bitwidth": 8, "is_symmetric": true,  "is_unsigned": false },
      "weight.bw":     { "bitwidth": 8, "is_symmetric": true,  "is_unsigned": false },
      "weight.br":     { "bitwidth": 8, "is_symmetric": true,  "is_unsigned": false },
      "matmul.Wx":     { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "matmul.Rh":     { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "gate.z_pre":    { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "gate.z_out":    { "bitwidth": 8, "is_symmetric": false, "is_unsigned": true },
      "gate.r_pre":    { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "gate.r_out":    { "bitwidth": 8, "is_symmetric": false, "is_unsigned": true },
      "gate.g_pre":    { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "gate.g_out":    { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "op.rRh":        { "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "op.old_contrib":{ "bitwidth": 8, "is_symmetric": false, "is_unsigned": false },
      "op.new_contrib":{ "bitwidth": 8, "is_symmetric": false, "is_unsigned": false }
    }
  }
}
```

### 示例 2: 简洁配置（只标记例外）

```json
{
  "GRU_config": {
    "operator_config": {
      "gate.z_out": { "bitwidth": 8, "is_symmetric": false, "is_unsigned": true },
      "gate.r_out": { "bitwidth": 8, "is_symmetric": false, "is_unsigned": true }
    }
  }
}
```

> **说明**：未配置的算子使用 C++ 默认值（8bit INT）

### 示例 3: 混合精度配置

```json
{
  "GRU_config": {
    "operator_config": {
      "weight.W":   { "bitwidth": 16, "is_symmetric": true,  "is_unsigned": false },
      "weight.R":   { "bitwidth": 16, "is_symmetric": true,  "is_unsigned": false },
      "matmul.Wx":  { "bitwidth": 16, "is_symmetric": false, "is_unsigned": false },
      "matmul.Rh":  { "bitwidth": 16, "is_symmetric": false, "is_unsigned": false },
      "gate.z_out": { "bitwidth": 8,  "is_symmetric": false, "is_unsigned": true },
      "gate.r_out": { "bitwidth": 8,  "is_symmetric": false, "is_unsigned": true }
    }
  }
}
```

---

## 量化公式参考

### 量化与反量化

```
量化：   int_value = round(float_value / scale) + zero_point
反量化： float_value = scale × (int_value - zero_point)

其中： scale = 2^(-exp2_inv)
```

### 量化范围

| 类型 | is_unsigned | 8bit 范围 | 公式 |
|------|-------------|-----------|------|
| INT | false | [-128, 127] | [-2^(n-1), 2^(n-1)-1] |
| UINT | true | [0, 255] | [0, 2^n-1] |

### 对称 vs 非对称

| 类型 | zero_point | 优点 | 缺点 |
|------|------------|------|------|
| 对称 | `zp = 0` | 计算简单高效 | 对非对称分布浪费量化范围 |
| 非对称 | `zp ≠ 0` | 充分利用量化范围 | 需要额外处理零点偏移 |

---

## 注意事项

1. **sigmoid 输出**（`gate.z_out`, `gate.r_out`）：
   - 默认使用 UINT 类型（`is_unsigned: true`）
   - 建议 `is_symmetric: false`（范围 [0,1] 不对称）

2. **tanh 输出**（`gate.g_out`）：
   - 范围 [-1,1] 理论上对称
   - 使用 INT 类型（默认，无需配置 `is_unsigned`）

3. **权重**：
   - 建议 `is_symmetric: true`
   - 权重分布通常接近对称，无需存储零点

4. **位宽配置需在校准前加载**

5. **高级用法**：可手动调用 `finalize_calibration(verbose=True)` 查看校准详情

---
