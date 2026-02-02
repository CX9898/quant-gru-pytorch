# Haste 量化 GRU 纯定点计算流程

## 原浮点 GRU 门控计算

### 公式定义

| 门控 | 公式 | 说明 |
|------|------|------|
| 更新门 u | $u = \sigma(W_u x + R_u h + b_{wu} + b_{ru})$ | 控制保留多少旧状态 |
| 重置门 r | $r = \sigma(W_r x + R_r h + b_{wr} + b_{rr})$ | 控制遗忘多少旧状态 |
| 候选门 n | $n = \tanh(W_n x + b_{wn} + r \odot (R_n h + b_{rn}))$ | 生成候选新状态 |
| 新状态 h | $h_{new} = u \odot h_{old} + (1 - u) \odot n$ | 融合旧状态和候选状态 |

### 代码对应 (`gru_forward_gpu.cu`)

```cpp
// 更新门 u
const T u_pre = Wx[u_idx] + Rh[u_idx] + bw[bu_idx] + br[bu_idx];
const T u = sigmoid(u_pre);

// 重置门 r
const T r_pre = Wx[r_idx] + Rh[r_idx] + bw[br_idx] + br[br_idx];
const T r = sigmoid(r_pre);

// 候选门 n（注意：r 先乘以 Rh+br，再加 Wx 和 bw）
const T Rh_add_br_n = Rh[n_idx] + br[bn_idx];
const T n_pre = Wx[n_idx] + r * Rh_add_br_n + bw[bn_idx];
const T n = tanh(n_pre);

// 新隐藏状态
const T old_contrib = u * h[output_idx];
const T one_minus_u = 1.0 - u;
const T new_contrib = one_minus_u * n;
T cur_h_value = old_contrib + new_contrib;
```

> **haste 实现特点**：候选门 n 的计算中，重置门 r 仅作用于 $(R_n h + b_{rn})$ 部分，而不是整个 $(W_n x + R_n h)$。这与某些标准 GRU 实现略有不同。

## 量化核心规则说明

### 量化类型

| 类型 | 说明 |
|------|------|
| 对称量化 | zp = 0，仅需 scale = 2^(-shift)，无偏移 |
| 非对称量化 | zp ≠ 0，需同时使用 scale 和 zp，支持完整范围映射 |
| per-channel 量化 | 每个输出通道单独计算量化参数（对应权重矩阵每一行） |
| 动态范围更新 | 按时间步 EMA 更新 min/max：`min = 0.9×min_old + 0.1×min_cur` |

### 量化/反量化公式

**缩放因子 (scale)**

所有缩放因子均采用 2 的负 n 次方形式，便于用高效的移位操作代替乘除：

$$scale = 2^{-shift}$$

例如：`shift=7` 对应 `scale=1/128=0.0078125`

**零点 (zero point)**

零点 `zp` 用于非对称量化，表示浮点零值对应的量化整数值：

- 对称量化：`zp = 0`
- 非对称量化：`zp = round(-min / scale)`

**通用量化/反量化公式**：

- **量化**：$q = \text{round}(x / scale) + zp$
- **反量化**：$x = (q - zp) \times scale$
- 对称量化时 zp=0，简化为 $q = \text{round}(x / scale)$

**本项目采用的 2 的幂次形式**：

由于所有 scale 均为 $2^{-shift}$，公式可简化为：

- **量化**：$q = \text{round}(x \times 2^{shift}) + zp = \text{round}(x \ll shift) + zp$
- **反量化**：$x = (q - zp) \times 2^{-shift} = (q - zp) \gg shift$

> **计算优化**：乘以 $2^{shift}$ 等价于左移 `shift` 位，除以 $2^{shift}$ 等价于右移 `shift` 位。这种设计使得定点运算可以完全用整数移位实现，避免浮点乘除，大幅提升计算效率。

### 量化运算基础推导

在量化计算中，我们需要频繁进行 **Rescale（尺度转换）**、**量化加法** 和 **量化乘法**。下面给出这三种基本操作的推导。

#### 基础操作 1：Rescale（尺度转换）

**问题**：已知量化值 $q_x$（scale=$S_x$, zp=$Z_x$），如何转换到新的量化参数（scale=$S_y$, zp=$Z_y$）？

**推导**：

$$\text{浮点值}: x = (q_x - Z_x) \cdot S_x$$

$$\text{用新参数量化}: q_y = \frac{x}{S_y} + Z_y = \frac{(q_x - Z_x) \cdot S_x}{S_y} + Z_y$$

$$\boxed{q_y = \frac{S_x}{S_y}(q_x - Z_x) + Z_y}$$

#### 基础操作 2：量化加法

**问题**：计算 $z = x + y$，其中 $x$ 和 $y$ 有不同的量化参数，如何得到 $q_z$？

**推导**：

$$z = x + y = (q_x - Z_x) \cdot S_x + (q_y - Z_y) \cdot S_y$$

$$q_z = \frac{z}{S_z} + Z_z = \frac{(q_x - Z_x) \cdot S_x + (q_y - Z_y) \cdot S_y}{S_z} + Z_z$$

$$\boxed{q_z = \frac{S_x}{S_z}(q_x - Z_x) + \frac{S_y}{S_z}(q_y - Z_y) + Z_z}$$

> **关键洞察**：加法需要先将各项 rescale 到同一 scale，然后直接相加整数值。

#### 基础操作 3：量化乘法

**问题**：计算 $z = x \times y$，如何得到 $q_z$？

**推导**：

$$z = x \times y = (q_x - Z_x) \cdot S_x \times (q_y - Z_y) \cdot S_y = S_x \cdot S_y \cdot (q_x - Z_x)(q_y - Z_y)$$

$$q_z = \frac{z}{S_z} + Z_z = \frac{S_x \cdot S_y}{S_z}(q_x - Z_x)(q_y - Z_y) + Z_z$$

$$\boxed{q_z = \frac{S_x \cdot S_y}{S_z}(q_x - Z_x)(q_y - Z_y) + Z_z}$$

---

## 张量维度说明

| 变量 | 维度 | 说明 |
|------|------|------|
| x | [T×N, C] | T=时间步, N=批量, C=输入维度 |
| h | [(T+1)×N, H] | H=隐藏维度, 包含初始 h0 |
| W | [H×3, C] | 输入权重矩阵 |
| R | [H×3, H] | 隐藏状态权重矩阵 |
| bw, br | [H×3] | 偏置向量 |
| weight_ih_linear | [T×N, H×3] | W @ x + bw 结果 |
| weight_hh_linear | [N, H×3] | R @ h + br 结果（每时间步） |
| v | [T×N, H×4] | 中间激活值 [u, r, n, weight_hh_linear_n] |

### H×3 维度的门控分片

`H×3` 维度按以下方式切分为三个门的数据：

```
索引范围:  [0, H)      [H, 2H)     [2H, 3H)
门控类型:  u (更新门)   r (重置门)   n (候选门)
```

代码中的索引定义：
```cpp
const int u_idx = weight_idx + 0 * hidden_dim;  // [0, H)
const int r_idx = weight_idx + 1 * hidden_dim;  // [H, 2H)
const int n_idx = weight_idx + 2 * hidden_dim;  // [2H, 3H)
```

因此：
- `weight_ih_linear[u_idx]` = (W×x+bw)_u, `weight_ih_linear[r_idx]` = (W×x+bw)_r, `weight_ih_linear[n_idx]` = (W×x+bw)_n
- `weight_hh_linear[u_idx]` = (R×h+br)_u, `weight_hh_linear[r_idx]` = (R×h+br)_r, `weight_hh_linear[n_idx]` = (R×h+br)_n

---

## 量化推理流程

### 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         量化 GRU 前向计算流程                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Step A: 预计算（模型加载时一次性执行）                              │   │
│  │   W_sum[c] = Σ_k W[c,k]    // 权重行求和，用于零点补偿              │   │
│  │   R_sum[c] = Σ_k R[c,k]                                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Step B-1: 输入 Linear 变换 GEMM（所有时间步一次性计算）              │   │
│  │   q_gemm = cuBLAS SGEMM(W, x)    // 输出未 rescale 的 GEMM 结果     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 时间步循环 for t = 0 to T-1                                        │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Step B-2: 隐状态 Linear 变换 GEMM（每时间步计算）                │  │   │
│  │  │   q_gemm = cuBLAS SGEMM(R, h[t])  // 输出未 rescale 的 GEMM 结果│  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │                              ↓                                    │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Step C: 门控计算（PointwiseOperationsFP Kernel 逐元素）        │  │   │
│  │  │   // 融合执行 Bias + Rescale（减少全局内存读取）                │  │   │
│  │  │   weight_ih_linear = BiasRescale(gemm_ih, bw)              │  │   │
│  │  │   weight_hh_linear = BiasRescale(gemm_hh, br)               │  │   │
│  │  │   u = sigmoid(weight_ih_linear_u + weight_hh_linear_u)     │  │   │
│  │  │   r = sigmoid(weight_ih_linear_r + weight_hh_linear_r)     │  │   │
│  │  │   n = tanh(weight_ih_linear_n + r * weight_hh_linear_n)    │  │   │
│  │  │   h[t+1] = u * h[t] + (1-u) * n                            │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step A: 预计算权重和（用于零点补偿）

由于激活值 x 和 h 是非对称量化（zp≠0），GEMM 结果需要**零点补偿**。

#### 为什么需要零点补偿？

**背景**：权重 W 采用对称量化（zp=0），激活值 x 采用非对称量化（zp≠0）。整数 GEMM $q_W \cdot q_x$ 的结果并不直接等于 $W \cdot x$ 的量化值。

**推导**：设 $W = q_W \cdot S_W$，$x = (q_x - Z_x) \cdot S_x$，则：

$$Y[c] = \sum_k W[c,k] \cdot x[k] = S_W \cdot S_x \sum_k q_W[c,k] \cdot (q_x[k] - Z_x)$$

$$= S_W \cdot S_x \left( \underbrace{(q_W \cdot q_x)[c]}_{\text{整数GEMM}} - \underbrace{Z_x \sum_k q_W[c,k]}_{\text{零点补偿项}} \right)$$

**预计算公式**（模型加载时一次性计算）：

$$W\_sum[c] = \sum_{k} q_W[c, k], \quad R\_sum[c] = \sum_{k} q_R[c, k]$$

> **运行时**：零点补偿项 = $W\_sum[c] \times zp\_x$ 或 $R\_sum[c] \times zp\_h$

---

### Step B: Linear 层计算

**实现策略**：GEMM 和 Bias + Rescale 分离执行，减少全局内存读取。
- **GEMM 阶段**：使用 cuBLAS SGEMM 单独执行矩阵乘法，输出未 rescale 的原始 GEMM 结果
- **Bias + Rescale 阶段**：在 PointwiseOperationsFP kernel 中融合执行，减少全局内存访问

#### B-1. 输入 Linear 变换（所有时间步一次性计算）

**GEMM 阶段**（`ComputeLinearX`）：
- 执行 cuBLAS SGEMM：$q_{gemm}[c] = q_W[c,:] \cdot q_x$
- 输出未 rescale 的原始 GEMM 结果到 `tmp_weight_ih_linear_`（scale = $S_{W[c]} \cdot S_x$）

**Bias + Rescale 阶段**（在 PointwiseOperationsFP kernel 中融合执行）：
对每个输出通道 c 执行以下步骤：
1. 零点补偿：$q_{gemm\_corrected}[c] = q_{gemm}[c] - W\_sum[c] \cdot Z_x$
2. Bias rescale 到 GEMM 空间：$q_{bias\_in\_gemm}[c] = \frac{S_{bw[c]}}{S_{W[c]} \cdot S_x} q_{bw}[c]$
3. GEMM 和 Bias 在 GEMM 空间相加：$q_{combined}[c] = q_{gemm\_corrected}[c] + q_{bias\_in\_gemm}[c]$
4. 一起 rescale 到输出空间：$q_{weight\_ih\_linear}[c] = \frac{S_{W[c]} \cdot S_x}{S_{weight\_ih\_linear}} q_{combined}[c] + Z_{weight\_ih\_linear}$

**完整公式**：

$$q_{weight\_ih\_linear}[c] = \frac{S_{W[c]} \cdot S_x}{S_{weight\_ih\_linear}} \left( (q_{gemm}[c] - W\_sum[c] \cdot Z_x) + \frac{S_{bw[c]}}{S_{W[c]} \cdot S_x} q_{bw}[c] \right) + Z_{weight\_ih\_linear}$$

#### B-2. 隐状态 Linear 变换（每时间步计算）

**GEMM 阶段**（`ComputeLinearH`）：
- 执行 cuBLAS SGEMM：$q_{gemm}[c] = q_R[c,:] \cdot q_h$
- 输出未 rescale 的原始 GEMM 结果到 `tmp_weight_hh_linear_`（scale = $S_{R[c]} \cdot S_h$）

**Bias + Rescale 阶段**（在 PointwiseOperationsFP kernel 中融合执行）：
对每个输出通道 c 执行以下步骤：
1. 零点补偿：$q_{gemm\_corrected}[c] = q_{gemm}[c] - R\_sum[c] \cdot Z_h$
2. Bias rescale 到 GEMM 空间：$q_{bias\_in\_gemm}[c] = \frac{S_{br[c]}}{S_{R[c]} \cdot S_h} q_{br}[c]$
3. GEMM 和 Bias 在 GEMM 空间相加：$q_{combined}[c] = q_{gemm\_corrected}[c] + q_{bias\_in\_gemm}[c]$
4. 一起 rescale 到输出空间：$q_{weight\_hh\_linear}[c] = \frac{S_{R[c]} \cdot S_h}{S_{weight\_hh\_linear}} q_{combined}[c] + Z_{weight\_hh\_linear}$

**完整公式**：

$$q_{weight\_hh\_linear}[c] = \frac{S_{R[c]} \cdot S_h}{S_{weight\_hh\_linear}} \left( (q_{gemm}[c] - R\_sum[c] \cdot Z_h) + \frac{S_{br[c]}}{S_{R[c]} \cdot S_h} q_{br}[c] \right) + Z_{weight\_hh\_linear}$$

> **优化说明**：
> - **分离执行**：GEMM 单独执行，Bias + Rescale 在 PointwiseOperationsFP kernel 中融合执行，减少全局内存读取
> - **Scale 优化**：Bias 先 rescale 到 GEMM 空间，在更大 scale 空间（$S_W \cdot S_x$ 或 $S_R \cdot S_h$）中相加，然后一起 rescale 到输出空间。虽然 rescale 次数相同（2次），但在更大 scale 空间中相加，精度损失更小

---

### Step C: 门控计算（CUDA Kernel 逐元素）

#### C-1. 更新门 u（Update Gate）

**浮点公式**：`u = sigmoid(weight_ih_linear_u + weight_hh_linear_u)`

> 注意：由于 bias 已在 Linear 层融合，这里只需要两项相加。

**符号定义**：

| 符号 | 含义 |
|------|------|
| $q_A$ | 张量 A 的量化整数值 |
| $S_A$ | 张量 A 的 scale（= $2^{-shift_A}$） |
| $Z_A$ | 张量 A 的零点 (zero point) |

**量化加法推导**（以更新门为例）：

浮点运算：$update\_gate\_input = weight\_ih\_linear + weight\_hh\_linear$

量化表示：
- $weight\_ih\_linear = (q_{weight\_ih\_linear} - Z_{weight\_ih\_linear}) \cdot S_{weight\_ih\_linear}$
- $weight\_hh\_linear = (q_{weight\_hh\_linear} - Z_{weight\_hh\_linear}) \cdot S_{weight\_hh\_linear}$
- $update\_gate\_input = (q_{update\_gate\_input} - Z_{update\_gate\_input}) \cdot S_{update\_gate\_input}$

代入 $update\_gate\_input = weight\_ih\_linear + weight\_hh\_linear$：

$$(q_{update\_gate\_input} - Z_{update\_gate\_input}) \cdot S_{update\_gate\_input} = (q_{weight\_ih\_linear} - Z_{weight\_ih\_linear}) \cdot S_{weight\_ih\_linear} + (q_{weight\_hh\_linear} - Z_{weight\_hh\_linear}) \cdot S_{weight\_hh\_linear}$$

解出 $q_{update\_gate\_input}$：

$$q_{update\_gate\_input} = \frac{S_{weight\_ih\_linear}}{S_{update\_gate\_input}}(q_{weight\_ih\_linear} - Z_{weight\_ih\_linear}) + \frac{S_{weight\_hh\_linear}}{S_{update\_gate\_input}}(q_{weight\_hh\_linear} - Z_{weight\_hh\_linear}) + Z_{update\_gate\_input}$$

**定点实现**（使用移位代替除法）：

```cpp
ih_shifted = rshift_round(q_weight_ih_linear - zp_weight_ih_linear, shift_ih_to_gate_input);
hh_shifted = rshift_round(q_weight_hh_linear - zp_weight_hh_linear, shift_hh_to_gate_input);
q_update_gate_input = ih_shifted + hh_shifted + zp_update_gate_input;
```

**激活函数**（分段线性近似）：

```cpp
update_gate = piecewise_linear(update_gate_input, sigmoid_update_gate_lut_, 
                               bitwidth_config_.update_gate_input_, 
                               bitwidth_config_.update_gate_output_);
```

输入/输出位宽通过 `bitwidth_config_` 动态配置，支持任意位宽组合。

---

#### C-2. 重置门 r（Reset Gate）

**浮点公式**：`r = sigmoid(weight_ih_linear_r + weight_hh_linear_r)`

**量化公式**：与 u 门结构完全相同

$$q_{reset\_gate\_input} = \frac{S_{weight\_ih\_linear}}{S_{reset\_gate\_input}}(q_{weight\_ih\_linear} - Z_{weight\_ih\_linear}) + \frac{S_{weight\_hh\_linear}}{S_{reset\_gate\_input}}(q_{weight\_hh\_linear} - Z_{weight\_hh\_linear}) + Z_{reset\_gate\_input}$$

**定点实现**：

```cpp
ih_shifted = rshift_round(q_weight_ih_linear - zp_weight_ih_linear, shift_ih_to_gate_input);
hh_shifted = rshift_round(q_weight_hh_linear - zp_weight_hh_linear, shift_hh_to_gate_input);
q_reset_gate_input = ih_shifted + hh_shifted + zp_reset_gate_input;

reset_gate = piecewise_linear(reset_gate_input, sigmoid_reset_gate_lut_, ...);
```

---

#### C-3. 候选门 n（New Gate）

**浮点公式**：`n = tanh(weight_ih_linear_n + r × weight_hh_linear_n)`

> **注意**：由于 Linear 层已融合 bias，`weight_hh_linear_n` 已包含 $R_n h + br_n$，不再需要单独的 `Rh_add_br` 中间量。

**优化策略**：消除 `mul_reset_hidden` 中间量化空间，乘法结果直接 rescale 到 `new_gate_input` 空间。

**量化公式**：

$$q_{new\_gate\_input} = \frac{S_{weight\_ih\_linear}}{S_{new\_gate\_input}}(q_{weight\_ih\_linear} - Z_{weight\_ih\_linear}) + \frac{S_{reset\_gate\_output} \cdot S_{weight\_hh\_linear}}{S_{new\_gate\_input}} (q_{reset\_gate\_output} - Z_{reset\_gate\_output})(q_{weight\_hh\_linear} - Z_{weight\_hh\_linear}) + Z_{new\_gate\_input}$$

**计算步骤**：

1. **计算 r × weight_hh_linear_n**：乘积 scale = $S_{reset\_gate\_output} \cdot S_{weight\_hh\_linear}$
   - $q_{reset\_hidden\_mul} = (q_{reset\_gate\_output} - Z_{reset\_gate\_output})(q_{weight\_hh\_linear} - Z_{weight\_hh\_linear})$

2. **直接 rescale 到 new_gate_input 空间**：
   - $q_{rh\_rescaled} = \frac{S_{reset\_gate\_output} \cdot S_{weight\_hh\_linear}}{S_{new\_gate\_input}} q_{reset\_hidden\_mul}$

3. **weight_ih_linear_n rescale 到 new_gate_input 空间**：
   - $q_{ih\_rescaled} = \frac{S_{weight\_ih\_linear}}{S_{new\_gate\_input}}(q_{weight\_ih\_linear} - Z_{weight\_ih\_linear})$

4. **相加并应用零点**：
   - $q_{new\_gate\_input} = q_{ih\_rescaled} + q_{rh\_rescaled} + Z_{new\_gate\_input}$

**定点实现**：

```cpp
// 计算 r × weight_hh_linear_n，直接 rescale 到 new_gate_input
r_diff = q_reset_gate_output - zp_reset_gate_output;
hh_diff = q_weight_hh_linear - zp_weight_hh_linear;
product = r_diff * hh_diff;
q_rh = rshift_round(product, shift_reset_mul_hh_to_new_gate_input);

// weight_ih_linear rescale 到 new_gate_input
ih_shifted = rshift_round(q_weight_ih_linear - zp_weight_ih_linear, shift_ih_to_new_gate_input);

// 相加
q_new_gate_input = ih_shifted + q_rh + zp_new_gate_input;

new_gate = piecewise_linear(new_gate_input, tanh_new_gate_lut_, ...);
```

> **优化说明**：消除 `mul_reset_hidden` 中间空间，减少一次 rescale 操作，简化参数管理。

---

#### C-4. 隐藏状态更新

**浮点公式**：`h_new = u × h_old + (1 - u) × n`

**优化策略**：统一 scale 空间，先将 `new_gate` 对齐到 `h` scale，使两个乘积的 scale 统一为 $S_{update\_gate\_output} \cdot S_h$，在统一 scale 下直接相加，最后一起 rescale 到 `h` scale。

**计算步骤**：

**(1) 对齐 scale：将 new_gate 从 new_gate_output scale 对齐到 h scale**

$$q_{new\_gate\_aligned\_to\_h} = \frac{S_{new\_gate\_output}}{S_h}(q_{new\_gate\_output} - Z_{new\_gate\_output}) + Z_h$$

**(2) 统一 scale 下计算两个乘积**（都在 $S_{update\_gate\_output} \cdot S_h$ scale）

- **old_contribution = u × h_old**：
  $$q_{old\_contribution\_mul} = (q_{update\_gate\_output} - Z_{update\_gate\_output})(q_h - Z_h)$$

- **new_contribution = (1 - u) × new_gate_aligned**：
  - 计算 $(1 - u)$：$q_{one\_in\_update\_gate\_output} = \text{round}(1.0 / S_{update\_gate\_output}) + Z_{update\_gate\_output}$
  - $q_{one\_minus\_u} = q_{one\_in\_update\_gate\_output} - q_{update\_gate\_output}$
  - $$q_{new\_contribution\_mul} = (q_{one\_minus\_u})(q_{new\_gate\_aligned\_to\_h} - Z_h)$$

**(3) 统一 scale 下相加**

$$q_{combined} = q_{old\_contribution\_mul} + q_{new\_contribution\_mul} \quad \text{(scale = } S_{update\_gate\_output} \cdot S_h \text{)}$$

**(4) 一起 rescale 到 h scale**

$$q_{h\_new} = \frac{S_{update\_gate\_output} \cdot S_h}{S_h} \cdot q_{combined} + Z_h = S_{update\_gate\_output} \cdot q_{combined} + Z_h$$

**定点实现**：

```cpp
// 步骤1: 将new_gate对齐到h scale
n_diff_from_zp = q_new_gate_output - zp_new_gate_output;
q_new_gate_aligned_to_h = rshift_round(n_diff_from_zp, shift_new_gate_output_to_h) + zp_h;

// 步骤2: 计算old_contribution = u * h_old
u_diff = q_update_gate_output - zp_update_gate_output;
h_diff = q_h_old - zp_h;
q_old_contribution_mul = u_diff * h_diff;  // scale = S_u * S_h

// 步骤3: 计算new_contribution = (1-u) * new_gate_aligned
one_minus_u = quant_one_in_update_gate_scale_ - q_update_gate_output;
n_diff_aligned = q_new_gate_aligned_to_h - zp_h;
q_new_contribution_mul = one_minus_u * n_diff_aligned;  // scale = S_u * S_h

// 步骤4: 统一scale下相加
q_combined = q_old_contribution_mul + q_new_contribution_mul;

// 步骤5: 一起rescale到h scale
q_h_new = rshift_round(q_combined, shift_update_gate_output) + zp_h;
```

> **优化说明**：
> - 消除 `mul_old_contribution` 和 `mul_new_contribution` 中间空间
> - 减少 1 次 rescale（从 3 次减少到 2 次：new_gate 对齐 + 最终 rescale）
> - 在更大 scale 空间（$S_{update\_gate\_output} \cdot S_h$）中相加，精度损失更小

---

## 附录

### A. 各参数量化配置

| 参数 | 数据类型 | 量化类型 | scale | zp | 计算公式 | 使用位置 |
|------|----------|----------|-------|-----|----------|----------|
| 输入 x | INT | 非对称 + 动态 | `shift_x_` | `zp_x_` | - | GRU 输入 |
| 隐藏状态 h | INT | 非对称 + 动态 | `shift_h_` | `zp_h_` | - | GRU 输出/下一步输入 |
| 权重 W | INT | 对称 + per-channel | `shift_W_[i]` | 0 | - | GEMM: W×x |
| 权重 R | INT | 对称 + per-channel | `shift_R_[i]` | 0 | - | GEMM: R×h |
| 偏置 bw | INT | 对称 + per-channel | `shift_bw_[i]` | 0 | - | Linear 融合 |
| 偏置 br | INT | 对称 + per-channel | `shift_br_[i]` | 0 | - | Linear 融合 |
| weight_ih_linear | INT | 非对称 | `shift_weight_ih_linear_` | `zp_weight_ih_linear_` | W × x + bw | u/r/n 门输入 |
| weight_hh_linear | INT | 非对称 | `shift_weight_hh_linear_` | `zp_weight_hh_linear_` | R × h + br | u/r/n 门输入 |
| update_gate_input | INT | 非对称 | `shift_update_gate_input_` | `zp_update_gate_input_` | ih_u + hh_u | sigmoid 输入 |
| update_gate_output | **UINT** | 非对称 | `shift_update_gate_output_` | `zp_update_gate_output_` | sigmoid(update_gate_input) | h 更新门控 |
| reset_gate_input | INT | 非对称 | `shift_reset_gate_input_` | `zp_reset_gate_input_` | ih_r + hh_r | sigmoid 输入 |
| reset_gate_output | **UINT** | 非对称 | `shift_reset_gate_output_` | `zp_reset_gate_output_` | sigmoid(reset_gate_input) | n 门乘法输入 |
| new_gate_input | INT | 非对称 | `shift_new_gate_input_` | `zp_new_gate_input_` | ih_n + r×hh_n（直接 rescale） | tanh 输入 |
| new_gate_output | INT | 对称 | `shift_new_gate_output_` | 0 | tanh(new_gate_input) | h 更新候选值 |

> **说明**：
> - `update_gate_output` 和 `reset_gate_output` 使用 **UINT**（无符号整数），因为 sigmoid 输出范围为 [0, 1]，使用无符号类型可以充分利用所有 bit 位。其他参数使用 **INT**（有符号整数）。
> - **优化**：已消除 `mul_reset_hidden`、`mul_old_contribution`、`mul_new_contribution` 三个中间量化空间，乘法结果直接 rescale 到目标空间，减少 rescale 操作次数。

### B. 量化参数详细说明

本节对每个量化参数进行详细介绍，包括含义、数据类型和典型取值。基础概念（scale、zp、量化公式）请参见前文「量化核心规则说明」章节。

#### B.1 输入/输出参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `shift_x_` | int8_t | 标量 | 输入 x 的缩放因子指数 |
| `zp_x_` | int32_t | 标量 | 输入 x 的零点 |
| `shift_h_` | int8_t | 标量 | 隐藏状态 h 的缩放因子指数 |
| `zp_h_` | int32_t | 标量 | 隐藏状态 h 的零点 |

**输入 x (`input.x`)**
- **数据流位置**：GRU 输入层
- **浮点范围**：通常为 `[-1.0, 1.0]`
- **典型配置**：INT8 对称量化，`shift=7`，scale=1/128
- **特殊说明**：支持动态范围更新（EMA）

**隐藏状态 h (`output.h`)**
- **数据流位置**：GRU 输出层 / 下一时间步输入
- **浮点范围**：由 tanh 约束，理论范围 `[-1.0, 1.0]`
- **典型配置**：INT8 对称量化，`shift=7`，scale=1/128
- **特殊说明**：h 既是输入也是输出，共用量化参数

#### B.2 权重参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `shift_W_` | vector\<int8_t\> | per-channel | 输入权重 W 的缩放因子，size = hidden×3 |
| `shift_R_` | vector\<int8_t\> | per-channel | 循环权重 R 的缩放因子，size = hidden×3 |

**输入权重 W (`weight.W`)**
- **维度**：`[hidden×3, input_size]`，按 u/r/n 门顺序排列
- **量化方式**：对称量化 + per-channel（每行独立 scale）
- **典型配置**：INT8，`shift` 通常在 8~12 之间
- **特殊说明**：per-channel 量化可保留更多精度

**循环权重 R (`weight.R`)**
- **维度**：`[hidden×3, hidden_size]`
- **量化方式**：对称量化 + per-channel
- **典型配置**：与 W 类似

#### B.3 偏置参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `shift_bw_` | vector\<int8_t\> | per-channel | 输入偏置缩放因子，size = hidden×3 |
| `shift_br_` | vector\<int8_t\> | per-channel | 循环偏置缩放因子，size = hidden×3 |

**输入偏置 bw (`weight.bw`)** 和 **循环偏置 br (`weight.br`)**
- **维度**：`[hidden×3]`
- **量化方式**：对称量化 + per-channel
- **特殊说明**：偏置零点恒为 0，仅需 scale 参数；偏置在 Linear 层与 GEMM 结果融合

#### B.4 Linear 输出参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `shift_weight_ih_linear_` | int8_t | 标量 | W×x+bw 融合输出的缩放因子 |
| `zp_weight_ih_linear_` | int32_t | 标量 | W×x+bw 的零点 |
| `shift_weight_hh_linear_` | int8_t | 标量 | R×h+br 融合输出的缩放因子 |
| `zp_weight_hh_linear_` | int32_t | 标量 | R×h+br 的零点 |

**weight_ih_linear (`linear.weight_ih`)**
- **数据流位置**：Linear 层输出，送入门控计算
- **计算公式**：`W×x + bw`（GEMM + bias 融合）
- **浮点范围**：取决于权重、偏置和输入的分布
- **典型配置**：INT8，`shift=6`，scale=1/64
- **零点补偿**：GEMM 部分需要减去 `W_sum × zp_x`

**weight_hh_linear (`linear.weight_hh`)**
- **数据流位置**：Linear 层输出，每时间步重新计算
- **计算公式**：`R×h + br`（GEMM + bias 融合）
- **浮点范围**：通常比 weight_ih_linear 范围小（h 已被约束在 [-1,1]）
- **典型配置**：INT8，`shift=8`，scale=1/256
- **零点补偿**：GEMM 部分需要减去 `R_sum × zp_h`

#### B.5 门激活函数参数

##### 预激活参数（激活函数输入）

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `shift_update_gate_input_` | int8_t | 标量 | update gate sigmoid 输入的缩放因子 |
| `zp_update_gate_input_` | int32_t | 标量 | update gate sigmoid 输入的零点 |
| `shift_reset_gate_input_` | int8_t | 标量 | reset gate sigmoid 输入的缩放因子 |
| `zp_reset_gate_input_` | int32_t | 标量 | reset gate sigmoid 输入的零点 |
| `shift_new_gate_input_` | int8_t | 标量 | new gate tanh 输入的缩放因子 |
| `zp_new_gate_input_` | int32_t | 标量 | new gate tanh 输入的零点 |

**update_gate_input (`gate.update_input`)** = `weight_ih_linear_u + weight_hh_linear_u`
- **浮点范围**：通常为 `[-4.0, 4.0]`
- **典型配置**：INT8，`shift=6`，scale=1/64，覆盖 [-2, 2]
- **作用**：作为 sigmoid LUT 的输入索引

**reset_gate_input (`gate.reset_input`)** = `weight_ih_linear_r + weight_hh_linear_r`
- **配置与 update_gate_input 相同**

**new_gate_input (`gate.new_input`)** = `weight_ih_linear_n + mul_reset_hidden`
- **浮点范围**：通常为 `[-4.0, 4.0]`
- **典型配置**：INT8，`shift=6`，scale=1/64
- **作用**：作为 tanh LUT 的输入索引

##### 后激活参数（激活函数输出）

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `shift_update_gate_output_` | int8_t | 标量 | update gate sigmoid 输出的缩放因子 |
| `zp_update_gate_output_` | int32_t | 标量 | update gate sigmoid 输出的零点 |
| `shift_reset_gate_output_` | int8_t | 标量 | reset gate sigmoid 输出的缩放因子 |
| `zp_reset_gate_output_` | int32_t | 标量 | reset gate sigmoid 输出的零点 |
| `shift_new_gate_output_` | int8_t | 标量 | new gate tanh 输出的缩放因子 |
| `zp_new_gate_output_` | int32_t | 标量 | new gate tanh 输出的零点（对称量化时为 0） |

**update_gate_output (`gate.update_output`)** = `sigmoid(update_gate_input)`
- **浮点范围**：`[0.0, 1.0]`（sigmoid 输出恒正）
- **默认配置**：UINT8 非对称量化（`update_gate_output_symmetric_ = false`）
- **特殊说明**：由于 sigmoid 输出范围 [0,1]，使用 UINT 可充分利用所有 bit 位表示正值范围

**reset_gate_output (`gate.reset_output`)** = `sigmoid(reset_gate_input)`
- **配置与 update_gate_output 相同**

**new_gate_output (`gate.new_output`)** = `tanh(new_gate_input)`
- **浮点范围**：`[-1.0, 1.0]`（tanh 输出）
- **典型配置**：INT8 对称量化，`shift=7`，scale=1/128
- **特殊说明**：对称量化，zp=0

#### B.6 隐藏状态更新参数

**优化说明**：已消除 `mul_reset_hidden`、`mul_old_contribution`、`mul_new_contribution` 三个中间量化空间。乘法结果直接 rescale 到目标空间，减少 rescale 操作次数。

**隐藏状态更新流程**：

1. **new_gate 对齐到 h scale**：$q_{new\_gate\_aligned\_to\_h} = \frac{S_{new\_gate\_output}}{S_h}(q_{new\_gate\_output} - Z_{new\_gate\_output}) + Z_h$

2. **统一 scale 下计算**（两个乘积都在 $S_{update\_gate\_output} \cdot S_h$ scale）：
   - $q_{old\_contribution\_mul} = (q_{update\_gate\_output} - Z_{update\_gate\_output})(q_h - Z_h)$
   - $q_{new\_contribution\_mul} = (q_{one\_in\_update\_gate\_output} - q_{update\_gate\_output})(q_{new\_gate\_aligned\_to\_h} - Z_h)$

3. **统一 scale 下相加**：$q_{combined} = q_{old\_contribution\_mul} + q_{new\_contribution\_mul}$

4. **一起 rescale 到 h**：$q_{h\_new} = S_{update\_gate\_output} \cdot q_{combined} + Z_h$

**关键参数**：
- `q_{one\_in\_update\_gate\_output}`：常数 1 在 update_gate_output 量化空间的值
  - 计算公式：$q_{one\_in\_update\_gate\_output} = \text{round}(1.0 / S_{update\_gate\_output}) + Z_{update\_gate\_output}$

#### B.8 重缩放参数（Device 端）

重缩放参数由 `GRUQuantParams` 预计算得出，存储在 `GateQuantParams` 和 `LinearQuantParams` 结构体中，供 GPU Kernel 使用。

**命名约定**：

- `shift_A_to_B`：表示从 A 空间到 B 空间的移位量，= shift_A - shift_B

**示例**：
```cpp
// weight_ih_linear rescale 到 update_gate_input
// shift = shift_weight_ih_linear - shift_update_gate_input
shift_weight_ih_linear_to_update_gate_input_ = shift_weight_ih_linear_ - shift_update_gate_input_;
```

**主要重缩放参数**：

**Linear 层参数**（优化：Bias 先转到 GEMM 空间）：
| 参数 | 说明 | 计算公式 |
|------|------|----------|
| `shift_gemm_x_to_weight_ih_linear_[i]` | W×x GEMM 输出 rescale | `shift_W_[i] + shift_x_ - shift_weight_ih_linear_` |
| `shift_bw_to_gemm_x_[i]` | bw 转到 GEMM 空间 | `shift_bw_[i] - (shift_W_[i] + shift_x_)` |
| `shift_gemm_h_to_weight_hh_linear_[i]` | R×h GEMM 输出 rescale | `shift_R_[i] + shift_h_ - shift_weight_hh_linear_` |
| `shift_br_to_gemm_h_[i]` | br 转到 GEMM 空间 | `shift_br_[i] - (shift_R_[i] + shift_h_)` |

**门控计算参数**：
| 参数 | 说明 | 计算公式 |
|------|------|----------|
| `shift_weight_ih_linear_to_update_gate_input_` | ih 到 update_gate_input | `shift_weight_ih_linear_ - shift_update_gate_input_` |
| `shift_weight_hh_linear_to_update_gate_input_` | hh 到 update_gate_input | `shift_weight_hh_linear_ - shift_update_gate_input_` |
| `shift_weight_ih_linear_to_reset_gate_input_` | ih 到 reset_gate_input | `shift_weight_ih_linear_ - shift_reset_gate_input_` |
| `shift_weight_hh_linear_to_reset_gate_input_` | hh 到 reset_gate_input | `shift_weight_hh_linear_ - shift_reset_gate_input_` |
| `shift_weight_ih_linear_to_new_gate_input_` | ih 到 new_gate_input | `shift_weight_ih_linear_ - shift_new_gate_input_` |
| `shift_reset_mul_hh_to_new_gate_input_` | r×hh 到 new_gate_input（直接 rescale） | `(shift_reset_gate_output + shift_weight_hh_linear_) - shift_new_gate_input_` |
| `quant_one_in_update_gate_scale_` | 常数 1 的量化表示 | `round(1.0 × 2^shift_update_gate_output) + zp_update_gate_output` |

**隐藏状态更新参数**（优化：统一 scale 空间）：
| 参数 | 说明 | 计算公式 |
|------|------|----------|
| `shift_new_gate_output_to_h_` | new_gate_output 对齐到 h | `shift_new_gate_output_ - shift_h_` |
| `shift_update_gate_output` | 统一 scale 到 h（最终 rescale） | `shift_update_gate_output_` |

#### B.9 分段线性近似 LUT

激活函数采用**分段线性近似 (Piecewise Linear Approximation)** 实现，避免运行时计算 sigmoid/tanh：

| 参数 | 类型 | 说明 |
|------|------|------|
| `sigmoid_update_gate_lut_` | SigmoidLUT | update gate Sigmoid 分段线性参数 |
| `sigmoid_reset_gate_lut_` | SigmoidLUT | reset gate Sigmoid 分段线性参数 |
| `tanh_new_gate_lut_` | SigmoidLUT | new gate Tanh 分段线性参数 |

**工作原理**：将 sigmoid/tanh 函数划分为多个分段，每段用线性函数 $y = b \cdot x + c$ 近似。LUT 存储每段的斜率 `q_b`、截距预计算项 `term_c_precomputed` 和分段边界 `threshold`。

LUT 在 `finalize_calibration` 时根据输入输出量化参数生成，通过 `piecewise_linear()` 函数进行计算。

### C. 代码对应关系

| 计算步骤 | 代码位置 |
|----------|----------|
| 浮点 GRU 前向 | `gru_forward_gpu.cu::PointwiseOperations()` |
| 量化 GRU 前向 | `gru_forward_gpu_quant.cu::PointwiseOperationsQuant()` |
| update gate 计算 | `quantize_ops_helper.h::computeUpdateGate()` |
| reset gate 计算 | `quantize_ops_helper.h::computeResetGate()` |
| new gate 计算 | `quantize_ops_helper.h::computeNewGate()` |
| h 更新计算 | `quantize_ops_helper.h::computeHiddenState()` |
| Linear 层计算 | `gru_forward_gpu_quant.cu::ComputeLinearX/ComputeLinearH()` |
| 校准接口 | `gru_interface.cc::calibrateGruRanges()` |
| 参数计算 | `gru_interface.cc::calculateGRUQuantitativeParameters()` |
| 量化参数结构体定义 | `quantize_param_types.h::GRUQuantParams` |
| 门计算参数结构体定义 | `quantize_param_types.h::GateQuantParams` |
| Linear 层参数结构体定义 | `quantize_param_types.h::LinearQuantParamsGPU` |
| LUT 生成 | `quantize_ops_helper.h::generate_piecewise_linear_lut_to_params()` |

