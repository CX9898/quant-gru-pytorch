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
│  │ Step B: 输入 Linear 变换（所有时间步一次性计算）                    │   │
│  │   weight_ih_linear = quantizedGemmBiasFused(W, x, bw)             │   │
│  │   // GEMM + 零点补偿 + per-channel rescale + bias                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 时间步循环 for t = 0 to T-1                                        │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Step B: 隐状态 Linear 变换（每时间步计算）                    │  │   │
│  │  │   weight_hh_linear = quantizedGemmBiasFused(R, h[t], br)   │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │                              ↓                                    │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Step C: 门控计算（CUDA Kernel 逐元素）                       │  │   │
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

### Step B: Linear 层融合计算

Linear 层使用 `quantizedGemmBiasFused` kernel，将以下操作融合在一起：

```cpp
// 融合 kernel 对每个输出通道 c 执行：
gemm_result = q_W[c,:] @ q_x                                    // 1. 整数 GEMM
gemm_corrected = gemm_result - W_sum[c] * zp_x                  // 2. 零点补偿
gemm_rescaled = rshift_round(gemm_corrected, shift_gemm[c])     // 3. GEMM per-channel rescale
bias_rescaled = rshift_round(q_bw[c], shift_bw[c])              // 4. bias per-channel rescale
weight_ih_linear[c] = gemm_rescaled + bias_rescaled + zp_out    // 5. 加法 + 输出零点
```

**输入 Linear 变换**（所有时间步一次性计算）：

$$q_{weight\_ih\_linear}[c] = \frac{S_{W[c]} \cdot S_x}{S_{weight\_ih\_linear}} (q_{gemm}[c] - W\_sum[c] \cdot zp_x) + \frac{S_{bw[c]}}{S_{weight\_ih\_linear}} q_{bw}[c] + Z_{weight\_ih\_linear}$$

**隐状态 Linear 变换**（每时间步计算）：

$$q_{weight\_hh\_linear}[c] = \frac{S_{R[c]} \cdot S_h}{S_{weight\_hh\_linear}} (q_{gemm}[c] - R\_sum[c] \cdot zp_h) + \frac{S_{br[c]}}{S_{weight\_hh\_linear}} q_{br}[c] + Z_{weight\_hh\_linear}$$

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

**量化加法推导**：

设浮点运算 $y = x_1 + x_2$，其中：
- $x_1$ 的量化参数为 $(S_1, Z_1)$：$x_1 = (q_1 - Z_1) \cdot S_1$
- $x_2$ 的量化参数为 $(S_2, Z_2)$：$x_2 = (q_2 - Z_2) \cdot S_2$
- $y$ 的量化参数为 $(S_y, Z_y)$：$y = (q_y - Z_y) \cdot S_y$

代入 $y = x_1 + x_2$：

$$(q_y - Z_y) \cdot S_y = (q_1 - Z_1) \cdot S_1 + (q_2 - Z_2) \cdot S_2$$

解出 $q_y$：

$$q_y = \frac{S_1}{S_y}(q_1 - Z_1) + \frac{S_2}{S_y}(q_2 - Z_2) + Z_y$$

**应用到更新门**：$x_1 = weight\_ih\_linear$，$x_2 = weight\_hh\_linear$，$y = update\_gate\_input$

$$q_{update\_gate\_input} = \frac{S_{ih}}{S_{gate}}(q_{ih} - Z_{ih}) + \frac{S_{hh}}{S_{gate}}(q_{hh} - Z_{hh}) + Z_{gate}$$

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

$$q_{reset\_gate\_input} = \frac{S_{ih\_linear}}{S_{reset\_gate\_input}}(q_{ih\_linear} - Z_{ih\_linear}) + \frac{S_{hh\_linear}}{S_{reset\_gate\_input}}(q_{hh\_linear} - Z_{hh\_linear}) + Z_{reset\_gate\_input}$$

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

**(1) 计算 mul_reset_hidden = r × weight_hh_linear_n（量化乘法）**

根据量化乘法公式，两个量化值相乘后 rescale 到目标空间：

$$q_{mul\_reset\_hidden} = \frac{S_{reset\_gate} \cdot S_{hh\_linear}}{S_{mul\_reset\_hidden}} (q_{reset\_gate} - Z_{reset\_gate})(q_{hh\_linear} - Z_{hh\_linear}) + Z_{mul\_reset\_hidden}$$

**定点实现**：

```cpp
r_diff = q_reset_gate - zp_reset_gate_output;
hh_diff = q_weight_hh_linear - zp_weight_hh_linear;
product = r_diff * hh_diff;
q_mul_reset_hidden = rshift_round(product, shift_r_hh_to_mul_reset_hidden) + zp_mul_reset_hidden;
```

**(2) 计算 new_gate_input = weight_ih_linear_n + mul_reset_hidden（量化加法）**

$$q_{new\_gate\_input} = \frac{S_{ih\_linear}}{S_{new\_gate\_input}}(q_{ih\_linear} - Z_{ih\_linear}) + \frac{S_{mul\_reset\_hidden}}{S_{new\_gate\_input}}(q_{mul\_reset\_hidden} - Z_{mul\_reset\_hidden}) + Z_{new\_gate\_input}$$

**定点实现**：

```cpp
ih_shifted = rshift_round(q_weight_ih_linear - zp_weight_ih_linear, shift_ih_to_new_gate_input);
rh_shifted = rshift_round(q_mul_reset_hidden - zp_mul_reset_hidden, shift_mul_reset_hidden_to_new_gate_input);
q_new_gate_input = ih_shifted + rh_shifted + zp_new_gate_input;

new_gate = piecewise_linear(new_gate_input, tanh_new_gate_lut_, ...);
```

---

#### C-4. 隐藏状态更新

**浮点公式**：`h_new = u × h_old + (1 - u) × n`

该公式分解为四个子计算：

**(1) 计算 mul_old_contribution = u × h_old（量化乘法）**

$$q_{old} = \frac{S_{update\_gate} \cdot S_h}{S_{old}} (q_{update\_gate} - Z_{update\_gate})(q_h - Z_h) + Z_{old}$$

**定点实现**：

```cpp
u_diff = q_update_gate - zp_update_gate_output;
h_diff = q_h_old - zp_h;
product = u_diff * h_diff;
q_mul_old_contribution = rshift_round(product, shift_u_h_to_old_contribution) + zp_mul_old_contribution;
```

**(2) 计算 (1 - u)（优化技巧）**

将常数 1 预量化到 update_gate_output 的量化空间，使 $(1-u)$ 复用同一 scale：

$$q_{one} = \text{round}(1.0 / S_{update\_gate}) + Z_{update\_gate} = \text{round}(2^{shift}) + Z_{update\_gate}$$

$$q_{one\_minus\_u} = q_{one} - q_{update\_gate} + Z_{update\_gate}$$

**定点实现**：

```cpp
one_minus_u = quant_one_in_update_gate_scale_ - q_update_gate;
// (1-u) 复用 update_gate_output 的 scale，无需额外参数
```

**(3) 计算 mul_new_contribution = (1 - u) × n（量化乘法）**

$$q_{new} = \frac{S_{update\_gate} \cdot S_{new\_gate}}{S_{new}} (q_{one} - q_{update\_gate})(q_{new\_gate} - Z_{new\_gate}) + Z_{new}$$

**定点实现**：

```cpp
n_diff = q_new_gate - zp_new_gate_output;
product = one_minus_u * n_diff;
q_mul_new_contribution = rshift_round(product, shift_u_n_to_new_contribution) + zp_mul_new_contribution;
```

**(4) 最终合并 h_new = mul_old_contribution + mul_new_contribution（量化加法）**

$$q_{h\_new} = \frac{S_{old}}{S_h}(q_{old} - Z_{old}) + \frac{S_{new}}{S_h}(q_{new} - Z_{new}) + Z_h$$

**定点实现**：

```cpp
old_shifted = rshift_round(q_mul_old_contribution - zp_mul_old_contribution, shift_old_to_h);
new_shifted = rshift_round(q_mul_new_contribution - zp_mul_new_contribution, shift_new_to_h);
q_h_new = old_shifted + new_shifted + zp_h;
```

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
| mul_reset_hidden | INT | 非对称 | `shift_mul_reset_hidden_` | `zp_mul_reset_hidden_` | r × weight_hh_linear_n | n 门中间乘法 |
| new_gate_input | INT | 非对称 | `shift_new_gate_input_` | `zp_new_gate_input_` | ih_n + mul_reset_hidden | tanh 输入 |
| new_gate_output | INT | 对称 | `shift_new_gate_output_` | 0 | tanh(new_gate_input) | h 更新候选值 |
| mul_old_contribution | INT | 非对称 | `shift_mul_old_contribution_` | `zp_mul_old_contribution_` | u × h_old | h 更新旧状态贡献 |
| mul_new_contribution | INT | 非对称 | `shift_mul_new_contribution_` | `zp_mul_new_contribution_` | (1 - u) × n | h 更新新状态贡献 |

> **说明**：`update_gate_output` 和 `reset_gate_output` 使用 **UINT**（无符号整数），因为 sigmoid 输出范围为 [0, 1]，使用无符号类型可以充分利用所有 bit 位。其他参数使用 **INT**（有符号整数）。

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

#### B.6 中间计算参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `shift_mul_reset_hidden_` | int8_t | 标量 | r × weight_hh_linear_n 的缩放因子 |
| `zp_mul_reset_hidden_` | int32_t | 标量 | r × weight_hh_linear_n 的零点 |

**mul_reset_hidden (`op.mul_reset_hidden`)** = `reset_gate × weight_hh_linear_n`
- **数据流位置**：候选门 n 计算的中间步骤
- **说明**：由于 Linear 层已融合 bias，不再需要单独的 `Rh_add_br` 中间量
- **浮点范围**：由于 r∈[0,1]，范围不超过 weight_hh_linear_n
- **典型配置**：INT8，`shift=8`，scale=1/256

#### B.7 隐藏状态更新参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `shift_mul_old_contribution_` | int8_t | 标量 | u × h_old 的缩放因子 |
| `zp_mul_old_contribution_` | int32_t | 标量 | u × h_old 的零点 |
| `shift_mul_new_contribution_` | int8_t | 标量 | (1-u) × n 的缩放因子 |
| `zp_mul_new_contribution_` | int32_t | 标量 | (1-u) × n 的零点 |

**mul_old_contribution (`op.mul_old_contribution`)** = `update_gate × h_old`
- **数据流位置**：隐藏状态更新公式的第一项
- **浮点范围**：`[-1.0, 1.0]`（u∈[0,1]，h∈[-1,1]）
- **典型配置**：INT8，`shift=8`，scale=1/256

**mul_new_contribution (`op.mul_new_contribution`)** = `(1-update_gate) × new_gate`
- **数据流位置**：隐藏状态更新公式的第二项
- **浮点范围**：`[-1.0, 1.0]`（(1-u)∈[0,1]，n∈[-1,1]）
- **典型配置**：INT8，`shift=7`，scale=1/128

最终隐藏状态更新：`h_new = mul_old_contribution + mul_new_contribution`

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

| 参数 | 说明 | 计算公式 |
|------|------|----------|
| `shift_gemm_x_to_weight_ih_linear_[i]` | W×x GEMM 输出 rescale | `shift_W_[i] + shift_x_ - shift_weight_ih_linear_` |
| `shift_bw_to_weight_ih_linear_[i]` | bw rescale | `shift_bw_[i] - shift_weight_ih_linear_` |
| `shift_gemm_h_to_weight_hh_linear_[i]` | R×h GEMM 输出 rescale | `shift_R_[i] + shift_h_ - shift_weight_hh_linear_` |
| `shift_br_to_weight_hh_linear_[i]` | br rescale | `shift_br_[i] - shift_weight_hh_linear_` |
| `shift_weight_ih_linear_to_update_gate_input_` | ih 到 update_gate_input | `shift_weight_ih_linear_ - shift_update_gate_input_` |
| `shift_weight_hh_linear_to_update_gate_input_` | hh 到 update_gate_input | `shift_weight_hh_linear_ - shift_update_gate_input_` |
| `quant_one_in_update_gate_scale_` | 常数 1 的量化表示 | `round(1.0 × 2^shift_update_gate_output) + zp_update_gate_output` |

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

