# Haste 量化 GRU 纯定点计算流程

## 原浮点 GRU 门控计算

### 公式定义

| 门控 | 公式 | 说明 |
|------|------|------|
| 更新门 z | $z = \sigma(W_z x + R_z h + b_{xz} + b_{rz})$ | 控制保留多少旧状态 |
| 重置门 r | $r = \sigma(W_r x + R_r h + b_{xr} + b_{rr})$ | 控制遗忘多少旧状态 |
| 候选门 g | $g = \tanh(W_g x + r \odot (R_g h + b_{rg}) + b_{xg})$ | 生成候选新状态 |
| 新状态 h | $h_{new} = z \odot h_{old} + (1 - z) \odot g$ | 融合旧状态和候选状态 |

### 代码对应 (`gru_forward_gpu.cu`)

```cpp
// 更新门 z
const T z_pre = Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx];
const T z = sigmoid(z_pre);

// 重置门 r
const T r_pre = Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx];
const T r = sigmoid(r_pre);

// 候选门 g（注意：r 先乘以 Rh+br，再加 Wx 和 bx）
const T Rh_add_br_g = Rh[g_idx] + br[bg_idx];
const T g_pre = Wx[g_idx] + r * Rh_add_br_g + bx[bg_idx];
const T g = tanh(g_pre);

// 新隐藏状态
const T old_contrib = z * h[output_idx];
const T one_minus_z = 1.0 - z;
const T new_contrib = one_minus_z * g;
T cur_h_value = old_contrib + new_contrib;
```

> **haste 实现特点**：候选门 g 的计算中，重置门 r 仅作用于 $(R_g h + b_{rg})$ 部分，而不是整个 $(W_g x + R_g h)$。这与某些标准 GRU 实现略有不同。

## 量化核心规则说明

### 量化类型

| 类型 | 说明 |
|------|------|
| 对称量化 | zp = 0，仅需 scale = 2^(-exp2_inv_xxx)，无偏移 |
| 非对称量化 | zp ≠ 0，需同时使用 scale 和 zp，支持完整范围映射 |
| per-channel 量化 | 每个输出通道单独计算量化参数（对应权重矩阵每一行） |
| 动态范围更新 | 按时间步 EMA 更新 min/max：`min = 0.9×min_old + 0.1×min_cur` |

### 量化/反量化公式

**缩放因子 (scale)**

所有缩放因子均采用 2 的负 n 次方形式，便于用高效的移位操作代替乘除：

$$scale = 2^{-exp2\_inv}$$

例如：`exp2_inv=7` 对应 `scale=1/128=0.0078125`

**零点 (zero point)**

零点 `zp` 用于非对称量化，表示浮点零值对应的量化整数值：

- 对称量化：`zp = 0`
- 非对称量化：`zp = round(-min / scale)`

**通用量化/反量化公式**：

- **量化**：$q = \text{round}(x / scale) + zp$
- **反量化**：$x = (q - zp) \times scale$
- 对称量化时 zp=0，简化为 $q = \text{round}(x / scale)$

**本项目采用的 2 的幂次形式**：

由于所有 scale 均为 $2^{-exp2\_inv}$，公式可简化为：

- **量化**：$q = \text{round}(x \times 2^{exp2\_inv}) + zp = \text{round}(x \ll exp2\_inv) + zp$
- **反量化**：$x = (q - zp) \times 2^{-exp2\_inv} = (q - zp) \gg exp2\_inv$

> **计算优化**：乘以 $2^{exp2\_inv}$ 等价于左移 `exp2_inv` 位，除以 $2^{exp2\_inv}$ 等价于右移 `exp2_inv` 位。这种设计使得定点运算可以完全用整数移位实现，避免浮点乘除，大幅提升计算效率。

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
| bx, br | [H×3] | 偏置向量 |
| Wx | [T×N, H×3] | W @ x 结果 |
| Rh | [N, H×3] | R @ h 结果（每时间步） |
| v | [T×N, H×4] | 中间激活值 [z, r, g, Rh_add_br] |

### H×3 维度的门控分片

`H×3` 维度按以下方式切分为三个门的数据：

```
索引范围:  [0, H)      [H, 2H)     [2H, 3H)
门控类型:  z (更新门)   r (重置门)   g (候选门)
```

代码中的索引定义：
```cpp
const int z_idx = weight_idx + 0 * hidden_dim;  // [0, H)
const int r_idx = weight_idx + 1 * hidden_dim;  // [H, 2H)
const int g_idx = weight_idx + 2 * hidden_dim;  // [2H, 3H)
```

因此：
- `Wx[z_idx]` = Wx_z, `Wx[r_idx]` = Wx_r, `Wx[g_idx]` = Wx_g
- `Rh[z_idx]` = Rh_z, `Rh[r_idx]` = Rh_r, `Rh[g_idx]` = Rh_g
- `bx[b_z_idx]` = bx_z, `bx[b_r_idx]` = bx_r, `bx[b_g_idx]` = bx_g

---

## 量化推理流程

### 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         量化 GRU 前向计算流程                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 预处理阶段（一次性）                                                │   │
│  │  1. Wx_tmp = GEMM(W, x)           // 所有时间步的 Wx              │   │
│  │  2. 预计算权重行求和 Σ_k W[c,k]    // 用于零点补偿                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 时间步循环 for t = 0 to T-1                                        │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Step A: GEMM                                               │  │   │
│  │  │   Rh_tmp = GEMM(R, h[t])                                   │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │                              ↓                                    │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Step B: GEMM 结果 Rescale + 零点补偿                         │  │   │
│  │  │   Wx[t] = rescale(Wx_tmp[t] - W_sum_mul_x_zp)              │  │   │
│  │  │   Rh    = rescale(Rh_tmp    - R_sum_mul_h_zp)              │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │                              ↓                                    │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Step C: 门控计算（CUDA Kernel 逐元素）                       │  │   │
│  │  │   z = sigmoid(Wx_z + Rh_z + bx_z + br_z)                   │  │   │
│  │  │   r = sigmoid(Wx_r + Rh_r + bx_r + br_r)                   │  │   │
│  │  │   g = tanh(Wx_g + r*(Rh_g + br_g) + bx_g)                  │  │   │
│  │  │   h[t+1] = z*h[t] + (1-z)*g                                │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 1: 预处理（GEMM + 零点补偿准备）

```
Wx_tmp = cuBLAS::GEMM(W, x)  // [H×3, T×N]
```

由于 x 和 h 是非对称量化（zp≠0），GEMM 结果需要**零点补偿**。

#### 为什么需要零点补偿？

**背景**：权重 W 采用对称量化（zp=0），激活值 x 采用非对称量化（zp≠0）。整数 GEMM $q_W \cdot q_x$ 的结果并不直接等于 $W \cdot x$ 的量化值。

**推导**：设 $W = q_W \cdot S_W$，$x = (q_x - Z_x) \cdot S_x$，则：

$$Y[c] = \sum_k W[c,k] \cdot x[k] = S_W \cdot S_x \sum_k q_W[c,k] \cdot (q_x[k] - Z_x)$$

$$= S_W \cdot S_x \left( \underbrace{(q_W \cdot q_x)[c]}_{\text{整数GEMM}} - \underbrace{Z_x \sum_k q_W[c,k]}_{\text{零点补偿项}} \right)$$

**预计算公式**：

$$W\_sum\_mul\_x\_zp[c] = Z_x \times \sum_{k} q_W[c, k]$$

$$R\_sum\_mul\_h\_zp[c] = Z_h \times \sum_{k} q_R[c, k]$$

> **优化**：$\sum_k q_W[c,k]$ 在模型加载时一次性计算，运行时只需乘以零点。

### Step 2: 时间步循环

```
for t in 0..T:
    1. Rh_tmp = cuBLAS::GEMM(R, h[t])
    2. CUDA Kernel 逐元素计算：z, r, g, h_new
```

---

## CUDA Kernel 逐元素计算详解

### Step B: GEMM 结果 Rescale

GEMM 后需要做两件事：**减去零点补偿** + **Rescale 到目标量化参数**。

根据上面的推导，整数 GEMM 结果表示的浮点值为：

$$Wx = S_W \cdot S_x \cdot (q_{Wx\_tmp} - W\_sum\_mul\_x\_zp)$$

将其 rescale 到目标参数 $(S_{Wx}, Z_{Wx})$：

$$q_{Wx} = \frac{Wx}{S_{Wx}} + Z_{Wx} = \frac{S_W \cdot S_x}{S_{Wx}}(q_{Wx\_tmp} - W\_sum\_mul\_x\_zp) + Z_{Wx}$$

**per-channel 公式**（对于门 $\gamma \in \{z, r, g\}$）：

$$q_{Wx_\gamma} = \frac{S_{W[\gamma\_idx]} \cdot S_x}{S_{Wx}} (q_{Wx\_tmp[\gamma\_idx]} - W\_sum\_mul\_x\_zp[\gamma\_idx]) + Z_{Wx}$$

$$q_{Rh_\gamma} = \frac{S_{R[\gamma\_idx]} \cdot S_h}{S_{Rh}} (q_{Rh\_tmp[\gamma\_idx]} - R\_sum\_mul\_h\_zp[\gamma\_idx]) + Z_{Rh}$$

---

### Step C: 门控计算

#### C-1. 更新门 z（Update Gate）

**浮点公式**：`z = sigmoid(Wx_z + Rh_z + bx_z + br_z)`

**推导过程**

设 `z_pre = Wx_z + Rh_z + bx_z + br_z`，根据**量化加法**公式，四个不同 scale 的量化值相加：

$$z\_pre = (q_{Wx_z} - Z_{Wx}) \cdot S_{Wx} + (q_{Rh_z} - Z_{Rh}) \cdot S_{Rh} + q_{bx_z} \cdot S_{bx} + q_{br_z} \cdot S_{br}$$

将 $z\_pre$ 量化到参数 $(S_{z\_pre}, Z_{z\_pre})$：

$$q_{z\_pre} = \frac{z\_pre}{S_{z\_pre}} + Z_{z\_pre}$$

$$= \frac{S_{Wx}}{S_{z\_pre}}(q_{Wx_z} - Z_{Wx}) + \frac{S_{Rh}}{S_{z\_pre}}(q_{Rh_z} - Z_{Rh}) + \frac{S_{bx}}{S_{z\_pre}}q_{bx_z} + \frac{S_{br}}{S_{z\_pre}}q_{br_z} + Z_{z\_pre}$$

> 注：偏置 $bx$, $br$ 是对称量化（zp=0），所以没有减零点项。

**量化计算公式**（使用 z_idx 索引）：

定义中间变量（rescale 后的整数值）：

$$q_{Wx\_z\_shifted} = \frac{S_{Wx}}{S_{z\_pre}} (q_{Wx_z} - Z_{Wx})$$

$$q_{Rh\_z\_shifted} = \frac{S_{Rh}}{S_{z\_pre}} (q_{Rh_z} - Z_{Rh})$$

$$q_{bx\_z\_shifted} = \frac{S_{bx[z\_idx]}}{S_{z\_pre}} q_{bx_z}$$

$$q_{br\_z\_shifted} = \frac{S_{br[z\_idx]}}{S_{z\_pre}} q_{br_z}$$

最终相加：

$$q_{z\_pre} = q_{Wx\_z\_shifted} + q_{Rh\_z\_shifted} + q_{bx\_z\_shifted} + q_{br\_z\_shifted} + Z_{z\_pre}$$

**激活函数**（根据位宽配置）：
- INT8: `z = sigmoid_int8_lut(clamp<int8>(q_z_pre))`
- INT16: `z = sigmoid_int16_lut(clamp<int16>(q_z_pre))`

---

#### C-2. 重置门 r（Reset Gate）

**浮点公式**：`r = sigmoid(Wx_r + Rh_r + bx_r + br_r)`

**量化计算**：与 z 门结构完全相同，仅将索引和 scale 替换为 r 对应的参数。

---

#### C-3. 候选门 g（Candidate Gate）

**浮点公式**：`g = tanh(Wx_g + r × (Rh_g + br_g) + bx_g)`

> **注意**：haste 实现中，重置门 r 仅作用于 $(Rh_g + br_g)$，不作用于 $Wx_g$。

**推导过程**

**Step 1: 计算 $Rh\_add\_br\_g = Rh_g + br_g$（量化加法）**

$$Rh\_add\_br\_g = (q_{Rh_g} - Z_{Rh}) \cdot S_{Rh} + q_{br_g} \cdot S_{br}$$

量化到参数 $(S_{Rh\_add\_br}, Z_{Rh\_add\_br})$：

$$q_{Rh\_add\_br\_g} = \frac{Rh\_add\_br\_g}{S_{Rh\_add\_br}} + Z_{Rh\_add\_br}$$

$$= \frac{S_{Rh}}{S_{Rh\_add\_br}} (q_{Rh_g} - Z_{Rh}) + \frac{S_{br}}{S_{Rh\_add\_br}} q_{br_g} + Z_{Rh\_add\_br}$$

**Step 2: 计算 $rRh = r \times Rh\_add\_br\_g$（量化乘法）**

$$rRh = r \times Rh\_add\_br\_g = (q_r - Z_{r\_out}) \cdot S_{r\_out} \times (q_{Rh\_add\_br\_g} - Z_{Rh\_add\_br}) \cdot S_{Rh\_add\_br}$$

量化到参数 $(S_{rRh}, Z_{rRh})$：

$$q_{rRh} = \frac{rRh}{S_{rRh}} + Z_{rRh} = \frac{S_{r\_out} \cdot S_{Rh\_add\_br}}{S_{rRh}} (q_r - Z_{r\_out})(q_{Rh\_add\_br\_g} - Z_{Rh\_add\_br}) + Z_{rRh}$$

**Step 3: 计算 $g\_pre = Wx_g + rRh + bx_g$（量化加法）**

$$g\_pre = (q_{Wx_g} - Z_{Wx}) \cdot S_{Wx} + (q_{rRh} - Z_{rRh}) \cdot S_{rRh} + q_{bx_g} \cdot S_{bx}$$

量化到参数 $(S_{g\_pre}, Z_{g\_pre})$：

$$q_{g\_pre} = \frac{S_{Wx}}{S_{g\_pre}}(q_{Wx_g} - Z_{Wx}) + \frac{S_{rRh}}{S_{g\_pre}}(q_{rRh} - Z_{rRh}) + \frac{S_{bx}}{S_{g\_pre}}q_{bx_g} + Z_{g\_pre}$$

**量化计算公式汇总**（使用 g_idx 索引）：

$$q_{Rh\_add\_br\_g} = \frac{S_{Rh}}{S_{Rh\_add\_br}} (q_{Rh_g} - Z_{Rh}) + \frac{S_{br[g\_idx]}}{S_{Rh\_add\_br}} q_{br_g} + Z_{Rh\_add\_br}$$

$$q_{rRh} = \frac{S_{r\_out} \cdot S_{Rh\_add\_br}}{S_{rRh}} (q_r - Z_{r\_out})(q_{Rh\_add\_br\_g} - Z_{Rh\_add\_br}) + Z_{rRh}$$

$$q_{g\_pre} = \frac{S_{Wx}}{S_{g\_pre}}(q_{Wx_g} - Z_{Wx}) + \frac{S_{rRh}}{S_{g\_pre}}(q_{rRh} - Z_{rRh}) + \frac{S_{bx[g\_idx]}}{S_{g\_pre}}q_{bx_g} + Z_{g\_pre}$$

**激活函数**（根据位宽配置）：
- INT8: `g = tanh_int8_lut(clamp<int8>(q_g_pre))`
- INT16: `g = tanh_int16_lut(clamp<int16>(q_g_pre))`

---

#### C-4. 隐藏状态更新

**浮点公式**：`h_new = z × h_old + (1 - z) × g`

**推导过程**

该公式包含两个乘法和一个加法：
1. $old\_contrib = z \times h_{old}$
2. $new\_contrib = (1 - z) \times g$
3. $h_{new} = old\_contrib + new\_contrib$

**C-4.1 计算 z × h_old**（量化乘法）

**推导**：根据量化乘法公式

$$old\_contrib = z \times h_{old} = (q_z - Z_{z\_out}) \cdot S_{z\_out} \times (q_{h\_old} - Z_h) \cdot S_h$$

量化到参数 $(S_{old\_contrib}, Z_{old\_contrib})$：

$$q_{old\_contrib} = \frac{old\_contrib}{S_{old\_contrib}} + Z_{old\_contrib} = \frac{S_{z\_out} \cdot S_h}{S_{old\_contrib}}(q_z - Z_{z\_out})(q_{h\_old} - Z_h) + Z_{old\_contrib}$$

**C-4.2 计算 (1 - z)**（量化减法的优化技巧）

**问题**：如何在量化域计算 $1 - z$？

**推导**：

设 z 的量化参数为 $(S_{z\_out}, Z_{z\_out})$，则：$z = (q_z - Z_{z\_out}) \cdot S_{z\_out}$

我们希望 $(1-z)$ 也复用同样的 scale $S_{z\_out}$，这样后续乘法更简单。

将常数 1 用 $z_{out}$ 的量化参数表示：

$$q_1 = \frac{1}{S_{z\_out}} + Z_{z\_out} = \text{round}(1.0 / S_{z\_out}) + Z_{z\_out}$$

> 由于 $S = 2^{-exp2\_inv}$，所以 $1/S = 2^{exp2\_inv}$

然后计算 $1 - z$：

$$(1 - z) = (q_1 - Z_{z\_out}) \cdot S_{z\_out} - (q_z - Z_{z\_out}) \cdot S_{z\_out} = (q_1 - q_z) \cdot S_{z\_out}$$

为了让结果仍使用 $(S_{z\_out}, Z_{z\_out})$ 表示，设 $q_{one\_minus\_z}$ 满足：

$$(1 - z) = (q_{one\_minus\_z} - Z_{z\_out}) \cdot S_{z\_out}$$

则：

$$q_{one\_minus\_z} - Z_{z\_out} = q_1 - q_z$$

$$q_{one\_minus\_z} = q_1 - q_z + Z_{z\_out}$$

**最终公式**：

$$q_{1\_in\_z\_scale} = \text{round}(1.0 \times 2^{exp2\_inv\_z\_out}) + Z_{z\_out}$$

$$q_{one\_minus\_z} = q_{1\_in\_z\_scale} - q_z + Z_{z\_out}$$

> **关键优化**：$(1-z)$ 直接复用 z_out 的量化参数，无需额外的 scale 和 zp！

**C-4.3 计算 (1 - z) × g**（量化乘法）

**推导**：$(1-z)$ 使用量化参数 $(S_{z\_out}, Z_{z\_out})$，$g$ 使用 $(S_{g\_out}, Z_{g\_out})$

$$new\_contrib = (1-z) \times g = (q_{one\_minus\_z} - Z_{z\_out}) \cdot S_{z\_out} \times (q_g - Z_{g\_out}) \cdot S_{g\_out}$$

量化到参数 $(S_{new\_contrib}, Z_{new\_contrib})$：

$$q_{new\_contrib} = \frac{S_{z\_out} \cdot S_{g\_out}}{S_{new\_contrib}}(q_{one\_minus\_z} - Z_{z\_out})(q_g - Z_{g\_out}) + Z_{new\_contrib}$$

**C-4.4 最终合并**（量化加法）

**推导**：$h_{new} = old\_contrib + new\_contrib$

$$h_{new} = (q_{old\_contrib} - Z_{old\_contrib}) \cdot S_{old\_contrib} + (q_{new\_contrib} - Z_{new\_contrib}) \cdot S_{new\_contrib}$$

量化到 h 的参数 $(S_h, Z_h)$：

$$q_{h\_new} = \frac{h_{new}}{S_h} + Z_h = \frac{S_{old\_contrib}}{S_h}(q_{old\_contrib} - Z_{old\_contrib}) + \frac{S_{new\_contrib}}{S_h}(q_{new\_contrib} - Z_{new\_contrib}) + Z_h$$

---

## 附录

### A. 各参数量化配置

| 参数 | 数据类型 | 量化类型 | scale | zp | 计算公式 | 使用位置 |
|------|----------|----------|-------|-----|----------|----------|
| 输入 x | INT | 非对称 + 动态 | `exp2_inv_x_` | `zp_x_` | - | GRU 输入 |
| 隐藏状态 h | INT | 非对称 + 动态 | `exp2_inv_h_` | `zp_h_` | - | GRU 输出/下一步输入 |
| 权重 W | INT | 对称 + per-channel | `exp2_inv_W_[i]` | 0 | - | GEMM: W×x |
| 权重 R | INT | 对称 + per-channel | `exp2_inv_R_[i]` | 0 | - | GEMM: R×h |
| 偏置 bx | INT | 对称 + per-channel | `exp2_inv_bx_[i]` | 0 | - | 门控加法 |
| 偏置 br | INT | 对称 + per-channel | `exp2_inv_br_[i]` | 0 | - | 门控加法 |
| Wx | INT | 非对称 | `exp2_inv_Wx_` | `zp_Wx_` | W × x | z/r/g 门输入 |
| Rh | INT | 非对称 | `exp2_inv_Rh_` | `zp_Rh_` | R × h | z/r/g 门输入 |
| z_pre | INT | 非对称 | `exp2_inv_z_pre_` | `zp_z_pre_` | Wx_z + Rh_z + bx_z + br_z | sigmoid 输入 |
| z_out | **UINT** | 非对称 | `exp2_inv_z_out_` | `zp_z_out_` | sigmoid(z_pre) | h 更新门控 |
| r_pre | INT | 非对称 | `exp2_inv_r_pre_` | `zp_r_pre_` | Wx_r + Rh_r + bx_r + br_r | sigmoid 输入 |
| r_out | **UINT** | 非对称 | `exp2_inv_r_out_` | `zp_r_out_` | sigmoid(r_pre) | g 门乘法输入 |
| Rh_add_br | INT | 非对称 | `exp2_inv_Rh_add_br_` | `zp_Rh_add_br_` | Rh_g + br_g | g 门中间加法 |
| rRh | INT | 非对称 | `exp2_inv_rRh_` | `zp_rRh_` | r_out × Rh_add_br | g 门中间乘法 |
| g_pre | INT | 非对称 | `exp2_inv_g_pre_` | `zp_g_pre_` | Wx_g + rRh + bx_g | tanh 输入 |
| g_out | INT | 对称 | `exp2_inv_g_out_` | 0 | tanh(g_pre) | h 更新候选值 |
| old_contrib | INT | 非对称 | `exp2_inv_old_contrib_` | `zp_old_contrib_` | z_out × h_old | h 更新旧状态贡献 |
| new_contrib | INT | 非对称 | `exp2_inv_new_contrib_` | `zp_new_contrib_` | (1 - z_out) × g_out | h 更新新状态贡献 |

> **说明**：`z_out` 和 `r_out` 使用 **UINT**（无符号整数），因为 sigmoid 输出范围为 [0, 1]，使用无符号类型可以充分利用所有 bit 位。其他参数使用 **INT**（有符号整数）。

### B. 量化参数详细说明

本节对每个量化参数进行详细介绍，包括含义、数据类型和典型取值。基础概念（scale、zp、量化公式）请参见前文「量化核心规则说明」章节。

#### B.1 输入/输出参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `exp2_inv_x_` | int8_t | 标量 | 输入 x 的缩放因子指数 |
| `zp_x_` | int32_t | 标量 | 输入 x 的零点 |
| `exp2_inv_h_` | int8_t | 标量 | 隐藏状态 h 的缩放因子指数 |
| `zp_h_` | int32_t | 标量 | 隐藏状态 h 的零点 |

**输入 x (`input.x`)**
- **数据流位置**：GRU 输入层
- **浮点范围**：通常为 `[-1.0, 1.0]`
- **典型配置**：INT8 对称量化，`exp2_inv=7`，scale=1/128
- **特殊说明**：支持动态范围更新（EMA）

**隐藏状态 h (`output.h`)**
- **数据流位置**：GRU 输出层 / 下一时间步输入
- **浮点范围**：由 tanh 约束，理论范围 `[-1.0, 1.0]`
- **典型配置**：INT8 对称量化，`exp2_inv=7`，scale=1/128
- **特殊说明**：h 既是输入也是输出，共用量化参数

#### B.2 权重参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `exp2_inv_W_` | vector\<int8_t\> | per-channel | 输入权重 W 的缩放因子，size = hidden×3 |
| `exp2_inv_R_` | vector\<int8_t\> | per-channel | 循环权重 R 的缩放因子，size = hidden×3 |

**输入权重 W (`weight.W`)**
- **维度**：`[hidden×3, input_size]`，按 z/r/g 门顺序排列
- **量化方式**：对称量化 + per-channel（每行独立 scale）
- **典型配置**：INT8，`exp2_inv` 通常在 8~12 之间
- **特殊说明**：per-channel 量化可保留更多精度

**循环权重 R (`weight.R`)**
- **维度**：`[hidden×3, hidden_size]`
- **量化方式**：对称量化 + per-channel
- **典型配置**：与 W 类似

#### B.3 偏置参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `exp2_inv_bx_` | vector\<int8_t\> | per-channel | 输入偏置缩放因子，size = hidden×3 |
| `exp2_inv_br_` | vector\<int8_t\> | per-channel | 循环偏置缩放因子，size = hidden×3 |

**输入偏置 bx (`weight.bx`)** 和 **循环偏置 br (`weight.br`)**
- **维度**：`[hidden×3]`
- **量化方式**：对称量化 + per-channel
- **特殊说明**：偏置零点恒为 0，仅需 scale 参数

#### B.4 GEMM 输出参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `exp2_inv_Wx_` | int8_t | 标量 | W×x 矩阵乘结果的缩放因子 |
| `zp_Wx_` | int32_t | 标量 | W×x 的零点 |
| `exp2_inv_Rh_` | int8_t | 标量 | R×h 矩阵乘结果的缩放因子 |
| `zp_Rh_` | int32_t | 标量 | R×h 的零点 |

**Wx 结果 (`matmul.Wx`)**
- **数据流位置**：W×x GEMM 输出，后续送入门控计算
- **浮点范围**：取决于权重和输入的分布
- **典型配置**：INT8，`exp2_inv=6`，scale=1/64
- **零点补偿**：需要减去 `W_sum × zp_x`

**Rh 结果 (`matmul.Rh`)**
- **数据流位置**：R×h GEMM 输出，每时间步重新计算
- **浮点范围**：通常比 Wx 范围小（h 已被约束在 [-1,1]）
- **典型配置**：INT8，`exp2_inv=8`，scale=1/256
- **零点补偿**：需要减去 `R_sum × zp_h`

#### B.5 门激活函数参数

##### 预激活参数（激活函数输入）

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `exp2_inv_z_pre_` | int8_t | 标量 | z 门 sigmoid 输入的缩放因子 |
| `zp_z_pre_` | int32_t | 标量 | z 门 sigmoid 输入的零点 |
| `exp2_inv_r_pre_` | int8_t | 标量 | r 门 sigmoid 输入的缩放因子 |
| `zp_r_pre_` | int32_t | 标量 | r 门 sigmoid 输入的零点 |
| `exp2_inv_g_pre_` | int8_t | 标量 | g 门 tanh 输入的缩放因子 |
| `zp_g_pre_` | int32_t | 标量 | g 门 tanh 输入的零点 |

**z_pre (`gate.z_pre`)** = `Wx_z + Rh_z + bx_z + br_z`
- **浮点范围**：通常为 `[-4.0, 4.0]`
- **典型配置**：INT8，`exp2_inv=6`，scale=1/64，覆盖 [-2, 2]
- **作用**：作为 sigmoid LUT 的输入索引

**r_pre (`gate.r_pre`)** = `Wx_r + Rh_r + bx_r + br_r`
- **配置与 z_pre 相同**

**g_pre (`gate.g_pre`)** = `Wx_g + r×(Rh_g + br_g) + bx_g`
- **浮点范围**：通常为 `[-4.0, 4.0]`
- **典型配置**：INT8，`exp2_inv=6`，scale=1/64
- **作用**：作为 tanh LUT 的输入索引

##### 后激活参数（激活函数输出）

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `exp2_inv_z_out_` | int8_t | 标量 | z 门 sigmoid 输出的缩放因子 |
| `zp_z_out_` | int32_t | 标量 | z 门 sigmoid 输出的零点 |
| `exp2_inv_r_out_` | int8_t | 标量 | r 门 sigmoid 输出的缩放因子 |
| `zp_r_out_` | int32_t | 标量 | r 门 sigmoid 输出的零点 |
| `exp2_inv_g_out_` | int8_t | 标量 | g 门 tanh 输出的缩放因子 |
| `zp_g_out_` | int32_t | 标量 | g 门 tanh 输出的零点（对称量化时为 0） |

**z_out (`gate.z_out`)** = `sigmoid(z_pre)`
- **浮点范围**：`[0.0, 1.0]`（sigmoid 输出恒正）
- **默认配置**：UINT8 非对称量化（`z_out_symmetric_ = false`）
- **特殊说明**：由于 sigmoid 输出范围 [0,1]，使用 UINT 可充分利用所有 bit 位表示正值范围

**r_out (`gate.r_out`)** = `sigmoid(r_pre)`
- **配置与 z_out 相同**

**g_out (`gate.g_out`)** = `tanh(g_pre)`
- **浮点范围**：`[-1.0, 1.0]`（tanh 输出）
- **典型配置**：INT8 对称量化，`exp2_inv=7`，scale=1/128
- **特殊说明**：对称量化，zp=0

#### B.6 中间计算参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `exp2_inv_Rh_add_br_` | int8_t | 标量 | Rh_g + br_g 的缩放因子 |
| `zp_Rh_add_br_` | int32_t | 标量 | Rh_g + br_g 的零点 |
| `exp2_inv_rRh_` | int8_t | 标量 | r × (Rh_g + br_g) 的缩放因子 |
| `zp_rRh_` | int32_t | 标量 | r × (Rh_g + br_g) 的零点 |

**Rh_add_br (`op.Rh_add_br`)** = `Rh_g + br_g`
- **数据流位置**：候选门 g 计算的中间步骤
- **典型配置**：INT8，`exp2_inv=8`，scale=1/256
- **特殊说明**：该值会与 r 门输出相乘

**rRh (`op.rRh`)** = `r × (Rh_g + br_g)`
- **数据流位置**：候选门 g 计算的中间步骤
- **浮点范围**：由于 r∈[0,1]，范围不超过 Rh_add_br
- **典型配置**：INT8，`exp2_inv=8`，scale=1/256

#### B.7 隐藏状态更新参数

| 参数 | 变量名 | 类型 | 说明 |
|------|--------|------|------|
| `exp2_inv_old_contrib_` | int8_t | 标量 | z × h_old 的缩放因子 |
| `zp_old_contrib_` | int32_t | 标量 | z × h_old 的零点 |
| `exp2_inv_new_contrib_` | int8_t | 标量 | (1-z) × g 的缩放因子 |
| `zp_new_contrib_` | int32_t | 标量 | (1-z) × g 的零点 |

**old_contrib (`op.old_contrib`)** = `z × h_old`
- **数据流位置**：隐藏状态更新公式的第一项
- **浮点范围**：`[-1.0, 1.0]`（z∈[0,1]，h∈[-1,1]）
- **典型配置**：INT8，`exp2_inv=8`，scale=1/256

**new_contrib (`op.new_contrib`)** = `(1-z) × g`
- **数据流位置**：隐藏状态更新公式的第二项
- **浮点范围**：`[-1.0, 1.0]`（(1-z)∈[0,1]，g∈[-1,1]）
- **典型配置**：INT8，`exp2_inv=7`，scale=1/128

最终隐藏状态更新：`h_new = old_contrib + new_contrib`

#### B.8 重缩放参数（Device 端）

重缩放参数由 `GRUQuantitativeParameters` 预计算得出，存储在 `QuantGRUReScale` 结构体中，供 GPU Kernel 使用。

**命名约定**：

- `n_A_div_B`：表示 `scale_A / scale_B ≈ 2^(-n)`
- `exp2_inv_A_div_B`：同上，强调指数形式

**示例**：
```cpp
// Wx rescale 到 z_pre
// rescale_factor = scale_Wx / scale_z_pre = 2^(-exp2_inv_Wx) / 2^(-exp2_inv_z_pre)
//                = 2^(exp2_inv_z_pre - exp2_inv_Wx)
// 存储 n = exp2_inv_Wx - exp2_inv_z_pre
exp2_inv_Wx_div_z_pre_ = exp2_inv_Wx_ - exp2_inv_z_pre_;
```

**主要重缩放参数**：

| 参数 | 说明 | 计算公式 |
|------|------|----------|
| `n_W_mul_x_div_Wx_[i]` | W×x GEMM 输出 rescale | `exp2_inv_W_[i] + exp2_inv_x_ - exp2_inv_Wx_` |
| `n_R_mul_h_div_Rh_[i]` | R×h GEMM 输出 rescale | `exp2_inv_R_[i] + exp2_inv_h_ - exp2_inv_Rh_` |
| `exp2_inv_Wx_div_z_pre_` | Wx 到 z_pre | `exp2_inv_Wx_ - exp2_inv_z_pre_` |
| `exp2_inv_Rh_div_z_pre_` | Rh 到 z_pre | `exp2_inv_Rh_ - exp2_inv_z_pre_` |
| `n_bx_div_z_[i]` | bx 到 z_pre | `exp2_inv_bx_[i] - exp2_inv_z_pre_` |
| `one_in_z_scale_` | 常数 1 的量化表示 | `round(1.0 × 2^exp2_inv_z_out) + zp_z_out` |

#### B.9 LUT 查找表

激活函数采用查找表 (LUT) 实现，避免运行时计算 sigmoid/tanh：

| 参数 | 类型 | 说明 |
|------|------|------|
| `sigmoid_z_lut_` | SigmoidLUT | z 门 sigmoid 查找表 |
| `sigmoid_r_lut_` | SigmoidLUT | r 门 sigmoid 查找表 |
| `tanh_g_lut_` | TanhLUT | g 门 tanh 查找表 |

LUT 在 `finalize_calibration` 时根据输入输出量化参数生成，支持分段线性插值以提高精度。

### C. 代码对应关系

| 计算步骤 | 代码位置 |
|----------|----------|
| 浮点 GRU 前向 | `gru_forward_gpu.cu::PointwiseOperations()` |
| 量化 GRU 前向 | `gru_forward_gpu_quant.cu::PointwiseOperationsQuant()` |
| z 门计算 | `gru_forward_gpu_quant.cu::computeZ()` |
| r 门计算 | `gru_forward_gpu_quant.cu::computeR()` |
| g 门计算 | `gru_forward_gpu_quant.cu::computeG()` |
| h 更新计算 | `gru_forward_gpu_quant.cu::computeH()` |
| 量化参数设置 | `gru_forward_gpu_quant.cu::setRescaleParam()` |
| 校准接口 | `gru_interface.cc::calibrateGruRanges()` |
| 参数计算 | `gru_interface.cc::calculateGRUQuantitativeParameters()` |
| 量化参数结构体定义 | `quantize_ops_helper.h::GRUQuantitativeParameters` |
| 重缩放参数结构体定义 | `quantize_ops_helper.h::QuantGRUReScale` |
| LUT 生成 | `quantize_ops_helper.h::generate_piecewise_linear_lut_to_params()` |

### D. JSON 配置文件格式

量化参数可导出为 JSON 格式（AIMET 兼容），文件结构如下：

```json
{
  "model_info": {
    "input_size": 120,
    "hidden_size": 120,
    "bias": true
  },
  "operators": {
    "input.x": {
      "dtype": "INT8",
      "symmetric": true,
      "scale": 0.0078125,
      "zero_point": 0,
      "real_min": -1.0,
      "real_max": 0.9921875,
      "enc_type": "PER_TENSOR",
      "n": 7
    },
    "gate.z_pre": { ... },
    "gate.z_out": { ... },
    "weight.W": {
      "dtype": "INT8",
      "symmetric": true,
      "scale": [...],
      "enc_type": "PER_CHANNEL",
      "n": [...]
    }
  }
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `dtype` | string | 数据类型：INT8, UINT8, INT16 等 |
| `symmetric` | bool | 是否对称量化 |
| `scale` | float/array | 缩放因子，per-channel 时为数组 |
| `zero_point` | int | 零点 |
| `real_min` | float | 量化可表示的最小浮点值 |
| `real_max` | float | 量化可表示的最大浮点值 |
| `enc_type` | string | 编码类型：PER_TENSOR 或 PER_CHANNEL |
| `n` | int/array | exp2_inv 值，即 scale = 2^(-n) |

---
