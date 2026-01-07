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

- **量化**：$q = \text{round}(x \times 2^{exp2\_inv}) + zp$
- **反量化**：$x = (q - zp) \times 2^{-exp2\_inv}$
- 对称量化时 zp=0，简化为 $q = \text{round}(x / scale)$

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

### 各参数量化配置

| 参数 | 量化类型 | scale | zp | 备注 |
|------|----------|-------|-----|------|
| 输入 x | 非对称 + 动态范围 | `exp2_inv_x_` | `zp_x_` | 时间步 EMA 更新 |
| 隐藏状态 h | 非对称 + 动态范围 | `exp2_inv_h_` | `zp_h_` | 时间步 EMA 更新 |
| 权重 W | 对称 + per-channel | `exp2_inv_W_[i]` | 0 | size = hidden×3 |
| 权重 R | 对称 + per-channel | `exp2_inv_R_[i]` | 0 | size = hidden×3 |
| Wx 结果 | 非对称 | `exp2_inv_Wx_` | `zp_Wx_` | GEMM 输出 |
| Rh 结果 | 非对称 | `exp2_inv_Rh_` | `zp_Rh_` | GEMM 输出 |
| 偏置 bx | 对称 + per-channel | `exp2_inv_bx_[i]` | 0 | size = hidden×3 |
| 偏置 br | 对称 + per-channel | `exp2_inv_br_[i]` | 0 | size = hidden×3 |
| z_pre | 非对称 | `exp2_inv_z_pre_` | `zp_z_pre_` | 更新门预激活 |
| r_pre | 非对称 | `exp2_inv_r_pre_` | `zp_r_pre_` | 重置门预激活 |
| g_pre | 非对称 | `exp2_inv_g_pre_` | `zp_g_pre_` | 候选门预激活 |
| z_out | 非对称 | `exp2_inv_z_out_` | `zp_z_out_` | sigmoid 输出 |
| r_out | 非对称 | `exp2_inv_r_out_` | `zp_r_out_` | sigmoid 输出 |
| g_out | 对称 | `exp2_inv_g_out_` | 0 | tanh 输出 |

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

### Step 1: 预计算 Wx（所有时间步一次性）

```
Wx_tmp = cuBLAS::GEMM(W, x)  // [H×3, T×N]
```

### Step 2: 零点补偿预计算

由于 x 和 h 是非对称量化，GEMM 结果需要零点补偿。

#### 为什么需要零点补偿？

**背景**：在混合量化场景中，权重 W 采用对称量化（zp=0），而激活值 x 采用非对称量化（zp≠0）。当我们用整数 GEMM 计算 $q_W \cdot q_x$ 时，结果并不直接等于浮点 $W \cdot x$ 的量化值，需要额外的零点补偿项。

**推导过程**：

设量化参数为：
- 权重对称量化：$q_W = \text{round}(W / S_W)$，即 $W = q_W \cdot S_W$
- 激活非对称量化：$q_x = \text{round}(x / S_x) + Z_x$，即 $x = (q_x - Z_x) \cdot S_x$

我们需要计算矩阵乘法 $Y = W \cdot x$，其中 $Y[c] = \sum_k W[c,k] \cdot x[k]$

将浮点值用量化值表示：

$$Y[c] = \sum_k (q_W[c,k] \cdot S_W) \cdot ((q_x[k] - Z_x) \cdot S_x)$$

$$= S_W \cdot S_x \sum_k q_W[c,k] \cdot (q_x[k] - Z_x)$$

$$= S_W \cdot S_x \left( \sum_k q_W[c,k] \cdot q_x[k] - Z_x \sum_k q_W[c,k] \right)$$

$$= S_W \cdot S_x \left( \underbrace{(q_W \cdot q_x)[c]}_{\text{整数GEMM结果}} - \underbrace{Z_x \sum_k q_W[c,k]}_{\text{零点补偿项}} \right)$$

因此，**整数 GEMM 的结果需要减去零点补偿项**才能得到正确的浮点乘积的量化表示。

#### 预计算公式

$$W\_sum\_mul\_x\_zp[c] = Z_x \times \sum_{k} q_W[c, k]$$

$$R\_sum\_mul\_h\_zp[c] = Z_h \times \sum_{k} q_R[c, k]$$

> **优化说明**：由于 $\sum_k q_W[c,k]$ 只与权重相关，可在模型加载时一次性计算。而 $Z_x$ 和 $Z_h$ 在动态量化时可能随输入变化，因此补偿值 = 零点 × 权重行求和。

### Step 3: 时间步循环

```
for t in 0..T:
    1. Rh_tmp = cuBLAS::GEMM(R, h[t])
    2. CUDA Kernel 逐元素计算：z, r, g, h_new
```

---

## CUDA Kernel 逐元素计算详解

### 公共步骤：GEMM 结果 rescale

GEMM 输出的 `Wx_tmp` 和 `Rh_tmp` 包含三个门（z/r/g）的数据，需要按门索引分别提取并 rescale。

#### 推导过程

根据 Step 2 的零点补偿推导，整数 GEMM 结果表示的浮点值为：

$$Wx = S_W \cdot S_x \cdot (q_{Wx\_tmp} - W\_sum\_mul\_x\_zp)$$

现在要将其量化到目标参数 $(S_{Wx}, Z_{Wx})$：

$$q_{Wx} = \frac{Wx}{S_{Wx}} + Z_{Wx} = \frac{S_W \cdot S_x}{S_{Wx}}(q_{Wx\_tmp} - W\_sum\_mul\_x\_zp) + Z_{Wx}$$

#### 最终公式

对于门 $\gamma \in \{z, r, g\}$，使用对应索引 $\gamma\_idx$ 提取数据：

$$q_{Wx_\gamma} = \frac{S_{W[\gamma\_idx]} \cdot S_x}{S_{Wx}} (q_{Wx\_tmp[\gamma\_idx]} - W\_sum\_mul\_x\_zp[\gamma\_idx]) + Z_{Wx}$$

$$q_{Rh_\gamma} = \frac{S_{R[\gamma\_idx]} \cdot S_h}{S_{Rh}} (q_{Rh\_tmp[\gamma\_idx]} - R\_sum\_mul\_h\_zp[\gamma\_idx]) + Z_{Rh}$$

> **per-channel 说明**：$S_W$, $S_R$, $S_{bx}$, $S_{br}$ 均为 per-channel 数组，大小 = H×3。每个门使用对应索引的 scale 值。

---

### 1. 更新门 z（Update Gate）

**浮点公式**：`z = sigmoid(Wx_z + Rh_z + bx_z + br_z)`

#### 推导过程

设 `z_pre = Wx_z + Rh_z + bx_z + br_z`，根据**量化加法**公式，四个不同 scale 的量化值相加：

$$z\_pre = (q_{Wx_z} - Z_{Wx}) \cdot S_{Wx} + (q_{Rh_z} - Z_{Rh}) \cdot S_{Rh} + q_{bx_z} \cdot S_{bx} + q_{br_z} \cdot S_{br}$$

将 $z\_pre$ 量化到参数 $(S_{z\_pre}, Z_{z\_pre})$：

$$q_{z\_pre} = \frac{z\_pre}{S_{z\_pre}} + Z_{z\_pre}$$

$$= \frac{S_{Wx}}{S_{z\_pre}}(q_{Wx_z} - Z_{Wx}) + \frac{S_{Rh}}{S_{z\_pre}}(q_{Rh_z} - Z_{Rh}) + \frac{S_{bx}}{S_{z\_pre}}q_{bx_z} + \frac{S_{br}}{S_{z\_pre}}q_{br_z} + Z_{z\_pre}$$

> 注：偏置 $bx$, $br$ 是对称量化（zp=0），所以没有减零点项。

#### 量化计算公式（使用 z_idx 索引）

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

### 2. 重置门 r（Reset Gate）

**浮点公式**：`r = sigmoid(Wx_r + Rh_r + bx_r + br_r)`

**量化计算**（使用 r_idx 索引）：与 z 门结构相同，使用 `Wx_r`, `Rh_r`, `bx_r`, `br_r`，目标 scale 替换为 `S_{r_pre}`

---

### 3. 候选门 g（Candidate Gate）

**浮点公式**：`g = tanh(Wx_g + r × (Rh_g + br_g) + bx_g)`

> **注意**：haste 实现中，重置门 r 仅作用于 $(Rh_g + br_g)$，不作用于 $Wx_g$。

#### 推导过程

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

#### 量化计算公式汇总（使用 g_idx 索引）

$$q_{Rh\_add\_br\_g} = \frac{S_{Rh}}{S_{Rh\_add\_br}} (q_{Rh_g} - Z_{Rh}) + \frac{S_{br[g\_idx]}}{S_{Rh\_add\_br}} q_{br_g} + Z_{Rh\_add\_br}$$

$$q_{rRh} = \frac{S_{r\_out} \cdot S_{Rh\_add\_br}}{S_{rRh}} (q_r - Z_{r\_out})(q_{Rh\_add\_br\_g} - Z_{Rh\_add\_br}) + Z_{rRh}$$

$$q_{g\_pre} = \frac{S_{Wx}}{S_{g\_pre}}(q_{Wx_g} - Z_{Wx}) + \frac{S_{rRh}}{S_{g\_pre}}(q_{rRh} - Z_{rRh}) + \frac{S_{bx[g\_idx]}}{S_{g\_pre}}q_{bx_g} + Z_{g\_pre}$$

**激活函数**（根据位宽配置）：
- INT8: `g = tanh_int8_lut(clamp<int8>(q_g_pre))`
- INT16: `g = tanh_int16_lut(clamp<int16>(q_g_pre))`

---

### 4. 隐藏状态更新

**浮点公式**：`h_new = z × h_old + (1 - z) × g`

#### 推导过程

该公式包含两个乘法和一个加法：
1. $old\_contrib = z \times h_{old}$
2. $new\_contrib = (1 - z) \times g$
3. $h_{new} = old\_contrib + new\_contrib$

#### 4.1 计算 z × h_old（量化乘法）

**推导**：根据量化乘法公式

$$old\_contrib = z \times h_{old} = (q_z - Z_{z\_out}) \cdot S_{z\_out} \times (q_{h\_old} - Z_h) \cdot S_h$$

量化到参数 $(S_{old\_contrib}, Z_{old\_contrib})$：

$$q_{old\_contrib} = \frac{old\_contrib}{S_{old\_contrib}} + Z_{old\_contrib} = \frac{S_{z\_out} \cdot S_h}{S_{old\_contrib}}(q_z - Z_{z\_out})(q_{h\_old} - Z_h) + Z_{old\_contrib}$$

#### 4.2 计算 (1 - z)（量化减法的优化技巧）

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

#### 4.3 计算 (1 - z) × g（量化乘法）

**推导**：$(1-z)$ 使用量化参数 $(S_{z\_out}, Z_{z\_out})$，$g$ 使用 $(S_{g\_out}, Z_{g\_out})$

$$new\_contrib = (1-z) \times g = (q_{one\_minus\_z} - Z_{z\_out}) \cdot S_{z\_out} \times (q_g - Z_{g\_out}) \cdot S_{g\_out}$$

量化到参数 $(S_{new\_contrib}, Z_{new\_contrib})$：

$$q_{new\_contrib} = \frac{S_{z\_out} \cdot S_{g\_out}}{S_{new\_contrib}}(q_{one\_minus\_z} - Z_{z\_out})(q_g - Z_{g\_out}) + Z_{new\_contrib}$$

#### 4.4 最终合并（量化加法）

**推导**：$h_{new} = old\_contrib + new\_contrib$

$$h_{new} = (q_{old\_contrib} - Z_{old\_contrib}) \cdot S_{old\_contrib} + (q_{new\_contrib} - Z_{new\_contrib}) \cdot S_{new\_contrib}$$

量化到 h 的参数 $(S_h, Z_h)$：

$$q_{h\_new} = \frac{h_{new}}{S_h} + Z_h = \frac{S_{old\_contrib}}{S_h}(q_{old\_contrib} - Z_{old\_contrib}) + \frac{S_{new\_contrib}}{S_h}(q_{new\_contrib} - Z_{new\_contrib}) + Z_h$$

---

## 代码对应关系

| 计算步骤 | 代码位置 |
|----------|----------|
| 浮点 GRU 前向 | `gru_forward_gpu.cu::PointwiseOperations()` |
| 量化 GRU 前向 | `gru_forward_gpu_quant.cu::PointwiseOperationsQuantDynamic()` |
| z 门计算 | `gru_forward_gpu_quant.cu::computeZ()` |
| r 门计算 | `gru_forward_gpu_quant.cu::computeR()` |
| g 门计算 | `gru_forward_gpu_quant.cu::computeG()` |
| h 更新计算 | `gru_forward_gpu_quant.cu::computeH()` |
| 量化参数设置 | `gru_forward_gpu_quant.cu::setRescaleParam()` |
| 校准接口 | `gru_interface.cc::calibrateGruRanges()` |
| 参数计算 | `gru_interface.cc::calculateGRUQuantitativeParameters()` |

---
