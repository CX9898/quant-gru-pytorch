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

由于 x 和 h 是非对称量化，GEMM 结果需要零点补偿：

$$W\_sum\_mul\_x\_zp[c] = zp_x \times \sum_{k} W[c, k]$$

$$R\_sum\_mul\_h\_zp[c] = zp_h \times \sum_{k} R[c, k]$$

### Step 3: 时间步循环

```
for t in 0..T:
    1. Rh_tmp = cuBLAS::GEMM(R, h[t])
    2. CUDA Kernel 逐元素计算：z, r, g, h_new
```

---

## CUDA Kernel 逐元素计算详解

### 公共步骤：GEMM 结果 rescale

GEMM 输出的 `Wx_tmp` 和 `Rh_tmp` 包含三个门（z/r/g）的数据，需要按门索引分别提取并 rescale：

对于门 $\gamma \in \{z, r, g\}$，使用对应索引 $\gamma\_idx$ 提取数据：

$$q_{Wx_\gamma} = \frac{S_{W[\gamma\_idx]} \cdot S_x}{S_{Wx}} (q_{Wx\_tmp[\gamma\_idx]} - W\_sum\_mul\_x\_zp[\gamma\_idx]) + Z_{Wx}$$

$$q_{Rh_\gamma} = \frac{S_{R[\gamma\_idx]} \cdot S_h}{S_{Rh}} (q_{Rh\_tmp[\gamma\_idx]} - R\_sum\_mul\_h\_zp[\gamma\_idx]) + Z_{Rh}$$

> **per-channel 说明**：$S_W$, $S_R$, $S_{bx}$, $S_{br}$ 均为 per-channel 数组，大小 = H×3。每个门使用对应索引的 scale 值。

---

### 1. 更新门 z（Update Gate）

**浮点公式**：`z = sigmoid(Wx_z + Rh_z + bx_z + br_z)`

**量化计算**（使用 z_idx 索引）：

$$q_{Wx\_z\_shifted} = \frac{S_{Wx}}{S_{z\_pre}} (q_{Wx_z} - Z_{Wx})$$

$$q_{Rh\_z\_shifted} = \frac{S_{Rh}}{S_{z\_pre}} (q_{Rh_z} - Z_{Rh})$$

$$q_{bx\_z\_shifted} = \frac{S_{bx[z\_idx]}}{S_{z\_pre}} q_{bx_z}$$

$$q_{br\_z\_shifted} = \frac{S_{br[z\_idx]}}{S_{z\_pre}} q_{br_z}$$

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

**量化计算**（使用 g_idx 索引）：

$$q_{Rh\_add\_br\_g} = \frac{S_{Rh}}{S_{Rh\_add\_br}} (q_{Rh_g} - Z_{Rh}) + \frac{S_{br[g\_idx]}}{S_{Rh\_add\_br}} q_{br_g} + Z_{Rh\_add\_br}$$

$$q_{rRh} = \frac{S_{r\_out} \cdot S_{Rh\_add\_br}}{S_{rRh}} (q_r - Z_{r\_out})(q_{Rh\_add\_br\_g} - Z_{Rh\_add\_br}) + Z_{rRh}$$

$$q_{g\_pre} = \frac{S_{Wx}}{S_{g\_pre}}(q_{Wx_g} - Z_{Wx}) + \frac{S_{rRh}}{S_{g\_pre}}(q_{rRh} - Z_{rRh}) + \frac{S_{bx[g\_idx]}}{S_{g\_pre}}q_{bx_g} + Z_{g\_pre}$$

**激活函数**（根据位宽配置）：
- INT8: `g = tanh_int8_lut(clamp<int8>(q_g_pre))`
- INT16: `g = tanh_int16_lut(clamp<int16>(q_g_pre))`

---

### 4. 隐藏状态更新

**浮点公式**：`h_new = z × h_old + (1 - z) × g`

**量化计算**：

#### 4.1 计算 z × h_old

$$q_{old\_contrib} = \frac{S_{z\_out} \cdot S_h}{S_{old\_contrib}}(q_z - Z_{z\_out})(q_{h\_old} - Z_h) + Z_{old\_contrib}$$

#### 4.2 计算 (1 - z)

> **关键优化**：直接复用 z_out 的 scale，无需额外量化参数

$$q_{1\_in\_z\_scale} = \text{round}(1.0 \times 2^{exp2\_inv\_z\_out}) + Z_{z\_out}$$

$$q_{one\_minus\_z} = q_{1\_in\_z\_scale} - q_z + Z_{z\_out}$$

#### 4.3 计算 (1 - z) × g

$$q_{new\_contrib} = \frac{S_{z\_out} \cdot S_{g\_out}}{S_{new\_contrib}}(q_{one\_minus\_z} - Z_{z\_out})(q_g - Z_{g\_out}) + Z_{new\_contrib}$$

#### 4.4 最终合并

$$q_{h\_new} = \frac{S_{old\_contrib}}{S_h}(q_{old\_contrib} - Z_{old\_contrib}) + \frac{S_{new\_contrib}}{S_h}(q_{new\_contrib} - Z_{new\_contrib}) + Z_h$$

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
