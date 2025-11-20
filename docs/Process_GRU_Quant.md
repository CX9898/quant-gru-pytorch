# haste 量化 GRU

## 原浮点的各个门控计算

![haste_GRU_formula.png](haste_GRU_formula.png)

## 量化核心规则说明

1. 对称量化：量化零点 zp = 0，仅需缩放因子（scale = 2^(-exp2_inv_xxx)），无偏移；
2. 非对称量化：量化零点 zp ≠ 0，需同时使用缩放因子和零点，支持浮点数范围完整映射；
3. per-channel 量化：每个输出通道（对应权重矩阵每一行）单独计算量化参数（scale/zp）；
4. 动态更新量化范围：仅适用于输入x和隐藏状态h，按时间步滑动更新min/max以适配数据分布。

各参数量化要求（按 GRU 计算流程排序）：
- 输入 x（当前时间步输入向量）：
  非对称量化 + 时间步动态范围更新；
  范围更新规则：最终 min = 前一时间步 min × 0.9 + 当前时间步 min × 0.1，最终 max = 前一时间步 max × 0.9 + 当前时间步 max × 0.1；
  缩放因子 scale_x = 2^(-exp2_inv_x_)，由最终 min/max 计算得出。

- 隐藏状态 h（上一时间步隐藏状态向量）：
  非对称量化 + 时间步动态范围更新；
  范围更新规则：与输入x一致（最终 min/max = 前序0.9 + 当前0.1加权）；
  缩放因子 scale_h = 2^(-exp2_inv_h_)，由最终 min/max 计算得出。

- 权重矩阵 W（输入x的权重，对应 Wx = W×x）：
  对称量化 + per-channel 量化（每个输出通道/每一行1个参数）；
  无量化零点（zp=0），仅需缩放因子 scale_W[i] = 2^(-exp2_inv_W_[i])。

- 权重矩阵 R（隐藏状态h的权重，对应 Rh = R×h）：
  对称量化 + per-channel 量化（每个输出通道/每一行1个参数）；
  无量化零点（zp=0），仅需缩放因子 scale_R[i] = 2^(-exp2_inv_R_[i])。

- 线性变换结果 Wx（W×x）：
  非对称量化；
  需缩放因子 scale_Wx = 2^(-exp2_inv_Wx_) 和量化零点 zp_Wx_。

- 线性变换结果 Rh（R×h）：
  非对称量化；
  需缩放因子 scale_Rh = 2^(-exp2_inv_Rh_) 和量化零点 zp_Rh_。

- 偏置项 bx（Wx 对应的偏置）：
  对称量化 + per-channel 量化（每个输出通道1个参数）；
  无量化零点（zp=0），仅需缩放因子 scale_bx[i] = 2^(-exp2_inv_bx_[i])。

- 偏置项 br（Rh 对应的偏置）：
  对称量化 + per-channel 量化（每个输出通道1个参数）；
  无量化零点（zp=0），仅需缩放因子 scale_br[i] = 2^(-exp2_inv_br_[i])。

- 门控预激活结果 z_pre（更新门：Wx_z + Rh_z + bx_z + br_z）：
  非对称量化；
  需缩放因子 scale_z_pre = 2^(-exp2_inv_z_pre_) 和量化零点 zp_z_pre_。

- 门控预激活结果 r_pre（重置门：Wx_r + Rh_r + bx_r + br_r）：
  非对称量化；
  需缩放因子 scale_r_pre = 2^(-exp2_inv_r_pre_) 和量化零点 zp_r_pre_。

- 门控预激活结果 g_pre（候选态：Wx_g + r×(Rh_g + br_g) + bx_g）：
  非对称量化；
  需缩放因子 scale_g_pre = 2^(-exp2_inv_g_pre_) 和量化零点 zp_g_pre_。

- 门控激活结果 z_out（更新门：sigmoid(z_pre)）：
  非对称量化；
  需缩放因子 scale_z_out = 2^(-exp2_inv_z_out_) 和量化零点 zp_z_out_。

- 门控激活结果 r_out（重置门：sigmoid(r_pre)）：
  非对称量化；
  需缩放因子 scale_r_out = 2^(-exp2_inv_r_out_) 和量化零点 zp_r_out_。

- 门控激活结果 g_out（候选态：tanh(g_pre)）：
  对称量化；
  无量化零点（zp=0），仅需缩放因子 scale_g_out = 2^(-exp2_inv_g_out_)。

补充说明：
- 所有缩放因子均以「2的负n次方」形式存储，exp2_inv_xxx 为对应指数（scale = 2^(-exp2_inv_xxx)）；
- 量化/反量化映射公式：量化值 q = round( (x - zp) / scale )，反量化值 x = (q - zp) × scale（对称量化时 zp=0，简化为 q=round(x/scale)、x=q×scale）。

---

## 效验



---

## 量化推理

1. 首先调用 cuBlas::GEMM 提前计算好所有时间步的 `Wx_tmp = W * x`. 在每个时间步cell的for循环开始传入对应时间步的Wx.
2. 因为x是非对称量化, Wx的GEMM结果每一个值需要进行零点补偿. 也就是减去`W_sum_mul_zp_x`
$$
S_{gi}(q_{gi} - Z_{gi}) = S_x(q_x - Z_x) \cdot S_{w_{ih}} q_{w_{ih}} + S_{b_{ih}}q_{b_{ih}}
$$
> 计算W的每个通道的输入维度之和乘以zp_x, 然后Wx的结果的每个通道的值要减去对应通道的零点补偿值
3. 因为 R * h 的GEMM也是因为h是非对称量化, 所以也需要提前计算权重R的每个通道的输入维度之和乘以zp_h, 用于后续零点补偿(`R_sum_mul_h_zp`).

4. for i in 0~steps: 循环每个时间步
   1. 每个cell第一步先调用 cuBlas::GEMM 计算 `Rh_tmp = R * h`
   2. 执行逐元素并行运算(CUDA Kernel)
      1. update gate z门计算: 原始haste浮点计算: `z = sigmoid(Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx])`
         - $q_{Wx} = \frac{S_W \cdot S_x}{S_{Wx}} (q_{Wx\_tmp} - q_{W\_sum\_mul\_x\_zp}) + Z_{Wx} $
         > 其中 $S_W$ 是per-channel的. 也就是储存为数组. size = hidden * 3. 后续其他门控计算步骤也都是
         - $q_{Rh} = \frac{S_R \cdot S_h}{S_{Rh}} (q_{Rh\_tmp} - q_{R\_sum\_mul\_h\_zp}) + Z_{Rh} $
         > 其中 $S_R$ 是per-channel的. 也就是储存为数组. size = hidden * 3. 后续其他门控计算步骤也都是
         - $q_{Wx\_shifted} = \frac{S_{Wx}}{S_{z\_pre}} (q_{Wx} - Z_{Wx})$
         - $q_{Rh\_shifted} = \frac{S_{Rh}}{S_{z\_pre}} (q_{Rh} - Z_{Rh})$
         - $q_{bx\_shifted} = \frac{S_{bx}}{S_{z\_pre}} (q_{bx})$
         > 其中 $S_{bx}$ 是per-channel的. 也就是储存为数组. size = hidden * 3. 后续其他门控计算步骤也都是
         - $q_{br\_shifted} = \frac{S_{br}}{S_{z\_pre}} (q_{br})$
         > 其中 $S_{br}$ 是per-channel的. 也就是储存为数组. size = hidden * 3. 后续其他门控计算步骤也都是
         - $q_{z\_pre\_i32} = q_{Wx\_shifted} + q_{Rh\_shifted} + q_{bx\_shifted} + q_{br\_shifted} + Z_{z\_pre}$
         - $q_{z\_pre\_i8} = clamp<int8>(q_{z\_pre\_i32})$ 截断到int8的范围
         - $z = sigmoid\_int8\_lut(q_{z\_pre\_i8})$
      2. reset gate r门计算: 原始haste浮点计算: `r = sigmoid(Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx])`
         - $q_{Wx} = \frac{S_W \cdot S_x}{S_{Wx}} (q_{Wx\_tmp} - q_{W\_sum\_mul\_x\_zp}) + Z_{Wx} $
         - $q_{Rh} = \frac{S_R \cdot S_h}{S_{Rh}} (q_{Rh\_tmp} - q_{R\_sum\_mul\_h\_zp}) + Z_{Rh} $
         - $q_{Wx\_shifted} = \frac{S_{Wx}}{S_{r\_pre}} (q_{Wx} - Z_{Wx})$
         - $q_{Rh\_shifted} = \frac{S_{Rh}}{S_{r\_pre}} (q_{Rh} - Z_{Rh})$
         - $q_{bx\_shifted} = \frac{S_{bx}}{S_{r\_pre}} (q_{bx})$
         - $q_{br\_shifted} = \frac{S_{br}}{S_{r\_pre}} (q_{br})$
         - $q_{r\_pre\_i32} = q_{Wx\_shifted} + q_{Rh\_shifted} + q_{bx\_shifted} + q_{br\_shifted} + Z_{r\_pre}$
         - $q_{r\_pre\_i8} = clamp<int8>(q_{r\_pre\_i32})$ 截断到int8的范围
         - $r = sigmoid\_int8\_lut(q_{r\_pre\_i8})$
      3. new gate g门计算: 原始haste浮点计算: `g = tanh (Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx])`
         - $q_{Wx} = \frac{S_W \cdot S_x}{S_{Wx}} (q_{Wx\_tmp} - q_{W\_sum\_mul\_x\_zp}) + Z_{Wx} $
         - $q_{Rh} = \frac{S_R \cdot S_h}{S_{Rh}} (q_{Rh\_tmp} - q_{R\_sum\_mul\_h\_zp}) + Z_{Rh} $
         - $q_{Rh\_add\_br} = \frac{S_{Rh}}{S_{Rh\_add\_br}} (q_{Rh} -  Z_{Rh}) + \frac{S_{br}}{S_{Rh\_add\_br}} (q_{br}) + Z_{Rh\_add\_br}$
         - $q_{rRh} = \frac{S_{r\_out} \cdot S_{Rh\_add\_br}}{S_{rRh}} (q_{r} - Z_{r_out})(q_{Rh\_add\_br} - Z_{Rh\_add\_br})$
         - $q_{Wx\_shifted} = \frac{S_{Wx}}{S_{g\_pre}} (q_{Wx} - Z_{Wx})$
         - $q_{bx\_shifted} = \frac{S_{bx}}{S_{g\_pre}} (q_{bx})$
         - $q_{rRh\_shifted} = \frac{S_{rRh}}{S_{g\_pre}} (q_{rRh} - Z_{rRh})$
         - $q_{g\_pre\_i32} = q_{Wx\_shifted} + q_{rRh\_shifted} + q_{bx\_shifted} + Z_{g\_pre}$
         - $q_{g\_pre\_i8} = clamp<int8>(q_{g\_pre\_i32})$ 截断到int8的范围
         - $g = tanh\_int8\_lut(q_{g\_pre\_i8})$
      4. 最终h: 原始haste浮点计算: `cur_h_value = z * h_old + (1.0 - z) * g`
         - $q_{old\_contrib} = \frac{S_{z_out} \cdot S_{h\_old}}{S_{old\_contrib}}(q_{z\_out} - Z_z\_out)(q_{h\_old} - Z_{h}) + Z_{old\_contrib}$ > 相当于 z * h_old
         - $q_{one\_minus\_update} =  \frac{1.0}{S_{one\_minus\_update}} - \frac{S_{z_out}}{S_{one\_minus\_update}}(q_{z\_out} - Z{z\_out}) + Z_{one\_minus\_update}$ > 相当于(1.0 - z)
         - $q_{new\_contrib} = \frac{S_{one\_minus\_update} \cdot S_{g\_out}}{S_{new\_contrib}}(q_{one\_minus\_update} - Z_{one\_minus\_update})(q_{g\_out - Z_{g\_out}}) + Z_{new\_contrib} $ > 相当于(1.0 - z) * g
         - $cur\_h\_value = \frac{S_{old\_contrib}}{S_h}(q_{old\_contrib} - Z_{old\_contrib}) + \frac{S_{new\_contrib}}{S_h}(q_{new\_contrib} - Z_{new\_contrib}) + Z_h$

---
