# 量化GRU

## 效验

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
         - $q_{Rh} = \frac{S_R \cdot S_h}{S_{Rh}} (q_{Rh\_tmp} - q_{R\_sum\_mul\_h\_zp}) + Z_{Rh} $
         - $q_{Wx\_shifted} = \frac{S_{Wx}}{S_{z\_pre}} (q_{Wx} - Z_{Wx})$
         - $q_{Rh\_shifted} = \frac{S_{Rh}}{S_{z\_pre}} (q_{Rh} - Z_{Rh})$
         - $q_{bx\_shifted} = \frac{S_{bx}}{S_{z\_pre}} (q_{bx})$
         - $q_{br\_shifted} = \frac{S_{br}}{S_{z\_pre}} (q_{br})$
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
         - $q_{old\_contrib} = \frac{S_{z_out} \cdot S_{h\_old}}{S_{old\_contrib}}(q_{z\_out} - Z_z\_out)(q_{h\_old} - Z_{h}) + Z_{old\_contrib}$
         - $q_{one\_minus\_update} =  \frac{1.0}{S_{one\_minus\_update}} - \frac{S_{z_out}}{S_{one\_minus\_update}}(q_{z\_out} - Z{z\_out}) + Z_{one\_minus\_update}$
         - $q_{new\_contrib} = \frac{S_{one\_minus\_update} \cdot S_{g\_out}}{S_{new\_contrib}}(q_{one\_minus\_update} - Z_{one\_minus\_update})(q_{g\_out - Z_{g\_out}}) + Z_{new\_contrib} $
         - $cur\_h\_value = \frac{S_{old\_contrib}}{S_h}(q_{old\_contrib} - Z_{old\_contrib}) + \frac{S_{new\_contrib}}{S_h}(q_{new\_contrib} - Z_{new\_contrib}) + Z_h$
