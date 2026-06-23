# 更新日志 (Changelog)

本文档记录 Quant-GRU-PyTorch 项目的所有重要更新和变更。

---

## [2026-06-23]

### 版本更新
1. **版本号更新**: 将版本号从 1.0.7 更新至 1.0.8

### Bug 修复
1. **支持 32bit bias 量化（修复标定阶段 `num_steps` 整型溢出）**: 标定阶段以 `int32` 计算量化级数 `num_steps = qmax_auto_scale() - qmin_auto_scale()`，在 32bit 下 `2147483647 - (-2147483648)` 发生 int32 溢出变为负数，触发 `num_steps must be > 0`，导致 `bias_ih`/`bias_hh` 设为 32bit 时校准直接报错。同时多处下游又把 `int64` 的 `num_steps` 截断回 `int`（如 `get_minimum_scale`），会得到负的 `minimum_scale` 并静默产生错误 scale。
   - `get_minimum_scale`：入参由 `int` 改为 `int64_t`，消除截断。
   - `pot_sqnr_calibrator.h` / `gru_interface.cc` / `calibration_gpu.cu`：减法前显式 `static_cast<int64_t>`，并去掉所有 `static_cast<int>(num_steps)`。
   - `quantize_ops_helper.h::calibrateQuantParams`（MINMAX 路径）：`num_steps`、`num_pos_steps`、`num_neg_steps` 统一改为 `int64_t`。

### 功能优化
1. **POT shift 越界防御断言**: 新增 `checkedShiftToInt8()`，在 POT scale 编码（`roundScaleToPowerOfTwo` / `convertToPot` / `calibrateQuantParams`）以及 bias 的 per-tensor/per-gate/per-channel rescale shift 计算处，对收窄到 `int8` 的 shift 做范围校验，超界时直接抛错，避免静默截断成错误 scale。

### 说明 / 已知限制
1. **forward 路径未改逻辑**: bias 前向通路本就是 `int32` 存储 + `int64` 累加，`qmin()/qmax()` 对 `bits>=32` 已特判返回 int32 极值，clamp 到 int32 全域为无操作，故 forward 无需为 32bit bias 改动；本次 forward 侧仅新增上述防御断言，不改变正确路径数值结果。

---

## [2026-06-22]

### 版本更新
1. **版本号更新**: 将版本号从 1.0.6 更新至 1.0.7

### 功能优化
1. **量化参数导出字段形态优化**: 导出量化参数时，`scale`、`zero_point`、`real_min`、`real_max` 等量化字段仅在存在多个元素时输出为列表；per-tensor 等单元素量化参数输出为标量，减少下游解析时对单元素列表的特殊处理。
2. **ONNX 导出 opset 支持范围放宽**: `ensure_quant_gru_onnx_registered(opset=...)` 支持 `opset>=13`，并按 opset 分别注册 symbolic；`QuantGRU` 的 ONNX forward 路径不再自行写死注册 opset 18，要求调用方在导出前显式注册与 `torch.onnx.export(opset_version=...)` 一致的 opset。

---

## [2026-06-12]

### 版本更新
1. **版本号更新**: 将版本号从 1.0.5 更新至 1.0.6

### 新功能
1. **量化值存储类型开关 `quant_storage_dtype`**: `QuantGRU` 新增 `quant_storage_dtype` 参数，可在量化前向中切换量化值的存储类型：
   - `"float32"`（默认，原行为）：量化值以 float32 存储，走 `forward_quant_float_storage`；
   - `"int32"`：量化值以 int32 存储，走纯定点核心 `forward_quant_int_storage`。
   反向传播按存储类型自动选择对应实现（int32 路径内部转 float 后复用原 backward），训练/推理两端均支持。
2. **纯定点 int 进 int 出推理接口（AIMET INT16_FIXED_EVAL 打通）**:
   - C++ 新增 `quantGRUForwardIntIO`：输入 `x_q` / `h0_q` 为上游已量化的 int32 值（同 `scale_x/zp_x`、`scale_h/zp_h` 网格），内部仅量化权重、跳过输入量化，直出 int32 `h_q`（不反量化）。
   - 新增 binding `forward_quantized_int_io`（int32 输入 → int32 `output_q`/`h_n_q`）。
   - `QuantGRU` 新增 `forward_quantized(input, hx=None)`：纯定点推理边界，仅接受已量化的 int32 输入（非整数输入抛 `TypeError`），输出恒为 int32 量化隐藏状态，满足 `(output_q - zp) * scale == 部署核 fp 输出`。
3. **AIMET 黑盒接入契约方法**: `QuantGRU` 新增 `aimet_capabilities()`、`aimet_configure(mode)`、`get_io_quant_meta()`，用于 AIMET 适配层的能力上报、模式配置与输入/输出/隐藏态量化元数据查询。
4. **有效定点 scale 解码**: 新增 binding `decode_effective_scale(scale, usePOT2)`，返回内核实际使用的（POT2/M16 编码后）有效 scale，保证 `get_io_quant_meta` 上报的 scale 与定点输出 bit-exact 一致。

### 功能优化
1. **输入/隐藏态校准改用全局 min/max（与 AIMET 对齐）**: 输入 `x` 与隐藏态 `h` 的 MinMax 校准范围统计由原先的「分时间步 EMA（decay=0.9，顺序相关）」改为顺序无关的全局 min/max。
   - 修复双向 GRU 正/反向因 EMA 顺序相关导致 `scale_x/zp_x` 不一致的问题：正/反向现共享同一输入网格，双向纯定点 int 进 int 出可达 bit-exact。
   - 与 AIMET 的全局 MinMax 校准语义一致，便于上游量化器对齐。

### 说明 / 已知限制
1. **校准跳过初始隐藏状态 h0**: MinMax 与直方图两条校准路径均跳过 `h[0]`（假设无状态 GRU 的 h0 恒为零，避免零值污染 `min_h_/max_h_`）。rxmet 实际接入路径（校准/推理均只传 input、h0 走零初始化）满足该前提。若未来出现有状态/流式 GRU 在校准阶段传入非零 h0，则需改为按需将 h0 纳入范围统计（已在代码注释中标注修法）。

---

## [2026-06-05]

### 版本更新
1. **版本号更新**: 将版本号从 1.0.4 更新至 1.0.5

### 功能优化
1. **量化参数导入/导出格式更新**: 导出改为 **scale-only**，算子级仅保留 `scale` 与 `zero_point`（及 `real_min`/`real_max`、`enc_type` 等语义字段），不再输出 `multiplier`、`shift`、`scale_encoding`。`scale` 字段统一表示运行时语义 scale（避免使用 `decode(fixed_scale)` 近似值）。导入侧优先解析 scale-only，同时继续兼容旧 `multiplier/shift` 与 `n/exp2_inv` 格式。
   ```json
   {
     "operators": {
       "input": {
         "dtype": "INT8",
         "symmetric": false,
         "scale": [0.010382890701293945],
         "zero_point": [-5],
         "enc_type": "PER_TENSOR",
         "real_min": [-1.2770955562591553],
         "real_max": [1.3705415725708008]
       }
     }
   }
   ```
2. **ONNX 导出路径更新**: 新增 `export_mode` 单节点 GRU 导出路径，通过 `ensure_quant_gru_onnx_registered(opset=18)` 注册 `custom_gru::quant_gru` / `custom_gru::quant_bigru` symbolic，并在 legacy exporter (`dynamo=False`) 下导出为标准 ONNX `GRU` 节点；支持单向/双向 GRU、`hx=None` 与显式 `hx` 输入。新增 `get_quant_gru_custom_opsets()`、`normalize_quant_gru_onnx_to_optimized_baseline()` 和 `prune_quant_gru_raw_l0_param_encodings()`，用于统一 custom opset 配置、规范化 ONNX 局部命名并清理 AIMET 原生 `*_l0` 参数 encoding。

### Bug 修复
1. 修复"对于一开始没有开启POT2模式, 中间开启了POT2模式调用了对应的接口, C++中进行了量化参数的调整, 但没有反应到python端", 数据流链路断开的bug. 修复点: 所有会改变量化参数生成语义的配置变更，都必须标记 `_quant_params_dirty=True`

---

## [2026-06-04]

### 新功能
1. **普通 scale 支持**: 量化 scale 从原先仅支持 POT2（2 的幂次）形式扩展为支持普通 scale，可使用任意 `M + shift` 组合表示量化缩放参数，提升量化参数配置的灵活性。

### Bug 修复
1. 修复 affine 模式下边界量化导致的精度回退问题。
2. 修复 LUT 表使用错误 scale 的问题。
3. 修复 `encodeMShift` 中 Q mantissa 缺少 16-bit offset 的问题。
4. 修复双向 GRU 导出 AIMET encodings 时，`weight_ih.weight` 和 `weight_hh.weight` 的 `zero_point` 未随反向分支拼接的问题；对称量化权重现在会输出与拼接后 `scale` 等长的全 0 `zero_point` 列表。

---

## [2026-04-21]

### 工程化优化
1. **GitHub Actions 工作流完善**: 新增并完善 Linux CUDA 构建、Docker CI 镜像构建和 Python 包发布流程。
2. **Docker 环境更新**: 更新 Docker 构建环境及配套说明文档，改善 CI 和本地构建的一致性。
3. **安装打包流程优化**: 更新 PyTorch 扩展的安装与打包流程，提升安装脚本的可维护性。

---

## [2026-03-24]

### 版本更新
1. **版本号更新**: 将版本号从 1.0.3 更新至 1.0.4

### Bug 修复
1. 修复双向GRU导入量化参数时反向的输入量化参数没有正确初始化的bug

---

## [2026-03-19]

### 版本更新
1. **版本号更新**: 将版本号从 1.0.2 更新至 1.0.3

### 新功能
1. **双向 GRU 的 AIMET encodings 导出/导入支持**:
   - 导出侧：`param_encodings` 中的权重/偏置量化参数按 **正向在前、反向追加在后** 的规则拼接（例如原本单向 128 个参数 -> 双向 256 个）
   - 导出侧：新增 `activation_encodings[module].internal_ops_reverse` 字段，用于承载反向方向的中间算子量化参数
   - 导入侧：支持从上述 AIMET encodings 结构回读并解析 `quant_params_reverse`

---

## [2026-03-13]

### 版本更新
1. **版本号更新**: 将版本号从 1.0.1 更新至 1.0.2

### 功能优化
1. **新增 quant_params 锁机制**: 在 `QuantGRU` 中新增 `set_quant_params_locked(locked: bool)` 接口，用于显式控制量化参数是否允许被后续校准覆盖。

---

## [2026-02-14]

### 版本更新
1. **版本号更新**: 将版本号从 1.0.0 更新至 1.0.1

### 功能优化
1. **安装脚本优化**: 改进 `setup.py` 安装脚本，现在在安装时会自动将 C++ binding 共享库文件 (`libgru_quant_shared.so`) 复制到安装目录的 `lib` 子目录中
   - 支持正常安装模式 (`pip install`) 和开发模式 (`pip install -e`) 下的自动复制
   - 通过 rpath 机制 (`$ORIGIN/lib`) 确保扩展模块在运行时能够正确找到共享库
   - 简化了安装流程，无需手动复制共享库文件

---

## [2026-02-13]

### 破坏性变更
1. **算子名称重构**: 为了提升配置的可读性和清晰度，统一重构了所有算子名称：
   - `x` → `input` (输入序列)
   - `h` → `output` (隐藏状态输出)
   - `W` → `weight_ih` (输入权重矩阵)
   - `R` → `weight_hh` (循环权重矩阵)
   - `bw` → `bias_ih` (输入偏置)
   - `br` → `bias_hh` (循环偏置)
   - 其他算子名称保持不变（如 `weight_ih_linear`, `update_gate_output` 等）
   
   **影响**: 
   - 旧的 JSON 配置文件需要更新算子名称才能正常工作
   - 代码中所有使用旧算子名称的地方（如 `adjust_quant_config("x", ...)`）需要更新为新名称
   - 请参考 `pytorch/config/gru_quant_bitwidth_config.json` 查看新的配置格式

### 功能优化
1. **取消梯度缩放优化**: 取消通过应用内部缩放梯度来解决 TF32 精度问题的优化，该优化原本用于在使用 TensorCore 加速训练时提升数值稳定性

---

## [2026-02-11]

### 功能优化
1. **量化参数导入导出与 AIMET 适配**: 量化参数导入导出功能与 AIMET 框架适配

---

## [2026-02-03]

### 性能优化
1. **Backward 开启 TensorCore 加速**: 对反向传播过程开启 TensorCore 加速，提升梯度计算性能

### 功能优化
1. **精度阈值动态缩放**: 在反向传播过程中，内部根据精度阈值进行动态缩放，确保计算精度和数值稳定性

---

## [2026-01-30]

### 新功能
1. **权重和bias支持per-tensor量化**: 支持权重和bias使用per-tensor量化和计算方式
2. **权重和bias支持per-gate量化**: 支持权重和bias使用per-gate量化和计算方式

---

## [2026-01-29]

### Bug 修复
1. 修复反向时对输入和权重没有进行量化反量化的量化误差感知的 bug

### 性能优化
1. **Forward 开启 TensorCore 加速**: 对 Forward 过程开启 TensorCore 加速，提升计算性能

---

## [2026-01-27]

### 功能优化
1. **调整移位顺序**: 优化移位操作的顺序，提升后量化精度

---

## [2026-01-26]

### Bug 修复
1. 修复梯度计算时使用了 TF32 导致梯度精度不足的 bug
2. 修复梯度计算时，被截断的值也传出了梯度的 bug

### 功能优化
1. 通过将 `roundf` 函数改为和 PyTorch 使用四舍五入逻辑相同的 `rintf` 函数来提升 PTQ 精度
2. **乘法 scale 融合优化**：将乘法操作（如 `reset_gate * weight_hh_linear`、`update_gate * h_old` 等）的结果直接对齐到目标量化 scale，省略了中间的量化步骤。该优化减少了量化误差累积（每次量化都会引入舍入误差，省略中间量化减少了一次舍入误差），提高了数值精度，同时减少了计算开销和参数管理复杂度

---

## [2026-01-15]

### 新功能
1. **量化参数导入导出**: 支持量化参数的导入和导出功能
2. **手动调整导出位宽**: 支持手动调整导出的量化参数位宽
3. **外部控制符号类型**: 支持算子位宽是 UINT 还是 INT 可以在外部 (JSON) 控制

### Bug 修复
- 修复校验时是否对称量化与是否有符号量化绑定的 bug

---

## [2025-01-13]

### 功能优化
1. **量化计算流程重构**: 从原来的在一个大 kernel 中进行 bias 的累加，改为在线性层同时进行 GEMM 和 bias 的相加，以保持与编译器和硬件实现一致
2. **统一量化参数命名格式**: 规范化量化参数的命名规则，提高代码可读性和维护性

---

## [2025-01-08]

### 新功能
1. **量化参数导入导出**: 支持量化参数的导入和导出功能
2. **手动调整导出位宽**: 支持手动调整导出的量化参数位宽
3. **外部控制符号类型**: 支持算子位宽是 UINT 还是 INT 可以在外部 (JSON) 控制

### Bug 修复
- 修复校验时是否对称量化与有符号量化绑定的 bug

---

## [2025-01-07]

### 新功能
1. **任意位宽支持**: 所有算子支持任意 bit 位宽设置

### Bug 修复
1. 修复分段拟合表全局共享导致多个 GRU 情况下 PTQ 精度下降的 bug
2. 修复校验时第一个时间步没有正确初始化的 bug
3. 修复直方图在边缘情况下的处理情况与 AIMET 实现不一致的 bug

### 性能优化
- 优化各个校验方法的速度

---

## [2024-12-31]

### 新功能
1. **灵活位宽配置**: 
   - 支持激活函数输入和输出位宽不一致
   - 支持权重和输入位宽不一致
2. **ONNX 浮点导出**: 支持 ONNX 浮点格式导出
3. **校准截断控制**: 支持校准时用截断比例控制
4. **C++ 定点 GRU 实现**: 实现 C++ 版本的纯定点 GRU 用于 Reference Model 验证

### 功能优化
- **校验流程优化**: 原来不能在 Forward 的时候同时进行校验，必须单独跑校验函数；现已优化支持同时进行

### Bug 修复
1. 修复混合精度配置下中间结果移除导致精度下降的 bug
2. 修复使用双向 GRU 时分段拟合表不匹配的 bug
3. 修复 QDQ 导出时，算子位宽与配置不匹配的 bug

---

## 版本说明

- 日期格式: `YYYY-MM-DD`
- 更新类型:
  - **新功能**: 新增的功能特性
  - **功能优化**: 对现有功能的改进和优化
  - **Bug 修复**: 修复的问题和错误
  - **性能优化**: 性能相关的改进
  - **破坏性变更**: 不向后兼容的变更（如有）
