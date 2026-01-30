#pragma once

// ============================================================================
// quantize_param_types.h - GRU 量化参数结构体定义
// ============================================================================
//
// 本文件包含 GRU 量化过程中使用的核心数据结构：
//   1. GRUQuantParams - Host 端完整量化参数（校准阶段使用）
//   2. GateQuantParams - 门计算参数（CPU/GPU 共用，推理阶段使用）
//   3. LinearQuantParamsGPU - Linear 层 per-channel 参数（GPU 版本）
//   4. LinearQuantParamsCPU - Linear 层 per-channel 参数（CPU 版本）
//
// 设计原则：
//   - 所有缩放因子均为 2 的负 n 次方：scale = 2^(-shift)
//   - 支持对称量化（zp=0）和非对称量化（zp≠0）
//   - 结构体按职责分离：GEMM 用 LinearQuantParams，门计算用 GateQuantParams
//
// 命名约定（与 optimized_quantizable_gru_2.md 文档对齐）：
//   - weight_ih_linear: W*x + bw 的输出（输入线性变换）
//   - weight_hh_linear: R*h + br 的输出（隐状态线性变换）
//   - reset_gate_input/output: reset gate 的输入/输出
//   - update_gate_input/output: update gate 的输入/输出
//   - new_gate_input/output: new gate（候选隐状态）的输入/输出
//   - mul_reset_hidden: r * weight_hh_linear 的输出
//   - mul_new_contribution: (1-u) * n 的输出
//   - mul_old_contribution: u * h 的输出
//
// ============================================================================

#include <array>
#include <vector>

#include "dev_vector.h"
#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"

// ============================================================================
// GRU 完整量化参数结构体（Host 端）
// ============================================================================

/**
 * @brief GRU 量化参数结构体（Host 端）
 *
 * 存储 GRU 网络量化过程中所有定点化/反量化所需的参数。
 * 主要在校准阶段使用，推理阶段会拆分为 GateQuantParams 和 LinearQuantParams。
 *
 * 命名约定：
 *   - shift_xxx: 缩放因子指数，scale = 2^(-shift)
 *   - zp_xxx: 零点（zero point）
 *
 * 量化公式：q = round(x / scale + zp)
 * 反量化公式：x = (q - zp) * scale
 */
struct GRUQuantParams {
    OperatorQuantConfig bitwidth_config_;  ///< 各算子的量化位宽配置

    // -------------------- 基础参数 --------------------
    int hidden_;       ///< 隐藏层大小，channel = hidden * 3
    int8_t shift_x_;   ///< 输入 x 的移位量
    int32_t zp_x_;     ///< 输入 x 的零点
    int8_t shift_h_;   ///< 隐状态 h 的移位量
    int32_t zp_h_;     ///< 隐状态 h 的零点

    // ========== 多粒度参数存储 ==========
    // Per-Tensor 参数（标量）
    int8_t shift_W_tensor_ = 0;
    int8_t shift_R_tensor_ = 0;
    int8_t shift_bw_tensor_ = 0;
    int8_t shift_br_tensor_ = 0;
    
    // Per-Gate 参数（数组：索引 0=z, 1=r, 2=g）
    std::array<int8_t, 3> shift_W_gate_ = {0, 0, 0};  // [z, r, g]
    std::array<int8_t, 3> shift_R_gate_ = {0, 0, 0};  // [z, r, g]
    std::array<int8_t, 3> shift_bw_gate_ = {0, 0, 0}; // [z, r, g]
    std::array<int8_t, 3> shift_br_gate_ = {0, 0, 0}; // [z, r, g]
    
    // Per-Channel 参数（现有字段，保持 std::vector）
    std::vector<int8_t> shift_W_;   ///< 输入权重 W 的移位量，size = hidden * 3
    std::vector<int8_t> shift_R_;   ///< 循环权重 R 的移位量，size = hidden * 3
    std::vector<int8_t> shift_bw_;  ///< 输入偏置移位量 (bias for W)
    std::vector<int8_t> shift_br_;  ///< 循环偏置移位量 (bias for R)

    // -------------------- Linear 输出参数 (GEMM+bias) --------------------
    int8_t shift_weight_ih_linear_;    ///< W*x + bw 的移位量
    int32_t zp_weight_ih_linear_;      ///< W*x + bw 的零点
    int8_t shift_weight_hh_linear_;    ///< R*h + br 的移位量
    int32_t zp_weight_hh_linear_;      ///< R*h + br 的零点

    // -------------------- 门激活函数输入参数（pre-activation）--------------------
    int8_t shift_update_gate_input_;   ///< update gate 激活前的移位量
    int32_t zp_update_gate_input_;     ///< update gate 激活前的零点
    int8_t shift_reset_gate_input_;    ///< reset gate 激活前的移位量
    int32_t zp_reset_gate_input_;      ///< reset gate 激活前的零点
    int8_t shift_new_gate_input_;      ///< new gate 激活前的移位量
    int32_t zp_new_gate_input_;        ///< new gate 激活前的零点

    // -------------------- 门激活函数输出参数（post-activation）--------------------
    int8_t shift_update_gate_output_;   ///< update gate 激活后的移位量（sigmoid 输出）
    int32_t zp_update_gate_output_;     ///< update gate 激活后的零点
    int8_t shift_reset_gate_output_;    ///< reset gate 激活后的移位量（sigmoid 输出）
    int32_t zp_reset_gate_output_;      ///< reset gate 激活后的零点
    int8_t shift_new_gate_output_;      ///< new gate 激活后的移位量（tanh 输出）
    int32_t zp_new_gate_output_;        ///< new gate 激活后的零点

    // -------------------- 中间计算参数 --------------------
    int8_t shift_mul_reset_hidden_;    ///< r * weight_hh_linear 的移位量
    int32_t zp_mul_reset_hidden_;      ///< r * weight_hh_linear 的零点

    // -------------------- 隐状态更新参数 --------------------
    int8_t shift_mul_new_contribution_;   ///< (1-u)*n 的移位量
    int32_t zp_mul_new_contribution_;     ///< (1-u)*n 的零点
    int8_t shift_mul_old_contribution_;   ///< u*h 的移位量
    int32_t zp_mul_old_contribution_;     ///< u*h 的零点

    // -------------------- LUT 表（每层独立，在 finalize_calibration 时生成）--------------------
    SigmoidLUT sigmoid_update_gate_lut_;  ///< update gate Sigmoid LUT
    SigmoidLUT sigmoid_reset_gate_lut_;   ///< reset gate Sigmoid LUT
    SigmoidLUT tanh_new_gate_lut_;        ///< new gate Tanh LUT
};

// ============================================================================
// GRU 门计算量化参数（纯标量，CPU/GPU 共用）
// ============================================================================

/**
 * @brief 门计算量化参数（纯标量，CPU/GPU 共用）
 *
 * 存储 computeUpdateGate/ResetGate/NewGate/HiddenState 等门计算函数所需的标量参数。
 * 这些参数不涉及 per-channel 数组，可以安全地在 CPU/GPU 间共享。
 *
 * 命名约定：
 *   - shift_A_to_B: 从 A 空间到 B 空间的移位量，= shift_A - shift_B
 *   - shift_xxx: 右移位数
 *   - zp_xxx: 零点
 */
struct GateQuantParams {
    // -------------------- Linear 输出零点 --------------------
    int32_t zp_weight_ih_linear_;  ///< W*x+bw 的零点
    int32_t zp_weight_hh_linear_;  ///< R*h+br 的零点
    int32_t zp_h_;                 ///< 隐状态 h 的零点

    // -------------------- Update Gate 参数 --------------------
    int32_t zp_update_gate_input_;                  ///< update gate 激活前零点
    int32_t zp_update_gate_output_;                 ///< update gate 激活后零点
    int8_t shift_weight_ih_linear_to_update_gate_input_;    ///< weight_ih_linear 到 update_gate_input 的移位
    int8_t shift_weight_hh_linear_to_update_gate_input_;    ///< weight_hh_linear 到 update_gate_input 的移位

    // -------------------- Reset Gate 参数 --------------------
    int32_t zp_reset_gate_input_;                  ///< reset gate 激活前零点
    int32_t zp_reset_gate_output_;                 ///< reset gate 激活后零点
    int8_t shift_weight_ih_linear_to_reset_gate_input_;    ///< weight_ih_linear 到 reset_gate_input 的移位
    int8_t shift_weight_hh_linear_to_reset_gate_input_;    ///< weight_hh_linear 到 reset_gate_input 的移位

    // -------------------- New Gate（候选隐状态）参数 --------------------
    int32_t zp_new_gate_input_;                  ///< new gate 激活前零点
    int32_t zp_new_gate_output_;                 ///< new gate 激活后零点
    int8_t shift_weight_ih_linear_to_new_gate_input_;      ///< weight_ih_linear 到 new_gate_input 的移位
    int8_t shift_reset_mul_hh_to_new_gate_input_;          ///< r*weight_hh_linear 直接对齐到 new_gate_input 的移位（融合）

    // -------------------- 隐状态更新参数（乘法scale融合）--------------------
    int32_t quant_one_in_update_gate_scale_;     ///< 常数 1 量化到 update_gate_output 空间的值 = 2^shift + zp
    int8_t shift_update_new_to_h_;               ///< (1-u)*n 直接对齐到 h 的移位（融合）
    int8_t shift_update_old_to_h_;               ///< u*h 直接对齐到 h 的移位（融合）

    // -------------------- 运行时配置 --------------------
    OperatorQuantConfig bitwidth_config_;  ///< 位宽配置

    // -------------------- LUT 表 --------------------
    SigmoidLUT sigmoid_update_gate_lut_;  ///< update gate Sigmoid LUT
    SigmoidLUT sigmoid_reset_gate_lut_;   ///< reset gate Sigmoid LUT
    SigmoidLUT tanh_new_gate_lut_;        ///< new gate Tanh LUT

#ifdef DEBUG
    GRUQuantParams test;  ///< 保存完整量化参数用于调试
#endif
};

// ============================================================================
// Linear 层量化参数（GPU/CPU 分离版本）
// ============================================================================

/**
 * @brief Linear 层量化参数（GPU 版本，使用 dev::vector）
 *
 * 存储 GEMM+bias 融合计算所需的参数，支持 per-tensor、per-gate、per-channel 三种粒度。
 * 仅在 ComputeLinearX/ComputeLinearH 等 GEMM 函数中使用。
 */
struct LinearQuantParamsGPU {
    int32_t zp_x_;  ///< 输入 x 的零点
    int32_t zp_h_;  ///< 隐状态 h 的零点

    // 粒度配置（从 OperatorQuantConfig 复制，用于 kernel 中判断）
    int8_t W_granularity_;   ///< W 的量化粒度：0=PER_TENSOR, 1=PER_GATE, 2=PER_CHANNEL
    int8_t R_granularity_;   ///< R 的量化粒度
    int8_t bw_granularity_;  ///< bw 的量化粒度
    int8_t br_granularity_;  ///< br 的量化粒度
    int hidden_size_;        ///< 隐藏层大小（用于计算 gate_idx）

    // Per-tensor 参数（标量，rescale 后的值，通过参数内存传递，访问快）
    int8_t shift_gemm_x_tensor_;   ///< W*x rescale per-tensor shift（已计算 rescale）
    int8_t shift_gemm_h_tensor_;   ///< R*h rescale per-tensor shift（已计算 rescale）
    int8_t shift_bw_tensor_;       ///< bw rescale per-tensor shift（已计算 rescale）
    int8_t shift_br_tensor_;       ///< br rescale per-tensor shift（已计算 rescale）

    // Per-gate 参数（3个值，rescale 后的值，通过参数内存传递，访问快）
    int8_t shift_gemm_x_gate_[3];   ///< W*x rescale per-gate shift [z, r, g]（已计算 rescale）
    int8_t shift_gemm_h_gate_[3];   ///< R*h rescale per-gate shift [z, r, g]（已计算 rescale）
    int8_t shift_bw_gate_[3];       ///< bw rescale per-gate shift [z, r, g]（已计算 rescale）
    int8_t shift_br_gate_[3];       ///< br rescale per-gate shift [z, r, g]（已计算 rescale）

    // Per-channel 参数（数组，全局内存，PER_CHANNEL 粒度时使用）
    dev::vector<int8_t> shift_gemm_x_to_weight_ih_linear_;  ///< W*x per-channel 移位
    dev::vector<int8_t> shift_bw_to_weight_ih_linear_;      ///< bw per-channel 移位
    dev::vector<int8_t> shift_gemm_h_to_weight_hh_linear_;  ///< R*h per-channel 移位
    dev::vector<int8_t> shift_br_to_weight_hh_linear_;      ///< br per-channel 移位

#ifdef DEBUG
    dev::vector<int8_t> shift_bw_;  ///< bw 移位量（调试用）
    dev::vector<int8_t> shift_br_;  ///< br 移位量（调试用）
#endif
};

/**
 * @brief Linear 层量化参数（CPU 版本，使用 std::vector）
 *
 * 与 LinearQuantParamsGPU 结构相同，但使用 std::vector 管理内存。
 */
struct LinearQuantParamsCPU {
    int32_t zp_x_;  ///< 输入 x 的零点
    int32_t zp_h_;  ///< 隐状态 h 的零点

    std::vector<int8_t> shift_gemm_x_to_weight_ih_linear_;  ///< W*x per-channel 移位
    std::vector<int8_t> shift_bw_to_weight_ih_linear_;      ///< bw per-channel 移位
    std::vector<int8_t> shift_gemm_h_to_weight_hh_linear_;  ///< R*h per-channel 移位
    std::vector<int8_t> shift_br_to_weight_hh_linear_;      ///< br per-channel 移位

#ifdef DEBUG
    std::vector<int8_t> shift_bw_;  ///< bw 移位量（调试用）
    std::vector<int8_t> shift_br_;  ///< br 移位量（调试用）
#endif
};

// ============================================================================
// 浮点存储版量化参数（用于 gru_forward_gpu_quant_fp.cu）
// ============================================================================
//
// 与整数版本的区别：
//   - 所有量化值使用 float 存储（值仍是定点整数）
//   - shift 预处理为除数：divisor = 2^shift
//   - 不包含 LUT（使用 real_sigmoid/real_tanh）
//
// ============================================================================

/**
 * @brief 门计算量化参数（浮点存储版本）
 *
 * shift 预处理为除数：divisor = 2^shift
 * 零点使用 float 存储
 * 不包含 LUT（使用 real_sigmoid/real_tanh）
 */
struct GateQuantParamsFP {
    // -------------------- Linear 输出零点 --------------------
    float zp_weight_ih_linear_;   ///< W*x+bw 的零点
    float zp_weight_hh_linear_;   ///< R*h+br 的零点
    float zp_h_;                  ///< 隐状态 h 的零点

    // -------------------- Update Gate 参数 --------------------
    float zp_update_gate_input_;                      ///< update gate 激活前零点
    float zp_update_gate_output_;                     ///< update gate 激活后零点
    float inv_div_weight_ih_linear_to_update_gate_input_; ///< 1.0f / (2^shift)，用于乘法替代除法
    float inv_div_weight_hh_linear_to_update_gate_input_; ///< 1.0f / (2^shift)，用于乘法替代除法

    // -------------------- Reset Gate 参数 --------------------
    float zp_reset_gate_input_;                       ///< reset gate 激活前零点
    float zp_reset_gate_output_;                      ///< reset gate 激活后零点
    float inv_div_weight_ih_linear_to_reset_gate_input_;  ///< 1.0f / (2^shift)，用于乘法替代除法
    float inv_div_weight_hh_linear_to_reset_gate_input_;  ///< 1.0f / (2^shift)，用于乘法替代除法

    // -------------------- New Gate 参数 --------------------
    float zp_new_gate_input_;                         ///< new gate 激活前零点
    float zp_new_gate_output_;                        ///< new gate 激活后零点
    float inv_div_weight_ih_linear_to_new_gate_input_;    ///< 1.0f / (2^shift)，用于乘法替代除法
    float inv_div_reset_mul_hh_to_new_gate_input_;        ///< 1.0f / (2^shift)，r*hh 到 new_gate_input 的倒数

    // -------------------- 隐状态更新参数 --------------------
    float quant_one_in_update_gate_scale_;  ///< 常数 1 量化到 update_gate_output 空间的值 = 2^shift + zp
    float inv_div_update_old_to_h_;             ///< 1.0f / (2^shift)，u*h 到 h 的倒数
    float inv_div_new_gate_output_to_h_;       ///< 1.0f / (2^shift)，new_gate_output 到 h 的倒数（用于先将new_gate对齐到h scale）

    // -------------------- 激活函数 scale（用于 real_sigmoid/real_tanh）--------------------
    float scale_update_gate_input_;   ///< = 2^(-shift)，反量化 scale
    float scale_update_gate_output_;  ///< = 2^(-shift)，量化 scale
    float scale_reset_gate_input_;    ///< = 2^(-shift)
    float scale_reset_gate_output_;   ///< = 2^(-shift)
    float scale_new_gate_input_;      ///< = 2^(-shift)
    float scale_new_gate_output_;     ///< = 2^(-shift)
};

/**
 * @brief Linear Rescale 参数（用于 kernel 和类成员，统一管理所有量化参数）
 * 
 * 用于减少 kernel 参数数量，将所有 BiasRescale 相关参数打包到一个结构体中
 * 同时作为类成员变量，直接管理数据，避免维护两套数据
 * 注意：粒度配置通过 OperatorQuantConfig 单独传递，不在此结构体中
 */
struct LinearRescaleParamsFP {
    // 基础零点参数
    float zp_x_;                 ///< 输入 x 的零点
    float zp_h_;                 ///< 隐状态 h 的零点
    // weight_ih_linear 相关参数
    const float *W_sum_mul_x_zp;                    ///< [3*hidden] 预计算的 sum(W)*zp_x（运行时设置）
    const float *inv_div_gemm_x_to_weight_ih_linear_;  ///< [3*hidden] PER_CHANNEL: 1.0f / (2^shift_gemm_x)，从GEMM_x空间到weight_ih_linear空间（指向 dev::vector）
    const float *inv_div_bw_to_gemm_x_;              ///< [3*hidden] PER_CHANNEL: 1.0f / (2^shift_bw)，从bw空间到GEMM_x空间（指向 dev::vector）
    float zp_weight_ih_linear_;                     ///< weight_ih_linear 输出零点
    // 注意：output_bw_ih_ 和 output_bw_hh_ 已移除，直接从 OperatorQuantConfig 中获取
    
    // Per-tensor 参数（标量，通过参数内存传递，访问快）
    float inv_div_gemm_x_tensor_;   ///< W*x rescale per-tensor 倒数
    float inv_div_gemm_h_tensor_;   ///< R*h rescale per-tensor 倒数
    float inv_div_bw_tensor_;       ///< bw rescale per-tensor 倒数
    float inv_div_br_tensor_;       ///< br rescale per-tensor 倒数

    // Per-gate 参数（3个值，通过参数内存传递，访问快）
    float inv_div_gemm_x_gate_[3];   ///< W*x rescale per-gate 倒数 [z, r, g]
    float inv_div_gemm_h_gate_[3];   ///< R*h rescale per-gate 倒数 [z, r, g]
    float inv_div_bw_gate_[3];       ///< bw rescale per-gate 倒数 [z, r, g]
    float inv_div_br_gate_[3];       ///< br rescale per-gate 倒数 [z, r, g]
    
    // weight_hh_linear 相关参数
    const float *R_sum_mul_h_zp;                    ///< [3*hidden] 预计算的 sum(R)*zp_h（运行时设置）
    const float *inv_div_gemm_h_to_weight_hh_linear_;  ///< [3*hidden] PER_CHANNEL: 1.0f / (2^shift_gemm_h)，从GEMM_h空间到weight_hh_linear空间（指向 dev::vector）
    const float *inv_div_br_to_gemm_h_;              ///< [3*hidden] PER_CHANNEL: 1.0f / (2^shift_br)，从br空间到GEMM_h空间（指向 dev::vector）
    float zp_weight_hh_linear_;                     ///< weight_hh_linear 输出零点
    // 注意：output_bw_ih_ 和 output_bw_hh_ 已移除，直接从 OperatorQuantConfig 中获取
};
