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

    // -------------------- 权重参数（per-channel）--------------------
    std::vector<int8_t> shift_W_;   ///< 输入权重 W 的移位量，size = hidden * 3
    std::vector<int8_t> shift_R_;   ///< 循环权重 R 的移位量，size = hidden * 3

    // -------------------- 偏置参数（per-channel）--------------------
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
    int8_t shift_reset_gate_mul_hh_to_mul_reset_hidden_;   ///< r*weight_hh_linear 的移位
    int32_t zp_mul_reset_hidden_;                ///< r*weight_hh_linear 的零点
    int8_t shift_weight_ih_linear_to_new_gate_input_;      ///< weight_ih_linear 到 new_gate_input 的移位
    int8_t shift_mul_reset_hidden_to_new_gate_input_;      ///< mul_reset_hidden 到 new_gate_input 的移位

    // -------------------- 隐状态更新参数 --------------------
    int32_t quant_one_in_update_gate_scale_;     ///< 常数 1 在 update_gate_output 量化空间的表示
    int32_t zp_mul_new_contribution_;            ///< (1-u)*n 的零点
    int8_t shift_update_new_to_mul_new_contribution_;    ///< (1-u)*n 的移位
    int32_t zp_mul_old_contribution_;            ///< u*h 的零点
    int8_t shift_update_h_to_mul_old_contribution_;      ///< u*h 的移位
    int8_t shift_mul_new_contribution_to_h_;     ///< mul_new_contribution 到 h 的移位
    int8_t shift_mul_old_contribution_to_h_;     ///< mul_old_contribution 到 h 的移位

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
 * 存储 GEMM+bias 融合计算所需的 per-channel 参数。
 * 仅在 ComputeLinearX/ComputeLinearH 等 GEMM 函数中使用。
 */
struct LinearQuantParamsGPU {
    int32_t zp_x_;  ///< 输入 x 的零点
    int32_t zp_h_;  ///< 隐状态 h 的零点

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
