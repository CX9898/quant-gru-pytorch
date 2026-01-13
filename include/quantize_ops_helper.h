#pragma once

// ============================================================================
// quantize_ops_helper.h - GRU 量化 CPU/GPU 共用函数
// ============================================================================
//
// 本文件包含：
//   1. CPU/GPU 共用的内联函数（__host__ __device__ __forceinline__）
//   2. 量化/反量化基础操作函数
//   3. 分段线性近似函数（Sigmoid/Tanh LUT）
//   4. GRU 门计算模板函数
//   5. 量化参数校准函数
//   6. 调试与工具函数
//
// 量化参数结构体定义已移至 quantize_param_types.h
//
// 设计原则：
//   - 所有缩放因子均为 2 的负 n 次方：scale = 2^(-exp2_inv)
//   - 支持对称量化（zp=0）和非对称量化（zp≠0）
//   - CPU/GPU 共用函数使用 __host__ __device__ __forceinline__ 标记，确保行为一致
//
// ============================================================================

#include "cuda_compat.h"
#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "gru_quantization_ranges.h"
#include "histogram_collector.h"  // for get_minimum_scale
#include "quantize_param_types.h" // 量化参数结构体定义

// #define DEBUG

/**
 * @brief 生成分段线性量化查找表并存储到参数中
 *
 * 将 LUT 存储到 GRUQuantParams 中，避免全局 __constant__ 内存覆盖问题。
 * 在 finalize_calibration 时调用一次，然后在 setRescaleParam 时复制到 GateQuantParams。
 *
 * @param params GRU 量化参数，会被修改以存储生成的 LUT
 */
void generate_piecewise_linear_lut_to_params(GRUQuantParams &params);

// ============================================================================
// CPU/GPU 共用基础运算函数
// ============================================================================

/**
 * @brief 计算 2 的负 n 次方缩放因子（CPU/GPU 共用）
 *
 * scale = 2^(-exp2_inv)
 *
 * 使用 ldexpf 直接操作浮点指数，比除法更高效。
 *
 * @param exp2_inv 缩放因子指数
 * @return 缩放因子 scale = 2^(-exp2_inv)
 */
__host__ __device__ __forceinline__ float exp2_scale(int8_t exp2_inv) {
    return ldexpf(1.0f, -static_cast<int>(exp2_inv));
}

/**
 * @brief 带四舍五入的右移操作（int32_t 版本）
 *
 * 实现 round(x / 2^n) 的定点运算，支持正负移位。
 *
 * @param x 被移位的值
 * @param n 移位量（正数右移，负数或零左移）
 * @return 移位后的结果
 *
 * @note 对负数采用向零舍入（round toward zero）
 */
__host__ __device__ __forceinline__ int32_t rshift_round(int32_t x, int8_t n) {
    if (n <= 0) return x << (-n);

    const int32_t offset = 1 << (n - 1);
    if (x >= 0) {
        return (x + offset) >> n;
    } else {
        return -((-x + offset) >> n);  // 向零舍入
    }
}

/**
 * @brief 带四舍五入的右移操作（int64_t 版本）
 *
 * 用于处理 16 位量化时可能超出 int32 范围的乘积。
 */
__host__ __device__ __forceinline__ int64_t rshift_round(int64_t x, int8_t n) {
    if (n <= 0) return x << (-n);

    const int64_t offset = static_cast<int64_t>(1) << (n - 1);
    if (x >= 0) {
        return (x + offset) >> n;
    } else {
        return -((-x + offset) >> n);  // 向零舍入
    }
}

// ============================================================================
// CPU/GPU 共用饱和截断函数
// ============================================================================

/**
 * @brief 将 int32_t 饱和截断到指定类型范围（模板版本）
 *
 * 编译时确定目标类型，零运行时开销。
 *
 * @tparam T 目标类型（int8_t, int16_t, int32_t, uint8_t, uint16_t）
 * @param x 输入值
 * @return 截断后的值
 */
template <typename T>
__host__ __device__ __forceinline__ T clamp_to_type(int32_t x);

template <>
__host__ __device__ __forceinline__ int8_t clamp_to_type<int8_t>(int32_t x) {
    return static_cast<int8_t>((x < -128) ? -128 : ((x > 127) ? 127 : x));
}

template <>
__host__ __device__ __forceinline__ int16_t clamp_to_type<int16_t>(int32_t x) {
    return static_cast<int16_t>((x < -32768) ? -32768 : ((x > 32767) ? 32767 : x));
}

template <>
__host__ __device__ __forceinline__ int32_t clamp_to_type<int32_t>(int32_t x) {
    return x;  // int32_t 无需截断
}

template <>
__host__ __device__ __forceinline__ uint8_t clamp_to_type<uint8_t>(int32_t x) {
    return static_cast<uint8_t>((x < 0) ? 0 : ((x > 255) ? 255 : x));
}

template <>
__host__ __device__ __forceinline__ uint16_t clamp_to_type<uint16_t>(int32_t x) {
    return static_cast<uint16_t>((x < 0) ? 0 : ((x > 65535) ? 65535 : x));
}

/**
 * @brief 按任意位宽饱和截断
 *
 * 适用于位宽在运行时确定的场景，支持 1-31 位任意位宽。
 *
 * @param val 输入值
 * @param bw 目标位宽配置
 * @return 截断后的值（始终返回 int32_t，但值已在目标范围内）
 */
__host__ __device__ __forceinline__ int32_t clamp_by_bitwidth(int32_t val, QuantBitWidth bw) {
    int32_t lo = bw.qmin();
    int32_t hi = bw.qmax();
    return (val < lo) ? lo : ((val > hi) ? hi : val);
}

// ============================================================================
// 分段线性近似函数（CPU/GPU 共用）
// ============================================================================
//
// 【原理】将非线性函数（Sigmoid/Tanh）在每个分段内用线性函数 y = b*x + c 近似
//
// 【量化公式】q_y = (q_b * (q_x - zp_x)) >> n_BX_total + term_c_precomputed
//
// 【计算流程】
//   1. find_segment: 根据输入找到所属分段
//   2. x_offset = q_x - zp_x: 去零点
//   3. bw = q_b * x_offset: 乘以斜率（INT64 避免溢出）
//   4. term_bx = bw >> n_BX_total: 重缩放
//   5. q_y = term_bx + term_c_precomputed: 加上预计算的截距项
//
// ============================================================================

/**
 * @brief 查找输入所属的分段索引
 *
 * @param q_x 量化输入值
 * @param segments 分段参数数组（NUM_SEGMENTS 个元素）
 * @return 分段索引 [0, NUM_SEGMENTS-1]
 */
__host__ __device__ __forceinline__ int find_segment(int32_t q_x, const SegmentParams *segments) {
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        if (q_x < segments[i].threshold) {
            return i;
        }
    }
    return NUM_SEGMENTS - 1;
}

/**
 * @brief 分段线性近似核心函数（不做饱和截断）
 *
 * @param q_x 量化输入值
 * @param lut 查找表（包含分段参数和量化参数）
 * @return 近似结果（int32_t，未截断）
 */
__host__ __device__ __forceinline__ int32_t piecewise_linear_raw(int32_t q_x,
                                                                   const SigmoidLUT &lut) {
    int seg_id = find_segment(q_x, lut.segments);
    const SegmentParams &seg = lut.segments[seg_id];

    int32_t x_offset = q_x - lut.zp_x;
    int64_t bx_64 = static_cast<int64_t>(seg.q_b) * static_cast<int64_t>(x_offset);

    int32_t term_bx = (seg.n_BX_total >= 0)
                          ? static_cast<int32_t>(rshift_round(bx_64, seg.n_BX_total))
                          : static_cast<int32_t>(bx_64 << (-seg.n_BX_total));

    return term_bx + seg.term_c_precomputed;
}

/**
 * @brief 分段线性近似函数（带输入/输出饱和截断）
 *
 * @param q_x 量化输入值
 * @param lut 查找表
 * @param pre_bw 输入位宽（用于输入截断）
 * @param out_bw 输出位宽（用于输出截断）
 * @return 近似结果（已截断到输出范围）
 */
__host__ __device__ __forceinline__ int32_t piecewise_linear(int32_t q_x, const SigmoidLUT &lut,
                                                              QuantBitWidth pre_bw,
                                                              QuantBitWidth out_bw) {
    int32_t q_x_clamped = clamp_by_bitwidth(q_x, pre_bw);
    int32_t result = piecewise_linear_raw(q_x_clamped, lut);
    return clamp_by_bitwidth(result, out_bw);
}

// ============================================================================
// GRU 门计算模板函数 (CPU/GPU 共用)
// ============================================================================

// 调试辅助函数
#if defined(DEBUG_QUANT) || defined(DEBUG_QUANT_DETAIL)
__host__ __device__ __forceinline__ float sigmoid_fp(float x) {
#ifdef __CUDA_ARCH__
    return 1.0f / (1.0f + expf(-x));
#else
    return 1.0f / (1.0f + std::exp(-x));
#endif
}

__host__ __device__ __forceinline__ float tanh_fp(float x) {
#ifdef __CUDA_ARCH__
    return tanhf(x);
#else
    return std::tanh(x);
#endif
}
#endif  // DEBUG_QUANT || DEBUG_QUANT_DETAIL

/**
 * @brief 计算更新门 update_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 * 
 * @param weight_ih_linear  输入 Linear 变换结果 W*x + bw
 * @param weight_hh_linear  隐状态 Linear 变换结果 R*h + br
 * @param params    门计算参数
 * @param debug_idx 调试索引，-1 表示不输出调试信息
 */
__host__ __device__ __forceinline__ 
int32_t computeUpdateGate(int32_t weight_ih_linear, int32_t weight_hh_linear, const GateQuantParams &params,
                 [[maybe_unused]] int debug_idx = -1) {
    // 重缩放到 update_gate_input 空间
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, params.shift_weight_ih_linear_to_update_gate_input_);
    const int32_t hh_shifted = rshift_round(weight_hh_linear - params.zp_weight_hh_linear_, params.shift_weight_hh_linear_to_update_gate_input_);

    const int32_t update_gate_input = ih_shifted + hh_shifted + params.zp_update_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    const int32_t update_gate = piecewise_linear(update_gate_input, params.sigmoid_update_gate_lut_, bw_cfg.update_gate_input_, bw_cfg.update_gate_output_);

#ifdef DEBUG_QUANT
    if (debug_idx == 0) {
        float update_gate_input_fp = (float)(update_gate_input - params.zp_update_gate_input_) /
                         (float)(1 << params.test.shift_update_gate_input_);
        float update_gate_fp = (float)(update_gate - params.zp_update_gate_output_) /
                     (float)(1 << params.test.shift_update_gate_output_);
        printf("[QUANT_I32] computeUpdateGate: update_gate_input_q=%d, update_gate_input_fp=%.6f, update_gate_q=%d, update_gate_fp=%.6f\n", 
               update_gate_input, update_gate_input_fp, update_gate, update_gate_fp);
    }
#endif

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0 && debug_idx < 3) {
        float update_gate_input_fp = (float)(update_gate_input - params.zp_update_gate_input_) /
                         (float)(1 << params.test.shift_update_gate_input_);
        float update_gate_quant_fp = (float)(update_gate - params.zp_update_gate_output_) /
                           (float)(1 << params.test.shift_update_gate_output_);
        float update_gate_theory = sigmoid_fp(update_gate_input_fp);
        float error = update_gate_quant_fp - update_gate_theory;
        printf("[UpdateGate] idx=%d input_q=%d input_fp=%.4f | output_q=%d output_fp=%.4f | theory=%.4f | err=%.6f\n",
               debug_idx, update_gate_input, update_gate_input_fp, update_gate, update_gate_quant_fp, update_gate_theory, error);
    }
#endif

    return update_gate;
}

/**
 * @brief 计算重置门 reset_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 */
__host__ __device__ __forceinline__ 
int32_t computeResetGate(int32_t weight_ih_linear, int32_t weight_hh_linear, const GateQuantParams &params,
                 [[maybe_unused]] int debug_idx = -1) {
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, params.shift_weight_ih_linear_to_reset_gate_input_);
    const int32_t hh_shifted = rshift_round(weight_hh_linear - params.zp_weight_hh_linear_, params.shift_weight_hh_linear_to_reset_gate_input_);

    const int32_t reset_gate_input = ih_shifted + hh_shifted + params.zp_reset_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    const int32_t reset_gate = piecewise_linear(reset_gate_input, params.sigmoid_reset_gate_lut_, bw_cfg.reset_gate_input_, bw_cfg.reset_gate_output_);

#ifdef DEBUG_QUANT
    if (debug_idx == 0) {
        float reset_gate_input_fp = (float)(reset_gate_input - params.zp_reset_gate_input_) /
                         (float)(1 << params.test.shift_reset_gate_input_);
        float reset_gate_fp = (float)(reset_gate - params.zp_reset_gate_output_) /
                     (float)(1 << params.test.shift_reset_gate_output_);
        printf("[QUANT_I32] computeResetGate: reset_gate_input_q=%d, reset_gate_input_fp=%.6f, reset_gate_q=%d, reset_gate_fp=%.6f\n", 
               reset_gate_input, reset_gate_input_fp, reset_gate, reset_gate_fp);
    }
#endif

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0 && debug_idx < 3) {
        float reset_gate_input_fp = (float)(reset_gate_input - params.zp_reset_gate_input_) /
                         (float)(1 << params.test.shift_reset_gate_input_);
        float reset_gate_quant_fp = (float)(reset_gate - params.zp_reset_gate_output_) /
                           (float)(1 << params.test.shift_reset_gate_output_);
        float reset_gate_theory = sigmoid_fp(reset_gate_input_fp);
        float error = reset_gate_quant_fp - reset_gate_theory;
        printf("[ResetGate] idx=%d input_q=%d input_fp=%.4f | output_q=%d output_fp=%.4f | theory=%.4f | err=%.6f\n",
               debug_idx, reset_gate_input, reset_gate_input_fp, reset_gate, reset_gate_quant_fp, reset_gate_theory, error);
    }
#endif

    return reset_gate;
}

/**
 * @brief 计算候选门 new_gate = tanh(weight_ih_linear + reset_gate * weight_hh_linear)
 * 
 * @param weight_ih_linear    输入 Linear 变换结果
 * @param weight_hh_linear    隐状态 Linear 变换结果
 * @param reset_gate          重置门输出
 * @param weight_hh_linear_g  [out] 中间结果，用于存储到 v（训练时反向传播需要）
 */
__host__ __device__ __forceinline__ 
int32_t computeNewGate(int32_t weight_ih_linear, int32_t weight_hh_linear, int32_t reset_gate,
                 const GateQuantParams &params, int32_t &weight_hh_linear_g,
                 [[maybe_unused]] int debug_idx = -1) {
    // Linear 融合后，weight_hh_linear 就是 R*h + br
    weight_hh_linear_g = weight_hh_linear;

    // 计算 reset_gate * weight_hh_linear (即 mul_reset_hidden)
    const int64_t r_diff = static_cast<int64_t>(reset_gate) - params.zp_reset_gate_output_;
    const int64_t hh_diff = static_cast<int64_t>(weight_hh_linear_g) - params.zp_weight_hh_linear_;
    const int64_t reset_hidden_mul = r_diff * hh_diff;

    int32_t mul_reset_hidden = static_cast<int32_t>(rshift_round(reset_hidden_mul, params.shift_reset_gate_mul_hh_to_mul_reset_hidden_)) +
                 params.zp_mul_reset_hidden_;
    mul_reset_hidden = clamp_by_bitwidth(mul_reset_hidden, params.bitwidth_config_.mul_reset_hidden_);

    // 计算 new_gate_input = weight_ih_linear + mul_reset_hidden
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, params.shift_weight_ih_linear_to_new_gate_input_);
    const int32_t rh_shifted = rshift_round(mul_reset_hidden - params.zp_mul_reset_hidden_, params.shift_mul_reset_hidden_to_new_gate_input_);

    const int32_t new_gate_input = ih_shifted + rh_shifted + params.zp_new_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    const int32_t new_gate = piecewise_linear(new_gate_input, params.tanh_new_gate_lut_, bw_cfg.new_gate_input_, bw_cfg.new_gate_output_);

#ifdef DEBUG_QUANT
    if (debug_idx == 0) {
        float hh_fp = (float)(weight_hh_linear_g - params.zp_weight_hh_linear_) /
                      (float)(1 << params.test.shift_weight_hh_linear_);
        float rh_fp = (float)(mul_reset_hidden - params.zp_mul_reset_hidden_) / 
                      (float)(1 << params.test.shift_mul_reset_hidden_);
        float new_gate_input_fp = (float)(new_gate_input - params.zp_new_gate_input_) /
                         (float)(1 << params.test.shift_new_gate_input_);
        float new_gate_fp = (float)(new_gate - params.zp_new_gate_output_) /
                     (float)(1 << params.test.shift_new_gate_output_);
        printf("[QUANT_I32] computeNewGate: hh_fp=%.6f, rh_fp=%.6f, new_gate_input_fp=%.6f, new_gate_fp=%.6f\n",
               hh_fp, rh_fp, new_gate_input_fp, new_gate_fp);
    }
#endif

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0 && debug_idx < 3) {
        float new_gate_input_fp = (float)(new_gate_input - params.zp_new_gate_input_) /
                         (float)(1 << params.test.shift_new_gate_input_);
        float new_gate_quant_fp = (float)(new_gate - params.zp_new_gate_output_) /
                           (float)(1 << params.test.shift_new_gate_output_);
        float new_gate_theory = tanh_fp(new_gate_input_fp);
        float error = new_gate_quant_fp - new_gate_theory;
        printf("[NewGate] idx=%d input_q=%d input_fp=%.4f | output_q=%d output_fp=%.4f | theory=%.4f | err=%.6f\n",
               debug_idx, new_gate_input, new_gate_input_fp, new_gate, new_gate_quant_fp, new_gate_theory, error);
    }
#endif

    return new_gate;
}

/**
 * @brief 计算隐藏状态 h_new = update_gate * h_old + (1 - update_gate) * new_gate
 */
__host__ __device__ __forceinline__ 
int32_t computeHiddenState(int32_t update_gate, int32_t new_gate, int32_t h_old, const GateQuantParams &params,
                 [[maybe_unused]] int debug_idx = -1) {
    // 计算 mul_old_contribution = update_gate * h_old
    const int64_t u_diff = static_cast<int64_t>(update_gate) - params.zp_update_gate_output_;
    const int64_t h_diff = static_cast<int64_t>(h_old) - params.zp_h_;
    const int64_t old_contribution_mul = u_diff * h_diff;

    int32_t mul_old_contribution = static_cast<int32_t>(
        rshift_round(old_contribution_mul, params.shift_update_h_to_mul_old_contribution_)) +
        params.zp_mul_old_contribution_;
    mul_old_contribution = clamp_by_bitwidth(mul_old_contribution, params.bitwidth_config_.mul_old_contribution_);

    // 计算 mul_new_contribution = (1 - update_gate) * new_gate
    const int64_t one_minus_u = static_cast<int64_t>(params.quant_one_in_update_gate_scale_) - update_gate;
    const int64_t n_diff = static_cast<int64_t>(new_gate) - params.zp_new_gate_output_;
    const int64_t new_contribution_mul = one_minus_u * n_diff;

    int32_t mul_new_contribution = static_cast<int32_t>(
        rshift_round(new_contribution_mul, params.shift_update_new_to_mul_new_contribution_)) +
        params.zp_mul_new_contribution_;
    mul_new_contribution = clamp_by_bitwidth(mul_new_contribution, params.bitwidth_config_.mul_new_contribution_);

    // 计算 h_new = mul_old_contribution + mul_new_contribution
    const int32_t h_new =
        rshift_round(mul_old_contribution - params.zp_mul_old_contribution_, params.shift_mul_old_contribution_to_h_) +
        rshift_round(mul_new_contribution - params.zp_mul_new_contribution_, params.shift_mul_new_contribution_to_h_) +
        params.zp_h_;

    const int32_t h = clamp_by_bitwidth(h_new, params.bitwidth_config_.h_);

#ifdef DEBUG_QUANT
    if (debug_idx == 0) {
        float u_fp = (float)(update_gate - params.zp_update_gate_output_) /
                     (float)(1 << params.test.shift_update_gate_output_);
        float n_fp = (float)(new_gate - params.zp_new_gate_output_) /
                     (float)(1 << params.test.shift_new_gate_output_);
        float h_old_fp = (float)(h_old - params.zp_h_) / 
                         (float)(1 << params.test.shift_h_);
        float h_fp = (float)(h - params.zp_h_) / 
                     (float)(1 << params.test.shift_h_);
        printf("[QUANT_I32] computeHiddenState: u_fp=%.6f, n_fp=%.6f, h_old_fp=%.6f, h_new_fp=%.6f\n", 
               u_fp, n_fp, h_old_fp, h_fp);
    }
#endif

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0 && debug_idx < 3) {
        float u_fp = (float)(update_gate - params.zp_update_gate_output_) /
                     (float)(1 << params.test.shift_update_gate_output_);
        float n_fp = (float)(new_gate - params.zp_new_gate_output_) /
                     (float)(1 << params.test.shift_new_gate_output_);
        float h_old_fp = (float)(h_old - params.zp_h_) / 
                         (float)(1 << params.test.shift_h_);
        float h_quant_fp = (float)(h - params.zp_h_) / 
                           (float)(1 << params.test.shift_h_);
        printf("[HiddenState] idx=%d u_fp=%.4f n_fp=%.4f h_old_fp=%.4f | h_q=%d h_fp=%.4f\n",
               debug_idx, u_fp, n_fp, h_old_fp, h, h_quant_fp);
    }
#endif

    return h;
}

// ============================================================================
// GPU Kernel 函数声明
// ============================================================================

/**
 * @brief 计算权重矩阵列和乘以零点（用于 GEMM 零点补偿）
 *
 * weight_sum[j] = sum_i(W_q[i,j]) * zp >> n[j]
 *
 * @tparam T 权重类型（int8_t/int16_t）
 * @param W_q 量化权重矩阵 [out_dim, in_dim]
 * @param weight_sum 输出数组 [out_dim]
 * @param zp 输入零点
 * @param n per-channel 重缩放移位
 * @param out_dim 输出维度 (M)
 * @param in_dim 输入维度 (K)
 * @param stream CUDA 流
 */
template <typename T>
void computeWeightSumMulzp(const T *W_q, int64_t *weight_sum, int32_t zp,
                           const int8_t *__restrict__ n, int out_dim, int in_dim,
                           cudaStream_t stream = 0);

/// @brief int32_t 输出版本（适用于 8 位量化，不会溢出）
template <typename T>
void computeWeightSumMulzp(const T *W_q, int32_t *weight_sum, int32_t zp,
                           const int8_t *__restrict__ n, int out_dim, int in_dim,
                           cudaStream_t stream = 0);

/**
 * @brief 应用零点补偿到 2D GEMM 输出
 *
 * Y[i,j] -= weight_sum[i] * x_zp[j]
 */
void applyZeroPointCompensation2D(int32_t *Y_int32, const int32_t *weight_sum, const int32_t *x_zp,
                                  int out_dim, int batch_size, cudaStream_t stream = 0);

// ============================================================================
// 量化参数校准与量化/反量化函数
// ============================================================================

/**
 * @brief 量化参数校准函数（与 AIMET MinMaxEncodingAnalyzer.compute_encodings_from_stats 完全一致）
 *
 * 根据数据范围和位宽配置计算量化参数，缩放因子对齐到 2 的负 n 次方。
 * 
 * 与 AIMET 的一致性:
 *   1. num_steps 检查
 *   2. 使用 get_minimum_scale(num_steps)
 *   3. 确保 0 在范围内
 *   4. 范围扩展逻辑（对称/非对称分别处理）
 *   5. 对称量化 delta 计算（区分 pos/neg steps）
 *   6. 2-bit 特殊处理
 *   7. Inf 保护
 *
 * @param[in] orig_min 原始数据最小值
 * @param[in] orig_max 原始数据最大值
 * @param[in] bw 位宽配置（含位数和符号）
 * @param[in] is_symmetric 是否对称量化
 * @param[out] aligned_min 对齐后的最小值
 * @param[out] aligned_max 对齐后的最大值
 * @param[out] exp2_inv 缩放因子指数，scale = 2^(-exp2_inv)
 * @param[out] zp 零点（对称量化时为 0）
 * @param[in] name 调试用名称（可选）
 */
inline void calibrateQuantParams(float orig_min, float orig_max, QuantBitWidth bw,
                                 bool is_symmetric, float &aligned_min, float &aligned_max,
                                 int8_t &exp2_inv, int32_t &zp, const std::string &name = "") {
    // 从位宽配置获取量化范围
    const int32_t quant_min = bw.qmin();
    const int32_t quant_max = bw.qmax();
    const int num_steps = quant_max - quant_min;  // 量化级数
    
    // 与 AIMET 一致: num_steps 检查
    if (num_steps <= 0) {
        throw std::runtime_error("calibrateQuantParams: num_steps must be > 0");
    }
    
    // 与 AIMET _get_minimum_scale 一致
    const float minimum_scale = get_minimum_scale(num_steps);
    
    // 与 AIMET 一致: 确保 0 在范围内
    // min_with_zero = clamp(min, max=0) -> min <= 0
    // max_with_zero = clamp(max, min=0) -> max >= 0
    float min_with_zero = std::min(orig_min, 0.0f);
    float max_with_zero = std::max(orig_max, 0.0f);
    
    // 与 AIMET 一致: 范围扩展（当 tensor_diff < minimum_scale 时）
    float tensor_diff = (max_with_zero - min_with_zero) / static_cast<float>(num_steps);
    float adjustment_step = (tensor_diff < minimum_scale) ? minimum_scale : 0.0f;
    
    float updated_min, updated_max;
    if (is_symmetric) {
        if (bw.is_unsigned_) {
            // UINT 对称量化：只扩展 max（数据范围 [0, max]）
            updated_max = max_with_zero + static_cast<float>(num_steps) * adjustment_step;
            updated_min = 0.0f;  // UINT 的 min 固定为 0
        } else {
            // INT 对称量化: 两边扩展
            updated_max = max_with_zero + std::floor(num_steps / 2.0f) * adjustment_step;
            updated_min = min_with_zero - std::ceil(num_steps / 2.0f) * adjustment_step;
        }
    } else {
        // 非对称量化: 只扩展 max
        updated_max = max_with_zero + static_cast<float>(num_steps) * adjustment_step;
        updated_min = min_with_zero;
    }
    
    float scale;
    if (is_symmetric) {
        zp = 0;
        
        // UINT + symmetric: 特殊处理（zp=0，范围 [0, qmax]）
        if (bw.is_unsigned_) {
            // UINT 对称量化：数据范围 [0, max]，量化范围 [0, qmax]
            // scale = max / qmax，确保 max 能被正确量化
            float data_max = std::max(updated_max, minimum_scale);
            float raw_scale = data_max / static_cast<float>(quant_max);
            raw_scale = std::max(raw_scale, minimum_scale);
            
            // POT 转换
            exp2_inv = static_cast<int8_t>(std::floor(std::log2(1.0f / raw_scale)));
            scale = std::pow(2.0f, -exp2_inv);
            aligned_min = 0.0f;
            aligned_max = scale * static_cast<float>(quant_max);
        } else {
            // INT 对称量化：与 AIMET 一致
            const int num_pos_steps = num_steps / 2;           // floor(N/2), 8-bit: 127
            const int num_neg_steps = (num_steps + 1) / 2;     // ceil(N/2),  8-bit: 128
            
            // 与 AIMET 一致: 2-bit 特殊处理
            // "For 2-bit quantization, using math.floor to compute num_pos_steps can result 
            //  in a wasted bin on the negative side given a symmetrically distributed weight."
            int additional_step_for_calibration = 0;
            if (num_steps == 3) {  // 2-bit strict symmetric
                additional_step_for_calibration = 1;
            }
            
            // 与 AIMET 一致: delta = max(max/(pos+additional), -min/neg)
            float delta_from_max = (num_pos_steps + additional_step_for_calibration > 0) 
                                 ? updated_max / (num_pos_steps + additional_step_for_calibration)
                                 : 0.0f;
            float delta_from_min = (num_neg_steps > 0) 
                                 ? -updated_min / num_neg_steps 
                                 : 0.0f;
            float delta = std::max(delta_from_max, delta_from_min);
            delta = std::max(delta, minimum_scale);  // 确保 delta >= minimum_scale
            
            // 与 AIMET 一致: 重新计算 min/max
            // offset = -num_neg_steps
            // updated_min = offset * delta = -num_neg_steps * delta
            // updated_max = num_pos_steps * delta
            updated_min = -static_cast<float>(num_neg_steps) * delta;
            updated_max = static_cast<float>(num_pos_steps) * delta;
            
            // POT 转换
            float raw_scale = delta;
            exp2_inv = static_cast<int8_t>(std::floor(std::log2(1.0f / raw_scale)));
            scale = std::pow(2.0f, -exp2_inv);
            aligned_max = scale * num_pos_steps;
            aligned_min = -scale * num_neg_steps;
        }
    } else {
        // 非对称量化
        float range = updated_max - updated_min;
        range = std::max(range, minimum_scale * static_cast<float>(num_steps));
        float raw_scale = range / static_cast<float>(num_steps);

        exp2_inv = static_cast<int8_t>(std::floor(std::log2(1.0f / raw_scale)));
        scale = std::pow(2.0f, -exp2_inv);

        aligned_min = std::floor(updated_min / scale) * scale;
        aligned_max = std::ceil(updated_max / scale) * scale;

        zp = static_cast<int32_t>(std::round(quant_min - aligned_min / scale));
    }
    
    // 与 AIMET 一致: Inf 保护
    constexpr float float_max = std::numeric_limits<float>::max();
    constexpr float float_min = std::numeric_limits<float>::lowest();
    aligned_max = std::min(aligned_max, float_max);
    aligned_min = std::max(aligned_min, float_min);

#ifdef DEBUG
    if (!name.empty()) {
        printf("[DEBUG][QuantParam][%s] bits=%d, unsigned=%d, orig=[%.4f,%.4f], "
               "zero_included=[%.4f,%.4f], aligned=[%.4f,%.4f], scale=%.6f, exp2_inv=%d, zp=%d\n",
               name.c_str(), bw.bits_, bw.is_unsigned_, orig_min, orig_max,
               min_with_zero, max_with_zero, aligned_min, aligned_max, scale, exp2_inv, zp);
    }
#endif
}

/**
 * @brief 单值量化（任意位宽版本，CPU/GPU 共用）
 *
 * q = clamp(round(src / scale + zp), qmin, qmax)
 * 使用 roundf 实现四舍五入（round half away from zero），确保 CPU/GPU 行为一致。
 *
 * @param src 浮点输入
 * @param exp2_inv 缩放因子指数
 * @param zp 零点
 * @param bw 位宽配置
 * @return 量化值（int32_t 存储）
 */
__host__ __device__ __forceinline__ int32_t quantize(float src, int8_t exp2_inv, int32_t zp,
                                                      QuantBitWidth bw) {
    float scale = exp2_scale(exp2_inv);
    float shifted = src / scale + static_cast<float>(zp);
    int32_t q = static_cast<int32_t>(roundf(shifted));
    return clamp_by_bitwidth(q, bw);
}

/**
 * @brief 单值量化（模板版本，CPU/GPU 共用，兼容旧代码）
 */
template <typename QuantT>
__host__ __device__ __forceinline__ QuantT quantize(float src, int8_t exp2_inv, int32_t zp) {
    float scale = exp2_scale(exp2_inv);
    float shifted = src / scale + static_cast<float>(zp);
    int32_t q = static_cast<int32_t>(roundf(shifted));
    return clamp_to_type<QuantT>(q);
}

/**
 * @brief 单值反量化（CPU/GPU 共用）
 *
 * x = (q - zp) * scale
 */
template <typename QuantT>
__host__ __device__ __forceinline__ float dequantize(QuantT q, int8_t exp2_inv, int32_t zp) {
    int32_t v = static_cast<int32_t>(q) - zp;
    return static_cast<float>(v) * exp2_scale(exp2_inv);
}

/// @brief 批量量化（任意位宽版本，输出 int32_t）
inline void quantification(const float *data, int32_t *quant_data, size_t size, 
                           int8_t exp2_inv, int32_t zp, QuantBitWidth bw) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        quant_data[i] = quantize(data[i], exp2_inv, zp, bw);
    }
}

/// @brief 批量量化（模板版本，兼容旧代码）
template <typename T, typename QuantT>
inline void quantification(const T *data, QuantT *quant_data, size_t size, int8_t exp2_inv,
                           int32_t zp) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        quant_data[i] = quantize<QuantT>(data[i], exp2_inv, zp);
    }
}

/// @brief Per-channel 批量量化（Host 端，统一 int32_t 输出，使用位宽配置）
inline void quantificationPerChannelBitwidth(const float *src, int32_t *quant_data, size_t input_size,
                                              size_t channel_size, const std::vector<int8_t> &exp2_invs,
                                              QuantBitWidth bw) {
#pragma omp parallel for
    for (size_t i = 0; i < channel_size; ++i) {
        const int8_t exp2_inv = exp2_invs[i];
        for (size_t j = 0; j < input_size; ++j) {
            const size_t idx = j * channel_size + i;
            quant_data[idx] = quantize(src[idx], exp2_inv, 0, bw);
        }
    }
}

/// @brief Per-channel 批量量化（Host 端，用于权重矩阵）
/// @deprecated 建议使用 quantificationPerChannelBitwidth
template <typename T, typename QuantT>
inline void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                                     size_t channel_size, const std::vector<int8_t> &exp2_invs) {
#pragma omp parallel for
    for (int i = 0; i < channel_size; ++i) {
        const int8_t exp2_inv = exp2_invs[i];
        for (int j = 0; j < input_size; ++j) {
            const int idx = j * channel_size + i;
            quant_data[idx] = quantize<QuantT>(src[idx], exp2_inv, 0);  // 对称量化
        }
    }
}

// ============================================================================
// GPU 量化/反量化 Kernel 声明
// ============================================================================

namespace dev {

/// @brief GPU 批量量化
template <typename T, typename QuantT>
void quantification(const T *data, QuantT *quant_data, size_t size, int8_t exp2_inv, int32_t zp);

/// @brief GPU 批量反量化
template <typename T, typename QuantT>
void dequantification(const QuantT *quant_data, T *data, size_t size, int8_t exp2_inv, int32_t zp);

/// @brief GPU 反量化 V 向量（各部分使用不同量化参数）
/// V 向量布局: [z_out, r_out, g_out, weight_hh_linear_g]
template <typename T>
void dequantificationV(const int32_t *quant_data, T *data, int time_steps, int batch_size,
                       int hidden_size, int8_t shift_update_gate, int32_t zp_update_gate, int8_t shift_reset_gate,
                       int32_t zp_reset_gate, int8_t shift_new_gate, int32_t zp_new_gate, 
                       int8_t shift_weight_hh_linear, int32_t zp_weight_hh_linear);

/// @brief GPU 量化（统一 int32_t 输出，使用位宽配置）
void quantificationBitwidth(const float *data, int32_t *quant_data, size_t size,
                             int8_t exp2_inv, int32_t zp, QuantBitWidth bw);

/// @brief GPU Per-channel 量化（统一 int32_t 输出，使用位宽配置）
void quantificationPerChannelBitwidth(const float *src, int32_t *quant_data, size_t input_size,
                                       size_t channel_size, const dev::vector<int8_t> &exp2_invs,
                                       QuantBitWidth bw);

/// @brief GPU 量化
/// @deprecated 建议使用 quantificationBitwidth
template <typename T, typename QuantT>
void quantification(const T *data, QuantT *quant_data, size_t size, int8_t exp2_inv, int32_t zp);

/// @brief GPU Per-channel 量化
/// @deprecated 建议使用 quantificationPerChannelBitwidth
template <typename T, typename QuantT>
void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                              size_t channel_size, const dev::vector<int8_t> &exp2_invs);

/// @brief GPU Per-channel 反量化
template <typename T, typename QuantT>
void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                size_t channel_size, const dev::vector<int8_t> &exp2_invs);

}  // namespace dev

// ============================================================================
// 工具函数
// ============================================================================

#include <limits>
#include <random>

/// @brief 获取全局随机数生成器（固定种子，确保可复现）
inline std::mt19937 &getGlobalRng() {
    static std::mt19937 gen(42);
    return gen;
}

/// @brief 设置全局随机种子
inline void setGlobalRandomSeed(unsigned int seed) { getGlobalRng().seed(seed); }

/**
 * @brief 用截断正态分布填充向量
 *
 * @param data 待填充向量
 * @param min_value 最小值
 * @param max_value 最大值
 *
 * @note 使用 3σ 覆盖范围，超出范围的值会重新采样
 */
inline void fillVectorWithNormalDistribution(std::vector<float> &data, float min_value,
                                             float max_value) {
    float mean = (min_value + max_value) / 2.0f;
    float stddev = (max_value - min_value) / 6.0f;

    std::mt19937 &gen = getGlobalRng();
    std::normal_distribution<float> dist(mean, stddev);

    for (auto &value : data) {
        float sample;
        do {
            sample = dist(gen);
        } while (sample < min_value || sample > max_value);
        value = sample;
    }
}

// ============================================================================
// LUT 系数量化辅助函数
// ============================================================================
// 这些函数用于 LUT 生成时量化分段线性近似的系数（斜率、截距等）

// -------------------- 系数量化（对称量化，zp=0）--------------------

/// @brief 量化系数为 INT8
inline int8_t quantize_coefficient_int8(float val_fp, int8_t shift_bits) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale));
    return static_cast<int8_t>(std::max(-128, std::min(127, q)));
}

/// @brief 量化系数为 INT16
inline int16_t quantize_coefficient_int16(float val_fp, int8_t shift_bits) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale));
    return static_cast<int16_t>(std::max(-32768, std::min(32767, q)));
}

/// @brief 量化系数为 INT32（用于 LUT 斜率 q_b，避免截断误差）
inline int32_t quantize_coefficient_int32(float val_fp, int8_t shift_bits) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int64_t q = static_cast<int64_t>(std::round(val_fp / scale));
    q = std::max(static_cast<int64_t>(INT32_MIN), std::min(static_cast<int64_t>(INT32_MAX), q));
    return static_cast<int32_t>(q);
}

// -------------------- 输入量化（非对称量化）--------------------

/// @brief 量化输入为 UINT8
inline uint8_t quantize_input_uint8(float val_fp, int8_t shift_bits, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    return static_cast<uint8_t>(std::max(0, std::min(255, q)));
}

/// @brief 量化输入为 INT8
inline int8_t quantize_input_int8(float val_fp, int8_t shift_bits, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    return static_cast<int8_t>(std::max(-128, std::min(127, q)));
}

/// @brief 量化输入为 UINT16
inline uint16_t quantize_input_uint16(float val_fp, int8_t shift_bits, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    return static_cast<uint16_t>(std::max(0, std::min(65535, q)));
}

/// @brief 量化输入为 INT16
inline int16_t quantize_input_int16(float val_fp, int8_t shift_bits, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    return static_cast<int16_t>(std::max(-32768, std::min(32767, q)));
}

// -------------------- Shift bits 自动确定 --------------------

/// @brief 根据最大值确定 INT8 的 shift_bits
inline int8_t determine_shift_bits_int8(float max_val) {
    if (max_val < 1e-9f) return 0;
    float scale = max_val / 127.0f;
    int8_t shift_bits = static_cast<int8_t>(std::floor(-std::log2(scale)));
    return std::max(static_cast<int8_t>(0), shift_bits);
}

/// @brief 根据最大值确定 INT16 的 shift_bits
inline int8_t determine_shift_bits_int16(float max_val) {
    if (max_val < 1e-9f) return 0;
    float scale = max_val / 32767.0f;
    int8_t shift_bits = static_cast<int8_t>(std::floor(-std::log2(scale)));
    return std::max(static_cast<int8_t>(0), shift_bits);
}

/// @brief 根据最大值确定 INT32 的 shift_bits（留出 rounding 余量）
inline int8_t determine_shift_bits_int32(float max_val) {
    if (max_val < 1e-9f) return 0;
    float scale = max_val / 2147483520.0f;  // 略小于 INT32_MAX
    int8_t shift_bits = static_cast<int8_t>(std::floor(-std::log2(scale)));
    return std::max(static_cast<int8_t>(0), shift_bits);
}

// ============================================================================
// 调试函数
// ============================================================================

/// @brief 打印 GRU 量化参数（调试用）
inline void printParms(const GRUQuantParams &quant_parms) {
    printf("GRUQuantParams:\n");
    printf("  hidden = %d\n", quant_parms.hidden_);

    // 输入/隐状态
    printf("  x:  exp2_inv=%2d, zp=%d\n", quant_parms.shift_x_, quant_parms.zp_x_);
    printf("  h:  exp2_inv=%2d, zp=%d\n", quant_parms.shift_h_, quant_parms.zp_h_);

    // Per-channel 权重
    auto print_vec = [](const char *name, const std::vector<int8_t> &vec) {
        printf("  %s (size %zu): ", name, vec.size());
        for (size_t i = 0; i < vec.size() && i < 5; ++i) printf("%d ", vec[i]);
        if (vec.size() > 5) printf("...");
        printf("\n");
    };
    print_vec("W ", quant_parms.shift_W_);
    print_vec("R ", quant_parms.shift_R_);
    print_vec("bw", quant_parms.shift_bw_);
    print_vec("br", quant_parms.shift_br_);

    // Linear 输出
    printf("  weight_ih_linear: shift=%2d, zp=%d\n", quant_parms.shift_weight_ih_linear_, quant_parms.zp_weight_ih_linear_);
    printf("  weight_hh_linear: shift=%2d, zp=%d\n", quant_parms.shift_weight_hh_linear_, quant_parms.zp_weight_hh_linear_);

    // 门参数
    printf("  update_gate_input:  exp2_inv=%2d, zp=%d\n", quant_parms.shift_update_gate_input_, quant_parms.zp_update_gate_input_);
    printf("  update_gate_output: exp2_inv=%2d, zp=%d\n", quant_parms.shift_update_gate_output_, quant_parms.zp_update_gate_output_);
    printf("  reset_gate_input:   exp2_inv=%2d, zp=%d\n", quant_parms.shift_reset_gate_input_, quant_parms.zp_reset_gate_input_);
    printf("  reset_gate_output:  exp2_inv=%2d, zp=%d\n", quant_parms.shift_reset_gate_output_, quant_parms.zp_reset_gate_output_);
    printf("  new_gate_input:     exp2_inv=%2d, zp=%d\n", quant_parms.shift_new_gate_input_, quant_parms.zp_new_gate_input_);
    printf("  new_gate_output:    exp2_inv=%2d, zp=%d\n", quant_parms.shift_new_gate_output_, quant_parms.zp_new_gate_output_);

    // 中间计算
    printf("  mul_reset_hidden:   exp2_inv=%2d, zp=%d\n", quant_parms.shift_mul_reset_hidden_, quant_parms.zp_mul_reset_hidden_);

    // 隐状态更新
    printf("  mul_new_contribution: exp2_inv=%2d, zp=%d\n", quant_parms.shift_mul_new_contribution_,
           quant_parms.zp_mul_new_contribution_);
    printf("  mul_old_contribution: exp2_inv=%2d, zp=%d\n", quant_parms.shift_mul_old_contribution_,
           quant_parms.zp_mul_old_contribution_);
}
