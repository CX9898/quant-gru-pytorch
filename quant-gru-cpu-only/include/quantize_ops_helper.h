#pragma once

// ============================================================================
// quantize_ops_helper.h - GRU 量化 CPU 共用函数
// ============================================================================
//
// 本文件包含：
//   1. 量化/反量化基础操作函数
//   2. 分段线性近似函数（Sigmoid/Tanh LUT）
//   3. GRU 门计算模板函数
//
// 设计原则：
//   - 所有缩放因子均为 2 的负 n 次方：scale = 2^(-exp2_inv)
//   - 支持对称量化（zp=0）和非对称量化（zp≠0）
//
// ============================================================================

#include <cmath>
#include <cstdint>
#include "quantize_param_types.h"

// ============================================================================
// 基础运算函数
// ============================================================================

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
inline int32_t rshift_round(int32_t x, int8_t n) {
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
inline int64_t rshift_round(int64_t x, int8_t n) {
    if (n <= 0) return x << (-n);

    const int64_t offset = static_cast<int64_t>(1) << (n - 1);
    if (x >= 0) {
        return (x + offset) >> n;
    } else {
        return -((-x + offset) >> n);  // 向零舍入
    }
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
inline int32_t clamp_by_bitwidth(int32_t val, QuantBitWidth bw) {
    int32_t lo = bw.qmin();
    int32_t hi = bw.qmax();
    return (val < lo) ? lo : ((val > hi) ? hi : val);
}

// ============================================================================
// 分段线性近似函数（Sigmoid/Tanh LUT）
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
inline int find_segment(int32_t q_x, const SegmentParams *segments) {
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
inline int32_t piecewise_linear_raw(int32_t q_x, const SigmoidLUT &lut) {
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
inline int32_t piecewise_linear(int32_t q_x, const SigmoidLUT &lut,
                                 QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    int32_t q_x_clamped = clamp_by_bitwidth(q_x, pre_bw);
    int32_t result = piecewise_linear_raw(q_x_clamped, lut);
    return clamp_by_bitwidth(result, out_bw);
}

// ============================================================================
// GRU 门计算模板函数
// ============================================================================

/**
 * @brief 计算更新门 update_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 * 
 * @param weight_ih_linear  输入 Linear 变换结果 W*x + bw
 * @param weight_hh_linear  隐状态 Linear 变换结果 R*h + br
 * @param params    门计算参数
 */
inline int32_t computeUpdateGate(int32_t weight_ih_linear, int32_t weight_hh_linear, 
                                  const GateQuantParams &params) {
    // 重缩放到 update_gate_input 空间
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, 
                                            params.shift_weight_ih_linear_to_update_gate_input_);
    const int32_t hh_shifted = rshift_round(weight_hh_linear - params.zp_weight_hh_linear_, 
                                            params.shift_weight_hh_linear_to_update_gate_input_);

    const int32_t update_gate_input = ih_shifted + hh_shifted + params.zp_update_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    return piecewise_linear(update_gate_input, params.sigmoid_update_gate_lut_, 
                            bw_cfg.update_gate_input_, bw_cfg.update_gate_output_);
}

/**
 * @brief 计算重置门 reset_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 */
inline int32_t computeResetGate(int32_t weight_ih_linear, int32_t weight_hh_linear, 
                                 const GateQuantParams &params) {
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, 
                                            params.shift_weight_ih_linear_to_reset_gate_input_);
    const int32_t hh_shifted = rshift_round(weight_hh_linear - params.zp_weight_hh_linear_, 
                                            params.shift_weight_hh_linear_to_reset_gate_input_);

    const int32_t reset_gate_input = ih_shifted + hh_shifted + params.zp_reset_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    return piecewise_linear(reset_gate_input, params.sigmoid_reset_gate_lut_, 
                            bw_cfg.reset_gate_input_, bw_cfg.reset_gate_output_);
}

/**
 * @brief 计算候选门 new_gate = tanh(weight_ih_linear + reset_gate * weight_hh_linear)
 * 
 * @param weight_ih_linear    输入 Linear 变换结果
 * @param weight_hh_linear    隐状态 Linear 变换结果
 * @param reset_gate          重置门输出
 * @param weight_hh_linear_g  [out] 中间结果，用于存储到 v（训练时反向传播需要）
 */
inline int32_t computeNewGate(int32_t weight_ih_linear, int32_t weight_hh_linear, int32_t reset_gate,
                               const GateQuantParams &params, int32_t &weight_hh_linear_g) {
    // Linear 融合后，weight_hh_linear 就是 R*h + br
    weight_hh_linear_g = weight_hh_linear;

    // 计算 reset_gate * weight_hh_linear (即 mul_reset_hidden)
    const int64_t r_diff = static_cast<int64_t>(reset_gate) - params.zp_reset_gate_output_;
    const int64_t hh_diff = static_cast<int64_t>(weight_hh_linear_g) - params.zp_weight_hh_linear_;
    const int64_t reset_hidden_mul = r_diff * hh_diff;

    int32_t mul_reset_hidden = static_cast<int32_t>(
        rshift_round(reset_hidden_mul, params.shift_reset_gate_mul_hh_to_mul_reset_hidden_)) +
        params.zp_mul_reset_hidden_;
    mul_reset_hidden = clamp_by_bitwidth(mul_reset_hidden, params.bitwidth_config_.mul_reset_hidden_);

    // 计算 new_gate_input = weight_ih_linear + mul_reset_hidden
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, 
                                            params.shift_weight_ih_linear_to_new_gate_input_);
    const int32_t rh_shifted = rshift_round(mul_reset_hidden - params.zp_mul_reset_hidden_, 
                                            params.shift_mul_reset_hidden_to_new_gate_input_);

    const int32_t new_gate_input = ih_shifted + rh_shifted + params.zp_new_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    return piecewise_linear(new_gate_input, params.tanh_new_gate_lut_, 
                            bw_cfg.new_gate_input_, bw_cfg.new_gate_output_);
}

/**
 * @brief 计算隐藏状态 h_new = update_gate * h_old + (1 - update_gate) * new_gate
 */
inline int32_t computeHiddenState(int32_t update_gate, int32_t new_gate, int32_t h_old, 
                                   const GateQuantParams &params) {
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

    return clamp_by_bitwidth(h_new, params.bitwidth_config_.h_);
}

// ============================================================================
// LUT 系数量化辅助函数
// ============================================================================

/**
 * @brief 量化系数到 int32_t
 */
inline int32_t quantize_coefficient_int32(float val, int8_t shift) {
    float scale = static_cast<float>(1 << shift);
    return static_cast<int32_t>(std::round(val * scale));
}

/**
 * @brief 计算 int8 范围内的最优移位量
 */
inline int8_t determine_shift_bits_int8(float max_val) {
    if (max_val < 1e-9f) return 7;
    int8_t shift = static_cast<int8_t>(std::floor(std::log2(127.0f / max_val)));
    return std::max(static_cast<int8_t>(0), std::min(shift, static_cast<int8_t>(15)));
}

/**
 * @brief 计算 int16 范围内的最优移位量
 */
inline int8_t determine_shift_bits_int16(float max_val) {
    if (max_val < 1e-9f) return 15;
    int8_t shift = static_cast<int8_t>(std::floor(std::log2(32767.0f / max_val)));
    return std::max(static_cast<int8_t>(0), std::min(shift, static_cast<int8_t>(30)));
}
