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
 * @brief 计算 2^(-exp2_inv)（与主项目保持一致）
 */
inline float exp2_scale(int8_t exp2_inv) {
    return ldexpf(1.0f, -static_cast<int>(exp2_inv));
}

/**
 * @brief 单精度浮点数银行家舍入（round half to even，与主项目保持一致）
 */
inline float round_f(float x) {
    return rintf(x);
}

/**
 * @brief 双精度浮点数银行家舍入（round half to even，与主项目保持一致）
 */
inline double round_d(double x) {
    return rint(x);
}

/**
 * @brief 浮点数舍入到 int32_t（银行家舍入，与主项目保持一致）
 */
inline int32_t round_to_int(float x) {
    return static_cast<int32_t>(round_f(x));
}

/**
 * @brief 双精度浮点数舍入到 int64_t（银行家舍入，与主项目保持一致）
 */
inline int64_t round_to_int64(double x) {
    return static_cast<int64_t>(round_d(x));
}

/**
 * @brief 带银行家舍入的右移操作（int32_t 版本，纯定点实现）
 *
 * 实现 round(x / 2^n) 的定点运算，支持正负移位。
 * 采用 round half to even（银行家舍入）策略，与主项目保持一致。
 *
 * @param x 被移位的值
 * @param n 移位量（正数右移，负数或零左移）
 * @return 移位后的结果
 *
 * @note 纯定点实现，避免浮点转换的精度损失
 */
inline int32_t rshift_round(int32_t x, int8_t n) {
    if (n <= 0) return x << (-n);
    
    // 处理负数：对绝对值舍入后取反
    const bool neg = (x < 0);
    const int32_t abs_x = neg ? -x : x;
    
    // 正数的银行家舍入
    const int32_t half = 1 << (n - 1);
    const int32_t mask = (1 << n) - 1;
    const int32_t q = abs_x >> n;      // 商（向下取整）
    const int32_t r = abs_x & mask;    // 余数
    
    int32_t result;
    if (r > half) {
        result = q + 1;                // 大于一半，进位
    } else if (r < half) {
        result = q;                    // 小于一半，舍去
    } else {
        // 正好一半：舍入到偶数
        result = (q & 1) ? (q + 1) : q;
    }
    
    return neg ? -result : result;
}

/**
 * @brief 带银行家舍入的右移操作（int64_t 版本，纯定点实现）
 *
 * 用于处理 16 位量化时可能超出 int32 范围的乘积。
 * 采用 round half to even（银行家舍入）策略，与主项目保持一致。
 */
inline int64_t rshift_round(int64_t x, int8_t n) {
    if (n <= 0) return x << (-n);
    
    // 处理负数
    const bool neg = (x < 0);
    const int64_t abs_x = neg ? -x : x;
    
    // 正数的银行家舍入
    const int64_t half = static_cast<int64_t>(1) << (n - 1);
    const int64_t mask = (static_cast<int64_t>(1) << n) - 1;
    const int64_t q = abs_x >> n;
    const int64_t r = abs_x & mask;
    
    int64_t result;
    if (r > half) {
        result = q + 1;
    } else if (r < half) {
        result = q;
    } else {
        result = (q & 1) ? (q + 1) : q;
    }
    
    return neg ? -result : result;
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
 * 乘法scale融合：r * weight_hh_linear 的结果直接对齐到 new_gate_input，省略中间层
 * 与主项目的 computeNewGate<false> 逻辑完全一致
 * 
 * @param weight_ih_linear    输入 Linear 变换结果
 * @param weight_hh_linear    隐状态 Linear 变换结果（即 R*h + br，用于反向传播时直接保存到 v）
 * @param reset_gate          重置门输出
 */
inline int32_t computeNewGate(int32_t weight_ih_linear, int32_t weight_hh_linear, int32_t reset_gate,
                               const GateQuantParams &params) {
    // Linear 融合后，weight_hh_linear 就是 R*h + br
    // 计算 reset_gate * weight_hh_linear，直接对齐到 new_gate_input（融合中间层）
    const int64_t r_diff = static_cast<int64_t>(reset_gate) - params.zp_reset_gate_output_;
    const int64_t hh_diff = static_cast<int64_t>(weight_hh_linear) - params.zp_weight_hh_linear_;
    const int64_t reset_hidden_mul = r_diff * hh_diff;

    // 乘法结果直接 shift 到 new_gate_input 空间（融合后省略中间 zp）
    const int32_t rh_shifted = static_cast<int32_t>(rshift_round(reset_hidden_mul, params.shift_reset_mul_hh_to_new_gate_input_));

    // weight_ih_linear shift 到 new_gate_input 空间
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, params.shift_weight_ih_linear_to_new_gate_input_);

    const int32_t new_gate_input = ih_shifted + rh_shifted + params.zp_new_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    const auto &lut = params.tanh_new_gate_lut_;
    
    // 与主项目的 computeNewGate<false> 逻辑一致：使用 piecewise_linear（推理模式）
    return piecewise_linear(new_gate_input, lut, bw_cfg.new_gate_input_, bw_cfg.new_gate_output_);
}

/**
 * @brief 计算隐藏状态 h_new = update_gate * h_old + (1 - update_gate) * new_gate
 * 
 * 优化策略：统一 scale 空间，先将 new_gate 对齐到 h scale，使两个乘积的 scale 统一为
 * S_{update_gate_output} * S_h，在统一 scale 下直接相加，最后一起 rescale 到 h scale。
 */
inline int32_t computeHiddenState(int32_t update_gate, int32_t new_gate, int32_t h_old, 
                                   const GateQuantParams &params) {
    // ========== 步骤1: 将 new_gate 从 new_gate_output scale 对齐到 h scale ==========
    // 这样后续计算时 old_contribution 和 new_contribution 的 scale 可以统一
    const int64_t n_diff_from_zp = static_cast<int64_t>(new_gate) - params.zp_new_gate_output_;
    const int32_t new_gate_aligned_to_h = static_cast<int32_t>(
        rshift_round(n_diff_from_zp, params.shift_new_gate_output_to_h_)) + params.zp_h_;

    // ========== 步骤2: 计算 old_contribution = update_gate * h_old ==========
    // u_diff 在 update_gate_output scale，h_diff 在 h scale
    // 乘积 scale = scale_update_gate_output * scale_h
    const int64_t u_diff = static_cast<int64_t>(update_gate) - params.zp_update_gate_output_;
    const int64_t h_diff = static_cast<int64_t>(h_old) - params.zp_h_;
    const int64_t old_contribution_mul = u_diff * h_diff;  // scale = S_u * S_h

    // ========== 步骤3: 计算 new_contribution = (1 - update_gate) * new_gate_aligned ==========
    // quant_one = 2^shift + zp，是常数 1 在 update_gate_output 量化空间的完整表示
    // one_minus_u = quant_one - update_gate = (2^shift + zp) - update_gate
    const int64_t one_minus_u = static_cast<int64_t>(params.quant_one_in_update_gate_scale_) - update_gate;
    const int64_t one_minus_u_diff = one_minus_u - params.zp_update_gate_output_;  // 在 update_gate_output scale
    const int64_t n_diff_aligned = static_cast<int64_t>(new_gate_aligned_to_h) - params.zp_h_;  // 在 h scale
    // one_minus_u_diff 在 update_gate_output scale，n_diff_aligned 在 h scale
    // 乘积 scale = scale_update_gate_output * scale_h
    const int64_t new_contribution_mul = one_minus_u_diff * n_diff_aligned;  // scale = S_u * S_h

    // ========== 步骤4: 合并两个贡献并 rescale 到 h scale ==========
    // old_contribution_mul 和 new_contribution_mul 都在 scale_update_gate_output * scale_h
    // scale，可以直接相加
    const int64_t combined = old_contribution_mul + new_contribution_mul;

    // rescale 到 h scale: 从 scale_update_gate_output * scale_h 到 scale_h
    // shift_update_old_to_h_ = shift_update_gate_output（因为 scale_h / scale_h = 1）
    const int32_t h_new = static_cast<int32_t>(
        rshift_round(combined, params.shift_update_old_to_h_)) + params.zp_h_;

    return clamp_by_bitwidth(h_new, params.bitwidth_config_.h_);
}

// ============================================================================
// LUT 系数量化辅助函数
// ============================================================================

/**
 * @brief 量化系数到 int32_t（与主项目完全一致）
 */
inline int32_t quantize_coefficient_int32(float val_fp, int8_t shift_bits) {
    float scale = exp2_scale(shift_bits);
    int64_t q = round_to_int64(static_cast<double>(val_fp / scale));
    q = std::max(static_cast<int64_t>(INT32_MIN), std::min(static_cast<int64_t>(INT32_MAX), q));
    return static_cast<int32_t>(q);
}

// -------------------- Shift bits 自动确定 --------------------

/**
 * @brief 根据最大值和位宽配置自动确定 shift_bits（与主项目完全一致）
 * @param max_val 浮点数的最大绝对值
 * @param bw 目标量化位宽
 * @return 使量化值能充分利用目标范围的 shift_bits
 */
inline int8_t determine_shift_bits(float max_val, QuantBitWidth bw) {
    if (max_val < 1e-9f) return 0;
    // 使用 qmax 作为量化范围上限（对称量化）
    float qmax = static_cast<float>(bw.qmax());
    float scale = max_val / qmax;
    int8_t shift_bits = static_cast<int8_t>(std::floor(-std::log2(scale)));
    return std::max(static_cast<int8_t>(0), shift_bits);
}
