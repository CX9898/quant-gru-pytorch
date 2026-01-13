#pragma once

#include <cmath>
#include <cstdint>
#include "quantize_param_types.h"

// ============================================================================
// 基础运算函数
// ============================================================================

inline int32_t rshift_round(int32_t x, int8_t n) {
    if (n <= 0) return x << (-n);
    const int32_t offset = 1 << (n - 1);
    return (x >= 0) ? ((x + offset) >> n) : -((-x + offset) >> n);
}

inline int64_t rshift_round(int64_t x, int8_t n) {
    if (n <= 0) return x << (-n);
    const int64_t offset = static_cast<int64_t>(1) << (n - 1);
    return (x >= 0) ? ((x + offset) >> n) : -((-x + offset) >> n);
}

inline int32_t clamp_by_bitwidth(int32_t val, QuantBitWidth bw) {
    int32_t lo = bw.qmin();
    int32_t hi = bw.qmax();
    return (val < lo) ? lo : ((val > hi) ? hi : val);
}

// ============================================================================
// 分段线性近似（Sigmoid/Tanh LUT）
// ============================================================================

inline int find_segment(int32_t q_x, const SegmentParams *segments) {
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        if (q_x < segments[i].threshold) return i;
    }
    return NUM_SEGMENTS - 1;
}

inline int32_t piecewise_linear(int32_t q_x, const SigmoidLUT &lut,
                                QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    q_x = clamp_by_bitwidth(q_x, pre_bw);
    
    int seg_id = find_segment(q_x, lut.segments);
    const SegmentParams &seg = lut.segments[seg_id];

    int64_t bx = static_cast<int64_t>(seg.q_b) * (q_x - lut.zp_x);
    int32_t term_bx = (seg.n_BX_total >= 0)
        ? static_cast<int32_t>(rshift_round(bx, seg.n_BX_total))
        : static_cast<int32_t>(bx << (-seg.n_BX_total));

    return clamp_by_bitwidth(term_bx + seg.term_c_precomputed, out_bw);
}

// ============================================================================
// GRU 门计算函数
// ============================================================================

inline int32_t computeUpdateGate(int32_t ih, int32_t hh, const GateQuantParams &p) {
    int32_t input = rshift_round(ih - p.zp_weight_ih_linear_, p.shift_weight_ih_linear_to_update_gate_input_)
                  + rshift_round(hh - p.zp_weight_hh_linear_, p.shift_weight_hh_linear_to_update_gate_input_)
                  + p.zp_update_gate_input_;
    return piecewise_linear(input, p.sigmoid_update_gate_lut_, 
                           p.bitwidth_config_.update_gate_input_, p.bitwidth_config_.update_gate_output_);
}

inline int32_t computeResetGate(int32_t ih, int32_t hh, const GateQuantParams &p) {
    int32_t input = rshift_round(ih - p.zp_weight_ih_linear_, p.shift_weight_ih_linear_to_reset_gate_input_)
                  + rshift_round(hh - p.zp_weight_hh_linear_, p.shift_weight_hh_linear_to_reset_gate_input_)
                  + p.zp_reset_gate_input_;
    return piecewise_linear(input, p.sigmoid_reset_gate_lut_, 
                           p.bitwidth_config_.reset_gate_input_, p.bitwidth_config_.reset_gate_output_);
}

inline int32_t computeNewGate(int32_t ih, int32_t hh, int32_t r, const GateQuantParams &p, int32_t &hh_out) {
    hh_out = hh;
    int64_t rh = (static_cast<int64_t>(r) - p.zp_reset_gate_output_) 
              * (static_cast<int64_t>(hh) - p.zp_weight_hh_linear_);
    int32_t mul_rh = clamp_by_bitwidth(
        static_cast<int32_t>(rshift_round(rh, p.shift_reset_gate_mul_hh_to_mul_reset_hidden_)) + p.zp_mul_reset_hidden_,
        p.bitwidth_config_.mul_reset_hidden_);

    int32_t input = rshift_round(ih - p.zp_weight_ih_linear_, p.shift_weight_ih_linear_to_new_gate_input_)
                  + rshift_round(mul_rh - p.zp_mul_reset_hidden_, p.shift_mul_reset_hidden_to_new_gate_input_)
                  + p.zp_new_gate_input_;
    return piecewise_linear(input, p.tanh_new_gate_lut_, 
                           p.bitwidth_config_.new_gate_input_, p.bitwidth_config_.new_gate_output_);
}

inline int32_t computeHiddenState(int32_t u, int32_t n, int32_t h, const GateQuantParams &p) {
    int64_t old_c = (static_cast<int64_t>(u) - p.zp_update_gate_output_) 
                  * (static_cast<int64_t>(h) - p.zp_h_);
    int32_t mul_old = clamp_by_bitwidth(
        static_cast<int32_t>(rshift_round(old_c, p.shift_update_h_to_mul_old_contribution_)) + p.zp_mul_old_contribution_,
        p.bitwidth_config_.mul_old_contribution_);

    int64_t new_c = (static_cast<int64_t>(p.quant_one_in_update_gate_scale_) - u) 
                  * (static_cast<int64_t>(n) - p.zp_new_gate_output_);
    int32_t mul_new = clamp_by_bitwidth(
        static_cast<int32_t>(rshift_round(new_c, p.shift_update_new_to_mul_new_contribution_)) + p.zp_mul_new_contribution_,
        p.bitwidth_config_.mul_new_contribution_);

    int32_t h_new = rshift_round(mul_old - p.zp_mul_old_contribution_, p.shift_mul_old_contribution_to_h_)
                  + rshift_round(mul_new - p.zp_mul_new_contribution_, p.shift_mul_new_contribution_to_h_)
                  + p.zp_h_;
    return clamp_by_bitwidth(h_new, p.bitwidth_config_.h_);
}

// ============================================================================
// LUT 生成辅助函数
// ============================================================================

inline int32_t quantize_coefficient_int32(float val, int8_t shift) {
    float scale = std::pow(2.0f, -static_cast<float>(shift));
    int64_t q = static_cast<int64_t>(std::round(val / scale));
    return static_cast<int32_t>(std::max(static_cast<int64_t>(INT32_MIN), 
                                         std::min(static_cast<int64_t>(INT32_MAX), q)));
}

inline int8_t determine_shift_bits_int8(float max_val) {
    if (max_val < 1e-9f) return 0;
    return std::max(static_cast<int8_t>(0), 
                    static_cast<int8_t>(std::floor(-std::log2(max_val / 127.0f))));
}

inline int8_t determine_shift_bits_int16(float max_val) {
    if (max_val < 1e-9f) return 0;
    return std::max(static_cast<int8_t>(0), 
                    static_cast<int8_t>(std::floor(-std::log2(max_val / 32767.0f))));
}
