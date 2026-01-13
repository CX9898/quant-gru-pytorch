#pragma once

// ============================================================================
// quantize_ops_helper.h - GRU 量化核心定义与 CPU 函数 (纯 CPU 版本)
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"

// ============================================================================
// Part 1: 量化参数结构体定义
// ============================================================================

struct GRUQuantitativeParameters {
    OperatorQuantConfig bitwidth_config_;

    int hidden_;
    int8_t exp2_inv_x_;
    int32_t zp_x_;
    int8_t exp2_inv_h_;
    int32_t zp_h_;

    std::vector<int8_t> exp2_inv_W_;
    std::vector<int8_t> exp2_inv_R_;

    int8_t exp2_inv_Wx_;
    int32_t zp_Wx_;
    int8_t exp2_inv_Rh_;
    int32_t zp_Rh_;

    std::vector<int8_t> exp2_inv_bx_;
    std::vector<int8_t> exp2_inv_br_;

    int8_t exp2_inv_z_pre_;
    int32_t zp_z_pre_;
    int8_t exp2_inv_r_pre_;
    int32_t zp_r_pre_;
    int8_t exp2_inv_g_pre_;
    int32_t zp_g_pre_;

    int8_t exp2_inv_z_out_;
    int32_t zp_z_out_;
    int8_t exp2_inv_r_out_;
    int32_t zp_r_out_;
    int8_t exp2_inv_g_out_;
    int32_t zp_g_out_;

    int8_t exp2_inv_Rh_add_br_;
    int32_t zp_Rh_add_br_;
    int8_t exp2_inv_rRh_;
    int32_t zp_rRh_;

    int8_t exp2_inv_new_contrib_;
    int32_t zp_new_contrib_;
    int8_t exp2_inv_old_contrib_;
    int32_t zp_old_contrib_;

    SigmoidLUT sigmoid_z_lut_;
    SigmoidLUT sigmoid_r_lut_;
    SigmoidLUT tanh_g_lut_;
};

void generate_piecewise_linear_lut_to_params(GRUQuantitativeParameters &params);

// ============================================================================
// Part 2: 基础运算函数
// ============================================================================

inline int32_t rshift_round(int32_t x, int8_t n) {
    if (n <= 0) return x << (-n);

    const int32_t offset = 1 << (n - 1);
    if (x >= 0) {
        return (x + offset) >> n;
    } else {
        return -((-x + offset) >> n);
    }
}

inline int64_t rshift_round(int64_t x, int8_t n) {
    if (n <= 0) return x << (-n);

    const int64_t offset = static_cast<int64_t>(1) << (n - 1);
    if (x >= 0) {
        return (x + offset) >> n;
    } else {
        return -((-x + offset) >> n);
    }
}

// ============================================================================
// Part 3: 饱和截断函数
// ============================================================================

template <typename T>
inline T clamp_to_type(int32_t x);

template <>
inline int8_t clamp_to_type<int8_t>(int32_t x) {
    return static_cast<int8_t>((x < -128) ? -128 : ((x > 127) ? 127 : x));
}

template <>
inline int16_t clamp_to_type<int16_t>(int32_t x) {
    return static_cast<int16_t>((x < -32768) ? -32768 : ((x > 32767) ? 32767 : x));
}

template <>
inline int32_t clamp_to_type<int32_t>(int32_t x) {
    return x;
}

template <>
inline uint8_t clamp_to_type<uint8_t>(int32_t x) {
    return static_cast<uint8_t>((x < 0) ? 0 : ((x > 255) ? 255 : x));
}

template <>
inline uint16_t clamp_to_type<uint16_t>(int32_t x) {
    return static_cast<uint16_t>((x < 0) ? 0 : ((x > 65535) ? 65535 : x));
}

inline int32_t clamp_by_bitwidth(int32_t val, QuantBitWidth bw) {
    switch (bw) {
        case QuantBitWidth::INT8:
            return (val < -128) ? -128 : ((val > 127) ? 127 : val);
        case QuantBitWidth::INT16:
            return (val < -32768) ? -32768 : ((val > 32767) ? 32767 : val);
        case QuantBitWidth::UINT8:
            return (val < 0) ? 0 : ((val > 255) ? 255 : val);
        case QuantBitWidth::UINT16:
            return (val < 0) ? 0 : ((val > 65535) ? 65535 : val);
        case QuantBitWidth::INT32:
        default:
            return val;
    }
}

// ============================================================================
// Part 4: 分段线性近似函数
// ============================================================================

inline int find_segment(int32_t q_x, const SegmentParams *segments) {
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        if (q_x < segments[i].threshold) {
            return i;
        }
    }
    return NUM_SEGMENTS - 1;
}

inline int32_t piecewise_linear_raw(int32_t q_x, const SigmoidLUT &lut) {
    int seg_id = find_segment(q_x, lut.segments);
    const SegmentParams &seg = lut.segments[seg_id];

    int32_t x_offset = q_x - lut.zp_x;
    int64_t bw_64 = static_cast<int64_t>(seg.q_b) * static_cast<int64_t>(x_offset);

    int32_t term_bx = (seg.n_BX_total >= 0)
                          ? static_cast<int32_t>(rshift_round(bw_64, seg.n_BX_total))
                          : static_cast<int32_t>(bw_64 << (-seg.n_BX_total));

    return term_bx + seg.term_c_precomputed;
}

inline int32_t piecewise_linear(int32_t q_x, const SigmoidLUT &lut,
                                QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    int32_t q_x_clamped = clamp_by_bitwidth(q_x, pre_bw);
    int32_t result = piecewise_linear_raw(q_x_clamped, lut);
    return clamp_by_bitwidth(result, out_bw);
}

// ============================================================================
// Part 5: 量化/反量化函数
// ============================================================================

template <typename QuantT>
inline QuantT quantize(float src, int8_t exp2_inv, int32_t zp) {
    float scale = static_cast<float>(1 << exp2_inv);
    float shifted = src * scale + static_cast<float>(zp);
    int32_t q = static_cast<int32_t>(std::round(shifted));

    constexpr int32_t qmin = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    constexpr int32_t qmax = static_cast<int32_t>(std::numeric_limits<QuantT>::max());
    q = (q < qmin) ? qmin : ((q > qmax) ? qmax : q);  // 手动 clamp，无需 C++17

    return static_cast<QuantT>(q);
}

template <typename QuantT>
inline float dequantize(QuantT q, int8_t exp2_inv, int32_t zp) {
    int32_t v = static_cast<int32_t>(q) - zp;
    if (exp2_inv >= 0) {
        return static_cast<float>(v) / static_cast<float>(1 << exp2_inv);
    } else {
        return static_cast<float>(v) * static_cast<float>(1 << (-exp2_inv));
    }
}

template <typename T, typename QuantT>
inline void quantification(const T *data, QuantT *quant_data, size_t size, int8_t exp2_inv,
                           int32_t zp) {
    for (size_t i = 0; i < size; ++i) {
        quant_data[i] = quantize<QuantT>(data[i], exp2_inv, zp);
    }
}

template <typename T, typename QuantT>
inline void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                                     size_t channel_size, const std::vector<int8_t> &exp2_invs) {
    for (size_t i = 0; i < channel_size; ++i) {
        const int8_t exp2_inv = exp2_invs[i];
        for (size_t j = 0; j < input_size; ++j) {
            const size_t idx = j * channel_size + i;
            quant_data[idx] = quantize<QuantT>(src[idx], exp2_inv, 0);
        }
    }
}

// ============================================================================
// Part 6: 调试函数
// ============================================================================

inline void printParms(const GRUQuantitativeParameters &quant_parms) {
    printf("GRUQuantitativeParameters:\n");
    printf("  hidden = %d\n", quant_parms.hidden_);
    printf("  x:  exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_x_, quant_parms.zp_x_);
    printf("  h:  exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_h_, quant_parms.zp_h_);
    printf("  Wx: exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_Wx_, quant_parms.zp_Wx_);
    printf("  Rh: exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_Rh_, quant_parms.zp_Rh_);
    printf("  z_pre:  exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_z_pre_, quant_parms.zp_z_pre_);
    printf("  z_out:  exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_);
    printf("  r_pre:  exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_r_pre_, quant_parms.zp_r_pre_);
    printf("  r_out:  exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_);
    printf("  g_pre:  exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_g_pre_, quant_parms.zp_g_pre_);
    printf("  g_out:  exp2_inv=%2d, zp=%d\n", quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_);
}

