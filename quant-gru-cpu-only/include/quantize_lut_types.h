#pragma once

// ============================================================================
// quantize_lut_types.h - 分段线性查找表（LUT）类型定义
// ============================================================================

#include <cstdint>
#include "quantize_bitwidth_config.h"

#define NUM_SEGMENTS 16

// ============================================================================
// 分段参数结构体
// ============================================================================

struct SegmentParams {
    int32_t q_b;                 ///< 量化斜率
    int8_t n_BX_total;           ///< 融合移位量
    int32_t term_c_precomputed;  ///< 预计算截距项
    int16_t threshold;           ///< 分段上界阈值
};

// ============================================================================
// 查找表结构体
// ============================================================================

struct SigmoidLUT {
    SegmentParams segments[NUM_SEGMENTS];
    int32_t zp_x;
    int8_t shift_bits_x;
    int8_t shift_bits_y;
    int32_t zp_y;
};

// ============================================================================
// LUT 生成函数声明
// ============================================================================

SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw);

SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                             QuantBitWidth input_bw, QuantBitWidth output_bw);

