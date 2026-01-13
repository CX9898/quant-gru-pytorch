#pragma once

#include <cstdint>
#include "quantize_bitwidth_config.h"

#define NUM_SEGMENTS 16

struct SegmentParams {
    int32_t q_b;
    int8_t n_BX_total;
    int32_t term_c_precomputed;
    int32_t threshold;
};

struct SigmoidLUT {
    SegmentParams segments[NUM_SEGMENTS];
    int32_t zp_x;
    int8_t shift_bits_x;
    int8_t shift_bits_y;
    int32_t zp_y;
};

SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw);

SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                             QuantBitWidth input_bw, QuantBitWidth output_bw);
