#pragma once

#include <cstdint>

#include "quantize_bitwidth_config.h"

// ============================================================================
// quantize_lut_types.h - 分段线性 LUT 类型定义（纯 C++，无 CUDA 依赖）
// ============================================================================
//
// 此文件定义 LUT 结构体和生成函数声明，供 CPU 和 GPU 代码共用。
// GPU 特定的常量内存声明在 quantize_ops.cuh 中。
//
// ============================================================================

#ifndef NUM_SEGMENTS
#define NUM_SEGMENTS 16
#endif

// ==================== 统一的 LUT 结构 ====================

/// @brief 统一的段参数结构（使用最大精度类型）
struct SegmentParams {
    int32_t q_b;                 // 量化后的系数 b (INT32，避免溢出截断)
    int8_t n_BX_total;           // 融合后的移位位数 (INT8，可能为负)
    int32_t term_c_precomputed;  // 预计算的 term_c (INT32)
    int16_t threshold;           // 段阈值 (INT16，可容纳 INT8 和 INT16 输入)
};

/// @brief 统一的查找表结构
struct SigmoidLUT {
    SegmentParams segments[NUM_SEGMENTS];
    int32_t zp_x;         // 输入 zero-point (INT32)
    int8_t shift_bits_x;  // 输入 shift_bits (INT8)
    int8_t shift_bits_y;  // 输出 shift_bits (INT8)
    int32_t zp_y;         // 输出 zero-point (INT32)
};

// 为了兼容性，保留旧类型名作为别名
using SegmentParams_INT8 = SegmentParams;
using SegmentParams_INT16 = SegmentParams;
using SegmentParams_INT8_to_INT16 = SegmentParams;
using SigmoidLUT_INT8 = SigmoidLUT;
using SigmoidLUT_INT16 = SigmoidLUT;
using SigmoidLUT_INT8_to_INT16 = SigmoidLUT;

// ==================== LUT 生成函数声明（纯 CPU 代码）====================
// 实现在 quantize_ops.cu 中（不依赖 CUDA kernel，可被 CPU/GPU 代码链接调用）

/**
 * @brief 生成 Sigmoid LUT
 * @param shift_bits_x 输入量化 shift bits
 * @param zp_x 输入 zero-point
 * @param shift_bits_y 输出量化 shift bits
 * @param zp_y 输出 zero-point
 * @param input_bw 输入位宽（决定输入范围）
 * @param output_bw 输出位宽（决定精度）
 * @return 生成的 SigmoidLUT 结构
 */
SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw);

/**
 * @brief 生成 Tanh LUT
 */
SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                             int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw);
