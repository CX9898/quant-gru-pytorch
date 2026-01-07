#pragma once

// ============================================================================
// quantize_lut_types.h - 分段线性查找表（LUT）类型定义
// ============================================================================
//
// 本文件定义分段线性近似所需的数据结构和生成函数声明。
// 用于将 Sigmoid/Tanh 等非线性函数在每个分段内用 y = b*x + c 近似。
//
// 设计特点：
//   - 纯 C++ 代码，无 CUDA 依赖
//   - CPU/GPU 代码共用同一套结构体
//
// ============================================================================

#include <cstdint>

#include "quantize_bitwidth_config.h"

#define NUM_SEGMENTS 16

// ============================================================================
// 分段参数结构体
// ============================================================================

/**
 * @brief 单个分段的参数
 *
 * 分段线性公式：q_y = (q_b * (q_x - zp_x)) >> n_BX_total + term_c_precomputed
 */
struct SegmentParams {
    int32_t q_b;                 ///< 量化斜率（INT32，避免截断误差）
    int8_t n_BX_total;           ///< 融合移位量 = shift_b + shift_x - shift_y（可能为负）
    int32_t term_c_precomputed;  ///< 预计算截距项（已含输出零点）
    int32_t threshold;           ///< 分段上界阈值（q_x < threshold 则属于此段），支持任意位宽
};

// ============================================================================
// 查找表结构体
// ============================================================================

/**
 * @brief Sigmoid/Tanh 分段线性查找表
 *
 * 包含 NUM_SEGMENTS 个分段参数和量化参数。
 */
struct SigmoidLUT {
    SegmentParams segments[NUM_SEGMENTS];  ///< 分段参数数组
    int32_t zp_x;                          ///< 输入零点
    int8_t shift_bits_x;                   ///< 输入缩放因子指数
    int8_t shift_bits_y;                   ///< 输出缩放因子指数
    int32_t zp_y;                          ///< 输出零点
};

// ============================================================================
// LUT 生成函数声明
// ============================================================================
// 实现在 quantize_ops.cu 中

/**
 * @brief 生成 Sigmoid 查找表
 *
 * @param shift_bits_x 输入缩放因子指数
 * @param zp_x 输入零点
 * @param shift_bits_y 输出缩放因子指数
 * @param zp_y 输出零点
 * @param input_bw 输入位宽（决定分段范围）
 * @param output_bw 输出位宽（决定输出精度）
 * @return 生成的 SigmoidLUT 结构
 */
SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw);

/**
 * @brief 生成 Tanh 查找表
 *
 * 参数含义同 generate_sigmoid_lut
 */
SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                             QuantBitWidth input_bw, QuantBitWidth output_bw);
