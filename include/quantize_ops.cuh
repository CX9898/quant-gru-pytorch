#pragma once

// ============================================================================
// quantize_ops.cuh - GPU 量化操作与 CUDA 常量内存声明
// ============================================================================
//
// 本文件包含：
//   1. GPU 专用优化函数（使用 CUDA 内置函数）
//   2. GRU 各门的激活函数封装
//
// 依赖关系：
//   - quantize_lut_types.h: LUT 结构体定义
//   - quantize_ops_helper.h: CPU/GPU 共用函数
//
// ============================================================================

#include <cuda_runtime.h>

#include <cstdint>

#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"
#include "quantize_ops_helper.h"

// ============================================================================
// GPU 专用优化函数
// ============================================================================

namespace dev {

/// @brief GPU 优化的四舍五入（使用 CUDA 内置函数）
__device__ __forceinline__ int32_t round(float val) {
    return __float2int_rn(val);  // round to nearest
}

/**
 * @brief GPU 优化的量化函数（统一 int32_t 输出，通过位宽配置 clamp）
 *
 * 使用 __fdividef 进行快速除法优化。
 * 所有量化值统一使用 int32_t 存储，通过 clamp_by_bitwidth 限制实际值范围。
 *
 * @param src 浮点输入
 * @param exp2_inv 缩放因子指数
 * @param zp 零点
 * @param bw 位宽配置
 * @return 量化值（int32_t 存储，实际值在位宽范围内）
 */
inline __device__ int32_t quantize(float src, int8_t exp2_inv, int32_t zp, QuantBitWidth bw) {
    float scale = (exp2_inv >= 0) ? __fdividef(1.0f, static_cast<float>(1 << exp2_inv))
                                  : static_cast<float>(1 << (-exp2_inv));
    float shifted = src / scale + static_cast<float>(zp);
    int32_t q = round(shifted);
    return clamp_by_bitwidth(q, bw);
}

/**
 * @brief GPU 优化的量化函数（兼容旧接口，使用模板类型推断 clamp 范围）
 * @deprecated 建议使用 quantize(float, int8_t, int32_t, QuantBitWidth) 版本
 */
template <typename QuantT>
inline __device__ QuantT quantize(float src, int8_t exp2_inv, int32_t zp) {
    float scale = (exp2_inv >= 0) ? __fdividef(1.0f, static_cast<float>(1 << exp2_inv))
                                  : static_cast<float>(1 << (-exp2_inv));
    float shifted = src / scale + static_cast<float>(zp);
    int32_t q = round(shifted);
    return clamp_to_type<QuantT>(q);
}

// ============================================================================
// GRU 门激活函数
// ============================================================================
//
// 以下函数是对 piecewise_linear() 的封装，自动选择正确的 LUT。
// 核心计算函数（find_segment, piecewise_linear_raw, piecewise_linear）
// 已统一定义在 quantize_ops_helper.h 中。
//

/**
 * @brief Z 门 Sigmoid 激活（使用参数传递的 LUT，推荐）
 *
 * @param q_x 量化输入
 * @param lut Sigmoid LUT（从 QuantGRUReScale 中获取）
 * @param pre_bw 输入位宽
 * @param out_bw 输出位宽
 * @return 量化输出
 */
__device__ __forceinline__ int32_t sigmoid_z(int32_t q_x, const SigmoidLUT& lut, 
                                              QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    return ::piecewise_linear(q_x, lut, pre_bw, out_bw);
}

/**
 * @brief R 门 Sigmoid 激活（使用参数传递的 LUT，推荐）
 */
__device__ __forceinline__ int32_t sigmoid_r(int32_t q_x, const SigmoidLUT& lut,
                                              QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    return ::piecewise_linear(q_x, lut, pre_bw, out_bw);
}

/**
 * @brief G 门 Tanh 激活（使用参数传递的 LUT，推荐）
 */
__device__ __forceinline__ int32_t tanh_g(int32_t q_x, const SigmoidLUT& lut,
                                           QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    return ::piecewise_linear(q_x, lut, pre_bw, out_bw);
}

}  // namespace dev
