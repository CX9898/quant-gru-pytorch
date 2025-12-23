#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "quantize_bitwidth_config.h"
#include "quantize_ops_helper.h"

// ==================== 分段线性量化数据结构 ====================
#define NUM_SEGMENTS 16

// INT16 版本的段参数结构
struct SegmentParams_INT16 {
    int32_t q_b;                 // 量化后的系数 b (INT32，避免溢出截断)
    int8_t n_BX_total;           // 融合后的移位位数 (INT8，可能为负)
    int32_t term_c_precomputed;  // 预计算的 term_c (INT32)
    int16_t threshold;           // 段阈值 (INT16，量化后的输入值)
};

// Sigmoid/Tanh 查找表结构（INT16）
struct SigmoidLUT_INT16 {
    SegmentParams_INT16 segments[NUM_SEGMENTS];
    int32_t zp_x;         // 输入 zero-point (INT32)
    int8_t shift_bits_x;  // 输入 shift_bits (INT8)
    int8_t shift_bits_y;  // 输出 shift_bits (INT8)
    int32_t zp_y;         // 输出 zero-point (INT32)
};

// INT8 版本的段参数结构
struct SegmentParams_INT8 {
    int8_t q_b;                  // 量化后的系数 b (INT8)
    int8_t n_BX_total;           // 融合后的移位位数 (INT8，可能为负)
    int16_t term_c_precomputed;  // 预计算的 term_c (INT16)
    int8_t threshold;            // 段阈值 (INT8，量化后的输入值)
};

// Sigmoid/Tanh 查找表结构（INT8）
struct SigmoidLUT_INT8 {
    SegmentParams_INT8 segments[NUM_SEGMENTS];
    int32_t zp_x;         // 输入 zero-point (INT32)
    int8_t shift_bits_x;  // 输入 shift_bits (INT8)
    int8_t shift_bits_y;  // 输出 shift_bits (INT8)
    int32_t zp_y;         // 输出 zero-point (INT32)
};

// ==================== INT8→INT16 混合版本 ====================
// 输入: INT8 (低存储开销)
// 输出: INT16/UINT16 (高精度输出)
// 适用场景: 需要高精度激活函数输出，但输入可以用低位宽表示

// INT8→INT16 混合版本的段参数结构
struct SegmentParams_INT8_to_INT16 {
    int16_t q_b;                 // 量化后的系数 b (INT16，支持更高精度斜率)
    int8_t n_BX_total;           // 融合后的移位位数 (INT8，可能为负)
    int32_t term_c_precomputed;  // 预计算的 term_c (INT32，支持INT16输出范围)
    int8_t threshold;            // 段阈值 (INT8，量化后的输入值)
};

// Sigmoid/Tanh 查找表结构（INT8→INT16）
struct SigmoidLUT_INT8_to_INT16 {
    SegmentParams_INT8_to_INT16 segments[NUM_SEGMENTS];
    int32_t zp_x;         // 输入 zero-point (INT32)
    int8_t shift_bits_x;  // 输入 shift_bits (INT8)
    int8_t shift_bits_y;  // 输出 shift_bits (INT8)
    int32_t zp_y;         // 输出 zero-point (INT32)
};

// 常量内存声明（CUDA设备端）
extern __constant__ SigmoidLUT_INT16 d_sigmoid_z_lut_int16;  // z 门的 Sigmoid LUT
extern __constant__ SigmoidLUT_INT16 d_sigmoid_r_lut_int16;  // r 门的 Sigmoid LUT
extern __constant__ SigmoidLUT_INT16 d_tanh_lut_int16;
extern __constant__ SigmoidLUT_INT8 d_sigmoid_z_lut_int8;  // z 门的 Sigmoid LUT
extern __constant__ SigmoidLUT_INT8 d_sigmoid_r_lut_int8;  // r 门的 Sigmoid LUT
extern __constant__ SigmoidLUT_INT8 d_tanh_lut_int8;
// INT8→INT16 混合版本
extern __constant__ SigmoidLUT_INT8_to_INT16 d_sigmoid_z_lut_int8_to_int16;
extern __constant__ SigmoidLUT_INT8_to_INT16 d_sigmoid_r_lut_int8_to_int16;
extern __constant__ SigmoidLUT_INT8_to_INT16 d_tanh_lut_int8_to_int16;

namespace dev {

template <typename T>
__device__ __forceinline__ T clamp(int x);

template <>
__device__ __forceinline__ int8_t clamp(int x) {
    return static_cast<int8_t>(max(-128, min(127, x)));
}

template <>
__device__ __forceinline__ int16_t clamp(int x) {
    return static_cast<int16_t>(max(-32768, min(32767, x)));
}

template <>
__device__ __forceinline__ int32_t clamp(int x) {
    // 使用 static_cast 确保字面量类型正确（-2147483648 会被识别为 long long）
    constexpr int32_t min_val = static_cast<int32_t>(-2147483648LL);
    constexpr int32_t max_val = 2147483647;
    return max(min_val, min(max_val, x));
}

template <>
__device__ __forceinline__ uint8_t clamp(int x) {
    return static_cast<uint8_t>(max(0, min(255, x)));
}

// uint16_t 特化（用于 sigmoid 输出 [0, 65535]）
template <>
__device__ __forceinline__ uint16_t clamp(int x) {
    return static_cast<uint16_t>(max(0, min(65535, x)));
}

// Round 函数
__device__ __forceinline__ int32_t round(float val) {
    // 使用 CUDA 内置函数 __float2int_rn 进行四舍五入（round to nearest）
    // 这比 roundf 更高效，因为它直接返回整数
    return __float2int_rn(val);
}

template <typename T>
struct QuantLimits;

template <>
struct QuantLimits<int8_t> {
    static __device__ __forceinline__ constexpr int32_t min() { return -128; }

    static __device__ __forceinline__ constexpr int32_t max() { return 127; }
};

template <>
struct QuantLimits<int16_t> {
    static __device__ __forceinline__ constexpr int32_t min() { return -32768; }

    static __device__ __forceinline__ constexpr int32_t max() { return 32767; }
};

// int32_t 特化
template <>
struct QuantLimits<int32_t> {
    static __host__ __device__ constexpr int min() {
        // 使用 LL 后缀确保类型正确，然后转换为 int32_t
        return static_cast<int32_t>(-2147483648LL);
    }

    static __host__ __device__ constexpr int max() { return 2147483647; }
};

// uint8_t 特化（用于 sigmoid 输出）
template <>
struct QuantLimits<uint8_t> {
    static __device__ __forceinline__ constexpr int32_t min() { return 0; }

    static __device__ __forceinline__ constexpr int32_t max() { return 255; }
};

// uint16_t 特化（用于 sigmoid 输出）
template <>
struct QuantLimits<uint16_t> {
    static __device__ __forceinline__ constexpr int32_t min() { return 0; }

    static __device__ __forceinline__ constexpr int32_t max() { return 65535; }
};

template <typename QuantT>
inline __device__ QuantT quantize(float src, int8_t exp2_inv, int32_t zp) {
    // CUDA device code: 与CPU版本保持一致，使用位运算
    // 量化公式：q = round(src / scale + zp)
    float scale;
    if (exp2_inv >= 0) {
        // scale = 2^(-exp2) = 1 / (1 << exp2)
        scale = __fdividef(1.0f, static_cast<float>(1 << exp2_inv));
    } else {
        // scale = 2^(-(-x)) = 2^x = (1 << -exp2_inv)
        scale = static_cast<float>(1 << (-exp2_inv));
    }
    // 正确的量化流程：先计算 src/scale + zp，然后四舍五入，最后截断到目标类型范围
    float shifted = src / scale + static_cast<float>(zp);
    int32_t q = round(shifted);  // 四舍五入
    q = clamp<QuantT>(q);        // 截断到目标量化类型的范围

    return static_cast<QuantT>(q);
}

// sigmoid LUT 查找函数：输入为 int8_t（有符号），输出为 uint8_t（无符号，因为 sigmoid ∈ [0,1]）
__device__ __forceinline__ uint8_t sigmoid_int8_lut(int8_t x, const uint8_t* lut) {
    // x in [-128,127], lut 长度 = 256
    const int idx = static_cast<uint8_t>(x + 128);  // 对齐 LUT 初始化
    return lut[idx];
}

__device__ __forceinline__ int8_t tanh_int8_lut(int8_t x, const int8_t* lut) {
    const int idx = static_cast<uint8_t>(x + 128);  // 对齐 LUT 初始化
    return lut[static_cast<uint8_t>(idx)];
}

// ==================== 分段线性量化设备端函数 ====================
//
// 【原理】将非线性函数(Sigmoid/Tanh)在每个分段内用线性函数 y = b*x + c 近似
//
// 【浮点公式】 y_fp = b_fp * x_fp + c_fp
//
// 【量化公式】 q_y = (q_b * (q_x - zp_x)) >> n_BX_total + term_c_precomputed
//
// 【参数说明】
//   q_x           : 量化输入（有符号整数 INT8/INT16）
//   zp_x          : 输入零点
//   q_b           : 量化斜率（有符号，对称量化）
//   n_BX_total    : 融合移位 = shift_bits_b + shift_bits_x - shift_bits_y
//   term_c_precomputed : 预计算常数项 = q_c >> (shift_bits_c - shift_bits_y)
//                        其中 q_c 已包含输出零点烘焙: c_adjusted = c_fp + zp_y * scale_y
//
// 【计算步骤】
//   Step 1: seg_id = find_segment(q_x)           // 段查找
//   Step 2: x_offset = q_x - zp_x                // 去零点
//   Step 3: bx = q_b * x_offset                  // 乘法（INT32）
//   Step 4: term_bx = bx >> n_BX_total           // 移位（右移或左移）
//   Step 5: q_y = term_bx + term_c_precomputed   // 相加
//   Step 6: q_y = clamp(q_y, Q_MIN, Q_MAX)       // 饱和
//
// =========================================================================

/**
 * @brief 段查找函数
 * @tparam SegParamsT 段参数类型
 * @param q_x 量化输入（int32_t，与 threshold 比较时自动类型提升）
 * @note 线性查找，对于 NUM_SEGMENTS=16 足够快
 */
template <typename SegParamsT>
__device__ __forceinline__ int find_segment(int32_t q_x, const SegParamsT* segments) {
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        if (q_x < segments[i].threshold) {
            return i;
        }
    }
    return NUM_SEGMENTS - 1;
}

// ==================== 分段线性量化核心函数 ====================
//
// 【设计理念】
//   kernel 中用 int32_t 存储所有中间结果，clamp 只是逻辑上的区别。
//   因此提供两层 API：
//     1. 核心函数：返回 int32_t，不做 clamp（最大灵活性）
//     2. 模板包装：按需 clamp 到目标类型（便捷使用）
//
// 【使用方式】
//   // 方式1：核心函数 + 延迟 clamp（推荐用于 kernel 内部）
//   int32_t y_raw = piecewise_linear_int16_raw(q_x, lut);
//   // ... 后续计算 ...
//   int32_t y_final = clamp_to<uint16_t>(y_raw);  // 存储前 clamp
//
//   // 方式2：模板包装，一步到位
//   uint16_t y = piecewise_linear_int16<uint16_t>(q_x, lut);
//
// =========================================================================

/**
 * @brief clamp 到目标类型范围（模板版本）
 * @tparam T 目标类型，支持 int8_t/uint8_t/int16_t/uint16_t
 */
template <typename T>
__device__ __forceinline__ int32_t clamp_to(int32_t val) {
    return max(QuantLimits<T>::min(), min(QuantLimits<T>::max(), val));
}

/**
 * @brief 根据位宽枚举 clamp 到对应范围（运行时版本）
 * @param val 原始值
 * @param bw 位宽枚举
 * @return clamp 后的值
 */
__device__ __forceinline__ int32_t clamp_by_bitwidth(int32_t val, QuantBitWidth bw) {
    switch (bw) {
        case QuantBitWidth::INT8:
            return max(-128, min(127, val));
        case QuantBitWidth::INT16:
            return max(-32768, min(32767, val));
        case QuantBitWidth::UINT8:
            return max(0, min(255, val));
        case QuantBitWidth::UINT16:
            return max(0, min(65535, val));
        case QuantBitWidth::INT32:
        default:
            return val;  // 不 clamp
    }
}

// ==================== 核心函数：返回 int32_t，不做 clamp ====================

/**
 * @brief 分段线性近似核心
 * @tparam LutT LUT 类型 (SigmoidLUT_INT8, SigmoidLUT_INT16, SigmoidLUT_INT8_to_INT16)
 * @param q_x 量化输入（int32_t，应已 clamp 到正确范围）
 * @param lut 分段线性查找表
 * @return int32_t 原始结果，未 clamp
 *
 * 【自动适配 q_b 类型】
 *   - q_b 是 int8_t 时，使用 int32 乘法
 *   - q_b 是 int16_t/int32_t 时，使用 int64 乘法（避免溢出）
 */
template <typename LutT>
__device__ __forceinline__ int32_t piecewise_linear_raw(int32_t q_x, const LutT& lut) {
    int seg_id = find_segment(q_x, lut.segments);
    const auto& seg = lut.segments[seg_id];

    int32_t x_offset = q_x - static_cast<int32_t>(lut.zp_x);

    // 根据 q_b 类型选择乘法精度
    int32_t term_bx;
    if constexpr (sizeof(decltype(seg.q_b)) > 1) {
        // q_b 是 int16_t 或 int32_t，需要 int64 避免溢出
        int64_t bx_64 = static_cast<int64_t>(seg.q_b) * static_cast<int64_t>(x_offset);
        term_bx = (seg.n_BX_total >= 0)
                      ? static_cast<int32_t>(rshift_round(bx_64, seg.n_BX_total))
                      : static_cast<int32_t>(bx_64 << (-seg.n_BX_total));
    } else {
        // q_b 是 int8_t，int32 就够了
        int32_t bx_32 = static_cast<int32_t>(seg.q_b) * x_offset;
        term_bx = (seg.n_BX_total >= 0) ? rshift_round(bx_32, seg.n_BX_total)
                                        : (bx_32 << (-seg.n_BX_total));
    }

    return term_bx + static_cast<int32_t>(seg.term_c_precomputed);
}

// ==================== 包装函数：核心函数 + clamp_by_bitwidth ====================

/**
 * @brief 通用分段线性近似（int32 输入，双位宽 clamp，需外部指定 LUT）
 */
template <typename LutT>
__device__ __forceinline__ int32_t piecewise_linear(int32_t q_x, const LutT& lut,
                                                    QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    int32_t q_x_clamped = clamp_by_bitwidth(q_x, pre_bw);
    int32_t result = piecewise_linear_raw(q_x_clamped, lut);
    return clamp_by_bitwidth(result, out_bw);
}

// ==================== 门专用函数：自动选择 LUT ====================

/**
 * @brief Z 门 Sigmoid（自动选择 LUT）
 * @param q_x 量化输入（int32_t）
 * @param pre_bw 输入位宽
 * @param out_bw 输出位宽
 */
__device__ __forceinline__ int32_t sigmoid_z(int32_t q_x, QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    if (pre_bw == QuantBitWidth::INT16) {
        return piecewise_linear(q_x, d_sigmoid_z_lut_int16, pre_bw, out_bw);
    } else if (out_bw == QuantBitWidth::UINT16 || out_bw == QuantBitWidth::INT16) {
        return piecewise_linear(q_x, d_sigmoid_z_lut_int8_to_int16, pre_bw, out_bw);
    } else {
        return piecewise_linear(q_x, d_sigmoid_z_lut_int8, pre_bw, out_bw);
    }
}

/**
 * @brief R 门 Sigmoid（自动选择 LUT）
 */
__device__ __forceinline__ int32_t sigmoid_r(int32_t q_x, QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    if (pre_bw == QuantBitWidth::INT16) {
        return piecewise_linear(q_x, d_sigmoid_r_lut_int16, pre_bw, out_bw);
    } else if (out_bw == QuantBitWidth::UINT16 || out_bw == QuantBitWidth::INT16) {
        return piecewise_linear(q_x, d_sigmoid_r_lut_int8_to_int16, pre_bw, out_bw);
    } else {
        return piecewise_linear(q_x, d_sigmoid_r_lut_int8, pre_bw, out_bw);
    }
}

/**
 * @brief G 门 Tanh（自动选择 LUT）
 */
__device__ __forceinline__ int32_t tanh_g(int32_t q_x, QuantBitWidth pre_bw, QuantBitWidth out_bw) {
    if (pre_bw == QuantBitWidth::INT16) {
        return piecewise_linear(q_x, d_tanh_lut_int16, pre_bw, out_bw);
    } else if (out_bw == QuantBitWidth::INT16) {
        return piecewise_linear(q_x, d_tanh_lut_int8_to_int16, pre_bw, out_bw);
    } else {
        return piecewise_linear(q_x, d_tanh_lut_int8, pre_bw, out_bw);
    }
}

}  // namespace dev
