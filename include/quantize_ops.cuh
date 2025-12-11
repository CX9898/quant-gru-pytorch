#pragma once

#include <cuda_runtime.h>

#include <cstdint>

extern __constant__ int8_t d_sigmoid_int8_z_lut[256];
extern __constant__ int8_t d_sigmoid_int8_r_lut[256];
extern __constant__ int8_t d_tanh_int8_g_lut[256];

// ==================== 分段线性量化数据结构 ====================
#define NUM_SEGMENTS 16

// INT16 版本的段参数结构
struct SegmentParams_INT16 {
    int16_t q_b;                 // 量化后的系数 b (INT16)
    int8_t n_BX_total;           // 融合后的移位位数 (INT8，可能为负)
    int32_t term_c_precomputed;  // 预计算的 term_c (INT32)
    uint16_t threshold;          // 段阈值 (UINT16，量化后的输入值)
};

// Sigmoid/Tanh 查找表结构（INT16）
struct SigmoidLUT_INT16 {
    SegmentParams_INT16 segments[NUM_SEGMENTS];
    int16_t zp_x;         // 输入 zero-point (INT16)
    int8_t shift_bits_x;  // 输入 shift_bits (INT8)
    int8_t shift_bits_y;  // 输出 shift_bits (INT8)
    int16_t zp_y;         // 输出 zero-point (INT16)
};

// INT8 版本的段参数结构
struct SegmentParams_INT8 {
    int8_t q_b;                  // 量化后的系数 b (INT8)
    int8_t n_BX_total;           // 融合后的移位位数 (INT8，可能为负)
    int16_t term_c_precomputed;  // 预计算的 term_c (INT16)
    uint8_t threshold;           // 段阈值 (UINT8，量化后的输入值)
};

// Sigmoid/Tanh 查找表结构（INT8）
struct SigmoidLUT_INT8 {
    SegmentParams_INT8 segments[NUM_SEGMENTS];
    int8_t zp_x;          // 输入 zero-point (INT8)
    int8_t shift_bits_x;  // 输入 shift_bits (INT8)
    int8_t shift_bits_y;  // 输出 shift_bits (INT8)
    int8_t zp_y;          // 输出 zero-point (INT8)
};

// 常量内存声明（CUDA设备端）
extern __constant__ SigmoidLUT_INT16 d_sigmoid_z_lut_int16;  // z 门的 Sigmoid LUT
extern __constant__ SigmoidLUT_INT16 d_sigmoid_r_lut_int16;  // r 门的 Sigmoid LUT
extern __constant__ SigmoidLUT_INT16 d_tanh_lut_int16;
extern __constant__ SigmoidLUT_INT8 d_sigmoid_z_lut_int8;  // z 门的 Sigmoid LUT
extern __constant__ SigmoidLUT_INT8 d_sigmoid_r_lut_int8;  // r 门的 Sigmoid LUT
extern __constant__ SigmoidLUT_INT8 d_tanh_lut_int8;

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

// Round 函数：只负责四舍五入，不限制范围
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

__device__ __forceinline__ int8_t sigmoid_int8_lut(int8_t x, const int8_t* lut) {
    // x in [-128,127], lut 长度 = 256
    const int idx = static_cast<uint8_t>(x + 128);  // 对齐 LUT 初始化
    return lut[idx];
}

__device__ __forceinline__ int8_t tanh_int8_lut(int8_t x, const int8_t* lut) {
    const int idx = static_cast<uint8_t>(x + 128);  // 对齐 LUT 初始化
    return lut[static_cast<uint8_t>(idx)];
}

__device__ __forceinline__ int8_t sigmoid_int16_lut(int16_t x) {  // (TODO: 二项式拟合查表方式)
    // 将 int16_t 范围 [-32768, 32767] 映射到 int8_t 范围 [-128, 127]
    // 公式：idx = round( (x + 32768) * (255.0f / 65535.0f) ) - 128
    // 整数优化：避免浮点运算，用移位实现近似缩放
    int32_t tmp = static_cast<int32_t>(x) + 32768;  // 转为 [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535;          // 四舍五入缩放到 [0, 255]
    int8_t idx = static_cast<int8_t>(tmp - 128);    // 转为 [-128, 127]
    //    return d_sigmoid_lut[static_cast<uint8_t>(idx)];

    // -10到10分成N32段, 每段用二次多项式拟合

    // PDQ
    // QAT 训练
}

__device__ __forceinline__ int8_t tanh_int16_lut(int16_t x) {  // (TODO: 二项式拟合查表方式)
    // 与 sigmoid 完全相同的索引映射逻辑
    int32_t tmp = static_cast<int32_t>(x) + 32768;  // int16_t [-32768, 32767] → [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535;          // 缩放到 [0, 255]（四舍五入）
    int8_t idx = static_cast<int8_t>(tmp - 128);    // → [-128, 127]
    //    return d_tanh_lut[static_cast<uint8_t>(idx)]; // 用索引访问 tanh LUT
}

// ==================== 分段线性量化设备端函数 ====================

// 带符号右移（四舍五入）
__device__ __forceinline__ int32_t rshift_round(int32_t val, int8_t shift) {
    if (shift <= 0) return val;
    if (shift >= 32) return (val >= 0) ? 0 : -1;

    // 四舍五入：加上 1 << (shift - 1)
    int32_t round_val =
        (val >= 0) ? (val + (1 << (shift - 1))) >> shift : (val - (1 << (shift - 1))) >> shift;
    return round_val;
}

// 段查找函数（线性查找，32段足够快）
__device__ __forceinline__ int find_segment_int16(uint16_t q_x,
                                                  const SegmentParams_INT16* segments) {
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        if (q_x < segments[i].threshold) {
            return i;
        }
    }
    return NUM_SEGMENTS - 1;  // 返回最后一个段
}

// INT8 版本的段查找函数
__device__ __forceinline__ int find_segment_int8(uint8_t q_x, const SegmentParams_INT8* segments) {
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        if (q_x < segments[i].threshold) {
            return i;
        }
    }
    return NUM_SEGMENTS - 1;  // 返回最后一个段
}

// Sigmoid 分段线性计算（核心函数，接受 LUT 参数）
__device__ __forceinline__ uint16_t sigmoid_piecewise_linear_int16(uint16_t q_x,
                                                                   const SigmoidLUT_INT16& lut) {
    // [1] 段查找
    int seg_id = find_segment_int16(q_x, lut.segments);
    const SegmentParams_INT16& seg = lut.segments[seg_id];

    // [2] 去零点
    // 🔥 修正：使用 int32_t 避免溢出（q_x 是 uint16_t [0, 65535]，zp_x 可能是正数如 24576）
    int32_t x_offset = static_cast<int32_t>(q_x) - static_cast<int32_t>(lut.zp_x);

    // [3] 乘法 + 移位融合
    // 公式: term_bx = (q_b * x_offset) >> n_BX_total
    int32_t bx_32 = static_cast<int32_t>(seg.q_b) * static_cast<int32_t>(x_offset);

    int32_t term_bx;
    if (seg.n_BX_total >= 0) {
        // 右移
        term_bx = rshift_round(bx_32, seg.n_BX_total);
    } else {
        // 左移（n_BX_total < 0）
        term_bx = bx_32 << (-seg.n_BX_total);
    }

    // [4] 相加（term_c 已预计算）
    int32_t y_32 = term_bx + seg.term_c_precomputed;

    // [5] 饱和到 UINT16 范围 [0, 65535]
    int32_t q_y = max(0, min(65535, y_32));

    return static_cast<uint16_t>(q_y);
}

// Tanh 分段线性计算（类似实现，接受 LUT 参数）
__device__ __forceinline__ uint16_t tanh_piecewise_linear_int16(uint16_t q_x,
                                                                const SigmoidLUT_INT16& lut) {
    // 与 sigmoid 相同的计算流程
    int seg_id = find_segment_int16(q_x, lut.segments);
    const SegmentParams_INT16& seg = lut.segments[seg_id];

    // 🔥 修正：使用 int32_t 避免溢出（q_x 是 uint16_t [0, 65535]，zp_x 可能是正数如 24576）
    int32_t x_offset = static_cast<int32_t>(q_x) - static_cast<int32_t>(lut.zp_x);
    int32_t bx_32 = static_cast<int32_t>(seg.q_b) * x_offset;

    int32_t term_bx;
    if (seg.n_BX_total >= 0) {
        term_bx = rshift_round(bx_32, seg.n_BX_total);
    } else {
        term_bx = bx_32 << (-seg.n_BX_total);
    }

    int32_t y_32 = term_bx + seg.term_c_precomputed;
    int32_t q_y = max(0, min(65535, y_32));

    return static_cast<uint16_t>(q_y);
}

// Sigmoid 分段线性计算（UINT8 版本，接受 LUT 参数）
__device__ __forceinline__ uint8_t sigmoid_piecewise_linear_int8(uint8_t q_x,
                                                                 const SigmoidLUT_INT8& lut) {
    // [1] 段查找（输入已经是 uint8_t [0, 255]）
    int seg_id = find_segment_int8(q_x, lut.segments);
    const SegmentParams_INT8& seg = lut.segments[seg_id];

    // [2] 去零点
    // 🔥 修正：使用 int32_t 避免溢出（q_x 是 uint8_t [0, 255]，zp_x 可能是正数）
    int32_t x_offset = static_cast<int32_t>(q_x) - static_cast<int32_t>(lut.zp_x);

    // [3] 乘法 + 移位融合
    // 公式: term_bx = (q_b * x_offset) >> n_BX_total
    int32_t bx_32 = static_cast<int32_t>(seg.q_b) * x_offset;

    int32_t term_bx;
    if (seg.n_BX_total >= 0) {
        // 右移
        term_bx = rshift_round(bx_32, seg.n_BX_total);
    } else {
        // 左移（n_BX_total < 0）
        term_bx = bx_32 << (-seg.n_BX_total);
    }

    // [4] 相加（term_c 已预计算）
    int32_t y_32 = term_bx + static_cast<int32_t>(seg.term_c_precomputed);

    // [5] 饱和到 UINT8 范围 [0, 255]（根据 Python 参考，非对称量化使用无符号整数）
    int32_t q_y = max(0, min(255, y_32));

    return static_cast<uint8_t>(q_y);
}

// Tanh 分段线性计算（UINT8 版本，接受 LUT 参数）
__device__ __forceinline__ uint8_t tanh_piecewise_linear_int8(uint8_t q_x,
                                                              const SigmoidLUT_INT8& lut) {
    // 与 sigmoid 相同的计算流程
    // [1] 段查找（输入已经是 uint8_t [0, 255]）
    int seg_id = find_segment_int8(q_x, lut.segments);
    const SegmentParams_INT8& seg = lut.segments[seg_id];

    // [2] 去零点
    // 🔥 修正：使用 int32_t 避免溢出（q_x 是 uint8_t [0, 255]，zp_x 可能是正数）
    int32_t x_offset = static_cast<int32_t>(q_x) - static_cast<int32_t>(lut.zp_x);

    // [3] 乘法 + 移位融合
    // 公式: term_bx = (q_b * x_offset) >> n_BX_total
    int32_t bx_32 = static_cast<int32_t>(seg.q_b) * x_offset;

    int32_t term_bx;
    if (seg.n_BX_total >= 0) {
        term_bx = rshift_round(bx_32, seg.n_BX_total);
    } else {
        term_bx = bx_32 << (-seg.n_BX_total);
    }

    // [4] 相加（term_c 已预计算）
    int32_t y_32 = term_bx + static_cast<int32_t>(seg.term_c_precomputed);

    // [5] 饱和到 UINT8 范围 [0, 255]（根据 Python 参考，非对称量化使用无符号整数）
    int32_t q_y = max(0, min(255, y_32));

    return static_cast<uint8_t>(q_y);
}

}  // namespace dev
