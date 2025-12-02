#pragma once

#include <cuda_runtime.h>
#include <cstdint>

extern __constant__ int8_t d_sigmoid_int8_z_lut[256];
extern __constant__ int8_t d_sigmoid_int8_r_lut[256];
extern __constant__ int8_t d_tanh_int8_g_lut[256];

namespace dev {

template<typename T>
__device__ __forceinline__ T clamp(int x);

template<>
__device__ __forceinline__ int8_t clamp(int x) {
    return static_cast<int8_t>(max(-128, min(127, x)));
}

template<>
__device__ __forceinline__ int16_t clamp(int x) {
    return static_cast<int8_t>(max(-32768, min(32767, x)));
}

template<>
__device__ __forceinline__ int32_t clamp(int x) {
    return static_cast<int8_t>(max(-32768, min(32767, x)));
}

template<typename T>
__device__ __forceinline__ T round(float val) {
    return clamp<T>(static_cast<int>(roundf(val)));
}

template<typename T>
struct QuantLimits;

template<>
struct QuantLimits<int8_t> {
  static __device__ __forceinline__ constexpr int32_t min() { return -128; }

  static __device__ __forceinline__ constexpr int32_t max() { return 127; }
};

template<>
struct QuantLimits<int16_t> {
  static __device__ __forceinline__ constexpr int32_t min() { return -32768; }

  static __device__ __forceinline__ constexpr int32_t max() { return 32767; }
};

// int32_t 特化
template<>
struct QuantLimits<int32_t> {
  static __host__ __device__ constexpr int min() { return -2147483648; }

  static __host__ __device__ constexpr int max() { return 2147483647; }
};


template<typename QuantT>
inline __device__ QuantT quantize(float src, int32_t exp2_inv, int32_t zp) {
    // CUDA device code: 与CPU版本保持一致，使用位运算
    float scale;
    if (exp2_inv >= 0) {
        // scale = 2^(-exp2) = 1 / (1 << exp2)
        scale = __fdividef(1.0f, static_cast<float>(1 << exp2_inv));
    } else {
        // scale = 2^(-(-x)) = 2^x = (1 << -exp2_inv)
        scale = static_cast<float>(1 << (-exp2_inv));
    }
    int32_t q = __float2int_rn(src / scale) + zp;

    q = clamp<QuantT>(q);

    return static_cast<QuantT>(q);
}

__device__ __forceinline__ int8_t sigmoid_int8_lut(int8_t x, const int8_t *lut) {
    // x in [-128,127], lut 长度 = 256
    const int idx = static_cast<uint8_t>(x + 128); // 对齐 LUT 初始化
    return lut[idx];
}

__device__ __forceinline__ int8_t tanh_int8_lut(int8_t x, const int8_t *lut) {
    const int idx = static_cast<uint8_t>(x + 128); // 对齐 LUT 初始化
    return lut[static_cast<uint8_t>(idx)];
}

__device__ __forceinline__ int8_t sigmoid_int16_lut(int16_t x) { // (TODO: 二项式拟合查表方式)
    // 将 int16_t 范围 [-32768, 32767] 映射到 int8_t 范围 [-128, 127]
    // 公式：idx = round( (x + 32768) * (255.0f / 65535.0f) ) - 128
    // 整数优化：避免浮点运算，用移位实现近似缩放
    int32_t tmp = static_cast<int32_t>(x) + 32768; // 转为 [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535; // 四舍五入缩放到 [0, 255]
    int8_t idx = static_cast<int8_t>(tmp - 128); // 转为 [-128, 127]
//    return d_sigmoid_lut[static_cast<uint8_t>(idx)];

    // -10到10分成N32段, 每段用二次多项式拟合

    // PDQ
    // QAT 训练
}

__device__ __forceinline__ int8_t tanh_int16_lut(int16_t x) { // (TODO: 二项式拟合查表方式)
    // 与 sigmoid 完全相同的索引映射逻辑
    int32_t tmp = static_cast<int32_t>(x) + 32768; // int16_t [-32768, 32767] → [0, 65535]
    tmp = (tmp * 255 + 65535 / 2) / 65535; // 缩放到 [0, 255]（四舍五入）
    int8_t idx = static_cast<int8_t>(tmp - 128); // → [-128, 127]
//    return d_tanh_lut[static_cast<uint8_t>(idx)]; // 用索引访问 tanh LUT
}

} // dev namespace
