#pragma once

#include <cmath>

#include "cuda_compat.h"

// ============================================================================
// 通用激活函数（CPU/GPU 共用）
// ============================================================================

template <typename T>
__host__ __device__ __forceinline__ T sigmoid(const T x) {
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x));
}

template <typename T>
__host__ __device__ __forceinline__ T tanh(const T x) {
    return std::tanh(x);
}

template <typename T>
__host__ __device__ __forceinline__ T d_sigmoid(const T sigmoid_output) {
    return sigmoid_output * (static_cast<T>(1.0) - sigmoid_output);
}

template <typename T>
__host__ __device__ __forceinline__ T d_tanh(const T tanh_output) {
    return (static_cast<T>(1.0) - tanh_output * tanh_output);
}

// ============================================================================
// double 类型 atomicAdd 兼容（CUDA < 6.0 不支持原生 double atomicAdd）
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)

__device__ __forceinline__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif
