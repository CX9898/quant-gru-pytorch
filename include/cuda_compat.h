#pragma once

// ============================================================================
//                       CUDA/C++ 兼容性宏定义
// ============================================================================
//
// 用途：在纯 C++ 编译器中定义 CUDA 修饰符为空，解决兼容性问题
//
// 使用方法：在需要使用 __host__ __device__ 等修饰符的头文件开头包含此文件
//
// ============================================================================

#ifndef __CUDACC__
    // 纯 C++ 编译时，定义 CUDA 修饰符为空
    #ifndef __host__
        #define __host__
    #endif
    #ifndef __device__
        #define __device__
    #endif
    #ifndef __forceinline__
        #define __forceinline__ inline
    #endif
    #ifndef __global__
        #define __global__
    #endif
#endif
