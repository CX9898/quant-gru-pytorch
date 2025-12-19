#pragma once

#include <cublas_v2.h>

#include <cstdio>
#include <vector>

// ==================== 矩阵/张量操作工具函数 ====================

/**
 * @brief 使用 cuBLAS 进行 2D 矩阵转置: [rows, cols] -> [cols, rows]
 *
 * @param handle cuBLAS 句柄
 * @param A 输入矩阵 [rows x cols]
 * @param A_t 输出矩阵 [cols x rows]
 * @param rows 输入矩阵行数
 * @param cols 输入矩阵列数
 *
 * @note 使用 cublasSgeam 实现转置：C = alpha * op(A) + beta * op(B)
 *       transa = CUBLAS_OP_T 对 A 进行转置，beta=0 使 B 不参与计算
 */
inline void transpose2D(cublasHandle_t handle, const float *A, float *A_t, int rows, int cols) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cublasSgeam: C = alpha * op(A) + beta * op(B)
    // 将 A [cols, rows] 转置为 A_t [rows, cols]
    //
    // 输入 A: 原始矩阵形状 [cols, rows]（列优先存储，lda = cols）
    // 输出 A_t: 转置后矩阵形状 [rows, cols]（列优先存储，ldc = rows）
    //
    // cublasSgeam 参数说明:
    //   transa = CUBLAS_OP_T: 对 A 进行转置
    //   transb = CUBLAS_OP_N: B 不转置（但 beta=0 所以 B 不会被使用）
    //   m = rows: 输出矩阵 C 的行数
    //   n = cols: 输出矩阵 C 的列数
    //   lda = cols: A 的 leading dimension（A 的行数）
    //   ldb = rows: B 的 leading dimension（需要 >= m，即使 beta=0 也要有效）
    //   ldc = rows: C 的 leading dimension
    cublasStatus_t status =
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, A,
                    cols,              // A: [cols, rows], lda = cols
                    &beta, A_t, rows,  // B: 使用 A_t 作为占位符, ldb = rows (>= m)
                    A_t, rows);        // C: [rows, cols], ldc = rows
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgeam failed with status %d\n", status);
    }
}

/**
 * @brief 3D 张量 permute: [T, B, I] -> [I, T, B]
 *
 * @param src 源张量数据，布局 [T, B, I]
 * @param dst 目标张量数据，布局 [I, T, B]
 * @param T 第一维大小（时间步）
 * @param B 第二维大小（批大小）
 * @param I 第三维大小（特征维度）
 *
 * @note 使用 CPU 实现，适合在初始化阶段调用（非性能关键路径）
 */
inline void permute3D_TBI_to_ITB(const std::vector<float> &src, std::vector<float> &dst, int T,
                                 int B, int I) {
    dst.resize(I * T * B);
    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < I; ++i) {
                // src[t, b, i] = src[t * B * I + b * I + i]
                // dst[i, t, b] = dst[i * T * B + t * B + b]
                dst[i * T * B + t * B + b] = src[t * B * I + b * I + i];
            }
        }
    }
}

/**
 * @brief 3D 张量 permute（模板版本）: [T, B, I] -> [I, T, B]
 *
 * @tparam T_elem 元素类型
 * @param src 源张量数据，布局 [T, B, I]
 * @param dst 目标张量数据，布局 [I, T, B]
 * @param T 第一维大小（时间步）
 * @param B 第二维大小（批大小）
 * @param I 第三维大小（特征维度）
 */
template <typename T_elem>
inline void permute3D_TBI_to_ITB(const std::vector<T_elem> &src, std::vector<T_elem> &dst, int T,
                                 int B, int I) {
    dst.resize(I * T * B);
    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < I; ++i) {
                dst[i * T * B + t * B + b] = src[t * B * I + b * I + i];
            }
        }
    }
}

