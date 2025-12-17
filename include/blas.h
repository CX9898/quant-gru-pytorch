#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

template <typename T>
struct blas {
    struct set_pointer_mode {
        set_pointer_mode(cublasHandle_t handle) : handle_(handle) {
            cublasGetPointerMode(handle_, &old_mode_);
            cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
        }

        ~set_pointer_mode() { cublasSetPointerMode(handle_, old_mode_); }

       private:
        cublasHandle_t handle_;
        cublasPointerMode_t old_mode_;
    };

    struct enable_tensor_cores {
        enable_tensor_cores(cublasHandle_t handle) : handle_(handle) {
            cublasGetMathMode(handle_, &old_mode_);
            cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
        }

        ~enable_tensor_cores() { cublasSetMathMode(handle_, old_mode_); }

       private:
        cublasHandle_t handle_;
        cublasMath_t old_mode_;
    };
};

inline cublasStatus_t int8_gemm_wrapper(cublasHandle_t handle, cublasOperation_t transa,
                                        cublasOperation_t transb, int m, int n, int k,
                                        const int32_t *alpha, const void *A, int lda, const void *B,
                                        int ldb, const int32_t *beta, void *C, int ldc) {
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_8I, lda, B, CUDA_R_8I,
                        ldb, beta, C, CUDA_R_32I, ldc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
}

// INT8 GEMM to int64: 直接用 int32 结果转 int64（int8*int8*256 不会溢出 int32）
inline cublasStatus_t int8_gemm_to_int64(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const int64_t *alpha, const void *A, int lda,
                                          const void *B, int ldb, const int64_t *beta, void *C,
                                          int ldc) {
    // 先分配 int32 缓冲区
    int32_t *C_i32;
    cudaMalloc(&C_i32, sizeof(int32_t) * m * n);
    
    int32_t alpha32 = static_cast<int32_t>(*alpha);
    int32_t beta32 = static_cast<int32_t>(*beta);
    
    cublasStatus_t status = int8_gemm_wrapper(handle, transa, transb, m, n, k, 
                                               &alpha32, A, lda, B, ldb, &beta32, C_i32, ldc);
    
    if (status == CUBLAS_STATUS_SUCCESS) {
        // 将 int32 转为 int64
        int32_t *C_i32_host = new int32_t[m * n];
        int64_t *C_i64_host = new int64_t[m * n];
        cudaMemcpy(C_i32_host, C_i32, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost);
        for (int i = 0; i < m * n; i++) {
            C_i64_host[i] = static_cast<int64_t>(C_i32_host[i]);
        }
        cudaMemcpy(C, C_i64_host, sizeof(int64_t) * m * n, cudaMemcpyHostToDevice);
        delete[] C_i32_host;
        delete[] C_i64_host;
    }
    
    cudaFree(C_i32);
    return status;
}

template <>
struct blas<int8_t> {
    static constexpr auto *gemm = &int8_gemm_wrapper;
    static constexpr auto *gemm_to_int64 = &int8_gemm_to_int64;
};

// INT16 GEMM: 输出 int64 以避免溢出
// 使用 double GEMM 保持精度，结果转为 int64
inline cublasStatus_t int16_gemm_to_int64(cublasHandle_t handle, cublasOperation_t transa,
                                          cublasOperation_t transb, int m, int n, int k,
                                          const int64_t *alpha, const void *A, int lda,
                                          const void *B, int ldb, const int64_t *beta, void *C,
                                          int ldc) {
    // 使用 double GEMM 以保持精度（int16*int16*256 需要约 33 位，double 有 53 位尾数）
    double *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(double) * m * k);
    cudaMalloc(&B_d, sizeof(double) * k * n);
    cudaMalloc(&C_d, sizeof(double) * m * n);
    
    // 初始化为 0 以确保确定性
    cudaMemset(C_d, 0, sizeof(double) * m * n);
    
    // CPU 转换（生产环境应改用 GPU kernel）
    int16_t *A_host = new int16_t[m * k];
    int16_t *B_host = new int16_t[k * n];
    double *A_d_host = new double[m * k];
    double *B_d_host = new double[k * n];
    
    cudaMemcpy(A_host, A, sizeof(int16_t) * m * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_host, B, sizeof(int16_t) * k * n, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < m * k; i++) A_d_host[i] = static_cast<double>(A_host[i]);
    for (int i = 0; i < k * n; i++) B_d_host[i] = static_cast<double>(B_host[i]);
    
    cudaMemcpy(A_d, A_d_host, sizeof(double) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_d_host, sizeof(double) * k * n, cudaMemcpyHostToDevice);
    
    // double GEMM
    double alpha_d = static_cast<double>(*alpha);
    double beta_d = static_cast<double>(*beta);
    cublasStatus_t status = cublasDgemm(handle, transa, transb, m, n, k, &alpha_d, A_d, lda, B_d, ldb, &beta_d, C_d, ldc);
    
    if (status == CUBLAS_STATUS_SUCCESS) {
        // 将 double 结果转为 int64
        double *C_d_host = new double[m * n];
        int64_t *C_host = new int64_t[m * n];
        cudaMemcpy(C_d_host, C_d, sizeof(double) * m * n, cudaMemcpyDeviceToHost);
        for (int i = 0; i < m * n; i++) {
            C_host[i] = static_cast<int64_t>(round(C_d_host[i]));
        }
        cudaMemcpy(C, C_host, sizeof(int64_t) * m * n, cudaMemcpyHostToDevice);
        delete[] C_d_host;
        delete[] C_host;
    }
    
    delete[] A_host;
    delete[] B_host;
    delete[] A_d_host;
    delete[] B_d_host;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    return status;
}

// INT16 GEMM fallback: 输出 int32（会有溢出风险）
inline cublasStatus_t int16_gemm_wrapper(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const int32_t *alpha, const void *A, int lda,
                                         const void *B, int ldb, const int32_t *beta, void *C,
                                         int ldc) {
    // 首先尝试原生 int16 GEMM
    cublasStatus_t status = cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16I, lda, B, CUDA_R_16I,
                        ldb, beta, C, CUDA_R_32I, ldc, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
    
    if (status == CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    
    // 原生 int16 不支持，使用 float fallback（有溢出风险）
    static bool warned = false;
    if (!warned) {
        printf("[WARNING] CUDA_R_16I GEMM not supported (status=%d), using float fallback with potential overflow\n", status);
        printf("[WARNING] Consider using int16_gemm_to_int64() for 16-bit quantization\n");
        warned = true;
    }
    
    float *A_f, *B_f, *C_f;
    cudaMalloc(&A_f, sizeof(float) * m * k);
    cudaMalloc(&B_f, sizeof(float) * k * n);
    cudaMalloc(&C_f, sizeof(float) * m * n);
    
    int16_t *A_host = new int16_t[m * k];
    int16_t *B_host = new int16_t[k * n];
    float *A_f_host = new float[m * k];
    float *B_f_host = new float[k * n];
    
    cudaMemcpy(A_host, A, sizeof(int16_t) * m * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_host, B, sizeof(int16_t) * k * n, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < m * k; i++) A_f_host[i] = static_cast<float>(A_host[i]);
    for (int i = 0; i < k * n; i++) B_f_host[i] = static_cast<float>(B_host[i]);
    
    cudaMemcpy(A_f, A_f_host, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(B_f, B_f_host, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    
    float alpha_f = static_cast<float>(*alpha);
    float beta_f = static_cast<float>(*beta);
    status = cublasSgemm(handle, transa, transb, m, n, k, &alpha_f, A_f, lda, B_f, ldb, &beta_f, C_f, ldc);
    
    if (status == CUBLAS_STATUS_SUCCESS) {
        float *C_f_host = new float[m * n];
        int32_t *C_host = new int32_t[m * n];
        cudaMemcpy(C_f_host, C_f, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        for (int i = 0; i < m * n; i++) {
            float val = roundf(C_f_host[i]);
            if (val > 2147483647.0f) val = 2147483647.0f;
            if (val < -2147483648.0f) val = -2147483648.0f;
            C_host[i] = static_cast<int32_t>(val);
        }
        cudaMemcpy(C, C_host, sizeof(int32_t) * m * n, cudaMemcpyHostToDevice);
        delete[] C_f_host;
        delete[] C_host;
    }
    
    delete[] A_host;
    delete[] B_host;
    delete[] A_f_host;
    delete[] B_f_host;
    cudaFree(A_f);
    cudaFree(B_f);
    cudaFree(C_f);
    
    return status;
}

template <>
struct blas<int16_t> {
    static constexpr auto *gemm = &int16_gemm_wrapper;
    static constexpr auto *gemm_to_int64 = &int16_gemm_to_int64;  // 无溢出版本
};

template <>
struct blas<__half> {
    static constexpr decltype(cublasHgemm) *gemm = &cublasHgemm;
};

template <>
struct blas<float> {
    static constexpr decltype(cublasSgemm) *gemm = &cublasSgemm;
};

template <>
struct blas<double> {
    static constexpr decltype(cublasDgemm) *gemm = &cublasDgemm;
};
