#pragma once

#include <cublas_v2.h>
#include <vector>

#include "gru.h"
#include "gru_quant.h"

// 初始化函数，供Python绑定调用
void init_gru_cublas(cublasHandle_t &g_blas_handle) {
    if (g_blas_handle == nullptr) {
        cublasCreate(&g_blas_handle);
    }
}

void calibrateGruScales(
    bool use_int16,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const std::vector<float> &W,
    const std::vector<float> &R,
    const std::vector<float> &bx,
    const std::vector<float> &br,
    const std::vector<float> &x,
    const cublasHandle_t &g_blas_handle,
    GRUQuantitativeParameters &quant_gru_scales);

GRUQuantitativeParameters calibrateGruScales(
    bool use_int16,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W,
    const float *R,
    const float *bx,
    const float *br,
    const float *x,
    const cublasHandle_t &g_blas_handle);

template<typename QuantT>
void quantitativeWeight(const int input_size, const int hidden_size,
                        const float *W, const float *R, const float *bx, const float *br,
                        const GRUQuantitativeParameters &quant_parms,
                        QuantT *W_quant, QuantT *R_quant, int32_t *bx_quant, int32_t *br_quant);

template<typename QuantT>
void quantGRUForward(const int time_steps, const int batch_size, const int input_size,
                     const int hidden_size, const QuantT *W, const QuantT *R, const int32_t *bx,
                     const int32_t *br, const float *x,
                     const GRUQuantitativeParameters &quant_parms,
                     const cublasHandle_t &g_blas_handle,
                     float *h// (time_steps) * batch_size * hidden_size
);

void hasteGRUForward(const int time_steps,
                     const int batch_size,
                     const int input_size,
                     const int hidden_size,
                     const float *W, const float *R, const float *bx,
                     const float *br, const float *x,
                     const cublasHandle_t &g_blas_handle,
                     float *h);

void forwardInterface(bool is_quant,
                      bool use_int16,
                      int time_steps, int batch_size, int input_size, int hidden_size,
                      const float *W,
                      const float *R,
                      const float *bx,
                      const float *br,
                      const float *x,
                      const GRUQuantitativeParameters &quant_gru_scales,
                      const cublasHandle_t &g_blas_handle,
                      float *h);
