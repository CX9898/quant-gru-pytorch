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
void quantGRUForward(bool is_training,  // 是否开启训练模式，true为训练，false为推理
                     const int time_steps, const int batch_size, const int input_size,
                     const int hidden_size, const QuantT *W, const QuantT *R, const int32_t *bx,
                     const int32_t *br, const float *x,
                     const float *h0,  // 初始隐藏状态，可以为 nullptr
                     const GRUQuantitativeParameters &quant_parms,
                     const cublasHandle_t &g_blas_handle,
                     float *h,  // (time_steps + 1) * batch_size * hidden_size，包含初始状态
                     float *v   // (time_steps * batch_size * hidden_size * 4)，反量化后的v，可以为 nullptr
);

void hasteGRUForward(bool is_training,  // 是否开启训练模式，true为训练，false为推理
                     const int time_steps,
                     const int batch_size,
                     const int input_size,
                     const int hidden_size,
                     const float *W, const float *R, const float *bx,
                     const float *br, const float *x,
                     const float *h0,  // 初始隐藏状态，可以为 nullptr
                     const cublasHandle_t &g_blas_handle,
                     float *h,  // (time_steps + 1) * batch_size * hidden_size，包含初始状态
                     float *v   // (time_steps * batch_size * hidden_size * 4)，中间值v，可以为 nullptr
);

void forwardInterface(bool is_training,  // 是否开启训练模式，true为训练，false为推理
                      bool is_quant,
                      bool use_int16,
                      int time_steps, int batch_size, int input_size, int hidden_size,
                      const float *W,
                      const float *R,
                      const float *bx,
                      const float *br,
                      const float *x,
                      const GRUQuantitativeParameters &quant_gru_scales,
                      const cublasHandle_t &g_blas_handle,
                      float *h,  // (time_steps + 1) * batch_size * hidden_size，包含初始状态
                      float *v);  // (time_steps * batch_size * hidden_size * 4)，中间值v，可以为 nullptr

void hasteGRUBackward(const int time_steps,
                      const int batch_size,
                      const int input_size,
                      const int hidden_size,
                      const float *W, const float *R, const float *bx,
                      const float *br, const float *x,
                      const float *dh_new,
                      const float *h,// (time_steps + 1) * batch_size * hidden_size
                      const float *v,// (time_steps * batch_size * hidden_size * 4)，中间值v，可以为 nullptr
                      const cublasHandle_t &g_blas_handle,
                      float *dx, // (time_steps *batch_size * input_size) 输入序列梯度
                      float *dW, // (input_size * hidden_size * 3)// 对输入权重的梯度
                      float *dR, // (hidden_size * hidden_size * 3) // 对循环权重的梯度
                      float *dbx,// (hidden_size * 3)// 对输入偏置的梯度
                      float *dbr,// (hidden_size * 3)// 对循环偏置的梯度
                      float *dh  // (batch_size * hidden_size)// 对最后隐藏状态的梯度
);