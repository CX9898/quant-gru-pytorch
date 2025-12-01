#pragma once

#include <vector>
#include <cublas_v2.h>

#include "gru.h"
#include "gru_quant.h"

GRUQuantitativeParameters calibrateGruScales(
    bool use_int16,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W,
    const float *R,
    const float *bx,
    const float *br,
    const float *x,
    const cublasHandle_t &g_blas_handle);

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
