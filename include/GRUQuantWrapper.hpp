#pragma once

#include <cublas_v2.h>

#include "quantize_ops_helper.hpp"

cublasHandle_t g_blas_handle = nullptr;

void initCublasHandle() {
    if (!g_blas_handle) {
        cublasCreate(&g_blas_handle);
    }
}

void destroyCublasHandle() {
    if (g_blas_handle) {
        cublasDestroy(g_blas_handle);
        g_blas_handle = nullptr;
    }
}

class GRUQuantWrapper {
 public:
  GRUQuantWrapper(bool use_int16,
                  int time_steps, int batch_size,
                  int input_size, int hidden_size)
      : use_int16_(use_int16),
        time_steps_(time_steps),
        batch_size_(batch_size),
        input_size_(input_size),
        hidden_size_(hidden_size) {
      initCublasHandle();
  }

  // Step 1 + 2：初始化量化权重
  void initWeights(const Tensor2f &W, const Tensor2f &R,
                   const Tensor1f &bx, const Tensor1f &br,
                   const Tensor3f &x) {
      calibrateGruScales(
          use_int16_,
          time_steps_, batch_size_, input_size_, hidden_size_,
          W.data(), R.data(), bx.data(), br.data(), x.data(),
          quant_gru_scales_  // 内部保存
      );

      GruQuantInit(
          time_steps_, batch_size_, input_size_, hidden_size_,
          W, R, bx, br, x,
          dh_dummy_, // 如果需要，可以传空或全零
          W_quant_, R_quant_, bx_quant_, br_quant_, x_quant_, dh_new_quant_,
          quant_gru_scales_ // 内部使用
      );
  }

  // Step 3：量化推理
  Tensor3i8 forward(const Tensor3f &x) {
      Tensor3i8 h_quant(time_steps_ + 1, batch_size_, hidden_size_);
      GruInferenceQuant(
          W_quant_, R_quant_, bx_quant_, br_quant_, x_quant_,
          quant_gru_scales_, // 内部使用
          h_quant
      );
      return h_quant;
  }

 private:
  bool use_int16_;
  int time_steps_, batch_size_, input_size_, hidden_size_;

  GRUQuantitativeParameters quant_gru_scales_; // 内部管理
  Tensor2i8 W_quant_, R_quant_;
  Tensor1i32 bx_quant_, br_quant_;
  Tensor3i8 x_quant_, dh_new_quant_;
  Tensor3f dh_dummy_; // 如果需要，可以传空张量
};

