#pragma once

#include <cublas_v2.h>
#include <torch/extension.h>

#include "quantize_ops_helper.hpp"
#include "gru.h"
#include "gru_quant.h"

// 全局 cublas handle
static cublasHandle_t g_blas_handle = nullptr;

inline void initCublasHandle() {
    if (!g_blas_handle) {
        cublasCreate(&g_blas_handle);
    }
}

inline void destroyCublasHandle() {
    if (g_blas_handle) {
        cublasDestroy(g_blas_handle);
        g_blas_handle = nullptr;
    }
}

// ======================================================
//   GRUQuantWrapper（新版）
// ======================================================
template<typename QuantT>
class GRUQuantWrapper {
 public:
  GRUQuantWrapper(bool use_int16, int time_steps, int batch_size,
                  int input_size, int hidden_size)
      : time_steps_(time_steps),
        batch_size_(batch_size),
        input_size_(input_size),
        hidden_size_(hidden_size) {
      initCublasHandle();
  }

  // --------------------------------------------------
  // Step 1 + 2: 初始化量化权重 (输入为 PyTorch Tensor)
  // --------------------------------------------------
  void initWeights(const at::Tensor &W, const at::Tensor &R,
                   const at::Tensor &bx, const at::Tensor &br,
                   const at::Tensor &x_for_calib) {
      TORCH_CHECK(W.is_cuda(), "W must be CUDA tensor");
      TORCH_CHECK(R.is_cuda(), "R must be CUDA tensor");
      TORCH_CHECK(x_for_calib.is_cuda(), "x_for_calib must be CUDA tensor");

      // 校准量化参数
      bool use_int16_ = std::is_same_v<QuantT, int16_t> ? true : false;
      calibrateGruScales(use_int16_, time_steps_, batch_size_, input_size_,
                         hidden_size_, W.data_ptr<float>(),
                         R.data_ptr<float>(), bx.data_ptr<float>(),
                         br.data_ptr<float>(), x_for_calib.data_ptr<float>(),
                         g_blas_handle, quant_parms_);

      torch::TensorOptions options_int;
      if constexpr (std::is_same_v<QuantT, int8_t>) {
          options_int = options_int.dtype(torch::kInt8).device(torch::kCUDA);
      } else if constexpr (std::is_same_v<QuantT, int16_t>) {
          options_int = options_int.dtype(torch::kInt16).device(torch::kCUDA);
      } else {
          fprintf(stderr, "Unsupported QuantT type!");
      }

      W_quant_ = torch::empty({3 * hidden_size_, input_size_}, options_int);
      R_quant_ = torch::empty({3 * hidden_size_, hidden_size_}, options_int);
      bx_quant_ =
          torch::empty({3 * hidden_size_},
                       torch::dtype(torch::kInt32).device(torch::kCUDA));
      br_quant_ =
          torch::empty({3 * hidden_size_},
                       torch::dtype(torch::kInt32).device(torch::kCUDA));

      dev::quantificationPerChannel(
          W.data_ptr<float>(), W_quant_.data_ptr<QuantT>(), input_size_,
          3 * hidden_size_, quant_parms_.exp2_inv_W_);
      dev::quantificationPerChannel(
          R.data_ptr<float>(), R_quant_.data_ptr<QuantT>(), hidden_size_,
          3 * hidden_size_, quant_parms_.exp2_inv_R_);

      dev::vector<int32_t> exp2_inv_bx(quant_parms_.exp2_inv_bx_);
      dev::quantificationPerChannel(bx.data_ptr<float>(),
                                    bx_quant_.data_ptr<int32_t>(), 1,
                                    3 * hidden_size_, exp2_inv_bx);
      dev::vector<int32_t> exp2_inv_br(quant_parms_.exp2_inv_br_);
      dev::quantificationPerChannel(br.data_ptr<float>(),
                                    br_quant_.data_ptr<int32_t>(), 1,
                                    3 * hidden_size_, exp2_inv_br);
  }

  // --------------------------------------------------
  // Step 3：量化前向推理
  // x: float32, shape = [T, B, input_size]
  // 返回 h_quant: int8/int16, shape = [T+1, B, hidden_size]
  // --------------------------------------------------
  at::Tensor forward(const at::Tensor &x) {
      TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
      TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

      torch::TensorOptions options_int;
      if constexpr (std::is_same_v<QuantT, int8_t>) {
          options_int = options_int.dtype(torch::kInt8).device(torch::kCUDA);
      } else if constexpr (std::is_same_v<QuantT, int16_t>) {
          options_int = options_int.dtype(torch::kInt16).device(torch::kCUDA);
      } else {
          fprintf(stderr, "Unsupported QuantT type!");
      }

      dev::vector<int8_t> x_quant(input_size_ *batch_size_
      *time_steps_);

      dev::quantification(x.data_ptr<float>(), x_quant.data(),
                          time_steps_ * batch_size_ * input_size_,
                          quant_parms_.exp2_inv_x_, quant_parms_.zp_x_);

      generate_int8_lut_from_exp2_inv(
          quant_parms_.exp2_inv_z_pre_, quant_parms_.zp_z_pre_,
          quant_parms_.exp2_inv_z_out_, quant_parms_.zp_z_out_,
          quant_parms_.exp2_inv_r_pre_, quant_parms_.zp_r_pre_,
          quant_parms_.exp2_inv_r_out_, quant_parms_.zp_r_out_,
          quant_parms_.exp2_inv_g_pre_, quant_parms_.zp_g_pre_,
          quant_parms_.exp2_inv_g_out_, quant_parms_.zp_g_out_);

      dev::vector<QuantT> h_quant((time_steps_ + 1) * batch_size_ * hidden_size_, quant_parms_.zp_h_);
      gru::ForwardPassQuant<int16_t> forward =
          gru::ForwardPassQuant<int16_t>(
              false,  // training
              batch_size_, input_size_, hidden_size_, g_blas_handle);

      forward.setRescaleParam(quant_parms_);

      forward.Run(time_steps_,
                  W_quant_.data_ptr<QuantT>(),
                  R_quant_.data_ptr<QuantT>(),
                  bx_quant_.data_ptr<int32_t>(),
                  br_quant_.data_ptr<int32_t>(),
                  x_quant.data(),
                  h_quant.data(),
                  nullptr,
                  tmp_Wx_quant_dev_.data_ptr<int32_t>(),
                  tmp_Rh_quant_dev_.data_ptr<int32_t>(),
                  0.0f,
                  nullptr);


      // TODO: 反量化输出
      return h_;
  }

 private:
  int time_steps_, batch_size_, input_size_, hidden_size_;

  // 量化参数
  GRUQuantitativeParameters quant_parms_;

  // 量化后的权重（GPU int8/int16）
  at::Tensor W_quant_, R_quant_;
  at::Tensor bx_quant_, br_quant_;

  at::Tensor tmp_Wx_quant_dev_, tmp_Rh_quant_dev_;

  at::Tensor h_;
};
