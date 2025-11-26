#pragma once

#include <cublas_v2.h>
#include <torch/extension.h>

#include "quantize_ops_helper.hpp"

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
class GRUQuantWrapper {
   public:
    GRUQuantWrapper(bool use_int16, int time_steps, int batch_size,
                    int input_size, int hidden_size)
        : use_int16_(use_int16),
          time_steps_(time_steps),
          batch_size_(batch_size),
          input_size_(input_size),
          hidden_size_(hidden_size) {
        initCublasHandle();
    }

    // --------------------------------------------------
    // Step 1 + 2: 初始化量化权重 (输入为 PyTorch Tensor)
    // --------------------------------------------------
    void initWeights(const at::Tensor& W, const at::Tensor& R,
                     const at::Tensor& bx, const at::Tensor& br,
                     const at::Tensor& x_for_calib,
                     const at::Tensor& dh_for_calib) {
        TORCH_CHECK(W.is_cuda(), "W must be CUDA tensor");
        TORCH_CHECK(R.is_cuda(), "R must be CUDA tensor");
        TORCH_CHECK(x_for_calib.is_cuda(), "x_for_calib must be CUDA tensor");

        // 校准量化参数
        calibrateGruScales(use_int16_, time_steps_, batch_size_, input_size_,
                           hidden_size_, W.data_ptr<float>(),
                           R.data_ptr<float>(), bx.data_ptr<float>(),
                           br.data_ptr<float>(), x_for_calib.data_ptr<float>(),
                           g_blas_handle, quant_parms_);

        // 分配 GPU 上的量化权重
        auto options_int = torch::TensorOptions()
                               .dtype(use_int16_ ? torch::kInt16 : torch::kInt8)
                               .device(torch::kCUDA);

        W_quant_ = torch::empty({3 * hidden_size_, input_size_}, options_int);
        R_quant_ = torch::empty({3 * hidden_size_, hidden_size_}, options_int);
        bx_quant_ =
            torch::empty({3 * hidden_size_},
                         torch::dtype(torch::kInt32).device(torch::kCUDA));
        br_quant_ =
            torch::empty({3 * hidden_size_},
                         torch::dtype(torch::kInt32).device(torch::kCUDA));

        if (use_int16_) {
            dev::quantificationPerChannel(
                W.data_ptr<float>(), W_quant_.data_ptr<int16_t>(), input_size_,
                3 * hidden_size_, quant_parms_.exp2_inv_W_);
            dev::quantificationPerChannel(
                R.data_ptr<float>(), R_quant_.data_ptr<int16_t>(), hidden_size_,
                3 * hidden_size_, quant_parms_.exp2_inv_R_);

        } else {
            dev::quantificationPerChannel(
                W.data_ptr<float>(), W_quant_.data_ptr<int8_t>(), input_size_,
                3 * hidden_size_, quant_parms_.exp2_inv_W_);
            dev::quantificationPerChannel(
                R.data_ptr<float>(), R_quant_.data_ptr<int8_t>(), hidden_size_,
                3 * hidden_size_, quant_parms_.exp2_inv_R_);
        }
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
    at::Tensor forward(const at::Tensor& x) {
        TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
        TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

        auto options_int = torch::TensorOptions()
                               .dtype(use_int16_ ? torch::kInt16 : torch::kInt8)
                               .device(torch::kCUDA);

        auto h_quant = torch::empty(
            {time_steps_ + 1, batch_size_, hidden_size_}, options_int);
        h_quant.fill_(quant_parms_.zp_h_);

        at::Tensor x_quant =
            torch::empty({input_size_, batch_size_, time_steps_}, options_int);

        if (use_int16_) {
            dev::quantification(x.data_ptr<float>(),
                                x_quant.data_ptr<int16_t>(),
                                time_steps_ * batch_size_ * input_size_,
                                quant_parms_.exp2_inv_x_, quant_parms_.zp_x_);
        } else {
            dev::quantification(x.data_ptr<float>(), x_quant.data_ptr<int8_t>(),
                                time_steps_ * batch_size_ * input_size_,
                                quant_parms_.exp2_inv_x_, quant_parms_.zp_x_);
        }

        generate_int8_lut_from_exp2_inv(
            quant_parms_.exp2_inv_z_pre_, quant_parms_.zp_z_pre_,
            quant_parms_.exp2_inv_z_out_, quant_parms_.zp_z_out_,
            quant_parms_.exp2_inv_r_pre_, quant_parms_.zp_r_pre_,
            quant_parms_.exp2_inv_r_out_, quant_parms_.zp_r_out_,
            quant_parms_.exp2_inv_g_pre_, quant_parms_.zp_g_pre_,
            quant_parms_.exp2_inv_g_out_, quant_parms_.zp_g_out_);

        // gru::ForwardPassQuant<int8_t> forward =
        // gru::ForwardPassQuant<int8_t>(
        //     false,  // training
        //     batch_size, input_size, hidden_size, g_blas_handle);
        //
        // forward.setRescaleParam(quant_parms_);
        //
        // forward.Run(time_steps, W_quant_.data(), R_dev.data(), bx_dev.data(),
        //     br_dev.data(), x_quant_dev.data(), h_quant_dev.data(),
        //     nullptr, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f,
        //     nullptr);

        // 直接返回 PyTorch Tensor (GPU)
        return h_quant;
    }

   private:
    bool use_int16_;
    int time_steps_, batch_size_, input_size_, hidden_size_;

    // 量化参数
    GRUQuantitativeParameters quant_parms_;

    // 量化后的权重（GPU int8/int16）
    at::Tensor W_quant_, R_quant_;
    at::Tensor bx_quant_, br_quant_;
};
