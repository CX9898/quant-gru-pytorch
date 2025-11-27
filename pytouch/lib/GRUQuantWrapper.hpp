#pragma once

#include <cublas_v2.h>
#include <torch/extension.h>

#include "gru.h"
#include "gru_quant.h"
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
template<typename QuantT>
class GRUQuantWrapper {
 public:
    GRUQuantWrapper(int time_steps, int batch_size,
                    int input_size, int hidden_size)
        : time_steps_(time_steps),
          batch_size_(batch_size),
          input_size_(input_size),
          hidden_size_(hidden_size) {
        initCublasHandle();

        torch::TensorOptions options_int;
        if constexpr (std::is_same_v<QuantT, int8_t>) {
            options_int = options_int.dtype(torch::kInt8).device(torch::kCUDA);
        } else if constexpr (std::is_same_v<QuantT, int16_t>) {
            options_int = options_int.dtype(torch::kInt16).device(torch::kCUDA);
        } else {
            fprintf(stderr, "Unsupported QuantT type!");
        }

        W_quant_.resize(3 * hidden_size * input_size);
        R_quant_.resize(3 * hidden_size * hidden_size);
        bx_quant_.resize(3 * hidden_size);
        br_quant_.resize(3 * hidden_size);

        tmp_Wx_quant_.resize(time_steps * batch_size * hidden_size *
                             3);
        tmp_Rh_quant_.resize(batch_size * hidden_size *
                             3);

        x_quant_.resize(time_steps * batch_size * input_size);
        h_quant_.resize((time_steps + 1) * batch_size * hidden_size);

        h_ = torch::empty({time_steps_ + 1, batch_size_, hidden_size_},  // 形状：[T+1, B, H]
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
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

        dev::quantificationPerChannel(
            W.data_ptr<float>(), W_quant_.data(), input_size_,
            3 * hidden_size_, quant_parms_.exp2_inv_W_);
        dev::quantificationPerChannel(
            R.data_ptr<float>(), R_quant_.data(), hidden_size_,
            3 * hidden_size_, quant_parms_.exp2_inv_R_);

        dev::vector<int32_t> exp2_inv_bx(quant_parms_.exp2_inv_bx_);
        dev::quantificationPerChannel(bx.data_ptr<float>(),
                                      bx_quant_.data(), 1,
                                      3 * hidden_size_, exp2_inv_bx);
        dev::vector<int32_t> exp2_inv_br(quant_parms_.exp2_inv_br_);
        dev::quantificationPerChannel(br.data_ptr<float>(),
                                      br_quant_.data(), 1,
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

        dev::quantification(x.data_ptr<float>(), x_quant_.data(),
                            time_steps_ * batch_size_ * input_size_,
                            quant_parms_.exp2_inv_x_, quant_parms_.zp_x_);

        generate_int8_lut_from_exp2_inv(
            quant_parms_.exp2_inv_z_pre_, quant_parms_.zp_z_pre_,
            quant_parms_.exp2_inv_z_out_, quant_parms_.zp_z_out_,
            quant_parms_.exp2_inv_r_pre_, quant_parms_.zp_r_pre_,
            quant_parms_.exp2_inv_r_out_, quant_parms_.zp_r_out_,
            quant_parms_.exp2_inv_g_pre_, quant_parms_.zp_g_pre_,
            quant_parms_.exp2_inv_g_out_, quant_parms_.zp_g_out_);

        h_quant_.setVal(quant_parms_.zp_h_);

        gru::ForwardPassQuant<QuantT> forward =
            gru::ForwardPassQuant<QuantT>(
                false, // training
                batch_size_, input_size_, hidden_size_, g_blas_handle);

        forward.setRescaleParam(quant_parms_);

        forward.Run(time_steps_,
                    W_quant_.data(),
                    R_quant_.data(),
                    bx_quant_.data(),
                    br_quant_.data(),
                    x_quant_.data(),
                    h_quant_.data(),
                    nullptr,
                    tmp_Wx_quant_.data(),
                    tmp_Rh_quant_.data(),
                    0.0f,
                    nullptr);

        dev::dequantification(h_quant_.data(),
                              h_.data_ptr<float>(),
                              h_quant_.size(),
                              quant_parms_.exp2_inv_h_,
                              quant_parms_.zp_h_);

        return h_;
    }

    at::Tensor backward(const at::Tensor &dh_new) {
        TORCH_CHECK(dh_new.is_cuda(), "dh_new must be CUDA tensor");
        TORCH_CHECK(dh_new.dtype() == torch::kFloat32, "dh_new must be float32");

        dev::dequantificationPerChannel(W_quant_.data(),
                                        W_dequant_.data(),
                                        W_dequant_.size(),
                                        quant_parms_.exp2_inv_h_,
                                        quant_parms_.zp_h_);

        dev::vector<float> dx_dev(time_steps_ * batch_size_ * input_size_);     // 输入序列梯度
        dev::vector<float> dW_dev(input_size_ * hidden_size_ * 3);              // 对输入权重的梯度
        dev::vector<float> dR_dev(hidden_size_ * hidden_size_ * 3);             // 对循环权重的梯度
        dev::vector<float> dbx_dev(hidden_size_ * 3);                           // 对输入偏置的梯度
        dev::vector<float> dbr_dev(hidden_size_ * 3);                           // 对循环偏置的梯度
        dev::vector<float> dh_dev(batch_size_ * hidden_size_);                  // 对最后隐藏状态的梯度
        dev::vector<float> dp_dev(time_steps_ * batch_size_ * hidden_size_ * 3);// 临时缓存梯度（内部结构用）
        dev::vector<float> dq_dev(time_steps_ * batch_size_ * hidden_size_ * 3);// 临时缓存梯度（内部结构用）

        //        gru::BackwardPass<float> backward(batch_size, input_size, hidden_size,
        //                                          g_blas_handle);
        //
        //        backward.Run(time_steps, W_dev.data(), R_dev.data(), bx_dev.data(),
        //                     br_dev.data(), x_dev.data(), h_dev.data(), v_dev.data(),
        //                     dh_new_dev.data(), dx_dev.data(), dW_dev.data(), dR_dev.data(),
        //                     dbx_dev.data(), dbr_dev.data(), dh_dev.data(), dp_dev.data(),
        //                     dq_dev.data(), nullptr);

        return h_;
    }

 private:
    int time_steps_, batch_size_, input_size_, hidden_size_;

    // 量化参数
    GRUQuantitativeParameters quant_parms_;

    // 量化后的权重（GPU int8/int16）
    dev::vector<QuantT> W_quant_, R_quant_;
    dev::vector<int32_t> bx_quant_, br_quant_;

    dev::vector<float> W_dequant_, R_dequant_;
    dev::vector<float> bx_dequant_, br_dequant_;

    // tmp
    dev::vector<QuantT> x_quant_, h_quant_;
    dev::vector<int32_t> tmp_Wx_quant_, tmp_Rh_quant_;

    // output
    at::Tensor h_;
    at::Tensor dx_;
    at::Tensor dW_;
    at::Tensor dR_;
    at::Tensor dbx_;
    at::Tensor dbr_;
    at::Tensor dh_;
};
