// #pragma once
//
// #include <torch/extension.h>
// #include <cublas_v2.h>
// #include "quantize_ops_helper.hpp"
//
// // 全局 cublas handle
// static cublasHandle_t g_blas_handle = nullptr;
//
// inline void initCublasHandle()
// {
//     if (!g_blas_handle)
//     {
//         cublasCreate(&g_blas_handle);
//     }
// }
//
// inline void destroyCublasHandle()
// {
//     if (g_blas_handle)
//     {
//         cublasDestroy(g_blas_handle);
//         g_blas_handle = nullptr;
//     }
// }
//
// // ======================================================
// //   GRUQuantWrapper（新版）
// // ======================================================
// class GRUQuantWrapper
// {
// public:
//     GRUQuantWrapper(bool use_int16,
//                     int time_steps, int batch_size,
//                     int input_size, int hidden_size)
//         : use_int16_(use_int16),
//           time_steps_(time_steps),
//           batch_size_(batch_size),
//           input_size_(input_size),
//           hidden_size_(hidden_size)
//     {
//         initCublasHandle();
//     }
//
//     // --------------------------------------------------
//     // Step 1 + 2: 初始化量化权重 (输入为 PyTorch Tensor)
//     // --------------------------------------------------
//     void initWeights(const at::Tensor& W,
//                      const at::Tensor& R,
//                      const at::Tensor& bx,
//                      const at::Tensor& br,
//                      const at::Tensor& x_for_calib,
//                      const at::Tensor& dh_for_calib)
//     {
//         TORCH_CHECK(W.is_cuda(), "W must be CUDA tensor");
//         TORCH_CHECK(R.is_cuda(), "R must be CUDA tensor");
//         TORCH_CHECK(x_for_calib.is_cuda(), "x_for_calib must be CUDA tensor");
//
//         // 校准量化参数
//         calibrateGruScales(
//             use_int16_,
//             time_steps_, batch_size_, input_size_, hidden_size_,
//             W.data_ptr<float>(),
//             R.data_ptr<float>(),
//             bx.data_ptr<float>(),
//             br.data_ptr<float>(),
//             x_for_calib.data_ptr<float>(),
//             g_blas_handle,
//             quant_gru_scales_
//         );
//
//         // 分配 GPU 上的量化权重
//         int dtype_bits = use_int16_ ? 16 : 8;
//         auto options_int =
//             torch::TensorOptions().dtype(dtype_bits == 16 ? torch::kInt16 : torch::kInt8)
//                                   .device(torch::kCUDA);
//
//         W_quant_ = torch::empty({3 * hidden_size_, input_size_}, options_int);
//         R_quant_ = torch::empty({3 * hidden_size_, hidden_size_}, options_int);
//         bx_quant_ = torch::empty({3 * hidden_size_}, torch::dtype(torch::kInt32).device(torch::kCUDA));
//         br_quant_ = torch::empty({3 * hidden_size_}, torch::dtype(torch::kInt32).device(torch::kCUDA));
//
//         quantification(x, x_quant, time_steps * batch_size * input_size, gruRescaleParams.exp2_inv_x_,
//                        gruRescaleParams.zp_x_);
//
//         // 量化权重
//         GruQuantInit(
//             time_steps_, batch_size_, input_size_, hidden_size_,
//             W.data_ptr<float>(), R.data_ptr<float>(), bx.data_ptr<float>(), br.data_ptr<float>(),
//             x_for_calib.data_ptr<float>(),
//             /*dh_dummy=*/dh_for_calib.data_ptr<float>(),
//             W_quant_.data_ptr<options_int>(), R_quant_.data_ptr<options_int>(),
//             bx_quant_.data_ptr<options_int>(), br_quant_.data_ptr<options_int>(),
//             /*x_quant=*/torch::Tensor(), // 这里只量化权重，不量化输入
//             /*dh_new_quant=*/torch::Tensor(),
//             quant_gru_scales_
//         );
//     }
//
//     // --------------------------------------------------
//     // Step 3：量化前向推理
//     // x: float32, shape = [T, B, input_size]
//     // 返回 h_quant: int8/int16, shape = [T+1, B, hidden_size]
//     // --------------------------------------------------
//     at::Tensor forward(const at::Tensor& x)
//     {
//         TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
//         TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
//
//         int dtype_bits = use_int16_ ? 16 : 8;
//         auto opt_q =
//             torch::TensorOptions().dtype(dtype_bits == 16 ? torch::kInt16 : torch::kInt8)
//                                   .device(torch::kCUDA);
//
//         auto h_quant = torch::empty({time_steps_ + 1, batch_size_, hidden_size_}, opt_q);
//
//         GruInferenceQuant(
//             W_quant_, R_quant_,
//             bx_quant_, br_quant_,
//             x, // float32 输入，内部会量化
//             quant_gru_scales_,
//             h_quant
//         );
//
//         // 直接返回 PyTorch Tensor (GPU)
//         return h_quant;
//     }
//
// private:
//     bool use_int16_;
//     int time_steps_, batch_size_, input_size_, hidden_size_;
//
//     // 量化参数
//     GRUQuantitativeParameters quant_gru_scales_;
//
//     // 量化后的权重（GPU int8/int16）
//     at::Tensor W_quant_, R_quant_;
//     at::Tensor bx_quant_, br_quant_;
// };
