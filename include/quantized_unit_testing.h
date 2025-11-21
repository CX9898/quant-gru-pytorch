#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <vector>

#include "gru_quant.h"
#include "blas.h"
#include "checkData.hpp"
#include "quantize_ops_helper.hpp"

inline bool checkScale(const std::vector<float> &src,
                       const std::vector<int8_t> &quant,
                       float scale,
                       int32_t zero_point,
                       const std::string &name = "") {
    bool is_pass = true;
    if (scale <= 1e-6f) {
        printf("Warning, %s: scale = %.15f <= 1e-6f\n",
               name.c_str(),
               scale);
    }

    std::vector<float> requant(src.size());
#pragma omp parallel for
    for (int i = 0; i < src.size(); ++i) {
        const float req_val = (quant[i] - zero_point) * scale;
        requant[i] = req_val;
    }
    is_pass &= checkCosineSimilarity(src, requant, name);
    is_pass &= checkMSE(src, requant);

    return is_pass;
}


template<typename T, typename QuantT>
inline bool checkScale(const std::vector<T> &src,
                       const std::vector<QuantT> &quant,
                       int32_t exp2_inv,
                       int32_t zero_point,
                       const std::string &name = "") {
    bool is_pass = true;
    std::vector<T> requant(src.size());
#pragma omp parallel for
    for (int i = 0; i < src.size(); ++i) {
        const float req_val = dequant_from_exp2(quant[i], exp2_inv, zero_point);
        requant[i] = req_val;
    }
    is_pass &= checkCosineSimilarity(src, requant, name);
    is_pass &= checkMSE(src, requant, name);
    return is_pass;
}

template<typename T, typename QuantT>
inline bool checkScale(const std::vector<T> &src,
                       int32_t exp2_inv,
                       int32_t zero_point,
                       const std::string &name = "") {

    std::vector<QuantT> quant(src.size());
#pragma omp parallel for
    for (int i = 0; i < src.size(); ++i) {
        const QuantT req_val = quant_from_exp2<T, QuantT>(src[i], exp2_inv, zero_point);
        quant[i] = req_val;
    }
    return checkScale<T, QuantT>(src, quant, exp2_inv, zero_point, name);
}

template<typename QuantT>
inline bool checkScalePerChannel(const std::vector<float> &src,
                                 size_t channel_size,
                                 size_t in_dim,
                                 const std::vector<QuantT> &quant,
                                 const std::vector<float> &scale,
                                 const std::string &name = "") {
    bool is_pass = true;
    std::vector<float> requant(src.size());
#pragma omp parallel for
    for (int i = 0; i < in_dim; ++i) {
        for (int j = 0; j < channel_size; ++j) {
            const int idx = i * channel_size + j;
            const float scale_val = scale[j];
            if (scale_val <= 1e-6f) {
                printf("Warning, %s: scale[%d] = %f\n",
                       name.c_str(),
                       j,
                       scale_val);
            }
            const float req_val = (quant[i * channel_size + j]) * scale_val;
            requant[idx] = req_val;
        }
    }

    is_pass &= checkCosineSimilarity(src, requant, name);
    is_pass &= checkMSE(src, requant, name);

    return is_pass;
}

template<typename T, typename QuantT>
inline bool checkScalePerChannel(const std::vector<T> &src,
                                 size_t channel_size,
                                 size_t in_dim,
                                 const std::vector<QuantT> &quant,
                                 const std::vector<int32_t> &exp2_inv,
                                 const std::string &name = "") {

    bool is_pass = true;
    std::vector<T> requant(src.size());
#pragma omp parallel for
    for (int i = 0; i < in_dim; ++i) {
        for (int j = 0; j < channel_size; ++j) {
            const int idx = i * channel_size + j;
            const int exp2_inv_val = exp2_inv[j];
            const int zp_val = 0;
            const T req_val = dequant_from_exp2(quant[idx], exp2_inv_val, zp_val);
            requant[idx] = req_val;
        }
    }
    is_pass &= checkCosineSimilarity(src, requant, name);
    is_pass &= checkMSE(src, requant, name);
    return is_pass;
}

template<typename T, typename QuantT>
inline bool checkScalePerChannel(const std::vector<T> &src,
                                 size_t channel_size,
                                 size_t in_dim,
                                 const std::vector<int32_t> &exp2_inv,
                                 const std::string &name = "") {

    std::vector<QuantT> quant(src.size());
#pragma omp parallel for
    for (int i = 0; i < in_dim; ++i) {
        for (int j = 0; j < channel_size; ++j) {
            const int idx = i * channel_size + j;
            const int exp2_inv_val = exp2_inv[j];
            const int zp_val = 0;
            const QuantT quant_val = quant_from_exp2<T, QuantT>(src[idx], exp2_inv_val, zp_val);
            quant[idx] = quant_val;
        }
    }
    return checkScalePerChannel<T, QuantT>(src, channel_size, in_dim, quant, exp2_inv, name);
}


template<typename QuantT>
struct Quantized_unit_testing {

  std::vector<float> W_;
  std::vector<float> R_;
  std::vector<float> bx_;
  std::vector<float> br_;
  std::vector<float> x_;
  std::vector<float> dh_;
  std::vector<QuantT> W_quant_;
  std::vector<QuantT> R_quant_;
  std::vector<int32_t> bx_quant_;
  std::vector<int32_t> br_quant_;
  std::vector<QuantT> x_quant_;
  std::vector<QuantT> dh_new_quant_;
  size_t hidden_size_;
  size_t input_size_;
  size_t batch_size_;
  size_t time_steps_;
  cublasHandle_t handle_;
  size_t channels_;
  GRUQuantitativeParameters quant_parms_;

  Quantized_unit_testing(float *W,
                         float *R,
                         float *bx,
                         float *br,
                         float *x,
                         float *dh,
                         QuantT *W_quant,
                         QuantT *R_quant,
                         int32_t *bx_quant,
                         int32_t *br_quant,
                         QuantT *x_quant,
                         QuantT *dh_new_quant,
                         size_t hidden_size,
                         size_t input_size,
                         size_t batch_size,
                         size_t time_steps,
                         cublasHandle_t handle,
                         const GRUQuantitativeParameters &quant_parms) {
      hidden_size_ = hidden_size;
      input_size_ = input_size;
      batch_size_ = batch_size;
      time_steps_ = time_steps;
      handle_ = handle;
      quant_parms_ = quant_parms;
      W_ = std::vector<float>(W, W + hidden_size * 3 * input_size);
      R_ = std::vector<float>(R, R + hidden_size * 3 * hidden_size);
      bx_ = std::vector<float>(bx, bx + hidden_size * 3);
      br_ = std::vector<float>(br, br + hidden_size * 3);
      x_ = std::vector<float>(x, x + input_size * batch_size * time_steps);
      dh_ = std::vector<float>(dh, dh + hidden_size * batch_size * time_steps);
      W_quant_ = std::vector<QuantT>(W_quant, W_quant + hidden_size * 3 * input_size);
      R_quant_ = std::vector<QuantT>(R_quant, R_quant + hidden_size * 3 * hidden_size);
      bx_quant_ = std::vector<int32_t>(bx_quant, bx_quant + hidden_size * 3);
      br_quant_ = std::vector<int32_t>(br_quant, br_quant + hidden_size * 3);
      x_quant_ = std::vector<QuantT>(x_quant, x_quant + input_size * batch_size * time_steps);
      dh_new_quant_ = std::vector<QuantT>(dh_new_quant, dh_new_quant + hidden_size * batch_size * time_steps);

      channels_ = hidden_size_ * 3;
  }

  bool checkWxGemm();

  bool checkQuantParameters();

  void printGRUQuantitativeParameters();
};

template<typename QuantT>
inline void Quantized_unit_testing<QuantT>::printGRUQuantitativeParameters() {
    printf("GRUQuantitativeParameters (量化参数):\n");
    printf("  hidden_ = %d\n", quant_parms_.hidden_);
    printf("  exp2_inv_x_ = %d, zp_x_ = %d\n",
           quant_parms_.exp2_inv_x_, quant_parms_.zp_x_);
    printf("  exp2_inv_h_ = %d, zp_h_ = %d\n",
           quant_parms_.exp2_inv_h_, quant_parms_.zp_h_);

    printf("  exp2_inv_W_ (size %zu): ", quant_parms_.exp2_inv_W_.size());
    for (size_t i = 0; i < quant_parms_.exp2_inv_W_.size() && i < 5; ++i) {
        printf("%d ", quant_parms_.exp2_inv_W_[i]);
    }
    if (quant_parms_.exp2_inv_W_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_R_ (size %zu): ", quant_parms_.exp2_inv_R_.size());
    for (size_t i = 0; i < quant_parms_.exp2_inv_R_.size() && i < 5; ++i) {
        printf("%d ", quant_parms_.exp2_inv_R_[i]);
    }
    if (quant_parms_.exp2_inv_R_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_bx_ (size %zu): ", quant_parms_.exp2_inv_bx_.size());
    for (size_t i = 0; i < quant_parms_.exp2_inv_bx_.size() && i < 5; ++i) {
        printf("%d ", quant_parms_.exp2_inv_bx_[i]);
    }
    if (quant_parms_.exp2_inv_bx_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_br_ (size %zu): ", quant_parms_.exp2_inv_br_.size());
    for (size_t i = 0; i < quant_parms_.exp2_inv_br_.size() && i < 5; ++i) {
        printf("%d ", quant_parms_.exp2_inv_br_[i]);
    }
    if (quant_parms_.exp2_inv_br_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_Wx_ = %d, zp_Wx_ = %d \n",
           quant_parms_.exp2_inv_Wx_, quant_parms_.zp_Wx_);
    printf("  exp2_inv_Rh_ = %d, zp_Rh_ = %d \n",
           quant_parms_.exp2_inv_Rh_, quant_parms_.zp_Rh_);
    printf("  exp2_inv_z_pre_ = %d, zp_z_pre_ = %d \n",
           quant_parms_.exp2_inv_z_pre_, quant_parms_.zp_z_pre_);
    printf("  exp2_inv_r_pre_ = %d, zp_r_pre_ = %d\n",
           quant_parms_.exp2_inv_r_pre_, quant_parms_.zp_r_pre_);
    printf("  exp2_inv_g_pre_ = %d, zp_g_pre_ = %d\n",
           quant_parms_.exp2_inv_g_pre_, quant_parms_.zp_g_pre_);
    printf("  exp2_inv_z_out_ = %d, zp_z_out_ = %d\n",
           quant_parms_.exp2_inv_z_out_, quant_parms_.zp_z_out_);
    printf("  exp2_inv_r_out_ = %d, zp_r_out_ = %d\n",
           quant_parms_.exp2_inv_r_out_, quant_parms_.zp_r_out_);
    printf("  exp2_inv_g_out_ = %d, zp_g_out_ = %d\n",
           quant_parms_.exp2_inv_g_out_, quant_parms_.zp_g_out_);
    printf("  exp2_inv_Rh_add_br_ = %d, zp_Rh_add_br_ = %d\n",
           quant_parms_.exp2_inv_Rh_add_br_, quant_parms_.zp_Rh_add_br_);
    printf("  exp2_inv_rRh_ = %d, zp_rRh_ = %d\n",
           quant_parms_.exp2_inv_rRh_, quant_parms_.zp_rRh_);
    printf("  exp2_inv_one_minus_update_ = %d, zp_one_minus_update_ = %d\n",
           quant_parms_.exp2_inv_one_minus_update_,
           quant_parms_.zp_one_minus_update_);
    printf("  exp2_inv_new_contrib_ = %d, zp_new_contrib_ = %d\n",
           quant_parms_.exp2_inv_new_contrib_,
           quant_parms_.zp_new_contrib_);
    printf("  exp2_inv_old_contrib_ = %d, zp_old_contrib_ = %d\n",
           quant_parms_.exp2_inv_old_contrib_,
           quant_parms_.zp_old_contrib_);

}

template<typename QuantT>
inline bool Quantized_unit_testing<QuantT>::checkWxGemm() {

    bool is_pass = true;
    const int M = hidden_size_ * 3;
    const int N = batch_size_ * time_steps_;
    const int K = input_size_;
    const int lda = M;
    const int ldb = K;
    const int ldc = M;
    std::vector<float> Wx_cpu(batch_size_ * time_steps_ * hidden_size_ * 3);

#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += W_[lda * k + m] * x_[n * ldb + k];
            }
            Wx_cpu[n * ldc + m] = sum;
        }
    }

    std::vector<float> Wx_requant_cpu(Wx_cpu.size());
    std::vector<int32_t> W_sum_mul_x_zp(M);
    const int32_t exp2_inv_x_val = quant_parms_.exp2_inv_x_;
    const int32_t zp_x_val = quant_parms_.zp_x_;
#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const int32_t exp2_inv_W_val = quant_parms_.exp2_inv_W_[m];
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                const int8_t W_quant_val = W_quant_[lda * k + m];
                const int8_t x_quant_val = x_quant_[n * ldb + k];
                const float W_quant_val_float = dequant_from_exp2(W_quant_val, exp2_inv_W_val, 0);
                const float x_quant_val_float = dequant_from_exp2(x_quant_val, exp2_inv_x_val, zp_x_val);
                sum += W_quant_val_float * x_quant_val_float;
            }
            const int Wx_idx = n * ldc + m;
            Wx_requant_cpu[Wx_idx] = sum;
        }
    }

    is_pass &= checkCosineSimilarity(Wx_cpu, Wx_requant_cpu, "Wx_requant_cpu");
    is_pass &= checkMSE(Wx_cpu, Wx_requant_cpu, "Wx_requant_cpu", 1e-3);

    // dev::vector<float> W_dev(W_);
    // dev::vector<float> x_dev(x_);
    // dev::vector<float> tmp_Wx_dev(batch_size_ * time_steps_ * hidden_size_ * 3);

    // const float alpha = 1.0f;
    // const float beta = 0.0f;
    // blas<float>::gemm(handle_,
    //                   CUBLAS_OP_N, CUBLAS_OP_N,
    //                   M, N, K,
    //                   &alpha,
    //                   W_dev.data(), lda,
    //                   x_dev.data(), ldb,
    //                   &beta,
    //                   tmp_Wx_dev.data(), ldc);
    // cudaDeviceSynchronize();

    // is_pass = checkData(Wx_cpu, tmp_Wx_dev);
    // if (!is_pass) {
    //     printf("Error, checkWxGemm failed\n");
    // }
    return is_pass;
}

template<typename QuantT>
inline bool Quantized_unit_testing<QuantT>::checkQuantParameters() {
    bool is_pass = true;

    is_pass &= checkScale(x_, x_quant_, quant_parms_.exp2_inv_x_, quant_parms_.zp_x_, "scale_x_");
    printf("checkScale: scale_x_ over\n");
    is_pass &= checkScalePerChannel(W_,
                                    channels_,
                                    input_size_,
                                    W_quant_,
                                    quant_parms_.exp2_inv_W_,
                                    "scale_W_");
    printf("checkScalePerChannel: scale_W_ over\n");
    is_pass &= checkScalePerChannel(R_,
                                    channels_,
                                    hidden_size_,
                                    R_quant_,
                                    quant_parms_.exp2_inv_R_,
                                    "scale_R_");
    printf("checkScalePerChannel: scale_R_ over\n");
    is_pass &= checkScalePerChannel(bx_, channels_, 1, bx_quant_, quant_parms_.exp2_inv_bx_, "scale_bx_");
    printf("checkScalePerChannel: scale_bx_ over\n");
    is_pass &= checkScalePerChannel(br_, channels_, 1, br_quant_, quant_parms_.exp2_inv_br_, "scale_br_");
    printf("checkScalePerChannel: scale_br_ over\n");
    is_pass &= checkWxGemm();
    printf("checkGemm: over\n");
    if (!is_pass) {
        printf("Error, checkQuantParameters failed\n");
    }
    return is_pass;
}
