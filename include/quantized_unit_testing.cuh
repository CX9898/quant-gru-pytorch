#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <string>
#include <vector>

#include "blas.h"
#include "checkData.hpp"
#include "gru_quant.h"
#include "quantize_ops_helper.hpp"

// 梯度输出结构体
struct GRUTrainGradients {
    std::vector<float> dx; // 输入序列梯度 [time_steps * batch_size * input_size]
    std::vector<float> dW; // 对输入权重的梯度 [input_size * hidden_size * 3]
    std::vector<float> dR; // 对循环权重的梯度 [hidden_size * hidden_size * 3]
    std::vector<float> dbx;// 对输入偏置的梯度 [hidden_size * 3]
    std::vector<float> dbr;// 对循环偏置的梯度 [hidden_size * 3]
    std::vector<float> dh; // 对最后隐藏状态的梯度 [batch_size * hidden_size]
    std::vector<float> v;  // V中间值 [time_steps * batch_size * hidden_size * 4]
    std::vector<float> h;  // 隐藏状态 [time_steps * batch_size * hidden_size] (不包含初始状态)
};

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
        const float req_val = dequantize(quant[i], exp2_inv, zero_point);
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
        const QuantT req_val = quantize<QuantT>(src[i], exp2_inv, zero_point);
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
            const T req_val = dequantize(quant[idx], exp2_inv_val, zp_val);
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
            const QuantT quant_val = quantize<QuantT>(src[idx], exp2_inv_val, zp_val);
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
                const float W_quant_val_float = dequantize(W_quant_val, exp2_inv_W_val, 0);
                const float x_quant_val_float = dequantize(x_quant_val, exp2_inv_x_val, zp_x_val);
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

void checkQuantificationHostAndDevice(
    const std::vector<float> &W,
    const std::vector<float> &R,
    const std::vector<float> &bx,
    const std::vector<float> &br,
    const std::vector<float> &x,
    const GRUQuantitativeParameters &quant_parms,
    int time_steps,
    int batch_size,
    int input_size,
    int hidden_size) {
    // ========== 验证CPU和GPU量化结果一致性 ==========
    printf("\n========== 验证CPU和GPU量化结果一致性 ==========\n");

    const int channel_size = hidden_size * 3;
    const std::size_t x_size = time_steps * batch_size * input_size;

    // CPU版本量化结果
    std::vector<int8_t> W_quant_cpu(input_size * hidden_size * 3);
    std::vector<int8_t> R_quant_cpu(hidden_size * hidden_size * 3);
    std::vector<int32_t> bx_quant_cpu(hidden_size * 3);
    std::vector<int32_t> br_quant_cpu(hidden_size * 3);
    std::vector<int8_t> x_quant_cpu(x_size);

    {
        quantificationPerChannel(W.data(), W_quant_cpu.data(), input_size, channel_size,
                                 quant_parms.exp2_inv_W_);
        quantificationPerChannel(R.data(), R_quant_cpu.data(), hidden_size, channel_size,
                                 quant_parms.exp2_inv_R_);
        quantificationPerChannel(bx.data(), bx_quant_cpu.data(), 1, channel_size,
                                 quant_parms.exp2_inv_bx_);
        quantificationPerChannel(br.data(), br_quant_cpu.data(), 1, channel_size,
                                 quant_parms.exp2_inv_br_);
        quantification(x.data(), x_quant_cpu.data(), x_size, quant_parms.exp2_inv_x_,
                       quant_parms.zp_x_);
    }

    // GPU版本量化结果
    dev::vector<float> W_dev(W);
    dev::vector<float> R_dev(R);
    dev::vector<float> bx_dev(bx);
    dev::vector<float> br_dev(br);
    dev::vector<float> x_dev(x);

    dev::vector<int8_t> W_quant_gpu(input_size * hidden_size * 3);
    dev::vector<int8_t> R_quant_gpu(hidden_size * hidden_size * 3);
    dev::vector<int32_t> bx_quant_gpu(hidden_size * 3);
    dev::vector<int32_t> br_quant_gpu(hidden_size * 3);
    dev::vector<int8_t> x_quant_gpu(x_size);

    dev::vector<int32_t> exp2_inv_W_dev(quant_parms.exp2_inv_W_);
    dev::vector<int32_t> exp2_inv_R_dev(quant_parms.exp2_inv_R_);
    dev::vector<int32_t> exp2_inv_bx_dev(quant_parms.exp2_inv_bx_);
    dev::vector<int32_t> exp2_inv_br_dev(quant_parms.exp2_inv_br_);

    {
        dev::quantificationPerChannel(W_dev.data(), W_quant_gpu.data(), input_size, channel_size,
                                      exp2_inv_W_dev);
        dev::quantificationPerChannel(R_dev.data(), R_quant_gpu.data(), hidden_size, channel_size,
                                      exp2_inv_R_dev);
        dev::quantificationPerChannel(bx_dev.data(), bx_quant_gpu.data(), 1, channel_size,
                                      exp2_inv_bx_dev);
        dev::quantificationPerChannel(br_dev.data(), br_quant_gpu.data(), 1, channel_size,
                                      exp2_inv_br_dev);
        dev::quantification(x_dev.data(), x_quant_gpu.data(), x_size, quant_parms.exp2_inv_x_,
                            quant_parms.zp_x_);
    }

    // 将GPU结果复制到CPU进行比较
    std::vector<int8_t> W_quant_gpu_host = d2h(W_quant_gpu);
    std::vector<int8_t> R_quant_gpu_host = d2h(R_quant_gpu);
    std::vector<int32_t> bx_quant_gpu_host = d2h(bx_quant_gpu);
    std::vector<int32_t> br_quant_gpu_host = d2h(br_quant_gpu);
    std::vector<int8_t> x_quant_gpu_host = d2h(x_quant_gpu);

    // 比较结果
    int mismatch_count = 0;
    const int max_show_mismatches = 10;

    // 比较W
    printf("\n比较W量化结果 (大小: %zu):\n", W_quant_cpu.size());
    for (size_t i = 0; i < W_quant_cpu.size(); ++i) {
        if (W_quant_cpu[i] != W_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  W[%zu]: CPU=%d, GPU=%d\n", i,
                       static_cast<int>(W_quant_cpu[i]),
                       static_cast<int>(W_quant_gpu_host[i]));
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ W量化结果完全一致\n");
    } else {
        printf("  ✗ W量化结果有 %d 个不匹配\n", mismatch_count);
    }

    // 比较R
    mismatch_count = 0;
    printf("\n比较R量化结果 (大小: %zu):\n", R_quant_cpu.size());
    for (size_t i = 0; i < R_quant_cpu.size(); ++i) {
        if (R_quant_cpu[i] != R_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  R[%zu]: CPU=%d, GPU=%d\n", i,
                       static_cast<int>(R_quant_cpu[i]),
                       static_cast<int>(R_quant_gpu_host[i]));
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ R量化结果完全一致\n");
    } else {
        printf("  ✗ R量化结果有 %d 个不匹配\n", mismatch_count);
    }

    // 比较bx
    mismatch_count = 0;
    printf("\n比较bx量化结果 (大小: %zu):\n", bx_quant_cpu.size());
    for (size_t i = 0; i < bx_quant_cpu.size(); ++i) {
        if (bx_quant_cpu[i] != bx_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  bx[%zu]: CPU=%d, GPU=%d\n", i,
                       bx_quant_cpu[i], bx_quant_gpu_host[i]);
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ bx量化结果完全一致\n");
    } else {
        printf("  ✗ bx量化结果有 %d 个不匹配\n", mismatch_count);
    }

    // 比较br
    mismatch_count = 0;
    printf("\n比较br量化结果 (大小: %zu):\n", br_quant_cpu.size());
    for (size_t i = 0; i < br_quant_cpu.size(); ++i) {
        if (br_quant_cpu[i] != br_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  br[%zu]: CPU=%d, GPU=%d\n", i,
                       br_quant_cpu[i], br_quant_gpu_host[i]);
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ br量化结果完全一致\n");
    } else {
        printf("  ✗ br量化结果有 %d 个不匹配\n", mismatch_count);
    }

    // 比较x
    mismatch_count = 0;
    printf("\n比较x量化结果 (大小: %zu):\n", x_quant_cpu.size());
    for (size_t i = 0; i < x_quant_cpu.size(); ++i) {
        if (x_quant_cpu[i] != x_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  x[%zu]: CPU=%d, GPU=%d\n", i,
                       static_cast<int>(x_quant_cpu[i]),
                       static_cast<int>(x_quant_gpu_host[i]));
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ x量化结果完全一致\n");
    } else {
        printf("  ✗ x量化结果有 %d 个不匹配\n", mismatch_count);
    }

    printf("\n===========================================================\n\n");
}

// ========== 统一的测试函数 ==========

/**
 * @brief 比较浮点和量化版本的V中间值
 * @param v_float 浮点版本的V中间值
 * @param v_quant_dequant 量化后反量化的V中间值
 * @param time_steps 时间步数
 * @param batch_size 批次大小
 * @param hidden_size 隐藏层维度
 * @param prefix 输出前缀（可选）
 */
inline void compareVIntermediateValues(
    const std::vector<float> &v_float,
    const std::vector<float> &v_quant_dequant,
    int time_steps,
    int batch_size,
    int hidden_size,
    const std::string &prefix = "") {
    printf("\n========== %s V Intermediate Values Comparison ==========\n", prefix.c_str());

    const int v_size_per_step = batch_size * hidden_size * 4;// 4个部分：z_out, r_out, g_out, Rh_add_br
    const int v_size_per_part = batch_size * hidden_size;    // 每个部分的大小

    // 验证大小
    if (v_float.size() != static_cast<size_t>(time_steps * v_size_per_step)) {
        printf("[Error] v_float size mismatch: expected %d, got %zu\n",
               time_steps * v_size_per_step, v_float.size());
        return;
    }
    if (v_quant_dequant.size() != static_cast<size_t>(time_steps * v_size_per_step)) {
        printf("[Error] v_quant_dequant size mismatch: expected %d, got %zu\n",
               time_steps * v_size_per_step, v_quant_dequant.size());
        return;
    }

    // V的4个部分名称
    const char *part_names[] = {"z_out", "r_out", "g_out", "Rh_add_br"};

    // 整体比较
    {
        const float mse = computeMSE(v_float, v_quant_dequant);
        const float cos_sim = computeCosineSimilarity(v_float, v_quant_dequant);
        printf("Overall V: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 按部分比较（所有时间步）
    for (int part = 0; part < 4; ++part) {
        std::vector<float> v_float_part(time_steps * v_size_per_part);
        std::vector<float> v_quant_part(time_steps * v_size_per_part);

        for (int t = 0; t < time_steps; ++t) {
            const int t_offset = t * v_size_per_step;
            const int part_offset = part * hidden_size;

            for (int b = 0; b < batch_size; ++b) {
                const int b_offset = b * hidden_size;
                for (int h = 0; h < hidden_size; ++h) {
                    const int src_idx = t_offset + b_offset + part_offset + h;
                    const int dst_idx = t * v_size_per_part + b_offset + h;
                    v_float_part[dst_idx] = v_float[src_idx];
                    v_quant_part[dst_idx] = v_quant_dequant[src_idx];
                }
            }
        }

        const float mse = computeMSE(v_float_part, v_quant_part);
        const float cos_sim = computeCosineSimilarity(v_float_part, v_quant_part);
        printf("%s: MSE = %e, Cosine Similarity = %f\n", part_names[part], mse, cos_sim);
    }

    // 按时间步比较（所有部分）
    printf("\nPer time step comparison:\n");
    for (int t = 0; t < time_steps && t < 10; ++t) {// 只显示前10个时间步
        const int t_offset = t * v_size_per_step;
        std::vector<float> v_float_step(v_size_per_step);
        std::vector<float> v_quant_step(v_size_per_step);

        for (int i = 0; i < v_size_per_step; ++i) {
            v_float_step[i] = v_float[t_offset + i];
            v_quant_step[i] = v_quant_dequant[t_offset + i];
        }

        const float mse = computeMSE(v_float_step, v_quant_step);
        const float cos_sim = computeCosineSimilarity(v_float_step, v_quant_step);
        printf("  Time step %d: MSE = %e, Cosine Similarity = %f\n", t, mse, cos_sim);
    }

    printf("===========================================================\n\n");
}

/**
 * @brief 比较浮点和量化版本的h隐藏状态（不包含初始状态）
 * @param h_float 浮点版本的h隐藏状态，size = time_steps * batch_size * hidden_size（不包含初始状态t=0）
 * @param h_quant_dequant 量化后反量化的h隐藏状态，size同上
 * @param time_steps 时间步数
 * @param batch_size 批次大小
 * @param hidden_size 隐藏层维度
 * @param prefix 输出前缀（可选）
 */
inline void compareHValues(
    const std::vector<float> &h_float,
    const std::vector<float> &h_quant_dequant,
    int time_steps,
    int batch_size,
    int hidden_size,
    const std::string &prefix = "") {
    printf("\n========== %s H Hidden States Comparison ==========\n", prefix.c_str());

    const int h_size_per_step = batch_size * hidden_size;// 每个时间步的大小

    // 验证大小
    if (h_quant_dequant.size() != h_float.size()) {
        printf("[Error] h_float and h_quant_dequant size mismatch: h_float_size = %zu, h_quant_dequant_size = %zu\n",
                   h_float.size(), h_quant_dequant.size());
        return;
    }

    // 整体比较
    {
        const float mse = computeMSE(h_float, h_quant_dequant);
        const float cos_sim = computeCosineSimilarity(h_float, h_quant_dequant);
        printf("Overall H: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 按时间步比较
    printf("\nPer time step comparison:\n");
    for (int t = 0; t < time_steps; ++t) {
        const int t_offset = t * h_size_per_step;
        std::vector<float> h_float_step(h_size_per_step);
        std::vector<float> h_quant_step(h_size_per_step);

        for (int i = 0; i < h_size_per_step; ++i) {
            h_float_step[i] = h_float[t_offset + i];
            h_quant_step[i] = h_quant_dequant[t_offset + i];
        }

        const float mse = computeMSE(h_float_step, h_quant_step);
        const float cos_sim = computeCosineSimilarity(h_float_step, h_quant_step);
        printf("  Time step %d: MSE = %e, Cosine Similarity = %f\n", t, mse, cos_sim);
    }

    // 按批次比较（所有时间步）
    printf("\nPer batch comparison:\n");
    for (int b = 0; b < batch_size && b < 5; ++b) {// 只显示前5个批次
        std::vector<float> h_float_batch(time_steps * hidden_size);
        std::vector<float> h_quant_batch(time_steps * hidden_size);

        for (int t = 0; t < time_steps; ++t) {
            const int t_offset = t * h_size_per_step;
            const int b_offset = b * hidden_size;

            for (int h = 0; h < hidden_size; ++h) {
                const int src_idx = t_offset + b_offset + h;
                const int dst_idx = t * hidden_size + h;
                h_float_batch[dst_idx] = h_float[src_idx];
                h_quant_batch[dst_idx] = h_quant_dequant[src_idx];
            }
        }

        const float mse = computeMSE(h_float_batch, h_quant_batch);
        const float cos_sim = computeCosineSimilarity(h_float_batch, h_quant_batch);
        printf("  Batch %d: MSE = %e, Cosine Similarity = %f\n", b, mse, cos_sim);
    }

    printf("===========================================================\n\n");
}

/**
 * @brief 比较两个GRU训练梯度的差异
 * @param gradients_float 浮点版本的梯度
 * @param gradients_quant 量化版本的梯度
 * @param prefix 输出前缀（可选）
 */
inline void compareGRUTrainGradients(const GRUTrainGradients &gradients_float,
                                     const GRUTrainGradients &gradients_quant,
                                     const std::string &prefix = "") {
    printf("\n========== %s GRU Train Gradients Comparison ==========\n", prefix.c_str());

    // 比较 dx
    {
        const float mse = computeMSE(gradients_float.dx, gradients_quant.dx);
        const float cos_sim = computeCosineSimilarity(gradients_float.dx, gradients_quant.dx);
        printf("dx: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dW
    {
        const float mse = computeMSE(gradients_float.dW, gradients_quant.dW);
        const float cos_sim = computeCosineSimilarity(gradients_float.dW, gradients_quant.dW);
        printf("dW: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dR
    {
        const float mse = computeMSE(gradients_float.dR, gradients_quant.dR);
        const float cos_sim = computeCosineSimilarity(gradients_float.dR, gradients_quant.dR);
        printf("dR: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dbx
    {
        const float mse = computeMSE(gradients_float.dbx, gradients_quant.dbx);
        const float cos_sim = computeCosineSimilarity(gradients_float.dbx, gradients_quant.dbx);
        printf("dbx: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dbr
    {
        const float mse = computeMSE(gradients_float.dbr, gradients_quant.dbr);
        const float cos_sim = computeCosineSimilarity(gradients_float.dbr, gradients_quant.dbr);
        printf("dbr: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dh
    {
        const float mse = computeMSE(gradients_float.dh, gradients_quant.dh);
        const float cos_sim = computeCosineSimilarity(gradients_float.dh, gradients_quant.dh);
        printf("dh: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    printf("===========================================================\n\n");
}

/**
 * @brief 检查量化h值与浮点h值的相似度（统一使用compareHValues的逻辑）
 * @param h_inference 浮点版本的h值，size = (time_steps+1) * batch_size * hidden_size（包含初始状态）
 * @param h_quant_inference 量化版本的h值，size同上
 * @param time_steps 时间步数
 * @param batch_size 批次大小
 * @param hidden_size 隐藏层维度
 * @param scaleParam 量化参数
 * @param prefix 输出前缀（可选）
 */
template<typename QuantT>
inline void checkHQuantizationWithCosine(
    const std::vector<float> &h_inference,// 浮点 h, size = (time_steps+1) * batch_size * hidden_size
    const std::vector<QuantT> &h_quant_inference,// 量化 h, size 同上
    int time_steps, int batch_size, int hidden_size,
    const GRUQuantitativeParameters &scaleParam,
    const std::string &prefix = "") {
    const int size_per_step = batch_size * hidden_size;

    // 验证输入数据大小
    if (h_inference.size() !=
        static_cast<size_t>((time_steps + 1) * size_per_step)) {
        printf("[Error] h_inference size mismatch: expected %d, got %zu\n",
               (time_steps + 1) * size_per_step, h_inference.size());
        return;
    }
    if (h_quant_inference.size() !=
        static_cast<size_t>((time_steps + 1) * size_per_step)) {
        printf(
            "[Error] h_quant_inference size mismatch: expected %d, got %zu\n",
            (time_steps + 1) * size_per_step, h_quant_inference.size());
        return;
    }

    // 打印量化参数信息
    printf("\n%s Quantization Parameters: exp2_inv_h_=%d, zp_h_=%d\n",
           prefix.empty() ? "H" : prefix.c_str(),
           scaleParam.exp2_inv_h_, scaleParam.zp_h_);

    // 反量化整个量化h值（跳过初始状态t=0，只处理t=1到t=time_steps）
    std::vector<float> h_quant_dequant(time_steps * size_per_step);
    for (int t = 1; t <= time_steps; ++t) {
        const size_t t_offset = static_cast<size_t>(t) * size_per_step;
        const size_t dst_offset = static_cast<size_t>(t - 1) * size_per_step;
        
        for (int idx = 0; idx < size_per_step; ++idx) {
            const QuantT quant_val = h_quant_inference[t_offset + idx];
            h_quant_dequant[dst_offset + idx] = dequantize<QuantT>(
                quant_val, scaleParam.exp2_inv_h_, scaleParam.zp_h_);
        }
    }

    // 提取浮点h值（跳过初始状态t=0，只处理t=1到t=time_steps）
    std::vector<float> h_float(time_steps * size_per_step);
    for (int t = 1; t <= time_steps; ++t) {
        const size_t src_offset = static_cast<size_t>(t) * size_per_step;
        const size_t dst_offset = static_cast<size_t>(t - 1) * size_per_step;
        
        for (int idx = 0; idx < size_per_step; ++idx) {
            h_float[dst_offset + idx] = h_inference[src_offset + idx];
        }
    }

    // 使用统一的compareHValues函数进行比较
    compareHValues(h_float, h_quant_dequant, time_steps, batch_size, hidden_size, prefix);
}
