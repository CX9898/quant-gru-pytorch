// =====================================================================
// GRU 接口层实现 (gru_interface.cpp)
// =====================================================================

#include "gru_interface.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "parallelAlgorithm.h"
#include "quantize_ops_helper.hpp"

// =====================================================================
// 量化校准实现
// =====================================================================

void calibrateGruRanges(int time_steps, int batch_size, int input_size, int hidden_size,
                        const float *W, const float *R, const float *bx, const float *br,
                        const float *x, const cublasHandle_t &g_blas_handle,
                        GRUQuantizationRanges &quant_ranges) {
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    // 初始化 quant_ranges（如果尚未初始化）
    if (quant_ranges.hidden_ != hidden_size) {
        quant_ranges.reset(hidden_size);
    }

    gru::ForwardPass<float> forward =
        gru::ForwardPass<float>(true,  // training
                                batch_size, input_size, hidden_size, g_blas_handle);

    forward.setCalibrationMode(true, quant_ranges);

    forward.Run(time_steps, W, R, bx, br, x, h_dev.data(), v_dev.data(), tmp_Wx_dev.data(),
                tmp_Rh_dev.data(), 0.0f, nullptr);

    // 同步所有 CUDA 操作，确保校准完成
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in calibrateGruRanges: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in calibrateGruRanges: ") + err_str);
    }

    // 获取更新后的范围
    quant_ranges = forward.getGRUQuantizationRanges();
}

// =====================================================================
// 量化校准实现
// =====================================================================

// 确保范围不小于最小阈值，避免范围过窄导致量化精度问题
inline void ensureMinRange(float &min_val, float &max_val, float min_range_threshold = 0.1f,
                           const char *name = nullptr) {
    float range = max_val - min_val;
    if (range < min_range_threshold) {
        float center = (min_val + max_val) / 2.0f;
        float old_min = min_val, old_max = max_val;
        min_val = center - min_range_threshold / 2.0f;
        max_val = center + min_range_threshold / 2.0f;
        if (name) {
            printf(
                "[ensureMinRange] %s: range %.4f < %.4f, expanded [%.4f, %.4f] -> [%.4f, %.4f]\n",
                name, range, min_range_threshold, old_min, old_max, min_val, max_val);
        }
    }
}

GRUQuantitativeParameters calculateGRUQuantitativeParameters(
    const GRUQuantizationRanges &quant_ranges, const OperatorQuantConfig &bitwidth_config) {
    GRUQuantitativeParameters quant_params;
    quant_params.hidden_ = quant_ranges.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;

    // 输入 x 的量化（使用配置的对称量化设置）
    dispatchByBitWidth(bitwidth_config.x_, [&](auto tag) {
        using XT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, XT>(quant_ranges.min_x_, quant_ranges.max_x_,
                                        bitwidth_config.x_symmetric_, aligned_min, aligned_max,
                                        quant_params.exp2_inv_x_, quant_params.zp_x_, "scale_x");
    });

    // 隐藏状态 h 的量化
    dispatchByBitWidth(bitwidth_config.h_, [&](auto tag) {
        using HT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, HT>(quant_ranges.min_h_, quant_ranges.max_h_,
                                        bitwidth_config.h_symmetric_, aligned_min, aligned_max,
                                        quant_params.exp2_inv_h_, quant_params.zp_h_, "scale_h");
    });

    // 权重 W 的量化（per-channel）
    const int channel_size = quant_ranges.hidden_ * 3;
    quant_params.exp2_inv_W_.resize(channel_size);
    dispatchByBitWidth(bitwidth_config.W_, [&](auto tag) {
        using WT = typename decltype(tag)::type;
        for (int c = 0; c < channel_size; ++c) {
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams<float, WT>(quant_ranges.min_W_[c], quant_ranges.max_W_[c],
                                            bitwidth_config.W_symmetric_, aligned_min, aligned_max,
                                            quant_params.exp2_inv_W_[c], zp_tmp, "scale_W");
        }
    });

    // 权重 R 的量化（per-channel）
    quant_params.exp2_inv_R_.resize(channel_size);
    dispatchByBitWidth(bitwidth_config.R_, [&](auto tag) {
        using RT = typename decltype(tag)::type;
        for (int c = 0; c < channel_size; ++c) {
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams<float, RT>(quant_ranges.min_R_[c], quant_ranges.max_R_[c],
                                            bitwidth_config.R_symmetric_, aligned_min, aligned_max,
                                            quant_params.exp2_inv_R_[c], zp_tmp, "scale_R");
        }
    });

    // Wx 结果的量化
    dispatchByBitWidth(bitwidth_config.Wx_, [&](auto tag) {
        using WxT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, WxT>(
            quant_ranges.min_Wx_, quant_ranges.max_Wx_, bitwidth_config.Wx_symmetric_, aligned_min,
            aligned_max, quant_params.exp2_inv_Wx_, quant_params.zp_Wx_, "scale_Wx");
    });

    // Rh 结果的量化
    dispatchByBitWidth(bitwidth_config.Rh_, [&](auto tag) {
        using RhT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, RhT>(
            quant_ranges.min_Rh_, quant_ranges.max_Rh_, bitwidth_config.Rh_symmetric_, aligned_min,
            aligned_max, quant_params.exp2_inv_Rh_, quant_params.zp_Rh_, "scale_Rh");
    });

    // 偏置 bx 的量化（per-channel）
    quant_params.exp2_inv_bx_.resize(channel_size);
    dispatchByBitWidth(bitwidth_config.bx_, [&](auto tag) {
        using BxT = typename decltype(tag)::type;
        for (int c = 0; c < channel_size; ++c) {
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams<float, BxT>(
                quant_ranges.min_bx_[c], quant_ranges.max_bx_[c], bitwidth_config.bx_symmetric_,
                aligned_min, aligned_max, quant_params.exp2_inv_bx_[c], zp_tmp, "scale_bx");
        }
    });

    // 偏置 br 的量化（per-channel）
    quant_params.exp2_inv_br_.resize(channel_size);
    dispatchByBitWidth(bitwidth_config.br_, [&](auto tag) {
        using BrT = typename decltype(tag)::type;
        for (int c = 0; c < channel_size; ++c) {
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams<float, BrT>(
                quant_ranges.min_br_[c], quant_ranges.max_br_[c], bitwidth_config.br_symmetric_,
                aligned_min, aligned_max, quant_params.exp2_inv_br_[c], zp_tmp, "scale_br");
        }
    });

    // z 门输入的量化
    dispatchByBitWidth(bitwidth_config.z_pre_, [&](auto tag) {
        using ZPreT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, ZPreT>(quant_ranges.min_z_pre_, quant_ranges.max_z_pre_,
                                           bitwidth_config.z_pre_symmetric_, aligned_min,
                                           aligned_max, quant_params.exp2_inv_z_pre_,
                                           quant_params.zp_z_pre_, "scale_z_pre");
    });

    // r 门输入的量化
    dispatchByBitWidth(bitwidth_config.r_pre_, [&](auto tag) {
        using RPreT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, RPreT>(quant_ranges.min_r_pre_, quant_ranges.max_r_pre_,
                                           bitwidth_config.r_pre_symmetric_, aligned_min,
                                           aligned_max, quant_params.exp2_inv_r_pre_,
                                           quant_params.zp_r_pre_, "scale_r_pre");
    });

    // g 门输入的量化
    dispatchByBitWidth(bitwidth_config.g_pre_, [&](auto tag) {
        using GPreT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, GPreT>(quant_ranges.min_g_pre_, quant_ranges.max_g_pre_,
                                           bitwidth_config.g_pre_symmetric_, aligned_min,
                                           aligned_max, quant_params.exp2_inv_g_pre_,
                                           quant_params.zp_g_pre_, "scale_g_pre");
    });

    // 激活函数输出的校准
    // INT8: 使用实际校准范围（精度更高）
    // INT16: 使用固定范围（LUT 精度足够，固定范围更稳定）
    constexpr float MIN_ACTIVATION_RANGE = 0.5f;

    // z 门输出的量化 - sigmoid 输出固定范围 [0, 1]
    dispatchByBitWidth(bitwidth_config.z_out_, [&](auto tag) {
        using ZOutT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        float min_val, max_val;
        if constexpr (sizeof(ZOutT) == 1) {
            // INT8: 使用实际校准范围
            min_val = quant_ranges.min_z_out_;
            max_val = quant_ranges.max_z_out_;
            ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, "z_out");
        } else {
            // INT16: 使用固定范围 [0, 1]
            min_val = 0.0f;
            max_val = 1.0f;
        }
        calibrateQuantParams<float, ZOutT>(min_val, max_val, bitwidth_config.z_out_symmetric_,
                                           aligned_min, aligned_max, quant_params.exp2_inv_z_out_,
                                           quant_params.zp_z_out_, "scale_z_out");
    });

    // r 门输出的量化 - sigmoid 输出固定范围 [0, 1]
    dispatchByBitWidth(bitwidth_config.r_out_, [&](auto tag) {
        using ROutT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        float min_val, max_val;
        if constexpr (sizeof(ROutT) == 1) {
            min_val = quant_ranges.min_r_out_;
            max_val = quant_ranges.max_r_out_;
            ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, "r_out");
        } else {
            min_val = 0.0f;
            max_val = 1.0f;
        }
        calibrateQuantParams<float, ROutT>(min_val, max_val, bitwidth_config.r_out_symmetric_,
                                           aligned_min, aligned_max, quant_params.exp2_inv_r_out_,
                                           quant_params.zp_r_out_, "scale_r_out");
    });

    // g 门输出的量化 - tanh 输出固定范围 [-1, 1]，使用对称量化
    dispatchByBitWidth(bitwidth_config.g_out_, [&](auto tag) {
        using GOutT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        float min_val, max_val;
        if constexpr (sizeof(GOutT) == 1) {
            min_val = quant_ranges.min_g_out_;
            max_val = quant_ranges.max_g_out_;
            ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, "g_out");
        } else {
            min_val = -1.0f;
            max_val = 1.0f;
        }
        calibrateQuantParams<float, GOutT>(min_val, max_val, bitwidth_config.g_out_symmetric_,
                                           aligned_min, aligned_max, quant_params.exp2_inv_g_out_,
                                           quant_params.zp_g_out_, "scale_g_out");
    });

    // Rh + br 的量化
    dispatchByBitWidth(bitwidth_config.Rh_add_br_, [&](auto tag) {
        using RhAddBrT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, RhAddBrT>(
            quant_ranges.min_Rh_add_br_g_, quant_ranges.max_Rh_add_br_g_,
            bitwidth_config.Rh_add_br_symmetric_, aligned_min, aligned_max,
            quant_params.exp2_inv_Rh_add_br_, quant_params.zp_Rh_add_br_, "scale_Rh_add_br");
    });

    // r × Rh 的量化
    dispatchByBitWidth(bitwidth_config.rRh_, [&](auto tag) {
        using rRhT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, rRhT>(quant_ranges.min_rRh_, quant_ranges.max_rRh_,
                                          bitwidth_config.rRh_symmetric_, aligned_min, aligned_max,
                                          quant_params.exp2_inv_rRh_, quant_params.zp_rRh_,
                                          "scale_rRh");
    });

    // (1.0 - z) * g 的量化
    dispatchByBitWidth(bitwidth_config.new_contrib_, [&](auto tag) {
        using NewContribT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, NewContribT>(
            quant_ranges.min_new_contrib_, quant_ranges.max_new_contrib_,
            bitwidth_config.new_contrib_symmetric_, aligned_min, aligned_max,
            quant_params.exp2_inv_new_contrib_, quant_params.zp_new_contrib_, "scale_new_contrib");
    });

    // z * h 的量化
    dispatchByBitWidth(bitwidth_config.old_contrib_, [&](auto tag) {
        using OldContribT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        calibrateQuantParams<float, OldContribT>(
            quant_ranges.min_old_contrib_, quant_ranges.max_old_contrib_,
            bitwidth_config.old_contrib_symmetric_, aligned_min, aligned_max,
            quant_params.exp2_inv_old_contrib_, quant_params.zp_old_contrib_, "scale_old_contrib");
    });

    return quant_params;
}

GRUQuantitativeParameters calibrateGruScales(int time_steps, int batch_size, int input_size,
                                             int hidden_size, const float *W, const float *R,
                                             const float *bx, const float *br, const float *x,
                                             const cublasHandle_t &g_blas_handle,
                                             const OperatorQuantConfig &bitwidth_config) {
    // 首先校准范围
    GRUQuantizationRanges quant_ranges(hidden_size);

    calibrateGruRanges(time_steps, batch_size, input_size, hidden_size, W, R, bx, br, x,
                       g_blas_handle, quant_ranges);

    // 然后根据范围计算量化参数
    return calculateGRUQuantitativeParameters(quant_ranges, bitwidth_config);
}

GRUQuantitativeParameters calibrateGruScalesAndInitLut(
    int time_steps, int batch_size, int input_size, int hidden_size, const float *W, const float *R,
    const float *bx, const float *br, const float *x, const cublasHandle_t &g_blas_handle,
    const OperatorQuantConfig &bitwidth_config) {
    // 先校准量化参数
    GRUQuantitativeParameters quant_params =
        calibrateGruScales(time_steps, batch_size, input_size, hidden_size, W, R, bx, br, x,
                           g_blas_handle, bitwidth_config);

    // 初始化 LUT 表（根据 bitwidth_config_ 自动选择方法）
    initialize_quantization_lut(quant_params);

    return quant_params;
}

// =====================================================================
// 前向传播实现
// =====================================================================

void hasteGRUForward(bool is_training, const int time_steps, const int batch_size,
                     const int input_size, const int hidden_size, const float *W, const float *R,
                     const float *bx, const float *br, const float *x, const float *h0,
                     const cublasHandle_t &g_blas_handle, float *h, float *v) {
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size *
                                  3);                             // 用于存放W * x的中间结果
    dev::vector<float> tmp_Rh_dev(batch_size * hidden_size * 3);  // 用于存放R * h的中间结果

    // 处理初始隐藏状态
    const int NH = batch_size * hidden_size;
    if (h0 != nullptr) {
        // 如果提供了初始状态，复制到 h[0]
        d2d(h, h0, NH);
    } else {
        // 否则初始化为零
        cudaMemset(h, 0, NH * sizeof(float));
    }

    gru::ForwardPass<float> forward =
        gru::ForwardPass<float>(is_training,  // training: true为训练，false为推理
                                batch_size, input_size, hidden_size, g_blas_handle);

    forward.Run(time_steps, W, R, bx, br, x, h, v, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f,
                nullptr);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in hasteGRUForward: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in hasteGRUForward: ") + err_str);
    }
}

// =====================================================================
// 反向传播实现
// =====================================================================

// ★★★ 重要：W_t、R_t、x_t 需要传入【转置后】的数据！★★★
void hasteGRUBackward(const int time_steps, const int batch_size, const int input_size,
                      const int hidden_size, const float *W_t, const float *R_t, const float *bx,
                      const float *br, const float *x_t, const float *dh_new, const float *h,
                      const float *v, const cublasHandle_t &g_blas_handle, float *dx, float *dW,
                      float *dR, float *dbx, float *dbr, float *dh) {
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）

    gru::BackwardPass<float> backward(batch_size, input_size, hidden_size, g_blas_handle);

    backward.Run(time_steps, W_t, R_t, bx, br, x_t, h, v, dh_new, dx, dW, dR, dbx, dbr, dh,
                 dp_dev.data(), dq_dev.data(), nullptr);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in hasteGRUBackward: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in hasteGRUBackward: ") + err_str);
    }
}

// =====================================================================
// 权重量化实现
// =====================================================================

template <typename QuantT>
void quantitativeWeight(const int input_size, const int hidden_size, const float *W, const float *R,
                        const float *bx, const float *br,
                        const GRUQuantitativeParameters &quant_parms, QuantT *W_quant,
                        QuantT *R_quant, int32_t *bx_quant, int32_t *br_quant) {
    // 显式创建dev::vector以避免临时对象问题
    dev::vector<int8_t> exp2_inv_W_dev(quant_parms.exp2_inv_W_);
    dev::vector<int8_t> exp2_inv_R_dev(quant_parms.exp2_inv_R_);
    dev::vector<int8_t> exp2_inv_bx_dev(quant_parms.exp2_inv_bx_);
    dev::vector<int8_t> exp2_inv_br_dev(quant_parms.exp2_inv_br_);

    dev::quantificationPerChannel(W, W_quant, input_size, 3 * hidden_size, exp2_inv_W_dev);
    dev::quantificationPerChannel(R, R_quant, hidden_size, 3 * hidden_size, exp2_inv_R_dev);
    dev::quantificationPerChannel(bx, bx_quant, 1, 3 * hidden_size, exp2_inv_bx_dev);
    dev::quantificationPerChannel(br, br_quant, 1, 3 * hidden_size, exp2_inv_br_dev);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantitativeWeight: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantitativeWeight: ") + err_str);
    }
}

// 量化 GRU 前向传播
template <typename QuantT>
void quantGRUForward(bool is_training, const int time_steps, const int batch_size,
                     const int input_size, const int hidden_size, const QuantT *W, const QuantT *R,
                     const int32_t *bx, const int32_t *br, const float *x, const float *h0,
                     const GRUQuantitativeParameters &quant_parms,
                     const cublasHandle_t &g_blas_handle, float *h, float *v) {
    const std::size_t x_size = time_steps * batch_size * input_size;

    dev::vector<QuantT> x_quant(x_size);
    dev::quantification(x, x_quant.data(), x_size, quant_parms.exp2_inv_x_, quant_parms.zp_x_);

    dev::vector<QuantT> h_quant((time_steps + 1) * batch_size * hidden_size);
    // 初始化 h0 区域（第一个时间步的隐藏状态）为零点值
    dev::fill_n(h_quant.data(), batch_size * hidden_size, quant_parms.zp_h_);

    // 处理初始隐藏状态
    if (h0 != nullptr) {
        // 如果提供了初始状态，直接量化到 h_quant[0]
        dev::quantification(h0, h_quant.data(), batch_size * hidden_size, quant_parms.exp2_inv_h_,
                            quant_parms.zp_h_);
    }

    dev::vector<int32_t> v_quant_dev(time_steps * batch_size * hidden_size *
                                     4);  // v 统一使用 int32_t 存储
    // dev::vector<int32_t> tmp_Wx_dev(time_steps * batch_size * hidden_size *
    //                                 3);                             // 用于存放W * x的中间结果
    // dev::vector<int32_t> tmp_Rh_dev(batch_size * hidden_size * 3);  // 用于存放R * h的中间结果

    gru::ForwardPassQuant<QuantT, QuantT, QuantT, QuantT> forward =
        gru::ForwardPassQuant<QuantT, QuantT, QuantT, QuantT>(
            is_training,  // training: true为训练，false为推理
            batch_size, input_size, hidden_size, g_blas_handle);

    // 得到量化GRU中使用的rescale参数
    forward.setRescaleParam(quant_parms);

    forward.Run(time_steps, W, R, bx, br, x_quant.data(), h_quant.data(), v_quant_dev.data(), 0.0f,
                nullptr);

    dev::dequantification(h_quant.data(), h, (time_steps + 1) * batch_size * hidden_size,
                          quant_parms.exp2_inv_h_, quant_parms.zp_h_);

    // 如果v不为nullptr，反量化v并输出
    if (v != nullptr) {
        dev::dequantificationV(v_quant_dev.data(), v, time_steps, batch_size, hidden_size,
                               quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_,
                               quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_,
                               quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_,
                               quant_parms.exp2_inv_Rh_add_br_, quant_parms.zp_Rh_add_br_);
    }

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUForward: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUForward: ") + err_str);
    }
}

// 统一前向传播接口
void forwardInterface(bool is_training, bool is_quant, int time_steps, int batch_size,
                      int input_size, int hidden_size, const float *W, const float *R,
                      const float *bx, const float *br, const float *x, const float *h0,
                      const GRUQuantitativeParameters &quant_gru_scales,
                      const cublasHandle_t &g_blas_handle, float *h, float *v) {
    if (is_quant) {
        // 根据 bitwidth_config_.W_ 决定权重量化位宽
        const auto &config = quant_gru_scales.bitwidth_config_;
        if (config.W_ == QuantBitWidth::INT16) {
            dev::vector<int16_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int16_t> R_quant(hidden_size * 3 * hidden_size);
            dev::vector<int32_t> bx_quant(hidden_size * 3);
            dev::vector<int32_t> br_quant(hidden_size * 3);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales,
                               W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward(is_training, time_steps, batch_size, input_size, hidden_size,
                            W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data(), x, h0,
                            quant_gru_scales, g_blas_handle, h, v);
        } else {
            dev::vector<int8_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int8_t> R_quant(hidden_size * 3 * hidden_size);
            dev::vector<int32_t> bx_quant(hidden_size * 3);
            dev::vector<int32_t> br_quant(hidden_size * 3);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales,
                               W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward(is_training, time_steps, batch_size, input_size, hidden_size,
                            W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data(), x, h0,
                            quant_gru_scales, g_blas_handle, h, v);
        }
    } else {
        hasteGRUForward(is_training, time_steps, batch_size, input_size, hidden_size, W, R, bx, br,
                        x, h0, g_blas_handle, h, v);
    }
}

// =====================================================================
// 模板显式实例化（供 Python 绑定使用）
// =====================================================================

template void quantitativeWeight<int8_t>(const int input_size, const int hidden_size,
                                         const float *W, const float *R, const float *bx,
                                         const float *br,
                                         const GRUQuantitativeParameters &quant_parms,
                                         int8_t *W_quant, int8_t *R_quant, int32_t *bx_quant,
                                         int32_t *br_quant);

template void quantitativeWeight<int16_t>(const int input_size, const int hidden_size,
                                          const float *W, const float *R, const float *bx,
                                          const float *br,
                                          const GRUQuantitativeParameters &quant_parms,
                                          int16_t *W_quant, int16_t *R_quant, int32_t *bx_quant,
                                          int32_t *br_quant);

template void quantGRUForward<int8_t>(bool is_training, const int time_steps, const int batch_size,
                                      const int input_size, const int hidden_size, const int8_t *W,
                                      const int8_t *R, const int32_t *bx, const int32_t *br,
                                      const float *x,
                                      const float *h0,  // 初始隐藏状态，可以为 nullptr
                                      const GRUQuantitativeParameters &quant_parms,
                                      const cublasHandle_t &g_blas_handle, float *h, float *v);

template void quantGRUForward<int16_t>(bool is_training, const int time_steps, const int batch_size,
                                       const int input_size, const int hidden_size,
                                       const int16_t *W, const int16_t *R, const int32_t *bx,
                                       const int32_t *br, const float *x, const float *h0,
                                       const GRUQuantitativeParameters &quant_parms,
                                       const cublasHandle_t &g_blas_handle, float *h, float *v);

// =====================================================================
// LUT 初始化实现
// =====================================================================

void initialize_quantization_lut(const GRUQuantitativeParameters &quant_params) {
    generate_piecewise_linear_lut(quant_params);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in initialize_quantization_lut: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in initialize_quantization_lut: ") +
                                 err_str);
    }
}
