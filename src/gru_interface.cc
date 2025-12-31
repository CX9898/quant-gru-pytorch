// =====================================================================
// GRU 接口层实现 (gru_interface.cpp)
// =====================================================================

#include "gru_interface.h"

#include <cuda_runtime.h>
#include <omp.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "histogram_collector.h"
#include "histogram_gpu.cuh"
#include "parallel_algorithm.h"
#include "pot_sqnr_calibrator.h"
#include "quantize_ops_helper.h"

// =====================================================================
// 量化参数计算
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
#ifdef DEBUG
        if (name) {
            printf(
                "[ensureMinRange] %s: range %.4f < %.4f, expanded [%.4f, %.4f] -> [%.4f, %.4f]\n",
                name, range, min_range_threshold, old_min, old_max, min_val, max_val);
        }
#endif
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
    // INT8 和 INT16 统一使用实际校准范围（LUT 表会自动使用相同的量化参数）
    // 注意：LUT 表使用 exp2_inv_*_out_ 和 zp_*_out_，与此处一致
    constexpr float MIN_ACTIVATION_RANGE = 0.5f;

    // z 门输出的量化 - sigmoid
    // 使用实际校准范围（INT8 和 INT16 统一使用实际范围）
    dispatchByBitWidth(bitwidth_config.z_out_, [&](auto tag) {
        using ZOutT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        float min_val = quant_ranges.min_z_out_;
        float max_val = quant_ranges.max_z_out_;
        ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, "z_out");
        calibrateQuantParams<float, ZOutT>(min_val, max_val, bitwidth_config.z_out_symmetric_,
                                           aligned_min, aligned_max, quant_params.exp2_inv_z_out_,
                                           quant_params.zp_z_out_, "scale_z_out");
    });

    // r 门输出的量化 - sigmoid
    // 使用实际校准范围（INT8 和 INT16 统一使用实际范围）
    dispatchByBitWidth(bitwidth_config.r_out_, [&](auto tag) {
        using ROutT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        float min_val = quant_ranges.min_r_out_;
        float max_val = quant_ranges.max_r_out_;
        ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, "r_out");
        calibrateQuantParams<float, ROutT>(min_val, max_val, bitwidth_config.r_out_symmetric_,
                                           aligned_min, aligned_max, quant_params.exp2_inv_r_out_,
                                           quant_params.zp_r_out_, "scale_r_out");
    });

    // g 门输出的量化 - tanh
    // 使用实际校准范围（INT8 和 INT16 统一使用实际范围）
    dispatchByBitWidth(bitwidth_config.g_out_, [&](auto tag) {
        using GOutT = typename decltype(tag)::type;
        float aligned_min, aligned_max;
        float min_val = quant_ranges.min_g_out_;
        float max_val = quant_ranges.max_g_out_;
        ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, "g_out");
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

    // 生成 LUT 并存储到参数中（避免全局 LUT 覆盖问题）
    generate_piecewise_linear_lut_to_params(quant_params);

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
// WeightT: 权重类型 (W, R 共享)
// ActivationT: 激活类型 (x, h 共享)
template <typename WeightT, typename ActivationT>
void quantGRUForward(bool is_training, const int time_steps, const int batch_size,
                     const int input_size, const int hidden_size, const WeightT *W,
                     const WeightT *R, const int32_t *bx, const int32_t *br, const float *x,
                     const float *h0, const GRUQuantitativeParameters &quant_parms,
                     const cublasHandle_t &g_blas_handle, float *h, float *v) {
    const std::size_t x_size = time_steps * batch_size * input_size;

    // 激活值使用 ActivationT 类型
    dev::vector<ActivationT> x_quant(x_size);
    dev::quantification(x, x_quant.data(), x_size, quant_parms.exp2_inv_x_, quant_parms.zp_x_);

    dev::vector<ActivationT> h_quant((time_steps + 1) * batch_size * hidden_size);
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

    // 使用独立的权重类型和激活类型
    // ForwardPassQuant<XT, HT, WT, RT>: XT=x类型, HT=h类型, WT=W类型, RT=R类型
    gru::ForwardPassQuant<ActivationT, ActivationT, WeightT, WeightT> forward =
        gru::ForwardPassQuant<ActivationT, ActivationT, WeightT, WeightT>(
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
// 支持的量化模式：
//   - W8A8:   权重 int8,  激活 int8  (默认)
//   - W8A16:  权重 int8,  激活 int16 (混合精度)
//   - W16A16: 权重 int16, 激活 int16
// calib_method: 校准方法，NONE 表示正常推理
void forwardInterface(bool is_training, bool is_quant, int time_steps, int batch_size,
                      int input_size, int hidden_size, const float *W, const float *R,
                      const float *bx, const float *br, const float *x, const float *h0,
                      const GRUQuantitativeParameters &quant_gru_scales,
                      const cublasHandle_t &g_blas_handle,
                      CalibrationMethod calib_method,
                      GRUHistogramCollectors *hist_collectors,
                      GRUQuantizationRanges *quant_ranges,
                      float *h, float *v) {
    // 如果需要校准，使用专门的校准前向传播
    if (calib_method != CalibrationMethod::NONE) {
        forwardWithCalibration(is_training, time_steps, batch_size, input_size, hidden_size,
                               W, R, bx, br, x, h0, g_blas_handle,
                               calib_method, hist_collectors, quant_ranges, h, v);
        return;
    }
    
    // 正常前向传播
    if (is_quant) {
        dev::vector<int32_t> bx_quant(hidden_size * 3);
        dev::vector<int32_t> br_quant(hidden_size * 3);

        const auto &config = quant_gru_scales.bitwidth_config_;

        // 一致性检查：W 和 R 必须相同位宽，x 和 h 必须相同位宽
        if (config.W_ != config.R_) {
            throw std::invalid_argument(
                "W_ and R_ must have the same bitwidth. "
                "Current: W_=" +
                std::to_string(static_cast<int>(config.W_)) +
                ", R_=" + std::to_string(static_cast<int>(config.R_)));
        }
        if (config.x_ != config.h_) {
            throw std::invalid_argument(
                "x_ and h_ must have the same bitwidth. "
                "Current: x_=" +
                std::to_string(static_cast<int>(config.x_)) +
                ", h_=" + std::to_string(static_cast<int>(config.h_)));
        }

        const bool weight_is_8bit = (config.W_ == QuantBitWidth::INT8);
        const bool weight_is_16bit = (config.W_ == QuantBitWidth::INT16);
        const bool activation_is_8bit = (config.x_ == QuantBitWidth::INT8);
        const bool activation_is_16bit = (config.x_ == QuantBitWidth::INT16);

        if (weight_is_16bit && activation_is_16bit) {
            // W16A16: 权重 int16, 激活 int16
            dev::vector<int16_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int16_t> R_quant(hidden_size * 3 * hidden_size);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales,
                               W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward<int16_t, int16_t>(is_training, time_steps, batch_size, input_size,
                                              hidden_size, W_quant.data(), R_quant.data(),
                                              bx_quant.data(), br_quant.data(), x, h0,
                                              quant_gru_scales, g_blas_handle, h, v);
        } else if (weight_is_8bit && activation_is_16bit) {
            // W8A16: 权重 int8, 激活 int16 (混合精度)
            dev::vector<int8_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int8_t> R_quant(hidden_size * 3 * hidden_size);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales,
                               W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward<int8_t, int16_t>(is_training, time_steps, batch_size, input_size,
                                             hidden_size, W_quant.data(), R_quant.data(),
                                             bx_quant.data(), br_quant.data(), x, h0,
                                             quant_gru_scales, g_blas_handle, h, v);
        } else if (weight_is_8bit && activation_is_8bit) {
            // W8A8: 权重 int8, 激活 int8 (默认)
            dev::vector<int8_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int8_t> R_quant(hidden_size * 3 * hidden_size);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales,
                               W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward<int8_t, int8_t>(is_training, time_steps, batch_size, input_size,
                                            hidden_size, W_quant.data(), R_quant.data(),
                                            bx_quant.data(), br_quant.data(), x, h0,
                                            quant_gru_scales, g_blas_handle, h, v);
        } else if (weight_is_16bit && activation_is_8bit) {
            // W16A8: 权重 int16, 激活 int8
            dev::vector<int16_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int16_t> R_quant(hidden_size * 3 * hidden_size);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales,
                               W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward<int16_t, int8_t>(is_training, time_steps, batch_size, input_size,
                                             hidden_size, W_quant.data(), R_quant.data(),
                                             bx_quant.data(), br_quant.data(), x, h0,
                                             quant_gru_scales, g_blas_handle, h, v);
        } else {
            // 不支持的位宽组合 - 生成详细错误信息
            auto bitwidthToString = [](QuantBitWidth bw) -> const char * {
                switch (bw) {
                    case QuantBitWidth::INT8:
                        return "INT8";
                    case QuantBitWidth::INT16:
                        return "INT16";
                    case QuantBitWidth::INT32:
                        return "INT32";
                    case QuantBitWidth::UINT8:
                        return "UINT8";
                    case QuantBitWidth::UINT16:
                        return "UINT16";
                    default:
                        return "UNKNOWN";
                }
            };
            std::string error_msg = "Unsupported quantization mode: W_=";
            error_msg += bitwidthToString(config.W_);
            error_msg += ", R_=";
            error_msg += bitwidthToString(config.R_);
            error_msg += ", x_=";
            error_msg += bitwidthToString(config.x_);
            error_msg += ", h_=";
            error_msg += bitwidthToString(config.h_);
            error_msg += ". Supported modes: W8A8, W8A16, W16A8, W16A16";
            throw std::invalid_argument(error_msg);
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

// quantGRUForward 显式实例化
// 支持的模式: W8A8, W8A16, W16A8, W16A16

// W8A8: 权重 int8, 激活 int8
template void quantGRUForward<int8_t, int8_t>(
    bool is_training, const int time_steps, const int batch_size, const int input_size,
    const int hidden_size, const int8_t *W, const int8_t *R, const int32_t *bx, const int32_t *br,
    const float *x, const float *h0, const GRUQuantitativeParameters &quant_parms,
    const cublasHandle_t &g_blas_handle, float *h, float *v);

// W8A16: 权重 int8, 激活 int16 (混合精度)
template void quantGRUForward<int8_t, int16_t>(
    bool is_training, const int time_steps, const int batch_size, const int input_size,
    const int hidden_size, const int8_t *W, const int8_t *R, const int32_t *bx, const int32_t *br,
    const float *x, const float *h0, const GRUQuantitativeParameters &quant_parms,
    const cublasHandle_t &g_blas_handle, float *h, float *v);

// W16A8: 权重 int16, 激活 int8
template void quantGRUForward<int16_t, int8_t>(
    bool is_training, const int time_steps, const int batch_size, const int input_size,
    const int hidden_size, const int16_t *W, const int16_t *R, const int32_t *bx, const int32_t *br,
    const float *x, const float *h0, const GRUQuantitativeParameters &quant_parms,
    const cublasHandle_t &g_blas_handle, float *h, float *v);

// W16A16: 权重 int16, 激活 int16
template void quantGRUForward<int16_t, int16_t>(
    bool is_training, const int time_steps, const int batch_size, const int input_size,
    const int hidden_size, const int16_t *W, const int16_t *R, const int32_t *bx, const int32_t *br,
    const float *x, const float *h0, const GRUQuantitativeParameters &quant_parms,
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

// =====================================================================
// AIMET 风格的真正直方图校准实现
// =====================================================================

// 辅助函数：从 GPU 数据收集直方图
template <typename T>
inline void collectHistogramFromDevice(HistogramCollector &collector, const T *data_dev,
                                       size_t size) {
    if (size == 0) return;

    // 拷贝到 host
    std::vector<T> data_host(size);
    cudaMemcpy(data_host.data(), data_dev, size * sizeof(T), cudaMemcpyDeviceToHost);

    // 转换为 float 并收集直方图
    std::vector<float> data_float(size);
    for (size_t i = 0; i < size; ++i) {
        data_float[i] = static_cast<float>(data_host[i]);
    }

    collector.collect(data_float.data(), data_float.size());
}

// 辅助函数：分时间步收集直方图（用于时序数据）
template <typename T>
inline void collectHistogramPerStep(HistogramCollector &collector, const T *data_dev, int steps,
                                    int step_size) {
    std::vector<T> data_host(steps * step_size);
    cudaMemcpy(data_host.data(), data_dev, steps * step_size * sizeof(T), cudaMemcpyDeviceToHost);

    // 分时间步收集（模拟多批次累积）
    std::vector<float> step_float(step_size);
    for (int t = 0; t < steps; ++t) {
        const T *step_data = data_host.data() + t * step_size;
        for (int i = 0; i < step_size; ++i) {
            step_float[i] = static_cast<float>(step_data[i]);
        }
        collector.collect(step_float.data(), step_float.size());
    }
}

// 辅助函数：per-channel 直方图收集
template <typename T>
inline void collectPerChannelHistograms(std::vector<HistogramCollector> &collectors,
                                        const T *data_dev, int input_size, int channel_size) {
    std::vector<T> data_host(input_size * channel_size);
    cudaMemcpy(data_host.data(), data_dev, input_size * channel_size * sizeof(T),
               cudaMemcpyDeviceToHost);

    // 为每个 channel 收集直方图
    for (int c = 0; c < channel_size; ++c) {
        std::vector<float> channel_data(input_size);
        for (int i = 0; i < input_size; ++i) {
            channel_data[i] = static_cast<float>(data_host[i * channel_size + c]);
        }
        collectors[c].collect(channel_data.data(), channel_data.size());
    }
}

// =====================================================================
// 公共直方图收集辅助函数（供 calibrateGruHistograms 和 forwardWithCalibration 复用）
// =====================================================================

void collectAllHistograms(
    GRUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *tmp_Wx, const float *tmp_Rh,
    const float *W, const float *R, const float *bx, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    // 预激活值（z_pre, r_pre, g_pre）- 可选，传 nullptr 则跳过
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size) {
    
    const int NH = batch_size * hidden_size;
    const int NI = batch_size * input_size;

    // 1. 收集输入 x 的直方图
    collectHistogramPerStep(hist_collectors.x_hist, x, time_steps, NI);

    // 2. 收集隐藏状态 h 的直方图（跳过初始状态）
    collectHistogramPerStep(hist_collectors.h_hist, h + NH, time_steps, NH);

    // 3. 收集 Wx 结果的直方图
    collectHistogramPerStep(hist_collectors.Wx_hist, tmp_Wx, time_steps, NH * 3);

    // 4. 收集 Rh 结果的直方图
    collectHistogramPerStep(hist_collectors.Rh_hist, tmp_Rh, time_steps, NH * 3);

    // 5. 收集权重的 per-channel 直方图（只在首次收集）
    if (!hist_collectors.W_hist[0].is_valid()) {
        collectPerChannelHistograms(hist_collectors.W_hist, W, input_size, hidden_size * 3);
        collectPerChannelHistograms(hist_collectors.R_hist, R, hidden_size, hidden_size * 3);
        collectPerChannelHistograms(hist_collectors.bx_hist, bx, 1, hidden_size * 3);
        collectPerChannelHistograms(hist_collectors.br_hist, br, 1, hidden_size * 3);
    }

    // 6. 从 v 中收集门的中间值直方图
    // v 布局: [T, B, H*4] = [z, r, g, Rh_add_br_g]
    std::vector<float> v_host = d2h(v, time_steps * batch_size * hidden_size * 4);
    std::vector<float> h_host = d2h(h, (time_steps + 1) * batch_size * hidden_size);

    const size_t output_size = time_steps * batch_size * hidden_size;
    std::vector<float> z_out(output_size);
    std::vector<float> r_out(output_size);
    std::vector<float> g_out(output_size);
    std::vector<float> Rh_add_br_g(output_size);
    std::vector<float> rRh_g(output_size);
    std::vector<float> new_contrib(output_size);
    std::vector<float> old_contrib(output_size);

    // 解析 v 中的值
    for (int t = 0; t < time_steps; ++t) {
        for (int b = 0; b < batch_size; ++b) {
            const size_t v_base = t * batch_size * hidden_size * 4 + b * hidden_size * 4;
            const size_t out_base = t * batch_size * hidden_size + b * hidden_size;

            for (int hh = 0; hh < hidden_size; ++hh) {
                const float z_val = v_host[v_base + 0 * hidden_size + hh];
                const float r_val = v_host[v_base + 1 * hidden_size + hh];
                const float g_val = v_host[v_base + 2 * hidden_size + hh];
                const float Rh_add_br_val = v_host[v_base + 3 * hidden_size + hh];

                z_out[out_base + hh] = z_val;
                r_out[out_base + hh] = r_val;
                g_out[out_base + hh] = g_val;
                Rh_add_br_g[out_base + hh] = Rh_add_br_val;
                rRh_g[out_base + hh] = r_val * Rh_add_br_val;
                new_contrib[out_base + hh] = (1.0f - z_val) * g_val;

                // h_old 是上一个时间步的隐藏状态
                const size_t h_base = t * batch_size * hidden_size + b * hidden_size;
                old_contrib[out_base + hh] = z_val * h_host[h_base + hh];
            }
        }
    }

    // 分时间步收集直方图
    for (int t = 0; t < time_steps; ++t) {
        const float *z_step = z_out.data() + t * batch_size * hidden_size;
        const float *r_step = r_out.data() + t * batch_size * hidden_size;
        const float *g_step = g_out.data() + t * batch_size * hidden_size;
        const float *Rh_add_br_step = Rh_add_br_g.data() + t * batch_size * hidden_size;
        const float *rRh_step = rRh_g.data() + t * batch_size * hidden_size;
        const float *new_contrib_step = new_contrib.data() + t * batch_size * hidden_size;
        const float *old_contrib_step = old_contrib.data() + t * batch_size * hidden_size;

        hist_collectors.z_out_hist.collect(z_step, NH);
        hist_collectors.r_out_hist.collect(r_step, NH);
        hist_collectors.g_out_hist.collect(g_step, NH);
        hist_collectors.Rh_add_br_g_hist.collect(Rh_add_br_step, NH);
        hist_collectors.rRh_hist.collect(rRh_step, NH);
        hist_collectors.new_contrib_hist.collect(new_contrib_step, NH);
        hist_collectors.old_contrib_hist.collect(old_contrib_step, NH);
    }

    // 7. 收集 z_pre, r_pre, g_pre 的直方图（如果提供）
    if (pres_size > 0 && z_pres && r_pres && g_pres) {
        std::vector<float> z_pres_host(pres_size);
        std::vector<float> r_pres_host(pres_size);
        std::vector<float> g_pres_host(pres_size);

        cudaMemcpy(z_pres_host.data(), z_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(r_pres_host.data(), r_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(g_pres_host.data(), g_pres, pres_size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int t = 0; t < time_steps; ++t) {
            const float *z_pre_step = z_pres_host.data() + t * batch_size * hidden_size;
            const float *r_pre_step = r_pres_host.data() + t * batch_size * hidden_size;
            const float *g_pre_step = g_pres_host.data() + t * batch_size * hidden_size;

            hist_collectors.z_pre_hist.collect(z_pre_step, NH);
            hist_collectors.r_pre_hist.collect(r_pre_step, NH);
            hist_collectors.g_pre_hist.collect(g_pre_step, NH);
        }
    }
}

// =====================================================================
// GPU 加速直方图收集辅助函数
// =====================================================================

/**
 * @brief 使用 GPU 收集所有直方图（高性能版本）
 * 
 * 所有直方图计算都在 GPU 上完成，避免大量 GPU->CPU 数据传输
 * 使用 CUDA streams 并行收集多个直方图
 */
void collectAllHistogramsGPU(
    GRUGPUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *tmp_Wx, const float *tmp_Rh,
    const float *W, const float *R, const float *bx, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size,
    cudaStream_t stream = 0) {
    
    const size_t x_size = time_steps * batch_size * input_size;
    const size_t h_size = time_steps * batch_size * hidden_size;
    const size_t Wx_size = time_steps * batch_size * hidden_size * 3;
    const size_t Rh_size = time_steps * batch_size * hidden_size * 3;
    const float *h_skip_initial = h + batch_size * hidden_size;

    // 创建 streams（按需数量）
    constexpr int NUM_STREAMS = 8;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // 并行收集 x, h, Wx, Rh 的直方图
    hist_collectors.x_hist.collect(x, x_size, streams[0]);
    hist_collectors.h_hist.collect(h_skip_initial, h_size, streams[1]);
    hist_collectors.Wx_hist.collect(tmp_Wx, Wx_size, streams[2]);
    hist_collectors.Rh_hist.collect(tmp_Rh, Rh_size, streams[3]);

    // 并行收集 per-channel 直方图（使用零拷贝批量版本）
    if (!hist_collectors.W_batch.is_valid()) {
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.W_batch, W, input_size, streams[4]);
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.R_batch, R, hidden_size, streams[5]);
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.bx_batch, bx, 1, streams[6]);
        gpu_hist::collect_per_channel_histograms_batch(hist_collectors.br_batch, br, 1, streams[7]);
    }

    // 统一等待基础数据收集完成
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // 收集门值直方图（需要 h 数据完成后）
    gpu_hist::collect_gate_histograms(hist_collectors, v, h, time_steps, batch_size, hidden_size,
                                       stream);

    // 并行收集 z_pre, r_pre, g_pre（如果提供）
    if (pres_size > 0 && z_pres && r_pres && g_pres) {
        hist_collectors.z_pre_hist.collect(z_pres, pres_size, streams[0]);
        hist_collectors.r_pre_hist.collect(r_pres, pres_size, streams[1]);
        hist_collectors.g_pre_hist.collect(g_pres, pres_size, streams[2]);
        
        for (int i = 0; i < 3; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
    }

    // 清理 streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

/**
 * @brief 将 GPU 直方图收集器转换为 CPU 版本
 * 
 * 用于与现有的 calculateGRUQuantitativeParametersFromHistograms 兼容
 */
GRUHistogramCollectors convertGPUHistogramsToCPU(const GRUGPUHistogramCollectors &gpu_collectors) {
    GRUHistogramCollectors cpu_collectors(gpu_collectors.hidden_, gpu_collectors.num_bins_);

    // 转换标量直方图
    auto convert_collector = [](HistogramCollector &dst, const GPUHistogramCollector &src) {
        if (src.is_valid()) {
            dst.histogram() = gpu_histogram_to_cpu(src.histogram());
        }
    };

    convert_collector(cpu_collectors.x_hist, gpu_collectors.x_hist);
    convert_collector(cpu_collectors.h_hist, gpu_collectors.h_hist);
    convert_collector(cpu_collectors.Wx_hist, gpu_collectors.Wx_hist);
    convert_collector(cpu_collectors.Rh_hist, gpu_collectors.Rh_hist);
    convert_collector(cpu_collectors.z_pre_hist, gpu_collectors.z_pre_hist);
    convert_collector(cpu_collectors.r_pre_hist, gpu_collectors.r_pre_hist);
    convert_collector(cpu_collectors.g_pre_hist, gpu_collectors.g_pre_hist);
    convert_collector(cpu_collectors.z_out_hist, gpu_collectors.z_out_hist);
    convert_collector(cpu_collectors.r_out_hist, gpu_collectors.r_out_hist);
    convert_collector(cpu_collectors.g_out_hist, gpu_collectors.g_out_hist);
    convert_collector(cpu_collectors.Rh_add_br_g_hist, gpu_collectors.Rh_add_br_g_hist);
    convert_collector(cpu_collectors.rRh_hist, gpu_collectors.rRh_hist);
    convert_collector(cpu_collectors.new_contrib_hist, gpu_collectors.new_contrib_hist);
    convert_collector(cpu_collectors.old_contrib_hist, gpu_collectors.old_contrib_hist);

    // 转换 per-channel 直方图（从批量结构）
    auto convert_batch = [](std::vector<HistogramCollector> &dst, 
                            const PerChannelHistogramBatch &src) {
        if (!src.is_valid()) return;
        
        // 一次性读取所有 counts 到 CPU
        std::vector<float> all_counts = src.all_counts_to_host();
        
        for (int c = 0; c < src.channel_size; ++c) {
            Histogram& hist = dst[c].histogram();
            hist.num_bins = src.num_bins;
            hist.min_val = src.mins[c];
            hist.max_val = src.maxs[c];
            hist.total_count = src.per_channel_count;
            hist.counts.assign(all_counts.begin() + c * src.num_bins,
                              all_counts.begin() + (c + 1) * src.num_bins);
        }
    };

    convert_batch(cpu_collectors.W_hist, gpu_collectors.W_batch);
    convert_batch(cpu_collectors.R_hist, gpu_collectors.R_batch);
    convert_batch(cpu_collectors.bx_hist, gpu_collectors.bx_batch);
    convert_batch(cpu_collectors.br_hist, gpu_collectors.br_batch);

    return cpu_collectors;
}

// =====================================================================
// 带校准的统一前向传播实现
// =====================================================================

void forwardWithCalibration(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bx, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    CalibrationMethod calib_method,
    GRUHistogramCollectors *hist_collectors,
    GRUQuantizationRanges *quant_ranges,
    float *h, float *v) {
    
    if (calib_method == CalibrationMethod::NONE) {
        throw std::invalid_argument("forwardWithCalibration called with NONE calibration method");
    }

    // 分配临时缓冲区
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);

    // 创建 ForwardPass 对象
    gru::ForwardPass<float> forward =
        gru::ForwardPass<float>(is_training, batch_size, input_size, hidden_size, g_blas_handle);

    // 根据校准方法设置校准模式
    if (calib_method == CalibrationMethod::MINMAX) {
        // MinMax 模式：设置校准模式，在 Run 过程中收集 min/max
        if (!quant_ranges) {
            throw std::invalid_argument("quant_ranges is required for MINMAX calibration");
        }
        if (quant_ranges->hidden_ != hidden_size) {
            quant_ranges->reset(hidden_size);
        }
        forward.setCalibrationMode(true, *quant_ranges);
    } else {
        // SQNR/Percentile 模式：需要直方图收集器
        if (!hist_collectors) {
            throw std::invalid_argument("hist_collectors is required for SQNR/Percentile calibration");
        }
        if (hist_collectors->hidden_ != hidden_size) {
            hist_collectors->reset(hidden_size);
        }
        // 直方图模式也需要临时的 quant_ranges 用于前向传播
        GRUQuantizationRanges temp_ranges(hidden_size);
        forward.setCalibrationMode(true, temp_ranges);
    }

    // 执行前向传播，收集中间结果
    forward.Run(time_steps, W, R, bx, br, x, h, v, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, h0);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in forwardWithCalibration: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in forwardWithCalibration: ") + err_str);
    }

    // MinMax 模式：从 ForwardPass 获取收集的范围
    if (calib_method == CalibrationMethod::MINMAX) {
        *quant_ranges = forward.getGRUQuantizationRanges();
        return;
    }

    // SQNR/Percentile 模式：使用公共辅助函数收集直方图
    collectAllHistograms(*hist_collectors, x, h, v,
                         tmp_Wx_dev.data(), tmp_Rh_dev.data(),
                         W, R, bx, br,
                         time_steps, batch_size, input_size, hidden_size,
                         forward.getZPres(), forward.getRPres(), forward.getGPres(),
                         forward.getPresSize());
}

// =====================================================================
// GPU 加速版本：带校准的前向传播
// =====================================================================

void forwardWithCalibrationGPU(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bx, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    CalibrationMethod calib_method,
    GRUGPUHistogramCollectors *gpu_hist_collectors,
    GRUQuantizationRanges *quant_ranges,
    float *h, float *v) {
    
    if (calib_method == CalibrationMethod::NONE) {
        throw std::invalid_argument("forwardWithCalibrationGPU called with NONE calibration method");
    }

    // 分配临时缓冲区
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);

    // 创建 ForwardPass 对象
    gru::ForwardPass<float> forward =
        gru::ForwardPass<float>(is_training, batch_size, input_size, hidden_size, g_blas_handle);

    // 根据校准方法设置校准模式
    if (calib_method == CalibrationMethod::MINMAX) {
        // MinMax 模式：设置校准模式，在 Run 过程中收集 min/max
        if (!quant_ranges) {
            throw std::invalid_argument("quant_ranges is required for MINMAX calibration");
        }
        if (quant_ranges->hidden_ != hidden_size) {
            quant_ranges->reset(hidden_size);
        }
        forward.setCalibrationMode(true, *quant_ranges);
    } else {
        // SQNR/Percentile 模式：需要 GPU 直方图收集器
        if (!gpu_hist_collectors) {
            throw std::invalid_argument("gpu_hist_collectors is required for SQNR/Percentile calibration");
        }
        if (gpu_hist_collectors->hidden_ != hidden_size) {
            gpu_hist_collectors->reset(hidden_size);
        }
        // 直方图模式也需要临时的 quant_ranges 用于前向传播
        GRUQuantizationRanges temp_ranges(hidden_size);
        forward.setCalibrationMode(true, temp_ranges);
    }

    // 执行前向传播，收集中间结果
    forward.Run(time_steps, W, R, bx, br, x, h, v, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, h0);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in forwardWithCalibrationGPU: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in forwardWithCalibrationGPU: ") + err_str);
    }

    // MinMax 模式：从 ForwardPass 获取收集的范围
    if (calib_method == CalibrationMethod::MINMAX) {
        *quant_ranges = forward.getGRUQuantizationRanges();
        return;
    }

    // SQNR/Percentile 模式：使用 GPU 加速直方图收集
    collectAllHistogramsGPU(*gpu_hist_collectors, x, h, v,
                            tmp_Wx_dev.data(), tmp_Rh_dev.data(),
                            W, R, bx, br,
                            time_steps, batch_size, input_size, hidden_size,
                            forward.getZPres(), forward.getRPres(), forward.getGPres(),
                            forward.getPresSize());
}

GRUQuantitativeParameters calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors &hist_collectors, const OperatorQuantConfig &bitwidth_config,
    bool verbose, bool use_percentile, float percentile_value) {
    GRUQuantitativeParameters quant_params;
    quant_params.hidden_ = hist_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;

    // 根据 use_percentile 选择校准方案
    CalibrationScheme scheme = use_percentile ? CalibrationScheme::PERCENTILE : CalibrationScheme::SQNR;
    const int channel_size = hist_collectors.hidden_ * 3;
    
    // 预分配 per-channel 向量
    quant_params.exp2_inv_W_.resize(channel_size);
    quant_params.exp2_inv_R_.resize(channel_size);
    quant_params.exp2_inv_bx_.resize(channel_size);
    quant_params.exp2_inv_br_.resize(channel_size);

    // ========== OpenMP 并行化 per-channel 计算 ==========
    // 使用 4 个线程分别处理 W, R, bx, br（每个约 768 channels）
    // 标量参数由线程 3 顺便处理（开销小，不影响负载均衡）
    #pragma omp parallel num_threads(4) shared(quant_params, hist_collectors, bitwidth_config, scheme, percentile_value, channel_size)
    {
        int tid = omp_get_thread_num();
        int32_t zp_tmp;
        
        if (tid == 0) {
            // 线程 0: 处理 W（per-channel）
            dispatchByBitWidth(bitwidth_config.W_, [&](auto tag) {
                using WT = typename decltype(tag)::type;
                for (int c = 0; c < channel_size; ++c) {
                    if (!hist_collectors.W_hist[c].is_valid()) {
                        throw std::runtime_error("W_hist[" + std::to_string(c) + "] is invalid.");
                    }
                    calibrateQuantParamsFromHistogram<WT>(hist_collectors.W_hist[c].histogram(),
                                                          bitwidth_config.W_symmetric_,
                                                          quant_params.exp2_inv_W_[c], zp_tmp, nullptr,
                                                          scheme, percentile_value);
                }
            });
        } else if (tid == 1) {
            // 线程 1: 处理 R（per-channel）
            dispatchByBitWidth(bitwidth_config.R_, [&](auto tag) {
                using RT = typename decltype(tag)::type;
                for (int c = 0; c < channel_size; ++c) {
                    if (!hist_collectors.R_hist[c].is_valid()) {
                        throw std::runtime_error("R_hist[" + std::to_string(c) + "] is invalid.");
                    }
                    calibrateQuantParamsFromHistogram<RT>(hist_collectors.R_hist[c].histogram(),
                                                          bitwidth_config.R_symmetric_,
                                                          quant_params.exp2_inv_R_[c], zp_tmp, nullptr,
                                                          scheme, percentile_value);
                }
            });
        } else if (tid == 2) {
            // 线程 2: 处理 bx（per-channel）
            dispatchByBitWidth(bitwidth_config.bx_, [&](auto tag) {
                using BxT = typename decltype(tag)::type;
                for (int c = 0; c < channel_size; ++c) {
                    if (!hist_collectors.bx_hist[c].is_valid()) {
                        throw std::runtime_error("bx_hist[" + std::to_string(c) + "] is invalid.");
                    }
                    calibrateQuantParamsFromHistogram<BxT>(
                        hist_collectors.bx_hist[c].histogram(), bitwidth_config.bx_symmetric_,
                        quant_params.exp2_inv_bx_[c], zp_tmp, nullptr,
                        scheme, percentile_value);
                }
            });
        } else {
            // 线程 3: 处理 br（per-channel）+ 所有标量参数
            dispatchByBitWidth(bitwidth_config.br_, [&](auto tag) {
                using BrT = typename decltype(tag)::type;
                for (int c = 0; c < channel_size; ++c) {
                    if (!hist_collectors.br_hist[c].is_valid()) {
                        throw std::runtime_error("br_hist[" + std::to_string(c) + "] is invalid.");
                    }
                    calibrateQuantParamsFromHistogram<BrT>(
                        hist_collectors.br_hist[c].histogram(), bitwidth_config.br_symmetric_,
                        quant_params.exp2_inv_br_[c], zp_tmp, nullptr,
                        scheme, percentile_value);
                }
            });
            
            // 标量参数（计算量小，由最后一个线程顺便处理）
            // x
            dispatchByBitWidth(bitwidth_config.x_, [&](auto tag) {
                using XT = typename decltype(tag)::type;
                if (!hist_collectors.x_hist.is_valid()) {
                    throw std::runtime_error("x_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<XT>(
                    hist_collectors.x_hist.histogram(), bitwidth_config.x_symmetric_,
                    quant_params.exp2_inv_x_, quant_params.zp_x_, verbose ? "scale_x" : nullptr,
                    scheme, percentile_value);
            });
            
            // h
            dispatchByBitWidth(bitwidth_config.h_, [&](auto tag) {
                using HT = typename decltype(tag)::type;
                if (!hist_collectors.h_hist.is_valid()) {
                    throw std::runtime_error("h_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<HT>(
                    hist_collectors.h_hist.histogram(), bitwidth_config.h_symmetric_,
                    quant_params.exp2_inv_h_, quant_params.zp_h_, verbose ? "scale_h" : nullptr,
                    scheme, percentile_value);
            });
            
            // Wx
            dispatchByBitWidth(bitwidth_config.Wx_, [&](auto tag) {
                using WxT = typename decltype(tag)::type;
                if (!hist_collectors.Wx_hist.is_valid()) {
                    throw std::runtime_error("Wx_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<WxT>(
                    hist_collectors.Wx_hist.histogram(), bitwidth_config.Wx_symmetric_,
                    quant_params.exp2_inv_Wx_, quant_params.zp_Wx_, verbose ? "scale_Wx" : nullptr,
                    scheme, percentile_value);
            });
            
            // Rh
            dispatchByBitWidth(bitwidth_config.Rh_, [&](auto tag) {
                using RhT = typename decltype(tag)::type;
                if (!hist_collectors.Rh_hist.is_valid()) {
                    throw std::runtime_error("Rh_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<RhT>(
                    hist_collectors.Rh_hist.histogram(), bitwidth_config.Rh_symmetric_,
                    quant_params.exp2_inv_Rh_, quant_params.zp_Rh_, verbose ? "scale_Rh" : nullptr,
                    scheme, percentile_value);
            });
            
            // z_pre
            dispatchByBitWidth(bitwidth_config.z_pre_, [&](auto tag) {
                using ZPreT = typename decltype(tag)::type;
                if (!hist_collectors.z_pre_hist.is_valid()) {
                    throw std::runtime_error("z_pre_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<ZPreT>(
                    hist_collectors.z_pre_hist.histogram(), bitwidth_config.z_pre_symmetric_,
                    quant_params.exp2_inv_z_pre_, quant_params.zp_z_pre_,
                    verbose ? "scale_z_pre" : nullptr,
                    scheme, percentile_value);
            });
            
            // r_pre
            dispatchByBitWidth(bitwidth_config.r_pre_, [&](auto tag) {
                using RPreT = typename decltype(tag)::type;
                if (!hist_collectors.r_pre_hist.is_valid()) {
                    throw std::runtime_error("r_pre_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<RPreT>(
                    hist_collectors.r_pre_hist.histogram(), bitwidth_config.r_pre_symmetric_,
                    quant_params.exp2_inv_r_pre_, quant_params.zp_r_pre_,
                    verbose ? "scale_r_pre" : nullptr,
                    scheme, percentile_value);
            });
            
            // g_pre
            dispatchByBitWidth(bitwidth_config.g_pre_, [&](auto tag) {
                using GPreT = typename decltype(tag)::type;
                if (!hist_collectors.g_pre_hist.is_valid()) {
                    throw std::runtime_error("g_pre_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<GPreT>(
                    hist_collectors.g_pre_hist.histogram(), bitwidth_config.g_pre_symmetric_,
                    quant_params.exp2_inv_g_pre_, quant_params.zp_g_pre_,
                    verbose ? "scale_g_pre" : nullptr,
                    scheme, percentile_value);
            });
            
            // z_out
            dispatchByBitWidth(bitwidth_config.z_out_, [&](auto tag) {
                using ZOutT = typename decltype(tag)::type;
                if (!hist_collectors.z_out_hist.is_valid()) {
                    throw std::runtime_error("z_out_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<ZOutT>(
                    hist_collectors.z_out_hist.histogram(), bitwidth_config.z_out_symmetric_,
                    quant_params.exp2_inv_z_out_, quant_params.zp_z_out_,
                    verbose ? "scale_z_out" : nullptr,
                    scheme, percentile_value);
            });
            
            // r_out
            dispatchByBitWidth(bitwidth_config.r_out_, [&](auto tag) {
                using ROutT = typename decltype(tag)::type;
                if (!hist_collectors.r_out_hist.is_valid()) {
                    throw std::runtime_error("r_out_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<ROutT>(
                    hist_collectors.r_out_hist.histogram(), bitwidth_config.r_out_symmetric_,
                    quant_params.exp2_inv_r_out_, quant_params.zp_r_out_,
                    verbose ? "scale_r_out" : nullptr,
                    scheme, percentile_value);
            });
            
            // g_out
            dispatchByBitWidth(bitwidth_config.g_out_, [&](auto tag) {
                using GOutT = typename decltype(tag)::type;
                if (!hist_collectors.g_out_hist.is_valid()) {
                    throw std::runtime_error("g_out_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<GOutT>(
                    hist_collectors.g_out_hist.histogram(), bitwidth_config.g_out_symmetric_,
                    quant_params.exp2_inv_g_out_, quant_params.zp_g_out_,
                    verbose ? "scale_g_out" : nullptr,
                    scheme, percentile_value);
            });
            
            // Rh_add_br
            dispatchByBitWidth(bitwidth_config.Rh_add_br_, [&](auto tag) {
                using RhAddBrT = typename decltype(tag)::type;
                if (!hist_collectors.Rh_add_br_g_hist.is_valid()) {
                    throw std::runtime_error("Rh_add_br_g_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<RhAddBrT>(
                    hist_collectors.Rh_add_br_g_hist.histogram(), bitwidth_config.Rh_add_br_symmetric_,
                    quant_params.exp2_inv_Rh_add_br_, quant_params.zp_Rh_add_br_,
                    verbose ? "scale_Rh_add_br" : nullptr,
                    scheme, percentile_value);
            });
            
            // rRh
            dispatchByBitWidth(bitwidth_config.rRh_, [&](auto tag) {
                using rRhT = typename decltype(tag)::type;
                if (!hist_collectors.rRh_hist.is_valid()) {
                    throw std::runtime_error("rRh_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<rRhT>(
                    hist_collectors.rRh_hist.histogram(), bitwidth_config.rRh_symmetric_,
                    quant_params.exp2_inv_rRh_, quant_params.zp_rRh_, verbose ? "scale_rRh" : nullptr,
                    scheme, percentile_value);
            });
            
            // new_contrib
            dispatchByBitWidth(bitwidth_config.new_contrib_, [&](auto tag) {
                using NewContribT = typename decltype(tag)::type;
                if (!hist_collectors.new_contrib_hist.is_valid()) {
                    throw std::runtime_error("new_contrib_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<NewContribT>(
                    hist_collectors.new_contrib_hist.histogram(),
                    bitwidth_config.new_contrib_symmetric_, quant_params.exp2_inv_new_contrib_,
                    quant_params.zp_new_contrib_, verbose ? "scale_new_contrib" : nullptr,
                    scheme, percentile_value);
            });
            
            // old_contrib
            dispatchByBitWidth(bitwidth_config.old_contrib_, [&](auto tag) {
                using OldContribT = typename decltype(tag)::type;
                if (!hist_collectors.old_contrib_hist.is_valid()) {
                    throw std::runtime_error("old_contrib_hist is invalid.");
                }
                calibrateQuantParamsFromHistogram<OldContribT>(
                    hist_collectors.old_contrib_hist.histogram(),
                    bitwidth_config.old_contrib_symmetric_, quant_params.exp2_inv_old_contrib_,
                    quant_params.zp_old_contrib_, verbose ? "scale_old_contrib" : nullptr,
                    scheme, percentile_value);
            });
        }
    }  // end omp parallel

    // 生成 LUT 并存储到参数中（避免全局 LUT 覆盖问题）
    generate_piecewise_linear_lut_to_params(quant_params);

    return quant_params;
}

/**
 * @brief 从 GPU 直方图收集器计算量化参数（GPU 加速 SQNR）
 *
 * 直接使用 GPU 上的直方图数据计算 SQNR，避免 GPU→CPU 传输
 */
GRUQuantitativeParameters calculateGRUQuantitativeParametersFromGPUHistograms(
    GRUGPUHistogramCollectors &gpu_collectors, const OperatorQuantConfig &bitwidth_config,
    bool verbose) {
    
    GRUQuantitativeParameters quant_params;
    quant_params.hidden_ = gpu_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;
    
    const int channel_size = gpu_collectors.hidden_ * 3;
    quant_params.exp2_inv_W_.resize(channel_size);
    quant_params.exp2_inv_R_.resize(channel_size);
    quant_params.exp2_inv_bx_.resize(channel_size);
    quant_params.exp2_inv_br_.resize(channel_size);
    
    // Helper: 计算单个标量直方图的 SQNR 参数
    auto compute_scalar_sqnr = [&](const GPUHistogramCollector& collector, 
                                    bool is_symmetric, QuantBitWidth quant_bw,
                                    int8_t& out_exp2_inv, int32_t& out_zp,
                                    const char* name) {
        int quant_bits = std::abs(static_cast<int>(quant_bw));
        bool is_unsigned = static_cast<int>(quant_bw) > 0;  // UINT 类型是正数
        if (!collector.is_valid()) {
            throw std::runtime_error(std::string("GPU histogram ") + (name ? name : "unknown") + " is invalid");
        }
        const auto& hist = collector.histogram();
        gpu_hist::compute_sqnr_params_gpu(
            hist.counts.data(), hist.min_val, hist.max_val,
            hist.num_bins, hist.total_count,
            is_symmetric, quant_bits, is_unsigned, out_exp2_inv, out_zp);
        
        if (verbose && name) {
            printf("[GPU-SQNR][%s] range=[%.4f,%.4f] exp2_inv=%d zp=%d\n",
                   name, hist.min_val, hist.max_val, out_exp2_inv, out_zp);
        }
    };
    
    // 标量直方图
    compute_scalar_sqnr(gpu_collectors.x_hist, bitwidth_config.x_symmetric_, 
                        bitwidth_config.x_, quant_params.exp2_inv_x_, quant_params.zp_x_, "x");
    compute_scalar_sqnr(gpu_collectors.h_hist, bitwidth_config.h_symmetric_,
                        bitwidth_config.h_, quant_params.exp2_inv_h_, quant_params.zp_h_, "h");
    compute_scalar_sqnr(gpu_collectors.Wx_hist, bitwidth_config.Wx_symmetric_,
                        bitwidth_config.Wx_, quant_params.exp2_inv_Wx_, quant_params.zp_Wx_, "Wx");
    compute_scalar_sqnr(gpu_collectors.Rh_hist, bitwidth_config.Rh_symmetric_,
                        bitwidth_config.Rh_, quant_params.exp2_inv_Rh_, quant_params.zp_Rh_, "Rh");
    compute_scalar_sqnr(gpu_collectors.z_pre_hist, bitwidth_config.z_pre_symmetric_,
                        bitwidth_config.z_pre_, quant_params.exp2_inv_z_pre_, quant_params.zp_z_pre_, "z_pre");
    compute_scalar_sqnr(gpu_collectors.r_pre_hist, bitwidth_config.r_pre_symmetric_,
                        bitwidth_config.r_pre_, quant_params.exp2_inv_r_pre_, quant_params.zp_r_pre_, "r_pre");
    compute_scalar_sqnr(gpu_collectors.g_pre_hist, bitwidth_config.g_pre_symmetric_,
                        bitwidth_config.g_pre_, quant_params.exp2_inv_g_pre_, quant_params.zp_g_pre_, "g_pre");
    compute_scalar_sqnr(gpu_collectors.z_out_hist, bitwidth_config.z_out_symmetric_,
                        bitwidth_config.z_out_, quant_params.exp2_inv_z_out_, quant_params.zp_z_out_, "z_out");
    compute_scalar_sqnr(gpu_collectors.r_out_hist, bitwidth_config.r_out_symmetric_,
                        bitwidth_config.r_out_, quant_params.exp2_inv_r_out_, quant_params.zp_r_out_, "r_out");
    compute_scalar_sqnr(gpu_collectors.g_out_hist, bitwidth_config.g_out_symmetric_,
                        bitwidth_config.g_out_, quant_params.exp2_inv_g_out_, quant_params.zp_g_out_, "g_out");
    compute_scalar_sqnr(gpu_collectors.Rh_add_br_g_hist, bitwidth_config.Rh_add_br_symmetric_,
                        bitwidth_config.Rh_add_br_, quant_params.exp2_inv_Rh_add_br_, 
                        quant_params.zp_Rh_add_br_, "Rh_add_br");
    compute_scalar_sqnr(gpu_collectors.rRh_hist, bitwidth_config.rRh_symmetric_,
                        bitwidth_config.rRh_, quant_params.exp2_inv_rRh_, quant_params.zp_rRh_, "rRh");
    compute_scalar_sqnr(gpu_collectors.new_contrib_hist, bitwidth_config.new_contrib_symmetric_,
                        bitwidth_config.new_contrib_, quant_params.exp2_inv_new_contrib_,
                        quant_params.zp_new_contrib_, "new_contrib");
    compute_scalar_sqnr(gpu_collectors.old_contrib_hist, bitwidth_config.old_contrib_symmetric_,
                        bitwidth_config.old_contrib_, quant_params.exp2_inv_old_contrib_,
                        quant_params.zp_old_contrib_, "old_contrib");
    
    // Per-channel 直方图（使用批量 GPU SQNR）
    auto getBits = [](QuantBitWidth bw) { return std::abs(static_cast<int>(bw)); };
    
    if (gpu_collectors.W_batch.is_valid()) {
        gpu_hist::compute_sqnr_per_channel_gpu(
            gpu_collectors.W_batch, bitwidth_config.W_symmetric_, 
            getBits(bitwidth_config.W_), quant_params.exp2_inv_W_);
    }
    if (gpu_collectors.R_batch.is_valid()) {
        gpu_hist::compute_sqnr_per_channel_gpu(
            gpu_collectors.R_batch, bitwidth_config.R_symmetric_,
            getBits(bitwidth_config.R_), quant_params.exp2_inv_R_);
    }
    if (gpu_collectors.bx_batch.is_valid()) {
        gpu_hist::compute_sqnr_per_channel_gpu(
            gpu_collectors.bx_batch, bitwidth_config.bx_symmetric_,
            getBits(bitwidth_config.bx_), quant_params.exp2_inv_bx_);
    }
    if (gpu_collectors.br_batch.is_valid()) {
        gpu_hist::compute_sqnr_per_channel_gpu(
            gpu_collectors.br_batch, bitwidth_config.br_symmetric_,
            getBits(bitwidth_config.br_), quant_params.exp2_inv_br_);
    }
    
    // 生成 LUT 并存储到参数中（避免全局 LUT 覆盖问题）
    generate_piecewise_linear_lut_to_params(quant_params);
    
    return quant_params;
}

// 注意：PERCENTILE 校准使用 CPU OpenMP 实现（比 GPU 更快）
// 使用 calculateGRUQuantitativeParametersFromHistograms() 并设置 use_percentile=true
