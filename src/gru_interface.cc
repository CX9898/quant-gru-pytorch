// =====================================================================
// GRU 接口层实现 (gru_interface.cpp)
// =====================================================================

#include "gru_interface.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "histogram_collector.h"
#include "parallel_algorithm.h"
#include "pot_sqnr_calibrator.h"
#include "quantize_ops_helper.h"

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
void forwardInterface(bool is_training, bool is_quant, int time_steps, int batch_size,
                      int input_size, int hidden_size, const float *W, const float *R,
                      const float *bx, const float *br, const float *x, const float *h0,
                      const GRUQuantitativeParameters &quant_gru_scales,
                      const cublasHandle_t &g_blas_handle, float *h, float *v) {
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

void calibrateGruHistograms(int time_steps, int batch_size, int input_size, int hidden_size,
                            const float *W, const float *R, const float *bx, const float *br,
                            const float *x, const cublasHandle_t &g_blas_handle,
                            GRUHistogramCollectors &hist_collectors) {
    // 分配临时缓冲区
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    // 初始化直方图收集器（如果需要）
    if (hist_collectors.hidden_ != hidden_size) {
        hist_collectors.reset(hidden_size);
    }

    // 创建量化范围用于前向传播（用于获取中间值）
    GRUQuantizationRanges quant_ranges(hidden_size);

    gru::ForwardPass<float> forward =
        gru::ForwardPass<float>(true,  // training mode to get v
                                batch_size, input_size, hidden_size, g_blas_handle);

    forward.setCalibrationMode(true, quant_ranges);

    forward.Run(time_steps, W, R, bx, br, x, h_dev.data(), v_dev.data(), tmp_Wx_dev.data(),
                tmp_Rh_dev.data(), 0.0f, nullptr);

    // 同步所有 CUDA 操作
    cudaDeviceSynchronize();

    const int NH = batch_size * hidden_size;
    const int NI = batch_size * input_size;

    // 1. 收集输入 x 的直方图
    collectHistogramPerStep(hist_collectors.x_hist, x, time_steps, NI);

    // 2. 收集隐藏状态 h 的直方图（跳过初始状态）
    collectHistogramPerStep(hist_collectors.h_hist, h_dev.data() + NH, time_steps, NH);

    // 3. 收集 Wx 结果的直方图
    collectHistogramPerStep(hist_collectors.Wx_hist, tmp_Wx_dev.data(), time_steps, NH * 3);

    // 4. 收集 Rh 结果的直方图
    collectHistogramPerStep(hist_collectors.Rh_hist, tmp_Rh_dev.data(), time_steps, NH * 3);

    // 5. 收集权重的 per-channel 直方图
    collectPerChannelHistograms(hist_collectors.W_hist, W, input_size, hidden_size * 3);
    collectPerChannelHistograms(hist_collectors.R_hist, R, hidden_size, hidden_size * 3);
    collectPerChannelHistograms(hist_collectors.bx_hist, bx, 1, hidden_size * 3);
    collectPerChannelHistograms(hist_collectors.br_hist, br, 1, hidden_size * 3);

    // 6. 从 v 中收集门的中间值直方图
    // v 布局: [T, B, H*4] = [z, r, g, Rh_add_br_g]
    std::vector<float> v_host = d2h(v_dev.data(), time_steps * batch_size * hidden_size * 4);
    std::vector<float> h_host = d2h(h_dev.data(), (time_steps + 1) * batch_size * hidden_size);

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

            for (int h = 0; h < hidden_size; ++h) {
                const float z_val = v_host[v_base + 0 * hidden_size + h];
                const float r_val = v_host[v_base + 1 * hidden_size + h];
                const float g_val = v_host[v_base + 2 * hidden_size + h];
                const float Rh_add_br_val = v_host[v_base + 3 * hidden_size + h];

                z_out[out_base + h] = z_val;
                r_out[out_base + h] = r_val;
                g_out[out_base + h] = g_val;
                Rh_add_br_g[out_base + h] = Rh_add_br_val;
                rRh_g[out_base + h] = r_val * Rh_add_br_val;
                new_contrib[out_base + h] = (1.0f - z_val) * g_val;

                // h_old 是上一个时间步的隐藏状态
                const size_t h_base = t * batch_size * hidden_size + b * hidden_size;
                old_contrib[out_base + h] = z_val * h_host[h_base + h];
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

        hist_collectors.z_out_hist.collect(z_step, batch_size * hidden_size);
        hist_collectors.r_out_hist.collect(r_step, batch_size * hidden_size);
        hist_collectors.g_out_hist.collect(g_step, batch_size * hidden_size);
        hist_collectors.Rh_add_br_g_hist.collect(Rh_add_br_step, batch_size * hidden_size);
        hist_collectors.rRh_hist.collect(rRh_step, batch_size * hidden_size);
        hist_collectors.new_contrib_hist.collect(new_contrib_step, batch_size * hidden_size);
        hist_collectors.old_contrib_hist.collect(old_contrib_step, batch_size * hidden_size);
    }

    // 7. 收集 z_pre, r_pre, g_pre 的直方图
    // 从 ForwardPass 中获取预激活值
    if (forward.getPresSize() > 0) {
        std::vector<float> z_pres_host(forward.getPresSize());
        std::vector<float> r_pres_host(forward.getPresSize());
        std::vector<float> g_pres_host(forward.getPresSize());

        cudaMemcpy(z_pres_host.data(), forward.getZPres(), forward.getPresSize() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(r_pres_host.data(), forward.getRPres(), forward.getPresSize() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(g_pres_host.data(), forward.getGPres(), forward.getPresSize() * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // 分时间步收集 z_pre, r_pre, g_pre 直方图
        for (int t = 0; t < time_steps; ++t) {
            const float *z_pre_step = z_pres_host.data() + t * batch_size * hidden_size;
            const float *r_pre_step = r_pres_host.data() + t * batch_size * hidden_size;
            const float *g_pre_step = g_pres_host.data() + t * batch_size * hidden_size;

            hist_collectors.z_pre_hist.collect(z_pre_step, batch_size * hidden_size);
            hist_collectors.r_pre_hist.collect(r_pre_step, batch_size * hidden_size);
            hist_collectors.g_pre_hist.collect(g_pre_step, batch_size * hidden_size);
        }
    }

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in calibrateGruHistograms: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in calibrateGruHistograms: ") + err_str);
    }
}

GRUQuantitativeParameters calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors &hist_collectors, const OperatorQuantConfig &bitwidth_config,
    bool verbose) {
    GRUQuantitativeParameters quant_params;
    quant_params.hidden_ = hist_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;

    // 输入 x 的量化 - 从直方图计算
    dispatchByBitWidth(bitwidth_config.x_, [&](auto tag) {
        using XT = typename decltype(tag)::type;
        if (!hist_collectors.x_hist.is_valid()) {
            throw std::runtime_error("x_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<XT>(
            hist_collectors.x_hist.histogram(), bitwidth_config.x_symmetric_,
            quant_params.exp2_inv_x_, quant_params.zp_x_, verbose ? "scale_x" : nullptr);
    });

    // 隐藏状态 h 的量化
    dispatchByBitWidth(bitwidth_config.h_, [&](auto tag) {
        using HT = typename decltype(tag)::type;
        if (!hist_collectors.h_hist.is_valid()) {
            throw std::runtime_error("h_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<HT>(
            hist_collectors.h_hist.histogram(), bitwidth_config.h_symmetric_,
            quant_params.exp2_inv_h_, quant_params.zp_h_, verbose ? "scale_h" : nullptr);
    });

    // 权重 W 的量化（per-channel）- 从直方图计算
    const int channel_size = hist_collectors.hidden_ * 3;
    quant_params.exp2_inv_W_.resize(channel_size);
    dispatchByBitWidth(bitwidth_config.W_, [&](auto tag) {
        using WT = typename decltype(tag)::type;
        for (int c = 0; c < channel_size; ++c) {
            int32_t zp_tmp;
            if (!hist_collectors.W_hist[c].is_valid()) {
                throw std::runtime_error("W_hist[" + std::to_string(c) + "] is invalid. Calibration data may be empty or corrupted.");
            }
            calibrateQuantParamsFromHistogram<WT>(hist_collectors.W_hist[c].histogram(),
                                                  bitwidth_config.W_symmetric_,
                                                  quant_params.exp2_inv_W_[c], zp_tmp, nullptr);
        }
    });

    // 权重 R 的量化（per-channel）
    quant_params.exp2_inv_R_.resize(channel_size);
    dispatchByBitWidth(bitwidth_config.R_, [&](auto tag) {
        using RT = typename decltype(tag)::type;
        for (int c = 0; c < channel_size; ++c) {
            int32_t zp_tmp;
            if (!hist_collectors.R_hist[c].is_valid()) {
                throw std::runtime_error("R_hist[" + std::to_string(c) + "] is invalid. Calibration data may be empty or corrupted.");
            }
            calibrateQuantParamsFromHistogram<RT>(hist_collectors.R_hist[c].histogram(),
                                                  bitwidth_config.R_symmetric_,
                                                  quant_params.exp2_inv_R_[c], zp_tmp, nullptr);
        }
    });

    // Wx 结果的量化 - 从直方图计算
    dispatchByBitWidth(bitwidth_config.Wx_, [&](auto tag) {
        using WxT = typename decltype(tag)::type;
        if (!hist_collectors.Wx_hist.is_valid()) {
            throw std::runtime_error("Wx_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<WxT>(
            hist_collectors.Wx_hist.histogram(), bitwidth_config.Wx_symmetric_,
            quant_params.exp2_inv_Wx_, quant_params.zp_Wx_, verbose ? "scale_Wx" : nullptr);
    });

    // Rh 结果的量化
    dispatchByBitWidth(bitwidth_config.Rh_, [&](auto tag) {
        using RhT = typename decltype(tag)::type;
        if (!hist_collectors.Rh_hist.is_valid()) {
            throw std::runtime_error("Rh_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<RhT>(
            hist_collectors.Rh_hist.histogram(), bitwidth_config.Rh_symmetric_,
            quant_params.exp2_inv_Rh_, quant_params.zp_Rh_, verbose ? "scale_Rh" : nullptr);
    });

    // 偏置 bx 的量化（per-channel）
    quant_params.exp2_inv_bx_.resize(channel_size);
    dispatchByBitWidth(bitwidth_config.bx_, [&](auto tag) {
        using BxT = typename decltype(tag)::type;
        for (int c = 0; c < channel_size; ++c) {
            int32_t zp_tmp;
            if (!hist_collectors.bx_hist[c].is_valid()) {
                throw std::runtime_error("bx_hist[" + std::to_string(c) + "] is invalid. Calibration data may be empty or corrupted.");
            }
            calibrateQuantParamsFromHistogram<BxT>(
                hist_collectors.bx_hist[c].histogram(), bitwidth_config.bx_symmetric_,
                quant_params.exp2_inv_bx_[c], zp_tmp, nullptr);
        }
    });

    // 偏置 br 的量化（per-channel）
    quant_params.exp2_inv_br_.resize(channel_size);
    dispatchByBitWidth(bitwidth_config.br_, [&](auto tag) {
        using BrT = typename decltype(tag)::type;
        for (int c = 0; c < channel_size; ++c) {
            int32_t zp_tmp;
            if (!hist_collectors.br_hist[c].is_valid()) {
                throw std::runtime_error("br_hist[" + std::to_string(c) + "] is invalid. Calibration data may be empty or corrupted.");
            }
            calibrateQuantParamsFromHistogram<BrT>(
                hist_collectors.br_hist[c].histogram(), bitwidth_config.br_symmetric_,
                quant_params.exp2_inv_br_[c], zp_tmp, nullptr);
        }
    });

    // z 门输入的量化 - 必须使用真实收集的 z_pre 直方图
    dispatchByBitWidth(bitwidth_config.z_pre_, [&](auto tag) {
        using ZPreT = typename decltype(tag)::type;
        if (!hist_collectors.z_pre_hist.is_valid()) {
            throw std::runtime_error("z_pre_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<ZPreT>(
            hist_collectors.z_pre_hist.histogram(), bitwidth_config.z_pre_symmetric_,
            quant_params.exp2_inv_z_pre_, quant_params.zp_z_pre_,
            verbose ? "scale_z_pre" : nullptr);
    });

    // r 门输入的量化 - 必须使用真实收集的 r_pre 直方图
    dispatchByBitWidth(bitwidth_config.r_pre_, [&](auto tag) {
        using RPreT = typename decltype(tag)::type;
        if (!hist_collectors.r_pre_hist.is_valid()) {
            throw std::runtime_error("r_pre_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<RPreT>(
            hist_collectors.r_pre_hist.histogram(), bitwidth_config.r_pre_symmetric_,
            quant_params.exp2_inv_r_pre_, quant_params.zp_r_pre_,
            verbose ? "scale_r_pre" : nullptr);
    });

    // g 门输入的量化 - 必须使用真实收集的 g_pre 直方图
    dispatchByBitWidth(bitwidth_config.g_pre_, [&](auto tag) {
        using GPreT = typename decltype(tag)::type;
        if (!hist_collectors.g_pre_hist.is_valid()) {
            throw std::runtime_error("g_pre_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<GPreT>(
            hist_collectors.g_pre_hist.histogram(), bitwidth_config.g_pre_symmetric_,
            quant_params.exp2_inv_g_pre_, quant_params.zp_g_pre_,
            verbose ? "scale_g_pre" : nullptr);
    });

    // 激活函数输出的校准 - 使用 AIMET SQNR 方法

    // z 门输出 - sigmoid
    dispatchByBitWidth(bitwidth_config.z_out_, [&](auto tag) {
        using ZOutT = typename decltype(tag)::type;
        if (!hist_collectors.z_out_hist.is_valid()) {
            throw std::runtime_error("z_out_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<ZOutT>(
            hist_collectors.z_out_hist.histogram(), bitwidth_config.z_out_symmetric_,
            quant_params.exp2_inv_z_out_, quant_params.zp_z_out_,
            verbose ? "scale_z_out" : nullptr);
    });

    // r 门输出 - sigmoid
    dispatchByBitWidth(bitwidth_config.r_out_, [&](auto tag) {
        using ROutT = typename decltype(tag)::type;
        if (!hist_collectors.r_out_hist.is_valid()) {
            throw std::runtime_error("r_out_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<ROutT>(
            hist_collectors.r_out_hist.histogram(), bitwidth_config.r_out_symmetric_,
            quant_params.exp2_inv_r_out_, quant_params.zp_r_out_,
            verbose ? "scale_r_out" : nullptr);
    });

    // g 门输出 - tanh
    dispatchByBitWidth(bitwidth_config.g_out_, [&](auto tag) {
        using GOutT = typename decltype(tag)::type;
        if (!hist_collectors.g_out_hist.is_valid()) {
            throw std::runtime_error("g_out_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<GOutT>(
            hist_collectors.g_out_hist.histogram(), bitwidth_config.g_out_symmetric_,
            quant_params.exp2_inv_g_out_, quant_params.zp_g_out_,
            verbose ? "scale_g_out" : nullptr);
    });

    // Rh + br 的量化
    dispatchByBitWidth(bitwidth_config.Rh_add_br_, [&](auto tag) {
        using RhAddBrT = typename decltype(tag)::type;
        if (!hist_collectors.Rh_add_br_g_hist.is_valid()) {
            throw std::runtime_error("Rh_add_br_g_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<RhAddBrT>(
            hist_collectors.Rh_add_br_g_hist.histogram(), bitwidth_config.Rh_add_br_symmetric_,
            quant_params.exp2_inv_Rh_add_br_, quant_params.zp_Rh_add_br_,
            verbose ? "scale_Rh_add_br" : nullptr);
    });

    // r × Rh 的量化
    dispatchByBitWidth(bitwidth_config.rRh_, [&](auto tag) {
        using rRhT = typename decltype(tag)::type;
        if (!hist_collectors.rRh_hist.is_valid()) {
            throw std::runtime_error("rRh_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<rRhT>(
            hist_collectors.rRh_hist.histogram(), bitwidth_config.rRh_symmetric_,
            quant_params.exp2_inv_rRh_, quant_params.zp_rRh_, verbose ? "scale_rRh" : nullptr);
    });

    // (1.0 - z) * g 的量化
    dispatchByBitWidth(bitwidth_config.new_contrib_, [&](auto tag) {
        using NewContribT = typename decltype(tag)::type;
        if (!hist_collectors.new_contrib_hist.is_valid()) {
            throw std::runtime_error("new_contrib_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<NewContribT>(
            hist_collectors.new_contrib_hist.histogram(),
            bitwidth_config.new_contrib_symmetric_, quant_params.exp2_inv_new_contrib_,
            quant_params.zp_new_contrib_, verbose ? "scale_new_contrib" : nullptr);
    });

    // z * h 的量化
    dispatchByBitWidth(bitwidth_config.old_contrib_, [&](auto tag) {
        using OldContribT = typename decltype(tag)::type;
        if (!hist_collectors.old_contrib_hist.is_valid()) {
            throw std::runtime_error("old_contrib_hist is invalid. Calibration data may be empty or corrupted.");
        }
        calibrateQuantParamsFromHistogram<OldContribT>(
            hist_collectors.old_contrib_hist.histogram(),
            bitwidth_config.old_contrib_symmetric_, quant_params.exp2_inv_old_contrib_,
            quant_params.zp_old_contrib_, verbose ? "scale_old_contrib" : nullptr);
    });

    return quant_params;
}
