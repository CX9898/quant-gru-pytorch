// =====================================================================
// GRU 接口层实现 (gru_interface.cpp)
// =====================================================================

#include "gru_interface.h"

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <limits>
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "gru_quant_cpu.h"
#include "histogram_calibration_utils.h"
#include "histogram_collector.h"
#include "calibration_gpu.cuh"
#include "parallel_algorithm.h"
#include "pot_sqnr_calibrator.h"
#include "quantize_ops_helper.h"

// =====================================================================
// 量化参数计算
// =====================================================================

GRUQuantParams calculateGRUQuantitativeParameters(
    const GRUQuantizationRanges &quant_ranges, const OperatorQuantConfig &bitwidth_config) {
    GRUQuantParams quant_params;
    quant_params.hidden_ = quant_ranges.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;

    // 辅助 lambda：简化单值校准调用
    auto calibrateSingle = [](const char* name, QuantBitWidth bw, bool symmetric,
                              float min_val, float max_val, int8_t& shift, int32_t& zp) {
        float aligned_min, aligned_max;
        calibrateQuantParams(min_val, max_val, bw, symmetric, aligned_min, aligned_max, shift, zp, name);
    };

    // 输入 x 和隐藏状态 h
    calibrateSingle("scale_x", bitwidth_config.x_, bitwidth_config.x_symmetric_,
                    quant_ranges.min_x_, quant_ranges.max_x_,
                    quant_params.shift_x_, quant_params.zp_x_);
    calibrateSingle("scale_h", bitwidth_config.h_, bitwidth_config.h_symmetric_,
                    quant_ranges.min_h_, quant_ranges.max_h_,
                    quant_params.shift_h_, quant_params.zp_h_);

    // 权重 W/R 和偏置 bw/br（根据 granularity 设置）
    const int channel_size = quant_ranges.hidden_ * 3;
    const int hidden_size = quant_ranges.hidden_;
    
    // 根据 granularity 设置参数
    // W
    if (bitwidth_config.W_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        // Per-tensor: 从 per-channel 数据计算全局 min/max
        float min_W = *std::min_element(quant_ranges.min_W_.begin(), quant_ranges.min_W_.end());
        float max_W = *std::max_element(quant_ranges.max_W_.begin(), quant_ranges.max_W_.end());
        float aligned_min, aligned_max;
        int32_t zp_tmp;
        calibrateQuantParams(min_W, max_W, bitwidth_config.W_,
                             bitwidth_config.W_symmetric_, aligned_min, aligned_max,
                             quant_params.shift_W_tensor_, zp_tmp);
    } else if (bitwidth_config.W_granularity_ == OperatorQuantConfig::PER_GATE) {
        // Per-gate: 按 gate 分组计算 min/max
        for (int gate = 0; gate < 3; ++gate) {
            float min_gate = std::numeric_limits<float>::max();
            float max_gate = std::numeric_limits<float>::lowest();
            for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) {
                min_gate = std::min(min_gate, quant_ranges.min_W_[c]);
                max_gate = std::max(max_gate, quant_ranges.max_W_[c]);
            }
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams(min_gate, max_gate, bitwidth_config.W_,
                                 bitwidth_config.W_symmetric_, aligned_min, aligned_max,
                                 quant_params.shift_W_gate_[gate], zp_tmp);
        }
    } else {  // PER_CHANNEL
        quant_params.shift_W_.resize(channel_size);
        for (int c = 0; c < channel_size; ++c) {
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams(quant_ranges.min_W_[c], quant_ranges.max_W_[c], bitwidth_config.W_,
                                 bitwidth_config.W_symmetric_, aligned_min, aligned_max,
                                 quant_params.shift_W_[c], zp_tmp);
        }
    }
    
    // R
    if (bitwidth_config.R_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        float min_R = *std::min_element(quant_ranges.min_R_.begin(), quant_ranges.min_R_.end());
        float max_R = *std::max_element(quant_ranges.max_R_.begin(), quant_ranges.max_R_.end());
        float aligned_min, aligned_max;
        int32_t zp_tmp;
        calibrateQuantParams(min_R, max_R, bitwidth_config.R_,
                             bitwidth_config.R_symmetric_, aligned_min, aligned_max,
                             quant_params.shift_R_tensor_, zp_tmp);
    } else if (bitwidth_config.R_granularity_ == OperatorQuantConfig::PER_GATE) {
        for (int gate = 0; gate < 3; ++gate) {
            float min_gate = std::numeric_limits<float>::max();
            float max_gate = std::numeric_limits<float>::lowest();
            for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) {
                min_gate = std::min(min_gate, quant_ranges.min_R_[c]);
                max_gate = std::max(max_gate, quant_ranges.max_R_[c]);
            }
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams(min_gate, max_gate, bitwidth_config.R_,
                                 bitwidth_config.R_symmetric_, aligned_min, aligned_max,
                                 quant_params.shift_R_gate_[gate], zp_tmp);
        }
    } else {  // PER_CHANNEL
        quant_params.shift_R_.resize(channel_size);
        for (int c = 0; c < channel_size; ++c) {
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams(quant_ranges.min_R_[c], quant_ranges.max_R_[c], bitwidth_config.R_,
                                 bitwidth_config.R_symmetric_, aligned_min, aligned_max,
                                 quant_params.shift_R_[c], zp_tmp);
        }
    }
    
    // bw
    if (bitwidth_config.bw_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        float min_bw = *std::min_element(quant_ranges.min_bw_.begin(), quant_ranges.min_bw_.end());
        float max_bw = *std::max_element(quant_ranges.max_bw_.begin(), quant_ranges.max_bw_.end());
        float aligned_min, aligned_max;
        int32_t zp_tmp;
        calibrateQuantParams(min_bw, max_bw, bitwidth_config.bw_,
                             bitwidth_config.bw_symmetric_, aligned_min, aligned_max,
                             quant_params.shift_bw_tensor_, zp_tmp);
    } else if (bitwidth_config.bw_granularity_ == OperatorQuantConfig::PER_GATE) {
        for (int gate = 0; gate < 3; ++gate) {
            float min_gate = std::numeric_limits<float>::max();
            float max_gate = std::numeric_limits<float>::lowest();
            for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) {
                min_gate = std::min(min_gate, quant_ranges.min_bw_[c]);
                max_gate = std::max(max_gate, quant_ranges.max_bw_[c]);
            }
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams(min_gate, max_gate, bitwidth_config.bw_,
                                 bitwidth_config.bw_symmetric_, aligned_min, aligned_max,
                                 quant_params.shift_bw_gate_[gate], zp_tmp);
        }
    } else {  // PER_CHANNEL
        quant_params.shift_bw_.resize(channel_size);
        for (int c = 0; c < channel_size; ++c) {
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams(quant_ranges.min_bw_[c], quant_ranges.max_bw_[c], bitwidth_config.bw_,
                                 bitwidth_config.bw_symmetric_, aligned_min, aligned_max,
                                 quant_params.shift_bw_[c], zp_tmp);
        }
    }
    
    // br
    if (bitwidth_config.br_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        float min_br = *std::min_element(quant_ranges.min_br_.begin(), quant_ranges.min_br_.end());
        float max_br = *std::max_element(quant_ranges.max_br_.begin(), quant_ranges.max_br_.end());
        float aligned_min, aligned_max;
        int32_t zp_tmp;
        calibrateQuantParams(min_br, max_br, bitwidth_config.br_,
                             bitwidth_config.br_symmetric_, aligned_min, aligned_max,
                             quant_params.shift_br_tensor_, zp_tmp);
    } else if (bitwidth_config.br_granularity_ == OperatorQuantConfig::PER_GATE) {
        for (int gate = 0; gate < 3; ++gate) {
            float min_gate = std::numeric_limits<float>::max();
            float max_gate = std::numeric_limits<float>::lowest();
            for (int c = gate * hidden_size; c < (gate + 1) * hidden_size; ++c) {
                min_gate = std::min(min_gate, quant_ranges.min_br_[c]);
                max_gate = std::max(max_gate, quant_ranges.max_br_[c]);
            }
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams(min_gate, max_gate, bitwidth_config.br_,
                                 bitwidth_config.br_symmetric_, aligned_min, aligned_max,
                                 quant_params.shift_br_gate_[gate], zp_tmp);
        }
    } else {  // PER_CHANNEL
        quant_params.shift_br_.resize(channel_size);
        for (int c = 0; c < channel_size; ++c) {
            float aligned_min, aligned_max;
            int32_t zp_tmp;
            calibrateQuantParams(quant_ranges.min_br_[c], quant_ranges.max_br_[c], bitwidth_config.br_,
                                 bitwidth_config.br_symmetric_, aligned_min, aligned_max,
                                 quant_params.shift_br_[c], zp_tmp);
        }
    }

    // Linear 输出 (GEMM+bias)
    calibrateSingle("scale_weight_ih_linear", bitwidth_config.weight_ih_linear_, bitwidth_config.weight_ih_linear_symmetric_,
                    quant_ranges.min_Wx_, quant_ranges.max_Wx_,
                    quant_params.shift_weight_ih_linear_, quant_params.zp_weight_ih_linear_);
    calibrateSingle("scale_weight_hh_linear", bitwidth_config.weight_hh_linear_, bitwidth_config.weight_hh_linear_symmetric_,
                    quant_ranges.min_Rh_, quant_ranges.max_Rh_,
                    quant_params.shift_weight_hh_linear_, quant_params.zp_weight_hh_linear_);

    // 门激活输入
    calibrateSingle("scale_update_gate_input", bitwidth_config.update_gate_input_, bitwidth_config.update_gate_input_symmetric_,
                    quant_ranges.min_update_gate_input_, quant_ranges.max_update_gate_input_,
                    quant_params.shift_update_gate_input_, quant_params.zp_update_gate_input_);
    calibrateSingle("scale_reset_gate_input", bitwidth_config.reset_gate_input_, bitwidth_config.reset_gate_input_symmetric_,
                    quant_ranges.min_reset_gate_input_, quant_ranges.max_reset_gate_input_,
                    quant_params.shift_reset_gate_input_, quant_params.zp_reset_gate_input_);
    calibrateSingle("scale_new_gate_input", bitwidth_config.new_gate_input_, bitwidth_config.new_gate_input_symmetric_,
                    quant_ranges.min_new_gate_input_, quant_ranges.max_new_gate_input_,
                    quant_params.shift_new_gate_input_, quant_params.zp_new_gate_input_);

    // 激活函数输出（确保最小范围）
    constexpr float MIN_ACTIVATION_RANGE = 0.5f;
    auto calibrateWithMinRange = [&](const char* name, QuantBitWidth bw, bool symmetric,
                                     float min_val, float max_val, int8_t& shift, int32_t& zp) {
        ensureMinRange(min_val, max_val, MIN_ACTIVATION_RANGE, name);
        float aligned_min, aligned_max;
        calibrateQuantParams(min_val, max_val, bw, symmetric, aligned_min, aligned_max, shift, zp, name);
    };

    calibrateWithMinRange("scale_update_gate_output", bitwidth_config.update_gate_output_, bitwidth_config.update_gate_output_symmetric_,
                          quant_ranges.min_update_gate_output_, quant_ranges.max_update_gate_output_,
                          quant_params.shift_update_gate_output_, quant_params.zp_update_gate_output_);
    calibrateWithMinRange("scale_reset_gate_output", bitwidth_config.reset_gate_output_, bitwidth_config.reset_gate_output_symmetric_,
                          quant_ranges.min_reset_gate_output_, quant_ranges.max_reset_gate_output_,
                          quant_params.shift_reset_gate_output_, quant_params.zp_reset_gate_output_);
    calibrateWithMinRange("scale_new_gate_output", bitwidth_config.new_gate_output_, bitwidth_config.new_gate_output_symmetric_,
                          quant_ranges.min_new_gate_output_, quant_ranges.max_new_gate_output_,
                          quant_params.shift_new_gate_output_, quant_params.zp_new_gate_output_);

    // 中间计算
    calibrateSingle("scale_mul_reset_hidden", bitwidth_config.mul_reset_hidden_, bitwidth_config.mul_reset_hidden_symmetric_,
                    quant_ranges.min_mul_reset_hidden_, quant_ranges.max_mul_reset_hidden_,
                    quant_params.shift_mul_reset_hidden_, quant_params.zp_mul_reset_hidden_);
    calibrateSingle("scale_mul_new_contribution", bitwidth_config.mul_new_contribution_, bitwidth_config.mul_new_contribution_symmetric_,
                    quant_ranges.min_mul_new_contribution_, quant_ranges.max_mul_new_contribution_,
                    quant_params.shift_mul_new_contribution_, quant_params.zp_mul_new_contribution_);
    calibrateSingle("scale_mul_old_contribution", bitwidth_config.mul_old_contribution_, bitwidth_config.mul_old_contribution_symmetric_,
                    quant_ranges.min_mul_old_contribution_, quant_ranges.max_mul_old_contribution_,
                    quant_params.shift_mul_old_contribution_, quant_params.zp_mul_old_contribution_);

    // 生成 LUT 并存储到参数中
    generate_piecewise_linear_lut_to_params(quant_params);

    return quant_params;
}

// =====================================================================
// 统一前向/反向传播接口（主入口）
// =====================================================================

// 量化模式选择宏：默认使用浮点存储版（方案2）
// 如需使用 INT32 版本（方案1），注释掉此行
#define USE_FP_STORAGE 1


// =====================================================================
// Haste GRU 前向传播实现
// =====================================================================

void hasteGRUForward(bool is_training, const int time_steps, const int batch_size,
                     const int input_size, const int hidden_size, const float *W, const float *R,
                     const float *bw, const float *br, const float *x, const float *h0,
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

    forward.Run(time_steps, W, R, bw, br, x, h, v, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f,
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
                      const int hidden_size, const float *W_t, const float *R_t, const float *bw,
                      const float *br, const float *x_t, const float *dh_new, const float *h,
                      const float *v, const cublasHandle_t &g_blas_handle, float *dx, float *dW,
                      float *dR, float *dbw, float *dbr, float *dh) {
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）

    gru::BackwardPass<float> backward(batch_size, input_size, hidden_size, g_blas_handle);

    backward.Run(time_steps, W_t, R_t, bw, br, x_t, h, v, dh_new, dx, dW, dR, dbw, dbr, dh,
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
// 量化 GRU 反向传播（支持 QAT mask）
// =====================================================================

void quantGRUBackward(const int time_steps, const int batch_size, const int input_size,
                      const int hidden_size, const float *W_t, const float *R_t, const float *bw,
                      const float *br, const float *x_t, const float *dh_new, const float *h,
                      const float *v, const cublasHandle_t &g_blas_handle, float *dx, float *dW,
                      float *dR, float *dbw, float *dbr, float *dh,
                      const GRUQuantParams *quant_params,
                      const uint8_t *x_mask, const uint8_t *h0_mask,
                      const uint8_t *W_mask, const uint8_t *R_mask,
                      const uint8_t *bw_mask, const uint8_t *br_mask,
                      const uint8_t *weight_ih_linear_mask, const uint8_t *weight_hh_linear_mask,
                      const uint8_t *gate_input_mask, const uint8_t *gate_output_mask,
                      const uint8_t *h_mask) {
    // 临时缓存梯度（内部结构用）
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size * 3);

    // 使用量化版反向传播类（支持 QAT mask）
    // 注意：不需要rescale补偿，因为反向传播使用的是反量化后的浮点值，梯度计算已经是正确的
    // quant_params 参数保留以保持接口兼容性，但不再使用
    (void)quant_params;  // suppress unused parameter warning
    gru::BackwardPassQuant<float> backward(batch_size, input_size, hidden_size, g_blas_handle);

    backward.Run(time_steps, W_t, R_t, bw, br, x_t, h, v, dh_new,
                 dx, dW, dR, dbw, dbr, dh,
                 dp_dev.data(), dq_dev.data(),
                 nullptr,  // zoneout_mask
                 // QAT masks
                 x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
                 weight_ih_linear_mask, weight_hh_linear_mask,
                 gate_input_mask, gate_output_mask, h_mask);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUBackward: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUBackward: ") + err_str);
    }
}

// =====================================================================
// 权重量化实现（统一 int32_t 输出）
// =====================================================================

void quantitativeWeight(const int input_size, const int hidden_size, const float *W, const float *R,
                        const float *bw, const float *br,
                        const GRUQuantParams &quant_parms, int32_t *W_quant,
                        int32_t *R_quant, int32_t *bw_quant, int32_t *br_quant) {
    // 显式创建dev::vector以避免临时对象问题
    dev::vector<int8_t> shift_W_dev(quant_parms.shift_W_);
    dev::vector<int8_t> shift_R_dev(quant_parms.shift_R_);
    dev::vector<int8_t> shift_bw_dev(quant_parms.shift_bw_);
    dev::vector<int8_t> shift_br_dev(quant_parms.shift_br_);

    // 统一 int32_t 输出，使用 clamp_by_bitwidth 限制到实际位宽
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    dev::quantificationPerChannelBitwidth<false>(W, W_quant, nullptr, input_size, 3 * hidden_size, 
                                                  shift_W_dev, bw_cfg.W_);
    dev::quantificationPerChannelBitwidth<false>(R, R_quant, nullptr, hidden_size, 3 * hidden_size, 
                                                  shift_R_dev, bw_cfg.R_);
    dev::quantificationPerChannelBitwidth<false>(bw, bw_quant, nullptr, 1, 3 * hidden_size, 
                                                  shift_bw_dev, bw_cfg.bw_);
    dev::quantificationPerChannelBitwidth<false>(br, br_quant, nullptr, 1, 3 * hidden_size, 
                                                 shift_br_dev, bw_cfg.br_);

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

// =====================================================================
// GRU 权重量化统一接口（封装 W, R, bw, br）- 浮点存储版本
// =====================================================================

// GRU 权重量化统一接口（根据 granularity 自动选择量化方式）
template <bool Training>
void quantizeGRUWeights(const float *W, const float *R, const float *bw, const float *br,
                        float *W_q_out, float *R_q_out, float *bw_q_out, float *br_q_out,
                        uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask,
                        size_t input_size, size_t hidden_size,
                        const GRUQuantParams &quant_params) {
    const auto &bw_cfg = quant_params.bitwidth_config_;
    
    // 占位符空 vector（当不是 PER_CHANNEL 时使用，函数内部不会访问）
    static const dev::vector<int8_t> empty_shift;
    
    // 只在 PER_CHANNEL 粒度时创建 shift 数组
    dev::vector<int8_t> shift_W_dev, shift_R_dev, shift_bw_dev, shift_br_dev;
    if (bw_cfg.W_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_W_dev = dev::vector<int8_t>(quant_params.shift_W_);
    }
    if (bw_cfg.R_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_R_dev = dev::vector<int8_t>(quant_params.shift_R_);
    }
    if (bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_bw_dev = dev::vector<int8_t>(quant_params.shift_bw_);
    }
    if (bw_cfg.br_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_br_dev = dev::vector<int8_t>(quant_params.shift_br_);
    }
    
    // 量化 W
    dev::quantificationWeightFP<Training>(W, W_q_out, W_mask, input_size, hidden_size,
                                          bw_cfg.W_granularity_,
                                          quant_params.shift_W_tensor_,
                                          quant_params.shift_W_gate_,
                                          bw_cfg.W_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_W_dev : empty_shift,
                                          bw_cfg.W_);
    
    // 量化 R
    dev::quantificationWeightFP<Training>(R, R_q_out, R_mask, hidden_size, hidden_size,
                                          bw_cfg.R_granularity_,
                                          quant_params.shift_R_tensor_,
                                          quant_params.shift_R_gate_,
                                          bw_cfg.R_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_R_dev : empty_shift,
                                          bw_cfg.R_);
    
    // 量化 bw
    dev::quantificationWeightFP<Training>(bw, bw_q_out, bw_mask, 1, hidden_size,
                                          bw_cfg.bw_granularity_,
                                          quant_params.shift_bw_tensor_,
                                          quant_params.shift_bw_gate_,
                                          bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_bw_dev : empty_shift,
                                          bw_cfg.bw_);
    
    // 量化 br
    dev::quantificationWeightFP<Training>(br, br_q_out, br_mask, 1, hidden_size,
                                          bw_cfg.br_granularity_,
                                          quant_params.shift_br_tensor_,
                                          quant_params.shift_br_gate_,
                                          bw_cfg.br_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_br_dev : empty_shift,
                                          bw_cfg.br_);
}

// 显式实例化模板
template void quantizeGRUWeights<false>(const float *W, const float *R, const float *bw, const float *br,
                                        float *W_q_out, float *R_q_out, float *bw_q_out, float *br_q_out,
                                        uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask,
                                        size_t input_size, size_t hidden_size,
                                        const GRUQuantParams &quant_params);
template void quantizeGRUWeights<true>(const float *W, const float *R, const float *bw, const float *br,
                                       float *W_q_out, float *R_q_out, float *bw_q_out, float *br_q_out,
                                       uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask,
                                       size_t input_size, size_t hidden_size,
                                       const GRUQuantParams &quant_params);

// 反量化 GRU 权重（W, R, bw, br）- 使用统一接口，内部根据 granularity 自动选择
void dequantizeGRUWeights(float *W_q, float *R_q, float *bw_q, float *br_q,
                          size_t input_size, size_t hidden_size,
                          const GRUQuantParams &quant_params) {
    const auto &bw_cfg = quant_params.bitwidth_config_;
    
    // 占位符空 vector（当不是 PER_CHANNEL 时使用，函数内部不会访问）
    static const dev::vector<int8_t> empty_shift;
    
    // 只在 PER_CHANNEL 粒度时创建 shift 数组
    dev::vector<int8_t> shift_W_dev, shift_R_dev, shift_bw_dev, shift_br_dev;
    if (bw_cfg.W_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_W_dev = dev::vector<int8_t>(quant_params.shift_W_);
    }
    if (bw_cfg.R_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_R_dev = dev::vector<int8_t>(quant_params.shift_R_);
    }
    if (bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_bw_dev = dev::vector<int8_t>(quant_params.shift_bw_);
    }
    if (bw_cfg.br_granularity_ == OperatorQuantConfig::PER_CHANNEL) {
        shift_br_dev = dev::vector<int8_t>(quant_params.shift_br_);
    }
    
    // 反量化 W
    dev::dequantificationWeightFPInplace(W_q, input_size, hidden_size,
                                         bw_cfg.W_granularity_,
                                         quant_params.shift_W_tensor_,
                                         quant_params.shift_W_gate_,
                                         bw_cfg.W_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_W_dev : empty_shift);
    
    // 反量化 R
    dev::dequantificationWeightFPInplace(R_q, hidden_size, hidden_size,
                                        bw_cfg.R_granularity_,
                                        quant_params.shift_R_tensor_,
                                        quant_params.shift_R_gate_,
                                        bw_cfg.R_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_R_dev : empty_shift);
    
    // 反量化 bw
    dev::dequantificationWeightFPInplace(bw_q, 1, hidden_size,
                                        bw_cfg.bw_granularity_,
                                        quant_params.shift_bw_tensor_,
                                        quant_params.shift_bw_gate_,
                                        bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_bw_dev : empty_shift);
    
    // 反量化 br
    dev::dequantificationWeightFPInplace(br_q, 1, hidden_size,
                                        bw_cfg.br_granularity_,
                                        quant_params.shift_br_tensor_,
                                        quant_params.shift_br_gate_,
                                        bw_cfg.br_granularity_ == OperatorQuantConfig::PER_CHANNEL ? shift_br_dev : empty_shift);
}

// =====================================================================
// 纯定点 GRU 前向传播（GPU 核心实现）
// =====================================================================

// GPU 纯定点 GRU 前向传播（int32 输入/输出）
// 这是量化 GRU 的核心计算，所有高层接口都调用此函数
void quantGRUForwardInt32(
    bool is_training, int time_steps, int batch_size, int input_size, int hidden_size,
    const int32_t *W_q, const int32_t *R_q, const int32_t *bw_q, const int32_t *br_q,
    const int32_t *x_q, const int32_t *h0_q,
    const GRUQuantParams &quant_params,
    const cublasHandle_t &g_blas_handle,
    int32_t *h_q, int32_t *v_q,
    uint8_t *weight_ih_linear_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask) {
    
    const int NH = batch_size * hidden_size;
    
    // 初始化 h_q[0]
    if (h0_q != nullptr) {
        cudaMemcpy(h_q, h0_q, NH * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    } else {
        dev::fill_n(h_q, NH, quant_params.zp_h_);
    }
    
    // 分配中间值缓冲区
    dev::vector<int32_t> v_internal;
    int32_t* v_ptr = v_q;
    if (v_q == nullptr) {
        // 内部分配临时缓冲区
        v_internal.resize(time_steps * batch_size * hidden_size * 4);
        v_ptr = v_internal.data();
    }
    
    // 创建 ForwardPassQuant 对象
    gru::ForwardPassQuant forward(is_training, batch_size, input_size, hidden_size, g_blas_handle);
    forward.setRescaleParam(quant_params);
    
    // 运行前向传播（传递 mask 指针）
    forward.Run(time_steps, W_q, R_q, bw_q, br_q, x_q, h_q, v_ptr, 0.0f, nullptr,
                weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask);
    
    // 同步 CUDA 操作
    cudaDeviceSynchronize();
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUForwardInt32: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUForwardInt32: ") + err_str);
    }
}

// =====================================================================
// 量化 GRU 前向传播（浮点接口）
// =====================================================================

// 量化 GRU 前向传播（浮点输入/输出，内部自动量化权重和激活）
void quantGRUForward(bool is_training, const int time_steps, const int batch_size,
                     const int input_size, const int hidden_size, const float *W,
                     const float *R, const float *bw, const float *br, const float *x,
                     const float *h0, const GRUQuantParams &quant_parms,
                     const cublasHandle_t &g_blas_handle, float *h, float *v,
                     // 输入量化 mask
                     uint8_t *x_mask,
                     uint8_t *h0_mask,
                     uint8_t *W_mask,
                     uint8_t *R_mask,
                     uint8_t *bw_mask,
                     uint8_t *br_mask,
                     // 计算过程 mask
                     uint8_t *weight_ih_linear_mask,
                     uint8_t *weight_hh_linear_mask,
                     uint8_t *gate_input_mask,
                     uint8_t *gate_output_mask,
                     uint8_t *h_mask) {
    const int hidden3 = hidden_size * 3;
    const std::size_t x_size = time_steps * batch_size * input_size;
    const std::size_t h_size = (time_steps + 1) * batch_size * hidden_size;
    const std::size_t h0_size = batch_size * hidden_size;
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    // 注意：hidden_size 已在函数参数中声明，不需要重新声明

    // 根据粒度配置构建 shift 数组（用于量化 kernel）
    std::vector<int8_t> shift_W_array(hidden3);
    std::vector<int8_t> shift_R_array(hidden3);
    std::vector<int8_t> shift_bw_array(hidden3);
    std::vector<int8_t> shift_br_array(hidden3);

    for (int idx = 0; idx < hidden3; ++idx) {
        // W
        if (bw_cfg.W_granularity_ == OperatorQuantConfig::PER_TENSOR) {
            shift_W_array[idx] = quant_parms.shift_W_tensor_;
        } else if (bw_cfg.W_granularity_ == OperatorQuantConfig::PER_GATE) {
            int gate_idx = idx / hidden_size;
            shift_W_array[idx] = quant_parms.shift_W_gate_[gate_idx];
        } else {  // PER_CHANNEL
            shift_W_array[idx] = quant_parms.shift_W_[idx];
        }
        
        // R
        if (bw_cfg.R_granularity_ == OperatorQuantConfig::PER_TENSOR) {
            shift_R_array[idx] = quant_parms.shift_R_tensor_;
        } else if (bw_cfg.R_granularity_ == OperatorQuantConfig::PER_GATE) {
            int gate_idx = idx / hidden_size;
            shift_R_array[idx] = quant_parms.shift_R_gate_[gate_idx];
        } else {  // PER_CHANNEL
            shift_R_array[idx] = quant_parms.shift_R_[idx];
        }
        
        // bw
        if (bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_TENSOR) {
            shift_bw_array[idx] = quant_parms.shift_bw_tensor_;
        } else if (bw_cfg.bw_granularity_ == OperatorQuantConfig::PER_GATE) {
            int gate_idx = idx / hidden_size;
            shift_bw_array[idx] = quant_parms.shift_bw_gate_[gate_idx];
        } else {  // PER_CHANNEL
            shift_bw_array[idx] = quant_parms.shift_bw_[idx];
        }
        
        // br
        if (bw_cfg.br_granularity_ == OperatorQuantConfig::PER_TENSOR) {
            shift_br_array[idx] = quant_parms.shift_br_tensor_;
        } else if (bw_cfg.br_granularity_ == OperatorQuantConfig::PER_GATE) {
            int gate_idx = idx / hidden_size;
            shift_br_array[idx] = quant_parms.shift_br_gate_[gate_idx];
        } else {  // PER_CHANNEL
            shift_br_array[idx] = quant_parms.shift_br_[idx];
        }
    }

    // 拷贝 shift 到 device（用于量化）
    dev::vector<int8_t> shift_W_dev(shift_W_array);
    dev::vector<int8_t> shift_R_dev(shift_R_array);
    dev::vector<int8_t> shift_bw_dev(shift_bw_array);
    dev::vector<int8_t> shift_br_dev(shift_br_array);

    // 1. 量化权重
    dev::vector<int32_t> W_q(input_size * hidden3);
    dev::vector<int32_t> R_q(hidden_size * hidden3);
    dev::vector<int32_t> bw_q(hidden3);
    dev::vector<int32_t> br_q(hidden3);

    // 量化权重（使用模板参数区分训练/推理）
    if (is_training) {
        dev::quantificationPerChannelBitwidth<true>(W, W_q.data(), W_mask, input_size, hidden3, shift_W_dev, bw_cfg.W_);
        dev::quantificationPerChannelBitwidth<true>(R, R_q.data(), R_mask, hidden_size, hidden3, shift_R_dev, bw_cfg.R_);
        dev::quantificationPerChannelBitwidth<true>(bw, bw_q.data(), bw_mask, 1, hidden3, shift_bw_dev, bw_cfg.bw_);
        dev::quantificationPerChannelBitwidth<true>(br, br_q.data(), br_mask, 1, hidden3, shift_br_dev, bw_cfg.br_);
    } else {
        dev::quantificationPerChannelBitwidth<false>(W, W_q.data(), nullptr, input_size, hidden3, shift_W_dev, bw_cfg.W_);
        dev::quantificationPerChannelBitwidth<false>(R, R_q.data(), nullptr, hidden_size, hidden3, shift_R_dev, bw_cfg.R_);
        dev::quantificationPerChannelBitwidth<false>(bw, bw_q.data(), nullptr, 1, hidden3, shift_bw_dev, bw_cfg.bw_);
        dev::quantificationPerChannelBitwidth<false>(br, br_q.data(), nullptr, 1, hidden3, shift_br_dev, bw_cfg.br_);
    }

    // 2. 量化输入 x
    dev::vector<int32_t> x_quant(x_size);
    if (is_training) {
        dev::quantificationBitwidth<true>(x, x_quant.data(), x_mask, x_size, 
                                          quant_parms.shift_x_, quant_parms.zp_x_, bw_cfg.x_);
    } else {
        dev::quantificationBitwidth<false>(x, x_quant.data(), nullptr, x_size, 
                                           quant_parms.shift_x_, quant_parms.zp_x_, bw_cfg.x_);
    }

    // 3. 量化初始隐藏状态 h0（空 vector 的 .data() 返回 nullptr）
    dev::vector<int32_t> h0_quant;
    if (h0 != nullptr) {
        h0_quant.resize(h0_size);
        if (is_training) {
            dev::quantificationBitwidth<true>(h0, h0_quant.data(), h0_mask, h0_size, 
                                              quant_parms.shift_h_, quant_parms.zp_h_, bw_cfg.h_);
        } else {
            dev::quantificationBitwidth<false>(h0, h0_quant.data(), nullptr, h0_size, 
                                               quant_parms.shift_h_, quant_parms.zp_h_, bw_cfg.h_);
        }
    }

    // 4. 分配输出缓冲区
    dev::vector<int32_t> h_quant(h_size);
    dev::vector<int32_t> v_quant(v != nullptr ? time_steps * batch_size * hidden_size * 4 : 0);

    // 5. 调用核心定点计算（传递 mask 指针）
    quantGRUForwardInt32(is_training, time_steps, batch_size, input_size, hidden_size,
                         W_q.data(), R_q.data(), bw_q.data(), br_q.data(),
                         x_quant.data(), h0_quant.data(),
                         quant_parms, g_blas_handle,
                         h_quant.data(), v != nullptr ? v_quant.data() : nullptr,
                         weight_ih_linear_mask, weight_hh_linear_mask, 
                         gate_input_mask, gate_output_mask, h_mask);

    // 6. 反量化输出 h
    dev::dequantification(h_quant.data(), h, h_size, quant_parms.shift_h_, quant_parms.zp_h_);

    // 7. 反量化中间值 v（如需要）
    if (v != nullptr) {
        dev::dequantificationV(v_quant.data(), v, time_steps, batch_size, hidden_size,
                               quant_parms.shift_update_gate_output_, quant_parms.zp_update_gate_output_,
                               quant_parms.shift_reset_gate_output_, quant_parms.zp_reset_gate_output_,
                               quant_parms.shift_new_gate_output_, quant_parms.zp_new_gate_output_,
                               quant_parms.shift_weight_hh_linear_, quant_parms.zp_weight_hh_linear_);
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

// =====================================================================
// CPU 量化 GRU 前向传播实现
// =====================================================================

// CPU 量化 GRU 前向传播（量化权重输入版本）
// 所有量化值使用 int32_t 存储，实际值通过位宽配置限制
void quantGRUForwardCPU(bool is_training, int time_steps, int batch_size, int input_size,
                        int hidden_size, const int32_t *W, const int32_t *R, const int32_t *bw,
                        const int32_t *br, const float *x, const float *h0,
                        const GRUQuantParams &quant_parms, float *h, float *v) {
    const std::size_t x_size = time_steps * batch_size * input_size;
    const std::size_t h_total_size = (time_steps + 1) * batch_size * hidden_size;
    const auto &bw_cfg = quant_parms.bitwidth_config_;

    // 量化输入序列
    std::vector<int32_t> x_quant(x_size);
    for (std::size_t i = 0; i < x_size; i++) {
        x_quant[i] = quantize(x[i], quant_parms.shift_x_, quant_parms.zp_x_, bw_cfg.x_);
    }

    // 分配隐藏状态缓冲区
    std::vector<int32_t> h_quant(h_total_size);

    // 初始化 h0 为零点值
    for (int i = 0; i < batch_size * hidden_size; i++) {
        h_quant[i] = quant_parms.zp_h_;
    }

    // 如果提供了初始状态，量化到 h_quant[0]
    if (h0 != nullptr) {
        for (int i = 0; i < batch_size * hidden_size; i++) {
            h_quant[i] = quantize(h0[i], quant_parms.shift_h_, quant_parms.zp_h_, bw_cfg.h_);
        }
    }

    // 分配中间值缓冲区（v: update_gate, reset_gate, new_gate, weight_hh_linear_g）
    std::vector<int32_t> v_quant(time_steps * batch_size * hidden_size * 4);

    // 创建 CPU 版本的 ForwardPassQuantCPU
    cpu::ForwardPassQuantCPU forward(is_training, batch_size, input_size, hidden_size);
    forward.setRescaleParam(quant_parms);

    // 运行前向传播
    forward.Run(time_steps, W, R, bw, br, x_quant.data(), h_quant.data(),
                v != nullptr ? v_quant.data() : nullptr, 0.0f, nullptr);

    // 反量化隐藏状态输出
    for (std::size_t i = 0; i < h_total_size; i++) {
        h[i] = dequantize(h_quant[i], quant_parms.shift_h_, quant_parms.zp_h_);
    }

    // 如果需要中间值，反量化 v
    if (v != nullptr) {
        const int hidden3 = hidden_size * 3;
        for (int t = 0; t < time_steps; t++) {
            for (int b = 0; b < batch_size; b++) {
                for (int j = 0; j < hidden_size; j++) {
                    const int base_idx = (t * batch_size + b) * hidden_size * 4 + j;
                    // v[0] = update_gate, v[1] = reset_gate, v[2] = new_gate, v[3] = weight_hh_linear_g
                    v[base_idx + hidden_size * 0] = dequantize(
                        v_quant[base_idx + hidden_size * 0], quant_parms.shift_update_gate_output_,
                        quant_parms.zp_update_gate_output_);
                    v[base_idx + hidden_size * 1] = dequantize(
                        v_quant[base_idx + hidden_size * 1], quant_parms.shift_reset_gate_output_,
                        quant_parms.zp_reset_gate_output_);
                    v[base_idx + hidden_size * 2] = dequantize(
                        v_quant[base_idx + hidden_size * 2], quant_parms.shift_new_gate_output_,
                        quant_parms.zp_new_gate_output_);
                    v[base_idx + hidden_size * 3] = dequantize(
                        v_quant[base_idx + hidden_size * 3], quant_parms.shift_weight_hh_linear_,
                        quant_parms.zp_weight_hh_linear_);
                }
            }
        }
    }
}

// CPU 量化 GRU 前向传播（浮点权重输入版本，内部量化）
void quantGRUForwardCPU(bool is_training, int time_steps, int batch_size, int input_size,
                        int hidden_size, const float *W, const float *R, const float *bw,
                        const float *br, const float *x, const float *h0,
                        const GRUQuantParams &quant_parms, float *h, float *v) {
    const int hidden3 = hidden_size * 3;
    const auto &bw_cfg = quant_parms.bitwidth_config_;

    // 量化权重矩阵（per-channel）
    std::vector<int32_t> W_quant(input_size * hidden3);
    std::vector<int32_t> R_quant(hidden_size * hidden3);

    for (int k = 0; k < input_size; k++) {
        for (int m = 0; m < hidden3; m++) {
            int idx = k * hidden3 + m;
            W_quant[idx] = quantize(W[idx], quant_parms.shift_W_[m], 0, bw_cfg.W_);
        }
    }

    for (int k = 0; k < hidden_size; k++) {
        for (int m = 0; m < hidden3; m++) {
            int idx = k * hidden3 + m;
            R_quant[idx] = quantize(R[idx], quant_parms.shift_R_[m], 0, bw_cfg.R_);
        }
    }

    // 量化偏置（per-channel）
    std::vector<int32_t> bw_quant(hidden3);
    std::vector<int32_t> br_quant(hidden3);
    for (int m = 0; m < hidden3; m++) {
        bw_quant[m] = quantize(bw[m], quant_parms.shift_bw_[m], 0, bw_cfg.bw_);
        br_quant[m] = quantize(br[m], quant_parms.shift_br_[m], 0, bw_cfg.br_);
    }

    // 调用量化权重版本
    quantGRUForwardCPU(is_training, time_steps, batch_size, input_size, hidden_size,
                       W_quant.data(), R_quant.data(), bw_quant.data(), br_quant.data(), x, h0,
                       quant_parms, h, v);
}

// =====================================================================
// 浮点存储版量化 GRU 前向传播（GPU-FP）
// =====================================================================

void quantGRUForwardFP(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br,
    const float *x, const float *h0,
    const GRUQuantParams &quant_params,
    const cublasHandle_t &g_blas_handle,
    float *h, float *v,
    // 输出量化后的值（必须由外部分配内存，训练和推理模式都需要）
    // 函数会直接写入这些指针指向的内存，无需拷贝
    float *W_q_out,
    float *R_q_out,
    float *bw_q_out,
    float *br_q_out,
    float *x_q_out,
    // 输入量化 mask（训练时外部分配，推理时可为 nullptr）
    uint8_t *x_mask,
    uint8_t *h0_mask,
    uint8_t *W_mask,
    uint8_t *R_mask,
    uint8_t *bw_mask,
    uint8_t *br_mask,
    // 计算过程 mask（训练时外部分配，推理时可为 nullptr）
    uint8_t *weight_ih_linear_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask) {
    
    const int hidden3 = hidden_size * 3;
    const auto &bw_cfg = quant_params.bitwidth_config_;
    const std::size_t x_size = time_steps * batch_size * input_size;
    const std::size_t h_size = (time_steps + 1) * batch_size * hidden_size;
    const int NH = batch_size * hidden_size;
    
    // 量化权重（W, R, bw, br）- 使用统一接口，内部根据 granularity 自动选择
    if (is_training) {
        quantizeGRUWeights<true>(W, R, bw, br,
                                 W_q_out, R_q_out, bw_q_out, br_q_out,
                                 W_mask, R_mask, bw_mask, br_mask,
                                 input_size, hidden_size, quant_params);
        // x (始终 per-tensor)
        dev::quantificationFP<true>(x, x_q_out, x_mask, x_size, quant_params.shift_x_,
                                    quant_params.zp_x_, bw_cfg.x_);
    } else {
        quantizeGRUWeights<false>(W, R, bw, br,
                                  W_q_out, R_q_out, bw_q_out, br_q_out,
                                  nullptr, nullptr, nullptr, nullptr,
                                  input_size, hidden_size, quant_params);
        // x (始终 per-tensor)
        dev::quantificationFP<false>(x, x_q_out, nullptr, x_size, quant_params.shift_x_,
                                     quant_params.zp_x_, bw_cfg.x_);
    }
    
    // 量化 h0 到 h 的前 NH 个元素（直接使用外部 h 缓冲区）
    if (h0 != nullptr) {
        if (is_training) {
            dev::quantificationFP<true>(h0, h, h0_mask, NH, quant_params.shift_h_,
                                       quant_params.zp_h_, bw_cfg.h_);
        } else {
            dev::quantificationFP<false>(h0, h, nullptr, NH, quant_params.shift_h_,
                                        quant_params.zp_h_, bw_cfg.h_);
        }
    } else {
        // 填充零点值（表示初始隐状态为 0）
        dev::fill_n(h, NH, static_cast<float>(quant_params.zp_h_));
    }
    
    // 前向传播：直接使用外部 h 和 v 缓冲区（会写入量化值）
    gru::ForwardPassQuantFP forward_fp(is_training, batch_size, input_size, hidden_size,
                                       g_blas_handle, nullptr);
    forward_fp.setRescaleParam(quant_params);
    forward_fp.Run(time_steps, W_q_out, R_q_out,
                   bw_q_out, br_q_out,
                   x_q_out, h,
                   v,
                   0.0f, nullptr,
                   weight_ih_linear_mask,
                   weight_hh_linear_mask,
                   gate_input_mask,
                   gate_output_mask,
                   h_mask);
    
    // 原地反量化输出
    dev::dequantificationFPInplace(h, h_size, quant_params.shift_h_, quant_params.zp_h_);
    if (v != nullptr) {
        dev::dequantificationVFPInplace(v, time_steps, batch_size, hidden_size,
                                         quant_params.shift_update_gate_output_, quant_params.zp_update_gate_output_,
                                         quant_params.shift_reset_gate_output_, quant_params.zp_reset_gate_output_,
                                         quant_params.shift_new_gate_output_, quant_params.zp_new_gate_output_,
                                         quant_params.shift_weight_hh_linear_, quant_params.zp_weight_hh_linear_);
    }
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in quantGRUForwardFP: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in quantGRUForwardFP: ") + err_str);
    }
}

// =====================================================================
// GPU 直方图收集器转换函数
// =====================================================================

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
    convert_collector(cpu_collectors.update_gate_input_hist, gpu_collectors.update_gate_input_hist);
    convert_collector(cpu_collectors.reset_gate_input_hist, gpu_collectors.reset_gate_input_hist);
    convert_collector(cpu_collectors.new_gate_input_hist, gpu_collectors.new_gate_input_hist);
    convert_collector(cpu_collectors.update_gate_output_hist, gpu_collectors.update_gate_output_hist);
    convert_collector(cpu_collectors.reset_gate_output_hist, gpu_collectors.reset_gate_output_hist);
    convert_collector(cpu_collectors.new_gate_output_hist, gpu_collectors.new_gate_output_hist);
    convert_collector(cpu_collectors.mul_reset_hidden_hist, gpu_collectors.mul_reset_hidden_hist);
    convert_collector(cpu_collectors.mul_new_contribution_hist, gpu_collectors.mul_new_contribution_hist);
    convert_collector(cpu_collectors.mul_old_contribution_hist, gpu_collectors.mul_old_contribution_hist);

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
    convert_batch(cpu_collectors.bw_hist, gpu_collectors.bw_batch);
    convert_batch(cpu_collectors.br_hist, gpu_collectors.br_batch);
    
    // 转换 per-tensor 直方图
    convert_collector(cpu_collectors.W_tensor_hist, gpu_collectors.W_tensor_hist);
    convert_collector(cpu_collectors.R_tensor_hist, gpu_collectors.R_tensor_hist);
    convert_collector(cpu_collectors.bw_tensor_hist, gpu_collectors.bw_tensor_hist);
    convert_collector(cpu_collectors.br_tensor_hist, gpu_collectors.br_tensor_hist);
    
    // 转换 per-gate 直方图
    for (int i = 0; i < 3; ++i) {
        convert_collector(cpu_collectors.W_gate_hist[i], gpu_collectors.W_gate_hist[i]);
        convert_collector(cpu_collectors.R_gate_hist[i], gpu_collectors.R_gate_hist[i]);
        convert_collector(cpu_collectors.bw_gate_hist[i], gpu_collectors.bw_gate_hist[i]);
        convert_collector(cpu_collectors.br_gate_hist[i], gpu_collectors.br_gate_hist[i]);
    }

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in convertGPUHistogramsToCPU: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in convertGPUHistogramsToCPU: ") + err_str);
    }

    return cpu_collectors;
}

GRUQuantParams calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors &hist_collectors, const OperatorQuantConfig &bitwidth_config,
    bool verbose, bool use_percentile, float percentile_value) {
    GRUQuantParams quant_params;
    quant_params.hidden_ = hist_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;

    const int channel_size = hist_collectors.hidden_ * 3;
    const int hidden_size = hist_collectors.hidden_;
    
    // 辅助 lambda：校准单个直方图
    auto histCalibrate = [&](const HistogramCollector& hist, QuantBitWidth bw, bool sym,
                             int8_t& shift, int32_t& zp, const char* name) {
        if (!hist.is_valid()) throw std::runtime_error(std::string(name) + " is invalid.");
        calibrateQuantParamsFromHistogram(hist.histogram(), bw, sym, shift, zp,
                                          verbose ? name : nullptr, use_percentile, percentile_value);
    };

    // 根据 granularity 设置权重参数
    // W
    if (bitwidth_config.W_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        // Per-tensor: 直接使用独立的 per-tensor 直方图
        int32_t zp_tmp;
        histCalibrate(hist_collectors.W_tensor_hist, bitwidth_config.W_, bitwidth_config.W_symmetric_,
                     quant_params.shift_W_tensor_, zp_tmp, "W");
    } else if (bitwidth_config.W_granularity_ == OperatorQuantConfig::PER_GATE) {
        // Per-gate: 直接使用独立的 per-gate 直方图
        int32_t zp_tmp;
        for (int gate = 0; gate < 3; ++gate) {
            const char* gate_names[] = {"W_gate_z", "W_gate_r", "W_gate_g"};
            histCalibrate(hist_collectors.W_gate_hist[gate], bitwidth_config.W_, bitwidth_config.W_symmetric_,
                         quant_params.shift_W_gate_[gate], zp_tmp, gate_names[gate]);
        }
    } else {  // PER_CHANNEL
        quant_params.shift_W_.resize(channel_size);
        for (int c = 0; c < channel_size; ++c) {
            int32_t zp_tmp;
            histCalibrate(hist_collectors.W_hist[c], bitwidth_config.W_,
                         bitwidth_config.W_symmetric_, quant_params.shift_W_[c], zp_tmp, "W");
        }
    }
    
    // R
    if (bitwidth_config.R_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        int32_t zp_tmp;
        histCalibrate(hist_collectors.R_tensor_hist, bitwidth_config.R_, bitwidth_config.R_symmetric_,
                     quant_params.shift_R_tensor_, zp_tmp, "R");
    } else if (bitwidth_config.R_granularity_ == OperatorQuantConfig::PER_GATE) {
        int32_t zp_tmp;
        for (int gate = 0; gate < 3; ++gate) {
            const char* gate_names[] = {"R_gate_z", "R_gate_r", "R_gate_g"};
            histCalibrate(hist_collectors.R_gate_hist[gate], bitwidth_config.R_, bitwidth_config.R_symmetric_,
                         quant_params.shift_R_gate_[gate], zp_tmp, gate_names[gate]);
        }
    } else {  // PER_CHANNEL
        quant_params.shift_R_.resize(channel_size);
        for (int c = 0; c < channel_size; ++c) {
            int32_t zp_tmp;
            histCalibrate(hist_collectors.R_hist[c], bitwidth_config.R_,
                         bitwidth_config.R_symmetric_, quant_params.shift_R_[c], zp_tmp, "R");
        }
    }
    
    // bw
    if (bitwidth_config.bw_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        int32_t zp_tmp;
        histCalibrate(hist_collectors.bw_tensor_hist, bitwidth_config.bw_, bitwidth_config.bw_symmetric_,
                     quant_params.shift_bw_tensor_, zp_tmp, "bw");
    } else if (bitwidth_config.bw_granularity_ == OperatorQuantConfig::PER_GATE) {
        int32_t zp_tmp;
        for (int gate = 0; gate < 3; ++gate) {
            const char* gate_names[] = {"bw_gate_z", "bw_gate_r", "bw_gate_g"};
            histCalibrate(hist_collectors.bw_gate_hist[gate], bitwidth_config.bw_, bitwidth_config.bw_symmetric_,
                         quant_params.shift_bw_gate_[gate], zp_tmp, gate_names[gate]);
        }
    } else {  // PER_CHANNEL
        quant_params.shift_bw_.resize(channel_size);
        for (int c = 0; c < channel_size; ++c) {
            int32_t zp_tmp;
            histCalibrate(hist_collectors.bw_hist[c], bitwidth_config.bw_,
                         bitwidth_config.bw_symmetric_, quant_params.shift_bw_[c], zp_tmp, "bw");
        }
    }
    
    // br
    if (bitwidth_config.br_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        int32_t zp_tmp;
        histCalibrate(hist_collectors.br_tensor_hist, bitwidth_config.br_, bitwidth_config.br_symmetric_,
                     quant_params.shift_br_tensor_, zp_tmp, "br");
    } else if (bitwidth_config.br_granularity_ == OperatorQuantConfig::PER_GATE) {
        int32_t zp_tmp;
        for (int gate = 0; gate < 3; ++gate) {
            const char* gate_names[] = {"br_gate_z", "br_gate_r", "br_gate_g"};
            histCalibrate(hist_collectors.br_gate_hist[gate], bitwidth_config.br_, bitwidth_config.br_symmetric_,
                         quant_params.shift_br_gate_[gate], zp_tmp, gate_names[gate]);
        }
    } else {  // PER_CHANNEL
        quant_params.shift_br_.resize(channel_size);
        for (int c = 0; c < channel_size; ++c) {
            int32_t zp_tmp;
            histCalibrate(hist_collectors.br_hist[c], bitwidth_config.br_,
                         bitwidth_config.br_symmetric_, quant_params.shift_br_[c], zp_tmp, "br");
        }
    }

    // 标量参数（不受 granularity 影响）
    histCalibrate(hist_collectors.x_hist, bitwidth_config.x_, bitwidth_config.x_symmetric_,
                  quant_params.shift_x_, quant_params.zp_x_, "scale_x");
    histCalibrate(hist_collectors.h_hist, bitwidth_config.h_, bitwidth_config.h_symmetric_,
                  quant_params.shift_h_, quant_params.zp_h_, "scale_h");
    histCalibrate(hist_collectors.Wx_hist, bitwidth_config.weight_ih_linear_, bitwidth_config.weight_ih_linear_symmetric_,
                  quant_params.shift_weight_ih_linear_, quant_params.zp_weight_ih_linear_, "scale_weight_ih_linear");
    histCalibrate(hist_collectors.Rh_hist, bitwidth_config.weight_hh_linear_, bitwidth_config.weight_hh_linear_symmetric_,
                  quant_params.shift_weight_hh_linear_, quant_params.zp_weight_hh_linear_, "scale_weight_hh_linear");
    histCalibrate(hist_collectors.update_gate_input_hist, bitwidth_config.update_gate_input_, bitwidth_config.update_gate_input_symmetric_,
                  quant_params.shift_update_gate_input_, quant_params.zp_update_gate_input_, "scale_update_gate_input");
    histCalibrate(hist_collectors.reset_gate_input_hist, bitwidth_config.reset_gate_input_, bitwidth_config.reset_gate_input_symmetric_,
                  quant_params.shift_reset_gate_input_, quant_params.zp_reset_gate_input_, "scale_reset_gate_input");
    histCalibrate(hist_collectors.new_gate_input_hist, bitwidth_config.new_gate_input_, bitwidth_config.new_gate_input_symmetric_,
                  quant_params.shift_new_gate_input_, quant_params.zp_new_gate_input_, "scale_new_gate_input");
    histCalibrate(hist_collectors.update_gate_output_hist, bitwidth_config.update_gate_output_, bitwidth_config.update_gate_output_symmetric_,
                  quant_params.shift_update_gate_output_, quant_params.zp_update_gate_output_, "scale_update_gate_output");
    histCalibrate(hist_collectors.reset_gate_output_hist, bitwidth_config.reset_gate_output_, bitwidth_config.reset_gate_output_symmetric_,
                  quant_params.shift_reset_gate_output_, quant_params.zp_reset_gate_output_, "scale_reset_gate_output");
    histCalibrate(hist_collectors.new_gate_output_hist, bitwidth_config.new_gate_output_, bitwidth_config.new_gate_output_symmetric_,
                  quant_params.shift_new_gate_output_, quant_params.zp_new_gate_output_, "scale_new_gate_output");
    histCalibrate(hist_collectors.mul_reset_hidden_hist, bitwidth_config.mul_reset_hidden_, bitwidth_config.mul_reset_hidden_symmetric_,
                  quant_params.shift_mul_reset_hidden_, quant_params.zp_mul_reset_hidden_, "scale_mul_reset_hidden");
    histCalibrate(hist_collectors.mul_new_contribution_hist, bitwidth_config.mul_new_contribution_, bitwidth_config.mul_new_contribution_symmetric_,
                  quant_params.shift_mul_new_contribution_, quant_params.zp_mul_new_contribution_, "scale_mul_new_contribution");
    histCalibrate(hist_collectors.mul_old_contribution_hist, bitwidth_config.mul_old_contribution_, bitwidth_config.mul_old_contribution_symmetric_,
                  quant_params.shift_mul_old_contribution_, quant_params.zp_mul_old_contribution_, "scale_mul_old_contribution");

    generate_piecewise_linear_lut_to_params(quant_params);
    return quant_params;
}

/**
 * @brief 从 GPU 直方图收集器计算量化参数（GPU 加速 SQNR）
 *
 * 直接使用 GPU 上的直方图数据计算 SQNR，避免 GPU→CPU 传输
 */
GRUQuantParams calculateGRUQuantitativeParametersFromGPUHistograms(
    GRUGPUHistogramCollectors &gpu_collectors, const OperatorQuantConfig &bitwidth_config,
    bool verbose) {
    
    GRUQuantParams quant_params;
    quant_params.hidden_ = gpu_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;
    
    const int channel_size = gpu_collectors.hidden_ * 3;
    quant_params.shift_W_.resize(channel_size);
    quant_params.shift_R_.resize(channel_size);
    quant_params.shift_bw_.resize(channel_size);
    quant_params.shift_br_.resize(channel_size);
    
    // Helper: 计算单个标量直方图的 SQNR 参数
    auto compute_scalar_sqnr = [&](const GPUHistogramCollector& collector, 
                                    bool is_symmetric, QuantBitWidth quant_bw,
                                    int8_t& out_shift, int32_t& out_zp,
                                    const char* name) {
        if (!collector.is_valid()) {
            throw std::runtime_error(std::string("GPU histogram ") + (name ? name : "unknown") + " is invalid");
        }
        const auto& hist = collector.histogram();
        const int64_t num_steps = quant_bw.qmax_auto_scale() - quant_bw.qmin_auto_scale();
        const bool is_unsigned = quant_bw.is_unsigned_;
        
        // 步骤 1: GPU SQNR 搜索获取连续 scale
        ContinuousScaleResult continuous_result = gpu_hist::searchSqnrGpu(
            hist.counts.data(), hist.min_val, hist.max_val,
            hist.num_bins, num_steps, is_symmetric, SqnrConfig(), is_unsigned);
        
        // 步骤 2: CPU POT 转换（与 AIMET 一致，无位宽约束）
        PotScaleResult pot_result = convertToPot(
            continuous_result.scale, continuous_result.min,
            quant_bw, is_symmetric);
        
        out_shift = pot_result.exp2_inv;
        out_zp = pot_result.zero_point;
        
        if (verbose && name) {
            printf("[GPU-SQNR][%s] bits=%d unsigned=%d range=[%.4f,%.4f] shift=%d zp=%d\n",
                   name, quant_bw.bits_, is_unsigned, hist.min_val, hist.max_val, out_shift, out_zp);
        }
    };
    
    // 标量直方图
    compute_scalar_sqnr(gpu_collectors.x_hist, bitwidth_config.x_symmetric_, 
                        bitwidth_config.x_, quant_params.shift_x_, quant_params.zp_x_, "x");
    compute_scalar_sqnr(gpu_collectors.h_hist, bitwidth_config.h_symmetric_,
                        bitwidth_config.h_, quant_params.shift_h_, quant_params.zp_h_, "h");
    compute_scalar_sqnr(gpu_collectors.Wx_hist, bitwidth_config.weight_ih_linear_symmetric_,
                        bitwidth_config.weight_ih_linear_, quant_params.shift_weight_ih_linear_, quant_params.zp_weight_ih_linear_, "weight_ih_linear");
    compute_scalar_sqnr(gpu_collectors.Rh_hist, bitwidth_config.weight_hh_linear_symmetric_,
                        bitwidth_config.weight_hh_linear_, quant_params.shift_weight_hh_linear_, quant_params.zp_weight_hh_linear_, "weight_hh_linear");
    compute_scalar_sqnr(gpu_collectors.update_gate_input_hist, bitwidth_config.update_gate_input_symmetric_,
                        bitwidth_config.update_gate_input_, quant_params.shift_update_gate_input_, quant_params.zp_update_gate_input_, "update_gate_input");
    compute_scalar_sqnr(gpu_collectors.reset_gate_input_hist, bitwidth_config.reset_gate_input_symmetric_,
                        bitwidth_config.reset_gate_input_, quant_params.shift_reset_gate_input_, quant_params.zp_reset_gate_input_, "reset_gate_input");
    compute_scalar_sqnr(gpu_collectors.new_gate_input_hist, bitwidth_config.new_gate_input_symmetric_,
                        bitwidth_config.new_gate_input_, quant_params.shift_new_gate_input_, quant_params.zp_new_gate_input_, "new_gate_input");
    compute_scalar_sqnr(gpu_collectors.update_gate_output_hist, bitwidth_config.update_gate_output_symmetric_,
                        bitwidth_config.update_gate_output_, quant_params.shift_update_gate_output_, quant_params.zp_update_gate_output_, "update_gate_output");
    compute_scalar_sqnr(gpu_collectors.reset_gate_output_hist, bitwidth_config.reset_gate_output_symmetric_,
                        bitwidth_config.reset_gate_output_, quant_params.shift_reset_gate_output_, quant_params.zp_reset_gate_output_, "reset_gate_output");
    compute_scalar_sqnr(gpu_collectors.new_gate_output_hist, bitwidth_config.new_gate_output_symmetric_,
                        bitwidth_config.new_gate_output_, quant_params.shift_new_gate_output_, quant_params.zp_new_gate_output_, "new_gate_output");
    compute_scalar_sqnr(gpu_collectors.mul_reset_hidden_hist, bitwidth_config.mul_reset_hidden_symmetric_,
                        bitwidth_config.mul_reset_hidden_, quant_params.shift_mul_reset_hidden_, quant_params.zp_mul_reset_hidden_, "mul_reset_hidden");
    compute_scalar_sqnr(gpu_collectors.mul_new_contribution_hist, bitwidth_config.mul_new_contribution_symmetric_,
                        bitwidth_config.mul_new_contribution_, quant_params.shift_mul_new_contribution_,
                        quant_params.zp_mul_new_contribution_, "mul_new_contribution");
    compute_scalar_sqnr(gpu_collectors.mul_old_contribution_hist, bitwidth_config.mul_old_contribution_symmetric_,
                        bitwidth_config.mul_old_contribution_, quant_params.shift_mul_old_contribution_,
                        quant_params.zp_mul_old_contribution_, "mul_old_contribution");
    
    // 根据粒度配置计算权重和偏置的量化参数
    // 辅助函数：计算 per-tensor 参数
    auto compute_tensor_sqnr = [&](const GPUHistogramCollector& collector,
                                    bool is_symmetric, QuantBitWidth quant_bw,
                                    int8_t& out_shift, const char* name) {
        if (!collector.is_valid()) {
            fprintf(stderr, "Warning: %s tensor histogram is invalid, shift parameter will remain 0\n", name);
            return;
        }
        const auto& hist = collector.histogram();
        const int64_t num_steps = quant_bw.qmax_auto_scale() - quant_bw.qmin_auto_scale();
        const bool is_unsigned = quant_bw.is_unsigned_;
        
        ContinuousScaleResult continuous_result = gpu_hist::searchSqnrGpu(
            hist.counts.data(), hist.min_val, hist.max_val,
            hist.num_bins, num_steps, is_symmetric, SqnrConfig(), is_unsigned);
        
        PotScaleResult pot_result = convertToPot(
            continuous_result.scale, continuous_result.min,
            quant_bw, is_symmetric);
        out_shift = pot_result.exp2_inv;
    };
    
    // 辅助函数：计算 per-gate 参数
    auto compute_gate_sqnr = [&](const std::array<GPUHistogramCollector, 3>& gate_collectors,
                                  bool is_symmetric, QuantBitWidth quant_bw,
                                  std::array<int8_t, 3>& out_gate,
                                  const char* name) {
        const char* gate_names[] = {"z", "r", "g"};
        for (int gate = 0; gate < 3; ++gate) {
            if (!gate_collectors[gate].is_valid()) {
                fprintf(stderr, "Warning: %s gate[%s] histogram is invalid, shift parameter will remain 0\n", 
                        name, gate_names[gate]);
                continue;
            }
            const auto& hist = gate_collectors[gate].histogram();
            const int64_t num_steps = quant_bw.qmax_auto_scale() - quant_bw.qmin_auto_scale();
            const bool is_unsigned = quant_bw.is_unsigned_;
            
            ContinuousScaleResult continuous_result = gpu_hist::searchSqnrGpu(
                hist.counts.data(), hist.min_val, hist.max_val,
                hist.num_bins, num_steps, is_symmetric, SqnrConfig(), is_unsigned);
            
            PotScaleResult pot_result = convertToPot(
                continuous_result.scale, continuous_result.min,
                quant_bw, is_symmetric);
            
            out_gate[gate] = pot_result.exp2_inv;
        }
    };
    
    // 辅助函数：计算 per-channel 参数
    auto compute_per_channel_sqnr = [&](const PerChannelHistogramBatch& batch,
                                         bool is_symmetric, QuantBitWidth quant_bw,
                                         std::vector<int8_t>& out_shift) {
        if (!batch.is_valid()) return;
        
        const int64_t num_steps = quant_bw.qmax_auto_scale() - quant_bw.qmin_auto_scale();
        const bool is_unsigned = quant_bw.is_unsigned_;
        
        // 步骤 1: GPU SQNR 搜索获取连续 scale
        std::vector<ContinuousScaleResult> continuous_results;
        gpu_hist::searchSqnrPerChannelGpu(batch, num_steps, is_symmetric, continuous_results,
                                           SqnrConfig(), is_unsigned);
        
        // 步骤 2: CPU POT 转换（与 AIMET 一致，无位宽约束）
        out_shift.resize(batch.channel_size);
        for (int c = 0; c < batch.channel_size; ++c) {
            PotScaleResult pot_result = convertToPot(
                continuous_results[c].scale, continuous_results[c].min,
                quant_bw, is_symmetric);
            out_shift[c] = pot_result.exp2_inv;
        }
    };
    
    // W
    if (bitwidth_config.W_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        compute_tensor_sqnr(gpu_collectors.W_tensor_hist, bitwidth_config.W_symmetric_,
                            bitwidth_config.W_, quant_params.shift_W_tensor_, "W");
    } else if (bitwidth_config.W_granularity_ == OperatorQuantConfig::PER_GATE) {
        compute_gate_sqnr(gpu_collectors.W_gate_hist, bitwidth_config.W_symmetric_,
                          bitwidth_config.W_, quant_params.shift_W_gate_, "W");
    } else {  // PER_CHANNEL
        compute_per_channel_sqnr(gpu_collectors.W_batch, bitwidth_config.W_symmetric_,
                                 bitwidth_config.W_, quant_params.shift_W_);
    }
    
    // R
    if (bitwidth_config.R_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        compute_tensor_sqnr(gpu_collectors.R_tensor_hist, bitwidth_config.R_symmetric_,
                            bitwidth_config.R_, quant_params.shift_R_tensor_, "R");
    } else if (bitwidth_config.R_granularity_ == OperatorQuantConfig::PER_GATE) {
        compute_gate_sqnr(gpu_collectors.R_gate_hist, bitwidth_config.R_symmetric_,
                          bitwidth_config.R_, quant_params.shift_R_gate_, "R");
    } else {  // PER_CHANNEL
        compute_per_channel_sqnr(gpu_collectors.R_batch, bitwidth_config.R_symmetric_,
                                 bitwidth_config.R_, quant_params.shift_R_);
    }
    
    // bw
    if (bitwidth_config.bw_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        compute_tensor_sqnr(gpu_collectors.bw_tensor_hist, bitwidth_config.bw_symmetric_,
                            bitwidth_config.bw_, quant_params.shift_bw_tensor_, "bw");
    } else if (bitwidth_config.bw_granularity_ == OperatorQuantConfig::PER_GATE) {
        compute_gate_sqnr(gpu_collectors.bw_gate_hist, bitwidth_config.bw_symmetric_,
                          bitwidth_config.bw_, quant_params.shift_bw_gate_, "bw");
    } else {  // PER_CHANNEL
        compute_per_channel_sqnr(gpu_collectors.bw_batch, bitwidth_config.bw_symmetric_,
                                 bitwidth_config.bw_, quant_params.shift_bw_);
    }
    
    // br
    if (bitwidth_config.br_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        compute_tensor_sqnr(gpu_collectors.br_tensor_hist, bitwidth_config.br_symmetric_,
                            bitwidth_config.br_, quant_params.shift_br_tensor_, "br");
    } else if (bitwidth_config.br_granularity_ == OperatorQuantConfig::PER_GATE) {
        compute_gate_sqnr(gpu_collectors.br_gate_hist, bitwidth_config.br_symmetric_,
                          bitwidth_config.br_, quant_params.shift_br_gate_, "br");
    } else {  // PER_CHANNEL
        compute_per_channel_sqnr(gpu_collectors.br_batch, bitwidth_config.br_symmetric_,
                                 bitwidth_config.br_, quant_params.shift_br_);
    }
    
    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 生成 LUT 并存储到参数中
    generate_piecewise_linear_lut_to_params(quant_params);
    
    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in calculateGRUQuantitativeParametersFromGPUHistograms: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error: ") + err_str);
    }
    
    return quant_params;
}

// Percentile 校准：使用 calculateGRUQuantitativeParametersFromHistograms() 并设置 use_percentile=true

// =====================================================================
// 统一校准前向传播（GPU）
// =====================================================================

void forwardWithCalibrationGPU(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    CalibrationMethod calib_method,
    GRUQuantizationRanges *quant_ranges,
    GRUGPUHistogramCollectors *gpu_hist_collectors,
    const OperatorQuantConfig &bitwidth_config,
    float *h, float *v) {
    
    // 参数校验
    if (calib_method == CalibrationMethod::MINMAX) {
        if (!quant_ranges) {
            throw std::invalid_argument("quant_ranges is required for MINMAX calibration");
        }
    } else if (calib_method == CalibrationMethod::SQNR || 
               calib_method == CalibrationMethod::PERCENTILE) {
        if (!gpu_hist_collectors) {
            throw std::invalid_argument("gpu_hist_collectors is required for SQNR/PERCENTILE calibration");
        }
        // 检查/重置直方图收集器
        if (gpu_hist_collectors->hidden_ != hidden_size) {
            gpu_hist_collectors->reset(hidden_size);
        }
    } else {
        throw std::invalid_argument("Invalid calibration method (must be MINMAX, SQNR, or PERCENTILE)");
    }
    
    // ========== 公共部分 ==========
    
    // 分配临时缓冲区
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);

    // 处理初始隐藏状态
    const int NH = batch_size * hidden_size;
    if (h0 != nullptr) {
        // 如果提供了初始状态，复制到 h[0]
        d2d(h, h0, NH);
    } else {
        // 否则初始化为零
        cudaMemset(h, 0, NH * sizeof(float));
    }

    // 创建 ForwardPass 对象
    gru::ForwardPass<float> forward(is_training, batch_size, input_size, hidden_size, g_blas_handle);

    // 设置校准模式
    forward.setCalibrationMode(true);

    // 执行前向传播
    forward.Run(time_steps, W, R, bw, br, x, h, v, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in forwardWithCalibrationGPU: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in forwardWithCalibrationGPU: ") + err_str);
    }
    
    // ========== 后处理（根据校准方法分发）==========
    
    if (calib_method == CalibrationMethod::MINMAX) {
        // MINMAX: 使用 GPU 版本原地更新量化范围（避免大量 D2H 传输）
        // 使用 Wx+bw 和 Rh+br 的结果（而非纯 GEMM 输出）进行校准
        updateGRUQuantizationRangesGPU(
            time_steps, batch_size, input_size, hidden_size,
            W, R, bw, br, x, h, v,
            forward.getWxAddBw(), forward.getRhAddBr(),
            forward.getZPres(), forward.getRPres(), forward.getGPres(),
            forward.getPresSize(),
            *quant_ranges);
    } else {
        // SQNR/Percentile: 收集直方图
        // 使用 Wx+bw 和 Rh+br 的结果（而非纯 GEMM 输出）进行直方图收集
        collectAllHistogramsGPU(*gpu_hist_collectors, x, h, v,
                                forward.getWxAddBw(), forward.getRhAddBr(),
                                W, R, bw, br,
                                time_steps, batch_size, input_size, hidden_size,
                                forward.getZPres(), forward.getRPres(), forward.getGPres(),
                                forward.getPresSize(), bitwidth_config);
    }
}

