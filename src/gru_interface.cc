// =====================================================================
// GRU 接口层实现 (gru_interface.cpp)
// =====================================================================

#include "gru_interface.h"

#include <cuda_runtime.h>
#include <omp.h>

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

GRUQuantitativeParameters calculateGRUQuantitativeParameters(
    const GRUQuantizationRanges &quant_ranges, const OperatorQuantConfig &bitwidth_config) {
    GRUQuantitativeParameters quant_params;
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

    // 权重 W/R 和偏置 bw/br（per-channel）
    const int channel_size = quant_ranges.hidden_ * 3;
    quant_params.shift_W_.resize(channel_size);
    quant_params.shift_R_.resize(channel_size);
    quant_params.shift_bw_.resize(channel_size);
    quant_params.shift_br_.resize(channel_size);

    for (int c = 0; c < channel_size; ++c) {
        float aligned_min, aligned_max;
        int32_t zp_tmp;
        calibrateQuantParams(quant_ranges.min_W_[c], quant_ranges.max_W_[c], bitwidth_config.W_,
                             bitwidth_config.W_symmetric_, aligned_min, aligned_max,
                             quant_params.shift_W_[c], zp_tmp);
        calibrateQuantParams(quant_ranges.min_R_[c], quant_ranges.max_R_[c], bitwidth_config.R_,
                             bitwidth_config.R_symmetric_, aligned_min, aligned_max,
                             quant_params.shift_R_[c], zp_tmp);
        calibrateQuantParams(quant_ranges.min_bw_[c], quant_ranges.max_bw_[c], bitwidth_config.bw_,
                             bitwidth_config.bw_symmetric_, aligned_min, aligned_max,
                             quant_params.shift_bw_[c], zp_tmp);
        calibrateQuantParams(quant_ranges.min_br_[c], quant_ranges.max_br_[c], bitwidth_config.br_,
                             bitwidth_config.br_symmetric_, aligned_min, aligned_max,
                             quant_params.shift_br_[c], zp_tmp);
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
// 前向传播实现
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
// 权重量化实现（统一 int32_t 输出）
// =====================================================================

void quantitativeWeight(const int input_size, const int hidden_size, const float *W, const float *R,
                        const float *bw, const float *br,
                        const GRUQuantitativeParameters &quant_parms, int32_t *W_quant,
                        int32_t *R_quant, int32_t *bw_quant, int32_t *br_quant) {
    // 显式创建dev::vector以避免临时对象问题
    dev::vector<int8_t> shift_W_dev(quant_parms.shift_W_);
    dev::vector<int8_t> shift_R_dev(quant_parms.shift_R_);
    dev::vector<int8_t> shift_bw_dev(quant_parms.shift_bw_);
    dev::vector<int8_t> shift_br_dev(quant_parms.shift_br_);

    // 统一 int32_t 输出，使用 clamp_by_bitwidth 限制到实际位宽
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    dev::quantificationPerChannelBitwidth(W, W_quant, input_size, 3 * hidden_size, 
                                           shift_W_dev, bw_cfg.W_);
    dev::quantificationPerChannelBitwidth(R, R_quant, hidden_size, 3 * hidden_size, 
                                           shift_R_dev, bw_cfg.R_);
    dev::quantificationPerChannelBitwidth(bw, bw_quant, 1, 3 * hidden_size, 
                                           shift_bw_dev, bw_cfg.bw_);
    dev::quantificationPerChannelBitwidth(br, br_quant, 1, 3 * hidden_size, 
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

// 量化 GRU 前向传播（统一 int32_t 存储）
// 所有量化值使用 int32_t 存储，实际值通过位宽配置限制
void quantGRUForward(bool is_training, const int time_steps, const int batch_size,
                     const int input_size, const int hidden_size, const int32_t *W,
                     const int32_t *R, const int32_t *bw, const int32_t *br, const float *x,
                     const float *h0, const GRUQuantitativeParameters &quant_parms,
                     const cublasHandle_t &g_blas_handle, float *h, float *v) {
    const std::size_t x_size = time_steps * batch_size * input_size;

    // 所有激活值使用 int32_t 存储，通过 clamp_by_bitwidth 限制到实际位宽
    const auto &bw_cfg = quant_parms.bitwidth_config_;
    dev::vector<int32_t> x_quant(x_size);
    dev::quantificationBitwidth(x, x_quant.data(), x_size, quant_parms.shift_x_, 
                                 quant_parms.zp_x_, bw_cfg.x_);

    dev::vector<int32_t> h_quant((time_steps + 1) * batch_size * hidden_size);
    // 初始化 h0 区域（第一个时间步的隐藏状态）为零点值
    dev::fill_n(h_quant.data(), batch_size * hidden_size, quant_parms.zp_h_);

    // 处理初始隐藏状态
    if (h0 != nullptr) {
        // 如果提供了初始状态，直接量化到 h_quant[0]
        dev::quantificationBitwidth(h0, h_quant.data(), batch_size * hidden_size, 
                                     quant_parms.shift_h_, quant_parms.zp_h_, bw_cfg.h_);
    }

    dev::vector<int32_t> v_quant_dev(time_steps * batch_size * hidden_size *
                                     4);  // v 统一使用 int32_t 存储

    // 非模板化的 ForwardPassQuant 类
    gru::ForwardPassQuant forward(
            is_training,  // training: true为训练，false为推理
            batch_size, input_size, hidden_size, g_blas_handle);

    // 得到量化GRU中使用的rescale参数
    forward.setRescaleParam(quant_parms);

    forward.Run(time_steps, W, R, bw, br, x_quant.data(), h_quant.data(), v_quant_dev.data(), 0.0f,
                nullptr);

    dev::dequantification(h_quant.data(), h, (time_steps + 1) * batch_size * hidden_size,
                          quant_parms.shift_h_, quant_parms.zp_h_);

    // 如果v不为nullptr，反量化v并输出
    // V 向量布局: [z_out, r_out, g_out, weight_hh_linear_g]
    if (v != nullptr) {
        dev::dequantificationV(v_quant_dev.data(), v, time_steps, batch_size, hidden_size,
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
                        const GRUQuantitativeParameters &quant_parms, float *h, float *v) {
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
                        const GRUQuantitativeParameters &quant_parms, float *h, float *v) {
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

// 统一前向传播接口（推理/训练）
// 所有量化值使用 int32_t 存储，实际位宽通过 bitwidth_config_ 控制
// 注意：校准请使用 forwardWithCalibrationMinMaxGPU 或 forwardWithCalibrationHistogramGPU
void forwardInterface(bool is_training, bool is_quant, int time_steps, int batch_size,
                      int input_size, int hidden_size, const float *W, const float *R,
                      const float *bw, const float *br, const float *x, const float *h0,
                      const GRUQuantitativeParameters &quant_gru_scales,
                      const cublasHandle_t &g_blas_handle,
                      float *h, float *v) {
    if (is_quant) {
        // 所有权重和激活统一使用 int32_t 存储
        dev::vector<int32_t> W_quant(hidden_size * 3 * input_size);
        dev::vector<int32_t> R_quant(hidden_size * 3 * hidden_size);
        dev::vector<int32_t> bw_quant(hidden_size * 3);
        dev::vector<int32_t> br_quant(hidden_size * 3);

        // 量化权重（统一输出 int32_t）
        quantitativeWeight(input_size, hidden_size, W, R, bw, br, quant_gru_scales,
                           W_quant.data(), R_quant.data(), bw_quant.data(), br_quant.data());
        
        // 调用统一的量化前向传播（非模板）
        quantGRUForward(is_training, time_steps, batch_size, input_size,
                        hidden_size, W_quant.data(), R_quant.data(),
                        bw_quant.data(), br_quant.data(), x, h0,
                        quant_gru_scales, g_blas_handle, h, v);
    } else {
        hasteGRUForward(is_training, time_steps, batch_size, input_size, hidden_size, W, R, bw, br,
                        x, h0, g_blas_handle, h, v);
    }
}

// =====================================================================
// 注：模板已移除，所有量化函数现在使用统一的 int32_t 存储
// quantitativeWeight 和 quantGRUForward 现在是普通函数（非模板）
// =====================================================================

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

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in convertGPUHistogramsToCPU: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in convertGPUHistogramsToCPU: ") + err_str);
    }

    return cpu_collectors;
}

GRUQuantitativeParameters calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors &hist_collectors, const OperatorQuantConfig &bitwidth_config,
    bool verbose, bool use_percentile, float percentile_value) {
    GRUQuantitativeParameters quant_params;
    quant_params.hidden_ = hist_collectors.hidden_;
    quant_params.bitwidth_config_ = bitwidth_config;

    const int channel_size = hist_collectors.hidden_ * 3;
    
    quant_params.shift_W_.resize(channel_size);
    quant_params.shift_R_.resize(channel_size);
    quant_params.shift_bw_.resize(channel_size);
    quant_params.shift_br_.resize(channel_size);

    // 辅助 lambda：校准单个直方图
    auto histCalibrate = [&](const HistogramCollector& hist, QuantBitWidth bw, bool sym,
                             int8_t& shift, int32_t& zp, const char* name) {
        if (!hist.is_valid()) throw std::runtime_error(std::string(name) + " is invalid.");
        calibrateQuantParamsFromHistogram(hist.histogram(), bw, sym, shift, zp,
                                          verbose ? name : nullptr, use_percentile, percentile_value);
    };

    // OpenMP 并行化 per-channel 计算
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        int32_t zp_tmp;
        
        if (tid == 0) {
            for (int c = 0; c < channel_size; ++c) {
                histCalibrate(hist_collectors.W_hist[c], bitwidth_config.W_,
                              bitwidth_config.W_symmetric_, quant_params.shift_W_[c], zp_tmp, "W");
            }
        } else if (tid == 1) {
            for (int c = 0; c < channel_size; ++c) {
                histCalibrate(hist_collectors.R_hist[c], bitwidth_config.R_,
                              bitwidth_config.R_symmetric_, quant_params.shift_R_[c], zp_tmp, "R");
            }
        } else if (tid == 2) {
            for (int c = 0; c < channel_size; ++c) {
                histCalibrate(hist_collectors.bw_hist[c], bitwidth_config.bw_,
                              bitwidth_config.bw_symmetric_, quant_params.shift_bw_[c], zp_tmp, "bw");
            }
        } else {
            for (int c = 0; c < channel_size; ++c) {
                histCalibrate(hist_collectors.br_hist[c], bitwidth_config.br_,
                              bitwidth_config.br_symmetric_, quant_params.shift_br_[c], zp_tmp, "br");
            }
            
            // 标量参数
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
            // Rh_add_br_g 已废弃，使用 weight_hh_linear 的量化参数
            histCalibrate(hist_collectors.mul_reset_hidden_hist, bitwidth_config.mul_reset_hidden_, bitwidth_config.mul_reset_hidden_symmetric_,
                          quant_params.shift_mul_reset_hidden_, quant_params.zp_mul_reset_hidden_, "scale_mul_reset_hidden");
            histCalibrate(hist_collectors.mul_new_contribution_hist, bitwidth_config.mul_new_contribution_, bitwidth_config.mul_new_contribution_symmetric_,
                          quant_params.shift_mul_new_contribution_, quant_params.zp_mul_new_contribution_, "scale_mul_new_contribution");
            histCalibrate(hist_collectors.mul_old_contribution_hist, bitwidth_config.mul_old_contribution_, bitwidth_config.mul_old_contribution_symmetric_,
                          quant_params.shift_mul_old_contribution_, quant_params.zp_mul_old_contribution_, "scale_mul_old_contribution");
        }
    }

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
        const int64_t num_steps = quant_bw.qmax() - quant_bw.qmin();
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
    // Rh_add_br_g 已废弃，使用 weight_hh_linear 的量化参数
    compute_scalar_sqnr(gpu_collectors.mul_reset_hidden_hist, bitwidth_config.mul_reset_hidden_symmetric_,
                        bitwidth_config.mul_reset_hidden_, quant_params.shift_mul_reset_hidden_, quant_params.zp_mul_reset_hidden_, "mul_reset_hidden");
    compute_scalar_sqnr(gpu_collectors.mul_new_contribution_hist, bitwidth_config.mul_new_contribution_symmetric_,
                        bitwidth_config.mul_new_contribution_, quant_params.shift_mul_new_contribution_,
                        quant_params.zp_mul_new_contribution_, "mul_new_contribution");
    compute_scalar_sqnr(gpu_collectors.mul_old_contribution_hist, bitwidth_config.mul_old_contribution_symmetric_,
                        bitwidth_config.mul_old_contribution_, quant_params.shift_mul_old_contribution_,
                        quant_params.zp_mul_old_contribution_, "mul_old_contribution");
    
    // Per-channel 直方图（使用批量 GPU SQNR + CPU POT 转换）
    auto compute_per_channel_sqnr = [&](const PerChannelHistogramBatch& batch,
                                         bool is_symmetric, QuantBitWidth quant_bw,
                                         std::vector<int8_t>& out_shift) {
        if (!batch.is_valid()) return;
        
        const int64_t num_steps = quant_bw.qmax() - quant_bw.qmin();
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
    
    compute_per_channel_sqnr(gpu_collectors.W_batch, bitwidth_config.W_symmetric_, 
                             bitwidth_config.W_, quant_params.shift_W_);
    compute_per_channel_sqnr(gpu_collectors.R_batch, bitwidth_config.R_symmetric_,
                             bitwidth_config.R_, quant_params.shift_R_);
    compute_per_channel_sqnr(gpu_collectors.bw_batch, bitwidth_config.bw_symmetric_,
                             bitwidth_config.bw_, quant_params.shift_bw_);
    compute_per_channel_sqnr(gpu_collectors.br_batch, bitwidth_config.br_symmetric_,
                             bitwidth_config.br_, quant_params.shift_br_);
    
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
                                forward.getPresSize());
    }
}

// =====================================================================
// CPU 直方图收集（用于性能对比）
// =====================================================================

void forwardWithHistogramCPU(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    GRUHistogramCollectors *hist_collectors,
    float *h, float *v) {
    
    if (!hist_collectors) {
        throw std::invalid_argument("hist_collectors is required for CPU histogram calibration");
    }
    
    // 分配临时缓冲区
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);

    // 处理初始隐藏状态
    const int NH = batch_size * hidden_size;
    if (h0 != nullptr) {
        d2d(h, h0, NH);
    } else {
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
        fprintf(stderr, "CUDA error in forwardWithHistogramCPU: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in forwardWithHistogramCPU: ") + err_str);
    }
    
    // CPU 直方图收集（需要先将 GPU 数据拷贝到 CPU）
    // 使用 Wx+bw 和 Rh+br 的结果（而非纯 GEMM 输出）进行直方图收集
    collectAllHistograms(*hist_collectors, x, h, v,
                         forward.getWxAddBw(), forward.getRhAddBr(),
                         W, R, bw, br,
                         time_steps, batch_size, input_size, hidden_size,
                         forward.getZPres(), forward.getRPres(), forward.getGPres(),
                         forward.getPresSize());
}
