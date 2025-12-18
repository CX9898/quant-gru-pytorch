// =====================================================================
// GRU 接口层 (gru_interface.hpp)
// =====================================================================
// 提供 GRU 的前向传播、反向传播、量化校准等统一接口。
// 包含浮点和量化两种实现，支持训练和推理模式。
//
// 维度约定:
//   T = time_steps (序列长度)
//   B = batch_size (批大小)
//   I = input_size (输入维度)
//   H = hidden_size (隐藏层维度)
//   C = input_size (与 I 相同，用于权重矩阵描述)
// =====================================================================

#pragma once

#include <cublas_v2.h>

#include <vector>

#include "gru.h"
#include "gru_quant.h"
#include "gru_quantization_ranges.hpp"

// =====================================================================
// cuBLAS 初始化
// =====================================================================

// 初始化 cuBLAS 句柄（供 Python 绑定调用）
inline void init_gru_cublas(cublasHandle_t &g_blas_handle) {
    if (g_blas_handle == nullptr) {
        cublasCreate(&g_blas_handle);
    }
}

// =====================================================================
// 量化校准接口
// =====================================================================

// 校准 GRU 量化范围（收集 min/max 数据分布）
// 输入:
//   W:  [C, H*3]   输入权重矩阵
//   R:  [H, H*3]   循环权重矩阵
//   bx: [H*3]      输入偏置
//   br: [H*3]      循环偏置
//   x:  [T, B, I]  输入序列
// 输出:
//   quant_ranges: 收集到的量化范围
void calibrateGruRanges(int time_steps, int batch_size, int input_size, int hidden_size,
                        const float *W, const float *R, const float *bx, const float *br,
                        const float *x, const cublasHandle_t &g_blas_handle,
                        GRUQuantizationRanges &quant_ranges);

// 根据量化范围和位宽配置计算量化参数（scale 和 zero point）
GRUQuantitativeParameters calculateGRUQuantitativeParameters(
    const GRUQuantizationRanges &quant_ranges,
    const OperatorQuantConfig &bitwidth_config = OperatorQuantConfig());

// =====================================================================
// AIMET 风格的真正直方图校准接口（多批次累积直方图 + SQNR 优化）
// =====================================================================

// 前向声明直方图收集器
struct GRUHistogramCollectors;

// 收集直方图数据（支持多批次累积）
// 输入:
//   W:  [C, H*3]   输入权重矩阵
//   R:  [H, H*3]   循环权重矩阵
//   bx: [H*3]      输入偏置
//   br: [H*3]      循环偏置
//   x:  [T, B, I]  输入序列
// 输出:
//   hist_collectors: 收集到的直方图（会累积更新）
void calibrateGruHistograms(int time_steps, int batch_size, int input_size, int hidden_size,
                            const float *W, const float *R, const float *bx, const float *br,
                            const float *x, const cublasHandle_t &g_blas_handle,
                            GRUHistogramCollectors &hist_collectors);

// 从直方图计算量化参数（真正的 AIMET 风格 SQNR 优化）
GRUQuantitativeParameters calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors &hist_collectors,
    const OperatorQuantConfig &bitwidth_config = OperatorQuantConfig(),
    bool verbose = false);

// 初始化量化 LUT 表
// 根据 bitwidth_config_ 自动选择相应的 LUT 初始化方法
void initialize_quantization_lut(const GRUQuantitativeParameters &quant_params);

// =====================================================================
// 权重量化接口
// =====================================================================

// 量化权重矩阵和偏置
// 输入（浮点）:
//   W:  [C, H*3]   输入权重矩阵
//   R:  [H, H*3]   循环权重矩阵
//   bx: [H*3]      输入偏置
//   br: [H*3]      循环偏置
// 输出（量化）:
//   W_quant:  [C, H*3]   量化后的输入权重
//   R_quant:  [H, H*3]   量化后的循环权重
//   bx_quant: [H*3]      量化后的输入偏置（int32）
//   br_quant: [H*3]      量化后的循环偏置（int32）
template <typename QuantT>
void quantitativeWeight(const int input_size, const int hidden_size,
                        const float *W, const float *R, const float *bx, const float *br,
                        const GRUQuantitativeParameters &quant_parms,
                        QuantT *W_quant, QuantT *R_quant, int32_t *bx_quant, int32_t *br_quant);

// =====================================================================
// 前向传播接口
// =====================================================================

// 量化 GRU 前向传播
// 输入（量化权重）:
//   W:  [C, H*3]   量化后的输入权重
//   R:  [H, H*3]   量化后的循环权重
//   bx: [H*3]      量化后的输入偏置
//   br: [H*3]      量化后的循环偏置
//   x:  [T, B, I]  浮点输入序列（内部会量化）
//   h0: [B, H]     初始隐藏状态（可为 nullptr）
// 输出:
//   h:  [(T+1), B, H]  所有时间步的隐藏状态（包含 h0）
//   v:  [T, B, H*4]    中间值（训练时需要，推理时可为 nullptr）
template <typename QuantT>
void quantGRUForward(
    bool is_training,
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const QuantT *W, const QuantT *R, const int32_t *bx, const int32_t *br, const float *x,
    const float *h0,
    const GRUQuantitativeParameters &quant_parms, const cublasHandle_t &g_blas_handle,
    float *h, float *v);

// 浮点 GRU 前向传播
// 输入:
//   W:  [C, H*3]   输入权重矩阵
//   R:  [H, H*3]   循环权重矩阵
//   bx: [H*3]      输入偏置
//   br: [H*3]      循环偏置
//   x:  [T, B, I]  输入序列
//   h0: [B, H]     初始隐藏状态（可为 nullptr）
// 输出:
//   h:  [(T+1), B, H]  所有时间步的隐藏状态（包含 h0）
//   v:  [T, B, H*4]    中间值（训练时需要，推理时可为 nullptr）
void hasteGRUForward(
    bool is_training,
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W, const float *R, const float *bx, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    float *h, float *v);

// 统一前向传播接口（根据 is_quant 自动选择浮点或量化实现）
void forwardInterface(
    bool is_training, bool is_quant,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bx, const float *br, const float *x,
    const float *h0,
    const GRUQuantitativeParameters &quant_gru_scales, const cublasHandle_t &g_blas_handle,
    float *h, float *v);

// =====================================================================
// 反向传播接口
// =====================================================================

// 浮点 GRU 反向传播
//
// ★★★ 重要：W、R、x 需要传入【转置后】的数据！★★★
//
// 输入（转置后的数据）:
//   W_t: [H*3, C]      转置后的输入权重（原 W 是 [C, H*3]）
//   R_t: [H*3, H]      转置后的循环权重（原 R 是 [H, H*3]）
//   bx:  [H*3]         输入偏置（不需要转置）
//   br:  [H*3]         循环偏置（不需要转置）
//   x_t: [I, T, B]     转置后的输入序列（原 x 是 [T, B, I]）
//   dh_new: [(T+1), B, H]  上游梯度
//   h:      [(T+1), B, H]  前向传播保存的隐藏状态
//   v:      [T, B, H*4]    前向传播保存的中间值
//
// 输出（梯度）:
//   dx:  [T, B, I]     输入序列梯度
//   dW:  [C, H*3]      输入权重梯度（注意：输出格式与输入 W_t 不同！）
//   dR:  [H, H*3]      循环权重梯度（注意：输出格式与输入 R_t 不同！）
//   dbx: [H*3]         输入偏置梯度
//   dbr: [H*3]         循环偏置梯度
//   dh:  [B, H]        初始隐藏状态梯度
void hasteGRUBackward(
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W_t, const float *R_t,
    const float *bx, const float *br,
    const float *x_t,
    const float *dh_new,
    const float *h, const float *v,
    const cublasHandle_t &g_blas_handle,
    float *dx, float *dW, float *dR, float *dbx, float *dbr, float *dh);
