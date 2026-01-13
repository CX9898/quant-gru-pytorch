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

#include "calibration_utils.h"
#include "gru.h"
#include "gru_quant.h"
#include "gru_quantization_ranges.h"

// =====================================================================
// 校准方法枚举
// =====================================================================

enum class CalibrationMethod : int8_t {
    NONE = 0,        // 不校准，正常 forward
    MINMAX = 1,      // 收集 min/max 范围
    SQNR = 2,        // 收集直方图，使用 SQNR 优化
    PERCENTILE = 3   // 收集直方图，使用百分位裁剪
};

// 辅助函数：判断是否需要直方图
inline bool needsHistogram(CalibrationMethod method) {
    return method == CalibrationMethod::SQNR || 
           method == CalibrationMethod::PERCENTILE;
}

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
// 量化参数计算接口
// =====================================================================

// 根据量化范围和位宽配置计算量化参数（scale 和 zero point）
GRUQuantParams calculateGRUQuantitativeParameters(
    const GRUQuantizationRanges &quant_ranges,
    const OperatorQuantConfig &bitwidth_config = OperatorQuantConfig());

// 前向声明直方图收集器
struct GRUHistogramCollectors;
struct GRUGPUHistogramCollectors;  // GPU 版本

// 从直方图计算量化参数（支持 SQNR 和 Percentile 两种校准方案）
// use_percentile: false = SQNR (AIMET tf_enhanced), true = Percentile
// percentile_value: 仅 Percentile 方案使用，默认 99.99%
GRUQuantParams calculateGRUQuantitativeParametersFromHistograms(
    const GRUHistogramCollectors &hist_collectors,
    const OperatorQuantConfig &bitwidth_config = OperatorQuantConfig(),
    bool verbose = false,
    bool use_percentile = false,
    float percentile_value = 99.99f);

// =====================================================================
// 权重量化接口
// =====================================================================

// 量化权重矩阵和偏置（统一 int32_t 存储）
// 所有量化值使用 int32_t 存储，实际位宽通过 bitwidth_config_ 控制
// 输入（浮点）:
//   W:  [C, H*3]   输入权重矩阵
//   R:  [H, H*3]   循环权重矩阵
//   bw: [H*3]      输入偏置 (bias for W)
//   br: [H*3]      循环偏置
// 输出（量化，int32_t 存储）:
//   W_quant:  [C, H*3]   量化后的输入权重
//   R_quant:  [H, H*3]   量化后的循环权重
//   bw_quant: [H*3]      量化后的输入偏置
//   br_quant: [H*3]      量化后的循环偏置
void quantitativeWeight(const int input_size, const int hidden_size,
                        const float *W, const float *R, const float *bw, const float *br,
                        const GRUQuantParams &quant_parms,
                        int32_t *W_quant, int32_t *R_quant, int32_t *bw_quant, int32_t *br_quant);

// =====================================================================
// GPU 量化 GRU 前向传播接口
// =====================================================================

// GPU 量化 GRU 前向传播（统一 int32_t 存储）
// 所有量化值使用 int32_t 存储，实际位宽通过 bitwidth_config_ 控制
// 输入:
//   W:  [C, H*3]   量化后的输入权重（int32_t 存储）
//   R:  [H, H*3]   量化后的循环权重（int32_t 存储）
//   bw: [H*3]      量化后的输入偏置（int32_t）
//   br: [H*3]      量化后的循环偏置（int32_t）
//   x:  [T, B, I]  浮点输入序列（内部会量化）
//   h0: [B, H]     初始隐藏状态（可为 nullptr）
// 输出:
//   h:  [(T+1), B, H]  所有时间步的隐藏状态（反量化后的浮点值）
//   v:  [T, B, H*4]    中间值（训练时需要，推理时可为 nullptr）
void quantGRUForward(
    bool is_training,
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const int32_t *W, const int32_t *R, const int32_t *bw, const int32_t *br, const float *x,
    const float *h0,
    const GRUQuantParams &quant_parms, const cublasHandle_t &g_blas_handle,
    float *h, float *v);

// =====================================================================
// CPU 量化 GRU 前向传播接口
// =====================================================================

// CPU 量化 GRU 前向传播（统一 int32_t 存储）
// 纯 CPU 实现，不依赖 CUDA，可在任意平台运行
// 输入:
//   W:  [C, H*3]   量化后的输入权重（int32_t 存储）
//   R:  [H, H*3]   量化后的循环权重（int32_t 存储）
//   bw: [H*3]      量化后的输入偏置（int32_t）
//   br: [H*3]      量化后的循环偏置（int32_t）
//   x:  [T, B, I]  浮点输入序列（内部会量化）
//   h0: [B, H]     初始隐藏状态（可为 nullptr）
// 输出:
//   h:  [(T+1), B, H]  所有时间步的隐藏状态（反量化后的浮点值）
//   v:  [T, B, H*4]    中间值（训练时需要，推理时可为 nullptr）
void quantGRUForwardCPU(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const int32_t *W, const int32_t *R, const int32_t *bw, const int32_t *br,
    const float *x, const float *h0,
    const GRUQuantParams &quant_parms,
    float *h, float *v);

// CPU 量化 GRU 前向传播（从浮点权重开始，内部量化）
// 输入:
//   W:  [C, H*3]   浮点输入权重
//   R:  [H, H*3]   浮点循环权重
//   bw: [H*3]      浮点输入偏置
//   br: [H*3]      浮点循环偏置
//   x:  [T, B, I]  浮点输入序列
//   h0: [B, H]     初始隐藏状态（可为 nullptr）
// 输出:
//   h:  [(T+1), B, H]  所有时间步的隐藏状态（反量化后的浮点值）
//   v:  [T, B, H*4]    中间值（训练时需要，推理时可为 nullptr）
void quantGRUForwardCPU(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br,
    const float *x, const float *h0,
    const GRUQuantParams &quant_parms,
    float *h, float *v);

// =====================================================================
// 纯定点 GRU 前向传播接口（核心实现）
// =====================================================================

// GPU 纯定点 GRU 前向传播（int32 输入/输出）
// 这是量化 GRU 的核心计算，所有高层接口都调用此函数
// 输入（全部 int32，GPU 内存）:
//   W_q:  [C, H*3]   量化后的输入权重
//   R_q:  [H, H*3]   量化后的循环权重
//   bw_q: [H*3]      量化后的输入偏置
//   br_q: [H*3]      量化后的循环偏置
//   x_q:  [T, B, I]  量化后的输入序列
//   h0_q: [B, H]     量化后的初始隐藏状态（可为 nullptr，则使用 zp_h）
// 输出（int32，GPU 内存）:
//   h_q:  [(T+1), B, H]  所有时间步的量化隐藏状态
//   v_q:  [T, B, H*4]    量化中间值（可为 nullptr）
void quantGRUForwardInt32(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const int32_t *W_q, const int32_t *R_q, const int32_t *bw_q, const int32_t *br_q,
    const int32_t *x_q, const int32_t *h0_q,
    const GRUQuantParams &quant_params,
    const cublasHandle_t &g_blas_handle,
    int32_t *h_q, int32_t *v_q);

// 浮点 GRU 前向传播
// 输入:
//   W:  [C, H*3]   输入权重矩阵
//   R:  [H, H*3]   循环权重矩阵
//   bw: [H*3]      输入偏置 (bias for W)
//   br: [H*3]      循环偏置
//   x:  [T, B, I]  输入序列
//   h0: [B, H]     初始隐藏状态（可为 nullptr）
// 输出:
//   h:  [(T+1), B, H]  所有时间步的隐藏状态（包含 h0）
//   v:  [T, B, H*4]    中间值（训练时需要，推理时可为 nullptr）
void hasteGRUForward(
    bool is_training,
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W, const float *R, const float *bw, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    float *h, float *v);

// 统一前向传播接口（推理/训练）
// 注意：校准请使用 forwardWithCalibrationMinMaxGPU 或 forwardWithCalibrationHistogramGPU
void forwardInterface(
    bool is_training, bool is_quant,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br, const float *x,
    const float *h0,
    const GRUQuantParams &quant_gru_scales, const cublasHandle_t &g_blas_handle,
    float *h, float *v);

// =====================================================================
// 统一校准前向传播（GPU）
// =====================================================================

// 根据 calib_method 自动选择 MINMAX 或 Histogram 校准
// - MINMAX: 使用 quant_ranges（原地累积更新）
// - SQNR/Percentile: 使用 gpu_hist_collectors（原地累积更新）
void forwardWithCalibrationGPU(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    CalibrationMethod calib_method,
    GRUQuantizationRanges *quant_ranges,              // MINMAX 时必须非空
    GRUGPUHistogramCollectors *gpu_hist_collectors,   // SQNR/Percentile 时必须非空
    float *h, float *v);

// CPU 直方图收集（用于性能对比）
// 前向传播在 GPU 上执行，直方图收集在 CPU 上执行
void forwardWithHistogramCPU(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bw, const float *br, const float *x,
    const float *h0,
    const cublasHandle_t &g_blas_handle,
    GRUHistogramCollectors *hist_collectors,
    float *h, float *v);

// 将 GPU 直方图收集器转换为 CPU 版本
// 用于与 calculateGRUQuantitativeParametersFromHistograms 配合使用
GRUHistogramCollectors convertGPUHistogramsToCPU(const GRUGPUHistogramCollectors &gpu_collectors);

// 从 GPU 直方图收集器直接计算量化参数（GPU 加速 SQNR）
// 避免 GPU→CPU 传输，直接在 GPU 上计算 SQNR
GRUQuantParams calculateGRUQuantitativeParametersFromGPUHistograms(
    GRUGPUHistogramCollectors &gpu_collectors,
    const OperatorQuantConfig &bitwidth_config = OperatorQuantConfig(),
    bool verbose = false);

// =====================================================================
// MINMAX 范围更新接口
// =====================================================================
// 
// updateGRUQuantizationRanges 定义在 include/calibration_utils.h 中
// 此处只是为了 API 文档完整性而保留注释

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
//   bw:  [H*3]         输入偏置（不需要转置）
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
//   dbw: [H*3]         输入偏置梯度
//   dbr: [H*3]         循环偏置梯度
//   dh:  [B, H]        初始隐藏状态梯度
void hasteGRUBackward(
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W_t, const float *R_t,
    const float *bw, const float *br,
    const float *x_t,
    const float *dh_new,
    const float *h, const float *v,
    const cublasHandle_t &g_blas_handle,
    float *dx, float *dW, float *dR, float *dbw, float *dbr, float *dh);
