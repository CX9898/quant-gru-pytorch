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

// GRU 权重量化统一接口（封装 W, R, bw, br）- 浮点存储版本
// 根据 granularity 自动选择 per-tensor、per-gate 或 per-channel 量化
// @tparam Training 是否训练模式（决定是否使用 mask）
// @param W 输入权重 W [input_size, hidden_size * 3]
// @param R 循环权重 R [hidden_size, hidden_size * 3]
// @param bw 输入偏置 bw [hidden_size * 3]
// @param br 循环偏置 br [hidden_size * 3]
// @param W_q_out 输出量化权重 W（必须由外部分配内存）
// @param R_q_out 输出量化权重 R（必须由外部分配内存）
// @param bw_q_out 输出量化偏置 bw（必须由外部分配内存）
// @param br_q_out 输出量化偏置 br（必须由外部分配内存）
// @param W_mask 训练模式时保存 W 的 clamp mask，推理模式时可为 nullptr
// @param R_mask 训练模式时保存 R 的 clamp mask，推理模式时可为 nullptr
// @param bw_mask 训练模式时保存 bw 的 clamp mask，推理模式时可为 nullptr
// @param br_mask 训练模式时保存 br 的 clamp mask，推理模式时可为 nullptr
// @param input_size 输入维度
// @param hidden_size 隐藏层维度
// @param quant_params 量化参数（包含 granularity 配置和 shift 值）
template <bool Training = false>
void quantizeGRUWeights(const float *W, const float *R, const float *bw, const float *br,
                        float *W_q_out, float *R_q_out, float *bw_q_out, float *br_q_out,
                        uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask,
                        size_t input_size, size_t hidden_size,
                        const GRUQuantParams &quant_params);

// 反量化 GRU 权重（W, R, bw, br）- 使用统一接口，内部根据 granularity 自动选择
// @param W_q 输入权重 W 的量化值（输入），反量化后的值（输出），原地修改
// @param R_q 循环权重 R 的量化值（输入），反量化后的值（输出），原地修改
// @param bw_q 输入偏置 bw 的量化值（输入），反量化后的值（输出），原地修改
// @param br_q 循环偏置 br 的量化值（输入），反量化后的值（输出），原地修改
// @param input_size 输入维度
// @param hidden_size 隐藏层维度
// @param quant_params 量化参数（包含 granularity 配置和 shift 值）
void dequantizeGRUWeights(float *W_q, float *R_q, float *bw_q, float *br_q,
                          size_t input_size, size_t hidden_size,
                          const GRUQuantParams &quant_params);

// =====================================================================
// GPU 量化 GRU 前向传播接口
// =====================================================================

// GPU 量化 GRU 前向传播（浮点输入/输出，内部自动量化权重和激活）
// 所有量化值使用 int32_t 存储，实际位宽通过 bitwidth_config_ 控制
// 输入:
//   W:  [C, H*3]   浮点输入权重（内部会量化）
//   R:  [H, H*3]   浮点循环权重（内部会量化）
//   bw: [H*3]      浮点输入偏置（内部会量化）
//   br: [H*3]      浮点循环偏置（内部会量化）
//   x:  [T, B, I]  浮点输入序列（内部会量化）
//   h0: [B, H]     初始隐藏状态（可为 nullptr）
// 输出:
//   h:  [(T+1), B, H]  所有时间步的隐藏状态（反量化后的浮点值）
//   v:  [T, B, H*4]    中间值（训练时需要，推理时可为 nullptr）
// QAT mask 输出（训练时需要外部分配，推理时可为 nullptr）:
//   输入量化 mask:
//     x_mask:  [T*B*I]    输入序列量化 mask
//     h0_mask: [B*H]      初始隐藏状态量化 mask（h0=nullptr时忽略）
//     W_mask:  [C*H*3]    输入权重量化 mask
//     R_mask:  [H*H*3]    循环权重量化 mask
//     bw_mask: [H*3]      输入偏置量化 mask
//     br_mask: [H*3]      循环偏置量化 mask
//   计算过程 mask:
//     weight_ih_linear_mask: [T*B*H*3]
//     weight_hh_linear_mask: [T*B*H*3]
//     gate_input_mask: [T*B*H*3] 门输入 clamp mask
//     gate_output_mask: [T*B*H*3] 门输出 clamp mask
//     h_mask: [T*B*H]
void quantGRUForward(
    bool is_training,
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W, const float *R, const float *bw, const float *br, const float *x,
    const float *h0,
    const GRUQuantParams &quant_parms, const cublasHandle_t &g_blas_handle,
    float *h, float *v,
    // 输入量化 mask（训练时外部分配，推理时可为 nullptr）
    uint8_t *x_mask = nullptr,
    uint8_t *h0_mask = nullptr,
    uint8_t *W_mask = nullptr,
    uint8_t *R_mask = nullptr,
    uint8_t *bw_mask = nullptr,
    uint8_t *br_mask = nullptr,
    // 计算过程 mask（训练时外部分配，推理时可为 nullptr）
    uint8_t *weight_ih_linear_mask = nullptr,
    uint8_t *weight_hh_linear_mask = nullptr,
    uint8_t *gate_input_mask = nullptr,
    uint8_t *gate_output_mask = nullptr,
    uint8_t *h_mask = nullptr);

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
// QAT mask 输出（外部分配，nullptr=不保存）:
//   weight_ih_linear_mask: [T*B, H*3]
//   weight_hh_linear_mask: [T*B, H*3]
//   gate_input_mask: [T*B, H*3] 门输入 clamp mask
//   gate_output_mask: [T*B, H*3] 门输出 clamp mask
//   h_mask: [T*B, H]
void quantGRUForwardInt32(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const int32_t *W_q, const int32_t *R_q, const int32_t *bw_q, const int32_t *br_q,
    const int32_t *x_q, const int32_t *h0_q,
    const GRUQuantParams &quant_params,
    const cublasHandle_t &g_blas_handle,
    int32_t *h_q, int32_t *v_q,
    uint8_t *weight_ih_linear_mask = nullptr,
    uint8_t *weight_hh_linear_mask = nullptr,
    uint8_t *gate_input_mask = nullptr,
    uint8_t *gate_output_mask = nullptr,
    uint8_t *h_mask = nullptr);

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
    const OperatorQuantConfig &bitwidth_config,      // 位宽配置（用于直方图收集）
    float *h, float *v);

// 将 GPU 直方图收集器转换为 CPU 版本
// 用于与 calculateGRUQuantitativeParametersFromHistograms 配合使用
GRUHistogramCollectors convertGPUHistogramsToCPU(const GRUGPUHistogramCollectors &gpu_collectors);

// 从 GPU 直方图收集器直接计算量化参数（GPU 加速 SQNR）
// 避免 GPU→CPU 传输，直接在 GPU 上计算 SQNR
GRUQuantParams calculateGRUQuantitativeParametersFromGPUHistograms(
    GRUGPUHistogramCollectors &gpu_collectors,
    const OperatorQuantConfig &bitwidth_config = OperatorQuantConfig());

// =====================================================================
// 反向传播接口
// =====================================================================

// =====================================================================
// 浮点存储版量化 GRU 前向传播接口（GPU-FP）
// =====================================================================
// 
// 与整数版的区别：
//   - 所有量化值使用 float 存储（值仍是定点整数）
//   - 使用 cuBLAS SGEMM + 单独的 bias/rescale kernel
//   - 只使用 real_sigmoid/real_tanh，不用 LUT
//   - shift 预处理为除数，避免运行时位移

// 量化权重为浮点存储格式（GPU 端）
// GPU 浮点存储版量化前向传播（浮点输入/输出，全部 device 内存）
// 输入（全部 device 内存）:
//   W:  [C, H*3]   浮点输入权重
//   R:  [H, H*3]   浮点循环权重
//   bw: [H*3]      浮点输入偏置
//   br: [H*3]      浮点循环偏置
//   x:  [T, B, I]  浮点输入序列
//   h0: [B, H]     初始隐藏状态（可为 nullptr）
// 输出（全部 device 内存）:
//   h:  [(T+1), B, H]  所有时间步的隐藏状态（反量化后的浮点值）
//   v:  [T, B, H*4]    中间值（训练时需要，推理时可为 nullptr）
// QAT mask 输出（外部分配，nullptr=不保存）:
//   x_mask: [T*B, I] 输入序列量化 mask
//   h0_mask: [B, H] 初始隐状态量化 mask
//   W_mask: [I, H*3] 输入权重量化 mask
//   R_mask: [H, H*3] 循环权重量化 mask
//   bw_mask: [H*3] 输入偏置量化 mask
//   br_mask: [H*3] 循环偏置量化 mask
//   weight_ih_linear_mask: [T*B, H*3] weight_ih_linear clamp mask
//   weight_hh_linear_mask: [T*B, H*3] weight_hh_linear clamp mask
//   gate_mask: [T*B, H*3] gate clamp mask
//   h_mask: [T*B, H] hidden state clamp mask
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
    float *W_q_out,  // [input_size, hidden_size * 3]
    float *R_q_out,  // [hidden_size, hidden_size * 3]
    float *bw_q_out, // [hidden_size * 3]
    float *br_q_out, // [hidden_size * 3]
    float *x_q_out,  // [time_steps, batch_size, input_size]
    // 输入量化 mask（外部分配，nullptr=不保存）
    uint8_t *x_mask,
    uint8_t *h0_mask,
    uint8_t *W_mask,
    uint8_t *R_mask,
    uint8_t *bw_mask,
    uint8_t *br_mask,
    // 计算过程 mask（外部分配，nullptr=不保存）
    uint8_t *weight_ih_linear_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask);


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

// ============================================================================
// 量化 GRU 反向传播接口（支持 QAT mask 和 rescale 补偿）
// ============================================================================
//
// 与 hasteGRUBackward 类似，但使用 BackwardPassQuant 并支持：
//   1. QAT mask: STE（被 clamp 的梯度置零）
//   2. Rescale 补偿: 梯度乘以 divisor 补偿前向传播中的 div_round 操作
//
// quant_params: 量化参数（用于计算 rescale 因子），nullptr=不应用 rescale
//
// QAT Mask 参数（可选，nullptr=不应用）：
//   - x_mask [T*B, I] → dx
//   - h0_mask [B, H] → dh
//   - W_mask [I, H*3] → dW
//   - R_mask [H, H*3] → dR
//   - bw_mask [H*3] → dbw
//   - br_mask [H*3] → dbr
//   - weight_ih_linear_mask [T*B, H*3] → dp
//   - weight_hh_linear_mask [T*B, H*3] → dq
//   - gate_input_mask [T*B, H*3] → 门输入梯度
//   - gate_output_mask [T*B, H*3] → 门输出梯度
//   - h_mask [T*B, H] → 隐状态梯度
//
void quantGRUBackward(
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W_t, const float *R_t,
    const float *bw, const float *br,
    const float *x_t,
    const float *dh_new,
    const float *h, const float *v,
    const cublasHandle_t &g_blas_handle,
    float *dx, float *dW, float *dR, float *dbw, float *dbr, float *dh,
    // 以下为量化相关参数（可选）
    const GRUQuantParams *quant_params = nullptr,  // 量化参数（用于 rescale 补偿），nullptr=不应用
    // QAT masks（可选）
    const uint8_t *x_mask = nullptr,
    const uint8_t *h0_mask = nullptr,
    const uint8_t *W_mask = nullptr,
    const uint8_t *R_mask = nullptr,
    const uint8_t *bw_mask = nullptr,
    const uint8_t *br_mask = nullptr,
    const uint8_t *weight_ih_linear_mask = nullptr,
    const uint8_t *weight_hh_linear_mask = nullptr,
    const uint8_t *gate_input_mask = nullptr,
    const uint8_t *gate_output_mask = nullptr,
    const uint8_t *h_mask = nullptr);

