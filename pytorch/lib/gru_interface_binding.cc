#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <iostream>

#include "gru_interface.h"
#include "gru_quantization_ranges.h"
#include "histogram_collector.h"
#include "calibration_gpu.cuh"
#include "quantize_ops_helper.h"

// 全局 cublas handle
static cublasHandle_t g_blas_handle = nullptr;

// 初始化 cublas handle 的包装函数
void init_gru_cublas_wrapper() { init_gru_cublas(g_blas_handle); }

// CalibrationMethod 字符串转换
// Python 端只需要传字符串，C++ wrapper 内部转换
inline CalibrationMethod stringToCalibMethod(const std::string &method) {
    if (method.empty() || method == "none") {
        return CalibrationMethod::NONE;
    } else if (method == "minmax") {
        return CalibrationMethod::MINMAX;
    } else if (method == "sqnr") {
        return CalibrationMethod::SQNR;
    } else if (method == "percentile") {
        return CalibrationMethod::PERCENTILE;
    } else {
        throw std::invalid_argument("Invalid calibration method: '" + method + 
            "'. Valid options: 'none', 'minmax', 'sqnr', 'percentile'");
    }
}

// ============================================================================
//                    位宽转换辅助函数（前置定义）
// ============================================================================
//
// 设计说明：
// - Python 端只配置位宽数量（1-32），不关心实际类型（INT/UINT）
// - C++ 端在 to_cpp() 时决定实际类型
// - 这些辅助函数需要在 OperatorQuantConfigPy 之前定义
//
// ============================================================================

/// 位宽转换：Python 端的位宽数值 → C++ QuantBitWidth 结构体
/// @param bitwidth 位宽 (1-32)
/// @param is_unsigned false=INT有符号(默认), true=UINT无符号
inline QuantBitWidth toBitwidth(int8_t bitwidth, bool is_unsigned = false) {
    if (bitwidth < 1 || bitwidth > 32) {
        throw std::invalid_argument("Invalid bitwidth: " + std::to_string(bitwidth) +
                                    ". Must be 1-32.");
    }
    return QuantBitWidth(bitwidth, is_unsigned);
}

// ============================================================================
//                    OperatorQuantConfig Python 绑定（前置定义）
// ============================================================================
//
// 注意：这个结构体需要在 GRUQuantParamsPy 之前定义，
// 因为 GRUQuantParamsPy 中包含 OperatorQuantConfigPy 成员
//
// ============================================================================

struct OperatorQuantConfigPy {
    // 位宽配置（存储位宽数量 1-32）
    // 命名与 C++ OperatorQuantConfig 保持一致
    int8_t x_, h_;
    int8_t W_, R_, bw_, br_;
    int8_t weight_ih_linear_, weight_hh_linear_;  // GEMM+bias 融合输出
    int8_t update_gate_input_, update_gate_output_;
    int8_t reset_gate_input_, reset_gate_output_;
    int8_t new_gate_input_, new_gate_output_;
    int8_t mul_reset_hidden_;
    int8_t mul_old_contribution_, mul_new_contribution_;

    // 对称量化配置
    bool x_symmetric_, h_symmetric_;
    bool W_symmetric_, R_symmetric_, bw_symmetric_, br_symmetric_;
    bool weight_ih_linear_symmetric_, weight_hh_linear_symmetric_;
    bool update_gate_input_symmetric_, update_gate_output_symmetric_;
    bool reset_gate_input_symmetric_, reset_gate_output_symmetric_;
    bool new_gate_input_symmetric_, new_gate_output_symmetric_;
    bool mul_reset_hidden_symmetric_;
    bool mul_old_contribution_symmetric_, mul_new_contribution_symmetric_;

    // 无符号量化配置（与 C++ is_unsigned_ 一致，只标记例外情况）
    bool x_unsigned_, h_unsigned_;
    bool W_unsigned_, R_unsigned_, bw_unsigned_, br_unsigned_;
    bool weight_ih_linear_unsigned_, weight_hh_linear_unsigned_;
    bool update_gate_input_unsigned_, update_gate_output_unsigned_;
    bool reset_gate_input_unsigned_, reset_gate_output_unsigned_;
    bool new_gate_input_unsigned_, new_gate_output_unsigned_;
    bool mul_reset_hidden_unsigned_;
    bool mul_old_contribution_unsigned_, mul_new_contribution_unsigned_;

    // 方法声明（实现在文件末尾）
    OperatorQuantConfigPy();                           // 默认构造函数：从 C++ 默认值初始化
    OperatorQuantConfig to_cpp() const;                // 转换为 C++ 结构体
    void from_cpp(const OperatorQuantConfig &cfg);     // 从 C++ 结构体读取
};

// GRUQuantizationRanges 的 Python 绑定（直接包装 C++ 对象）
struct GRUQuantizationRangesPy {
    GRUQuantizationRanges cpp_ranges;  // 直接包装 C++ 对象

    // 默认构造函数
    GRUQuantizationRangesPy() = default;

    // 带 hidden 参数的构造函数
    explicit GRUQuantizationRangesPy(int hidden) : cpp_ranges(hidden) {}

    // 重置所有范围
    void reset(int hidden = -1) { cpp_ranges.reset(hidden); }

    // 直接访问内部 C++ 对象（用于 C++ 层调用）
    GRUQuantizationRanges& get_cpp() { return cpp_ranges; }
    const GRUQuantizationRanges& get_cpp() const { return cpp_ranges; }
};

// GRUQuantParams 的 Python 绑定
// 命名与 C++ GRUQuantParams 保持一致
struct GRUQuantParamsPy {
    int hidden_;
    // 基础参数
    int8_t shift_x_;
    int32_t zp_x_;
    int8_t shift_h_;
    int32_t zp_h_;
    // 权重参数（per-channel）
    std::vector<int8_t> shift_W_;
    std::vector<int8_t> shift_R_;
    std::vector<int8_t> shift_bw_;
    std::vector<int8_t> shift_br_;
    // Linear 输出参数 (GEMM+bias)
    int8_t shift_weight_ih_linear_;
    int32_t zp_weight_ih_linear_;
    int8_t shift_weight_hh_linear_;
    int32_t zp_weight_hh_linear_;
    // 门激活函数输入参数（pre-activation）
    int8_t shift_update_gate_input_;
    int32_t zp_update_gate_input_;
    int8_t shift_reset_gate_input_;
    int32_t zp_reset_gate_input_;
    int8_t shift_new_gate_input_;
    int32_t zp_new_gate_input_;
    // 门激活函数输出参数（post-activation）
    int8_t shift_update_gate_output_;
    int32_t zp_update_gate_output_;
    int8_t shift_reset_gate_output_;
    int32_t zp_reset_gate_output_;
    int8_t shift_new_gate_output_;
    int32_t zp_new_gate_output_;
    // 中间计算参数
    int8_t shift_mul_reset_hidden_;
    int32_t zp_mul_reset_hidden_;
    // 隐状态更新参数
    int8_t shift_mul_new_contribution_;
    int32_t zp_mul_new_contribution_;
    int8_t shift_mul_old_contribution_;
    int32_t zp_mul_old_contribution_;

    // ⚠️ 关键字段：位宽配置必须在 Python 和 C++ 之间正确传递
    // 否则 forwardInterface 会使用默认的 8 位配置
    OperatorQuantConfigPy bitwidth_config_;

    // 方法声明（实现在文件末尾）
    void from_cpp(const GRUQuantParams &cpp_params);  // 从 C++ 结构体转换
    GRUQuantParams to_cpp() const;                     // 转换为 C++ 结构体
};

// 根据量化范围计算量化参数的包装函数（支持自定义位宽配置）
GRUQuantParamsPy calculate_gru_quantitative_parameters_wrapper(
    const GRUQuantizationRangesPy &quant_ranges,
    const OperatorQuantConfigPy &bitwidth_config = OperatorQuantConfigPy()) {
    // 直接使用包装的 C++ 对象
    OperatorQuantConfig cpp_bitwidth = bitwidth_config.to_cpp();

    // 调用 C++ 函数
    GRUQuantParams quant_params =
        calculateGRUQuantitativeParameters(quant_ranges.cpp_ranges, cpp_bitwidth);

    GRUQuantParamsPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// =====================================================================
// AIMET 风格直方图校准接口（默认使用 GPU 加速）
// =====================================================================

// GRUHistogramCollectors 的 Python 绑定包装
// 内部使用 GPU 版本实现，Python 接口保持不变
struct GRUHistogramCollectorsPy {
    GRUGPUHistogramCollectors gpu_collectors;  // 使用 GPU 版本

    GRUHistogramCollectorsPy() = default;
    explicit GRUHistogramCollectorsPy(int hidden, int num_bins = 2048) : gpu_collectors(hidden, num_bins) {}

    void reset(int hidden = -1, int num_bins = -1) { gpu_collectors.reset(hidden, num_bins); }
    bool is_valid() const { return gpu_collectors.is_valid(); }
    int hidden() const { return gpu_collectors.hidden_; }
    int num_bins() const { return gpu_collectors.num_bins_; }
    
    // 转换为 CPU 版本（内部使用）
    GRUHistogramCollectors to_cpu() const {
        return convertGPUHistogramsToCPU(gpu_collectors);
    }
    
    // 打印统计信息
    void print() const {
        std::cout << "GRUHistogramCollectors (GPU):" << std::endl;
        std::cout << "  hidden: " << gpu_collectors.hidden_ << std::endl;
        std::cout << "  num_bins: " << gpu_collectors.num_bins_ << std::endl;
        std::cout << "  is_valid: " << (is_valid() ? "true" : "false") << std::endl;
    }
};

// 从直方图计算量化参数的包装函数
// 内部自动选择 GPU/CPU 实现：SQNR 用 GPU，Percentile 用 CPU
GRUQuantParamsPy calculate_gru_quantitative_parameters_from_histograms_wrapper(
    GRUHistogramCollectorsPy &hist_collectors,
    const OperatorQuantConfigPy &bitwidth_config = OperatorQuantConfigPy(),
    bool verbose = false,
    bool use_percentile = false,
    float percentile_value = 99.99f) {

    OperatorQuantConfig cpp_bitwidth = bitwidth_config.to_cpp();
    GRUQuantParams quant_params;
    
    if (use_percentile) {
        // Percentile 方案：转换为 CPU 直方图后计算
        GRUHistogramCollectors cpu_collectors = hist_collectors.to_cpu();
        quant_params = calculateGRUQuantitativeParametersFromHistograms(
            cpu_collectors, cpp_bitwidth, verbose, true, percentile_value);
    } else {
        // SQNR 方案：直接使用 GPU 计算
        quant_params = calculateGRUQuantitativeParametersFromGPUHistograms(
            hist_collectors.gpu_collectors, cpp_bitwidth, verbose);
    }

    GRUQuantParamsPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// =====================================================================
// forward: 正常前向传播（推理/训练）
// =====================================================================
// 返回: (h, v, x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
//        weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask)
//   h: [T+1, B, H] 隐藏状态序列（包含初始状态）
//   v: [T, B, H*4] 中间值（训练时需要，推理时可忽略）
//   输入量化 mask:
//   x_mask: [T, B, I] 输入序列量化 mask
//   h0_mask: [B, H] 初始隐状态量化 mask
//   W_mask: [I, H*3] 输入权重量化 mask
//   R_mask: [H, H*3] 循环权重量化 mask
//   bw_mask: [H*3] 输入偏置量化 mask
//   br_mask: [H*3] 循环偏置量化 mask
//   计算过程 mask:
//   weight_ih_linear_mask: [T, B, H*3] QAT mask
//   weight_hh_linear_mask: [T, B, H*3] QAT mask
//   gate_input_mask: [T, B, H*3] 门输入 clamp mask
//   gate_output_mask: [T, B, H*3] 门输出 clamp mask
//   h_mask: [T, B, H] QAT mask
std::tuple<torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
forward_wrapper(
    bool is_training,  // 是否开启训练模式
    bool is_quant,     // 是否使用量化推理
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &h0,  // 初始隐藏状态，可以为空张量
    const GRUQuantParamsPy &quant_params) {
    
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // h0 可以为空张量（未提供初始状态）
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32,
                    "h0 must be CUDA float32 tensor");
        TORCH_CHECK(h0.sizes() == torch::IntArrayRef({batch_size, hidden_size}),
                    "h0 must have shape [batch_size, hidden_size]");
        h0_ptr = h0.data_ptr<float>();
    }

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 创建 QAT mask 张量（训练+量化模式时分配，否则为空张量）
    // 输入量化 mask
    torch::Tensor x_mask_tensor, h0_mask_tensor, W_mask_tensor, R_mask_tensor, bw_mask_tensor, br_mask_tensor;
    // 计算过程 mask
    torch::Tensor weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask;
    
    // mask 指针
    uint8_t *x_mask_ptr = nullptr, *h0_mask_ptr = nullptr;
    uint8_t *W_mask_ptr = nullptr, *R_mask_ptr = nullptr;
    uint8_t *bw_mask_ptr = nullptr, *br_mask_ptr = nullptr;
    uint8_t *weight_ih_mask_ptr = nullptr, *weight_hh_mask_ptr = nullptr;
    uint8_t *gate_input_mask_ptr = nullptr, *gate_output_mask_ptr = nullptr, *h_mask_ptr = nullptr;
    
    const bool need_mask = is_training && is_quant;
    if (need_mask) {
        const int hidden3 = hidden_size * 3;
        
        // 输入量化 mask
        x_mask_tensor = torch::empty({time_steps, batch_size, input_size},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        h0_mask_tensor = (h0_ptr != nullptr) 
            ? torch::empty({batch_size, hidden_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA))
            : torch::empty({0}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        W_mask_tensor = torch::empty({input_size, hidden3},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        R_mask_tensor = torch::empty({hidden_size, hidden3},
                                     torch::dtype(torch::kUInt8).device(torch::kCUDA));
        bw_mask_tensor = torch::empty({hidden3},
                                      torch::dtype(torch::kUInt8).device(torch::kCUDA));
        br_mask_tensor = torch::empty({hidden3},
                                      torch::dtype(torch::kUInt8).device(torch::kCUDA));
        
        // 计算过程 mask
        weight_ih_linear_mask = torch::empty({time_steps, batch_size, hidden3},
                                             torch::dtype(torch::kUInt8).device(torch::kCUDA));
        weight_hh_linear_mask = torch::empty({time_steps, batch_size, hidden3},
                                             torch::dtype(torch::kUInt8).device(torch::kCUDA));
        gate_input_mask = torch::empty({time_steps, batch_size, hidden3},
                                       torch::dtype(torch::kUInt8).device(torch::kCUDA));
        gate_output_mask = torch::empty({time_steps, batch_size, hidden3},
                                        torch::dtype(torch::kUInt8).device(torch::kCUDA));
        h_mask = torch::empty({time_steps, batch_size, hidden_size},
                              torch::dtype(torch::kUInt8).device(torch::kCUDA));
        
        // 获取指针
        x_mask_ptr = x_mask_tensor.data_ptr<uint8_t>();
        h0_mask_ptr = (h0_ptr != nullptr) ? h0_mask_tensor.data_ptr<uint8_t>() : nullptr;
        W_mask_ptr = W_mask_tensor.data_ptr<uint8_t>();
        R_mask_ptr = R_mask_tensor.data_ptr<uint8_t>();
        bw_mask_ptr = bw_mask_tensor.data_ptr<uint8_t>();
        br_mask_ptr = br_mask_tensor.data_ptr<uint8_t>();
        weight_ih_mask_ptr = weight_ih_linear_mask.data_ptr<uint8_t>();
        weight_hh_mask_ptr = weight_hh_linear_mask.data_ptr<uint8_t>();
        gate_input_mask_ptr = gate_input_mask.data_ptr<uint8_t>();
        gate_output_mask_ptr = gate_output_mask.data_ptr<uint8_t>();
        h_mask_ptr = h_mask.data_ptr<uint8_t>();
    } else {
        // 返回空张量（numel() == 0）
        auto empty_mask = torch::empty({0}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        x_mask_tensor = empty_mask.clone();
        h0_mask_tensor = empty_mask.clone();
        W_mask_tensor = empty_mask.clone();
        R_mask_tensor = empty_mask.clone();
        bw_mask_tensor = empty_mask.clone();
        br_mask_tensor = empty_mask.clone();
        weight_ih_linear_mask = empty_mask.clone();
        weight_hh_linear_mask = empty_mask.clone();
        gate_input_mask = empty_mask.clone();
        gate_output_mask = empty_mask.clone();
        h_mask = empty_mask.clone();
    }

    // 只有量化时才调用 to_cpp()（避免非量化时白白生成 LUT）
    GRUQuantParams cpp_params;
    if (is_quant) {
        cpp_params = quant_params.to_cpp();  // 包含 LUT 生成
    }
    
    forwardInterface(is_training, is_quant, time_steps, batch_size, input_size, hidden_size,
                     W.data_ptr<float>(), R.data_ptr<float>(), 
                     bw.data_ptr<float>(), br.data_ptr<float>(), 
                     x.data_ptr<float>(), h0_ptr, cpp_params, g_blas_handle,
                     h.data_ptr<float>(), v.data_ptr<float>(),
                     x_mask_ptr, h0_mask_ptr, W_mask_ptr, R_mask_ptr, bw_mask_ptr, br_mask_ptr,
                     weight_ih_mask_ptr, weight_hh_mask_ptr, gate_input_mask_ptr, gate_output_mask_ptr, h_mask_ptr);

    return std::make_tuple(h, v, 
                           x_mask_tensor, h0_mask_tensor, W_mask_tensor, R_mask_tensor, bw_mask_tensor, br_mask_tensor,
                           weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask);
}

// =====================================================================
// forward_calibrate: 校准前向传播（统一接口）
// =====================================================================
// 返回: (h, v)
//   h: [T+1, B, H] 隐藏状态序列
//   v: [T, B, H*4] 中间值
//
// 校准数据通过指针参数原地累积更新：
// - MINMAX: quant_ranges 被原地更新
// - SQNR/Percentile: hist_collectors 被原地更新
std::tuple<torch::Tensor, torch::Tensor> forward_calibrate_wrapper(
    bool is_training,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &h0,
    const std::string &calib_method_str,  // 校准方法: 'minmax', 'sqnr', 'percentile'
    GRUQuantizationRangesPy *quant_ranges = nullptr,      // MINMAX 需要
    GRUHistogramCollectorsPy *hist_collectors = nullptr) {  // SQNR/Percentile 需要
    
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // 字符串转枚举
    CalibrationMethod calib_method = stringToCalibMethod(calib_method_str);
    
    // 校准模式不能是 NONE
    TORCH_CHECK(calib_method != CalibrationMethod::NONE,
                "forward_calibrate requires a calibration method ('minmax', 'sqnr', 'percentile')");
    
    // 验证校准参数
    if (calib_method == CalibrationMethod::MINMAX) {
        TORCH_CHECK(quant_ranges != nullptr, 
                    "quant_ranges is required for MINMAX calibration");
    } else {
        TORCH_CHECK(hist_collectors != nullptr, 
                    "hist_collectors is required for SQNR/Percentile calibration");
    }

    // h0 可以为空张量
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32,
                    "h0 must be CUDA float32 tensor");
        TORCH_CHECK(h0.sizes() == torch::IntArrayRef({batch_size, hidden_size}),
                    "h0 must have shape [batch_size, hidden_size]");
        h0_ptr = h0.data_ptr<float>();
    }

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // 统一调用校准前向传播
    forwardWithCalibrationGPU(
        is_training, time_steps, batch_size, input_size, hidden_size,
        W.data_ptr<float>(), R.data_ptr<float>(), 
        bw.data_ptr<float>(), br.data_ptr<float>(), 
        x.data_ptr<float>(), h0_ptr, g_blas_handle,
        calib_method,
        quant_ranges ? &(quant_ranges->cpp_ranges) : nullptr,
        hist_collectors ? &(hist_collectors->gpu_collectors) : nullptr,
        h.data_ptr<float>(), v.data_ptr<float>());

    return std::make_tuple(h, v);
}

// =====================================================================
// quant_gru_forward_int32: 纯定点 GRU 前向传播（用于验证）
// =====================================================================
// 输入输出都是 int32 张量，不涉及浮点转换
// 用于验证 CPU 定点实现与 GPU 定点实现的一致性
torch::Tensor quant_gru_forward_int32_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W_q, const torch::Tensor &R_q,
    const torch::Tensor &bw_q, const torch::Tensor &br_q,
    const torch::Tensor &x_q, const torch::Tensor &h0_q,
    const GRUQuantParamsPy &quant_params) {
    
    // 检查输入类型和设备
    TORCH_CHECK(W_q.is_cuda() && W_q.dtype() == torch::kInt32, "W_q must be CUDA int32 tensor");
    TORCH_CHECK(R_q.is_cuda() && R_q.dtype() == torch::kInt32, "R_q must be CUDA int32 tensor");
    TORCH_CHECK(bw_q.is_cuda() && bw_q.dtype() == torch::kInt32, "bw_q must be CUDA int32 tensor");
    TORCH_CHECK(br_q.is_cuda() && br_q.dtype() == torch::kInt32, "br_q must be CUDA int32 tensor");
    TORCH_CHECK(x_q.is_cuda() && x_q.dtype() == torch::kInt32, "x_q must be CUDA int32 tensor");
    
    // h0_q 可以为空
    const int32_t *h0_ptr = nullptr;
    if (h0_q.defined() && h0_q.numel() > 0) {
        TORCH_CHECK(h0_q.is_cuda() && h0_q.dtype() == torch::kInt32, "h0_q must be CUDA int32 tensor");
        h0_ptr = h0_q.data_ptr<int32_t>();
    }
    
    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }
    
    // 创建输出张量
    auto h_q = torch::empty({time_steps + 1, batch_size, hidden_size},
                            torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    // 转换量化参数
    GRUQuantParams cpp_params = quant_params.to_cpp();
    
    // 调用 C++ 函数（推理模式，不输出 v）
    quantGRUForwardInt32(
        false, time_steps, batch_size, input_size, hidden_size,
        W_q.data_ptr<int32_t>(), R_q.data_ptr<int32_t>(),
        bw_q.data_ptr<int32_t>(), br_q.data_ptr<int32_t>(),
        x_q.data_ptr<int32_t>(), h0_ptr,
        cpp_params, g_blas_handle,
        h_q.data_ptr<int32_t>(), nullptr);
    
    return h_q;
}

// GRU 反向传播包装函数
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
haste_gru_backward_wrapper(int time_steps, int batch_size, int input_size, int hidden_size,
                           const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
                           const torch::Tensor &br, const torch::Tensor &x,
                           const torch::Tensor &dh_new,  // 来自上层网络或损失函数的反向梯度
                           const torch::Tensor &h,       // 前向传播的隐藏状态
                           const torch::Tensor &v) {     // 前向传播的中间值，必需

    // 检查输入张量的类型和设备
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");
    TORCH_CHECK(dh_new.is_cuda() && dh_new.dtype() == torch::kFloat32,
                "dh_new must be CUDA float32 tensor");
    TORCH_CHECK(h.is_cuda() && h.dtype() == torch::kFloat32, "h must be CUDA float32 tensor");
    TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kFloat32, "v must be CUDA float32 tensor");

    // 检查张量形状
    // 根据 haste 的实现，gru_backward 期望转置后的格式：
    // x_t: [input_size, time_steps, batch_size] (转置后的 x)
    // kernel_t: [hidden_size * 3, input_size] (转置后的 kernel)
    // recurrent_kernel_t: [hidden_size * 3, hidden_size] (转置后的 recurrent_kernel)

    // 检查 x 的形状，需要转置为 [input_size, time_steps, batch_size]
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({time_steps, batch_size, input_size}),
                "x must have shape [time_steps, batch_size, input_size]");
    torch::Tensor x_t = x.permute({2, 0, 1}).contiguous();  // [T,B,I] -> [I,T,B]

    // 检查 W 的形状，需要转置为 [hidden_size * 3, input_size]
    TORCH_CHECK(W.sizes() == torch::IntArrayRef({input_size, hidden_size * 3}),
                "W must have shape [input_size, hidden_size * 3]");
    torch::Tensor W_t = W.t().contiguous();  // [C, H*3] -> [H*3, C]

    // 检查 R 的形状，需要转置为 [hidden_size * 3, hidden_size]
    TORCH_CHECK(R.sizes() == torch::IntArrayRef({hidden_size, hidden_size * 3}),
                "R must have shape [hidden_size, hidden_size * 3]");
    torch::Tensor R_t = R.t().contiguous();  // [H, H*3] -> [H*3, H]

    TORCH_CHECK(bw.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "bw must have shape [hidden_size * 3]");
    TORCH_CHECK(br.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "br must have shape [hidden_size * 3]");
    TORCH_CHECK(dh_new.sizes() == torch::IntArrayRef({time_steps + 1, batch_size, hidden_size}),
                "dh_new must have shape [time_steps + 1, batch_size, hidden_size]");
    TORCH_CHECK(h.sizes() == torch::IntArrayRef({time_steps + 1, batch_size, hidden_size}),
                "h must have shape [time_steps + 1, batch_size, hidden_size]");
    TORCH_CHECK(v.sizes() == torch::IntArrayRef({time_steps, batch_size, hidden_size * 4}),
                "v must have shape [time_steps, batch_size, hidden_size * 4]");

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto dx = torch::empty({time_steps, batch_size, input_size},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dW = torch::zeros({input_size, hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dR = torch::zeros({hidden_size, hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbw = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbr = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dh =
        torch::zeros({batch_size, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 调用 C++ 函数
    // 注意：需要将张量展平为连续内存布局
    // C++ BackwardPass 期望转置后的格式（与 haste 一致）：
    // W_t: [H*3, C], R_t: [H*3, H], x_t: [I, T, B]
    hasteGRUBackward(time_steps, batch_size, input_size, hidden_size,
                     W_t.data_ptr<float>(),  // [H*3, C] - 转置后的 W
                     R_t.data_ptr<float>(),  // [H*3, H] - 转置后的 R
                     bw.data_ptr<float>(), br.data_ptr<float>(),
                     x_t.data_ptr<float>(),  // [I, T, B] - 转置后的 x
                     dh_new.data_ptr<float>(), h.data_ptr<float>(), v.data_ptr<float>(),
                     g_blas_handle, dx.data_ptr<float>(), dW.data_ptr<float>(),
                     dR.data_ptr<float>(), dbw.data_ptr<float>(), dbr.data_ptr<float>(),
                     dh.data_ptr<float>());

    return std::make_tuple(dx, dW, dR, dbw, dbr, dh);
}

// ============================================================================
// 统一反向传播包装函数（支持 QAT mask）
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_wrapper(bool is_quant,
                 int time_steps, int batch_size, int input_size, int hidden_size,
                 const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bw,
                 const torch::Tensor &br, const torch::Tensor &x,
                 const torch::Tensor &dh_new,
                 const torch::Tensor &h,
                 const torch::Tensor &v,
                 // 以下为量化相关参数
                 const GRUQuantParamsPy &quant_params,
                 // QAT masks
                 const torch::Tensor &x_mask,
                 const torch::Tensor &h0_mask,
                 const torch::Tensor &W_mask,
                 const torch::Tensor &R_mask,
                 const torch::Tensor &bw_mask,
                 const torch::Tensor &br_mask,
                 const torch::Tensor &weight_ih_linear_mask,
                 const torch::Tensor &weight_hh_linear_mask,
                 const torch::Tensor &gate_input_mask,
                 const torch::Tensor &gate_output_mask,
                 const torch::Tensor &h_mask) {

    // 检查输入张量的类型和设备
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bw.is_cuda() && bw.dtype() == torch::kFloat32, "bw must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");
    TORCH_CHECK(dh_new.is_cuda() && dh_new.dtype() == torch::kFloat32,
                "dh_new must be CUDA float32 tensor");
    TORCH_CHECK(h.is_cuda() && h.dtype() == torch::kFloat32, "h must be CUDA float32 tensor");
    TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kFloat32, "v must be CUDA float32 tensor");

    // 检查张量形状
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({time_steps, batch_size, input_size}),
                "x must have shape [time_steps, batch_size, input_size]");
    torch::Tensor x_t = x.permute({2, 0, 1}).contiguous();  // [T,B,I] -> [I,T,B]

    TORCH_CHECK(W.sizes() == torch::IntArrayRef({input_size, hidden_size * 3}),
                "W must have shape [input_size, hidden_size * 3]");
    torch::Tensor W_t = W.t().contiguous();  // [C, H*3] -> [H*3, C]

    TORCH_CHECK(R.sizes() == torch::IntArrayRef({hidden_size, hidden_size * 3}),
                "R must have shape [hidden_size, hidden_size * 3]");
    torch::Tensor R_t = R.t().contiguous();  // [H, H*3] -> [H*3, H]

    TORCH_CHECK(bw.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "bw must have shape [hidden_size * 3]");
    TORCH_CHECK(br.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "br must have shape [hidden_size * 3]");
    TORCH_CHECK(dh_new.sizes() == torch::IntArrayRef({time_steps + 1, batch_size, hidden_size}),
                "dh_new must have shape [time_steps + 1, batch_size, hidden_size]");
    TORCH_CHECK(h.sizes() == torch::IntArrayRef({time_steps + 1, batch_size, hidden_size}),
                "h must have shape [time_steps + 1, batch_size, hidden_size]");
    TORCH_CHECK(v.sizes() == torch::IntArrayRef({time_steps, batch_size, hidden_size * 4}),
                "v must have shape [time_steps, batch_size, hidden_size * 4]");

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 创建输出张量
    auto dx = torch::empty({time_steps, batch_size, input_size},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dW = torch::zeros({input_size, hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dR = torch::zeros({hidden_size, hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbw = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbr = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dh = torch::zeros({batch_size, hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 获取 mask 指针（空 tensor 时为 nullptr）
    auto get_mask_ptr = [](const torch::Tensor &t) -> const uint8_t* {
        return t.numel() > 0 ? t.data_ptr<uint8_t>() : nullptr;
    };

    // 转换量化参数（保留以保持接口兼容性，但反向传播中不再使用）
    // 注意：反向传播不需要quant_params，因为使用的是反量化后的浮点值
    const GRUQuantParams *quant_params_ptr = nullptr;
    GRUQuantParams cpp_params;
    if (is_quant && quant_params.hidden_ > 0) {
        cpp_params = quant_params.to_cpp();
        quant_params_ptr = &cpp_params;
    }

    // 调用 C++ 统一反向接口
    backwardInterface(
        is_quant,
        time_steps, batch_size, input_size, hidden_size,
        W_t.data_ptr<float>(),  // [H*3, C] - 转置后的 W
        R_t.data_ptr<float>(),  // [H*3, H] - 转置后的 R
        bw.data_ptr<float>(), br.data_ptr<float>(),
        x_t.data_ptr<float>(),  // [I, T, B] - 转置后的 x
        dh_new.data_ptr<float>(), h.data_ptr<float>(), v.data_ptr<float>(),
        g_blas_handle,
        dx.data_ptr<float>(), dW.data_ptr<float>(),
        dR.data_ptr<float>(), dbw.data_ptr<float>(), dbr.data_ptr<float>(),
        dh.data_ptr<float>(),
        // 量化相关参数（用于 mask 生成，不用于 rescale 补偿）
        quant_params_ptr,
        // QAT masks
        get_mask_ptr(x_mask),
        get_mask_ptr(h0_mask),
        get_mask_ptr(W_mask),
        get_mask_ptr(R_mask),
        get_mask_ptr(bw_mask),
        get_mask_ptr(br_mask),
        get_mask_ptr(weight_ih_linear_mask),
        get_mask_ptr(weight_hh_linear_mask),
        get_mask_ptr(gate_input_mask),
        get_mask_ptr(gate_output_mask),
        get_mask_ptr(h_mask));

    return std::make_tuple(dx, dW, dR, dbw, dbr, dh);
}

// ============================================================================
//                    OperatorQuantConfigPy 方法实现
// ============================================================================

// 默认构造函数：从 C++ OperatorQuantConfig 获取默认值
// 这样保证 Python 端和 C++ 端的默认值始终一致
OperatorQuantConfigPy::OperatorQuantConfigPy() {
    OperatorQuantConfig cpp_defaults;  // 使用 C++ 的默认值
    from_cpp(cpp_defaults);
}

// 转换为 C++ 结构体
OperatorQuantConfig OperatorQuantConfigPy::to_cpp() const {
    OperatorQuantConfig cfg;
    // 位宽配置（直接使用 is_unsigned 标志，与 C++ 语义一致）
    cfg.x_ = toBitwidth(x_, x_unsigned_);
    cfg.h_ = toBitwidth(h_, h_unsigned_);
    cfg.W_ = toBitwidth(W_, W_unsigned_);
    cfg.R_ = toBitwidth(R_, R_unsigned_);
    cfg.bw_ = toBitwidth(bw_, bw_unsigned_);
    cfg.br_ = toBitwidth(br_, br_unsigned_);
    cfg.weight_ih_linear_ = toBitwidth(weight_ih_linear_, weight_ih_linear_unsigned_);
    cfg.weight_hh_linear_ = toBitwidth(weight_hh_linear_, weight_hh_linear_unsigned_);
    cfg.update_gate_input_ = toBitwidth(update_gate_input_, update_gate_input_unsigned_);
    cfg.update_gate_output_ = toBitwidth(update_gate_output_, update_gate_output_unsigned_);
    cfg.reset_gate_input_ = toBitwidth(reset_gate_input_, reset_gate_input_unsigned_);
    cfg.reset_gate_output_ = toBitwidth(reset_gate_output_, reset_gate_output_unsigned_);
    cfg.new_gate_input_ = toBitwidth(new_gate_input_, new_gate_input_unsigned_);
    cfg.new_gate_output_ = toBitwidth(new_gate_output_, new_gate_output_unsigned_);
    cfg.mul_reset_hidden_ = toBitwidth(mul_reset_hidden_, mul_reset_hidden_unsigned_);
    cfg.mul_old_contribution_ = toBitwidth(mul_old_contribution_, mul_old_contribution_unsigned_);
    cfg.mul_new_contribution_ = toBitwidth(mul_new_contribution_, mul_new_contribution_unsigned_);
    // 对称量化配置
    cfg.x_symmetric_ = x_symmetric_;
    cfg.h_symmetric_ = h_symmetric_;
    cfg.W_symmetric_ = W_symmetric_;
    cfg.R_symmetric_ = R_symmetric_;
    cfg.bw_symmetric_ = bw_symmetric_;
    cfg.br_symmetric_ = br_symmetric_;
    cfg.weight_ih_linear_symmetric_ = weight_ih_linear_symmetric_;
    cfg.weight_hh_linear_symmetric_ = weight_hh_linear_symmetric_;
    cfg.update_gate_input_symmetric_ = update_gate_input_symmetric_;
    cfg.update_gate_output_symmetric_ = update_gate_output_symmetric_;
    cfg.reset_gate_input_symmetric_ = reset_gate_input_symmetric_;
    cfg.reset_gate_output_symmetric_ = reset_gate_output_symmetric_;
    cfg.new_gate_input_symmetric_ = new_gate_input_symmetric_;
    cfg.new_gate_output_symmetric_ = new_gate_output_symmetric_;
    cfg.mul_reset_hidden_symmetric_ = mul_reset_hidden_symmetric_;
    cfg.mul_old_contribution_symmetric_ = mul_old_contribution_symmetric_;
    cfg.mul_new_contribution_symmetric_ = mul_new_contribution_symmetric_;
    return cfg;
}

// 从 C++ 结构体读取
void OperatorQuantConfigPy::from_cpp(const OperatorQuantConfig &cfg) {
    // 位宽配置
    x_ = cfg.x_.bits_;
    h_ = cfg.h_.bits_;
    W_ = cfg.W_.bits_;
    R_ = cfg.R_.bits_;
    bw_ = cfg.bw_.bits_;
    br_ = cfg.br_.bits_;
    weight_ih_linear_ = cfg.weight_ih_linear_.bits_;
    weight_hh_linear_ = cfg.weight_hh_linear_.bits_;
    update_gate_input_ = cfg.update_gate_input_.bits_;
    update_gate_output_ = cfg.update_gate_output_.bits_;
    reset_gate_input_ = cfg.reset_gate_input_.bits_;
    reset_gate_output_ = cfg.reset_gate_output_.bits_;
    new_gate_input_ = cfg.new_gate_input_.bits_;
    new_gate_output_ = cfg.new_gate_output_.bits_;
    mul_reset_hidden_ = cfg.mul_reset_hidden_.bits_;
    mul_old_contribution_ = cfg.mul_old_contribution_.bits_;
    mul_new_contribution_ = cfg.mul_new_contribution_.bits_;
    // 对称量化配置
    x_symmetric_ = cfg.x_symmetric_;
    h_symmetric_ = cfg.h_symmetric_;
    W_symmetric_ = cfg.W_symmetric_;
    R_symmetric_ = cfg.R_symmetric_;
    bw_symmetric_ = cfg.bw_symmetric_;
    br_symmetric_ = cfg.br_symmetric_;
    weight_ih_linear_symmetric_ = cfg.weight_ih_linear_symmetric_;
    weight_hh_linear_symmetric_ = cfg.weight_hh_linear_symmetric_;
    update_gate_input_symmetric_ = cfg.update_gate_input_symmetric_;
    update_gate_output_symmetric_ = cfg.update_gate_output_symmetric_;
    reset_gate_input_symmetric_ = cfg.reset_gate_input_symmetric_;
    reset_gate_output_symmetric_ = cfg.reset_gate_output_symmetric_;
    new_gate_input_symmetric_ = cfg.new_gate_input_symmetric_;
    new_gate_output_symmetric_ = cfg.new_gate_output_symmetric_;
    mul_reset_hidden_symmetric_ = cfg.mul_reset_hidden_symmetric_;
    mul_old_contribution_symmetric_ = cfg.mul_old_contribution_symmetric_;
    mul_new_contribution_symmetric_ = cfg.mul_new_contribution_symmetric_;
    // 无符号配置（直接从 C++ is_unsigned_ 读取，无需转换）
    x_unsigned_ = cfg.x_.is_unsigned_;
    h_unsigned_ = cfg.h_.is_unsigned_;
    W_unsigned_ = cfg.W_.is_unsigned_;
    R_unsigned_ = cfg.R_.is_unsigned_;
    bw_unsigned_ = cfg.bw_.is_unsigned_;
    br_unsigned_ = cfg.br_.is_unsigned_;
    weight_ih_linear_unsigned_ = cfg.weight_ih_linear_.is_unsigned_;
    weight_hh_linear_unsigned_ = cfg.weight_hh_linear_.is_unsigned_;
    update_gate_input_unsigned_ = cfg.update_gate_input_.is_unsigned_;
    update_gate_output_unsigned_ = cfg.update_gate_output_.is_unsigned_;
    reset_gate_input_unsigned_ = cfg.reset_gate_input_.is_unsigned_;
    reset_gate_output_unsigned_ = cfg.reset_gate_output_.is_unsigned_;
    new_gate_input_unsigned_ = cfg.new_gate_input_.is_unsigned_;
    new_gate_output_unsigned_ = cfg.new_gate_output_.is_unsigned_;
    mul_reset_hidden_unsigned_ = cfg.mul_reset_hidden_.is_unsigned_;
    mul_old_contribution_unsigned_ = cfg.mul_old_contribution_.is_unsigned_;
    mul_new_contribution_unsigned_ = cfg.mul_new_contribution_.is_unsigned_;
}

// ============================================================================
//                    GRUQuantParamsPy 方法实现
// ============================================================================

// 从 C++ 结构体转换
void GRUQuantParamsPy::from_cpp(const GRUQuantParams &cpp_params) {
    hidden_ = cpp_params.hidden_;
    // 基础参数
    shift_x_ = cpp_params.shift_x_;
    zp_x_ = cpp_params.zp_x_;
    shift_h_ = cpp_params.shift_h_;
    zp_h_ = cpp_params.zp_h_;
    // 权重参数
    shift_W_ = cpp_params.shift_W_;
    shift_R_ = cpp_params.shift_R_;
    shift_bw_ = cpp_params.shift_bw_;
    shift_br_ = cpp_params.shift_br_;
    // Linear 输出参数
    shift_weight_ih_linear_ = cpp_params.shift_weight_ih_linear_;
    zp_weight_ih_linear_ = cpp_params.zp_weight_ih_linear_;
    shift_weight_hh_linear_ = cpp_params.shift_weight_hh_linear_;
    zp_weight_hh_linear_ = cpp_params.zp_weight_hh_linear_;
    // 门激活函数输入参数
    shift_update_gate_input_ = cpp_params.shift_update_gate_input_;
    zp_update_gate_input_ = cpp_params.zp_update_gate_input_;
    shift_reset_gate_input_ = cpp_params.shift_reset_gate_input_;
    zp_reset_gate_input_ = cpp_params.zp_reset_gate_input_;
    shift_new_gate_input_ = cpp_params.shift_new_gate_input_;
    zp_new_gate_input_ = cpp_params.zp_new_gate_input_;
    // 门激活函数输出参数
    shift_update_gate_output_ = cpp_params.shift_update_gate_output_;
    zp_update_gate_output_ = cpp_params.zp_update_gate_output_;
    shift_reset_gate_output_ = cpp_params.shift_reset_gate_output_;
    zp_reset_gate_output_ = cpp_params.zp_reset_gate_output_;
    shift_new_gate_output_ = cpp_params.shift_new_gate_output_;
    zp_new_gate_output_ = cpp_params.zp_new_gate_output_;
    // 中间计算参数
    shift_mul_reset_hidden_ = cpp_params.shift_mul_reset_hidden_;
    zp_mul_reset_hidden_ = cpp_params.zp_mul_reset_hidden_;
    // 隐状态更新参数
    shift_mul_new_contribution_ = cpp_params.shift_mul_new_contribution_;
    zp_mul_new_contribution_ = cpp_params.zp_mul_new_contribution_;
    shift_mul_old_contribution_ = cpp_params.shift_mul_old_contribution_;
    zp_mul_old_contribution_ = cpp_params.zp_mul_old_contribution_;

    // ⚠️ 关键：复制位宽配置
    bitwidth_config_.from_cpp(cpp_params.bitwidth_config_);
}

// 转换为 C++ 结构体
GRUQuantParams GRUQuantParamsPy::to_cpp() const {
    GRUQuantParams cpp_params;
    cpp_params.hidden_ = hidden_;
    // 基础参数
    cpp_params.shift_x_ = shift_x_;
    cpp_params.zp_x_ = zp_x_;
    cpp_params.shift_h_ = shift_h_;
    cpp_params.zp_h_ = zp_h_;
    // 权重参数
    cpp_params.shift_W_ = shift_W_;
    cpp_params.shift_R_ = shift_R_;
    cpp_params.shift_bw_ = shift_bw_;
    cpp_params.shift_br_ = shift_br_;
    // Linear 输出参数
    cpp_params.shift_weight_ih_linear_ = shift_weight_ih_linear_;
    cpp_params.zp_weight_ih_linear_ = zp_weight_ih_linear_;
    cpp_params.shift_weight_hh_linear_ = shift_weight_hh_linear_;
    cpp_params.zp_weight_hh_linear_ = zp_weight_hh_linear_;
    // 门激活函数输入参数
    cpp_params.shift_update_gate_input_ = shift_update_gate_input_;
    cpp_params.zp_update_gate_input_ = zp_update_gate_input_;
    cpp_params.shift_reset_gate_input_ = shift_reset_gate_input_;
    cpp_params.zp_reset_gate_input_ = zp_reset_gate_input_;
    cpp_params.shift_new_gate_input_ = shift_new_gate_input_;
    cpp_params.zp_new_gate_input_ = zp_new_gate_input_;
    // 门激活函数输出参数
    cpp_params.shift_update_gate_output_ = shift_update_gate_output_;
    cpp_params.zp_update_gate_output_ = zp_update_gate_output_;
    cpp_params.shift_reset_gate_output_ = shift_reset_gate_output_;
    cpp_params.zp_reset_gate_output_ = zp_reset_gate_output_;
    cpp_params.shift_new_gate_output_ = shift_new_gate_output_;
    cpp_params.zp_new_gate_output_ = zp_new_gate_output_;
    // 中间计算参数
    cpp_params.shift_mul_reset_hidden_ = shift_mul_reset_hidden_;
    cpp_params.zp_mul_reset_hidden_ = zp_mul_reset_hidden_;
    // 隐状态更新参数
    cpp_params.shift_mul_new_contribution_ = shift_mul_new_contribution_;
    cpp_params.zp_mul_new_contribution_ = zp_mul_new_contribution_;
    cpp_params.shift_mul_old_contribution_ = shift_mul_old_contribution_;
    cpp_params.zp_mul_old_contribution_ = zp_mul_old_contribution_;

    // ⚠️ 关键：复制位宽配置
    cpp_params.bitwidth_config_ = bitwidth_config_.to_cpp();

    // 重新生成 LUT（因为 LUT 不能直接序列化到 Python）
    // 这会在每次 to_cpp() 时重新生成，但开销可以接受（只在 forward 前调用一次）
    generate_piecewise_linear_lut_to_params(cpp_params);

    return cpp_params;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GRU Interface Python Bindings";

    // 初始化 cublas handle
    m.def("init_gru_cublas", &init_gru_cublas_wrapper, "Initialize cuBLAS handle for GRU");

    // GRUQuantizationRanges 绑定（通过 lambda 访问内部 C++ 对象的字段）
    // 宏定义简化属性绑定
    #define DEF_PROP(name) \
        .def_property(#name, \
            [](const GRUQuantizationRangesPy &self) { return self.cpp_ranges.name; }, \
            [](GRUQuantizationRangesPy &self, decltype(GRUQuantizationRanges::name) v) { self.cpp_ranges.name = v; })

    py::class_<GRUQuantizationRangesPy>(m, "GRUQuantizationRanges")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("hidden"))
        DEF_PROP(hidden_)
        DEF_PROP(min_x_)
        DEF_PROP(max_x_)
        DEF_PROP(min_h_)
        DEF_PROP(max_h_)
        DEF_PROP(min_W_)
        DEF_PROP(max_W_)
        DEF_PROP(min_R_)
        DEF_PROP(max_R_)
        DEF_PROP(min_Wx_)
        DEF_PROP(max_Wx_)
        DEF_PROP(min_Rh_)
        DEF_PROP(max_Rh_)
        DEF_PROP(min_bw_)
        DEF_PROP(max_bw_)
        DEF_PROP(min_br_)
        DEF_PROP(max_br_)
        DEF_PROP(min_update_gate_input_)
        DEF_PROP(max_update_gate_input_)
        DEF_PROP(min_reset_gate_input_)
        DEF_PROP(max_reset_gate_input_)
        DEF_PROP(min_new_gate_input_)
        DEF_PROP(max_new_gate_input_)
        DEF_PROP(min_update_gate_output_)
        DEF_PROP(max_update_gate_output_)
        DEF_PROP(min_reset_gate_output_)
        DEF_PROP(max_reset_gate_output_)
        DEF_PROP(min_new_gate_output_)
        DEF_PROP(max_new_gate_output_)
        DEF_PROP(min_mul_reset_hidden_)
        DEF_PROP(max_mul_reset_hidden_)
        DEF_PROP(min_mul_new_contribution_)
        DEF_PROP(max_mul_new_contribution_)
        DEF_PROP(min_mul_old_contribution_)
        DEF_PROP(max_mul_old_contribution_)
        .def("reset", &GRUQuantizationRangesPy::reset,
             "Reset all ranges to invalid values. If hidden > 0, also update hidden_ and resize "
             "per-channel vectors.",
             py::arg("hidden") = -1);

    #undef DEF_PROP

    // OperatorQuantConfig 绑定（位宽配置 + 对称量化配置）
    // 位宽值: 1-32（Python 端只看到位宽数量，C++ 端决定实际类型）
    // 对称量化: is_symmetric=true 对称量化(zp=0), is_symmetric=false 非对称量化(zp≠0)
    py::class_<OperatorQuantConfigPy>(m, "OperatorQuantConfig")
        .def(py::init<>())
        // 位宽配置
        .def_readwrite("x_", &OperatorQuantConfigPy::x_)
        .def_readwrite("h_", &OperatorQuantConfigPy::h_)
        .def_readwrite("W_", &OperatorQuantConfigPy::W_)
        .def_readwrite("R_", &OperatorQuantConfigPy::R_)
        .def_readwrite("bw_", &OperatorQuantConfigPy::bw_)
        .def_readwrite("br_", &OperatorQuantConfigPy::br_)
        .def_readwrite("weight_ih_linear_", &OperatorQuantConfigPy::weight_ih_linear_)
        .def_readwrite("weight_hh_linear_", &OperatorQuantConfigPy::weight_hh_linear_)
        .def_readwrite("update_gate_input_", &OperatorQuantConfigPy::update_gate_input_)
        .def_readwrite("update_gate_output_", &OperatorQuantConfigPy::update_gate_output_)
        .def_readwrite("reset_gate_input_", &OperatorQuantConfigPy::reset_gate_input_)
        .def_readwrite("reset_gate_output_", &OperatorQuantConfigPy::reset_gate_output_)
        .def_readwrite("new_gate_input_", &OperatorQuantConfigPy::new_gate_input_)
        .def_readwrite("new_gate_output_", &OperatorQuantConfigPy::new_gate_output_)
        .def_readwrite("mul_reset_hidden_", &OperatorQuantConfigPy::mul_reset_hidden_)
        .def_readwrite("mul_old_contribution_", &OperatorQuantConfigPy::mul_old_contribution_)
        .def_readwrite("mul_new_contribution_", &OperatorQuantConfigPy::mul_new_contribution_)
        // 对称量化配置
        .def_readwrite("x_symmetric_", &OperatorQuantConfigPy::x_symmetric_)
        .def_readwrite("h_symmetric_", &OperatorQuantConfigPy::h_symmetric_)
        .def_readwrite("W_symmetric_", &OperatorQuantConfigPy::W_symmetric_)
        .def_readwrite("R_symmetric_", &OperatorQuantConfigPy::R_symmetric_)
        .def_readwrite("bw_symmetric_", &OperatorQuantConfigPy::bw_symmetric_)
        .def_readwrite("br_symmetric_", &OperatorQuantConfigPy::br_symmetric_)
        .def_readwrite("weight_ih_linear_symmetric_", &OperatorQuantConfigPy::weight_ih_linear_symmetric_)
        .def_readwrite("weight_hh_linear_symmetric_", &OperatorQuantConfigPy::weight_hh_linear_symmetric_)
        .def_readwrite("update_gate_input_symmetric_", &OperatorQuantConfigPy::update_gate_input_symmetric_)
        .def_readwrite("update_gate_output_symmetric_", &OperatorQuantConfigPy::update_gate_output_symmetric_)
        .def_readwrite("reset_gate_input_symmetric_", &OperatorQuantConfigPy::reset_gate_input_symmetric_)
        .def_readwrite("reset_gate_output_symmetric_", &OperatorQuantConfigPy::reset_gate_output_symmetric_)
        .def_readwrite("new_gate_input_symmetric_", &OperatorQuantConfigPy::new_gate_input_symmetric_)
        .def_readwrite("new_gate_output_symmetric_", &OperatorQuantConfigPy::new_gate_output_symmetric_)
        .def_readwrite("mul_reset_hidden_symmetric_", &OperatorQuantConfigPy::mul_reset_hidden_symmetric_)
        .def_readwrite("mul_old_contribution_symmetric_", &OperatorQuantConfigPy::mul_old_contribution_symmetric_)
        .def_readwrite("mul_new_contribution_symmetric_", &OperatorQuantConfigPy::mul_new_contribution_symmetric_)
        // 无符号配置（与 C++ is_unsigned_ 语义一致，只标记例外）
        .def_readwrite("x_unsigned_", &OperatorQuantConfigPy::x_unsigned_)
        .def_readwrite("h_unsigned_", &OperatorQuantConfigPy::h_unsigned_)
        .def_readwrite("W_unsigned_", &OperatorQuantConfigPy::W_unsigned_)
        .def_readwrite("R_unsigned_", &OperatorQuantConfigPy::R_unsigned_)
        .def_readwrite("bw_unsigned_", &OperatorQuantConfigPy::bw_unsigned_)
        .def_readwrite("br_unsigned_", &OperatorQuantConfigPy::br_unsigned_)
        .def_readwrite("weight_ih_linear_unsigned_", &OperatorQuantConfigPy::weight_ih_linear_unsigned_)
        .def_readwrite("weight_hh_linear_unsigned_", &OperatorQuantConfigPy::weight_hh_linear_unsigned_)
        .def_readwrite("update_gate_input_unsigned_", &OperatorQuantConfigPy::update_gate_input_unsigned_)
        .def_readwrite("update_gate_output_unsigned_", &OperatorQuantConfigPy::update_gate_output_unsigned_)
        .def_readwrite("reset_gate_input_unsigned_", &OperatorQuantConfigPy::reset_gate_input_unsigned_)
        .def_readwrite("reset_gate_output_unsigned_", &OperatorQuantConfigPy::reset_gate_output_unsigned_)
        .def_readwrite("new_gate_input_unsigned_", &OperatorQuantConfigPy::new_gate_input_unsigned_)
        .def_readwrite("new_gate_output_unsigned_", &OperatorQuantConfigPy::new_gate_output_unsigned_)
        .def_readwrite("mul_reset_hidden_unsigned_", &OperatorQuantConfigPy::mul_reset_hidden_unsigned_)
        .def_readwrite("mul_old_contribution_unsigned_", &OperatorQuantConfigPy::mul_old_contribution_unsigned_)
        .def_readwrite("mul_new_contribution_unsigned_", &OperatorQuantConfigPy::mul_new_contribution_unsigned_);

    // GRUQuantParams 绑定
    py::class_<GRUQuantParamsPy>(m, "GRUQuantParams")
        .def(py::init<>())
        .def_readwrite("hidden_", &GRUQuantParamsPy::hidden_)
        // 基础参数
        .def_readwrite("shift_x_", &GRUQuantParamsPy::shift_x_)
        .def_readwrite("zp_x_", &GRUQuantParamsPy::zp_x_)
        .def_readwrite("shift_h_", &GRUQuantParamsPy::shift_h_)
        .def_readwrite("zp_h_", &GRUQuantParamsPy::zp_h_)
        // 权重参数（per-channel）
        .def_readwrite("shift_W_", &GRUQuantParamsPy::shift_W_)
        .def_readwrite("shift_R_", &GRUQuantParamsPy::shift_R_)
        .def_readwrite("shift_bw_", &GRUQuantParamsPy::shift_bw_)
        .def_readwrite("shift_br_", &GRUQuantParamsPy::shift_br_)
        // Linear 输出参数 (GEMM+bias)
        .def_readwrite("shift_weight_ih_linear_", &GRUQuantParamsPy::shift_weight_ih_linear_)
        .def_readwrite("zp_weight_ih_linear_", &GRUQuantParamsPy::zp_weight_ih_linear_)
        .def_readwrite("shift_weight_hh_linear_", &GRUQuantParamsPy::shift_weight_hh_linear_)
        .def_readwrite("zp_weight_hh_linear_", &GRUQuantParamsPy::zp_weight_hh_linear_)
        // 门激活函数输入参数（pre-activation）
        .def_readwrite("shift_update_gate_input_", &GRUQuantParamsPy::shift_update_gate_input_)
        .def_readwrite("zp_update_gate_input_", &GRUQuantParamsPy::zp_update_gate_input_)
        .def_readwrite("shift_reset_gate_input_", &GRUQuantParamsPy::shift_reset_gate_input_)
        .def_readwrite("zp_reset_gate_input_", &GRUQuantParamsPy::zp_reset_gate_input_)
        .def_readwrite("shift_new_gate_input_", &GRUQuantParamsPy::shift_new_gate_input_)
        .def_readwrite("zp_new_gate_input_", &GRUQuantParamsPy::zp_new_gate_input_)
        // 门激活函数输出参数（post-activation）
        .def_readwrite("shift_update_gate_output_", &GRUQuantParamsPy::shift_update_gate_output_)
        .def_readwrite("zp_update_gate_output_", &GRUQuantParamsPy::zp_update_gate_output_)
        .def_readwrite("shift_reset_gate_output_", &GRUQuantParamsPy::shift_reset_gate_output_)
        .def_readwrite("zp_reset_gate_output_", &GRUQuantParamsPy::zp_reset_gate_output_)
        .def_readwrite("shift_new_gate_output_", &GRUQuantParamsPy::shift_new_gate_output_)
        .def_readwrite("zp_new_gate_output_", &GRUQuantParamsPy::zp_new_gate_output_)
        // 中间计算参数
        .def_readwrite("shift_mul_reset_hidden_", &GRUQuantParamsPy::shift_mul_reset_hidden_)
        .def_readwrite("zp_mul_reset_hidden_", &GRUQuantParamsPy::zp_mul_reset_hidden_)
        // 隐状态更新参数
        .def_readwrite("shift_mul_new_contribution_", &GRUQuantParamsPy::shift_mul_new_contribution_)
        .def_readwrite("zp_mul_new_contribution_", &GRUQuantParamsPy::zp_mul_new_contribution_)
        .def_readwrite("shift_mul_old_contribution_", &GRUQuantParamsPy::shift_mul_old_contribution_)
        .def_readwrite("zp_mul_old_contribution_", &GRUQuantParamsPy::zp_mul_old_contribution_)
        // ⚠️ 关键字段：位宽配置，决定 forwardInterface 使用 int8 还是 int16
        .def_readwrite("bitwidth_config_", &GRUQuantParamsPy::bitwidth_config_);

    // 根据量化范围计算量化参数（支持自定义位宽配置）
    m.def("calculate_gru_quantitative_parameters", &calculate_gru_quantitative_parameters_wrapper,
          "Calculate GRU quantitative parameters from quantization ranges", py::arg("quant_ranges"),
          py::arg("bitwidth_config") = OperatorQuantConfigPy());

    // =====================================================================
    // AIMET 风格直方图校准接口（支持多批次累积，最高精度）
    // =====================================================================

    // GRUHistogramCollectors 绑定
    py::class_<GRUHistogramCollectorsPy>(m, "GRUHistogramCollectors")
        .def(py::init<>())
        .def(py::init<int, int>(), py::arg("hidden"), py::arg("num_bins") = 2048)
        .def("reset", &GRUHistogramCollectorsPy::reset,
             "Reset histogram collectors", py::arg("hidden") = -1, py::arg("num_bins") = -1)
        .def("is_valid", &GRUHistogramCollectorsPy::is_valid, "Check if histograms have valid data")
        .def("hidden", &GRUHistogramCollectorsPy::hidden, "Get hidden size")
        .def("num_bins", &GRUHistogramCollectorsPy::num_bins, "Get number of histogram bins")
        .def("print", &GRUHistogramCollectorsPy::print, "Print histogram statistics");

    // 从直方图计算量化参数（支持 SQNR 和 Percentile 两种方案）
    m.def("calculate_gru_quantitative_parameters_from_histograms",
          &calculate_gru_quantitative_parameters_from_histograms_wrapper,
          "Calculate GRU quantitative parameters from accumulated histograms.\n"
          "Supports two calibration schemes:\n"
          "  - SQNR (default): AIMET tf_enhanced style, searches optimal scale to minimize quantization noise\n"
          "  - Percentile: AIMET percentile style, uses percentile range for clipping",
          py::arg("hist_collectors"), py::arg("bitwidth_config") = OperatorQuantConfigPy(),
          py::arg("verbose") = false,
          py::arg("use_percentile") = false,
          py::arg("percentile_value") = 99.99f);

    // =====================================================================
    // forward: 正常前向传播（推理/训练，支持 QAT mask）
    // =====================================================================
    m.def("forward", &forward_wrapper,
          "GRU forward pass for inference/training with QAT mask support.\n"
          "\n"
          "Args:\n"
          "  is_training: Enable training mode (saves intermediate values)\n"
          "  is_quant: Use quantized inference (requires quant_params)\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight matrices and biases\n"
          "  x: Input tensor [T, B, I]\n"
          "  h0: Initial hidden state [B, H], optional\n"
          "  quant_params: Quantization parameters (used when is_quant=True)\n"
          "\n"
          "Returns:\n"
          "  tuple(h, v, x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,\n"
          "        weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask)\n"
          "  - h: Hidden states [T+1, B, H]\n"
          "  - v: Intermediate values [T, B, H*4]\n"
          "  Input quantization masks (empty if not training+quant):\n"
          "  - x_mask: [T, B, I] input sequence quantization mask\n"
          "  - h0_mask: [B, H] initial hidden state quantization mask\n"
          "  - W_mask: [I, H*3] input weight quantization mask\n"
          "  - R_mask: [H, H*3] recurrent weight quantization mask\n"
          "  - bw_mask: [H*3] input bias quantization mask\n"
          "  - br_mask: [H*3] recurrent bias quantization mask\n"
          "  Computation masks (empty if not training+quant):\n"
          "  - weight_ih_linear_mask: [T, B, H*3] (uint8, 1=clamped)\n"
          "  - weight_hh_linear_mask: [T, B, H*3] (uint8, 1=clamped)\n"
          "  - gate_input_mask: [T, B, H*3] gate input clamp mask (uint8, 1=clamped)\n"
          "  - gate_output_mask: [T, B, H*3] gate output clamp mask (uint8, 1=clamped)\n"
          "  - h_mask: [T, B, H] hidden state mask (uint8, 1=clamped)\n"
          "\n"
          "Note: QAT masks are only populated when is_training=True and is_quant=True.\n"
          "      Check mask.numel() > 0 to determine if masks are valid.",
          py::arg("is_training"), py::arg("is_quant"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor(),
          py::arg("quant_params"));

    // =====================================================================
    // forward_calibrate: 校准前向传播（统一接口，原地累积）
    // =====================================================================
    m.def("forward_calibrate", &forward_calibrate_wrapper,
          "GRU forward pass for quantization calibration.\n"
          "Calibration data is accumulated in-place via pointer parameters.\n"
          "\n"
          "Args:\n"
          "  is_training: Enable training mode\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight matrices and biases (float)\n"
          "  x: Input tensor [T, B, I]\n"
          "  h0: Initial hidden state [B, H], optional\n"
          "  calib_method: Calibration method ('minmax', 'sqnr', 'percentile')\n"
          "  quant_ranges: Required for 'minmax' mode (updated in-place)\n"
          "  hist_collectors: Required for 'sqnr'/'percentile' mode (updated in-place)\n"
          "\n"
          "Returns:\n"
          "  tuple(h, v)\n"
          "  - h: Hidden states [T+1, B, H]\n"
          "  - v: Intermediate values [T, B, H*4]\n"
          "\n"
          "Usage:\n"
          "  # MINMAX calibration:\n"
          "  ranges = GRUQuantizationRanges(hidden_size)\n"
          "  for batch in batches:\n"
          "      h, v = forward_calibrate(..., 'minmax', ranges)\n"
          "  params = calculate_gru_quantitative_parameters(ranges)\n"
          "  \n"
          "  # SQNR/Percentile calibration:\n"
          "  hist = GRUHistogramCollectors(hidden_size)\n"
          "  for batch in batches:\n"
          "      h, v = forward_calibrate(..., 'sqnr', None, hist)\n"
          "  params = calculate_gru_quantitative_parameters_from_histograms(hist)",
          py::arg("is_training"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor(),
          py::arg("calib_method"),
          py::arg("quant_ranges") = nullptr,
          py::arg("hist_collectors") = nullptr);

    // =====================================================================
    // quant_gru_forward_int32: 纯定点 GRU 前向传播（用于验证）
    // =====================================================================
    m.def("quant_gru_forward_int32", &quant_gru_forward_int32_wrapper,
          "Quantized GRU forward pass with int32 input/output for verification.\n"
          "\n"
          "Args:\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W_q, R_q, bw_q, br_q: Quantized weights (int32 CUDA tensors)\n"
          "  x_q: Quantized input [T, B, I] (int32 CUDA tensor)\n"
          "  h0_q: Quantized initial hidden state [B, H] (int32 CUDA tensor, optional)\n"
          "  quant_params: Quantization parameters\n"
          "\n"
          "Returns:\n"
          "  h_q: Quantized hidden states [T+1, B, H] (int32 CUDA tensor)",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W_q"), py::arg("R_q"), py::arg("bw_q"), py::arg("br_q"),
          py::arg("x_q"), py::arg("h0_q"),
          py::arg("quant_params"));

    // GRU 反向传播（非量化版，保留兼容性）
    m.def("haste_gru_backward", &haste_gru_backward_wrapper, "Non-quantized GRU backward pass",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"),
          py::arg("x"), py::arg("dh_new"), py::arg("h"),
          py::arg("v"));  // 中间值v，必需；返回 (dx, dW, dR, dbw, dbr, dh) 元组

    // GRU 统一反向传播接口（支持 QAT mask）
    m.def("backward", &backward_wrapper,
          "Unified GRU backward pass with optional QAT mask support.\n"
          "\n"
          "Args:\n"
          "  is_quant: Whether to use quantized backward (applies masks via STE)\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bw, br: Weight tensors in Haste format\n"
          "  x: Input tensor [T, B, I]\n"
          "  dh_new: Upstream gradient [T+1, B, H]\n"
          "  h: Hidden states from forward [T+1, B, H]\n"
          "  v: Intermediate values from forward [T, B, H*4]\n"
          "  quant_params: Quantization parameters (保留以保持接口兼容性，但不再使用)\n"
          "  x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask: Input/weight clamp masks\n"
          "  weight_ih_linear_mask, weight_hh_linear_mask: Linear output clamp masks\n"
          "  gate_input_mask, gate_output_mask: Gate input/output clamp masks\n"
          "  h_mask: Hidden state clamp mask\n"
          "\n"
          "Returns:\n"
          "  (dx, dW, dR, dbw, dbr, dh) gradient tuple\n"
          "\n"
          "Note:\n"
          "  - When is_quant=True, masks are applied in C++ via Straight-Through Estimator (STE)\n"
          "  - When is_quant=False, masks and quant_params are ignored\n"
          "  - No rescale compensation needed: backward pass uses dequantized float values, gradient computation is already correct",
          py::arg("is_quant"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bw"), py::arg("br"),
          py::arg("x"), py::arg("dh_new"), py::arg("h"), py::arg("v"),
          py::arg("quant_params"),
          py::arg("x_mask") = torch::Tensor(),
          py::arg("h0_mask") = torch::Tensor(),
          py::arg("W_mask") = torch::Tensor(),
          py::arg("R_mask") = torch::Tensor(),
          py::arg("bw_mask") = torch::Tensor(),
          py::arg("br_mask") = torch::Tensor(),
          py::arg("weight_ih_linear_mask") = torch::Tensor(),
          py::arg("weight_hh_linear_mask") = torch::Tensor(),
          py::arg("gate_input_mask") = torch::Tensor(),
          py::arg("gate_output_mask") = torch::Tensor(),
          py::arg("h_mask") = torch::Tensor());
}
