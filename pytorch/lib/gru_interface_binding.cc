#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <iostream>

#include "gru_interface.h"
#include "gru_quantization_ranges.h"
#include "histogram_collector.h"
#include "histogram_gpu.cuh"
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
// - Python 端只配置位宽数量（8, 16, 32），不关心实际类型（INT/UINT）
// - C++ 端在 to_cpp() 时决定实际类型
// - 这些辅助函数需要在 OperatorQuantConfigPy 之前定义
//
// ============================================================================

/// 位宽转换：Python 端的位宽数值 → C++ QuantBitWidth 结构体
inline QuantBitWidth bitwidthToSigned(int8_t bitwidth) {
    if (bitwidth < 1 || bitwidth > 32) {
        throw std::invalid_argument("Invalid bitwidth: " + std::to_string(bitwidth) +
                                    ". Must be 1-32.");
    }
    return QuantBitWidth(bitwidth, true);  // 有符号
}

inline QuantBitWidth bitwidthToUnsigned(int8_t bitwidth) {
    if (bitwidth < 1 || bitwidth > 32) {
        throw std::invalid_argument("Invalid bitwidth: " + std::to_string(bitwidth) +
                                    ". Must be 1-32.");
    }
    return QuantBitWidth(bitwidth, false);  // 无符号
}

// ============================================================================
//                    OperatorQuantConfig Python 绑定（前置定义）
// ============================================================================
//
// 注意：这个结构体需要在 GRUQuantitativeParametersPy 之前定义，
// 因为 GRUQuantitativeParametersPy 中包含 OperatorQuantConfigPy 成员
//
// ============================================================================

struct OperatorQuantConfigPy {
    // 位宽配置（存储位宽数量 8, 16, 32）
    int8_t x_ = 8, h_ = 8;
    int8_t W_ = 8, R_ = 8, bx_ = 8, br_ = 8;
    int8_t Wx_ = 8, Rh_ = 8;
    int8_t z_pre_ = 8, z_out_ = 8;
    int8_t r_pre_ = 8, r_out_ = 8;
    int8_t g_pre_ = 8, g_out_ = 8;
    int8_t Rh_add_br_ = 8, rRh_ = 8, old_contrib_ = 8, new_contrib_ = 8;

    // 对称量化配置
    bool x_symmetric_ = false, h_symmetric_ = false;
    bool W_symmetric_ = true, R_symmetric_ = true, bx_symmetric_ = true, br_symmetric_ = true;
    bool Wx_symmetric_ = false, Rh_symmetric_ = false;
    bool z_pre_symmetric_ = false, z_out_symmetric_ = false;
    bool r_pre_symmetric_ = false, r_out_symmetric_ = false;
    bool g_pre_symmetric_ = false, g_out_symmetric_ = false;
    bool Rh_add_br_symmetric_ = false, rRh_symmetric_ = false;
    bool old_contrib_symmetric_ = false, new_contrib_symmetric_ = false;

    OperatorQuantConfig to_cpp() const {
        OperatorQuantConfig cfg;
        // 位宽配置（sigmoid 输出为无符号，其他有符号）
        cfg.x_ = bitwidthToSigned(x_);
        cfg.h_ = bitwidthToSigned(h_);
        cfg.W_ = bitwidthToSigned(W_);
        cfg.R_ = bitwidthToSigned(R_);
        cfg.bx_ = bitwidthToSigned(bx_);
        cfg.br_ = bitwidthToSigned(br_);
        cfg.Wx_ = bitwidthToSigned(Wx_);
        cfg.Rh_ = bitwidthToSigned(Rh_);
        cfg.z_pre_ = bitwidthToSigned(z_pre_);
        cfg.z_out_ = bitwidthToUnsigned(z_out_);  // sigmoid → unsigned
        cfg.r_pre_ = bitwidthToSigned(r_pre_);
        cfg.r_out_ = bitwidthToUnsigned(r_out_);  // sigmoid → unsigned
        cfg.g_pre_ = bitwidthToSigned(g_pre_);
        cfg.g_out_ = bitwidthToSigned(g_out_);
        cfg.Rh_add_br_ = bitwidthToSigned(Rh_add_br_);
        cfg.rRh_ = bitwidthToSigned(rRh_);
        cfg.old_contrib_ = bitwidthToSigned(old_contrib_);
        cfg.new_contrib_ = bitwidthToSigned(new_contrib_);
        // 对称量化配置
        cfg.x_symmetric_ = x_symmetric_;
        cfg.h_symmetric_ = h_symmetric_;
        cfg.W_symmetric_ = W_symmetric_;
        cfg.R_symmetric_ = R_symmetric_;
        cfg.bx_symmetric_ = bx_symmetric_;
        cfg.br_symmetric_ = br_symmetric_;
        cfg.Wx_symmetric_ = Wx_symmetric_;
        cfg.Rh_symmetric_ = Rh_symmetric_;
        cfg.z_pre_symmetric_ = z_pre_symmetric_;
        cfg.z_out_symmetric_ = z_out_symmetric_;
        cfg.r_pre_symmetric_ = r_pre_symmetric_;
        cfg.r_out_symmetric_ = r_out_symmetric_;
        cfg.g_pre_symmetric_ = g_pre_symmetric_;
        cfg.g_out_symmetric_ = g_out_symmetric_;
        cfg.Rh_add_br_symmetric_ = Rh_add_br_symmetric_;
        cfg.rRh_symmetric_ = rRh_symmetric_;
        cfg.old_contrib_symmetric_ = old_contrib_symmetric_;
        cfg.new_contrib_symmetric_ = new_contrib_symmetric_;
        return cfg;
    }

    void from_cpp(const OperatorQuantConfig &cfg) {
        // 直接从 C++ 结构体读取位宽（忽略 is_signed，Python 端不关心）
        x_ = cfg.x_.bits_;
        h_ = cfg.h_.bits_;
        W_ = cfg.W_.bits_;
        R_ = cfg.R_.bits_;
        bx_ = cfg.bx_.bits_;
        br_ = cfg.br_.bits_;
        Wx_ = cfg.Wx_.bits_;
        Rh_ = cfg.Rh_.bits_;
        z_pre_ = cfg.z_pre_.bits_;
        z_out_ = cfg.z_out_.bits_;
        r_pre_ = cfg.r_pre_.bits_;
        r_out_ = cfg.r_out_.bits_;
        g_pre_ = cfg.g_pre_.bits_;
        g_out_ = cfg.g_out_.bits_;
        Rh_add_br_ = cfg.Rh_add_br_.bits_;
        rRh_ = cfg.rRh_.bits_;
        old_contrib_ = cfg.old_contrib_.bits_;
        new_contrib_ = cfg.new_contrib_.bits_;
        // 对称量化配置
        x_symmetric_ = cfg.x_symmetric_;
        h_symmetric_ = cfg.h_symmetric_;
        W_symmetric_ = cfg.W_symmetric_;
        R_symmetric_ = cfg.R_symmetric_;
        bx_symmetric_ = cfg.bx_symmetric_;
        br_symmetric_ = cfg.br_symmetric_;
        Wx_symmetric_ = cfg.Wx_symmetric_;
        Rh_symmetric_ = cfg.Rh_symmetric_;
        z_pre_symmetric_ = cfg.z_pre_symmetric_;
        z_out_symmetric_ = cfg.z_out_symmetric_;
        r_pre_symmetric_ = cfg.r_pre_symmetric_;
        r_out_symmetric_ = cfg.r_out_symmetric_;
        g_pre_symmetric_ = cfg.g_pre_symmetric_;
        g_out_symmetric_ = cfg.g_out_symmetric_;
        Rh_add_br_symmetric_ = cfg.Rh_add_br_symmetric_;
        rRh_symmetric_ = cfg.rRh_symmetric_;
        old_contrib_symmetric_ = cfg.old_contrib_symmetric_;
        new_contrib_symmetric_ = cfg.new_contrib_symmetric_;
    }
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

// GRUQuantitativeParameters 的 Python 绑定
struct GRUQuantitativeParametersPy {
    int hidden_;
    int8_t exp2_inv_x_;
    int32_t zp_x_;
    int8_t exp2_inv_h_;
    int32_t zp_h_;
    std::vector<int8_t> exp2_inv_W_;
    std::vector<int8_t> exp2_inv_R_;
    int8_t exp2_inv_Wx_;
    int32_t zp_Wx_;
    int8_t exp2_inv_Rh_;
    int32_t zp_Rh_;
    std::vector<int8_t> exp2_inv_bx_;
    std::vector<int8_t> exp2_inv_br_;
    int8_t exp2_inv_z_pre_;
    int32_t zp_z_pre_;
    int8_t exp2_inv_r_pre_;
    int32_t zp_r_pre_;
    int8_t exp2_inv_g_pre_;
    int32_t zp_g_pre_;
    int8_t exp2_inv_z_out_;
    int32_t zp_z_out_;
    int8_t exp2_inv_r_out_;
    int32_t zp_r_out_;
    int8_t exp2_inv_g_out_;
    int32_t zp_g_out_;
    int8_t exp2_inv_Rh_add_br_;
    int32_t zp_Rh_add_br_;
    int8_t exp2_inv_rRh_;
    int32_t zp_rRh_;
    int8_t exp2_inv_new_contrib_;
    int32_t zp_new_contrib_;
    int8_t exp2_inv_old_contrib_;
    int32_t zp_old_contrib_;

    // ⚠️ 关键字段：位宽配置必须在 Python 和 C++ 之间正确传递
    // 否则 forwardInterface 会使用默认的 8 位配置
    OperatorQuantConfigPy bitwidth_config_;

    // 从 C++ 结构体转换
    void from_cpp(const GRUQuantitativeParameters &cpp_params) {
        hidden_ = cpp_params.hidden_;
        exp2_inv_x_ = cpp_params.exp2_inv_x_;
        zp_x_ = cpp_params.zp_x_;
        exp2_inv_h_ = cpp_params.exp2_inv_h_;
        zp_h_ = cpp_params.zp_h_;
        exp2_inv_W_ = cpp_params.exp2_inv_W_;
        exp2_inv_R_ = cpp_params.exp2_inv_R_;
        exp2_inv_Wx_ = cpp_params.exp2_inv_Wx_;
        zp_Wx_ = cpp_params.zp_Wx_;
        exp2_inv_Rh_ = cpp_params.exp2_inv_Rh_;
        zp_Rh_ = cpp_params.zp_Rh_;
        exp2_inv_bx_ = cpp_params.exp2_inv_bx_;
        exp2_inv_br_ = cpp_params.exp2_inv_br_;
        exp2_inv_z_pre_ = cpp_params.exp2_inv_z_pre_;
        zp_z_pre_ = cpp_params.zp_z_pre_;
        exp2_inv_r_pre_ = cpp_params.exp2_inv_r_pre_;
        zp_r_pre_ = cpp_params.zp_r_pre_;
        exp2_inv_g_pre_ = cpp_params.exp2_inv_g_pre_;
        zp_g_pre_ = cpp_params.zp_g_pre_;
        exp2_inv_z_out_ = cpp_params.exp2_inv_z_out_;
        zp_z_out_ = cpp_params.zp_z_out_;
        exp2_inv_r_out_ = cpp_params.exp2_inv_r_out_;
        zp_r_out_ = cpp_params.zp_r_out_;
        exp2_inv_g_out_ = cpp_params.exp2_inv_g_out_;
        zp_g_out_ = cpp_params.zp_g_out_;
        exp2_inv_Rh_add_br_ = cpp_params.exp2_inv_Rh_add_br_;
        zp_Rh_add_br_ = cpp_params.zp_Rh_add_br_;
        exp2_inv_rRh_ = cpp_params.exp2_inv_rRh_;
        zp_rRh_ = cpp_params.zp_rRh_;
        exp2_inv_new_contrib_ = cpp_params.exp2_inv_new_contrib_;
        zp_new_contrib_ = cpp_params.zp_new_contrib_;
        exp2_inv_old_contrib_ = cpp_params.exp2_inv_old_contrib_;
        zp_old_contrib_ = cpp_params.zp_old_contrib_;

        // ⚠️ 关键：复制位宽配置
        bitwidth_config_.from_cpp(cpp_params.bitwidth_config_);
    }

    // 转换为 C++ 结构体
    GRUQuantitativeParameters to_cpp() const {
        GRUQuantitativeParameters cpp_params;
        cpp_params.hidden_ = hidden_;
        cpp_params.exp2_inv_x_ = exp2_inv_x_;
        cpp_params.zp_x_ = zp_x_;
        cpp_params.exp2_inv_h_ = exp2_inv_h_;
        cpp_params.zp_h_ = zp_h_;
        cpp_params.exp2_inv_W_ = exp2_inv_W_;
        cpp_params.exp2_inv_R_ = exp2_inv_R_;
        cpp_params.exp2_inv_Wx_ = exp2_inv_Wx_;
        cpp_params.zp_Wx_ = zp_Wx_;
        cpp_params.exp2_inv_Rh_ = exp2_inv_Rh_;
        cpp_params.zp_Rh_ = zp_Rh_;
        cpp_params.exp2_inv_bx_ = exp2_inv_bx_;
        cpp_params.exp2_inv_br_ = exp2_inv_br_;
        cpp_params.exp2_inv_z_pre_ = exp2_inv_z_pre_;
        cpp_params.zp_z_pre_ = zp_z_pre_;
        cpp_params.exp2_inv_r_pre_ = exp2_inv_r_pre_;
        cpp_params.zp_r_pre_ = zp_r_pre_;
        cpp_params.exp2_inv_g_pre_ = exp2_inv_g_pre_;
        cpp_params.zp_g_pre_ = zp_g_pre_;
        cpp_params.exp2_inv_z_out_ = exp2_inv_z_out_;
        cpp_params.zp_z_out_ = zp_z_out_;
        cpp_params.exp2_inv_r_out_ = exp2_inv_r_out_;
        cpp_params.zp_r_out_ = zp_r_out_;
        cpp_params.exp2_inv_g_out_ = exp2_inv_g_out_;
        cpp_params.zp_g_out_ = zp_g_out_;
        cpp_params.exp2_inv_Rh_add_br_ = exp2_inv_Rh_add_br_;
        cpp_params.zp_Rh_add_br_ = zp_Rh_add_br_;
        cpp_params.exp2_inv_rRh_ = exp2_inv_rRh_;
        cpp_params.zp_rRh_ = zp_rRh_;
        cpp_params.exp2_inv_new_contrib_ = exp2_inv_new_contrib_;
        cpp_params.zp_new_contrib_ = zp_new_contrib_;
        cpp_params.exp2_inv_old_contrib_ = exp2_inv_old_contrib_;
        cpp_params.zp_old_contrib_ = zp_old_contrib_;

        // ⚠️ 关键：复制位宽配置
        cpp_params.bitwidth_config_ = bitwidth_config_.to_cpp();

        // 重新生成 LUT（因为 LUT 不能直接序列化到 Python）
        // 这会在每次 to_cpp() 时重新生成，但开销可以接受（只在 forward 前调用一次）
        generate_piecewise_linear_lut_to_params(cpp_params);

        return cpp_params;
    }
};

// 根据量化范围计算量化参数的包装函数（支持自定义位宽配置）
GRUQuantitativeParametersPy calculate_gru_quantitative_parameters_wrapper(
    const GRUQuantizationRangesPy &quant_ranges,
    const OperatorQuantConfigPy &bitwidth_config = OperatorQuantConfigPy()) {
    // 直接使用包装的 C++ 对象
    OperatorQuantConfig cpp_bitwidth = bitwidth_config.to_cpp();

    // 调用 C++ 函数
    GRUQuantitativeParameters quant_params =
        calculateGRUQuantitativeParameters(quant_ranges.cpp_ranges, cpp_bitwidth);

    GRUQuantitativeParametersPy py_params;
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
GRUQuantitativeParametersPy calculate_gru_quantitative_parameters_from_histograms_wrapper(
    GRUHistogramCollectorsPy &hist_collectors,
    const OperatorQuantConfigPy &bitwidth_config = OperatorQuantConfigPy(),
    bool verbose = false,
    bool use_percentile = false,
    float percentile_value = 99.99f) {

    OperatorQuantConfig cpp_bitwidth = bitwidth_config.to_cpp();
    GRUQuantitativeParameters quant_params;
    
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

    GRUQuantitativeParametersPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// =====================================================================
// forward: 正常前向传播（推理/训练）
// =====================================================================
// 返回: (h, v)
//   h: [T+1, B, H] 隐藏状态序列（包含初始状态）
//   v: [T, B, H*4] 中间值（训练时需要，推理时可忽略）
std::tuple<torch::Tensor, torch::Tensor> forward_wrapper(
    bool is_training,  // 是否开启训练模式
    bool is_quant,     // 是否使用量化推理
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bx,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &h0,  // 初始隐藏状态，可以为空张量
    const GRUQuantitativeParametersPy &quant_params) {
    
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
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

    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();
    
    forwardInterface(is_training, is_quant, time_steps, batch_size, input_size, hidden_size,
                     W.data_ptr<float>(), R.data_ptr<float>(), 
                     bx.data_ptr<float>(), br.data_ptr<float>(), 
                     x.data_ptr<float>(), h0_ptr, cpp_params, g_blas_handle,
                     h.data_ptr<float>(), v.data_ptr<float>());

    return std::make_tuple(h, v);
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
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bx,
    const torch::Tensor &br, const torch::Tensor &x,
    const torch::Tensor &h0,
    const std::string &calib_method_str,  // 校准方法: 'minmax', 'sqnr', 'percentile'
    GRUQuantizationRangesPy *quant_ranges = nullptr,      // MINMAX 需要
    GRUHistogramCollectorsPy *hist_collectors = nullptr) {  // SQNR/Percentile 需要
    
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
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
        bx.data_ptr<float>(), br.data_ptr<float>(), 
        x.data_ptr<float>(), h0_ptr, g_blas_handle,
        calib_method,
        quant_ranges ? &(quant_ranges->cpp_ranges) : nullptr,
        hist_collectors ? &(hist_collectors->gpu_collectors) : nullptr,
        h.data_ptr<float>(), v.data_ptr<float>());

    return std::make_tuple(h, v);
}

// GRU 反向传播包装函数
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
haste_gru_backward_wrapper(int time_steps, int batch_size, int input_size, int hidden_size,
                           const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bx,
                           const torch::Tensor &br, const torch::Tensor &x,
                           const torch::Tensor &dh_new,  // 来自上层网络或损失函数的反向梯度
                           const torch::Tensor &h,       // 前向传播的隐藏状态
                           const torch::Tensor &v) {     // 前向传播的中间值，必需

    // 检查输入张量的类型和设备
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
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

    TORCH_CHECK(bx.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "bx must have shape [hidden_size * 3]");
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
    auto dbx = torch::zeros({hidden_size * 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
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
                     bx.data_ptr<float>(), br.data_ptr<float>(),
                     x_t.data_ptr<float>(),  // [I, T, B] - 转置后的 x
                     dh_new.data_ptr<float>(), h.data_ptr<float>(), v.data_ptr<float>(),
                     g_blas_handle, dx.data_ptr<float>(), dW.data_ptr<float>(),
                     dR.data_ptr<float>(), dbx.data_ptr<float>(), dbr.data_ptr<float>(),
                     dh.data_ptr<float>());

    return std::make_tuple(dx, dW, dR, dbx, dbr, dh);
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
        DEF_PROP(min_bx_)
        DEF_PROP(max_bx_)
        DEF_PROP(min_br_)
        DEF_PROP(max_br_)
        DEF_PROP(min_z_pre_)
        DEF_PROP(max_z_pre_)
        DEF_PROP(min_r_pre_)
        DEF_PROP(max_r_pre_)
        DEF_PROP(min_g_pre_)
        DEF_PROP(max_g_pre_)
        DEF_PROP(min_z_out_)
        DEF_PROP(max_z_out_)
        DEF_PROP(min_r_out_)
        DEF_PROP(max_r_out_)
        DEF_PROP(min_g_out_)
        DEF_PROP(max_g_out_)
        DEF_PROP(min_Rh_add_br_g_)
        DEF_PROP(max_Rh_add_br_g_)
        DEF_PROP(min_rRh_)
        DEF_PROP(max_rRh_)
        DEF_PROP(min_new_contrib_)
        DEF_PROP(max_new_contrib_)
        DEF_PROP(min_old_contrib_)
        DEF_PROP(max_old_contrib_)
        .def("reset", &GRUQuantizationRangesPy::reset,
             "Reset all ranges to invalid values. If hidden > 0, also update hidden_ and resize "
             "per-channel vectors.",
             py::arg("hidden") = -1);

    #undef DEF_PROP

    // OperatorQuantConfig 绑定（位宽配置 + 对称量化配置）
    // 位宽值: 8, 16, 32（Python 端只看到位宽数量，C++ 端决定实际类型）
    // 对称量化: is_symmetric=true 对称量化(zp=0), is_symmetric=false 非对称量化(zp≠0)
    py::class_<OperatorQuantConfigPy>(m, "OperatorQuantConfig")
        .def(py::init<>())
        // 位宽配置
        .def_readwrite("x_", &OperatorQuantConfigPy::x_)
        .def_readwrite("h_", &OperatorQuantConfigPy::h_)
        .def_readwrite("W_", &OperatorQuantConfigPy::W_)
        .def_readwrite("R_", &OperatorQuantConfigPy::R_)
        .def_readwrite("bx_", &OperatorQuantConfigPy::bx_)
        .def_readwrite("br_", &OperatorQuantConfigPy::br_)
        .def_readwrite("Wx_", &OperatorQuantConfigPy::Wx_)
        .def_readwrite("Rh_", &OperatorQuantConfigPy::Rh_)
        .def_readwrite("z_pre_", &OperatorQuantConfigPy::z_pre_)
        .def_readwrite("z_out_", &OperatorQuantConfigPy::z_out_)
        .def_readwrite("r_pre_", &OperatorQuantConfigPy::r_pre_)
        .def_readwrite("r_out_", &OperatorQuantConfigPy::r_out_)
        .def_readwrite("g_pre_", &OperatorQuantConfigPy::g_pre_)
        .def_readwrite("g_out_", &OperatorQuantConfigPy::g_out_)
        .def_readwrite("Rh_add_br_", &OperatorQuantConfigPy::Rh_add_br_)
        .def_readwrite("rRh_", &OperatorQuantConfigPy::rRh_)
        .def_readwrite("old_contrib_", &OperatorQuantConfigPy::old_contrib_)
        .def_readwrite("new_contrib_", &OperatorQuantConfigPy::new_contrib_)
        // 对称量化配置
        .def_readwrite("x_symmetric_", &OperatorQuantConfigPy::x_symmetric_)
        .def_readwrite("h_symmetric_", &OperatorQuantConfigPy::h_symmetric_)
        .def_readwrite("W_symmetric_", &OperatorQuantConfigPy::W_symmetric_)
        .def_readwrite("R_symmetric_", &OperatorQuantConfigPy::R_symmetric_)
        .def_readwrite("bx_symmetric_", &OperatorQuantConfigPy::bx_symmetric_)
        .def_readwrite("br_symmetric_", &OperatorQuantConfigPy::br_symmetric_)
        .def_readwrite("Wx_symmetric_", &OperatorQuantConfigPy::Wx_symmetric_)
        .def_readwrite("Rh_symmetric_", &OperatorQuantConfigPy::Rh_symmetric_)
        .def_readwrite("z_pre_symmetric_", &OperatorQuantConfigPy::z_pre_symmetric_)
        .def_readwrite("z_out_symmetric_", &OperatorQuantConfigPy::z_out_symmetric_)
        .def_readwrite("r_pre_symmetric_", &OperatorQuantConfigPy::r_pre_symmetric_)
        .def_readwrite("r_out_symmetric_", &OperatorQuantConfigPy::r_out_symmetric_)
        .def_readwrite("g_pre_symmetric_", &OperatorQuantConfigPy::g_pre_symmetric_)
        .def_readwrite("g_out_symmetric_", &OperatorQuantConfigPy::g_out_symmetric_)
        .def_readwrite("Rh_add_br_symmetric_", &OperatorQuantConfigPy::Rh_add_br_symmetric_)
        .def_readwrite("rRh_symmetric_", &OperatorQuantConfigPy::rRh_symmetric_)
        .def_readwrite("old_contrib_symmetric_", &OperatorQuantConfigPy::old_contrib_symmetric_)
        .def_readwrite("new_contrib_symmetric_", &OperatorQuantConfigPy::new_contrib_symmetric_);

    // GRUQuantitativeParameters 绑定
    py::class_<GRUQuantitativeParametersPy>(m, "GRUQuantitativeParameters")
        .def(py::init<>())
        .def_readwrite("hidden_", &GRUQuantitativeParametersPy::hidden_)
        .def_readwrite("exp2_inv_x_", &GRUQuantitativeParametersPy::exp2_inv_x_)
        .def_readwrite("zp_x_", &GRUQuantitativeParametersPy::zp_x_)
        .def_readwrite("exp2_inv_h_", &GRUQuantitativeParametersPy::exp2_inv_h_)
        .def_readwrite("zp_h_", &GRUQuantitativeParametersPy::zp_h_)
        .def_readwrite("exp2_inv_W_", &GRUQuantitativeParametersPy::exp2_inv_W_)
        .def_readwrite("exp2_inv_R_", &GRUQuantitativeParametersPy::exp2_inv_R_)
        .def_readwrite("exp2_inv_Wx_", &GRUQuantitativeParametersPy::exp2_inv_Wx_)
        .def_readwrite("zp_Wx_", &GRUQuantitativeParametersPy::zp_Wx_)
        .def_readwrite("exp2_inv_Rh_", &GRUQuantitativeParametersPy::exp2_inv_Rh_)
        .def_readwrite("zp_Rh_", &GRUQuantitativeParametersPy::zp_Rh_)
        .def_readwrite("exp2_inv_bx_", &GRUQuantitativeParametersPy::exp2_inv_bx_)
        .def_readwrite("exp2_inv_br_", &GRUQuantitativeParametersPy::exp2_inv_br_)
        .def_readwrite("exp2_inv_z_pre_", &GRUQuantitativeParametersPy::exp2_inv_z_pre_)
        .def_readwrite("zp_z_pre_", &GRUQuantitativeParametersPy::zp_z_pre_)
        .def_readwrite("exp2_inv_r_pre_", &GRUQuantitativeParametersPy::exp2_inv_r_pre_)
        .def_readwrite("zp_r_pre_", &GRUQuantitativeParametersPy::zp_r_pre_)
        .def_readwrite("exp2_inv_g_pre_", &GRUQuantitativeParametersPy::exp2_inv_g_pre_)
        .def_readwrite("zp_g_pre_", &GRUQuantitativeParametersPy::zp_g_pre_)
        .def_readwrite("exp2_inv_z_out_", &GRUQuantitativeParametersPy::exp2_inv_z_out_)
        .def_readwrite("zp_z_out_", &GRUQuantitativeParametersPy::zp_z_out_)
        .def_readwrite("exp2_inv_r_out_", &GRUQuantitativeParametersPy::exp2_inv_r_out_)
        .def_readwrite("zp_r_out_", &GRUQuantitativeParametersPy::zp_r_out_)
        .def_readwrite("exp2_inv_g_out_", &GRUQuantitativeParametersPy::exp2_inv_g_out_)
        .def_readwrite("zp_g_out_", &GRUQuantitativeParametersPy::zp_g_out_)
        .def_readwrite("exp2_inv_Rh_add_br_", &GRUQuantitativeParametersPy::exp2_inv_Rh_add_br_)
        .def_readwrite("zp_Rh_add_br_", &GRUQuantitativeParametersPy::zp_Rh_add_br_)
        .def_readwrite("exp2_inv_rRh_", &GRUQuantitativeParametersPy::exp2_inv_rRh_)
        .def_readwrite("zp_rRh_", &GRUQuantitativeParametersPy::zp_rRh_)
        .def_readwrite("exp2_inv_new_contrib_", &GRUQuantitativeParametersPy::exp2_inv_new_contrib_)
        .def_readwrite("zp_new_contrib_", &GRUQuantitativeParametersPy::zp_new_contrib_)
        .def_readwrite("exp2_inv_old_contrib_", &GRUQuantitativeParametersPy::exp2_inv_old_contrib_)
        .def_readwrite("zp_old_contrib_", &GRUQuantitativeParametersPy::zp_old_contrib_)
        // ⚠️ 关键字段：位宽配置，决定 forwardInterface 使用 int8 还是 int16
        .def_readwrite("bitwidth_config_", &GRUQuantitativeParametersPy::bitwidth_config_);

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
    // forward: 正常前向传播（推理/训练）
    // =====================================================================
    m.def("forward", &forward_wrapper,
          "GRU forward pass for inference/training.\n"
          "\n"
          "Args:\n"
          "  is_training: Enable training mode (saves intermediate values)\n"
          "  is_quant: Use quantized inference (requires quant_params)\n"
          "  time_steps, batch_size, input_size, hidden_size: Dimension parameters\n"
          "  W, R, bx, br: Weight matrices and biases\n"
          "  x: Input tensor [T, B, I]\n"
          "  h0: Initial hidden state [B, H], optional\n"
          "  quant_params: Quantization parameters (used when is_quant=True)\n"
          "\n"
          "Returns:\n"
          "  tuple(h, v)\n"
          "  - h: Hidden states [T+1, B, H]\n"
          "  - v: Intermediate values [T, B, H*4]",
          py::arg("is_training"), py::arg("is_quant"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"), py::arg("x"),
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
          "  W, R, bx, br: Weight matrices and biases (float)\n"
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
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor(),
          py::arg("calib_method"),
          py::arg("quant_ranges") = nullptr,
          py::arg("hist_collectors") = nullptr);

    // GRU 反向传播
    m.def("haste_gru_backward", &haste_gru_backward_wrapper, "Non-quantized GRU backward pass",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("x"), py::arg("dh_new"), py::arg("h"),
          py::arg("v"));  // 中间值v，必需；返回 (dx, dW, dR, dbx, dbr, dh) 元组
}
