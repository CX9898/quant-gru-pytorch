#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "gru_interface.hpp"
#include "gru_quantization_ranges.hpp"
#include "quantize_ops_helper.hpp"

// 全局 cublas handle
static cublasHandle_t g_blas_handle = nullptr;

// 初始化 cublas handle 的包装函数
void init_gru_cublas_wrapper() { init_gru_cublas(g_blas_handle); }

// GRUQuantizationRanges 的 Python 绑定
struct GRUQuantizationRangesPy {
    int hidden_ = 0;

    // 输入和隐藏状态
    float min_x_ = std::numeric_limits<float>::max();
    float max_x_ = std::numeric_limits<float>::lowest();
    float min_h_ = std::numeric_limits<float>::max();
    float max_h_ = std::numeric_limits<float>::lowest();

    // 权重矩阵（per-channel）
    std::vector<float> min_W_, max_W_;
    std::vector<float> min_R_, max_R_;

    // 矩阵乘法结果
    float min_Wx_ = std::numeric_limits<float>::max();
    float max_Wx_ = std::numeric_limits<float>::lowest();
    float min_Rh_ = std::numeric_limits<float>::max();
    float max_Rh_ = std::numeric_limits<float>::lowest();

    // 偏置（per-channel）
    std::vector<float> min_bx_, max_bx_;
    std::vector<float> min_br_, max_br_;

    // 门的预激活值
    float min_z_pre_ = std::numeric_limits<float>::max();
    float max_z_pre_ = std::numeric_limits<float>::lowest();
    float min_r_pre_ = std::numeric_limits<float>::max();
    float max_r_pre_ = std::numeric_limits<float>::lowest();
    float min_g_pre_ = std::numeric_limits<float>::max();
    float max_g_pre_ = std::numeric_limits<float>::lowest();

    // 门的输出值
    float min_z_out_ = std::numeric_limits<float>::max();
    float max_z_out_ = std::numeric_limits<float>::lowest();
    float min_r_out_ = std::numeric_limits<float>::max();
    float max_r_out_ = std::numeric_limits<float>::lowest();
    float min_g_out_ = std::numeric_limits<float>::max();
    float max_g_out_ = std::numeric_limits<float>::lowest();

    // 中间计算结果
    float min_Rh_add_br_ = std::numeric_limits<float>::max();
    float max_Rh_add_br_ = std::numeric_limits<float>::lowest();
    float min_rRh_ = std::numeric_limits<float>::max();
    float max_rRh_ = std::numeric_limits<float>::lowest();

    // 最终输出计算
    float min_one_minus_update_ = std::numeric_limits<float>::max();
    float max_one_minus_update_ = std::numeric_limits<float>::lowest();
    float min_new_contrib_ = std::numeric_limits<float>::max();
    float max_new_contrib_ = std::numeric_limits<float>::lowest();
    float min_old_contrib_ = std::numeric_limits<float>::max();
    float max_old_contrib_ = std::numeric_limits<float>::lowest();

    // 从 C++ 结构体转换
    void from_cpp(const GRUQuantizationRanges &cpp_ranges) {
        hidden_ = cpp_ranges.hidden_;
        min_x_ = cpp_ranges.min_x_;
        max_x_ = cpp_ranges.max_x_;
        min_h_ = cpp_ranges.min_h_;
        max_h_ = cpp_ranges.max_h_;
        min_W_ = cpp_ranges.min_W_;
        max_W_ = cpp_ranges.max_W_;
        min_R_ = cpp_ranges.min_R_;
        max_R_ = cpp_ranges.max_R_;
        min_Wx_ = cpp_ranges.min_Wx_;
        max_Wx_ = cpp_ranges.max_Wx_;
        min_Rh_ = cpp_ranges.min_Rh_;
        max_Rh_ = cpp_ranges.max_Rh_;
        min_bx_ = cpp_ranges.min_bx_;
        max_bx_ = cpp_ranges.max_bx_;
        min_br_ = cpp_ranges.min_br_;
        max_br_ = cpp_ranges.max_br_;
        min_z_pre_ = cpp_ranges.min_z_pre_;
        max_z_pre_ = cpp_ranges.max_z_pre_;
        min_r_pre_ = cpp_ranges.min_r_pre_;
        max_r_pre_ = cpp_ranges.max_r_pre_;
        min_g_pre_ = cpp_ranges.min_g_pre_;
        max_g_pre_ = cpp_ranges.max_g_pre_;
        min_z_out_ = cpp_ranges.min_z_out_;
        max_z_out_ = cpp_ranges.max_z_out_;
        min_r_out_ = cpp_ranges.min_r_out_;
        max_r_out_ = cpp_ranges.max_r_out_;
        min_g_out_ = cpp_ranges.min_g_out_;
        max_g_out_ = cpp_ranges.max_g_out_;
        min_Rh_add_br_ = cpp_ranges.min_Rh_add_br_;
        max_Rh_add_br_ = cpp_ranges.max_Rh_add_br_;
        min_rRh_ = cpp_ranges.min_rRh_;
        max_rRh_ = cpp_ranges.max_rRh_;
        min_one_minus_update_ = cpp_ranges.min_one_minus_update_;
        max_one_minus_update_ = cpp_ranges.max_one_minus_update_;
        min_new_contrib_ = cpp_ranges.min_new_contrib_;
        max_new_contrib_ = cpp_ranges.max_new_contrib_;
        min_old_contrib_ = cpp_ranges.min_old_contrib_;
        max_old_contrib_ = cpp_ranges.max_old_contrib_;
    }

    // 转换为 C++ 结构体
    GRUQuantizationRanges to_cpp() const {
        GRUQuantizationRanges cpp_ranges;
        cpp_ranges.hidden_ = hidden_;
        cpp_ranges.min_x_ = min_x_;
        cpp_ranges.max_x_ = max_x_;
        cpp_ranges.min_h_ = min_h_;
        cpp_ranges.max_h_ = max_h_;
        cpp_ranges.min_W_ = min_W_;
        cpp_ranges.max_W_ = max_W_;
        cpp_ranges.min_R_ = min_R_;
        cpp_ranges.max_R_ = max_R_;
        cpp_ranges.min_Wx_ = min_Wx_;
        cpp_ranges.max_Wx_ = max_Wx_;
        cpp_ranges.min_Rh_ = min_Rh_;
        cpp_ranges.max_Rh_ = max_Rh_;
        cpp_ranges.min_bx_ = min_bx_;
        cpp_ranges.max_bx_ = max_bx_;
        cpp_ranges.min_br_ = min_br_;
        cpp_ranges.max_br_ = max_br_;
        cpp_ranges.min_z_pre_ = min_z_pre_;
        cpp_ranges.max_z_pre_ = max_z_pre_;
        cpp_ranges.min_r_pre_ = min_r_pre_;
        cpp_ranges.max_r_pre_ = max_r_pre_;
        cpp_ranges.min_g_pre_ = min_g_pre_;
        cpp_ranges.max_g_pre_ = max_g_pre_;
        cpp_ranges.min_z_out_ = min_z_out_;
        cpp_ranges.max_z_out_ = max_z_out_;
        cpp_ranges.min_r_out_ = min_r_out_;
        cpp_ranges.max_r_out_ = max_r_out_;
        cpp_ranges.min_g_out_ = min_g_out_;
        cpp_ranges.max_g_out_ = max_g_out_;
        cpp_ranges.min_Rh_add_br_ = min_Rh_add_br_;
        cpp_ranges.max_Rh_add_br_ = max_Rh_add_br_;
        cpp_ranges.min_rRh_ = min_rRh_;
        cpp_ranges.max_rRh_ = max_rRh_;
        cpp_ranges.min_one_minus_update_ = min_one_minus_update_;
        cpp_ranges.max_one_minus_update_ = max_one_minus_update_;
        cpp_ranges.min_new_contrib_ = min_new_contrib_;
        cpp_ranges.max_new_contrib_ = max_new_contrib_;
        cpp_ranges.min_old_contrib_ = min_old_contrib_;
        cpp_ranges.max_old_contrib_ = max_old_contrib_;
        return cpp_ranges;
    }

    // 重置为无效值
    void reset() {
        min_x_ = std::numeric_limits<float>::max();
        max_x_ = std::numeric_limits<float>::lowest();
        min_h_ = std::numeric_limits<float>::max();
        max_h_ = std::numeric_limits<float>::lowest();
        min_Wx_ = std::numeric_limits<float>::max();
        max_Wx_ = std::numeric_limits<float>::lowest();
        min_Rh_ = std::numeric_limits<float>::max();
        max_Rh_ = std::numeric_limits<float>::lowest();
        min_z_pre_ = std::numeric_limits<float>::max();
        max_z_pre_ = std::numeric_limits<float>::lowest();
        min_r_pre_ = std::numeric_limits<float>::max();
        max_r_pre_ = std::numeric_limits<float>::lowest();
        min_g_pre_ = std::numeric_limits<float>::max();
        max_g_pre_ = std::numeric_limits<float>::lowest();
        min_z_out_ = std::numeric_limits<float>::max();
        max_z_out_ = std::numeric_limits<float>::lowest();
        min_r_out_ = std::numeric_limits<float>::max();
        max_r_out_ = std::numeric_limits<float>::lowest();
        min_g_out_ = std::numeric_limits<float>::max();
        max_g_out_ = std::numeric_limits<float>::lowest();
        min_Rh_add_br_ = std::numeric_limits<float>::max();
        max_Rh_add_br_ = std::numeric_limits<float>::lowest();
        min_rRh_ = std::numeric_limits<float>::max();
        max_rRh_ = std::numeric_limits<float>::lowest();
        min_one_minus_update_ = std::numeric_limits<float>::max();
        max_one_minus_update_ = std::numeric_limits<float>::lowest();
        min_new_contrib_ = std::numeric_limits<float>::max();
        max_new_contrib_ = std::numeric_limits<float>::lowest();
        min_old_contrib_ = std::numeric_limits<float>::max();
        max_old_contrib_ = std::numeric_limits<float>::lowest();

        if (hidden_ > 0) {
            resize_per_channel_vectors(hidden_);
        }
    }

    // 调整 per-channel 向量大小
    void resize_per_channel_vectors(int hidden) {
        hidden_ = hidden;
        const int channel_size = hidden * 3;
        min_W_.assign(channel_size, std::numeric_limits<float>::max());
        max_W_.assign(channel_size, std::numeric_limits<float>::lowest());
        min_R_.assign(channel_size, std::numeric_limits<float>::max());
        max_R_.assign(channel_size, std::numeric_limits<float>::lowest());
        min_bx_.assign(channel_size, std::numeric_limits<float>::max());
        max_bx_.assign(channel_size, std::numeric_limits<float>::lowest());
        min_br_.assign(channel_size, std::numeric_limits<float>::max());
        max_br_.assign(channel_size, std::numeric_limits<float>::lowest());
    }
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
    int8_t exp2_inv_one_minus_update_;
    int32_t zp_one_minus_update_;
    int8_t exp2_inv_new_contrib_;
    int32_t zp_new_contrib_;
    int8_t exp2_inv_old_contrib_;
    int32_t zp_old_contrib_;

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
        exp2_inv_one_minus_update_ = cpp_params.exp2_inv_one_minus_update_;
        zp_one_minus_update_ = cpp_params.zp_one_minus_update_;
        exp2_inv_new_contrib_ = cpp_params.exp2_inv_new_contrib_;
        zp_new_contrib_ = cpp_params.zp_new_contrib_;
        exp2_inv_old_contrib_ = cpp_params.exp2_inv_old_contrib_;
        zp_old_contrib_ = cpp_params.zp_old_contrib_;
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
        cpp_params.exp2_inv_one_minus_update_ = exp2_inv_one_minus_update_;
        cpp_params.zp_one_minus_update_ = zp_one_minus_update_;
        cpp_params.exp2_inv_new_contrib_ = exp2_inv_new_contrib_;
        cpp_params.zp_new_contrib_ = zp_new_contrib_;
        cpp_params.exp2_inv_old_contrib_ = exp2_inv_old_contrib_;
        cpp_params.zp_old_contrib_ = zp_old_contrib_;
        return cpp_params;
    }
};

// 校准 GRU 量化范围的包装函数（支持累积更新）
void calibrate_gru_ranges_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bx,
    const torch::Tensor &br, const torch::Tensor &x,
    GRUQuantizationRangesPy &quant_ranges) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // 检查 x 的形状，期望 [time_steps, batch_size, input_size]
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({time_steps, batch_size, input_size}),
                "x must have shape [time_steps, batch_size, input_size]");

    // 确保 x 是连续的
    torch::Tensor x_contiguous = x.contiguous();

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 转换为 C++ 结构体
    GRUQuantizationRanges cpp_ranges = quant_ranges.to_cpp();

    // 调用 C++ 函数（累积更新范围）
    calibrateGruRanges(time_steps, batch_size, input_size, hidden_size,
                       W.data_ptr<float>(), R.data_ptr<float>(), bx.data_ptr<float>(),
                       br.data_ptr<float>(), x_contiguous.data_ptr<float>(), g_blas_handle,
                       cpp_ranges);

    // 更新 Python 对象
    quant_ranges.from_cpp(cpp_ranges);
}

// 根据量化范围计算量化参数的包装函数
GRUQuantitativeParametersPy calculate_gru_quantitative_parameters_wrapper(
    const GRUQuantizationRangesPy &quant_ranges) {
    // 转换为 C++ 结构体
    GRUQuantizationRanges cpp_ranges = quant_ranges.to_cpp();

    // 调用 C++ 函数（使用默认的位宽配置）
    GRUQuantitativeParameters quant_params = calculateGRUQuantitativeParameters(cpp_ranges);

    GRUQuantitativeParametersPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// 校准 GRU 量化参数的包装函数
GRUQuantitativeParametersPy calibrate_gru_scales_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bx,
    const torch::Tensor &br, const torch::Tensor &x) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // 检查 x 的形状，期望 [time_steps, batch_size, input_size]
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({time_steps, batch_size, input_size}),
                "x must have shape [time_steps, batch_size, input_size]");

    // 确保 x 是连续的（Haste GRU 期望 [T, N, C] 格式，但需要连续内存布局）
    torch::Tensor x_contiguous = x.contiguous();

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 调用 C++ 函数（使用默认的 INT8 位宽配置）
    GRUQuantitativeParameters quant_params =
        calibrateGruScales(time_steps, batch_size, input_size, hidden_size,
                           W.data_ptr<float>(), R.data_ptr<float>(), bx.data_ptr<float>(),
                           br.data_ptr<float>(), x_contiguous.data_ptr<float>(), g_blas_handle);

    GRUQuantitativeParametersPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// 校准 GRU 量化参数并初始化 LUT 表的包装函数（组合函数）
GRUQuantitativeParametersPy calibrate_gru_scales_and_init_lut_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W, const torch::Tensor &R, const torch::Tensor &bx,
    const torch::Tensor &br, const torch::Tensor &x) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 调用组合函数（内部会处理 LUT 初始化，使用默认的 INT8 位宽配置）
    GRUQuantitativeParameters quant_params =
        calibrateGruScalesAndInitLut(time_steps, batch_size, input_size, hidden_size,
                                     W.data_ptr<float>(), R.data_ptr<float>(), bx.data_ptr<float>(),
                                     br.data_ptr<float>(), x.data_ptr<float>(), g_blas_handle);

    GRUQuantitativeParametersPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// 量化权重的包装函数（int8）
void quantitative_weight_int8_wrapper(int input_size, int hidden_size, const torch::Tensor &W,
                                      const torch::Tensor &R, const torch::Tensor &bx,
                                      const torch::Tensor &br,
                                      const GRUQuantitativeParametersPy &quant_params,
                                      torch::Tensor &W_quant, torch::Tensor &R_quant,
                                      torch::Tensor &bx_quant, torch::Tensor &br_quant) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(W_quant.is_cuda() && W_quant.dtype() == torch::kInt8,
                "W_quant must be CUDA int8 tensor");
    TORCH_CHECK(R_quant.is_cuda() && R_quant.dtype() == torch::kInt8,
                "R_quant must be CUDA int8 tensor");
    TORCH_CHECK(bx_quant.is_cuda() && bx_quant.dtype() == torch::kInt32,
                "bx_quant must be CUDA int32 tensor");
    TORCH_CHECK(br_quant.is_cuda() && br_quant.dtype() == torch::kInt32,
                "br_quant must be CUDA int32 tensor");

    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();

    quantitativeWeight<int8_t>(input_size, hidden_size, W.data_ptr<float>(), R.data_ptr<float>(),
                               bx.data_ptr<float>(), br.data_ptr<float>(), cpp_params,
                               W_quant.data_ptr<int8_t>(), R_quant.data_ptr<int8_t>(),
                               bx_quant.data_ptr<int32_t>(), br_quant.data_ptr<int32_t>());
}

// 量化权重的包装函数（int16）
void quantitative_weight_int16_wrapper(int input_size, int hidden_size, const torch::Tensor &W,
                                       const torch::Tensor &R, const torch::Tensor &bx,
                                       const torch::Tensor &br,
                                       const GRUQuantitativeParametersPy &quant_params,
                                       torch::Tensor &W_quant, torch::Tensor &R_quant,
                                       torch::Tensor &bx_quant, torch::Tensor &br_quant) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(W_quant.is_cuda() && W_quant.dtype() == torch::kInt16,
                "W_quant must be CUDA int16 tensor");
    TORCH_CHECK(R_quant.is_cuda() && R_quant.dtype() == torch::kInt16,
                "R_quant must be CUDA int16 tensor");
    TORCH_CHECK(bx_quant.is_cuda() && bx_quant.dtype() == torch::kInt32,
                "bx_quant must be CUDA int32 tensor");
    TORCH_CHECK(br_quant.is_cuda() && br_quant.dtype() == torch::kInt32,
                "br_quant must be CUDA int32 tensor");

    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();

    quantitativeWeight<int16_t>(input_size, hidden_size, W.data_ptr<float>(), R.data_ptr<float>(),
                                bx.data_ptr<float>(), br.data_ptr<float>(), cpp_params,
                                W_quant.data_ptr<int16_t>(), R_quant.data_ptr<int16_t>(),
                                bx_quant.data_ptr<int32_t>(), br_quant.data_ptr<int32_t>());
}

// 量化 GRU 前向传播（int8）
std::tuple<torch::Tensor, torch::Tensor> quant_gru_forward_int8_wrapper(
    bool is_training,  // 是否开启训练模式，true为训练，false为推理
    int time_steps, int batch_size, int input_size, int hidden_size, const torch::Tensor &W_quant,
    const torch::Tensor &R_quant, const torch::Tensor &bx_quant, const torch::Tensor &br_quant,
    const torch::Tensor &x,
    const torch::Tensor &h0,  // 初始隐藏状态，可以为空张量
    const GRUQuantitativeParametersPy &quant_params) {
    TORCH_CHECK(W_quant.is_cuda() && W_quant.dtype() == torch::kInt8,
                "W_quant must be CUDA int8 tensor");
    TORCH_CHECK(R_quant.is_cuda() && R_quant.dtype() == torch::kInt8,
                "R_quant must be CUDA int8 tensor");
    TORCH_CHECK(bx_quant.is_cuda() && bx_quant.dtype() == torch::kInt32,
                "bx_quant must be CUDA int32 tensor");
    TORCH_CHECK(br_quant.is_cuda() && br_quant.dtype() == torch::kInt32,
                "br_quant must be CUDA int32 tensor");
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

    // 创建输出张量，包含初始状态，大小为 (time_steps + 1) * batch_size * hidden_size
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 创建v输出张量，大小为 time_steps * batch_size * hidden_size * 4
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();

    quantGRUForward<int8_t>(is_training, time_steps, batch_size, input_size, hidden_size,
                            W_quant.data_ptr<int8_t>(), R_quant.data_ptr<int8_t>(),
                            bx_quant.data_ptr<int32_t>(), br_quant.data_ptr<int32_t>(),
                            x.data_ptr<float>(),
                            h0_ptr,  // 初始隐藏状态，可以为 nullptr
                            cpp_params, g_blas_handle, h.data_ptr<float>(),
                            v.data_ptr<float>()  // 反量化后的v输出
    );

    return std::make_tuple(h, v);
}

// 量化 GRU 前向传播（int16）
std::tuple<torch::Tensor, torch::Tensor> quant_gru_forward_int16_wrapper(
    bool is_training,  // 是否开启训练模式，true为训练，false为推理
    int time_steps, int batch_size, int input_size, int hidden_size, const torch::Tensor &W_quant,
    const torch::Tensor &R_quant, const torch::Tensor &bx_quant, const torch::Tensor &br_quant,
    const torch::Tensor &x,
    const torch::Tensor &h0,  // 初始隐藏状态，可以为空张量
    const GRUQuantitativeParametersPy &quant_params) {
    TORCH_CHECK(W_quant.is_cuda() && W_quant.dtype() == torch::kInt16,
                "W_quant must be CUDA int16 tensor");
    TORCH_CHECK(R_quant.is_cuda() && R_quant.dtype() == torch::kInt16,
                "R_quant must be CUDA int16 tensor");
    TORCH_CHECK(bx_quant.is_cuda() && bx_quant.dtype() == torch::kInt32,
                "bx_quant must be CUDA int32 tensor");
    TORCH_CHECK(br_quant.is_cuda() && br_quant.dtype() == torch::kInt32,
                "br_quant must be CUDA int32 tensor");
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

    // 创建输出张量，包含初始状态，大小为 (time_steps + 1) * batch_size * hidden_size
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 创建v输出张量，大小为 time_steps * batch_size * hidden_size * 4
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();

    quantGRUForward<int16_t>(is_training, time_steps, batch_size, input_size, hidden_size,
                             W_quant.data_ptr<int16_t>(), R_quant.data_ptr<int16_t>(),
                             bx_quant.data_ptr<int32_t>(), br_quant.data_ptr<int32_t>(),
                             x.data_ptr<float>(),
                             h0_ptr,  // 初始隐藏状态，可以为 nullptr
                             cpp_params, g_blas_handle, h.data_ptr<float>(),
                             v.data_ptr<float>()  // 反量化后的v输出
    );

    return std::make_tuple(h, v);
}

// 非量化 GRU 前向传播
std::tuple<torch::Tensor, torch::Tensor> haste_gru_forward_wrapper(
    bool is_training,  // 是否开启训练模式，true为训练，false为推理
    int time_steps, int batch_size, int input_size, int hidden_size, const torch::Tensor &W,
    const torch::Tensor &R, const torch::Tensor &bx, const torch::Tensor &br,
    const torch::Tensor &x,
    const torch::Tensor &h0) {  // 初始隐藏状态，可以为空张量

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

    // 创建输出张量，包含初始状态，大小为 (time_steps + 1) * batch_size * hidden_size
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 创建v输出张量，大小为 time_steps * batch_size * hidden_size * 4
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    hasteGRUForward(is_training, time_steps, batch_size, input_size, hidden_size,
                    W.data_ptr<float>(), R.data_ptr<float>(), bx.data_ptr<float>(),
                    br.data_ptr<float>(), x.data_ptr<float>(),
                    h0_ptr,  // 初始隐藏状态，可以为 nullptr
                    g_blas_handle, h.data_ptr<float>(),
                    v.data_ptr<float>()  // 中间值v输出
    );

    return std::make_tuple(h, v);
}

// forwardInterface 的包装函数
std::tuple<torch::Tensor, torch::Tensor> forward_interface_wrapper(
    bool is_training,  // 是否开启训练模式，true为训练，false为推理
    bool is_quant, int time_steps, int batch_size, int input_size, int hidden_size,
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

    // 创建输出张量，包含初始状态，大小为 (time_steps + 1) * batch_size * hidden_size
    auto h = torch::empty({time_steps + 1, batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 创建v输出张量，大小为 time_steps * batch_size * hidden_size * 4
    auto v = torch::empty({time_steps, batch_size, hidden_size * 4},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();

    forwardInterface(is_training, is_quant, time_steps, batch_size, input_size,
                     hidden_size, W.data_ptr<float>(), R.data_ptr<float>(), bx.data_ptr<float>(),
                     br.data_ptr<float>(), x.data_ptr<float>(),
                     h0_ptr,  // 初始隐藏状态，可以为 nullptr
                     cpp_params, g_blas_handle, h.data_ptr<float>(),
                     v.data_ptr<float>()  // 传递v参数
    );

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

// 初始化量化 LUT 表的包装函数
// 将 Python 绑定层的参数转换为 C++ 接口层的参数
void initialize_quantization_lut_wrapper(const GRUQuantitativeParametersPy &quant_params) {
    // 转换为 C++ 结构体并调用 gru_interface 中的函数
    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();
    initialize_quantization_lut(cpp_params);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GRU Interface Python Bindings";

    // 初始化 cublas handle
    m.def("init_gru_cublas", &init_gru_cublas_wrapper, "Initialize cuBLAS handle for GRU");

    // GRUQuantizationRanges 绑定
    py::class_<GRUQuantizationRangesPy>(m, "GRUQuantizationRanges")
        .def(py::init<>())
        .def_readwrite("hidden_", &GRUQuantizationRangesPy::hidden_)
        .def_readwrite("min_x_", &GRUQuantizationRangesPy::min_x_)
        .def_readwrite("max_x_", &GRUQuantizationRangesPy::max_x_)
        .def_readwrite("min_h_", &GRUQuantizationRangesPy::min_h_)
        .def_readwrite("max_h_", &GRUQuantizationRangesPy::max_h_)
        .def_readwrite("min_W_", &GRUQuantizationRangesPy::min_W_)
        .def_readwrite("max_W_", &GRUQuantizationRangesPy::max_W_)
        .def_readwrite("min_R_", &GRUQuantizationRangesPy::min_R_)
        .def_readwrite("max_R_", &GRUQuantizationRangesPy::max_R_)
        .def_readwrite("min_Wx_", &GRUQuantizationRangesPy::min_Wx_)
        .def_readwrite("max_Wx_", &GRUQuantizationRangesPy::max_Wx_)
        .def_readwrite("min_Rh_", &GRUQuantizationRangesPy::min_Rh_)
        .def_readwrite("max_Rh_", &GRUQuantizationRangesPy::max_Rh_)
        .def_readwrite("min_bx_", &GRUQuantizationRangesPy::min_bx_)
        .def_readwrite("max_bx_", &GRUQuantizationRangesPy::max_bx_)
        .def_readwrite("min_br_", &GRUQuantizationRangesPy::min_br_)
        .def_readwrite("max_br_", &GRUQuantizationRangesPy::max_br_)
        .def_readwrite("min_z_pre_", &GRUQuantizationRangesPy::min_z_pre_)
        .def_readwrite("max_z_pre_", &GRUQuantizationRangesPy::max_z_pre_)
        .def_readwrite("min_r_pre_", &GRUQuantizationRangesPy::min_r_pre_)
        .def_readwrite("max_r_pre_", &GRUQuantizationRangesPy::max_r_pre_)
        .def_readwrite("min_g_pre_", &GRUQuantizationRangesPy::min_g_pre_)
        .def_readwrite("max_g_pre_", &GRUQuantizationRangesPy::max_g_pre_)
        .def_readwrite("min_z_out_", &GRUQuantizationRangesPy::min_z_out_)
        .def_readwrite("max_z_out_", &GRUQuantizationRangesPy::max_z_out_)
        .def_readwrite("min_r_out_", &GRUQuantizationRangesPy::min_r_out_)
        .def_readwrite("max_r_out_", &GRUQuantizationRangesPy::max_r_out_)
        .def_readwrite("min_g_out_", &GRUQuantizationRangesPy::min_g_out_)
        .def_readwrite("max_g_out_", &GRUQuantizationRangesPy::max_g_out_)
        .def_readwrite("min_Rh_add_br_", &GRUQuantizationRangesPy::min_Rh_add_br_)
        .def_readwrite("max_Rh_add_br_", &GRUQuantizationRangesPy::max_Rh_add_br_)
        .def_readwrite("min_rRh_", &GRUQuantizationRangesPy::min_rRh_)
        .def_readwrite("max_rRh_", &GRUQuantizationRangesPy::max_rRh_)
        .def_readwrite("min_one_minus_update_", &GRUQuantizationRangesPy::min_one_minus_update_)
        .def_readwrite("max_one_minus_update_", &GRUQuantizationRangesPy::max_one_minus_update_)
        .def_readwrite("min_new_contrib_", &GRUQuantizationRangesPy::min_new_contrib_)
        .def_readwrite("max_new_contrib_", &GRUQuantizationRangesPy::max_new_contrib_)
        .def_readwrite("min_old_contrib_", &GRUQuantizationRangesPy::min_old_contrib_)
        .def_readwrite("max_old_contrib_", &GRUQuantizationRangesPy::max_old_contrib_)
        .def("reset", &GRUQuantizationRangesPy::reset, "Reset all ranges to invalid values")
        .def("resize_per_channel_vectors", &GRUQuantizationRangesPy::resize_per_channel_vectors,
             "Resize per-channel vectors", py::arg("hidden"));

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
        .def_readwrite("exp2_inv_one_minus_update_",
                       &GRUQuantitativeParametersPy::exp2_inv_one_minus_update_)
        .def_readwrite("zp_one_minus_update_", &GRUQuantitativeParametersPy::zp_one_minus_update_)
        .def_readwrite("exp2_inv_new_contrib_", &GRUQuantitativeParametersPy::exp2_inv_new_contrib_)
        .def_readwrite("zp_new_contrib_", &GRUQuantitativeParametersPy::zp_new_contrib_)
        .def_readwrite("exp2_inv_old_contrib_", &GRUQuantitativeParametersPy::exp2_inv_old_contrib_)
        .def_readwrite("zp_old_contrib_", &GRUQuantitativeParametersPy::zp_old_contrib_);

    // 校准量化范围（支持多次调用累积更新）
    m.def("calibrate_gru_ranges", &calibrate_gru_ranges_wrapper,
          "Calibrate GRU quantization ranges (supports accumulative updates)",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("x"), py::arg("quant_ranges"));

    // 根据量化范围计算量化参数
    m.def("calculate_gru_quantitative_parameters", &calculate_gru_quantitative_parameters_wrapper,
          "Calculate GRU quantitative parameters from quantization ranges",
          py::arg("quant_ranges"));

    // 校准量化参数（一次性完成，向后兼容）
    m.def("calibrate_gru_scales", &calibrate_gru_scales_wrapper,
          "Calibrate GRU quantization scales (one-shot, for backward compatibility)",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("x"));

    // 校准量化参数并初始化 LUT 表（组合函数，方便使用）
    m.def("calibrate_gru_scales_and_init_lut", &calibrate_gru_scales_and_init_lut_wrapper,
          "Calibrate GRU quantization scales and initialize LUT tables (convenience function)",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("x"));

    // 量化权重（int8）
    m.def("quantitative_weight_int8", &quantitative_weight_int8_wrapper,
          "Quantize GRU weights to int8", py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"), py::arg("quant_params"),
          py::arg("W_quant"), py::arg("R_quant"), py::arg("bx_quant"), py::arg("br_quant"));

    // 量化权重（int16）
    m.def("quantitative_weight_int16", &quantitative_weight_int16_wrapper,
          "Quantize GRU weights to int16", py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"), py::arg("quant_params"),
          py::arg("W_quant"), py::arg("R_quant"), py::arg("bx_quant"), py::arg("br_quant"));

    // 量化 GRU 前向传播（int8）
    m.def("quant_gru_forward_int8", &quant_gru_forward_int8_wrapper,
          "Quantized GRU forward pass (int8)",
          py::arg("is_training"),  // 是否开启训练模式，true为训练，false为推理
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W_quant"), py::arg("R_quant"), py::arg("bx_quant"),
          py::arg("br_quant"), py::arg("x"), py::arg("h0") = torch::Tensor(),  // 初始隐藏状态，可选
          py::arg("quant_params"));  // 返回 (h, v) 元组，h包含初始状态，v为反量化后的中间值

    // 量化 GRU 前向传播（int16）
    m.def("quant_gru_forward_int16", &quant_gru_forward_int16_wrapper,
          "Quantized GRU forward pass (int16)",
          py::arg("is_training"),  // 是否开启训练模式，true为训练，false为推理
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W_quant"), py::arg("R_quant"), py::arg("bx_quant"),
          py::arg("br_quant"), py::arg("x"), py::arg("h0") = torch::Tensor(),  // 初始隐藏状态，可选
          py::arg("quant_params"));  // 返回 (h, v) 元组，h包含初始状态，v为反量化后的中间值

    // 非量化 GRU 前向传播
    m.def("haste_gru_forward", &haste_gru_forward_wrapper, "Non-quantized GRU forward pass",
          py::arg("is_training"),  // 是否开启训练模式，true为训练，false为推理
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("x"),
          py::arg("h0") =
              torch::Tensor());  // 初始隐藏状态，可选；返回 (h, v) 元组，h包含初始状态，v为中间值

    // forwardInterface 统一接口
    m.def("forward_interface", &forward_interface_wrapper,
          "Unified GRU forward interface supporting both quantized and non-quantized modes",
          py::arg("is_training"),  // 是否开启训练模式，true为训练，false为推理
          py::arg("is_quant"), py::arg("time_steps"), py::arg("batch_size"),
          py::arg("input_size"), py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bx"),
          py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor(),  // 初始隐藏状态，可选
          py::arg("quant_params"));         // 返回 (h, v) 元组，h包含初始状态，v为中间值

    // GRU 反向传播
    m.def("haste_gru_backward", &haste_gru_backward_wrapper, "Non-quantized GRU backward pass",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("x"), py::arg("dh_new"), py::arg("h"),
          py::arg("v"));  // 中间值v，必需；返回 (dx, dW, dR, dbx, dbr, dh) 元组

    // 初始化量化 LUT 表（仅在初始化时调用一次）
    // 接收量化参数对象，内部根据 bitwidth_config_ 自动选择相应的 LUT 初始化方法
    m.def("initialize_quantization_lut", &initialize_quantization_lut_wrapper,
          "Initialize quantization LUT tables from quantization parameters (should be called only "
          "once during initialization)",
          py::arg("quant_params"));
}
