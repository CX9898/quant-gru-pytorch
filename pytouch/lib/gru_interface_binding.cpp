#include "gru_interface.hpp"
#include "quantize_ops_helper.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// 全局 cublas handle
static cublasHandle_t g_blas_handle = nullptr;

// 初始化 cublas handle 的包装函数
void init_gru_cublas_wrapper() {
    init_gru_cublas(g_blas_handle);
}

// GRUQuantitativeParameters 的 Python 绑定
struct GRUQuantitativeParametersPy {
    int hidden_;
    int32_t exp2_inv_x_;
    int32_t zp_x_;
    int32_t exp2_inv_h_;
    int32_t zp_h_;
    std::vector<int32_t> exp2_inv_W_;
    std::vector<int32_t> exp2_inv_R_;
    int32_t exp2_inv_Wx_;
    int32_t zp_Wx_;
    int32_t exp2_inv_Rh_;
    int32_t zp_Rh_;
    std::vector<int32_t> exp2_inv_bx_;
    std::vector<int32_t> exp2_inv_br_;
    int32_t exp2_inv_z_pre_;
    int32_t zp_z_pre_;
    int32_t exp2_inv_r_pre_;
    int32_t zp_r_pre_;
    int32_t exp2_inv_g_pre_;
    int32_t zp_g_pre_;
    int32_t exp2_inv_z_out_;
    int32_t zp_z_out_;
    int32_t exp2_inv_r_out_;
    int32_t zp_r_out_;
    int32_t exp2_inv_g_out_;
    int32_t zp_g_out_;
    int32_t exp2_inv_Rh_add_br_;
    int32_t zp_Rh_add_br_;
    int32_t exp2_inv_rRh_;
    int32_t zp_rRh_;
    int32_t exp2_inv_one_minus_update_;
    int32_t zp_one_minus_update_;
    int32_t exp2_inv_new_contrib_;
    int32_t zp_new_contrib_;
    int32_t exp2_inv_old_contrib_;
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

// 校准 GRU 量化参数的包装函数
GRUQuantitativeParametersPy calibrate_gru_scales_wrapper(
    bool use_int16,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W,
    const torch::Tensor &R,
    const torch::Tensor &bx,
    const torch::Tensor &br,
    const torch::Tensor &x) {

    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // 确保 cublas handle 已初始化
    if (g_blas_handle == nullptr) {
        init_gru_cublas(g_blas_handle);
    }

    // 调用 C++ 函数
    GRUQuantitativeParameters quant_params = calibrateGruScales(
        use_int16,
        time_steps, batch_size, input_size, hidden_size,
        W.data_ptr<float>(),
        R.data_ptr<float>(),
        bx.data_ptr<float>(),
        br.data_ptr<float>(),
        x.data_ptr<float>(),
        g_blas_handle);

    GRUQuantitativeParametersPy py_params;
    py_params.from_cpp(quant_params);
    return py_params;
}

// 量化权重的包装函数（int8）
void quantitative_weight_int8_wrapper(
    int input_size, int hidden_size,
    const torch::Tensor &W,
    const torch::Tensor &R,
    const torch::Tensor &bx,
    const torch::Tensor &br,
    const GRUQuantitativeParametersPy &quant_params,
    torch::Tensor &W_quant,
    torch::Tensor &R_quant,
    torch::Tensor &bx_quant,
    torch::Tensor &br_quant) {

    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(W_quant.is_cuda() && W_quant.dtype() == torch::kInt8, "W_quant must be CUDA int8 tensor");
    TORCH_CHECK(R_quant.is_cuda() && R_quant.dtype() == torch::kInt8, "R_quant must be CUDA int8 tensor");
    TORCH_CHECK(bx_quant.is_cuda() && bx_quant.dtype() == torch::kInt32, "bx_quant must be CUDA int32 tensor");
    TORCH_CHECK(br_quant.is_cuda() && br_quant.dtype() == torch::kInt32, "br_quant must be CUDA int32 tensor");

    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();

    quantitativeWeight<int8_t>(
        input_size, hidden_size,
        W.data_ptr<float>(),
        R.data_ptr<float>(),
        bx.data_ptr<float>(),
        br.data_ptr<float>(),
        cpp_params,
        W_quant.data_ptr<int8_t>(),
        R_quant.data_ptr<int8_t>(),
        bx_quant.data_ptr<int32_t>(),
        br_quant.data_ptr<int32_t>());
}

// 量化权重的包装函数（int16）
void quantitative_weight_int16_wrapper(
    int input_size, int hidden_size,
    const torch::Tensor &W,
    const torch::Tensor &R,
    const torch::Tensor &bx,
    const torch::Tensor &br,
    const GRUQuantitativeParametersPy &quant_params,
    torch::Tensor &W_quant,
    torch::Tensor &R_quant,
    torch::Tensor &bx_quant,
    torch::Tensor &br_quant) {

    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(W_quant.is_cuda() && W_quant.dtype() == torch::kInt16, "W_quant must be CUDA int16 tensor");
    TORCH_CHECK(R_quant.is_cuda() && R_quant.dtype() == torch::kInt16, "R_quant must be CUDA int16 tensor");
    TORCH_CHECK(bx_quant.is_cuda() && bx_quant.dtype() == torch::kInt32, "bx_quant must be CUDA int32 tensor");
    TORCH_CHECK(br_quant.is_cuda() && br_quant.dtype() == torch::kInt32, "br_quant must be CUDA int32 tensor");

    GRUQuantitativeParameters cpp_params = quant_params.to_cpp();

    quantitativeWeight<int16_t>(
        input_size, hidden_size,
        W.data_ptr<float>(),
        R.data_ptr<float>(),
        bx.data_ptr<float>(),
        br.data_ptr<float>(),
        cpp_params,
        W_quant.data_ptr<int16_t>(),
        R_quant.data_ptr<int16_t>(),
        bx_quant.data_ptr<int32_t>(),
        br_quant.data_ptr<int32_t>());
}

// 量化 GRU 前向传播（int8）
std::tuple<torch::Tensor, torch::Tensor> quant_gru_forward_int8_wrapper(
    bool is_training,// 是否开启训练模式，true为训练，false为推理
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W_quant,
    const torch::Tensor &R_quant,
    const torch::Tensor &bx_quant,
    const torch::Tensor &br_quant,
    const torch::Tensor &x,
    const torch::Tensor &h0,// 初始隐藏状态，可以为空张量
    const GRUQuantitativeParametersPy &quant_params) {

    TORCH_CHECK(W_quant.is_cuda() && W_quant.dtype() == torch::kInt8, "W_quant must be CUDA int8 tensor");
    TORCH_CHECK(R_quant.is_cuda() && R_quant.dtype() == torch::kInt8, "R_quant must be CUDA int8 tensor");
    TORCH_CHECK(bx_quant.is_cuda() && bx_quant.dtype() == torch::kInt32, "bx_quant must be CUDA int32 tensor");
    TORCH_CHECK(br_quant.is_cuda() && br_quant.dtype() == torch::kInt32, "br_quant must be CUDA int32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // h0 可以为空张量（未提供初始状态）
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32, "h0 must be CUDA float32 tensor");
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

    quantGRUForward<int8_t>(
        is_training,
        time_steps, batch_size, input_size, hidden_size,
        W_quant.data_ptr<int8_t>(),
        R_quant.data_ptr<int8_t>(),
        bx_quant.data_ptr<int32_t>(),
        br_quant.data_ptr<int32_t>(),
        x.data_ptr<float>(),
        h0_ptr,// 初始隐藏状态，可以为 nullptr
        cpp_params,
        g_blas_handle,
        h.data_ptr<float>(),
        v.data_ptr<float>()// 反量化后的v输出
    );

    return std::make_tuple(h, v);
}

// 量化 GRU 前向传播（int16）
std::tuple<torch::Tensor, torch::Tensor> quant_gru_forward_int16_wrapper(
    bool is_training,// 是否开启训练模式，true为训练，false为推理
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W_quant,
    const torch::Tensor &R_quant,
    const torch::Tensor &bx_quant,
    const torch::Tensor &br_quant,
    const torch::Tensor &x,
    const torch::Tensor &h0,// 初始隐藏状态，可以为空张量
    const GRUQuantitativeParametersPy &quant_params) {

    TORCH_CHECK(W_quant.is_cuda() && W_quant.dtype() == torch::kInt16, "W_quant must be CUDA int16 tensor");
    TORCH_CHECK(R_quant.is_cuda() && R_quant.dtype() == torch::kInt16, "R_quant must be CUDA int16 tensor");
    TORCH_CHECK(bx_quant.is_cuda() && bx_quant.dtype() == torch::kInt32, "bx_quant must be CUDA int32 tensor");
    TORCH_CHECK(br_quant.is_cuda() && br_quant.dtype() == torch::kInt32, "br_quant must be CUDA int32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // h0 可以为空张量（未提供初始状态）
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32, "h0 must be CUDA float32 tensor");
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

    quantGRUForward<int16_t>(
        is_training,
        time_steps, batch_size, input_size, hidden_size,
        W_quant.data_ptr<int16_t>(),
        R_quant.data_ptr<int16_t>(),
        bx_quant.data_ptr<int32_t>(),
        br_quant.data_ptr<int32_t>(),
        x.data_ptr<float>(),
        h0_ptr,// 初始隐藏状态，可以为 nullptr
        cpp_params,
        g_blas_handle,
        h.data_ptr<float>(),
        v.data_ptr<float>()// 反量化后的v输出
    );

    return std::make_tuple(h, v);
}

// 非量化 GRU 前向传播
std::tuple<torch::Tensor, torch::Tensor> haste_gru_forward_wrapper(
    bool is_training,// 是否开启训练模式，true为训练，false为推理
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W,
    const torch::Tensor &R,
    const torch::Tensor &bx,
    const torch::Tensor &br,
    const torch::Tensor &x,
    const torch::Tensor &h0) { // 初始隐藏状态，可以为空张量

    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

    // h0 可以为空张量（未提供初始状态）
    const float *h0_ptr = nullptr;
    if (h0.defined() && h0.numel() > 0) {
        TORCH_CHECK(h0.is_cuda() && h0.dtype() == torch::kFloat32, "h0 must be CUDA float32 tensor");
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

    hasteGRUForward(
        is_training,
        time_steps, batch_size, input_size, hidden_size,
        W.data_ptr<float>(),
        R.data_ptr<float>(),
        bx.data_ptr<float>(),
        br.data_ptr<float>(),
        x.data_ptr<float>(),
        h0_ptr,// 初始隐藏状态，可以为 nullptr
        g_blas_handle,
        h.data_ptr<float>(),
        v.data_ptr<float>()// 中间值v输出
    );

    return std::make_tuple(h, v);
}

// forwardInterface 的包装函数
std::tuple<torch::Tensor, torch::Tensor> forward_interface_wrapper(
    bool is_training,  // 是否开启训练模式，true为训练，false为推理
    bool is_quant,
    bool use_int16,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W,
    const torch::Tensor &R,
    const torch::Tensor &bx,
    const torch::Tensor &br,
    const torch::Tensor &x,
    const GRUQuantitativeParametersPy &quant_params) {

    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");

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

    forwardInterface(
        is_training,
        is_quant,
        use_int16,
        time_steps, batch_size, input_size, hidden_size,
        W.data_ptr<float>(),
        R.data_ptr<float>(),
        bx.data_ptr<float>(),
        br.data_ptr<float>(),
        x.data_ptr<float>(),
        cpp_params,
        g_blas_handle,
        h.data_ptr<float>(),
        v.data_ptr<float>()  // 传递v参数
    );

    return std::make_tuple(h, v);
}

// GRU 反向传播包装函数
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
haste_gru_backward_wrapper(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const torch::Tensor &W,
    const torch::Tensor &R,
    const torch::Tensor &bx,
    const torch::Tensor &br,
    const torch::Tensor &x,
    const torch::Tensor &dh_new,  // 来自上层网络或损失函数的反向梯度
    const torch::Tensor &h,        // 前向传播的隐藏状态
    const torch::Tensor &v) {      // 前向传播的中间值，必需

    // 检查输入张量的类型和设备
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kFloat32, "W must be CUDA float32 tensor");
    TORCH_CHECK(R.is_cuda() && R.dtype() == torch::kFloat32, "R must be CUDA float32 tensor");
    TORCH_CHECK(bx.is_cuda() && bx.dtype() == torch::kFloat32, "bx must be CUDA float32 tensor");
    TORCH_CHECK(br.is_cuda() && br.dtype() == torch::kFloat32, "br must be CUDA float32 tensor");
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32 tensor");
    TORCH_CHECK(dh_new.is_cuda() && dh_new.dtype() == torch::kFloat32, "dh_new must be CUDA float32 tensor");
    TORCH_CHECK(h.is_cuda() && h.dtype() == torch::kFloat32, "h must be CUDA float32 tensor");
    TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kFloat32, "v must be CUDA float32 tensor");

    // 检查张量形状
    TORCH_CHECK(W.sizes() == torch::IntArrayRef({input_size, hidden_size * 3}),
                "W must have shape [input_size, hidden_size * 3]");
    TORCH_CHECK(R.sizes() == torch::IntArrayRef({hidden_size, hidden_size * 3}),
                "R must have shape [hidden_size, hidden_size * 3]");
    TORCH_CHECK(bx.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "bx must have shape [hidden_size * 3]");
    TORCH_CHECK(br.sizes() == torch::IntArrayRef({hidden_size * 3}),
                "br must have shape [hidden_size * 3]");
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({time_steps, batch_size, input_size}),
                "x must have shape [time_steps, batch_size, input_size]");
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
    auto dbx = torch::zeros({hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dbr = torch::zeros({hidden_size * 3},
                           torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dh = torch::zeros({batch_size, hidden_size},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // 调用 C++ 函数
    // 注意：需要将张量展平为连续内存布局
    hasteGRUBackward(
        time_steps, batch_size, input_size, hidden_size,
        W.data_ptr<float>(),
        R.data_ptr<float>(),
        bx.data_ptr<float>(),
        br.data_ptr<float>(),
        x.data_ptr<float>(),
        dh_new.data_ptr<float>(),
        h.data_ptr<float>(),
        v.data_ptr<float>(),
        g_blas_handle,
        dx.data_ptr<float>(),
        dW.data_ptr<float>(),
        dR.data_ptr<float>(),
        dbx.data_ptr<float>(),
        dbr.data_ptr<float>(),
        dh.data_ptr<float>()
    );

    return std::make_tuple(dx, dW, dR, dbx, dbr, dh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GRU Interface Python Bindings";

    // 初始化 cublas handle
    m.def("init_gru_cublas", &init_gru_cublas_wrapper, "Initialize cuBLAS handle for GRU");

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
        .def_readwrite("exp2_inv_one_minus_update_", &GRUQuantitativeParametersPy::exp2_inv_one_minus_update_)
        .def_readwrite("zp_one_minus_update_", &GRUQuantitativeParametersPy::zp_one_minus_update_)
        .def_readwrite("exp2_inv_new_contrib_", &GRUQuantitativeParametersPy::exp2_inv_new_contrib_)
        .def_readwrite("zp_new_contrib_", &GRUQuantitativeParametersPy::zp_new_contrib_)
        .def_readwrite("exp2_inv_old_contrib_", &GRUQuantitativeParametersPy::exp2_inv_old_contrib_)
        .def_readwrite("zp_old_contrib_", &GRUQuantitativeParametersPy::zp_old_contrib_);

    // 校准量化参数
    m.def("calibrate_gru_scales", &calibrate_gru_scales_wrapper,
          "Calibrate GRU quantization scales",
          py::arg("use_int16"), py::arg("time_steps"), py::arg("batch_size"),
          py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"), py::arg("x"));

    // 量化权重（int8）
    m.def("quantitative_weight_int8", &quantitative_weight_int8_wrapper,
          "Quantize GRU weights to int8",
          py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("quant_params"),
          py::arg("W_quant"), py::arg("R_quant"), py::arg("bx_quant"), py::arg("br_quant"));

    // 量化权重（int16）
    m.def("quantitative_weight_int16", &quantitative_weight_int16_wrapper,
          "Quantize GRU weights to int16",
          py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("quant_params"),
          py::arg("W_quant"), py::arg("R_quant"), py::arg("bx_quant"), py::arg("br_quant"));

    // 量化 GRU 前向传播（int8）
    m.def("quant_gru_forward_int8", &quant_gru_forward_int8_wrapper,
          "Quantized GRU forward pass (int8)",
          py::arg("is_training"),// 是否开启训练模式，true为训练，false为推理
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W_quant"), py::arg("R_quant"), py::arg("bx_quant"), py::arg("br_quant"),
          py::arg("x"), py::arg("h0") = torch::Tensor(),// 初始隐藏状态，可选
          py::arg("quant_params"));// 返回 (h, v) 元组，h包含初始状态，v为反量化后的中间值

    // 量化 GRU 前向传播（int16）
    m.def("quant_gru_forward_int16", &quant_gru_forward_int16_wrapper,
          "Quantized GRU forward pass (int16)",
          py::arg("is_training"),// 是否开启训练模式，true为训练，false为推理
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W_quant"), py::arg("R_quant"), py::arg("bx_quant"), py::arg("br_quant"),
          py::arg("x"), py::arg("h0") = torch::Tensor(),// 初始隐藏状态，可选
          py::arg("quant_params"));// 返回 (h, v) 元组，h包含初始状态，v为反量化后的中间值

    // 非量化 GRU 前向传播
    m.def("haste_gru_forward", &haste_gru_forward_wrapper,
          "Non-quantized GRU forward pass",
          py::arg("is_training"),// 是否开启训练模式，true为训练，false为推理
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"), py::arg("x"),
          py::arg("h0") = torch::Tensor());// 初始隐藏状态，可选；返回 (h, v) 元组，h包含初始状态，v为中间值

    // forwardInterface 统一接口
    m.def("forward_interface", &forward_interface_wrapper,
          "Unified GRU forward interface supporting both quantized and non-quantized modes",
          py::arg("is_training"),// 是否开启训练模式，true为训练，false为推理
          py::arg("is_quant"), py::arg("use_int16"),
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"), py::arg("x"),
          py::arg("quant_params"));// 返回 (h, v) 元组，h包含初始状态，v为中间值

    // GRU 反向传播
    m.def("haste_gru_backward", &haste_gru_backward_wrapper,
          "Non-quantized GRU backward pass",
          py::arg("time_steps"), py::arg("batch_size"), py::arg("input_size"), py::arg("hidden_size"),
          py::arg("W"), py::arg("R"), py::arg("bx"), py::arg("br"),
          py::arg("x"), py::arg("dh_new"), py::arg("h"), py::arg("v"));// 中间值v，必需；返回 (dx, dW, dR, dbx, dbr, dh) 元组
}
