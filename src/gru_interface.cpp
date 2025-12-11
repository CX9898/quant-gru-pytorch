#include "gru_interface.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

#include "parallelAlgorithm.h"
#include "quantize_ops_helper.hpp"

void calibrateGruScales(bool use_int16, int time_steps, int batch_size, int input_size,
                        int hidden_size, const std::vector<float> &W, const std::vector<float> &R,
                        const std::vector<float> &bx, const std::vector<float> &br,
                        const std::vector<float> &x, const cublasHandle_t &g_blas_handle,
                        GRUQuantitativeParameters &quant_gru_scales) {
    // Copy weights over to GPU.
    dev::vector<float> W_dev(W);
    dev::vector<float> R_dev(R);
    dev::vector<float> bx_dev(bx);
    dev::vector<float> br_dev(br);
    dev::vector<float> x_dev(x);

    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    gru::ForwardPass<float> forward =
        gru::ForwardPass<float>(true,  // training
                                batch_size, input_size, hidden_size, g_blas_handle);

    forward.setCalibrationMode(true, use_int16);

    forward.Run(time_steps, W_dev.data(), R_dev.data(), bx_dev.data(), br_dev.data(), x_dev.data(),
                h_dev.data(), v_dev.data(), tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);

    quant_gru_scales = forward.getGRUQuantitativeParameters();
}

GRUQuantitativeParameters calibrateGruScales(bool use_int16, int time_steps, int batch_size,
                                             int input_size, int hidden_size, const float *W,
                                             const float *R, const float *bx, const float *br,
                                             const float *x, const cublasHandle_t &g_blas_handle) {
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    gru::ForwardPass<float> forward =
        gru::ForwardPass<float>(true,  // training
                                batch_size, input_size, hidden_size, g_blas_handle);

    forward.setCalibrationMode(true, use_int16);

    forward.Run(time_steps, W, R, bx, br, x, h_dev.data(), v_dev.data(), tmp_Wx_dev.data(),
                tmp_Rh_dev.data(), 0.0f, nullptr);

    // 同步所有 CUDA 操作，确保校准完成
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in calibrateGruScales: %s\n", err_str);
        // 抛出异常，让调用者知道校准失败
        throw std::runtime_error(std::string("CUDA error in calibrateGruScales: ") + err_str);
    }

    // 确保 cublas stream 已恢复（forward.Run() 内部会恢复，但这里再次确认）
    cudaStream_t current_stream;
    cublasGetStream(g_blas_handle, &current_stream);
    // 如果 stream 不是默认的，同步它
    if (current_stream != nullptr) {
        cudaStreamSynchronize(current_stream);
    }

    return forward.getGRUQuantitativeParameters();
}

// 校准量化参数并初始化 LUT 表（组合函数，方便使用）
// 内部会根据 use_int16 参数自动选择相应的 LUT 初始化方法
GRUQuantitativeParameters calibrateGruScalesAndInitLut(bool use_int16, int time_steps,
                                                       int batch_size, int input_size,
                                                       int hidden_size, const float *W,
                                                       const float *R, const float *bx,
                                                       const float *br, const float *x,
                                                       const cublasHandle_t &g_blas_handle) {
    // 先校准量化参数
    GRUQuantitativeParameters quant_params = calibrateGruScales(
        use_int16, time_steps, batch_size, input_size, hidden_size, W, R, bx, br, x, g_blas_handle);

    // 初始化 LUT 表（内部会根据 use_int16 自动选择方法）
    initialize_quantization_lut(quant_params, use_int16);

    return quant_params;
}

void hasteGRUForward(
    bool is_training,  // 是否开启训练模式，true为训练，false为推理
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W, const float *R, const float *bx, const float *br, const float *x,
    const float *h0,  // 初始隐藏状态，可以为 nullptr
    const cublasHandle_t &g_blas_handle,
    float *h,  // (time_steps + 1) * batch_size * hidden_size
    float *v   // (time_steps * batch_size * hidden_size * 4)，中间值v，可以为 nullptr
) {
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
}

void hasteGRUBackward(
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W, const float *R, const float *bx, const float *br, const float *x,
    const float *dh_new,
    const float *h,  // (time_steps + 1) * batch_size * hidden_size
    const float *v,  // (time_steps * batch_size * hidden_size * 4)，中间值v，可以为 nullptr
    const cublasHandle_t &g_blas_handle,
    float *dx,   // (time_steps *batch_size * input_size) 输入序列梯度
    float *dW,   // (input_size * hidden_size * 3)// 对输入权重的梯度
    float *dR,   // (hidden_size * hidden_size * 3) // 对循环权重的梯度
    float *dbx,  // (hidden_size * 3)// 对输入偏置的梯度
    float *dbr,  // (hidden_size * 3)// 对循环偏置的梯度
    float *dh    // (batch_size * hidden_size)// 对最后隐藏状态的梯度
) {
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）

    gru::BackwardPass<float> backward(batch_size, input_size, hidden_size, g_blas_handle);

    backward.Run(time_steps, W, R, bx, br, x, h, v, dh_new, dx, dW, dR, dbx, dbr, dh, dp_dev.data(),
                 dq_dev.data(), nullptr);
}

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
}

template <typename QuantT>
void quantGRUForward(
    bool is_training,  // 是否开启训练模式，true为训练，false为推理
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const QuantT *W, const QuantT *R, const int32_t *bx, const int32_t *br, const float *x,
    const float *h0,  // 初始隐藏状态，可以为 nullptr
    const GRUQuantitativeParameters &quant_parms, const cublasHandle_t &g_blas_handle,
    float *h,  // (time_steps + 1) * batch_size * hidden_size
    float *v   // (time_steps * batch_size * hidden_size * 4)，反量化后的v，可以为 nullptr
) {
    const std::size_t x_size = time_steps * batch_size * input_size;

    dev::vector<QuantT> x_quant(x_size);
    dev::quantification(x, x_quant.data(), x_size, quant_parms.exp2_inv_x_, quant_parms.zp_x_);

    dev::vector<QuantT> h_quant((time_steps + 1) * batch_size * hidden_size);
    // 初始化 h0 区域（第一个时间步的隐藏状态）为零点值
    dev::fill_n(h_quant.data(), batch_size * hidden_size, quant_parms.zp_h_);

    // 处理初始隐藏状态
    if (h0 != nullptr) {
        // 如果提供了初始状态，直接量化到 h_quant[0]
        dev::quantification(h0, h_quant.data(), batch_size * hidden_size, quant_parms.exp2_inv_h_,
                            quant_parms.zp_h_);
    }

    dev::vector<QuantT> v_quant_dev(time_steps * batch_size * hidden_size * 4);
    dev::vector<int32_t> tmp_Wx_dev(time_steps * batch_size * hidden_size *
                                    3);                             // 用于存放W * x的中间结果
    dev::vector<int32_t> tmp_Rh_dev(batch_size * hidden_size * 3);  // 用于存放R * h的中间结果

    gru::ForwardPassQuant<QuantT> forward =
        gru::ForwardPassQuant<QuantT>(is_training,  // training: true为训练，false为推理
                                      batch_size, input_size, hidden_size, g_blas_handle);

    // 得到量化GRU中使用的rescale参数
    forward.setRescaleParam(quant_parms);

    forward.Run(time_steps, W, R, bx, br, x_quant.data(), h_quant.data(), v_quant_dev.data(),
                tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);

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
}

void forwardInterface(
    bool is_training,  // 是否开启训练模式，true为训练，false为推理
    bool is_quant, bool use_int16, int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bx, const float *br, const float *x,
    const float *h0,  // 初始隐藏状态，可以为 nullptr
    const GRUQuantitativeParameters &quant_gru_scales, const cublasHandle_t &g_blas_handle,
    float *h,    // (time_steps + 1) * batch_size * hidden_size，包含初始状态
    float *v) {  // (time_steps * batch_size * hidden_size * 4)，中间值v，可以为 nullptr
    if (is_quant) {
        if (use_int16) {
            dev::vector<int16_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int16_t> R_quant(hidden_size * 3 * hidden_size);
            dev::vector<int32_t> bx_quant(hidden_size * 3);
            dev::vector<int32_t> br_quant(hidden_size * 3);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales,
                               W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward(is_training, time_steps, batch_size, input_size, hidden_size,
                            W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data(), x, h0,
                            quant_gru_scales, g_blas_handle, h, v);
        } else {
            dev::vector<int8_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int8_t> R_quant(hidden_size * 3 * hidden_size);
            dev::vector<int32_t> bx_quant(hidden_size * 3);
            dev::vector<int32_t> br_quant(hidden_size * 3);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales,
                               W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward(is_training, time_steps, batch_size, input_size, hidden_size,
                            W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data(), x, h0,
                            quant_gru_scales, g_blas_handle, h, v);
        }
    } else {
        hasteGRUForward(is_training, time_steps, batch_size, input_size, hidden_size, W, R, bx, br,
                        x, h0, g_blas_handle, h, v);
    }
}

template <typename QuantT>
void GruQuantInit(
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W,   // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float *R,   // 隐藏层到隐藏层的循环权重矩阵
    const float *bx,  // 输入偏置项（input bias），来自输入路径
    const float *br,  // 循环偏置项（recurrent bias），来自循环路径
    const float *x,   // 输入序列张量
    QuantT *W_quant, QuantT *R_quant, int32_t *bx_quant, int32_t *br_quant, QuantT *x_quant,
    const GRUQuantitativeParameters &gruRescaleParams) {
    const int channel_size = hidden_size * 3;
    // N : batch_size
    // C : input_size

    // 权重是per-channel的，大小为H * 3（hidden_size * 3）
    // W: [H*3, C]，W_quant: [H*3, C]，scale_W_: [H*3]
    quantificationPerChannel(W, W_quant, input_size, channel_size, gruRescaleParams.exp2_inv_W_);
    // R: [H*3, H]，R_quant: [H*3, H]，scale_R_: [H*3]
    quantificationPerChannel(R, R_quant, hidden_size, channel_size, gruRescaleParams.exp2_inv_R_);

    // 偏置per-channel，H*3
    // bx_quant: [H*3], scale_bx_: [H*3]
    quantificationPerChannel(bx, bx_quant, 1, channel_size, gruRescaleParams.exp2_inv_bx_);
    // br_quant: [H*3], scale_br_: [H*3]
    quantificationPerChannel(br, br_quant, 1, channel_size, gruRescaleParams.exp2_inv_br_);

    // x: [C, N, T], x_quant: [C, N, T]
    // 量化用全局scale_x_和zp_x_
    quantification(x, x_quant, time_steps * batch_size * input_size, gruRescaleParams.exp2_inv_x_,
                   gruRescaleParams.zp_x_);
}

template void GruQuantInit<int8_t>(
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W,   // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float *R,   // 隐藏层到隐藏层的循环权重矩阵
    const float *bx,  // 输入偏置项（input bias），来自输入路径
    const float *br,  // 循环偏置项（recurrent bias），来自循环路径
    const float *x,   // 输入序列张量
    int8_t *W_quant, int8_t *R_quant, int32_t *bx_quant, int32_t *br_quant, int8_t *x_quant,
    const GRUQuantitativeParameters &gruRescaleParams);

template void GruQuantInit<int16_t>(
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W,   // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float *R,   // 隐藏层到隐藏层的循环权重矩阵
    const float *bx,  // 输入偏置项（input bias），来自输入路径
    const float *br,  // 循环偏置项（recurrent bias），来自循环路径
    const float *x,   // 输入序列张量
    int16_t *W_quant, int16_t *R_quant, int32_t *bx_quant, int32_t *br_quant, int16_t *x_quant,
    const GRUQuantitativeParameters &gruRescaleParams);

// 显式实例化 quantitativeWeight 和 quantGRUForward 模板函数，供 Python 绑定使用
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

template void quantGRUForward<int8_t>(bool is_training, const int time_steps, const int batch_size,
                                      const int input_size, const int hidden_size, const int8_t *W,
                                      const int8_t *R, const int32_t *bx, const int32_t *br,
                                      const float *x,
                                      const float *h0,  // 初始隐藏状态，可以为 nullptr
                                      const GRUQuantitativeParameters &quant_parms,
                                      const cublasHandle_t &g_blas_handle, float *h, float *v);

template void quantGRUForward<int16_t>(bool is_training, const int time_steps, const int batch_size,
                                       const int input_size, const int hidden_size,
                                       const int16_t *W, const int16_t *R, const int32_t *bx,
                                       const int32_t *br, const float *x,
                                       const float *h0,  // 初始隐藏状态，可以为 nullptr
                                       const GRUQuantitativeParameters &quant_parms,
                                       const cublasHandle_t &g_blas_handle, float *h, float *v);

// 初始化量化 LUT 表（仅在初始化时调用一次）
// 接收量化参数对象和量化类型，内部根据类型自动选择相应的 LUT 初始化方法
// 支持 int8 和 int16，未来可扩展支持其他类型
// 解耦实现细节，上层无需关注具体参数和初始化方法
void initialize_quantization_lut(const GRUQuantitativeParameters &quant_params, bool use_int16) {
    if (use_int16) {
        // int16 使用分段线性量化表
        // x_min 和 x_max 定义了 LUT 表的输入范围：
        // 注意：虽然三个门使用相同的范围，但函数支持为每个门单独设置范围以提供灵活性
        // 注意：模板参数使用 int16_t（表示16位量化），函数内部会自动处理无符号量化范围
        generate_piecewise_linear_lut_from_exp2_inv<int16_t>(
            quant_params.exp2_inv_z_pre_, quant_params.zp_z_pre_, quant_params.exp2_inv_z_out_,
            quant_params.zp_z_out_, quant_params.exp2_inv_r_pre_, quant_params.zp_r_pre_,
            quant_params.exp2_inv_r_out_, quant_params.zp_r_out_, quant_params.exp2_inv_g_pre_,
            quant_params.zp_g_pre_, quant_params.exp2_inv_g_out_, quant_params.zp_g_out_);
    } else {
        // int8 使用传统的 LUT 表
        generate_int8_lut_from_exp2_inv(
            quant_params.exp2_inv_z_pre_, quant_params.zp_z_pre_, quant_params.exp2_inv_z_out_,
            quant_params.zp_z_out_, quant_params.exp2_inv_r_pre_, quant_params.zp_r_pre_,
            quant_params.exp2_inv_r_out_, quant_params.zp_r_out_, quant_params.exp2_inv_g_pre_,
            quant_params.zp_g_pre_, quant_params.exp2_inv_g_out_, quant_params.zp_g_out_);
    }
}
