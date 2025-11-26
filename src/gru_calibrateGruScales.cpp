#include <cublas_v2.h>

#include "gru.h"

void calibrateGruScales(
    bool use_int16,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float* W,
    const float* R,
    const float* bx,
    const float* br,
    const float* x,
    const cublasHandle_t& g_blas_handle,
    GRUQuantitativeParameters& quant_gru_scales
)
{
    // Copy weights over to GPU.
    dev::vector<float> W_dev(W, hidden_size * 3 * input_size);
    dev::vector<float> R_dev(R, hidden_size * 3 * hidden_size);
    dev::vector<float> bx_dev(bx, hidden_size * 3);
    dev::vector<float> br_dev(br, hidden_size * 3);
    dev::vector<float> x_dev(x, input_size * batch_size * time_steps);
    //    dev::vector<float> dh_new_dev(dh_new);

    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    gru::ForwardPass<float> forward = gru::ForwardPass<float>(
        true, // training
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    forward.setCalibrationMode(true, use_int16);

    forward.Run(
        time_steps,
        W_dev.data(),
        R_dev.data(),
        bx_dev.data(),
        br_dev.data(),
        x_dev.data(),
        h_dev.data(),
        v_dev.data(),
        tmp_Wx_dev.data(),
        tmp_Rh_dev.data(),
        0.0f,
        nullptr);

    quant_gru_scales = forward.getGRUQuantitativeParameters();
}

template <typename QuantT>
void GruQuantInit(
    const int time_steps,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float* W, // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float* R, // 隐藏层到隐藏层的循环权重矩阵
    const float* bx, // 输入偏置项（input bias），来自输入路径
    const float* br, // 循环偏置项（recurrent bias），来自循环路径
    const float* x, // 输入序列张量
    const float* dh_new, // 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    QuantT* W_quant,
    QuantT* R_quant,
    int32_t* bx_quant,
    int32_t* br_quant,
    QuantT* x_quant,
    QuantT* dh_new_quant,
    const GRUQuantitativeParameters& gruRescaleParams
)
{
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

    // dh_new: [H, N, T+1], dh_new_quant: [H, N, T+1]
    // TODO: 是不是应该dh有自己的scale
    quantification(dh_new, dh_new_quant, (time_steps + 1) * batch_size * hidden_size, gruRescaleParams.exp2_inv_h_,
                   gruRescaleParams.zp_h_);
}

template
void GruQuantInit<int8_t>(
    const int time_steps,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float* W, // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float* R, // 隐藏层到隐藏层的循环权重矩阵
    const float* bx, // 输入偏置项（input bias），来自输入路径
    const float* br, // 循环偏置项（recurrent bias），来自循环路径
    const float* x, // 输入序列张量
    const float* dh_new, // 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    int8_t* W_quant,
    int8_t* R_quant,
    int32_t* bx_quant,
    int32_t* br_quant,
    int8_t* x_quant,
    int8_t* dh_new_quant,
    const GRUQuantitativeParameters& gruRescaleParams
);

template
void GruQuantInit<int16_t>(
    const int time_steps,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float* W, // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float* R, // 隐藏层到隐藏层的循环权重矩阵
    const float* bx, // 输入偏置项（input bias），来自输入路径
    const float* br, // 循环偏置项（recurrent bias），来自循环路径
    const float* x, // 输入序列张量
    const float* dh_new, // 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    int16_t* W_quant,
    int16_t* R_quant,
    int32_t* bx_quant,
    int32_t* br_quant,
    int16_t* x_quant,
    int16_t* dh_new_quant,
    const GRUQuantitativeParameters& gruRescaleParams
);
