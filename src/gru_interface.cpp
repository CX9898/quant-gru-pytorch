#include "gru_interface.hpp"


void calibrateGruScales(
    bool use_int16,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const std::vector<float> &W,
    const std::vector<float> &R,
    const std::vector<float> &bx,
    const std::vector<float> &br,
    const std::vector<float> &x,
    const cublasHandle_t &g_blas_handle,
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

    gru::ForwardPass<float> forward = gru::ForwardPass<float>(
        true,// training
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

GRUQuantitativeParameters calibrateGruScales(
    bool use_int16,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W,
    const float *R,
    const float *bx,
    const float *br,
    const float *x,
    const cublasHandle_t &g_blas_handle) {
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    gru::ForwardPass<float> forward = gru::ForwardPass<float>(
        true,// training
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    forward.setCalibrationMode(true, use_int16);

    forward.Run(
        time_steps,
        W,
        R,
        bx,
        br,
        x,
        h_dev.data(),
        v_dev.data(),
        tmp_Wx_dev.data(),
        tmp_Rh_dev.data(),
        0.0f,
        nullptr);

    return forward.getGRUQuantitativeParameters();
}

void hasteGRUForward(const int time_steps,
                     const int batch_size,
                     const int input_size,
                     const int hidden_size,
                     const float *W, const float *R, const float *bx,
                     const float *br, const float *x,
                     const cublasHandle_t &g_blas_handle,
                     float *h) {
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size *
                                  3);// 用于存放W * x的中间结果
    dev::vector<float> tmp_Rh_dev(batch_size * hidden_size *
                                  3);// 用于存放R * h的中间结果

    gru::ForwardPass<float> forward = gru::ForwardPass<float>(
        false,// training
        batch_size, input_size, hidden_size, g_blas_handle);

    forward.Run(time_steps, W, R, bx,
                br, x, h_dev.data(), nullptr,
                tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);

    d2d(h, h_dev.data() + batch_size * hidden_size, time_steps * batch_size * hidden_size);
}

template<typename QuantT>
void quantitativeWeight(const int input_size, const int hidden_size,
                        const float *W, const float *R, const float *bx, const float *br,
                        const GRUQuantitativeParameters &quant_parms,
                        QuantT *W_quant, QuantT *R_quant, int32_t *bx_quant, int32_t *br_quant) {
    dev::quantificationPerChannel(
        W, W_quant, input_size,
        3 * hidden_size, quant_parms.exp2_inv_W_);
    dev::quantificationPerChannel(
        R, R_quant, hidden_size,
        3 * hidden_size, quant_parms.exp2_inv_R_);

    dev::vector<int32_t> exp2_inv_bx(quant_parms.exp2_inv_bx_);
    dev::quantificationPerChannel(bx,
                                  bx_quant, 1,
                                  3 * hidden_size, exp2_inv_bx);
    dev::vector<int32_t> exp2_inv_br(quant_parms.exp2_inv_br_);
    dev::quantificationPerChannel(br,
                                  br_quant, 1,
                                  3 * hidden_size, exp2_inv_br);
}

template<typename QuantT>
void quantGRUForward(const int time_steps, const int batch_size, const int input_size,
                     const int hidden_size, const QuantT *W, const QuantT *R, const int32_t *bx,
                     const int32_t *br, const float *x,
                     const GRUQuantitativeParameters &quant_parms,
                     const cublasHandle_t &g_blas_handle,
                     float *h// (time_steps) * batch_size * hidden_size
) {

    const std::size_t x_size = time_steps * batch_size * input_size;

    dev::vector<QuantT> x_quant(x_size);
    dev::quantification(x, x_quant.data(), x_size, quant_parms.exp2_inv_x_,
                        quant_parms.zp_x_);

    dev::vector<QuantT> h_quant((time_steps + 1) * batch_size * hidden_size);
    h_quant.zero();

    generate_int8_lut_from_exp2_inv(
        quant_parms.exp2_inv_z_pre_, quant_parms.zp_z_pre_,
        quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_,
        quant_parms.exp2_inv_r_pre_, quant_parms.zp_r_pre_,
        quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_,
        quant_parms.exp2_inv_g_pre_, quant_parms.zp_g_pre_,
        quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_);


    dev::vector<int32_t> tmp_Wx_dev(time_steps * batch_size * hidden_size *
                                    3);// 用于存放W * x的中间结果
    dev::vector<int32_t> tmp_Rh_dev(batch_size * hidden_size *
                                    3);// 用于存放R * h的中间结果

    {
        gru::ForwardPassQuant<QuantT> forward = gru::ForwardPassQuant<QuantT>(
            false,// training
            batch_size, input_size, hidden_size, g_blas_handle);

        // 得到量化GRU中使用的rescale参数
        forward.setRescaleParam(quant_parms);

        forward.Run(time_steps, W, R, bx,
                    br, x_quant.data(), h_quant.data(),
                    nullptr, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f,
                    nullptr);
    }

    dev::dequantification(h_quant.data() + batch_size * hidden_size,
                          h,
                          time_steps * batch_size * hidden_size,
                          quant_parms.exp2_inv_h_, quant_parms.zp_h_);
}

void forwardInterface(bool is_quant,
                      bool use_int16,
                      int time_steps, int batch_size, int input_size, int hidden_size,
                      const float *W,
                      const float *R,
                      const float *bx,
                      const float *br,
                      const float *x,
                      const GRUQuantitativeParameters &quant_gru_scales,
                      const cublasHandle_t &g_blas_handle,
                      float *h) {
    if (is_quant) {
        if (use_int16) {
            dev::vector<int16_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int16_t> R_quant(hidden_size * 3 * hidden_size);
            dev::vector<int32_t> bx_quant(hidden_size * 3);
            dev::vector<int32_t> br_quant(hidden_size * 3);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales, W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward(time_steps, batch_size, input_size, hidden_size,
                            W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data(), x, quant_gru_scales, g_blas_handle, h);
        } else {
            dev::vector<int8_t> W_quant(hidden_size * 3 * input_size);
            dev::vector<int8_t> R_quant(hidden_size * 3 * hidden_size);
            dev::vector<int32_t> bx_quant(hidden_size * 3);
            dev::vector<int32_t> br_quant(hidden_size * 3);
            quantitativeWeight(input_size, hidden_size, W, R, bx, br, quant_gru_scales, W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data());
            quantGRUForward(time_steps, batch_size, input_size, hidden_size,
                            W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data(), x, quant_gru_scales, g_blas_handle, h);
        }
    } else {
        hasteGRUForward(time_steps, batch_size, input_size, hidden_size, W, R, bx, br, x, g_blas_handle, h);
    }
}

template<typename QuantT>
void GruQuantInit(
    const int time_steps,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float *W,     // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float *R,     // 隐藏层到隐藏层的循环权重矩阵
    const float *bx,    // 输入偏置项（input bias），来自输入路径
    const float *br,    // 循环偏置项（recurrent bias），来自循环路径
    const float *x,     // 输入序列张量
    const float *dh_new,// 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    QuantT *W_quant,
    QuantT *R_quant,
    int32_t *bx_quant,
    int32_t *br_quant,
    QuantT *x_quant,
    QuantT *dh_new_quant,
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
    const int time_steps,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float *W,     // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float *R,     // 隐藏层到隐藏层的循环权重矩阵
    const float *bx,    // 输入偏置项（input bias），来自输入路径
    const float *br,    // 循环偏置项（recurrent bias），来自循环路径
    const float *x,     // 输入序列张量
    const float *dh_new,// 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    int8_t *W_quant,
    int8_t *R_quant,
    int32_t *bx_quant,
    int32_t *br_quant,
    int8_t *x_quant,
    int8_t *dh_new_quant,
    const GRUQuantitativeParameters &gruRescaleParams);

template void GruQuantInit<int16_t>(
    const int time_steps,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float *W,     // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float *R,     // 隐藏层到隐藏层的循环权重矩阵
    const float *bx,    // 输入偏置项（input bias），来自输入路径
    const float *br,    // 循环偏置项（recurrent bias），来自循环路径
    const float *x,     // 输入序列张量
    const float *dh_new,// 来自上层网络或损失函数的反向梯度. [hidden_size, batch_size, time_steps]
    int16_t *W_quant,
    int16_t *R_quant,
    int32_t *bx_quant,
    int32_t *br_quant,
    int16_t *x_quant,
    int16_t *dh_new_quant,
    const GRUQuantitativeParameters &gruRescaleParams);

// 显式实例化 quantitativeWeight 和 quantGRUForward 模板函数，供 Python 绑定使用
template void quantitativeWeight<int8_t>(
    const int input_size, const int hidden_size,
    const float *W, const float *R, const float *bx, const float *br,
    const GRUQuantitativeParameters &quant_parms,
    int8_t *W_quant, int8_t *R_quant, int32_t *bx_quant, int32_t *br_quant);

template void quantitativeWeight<int16_t>(
    const int input_size, const int hidden_size,
    const float *W, const float *R, const float *bx, const float *br,
    const GRUQuantitativeParameters &quant_parms,
    int16_t *W_quant, int16_t *R_quant, int32_t *bx_quant, int32_t *br_quant);

template void quantGRUForward<int8_t>(
    const int time_steps, const int batch_size, const int input_size,
    const int hidden_size, const int8_t *W, const int8_t *R, const int32_t *bx,
    const int32_t *br, const float *x,
    const GRUQuantitativeParameters &quant_parms,
    const cublasHandle_t &g_blas_handle,
    float *h);

template void quantGRUForward<int16_t>(
    const int time_steps, const int batch_size, const int input_size,
    const int hidden_size, const int16_t *W, const int16_t *R, const int32_t *bx,
    const int32_t *br, const float *x,
    const GRUQuantitativeParameters &quant_parms,
    const cublasHandle_t &g_blas_handle,
    float *h);
