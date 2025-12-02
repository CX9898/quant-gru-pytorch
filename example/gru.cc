#include "gru.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "checkData.hpp"
#include "devVector.h"
#include "gru_interface.hpp"
#include "gru_quant.h"
#include "quantized_unit_testing.cuh"

constexpr int BATCH_SIZE = 64;   // 批大小
constexpr int SEQUENCE_LEN = 500;// 序列长度(T), 每个样本有T个时间步
constexpr int HIDDEN_DIMS = 256; // 隐藏层维度(H), h_t的维度
constexpr int INPUT_DIMS = 256;  // 输入维度(I), x_t的维度

cublasHandle_t g_blas_handle;// 改为非static以便在wrapper中访问

// 初始化函数，供Python绑定调用
void init_gru_cublas() {
    if (g_blas_handle == nullptr) {
        cublasCreate(&g_blas_handle);
    }
}

class ScopeTimer {
    // 测量时间类
 public:
    ScopeTimer(const std::string &msg) : msg_(msg) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaDeviceSynchronize();
        cudaEventRecord(start_);
    }

    ~ScopeTimer() {
        float elapsed_ms;
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed_ms, start_, stop_);
        printf("%s %fms\n", msg_.c_str(), elapsed_ms);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

 private:
    std::string msg_;
    cudaEvent_t start_, stop_;
};

template<typename QuantT>
void GruInferenceQuant(
    const int time_steps, const int batch_size, const int input_size,
    const int hidden_size, const QuantT *W, const QuantT *R, const int32_t *bx,
    const int32_t *br, const float *x,
    const GRUQuantitativeParameters &quant_parms,
    int8_t *h_quant_out// (time_steps + 1) * batch_size * hidden_size
) {
    generate_int8_lut_from_exp2_inv(
        quant_parms.exp2_inv_z_pre_, quant_parms.zp_z_pre_,
        quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_,
        quant_parms.exp2_inv_r_pre_, quant_parms.zp_r_pre_,
        quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_,
        quant_parms.exp2_inv_g_pre_, quant_parms.zp_g_pre_,
        quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_);

    // Copy weights over to GPU.
    dev::vector<QuantT> W_dev(W, input_size * hidden_size * 3);
    dev::vector<QuantT> R_dev(R, hidden_size * hidden_size * 3);
    dev::vector<int32_t> bx_dev(bx, hidden_size * 3);
    dev::vector<int32_t> br_dev(br, hidden_size * 3);

    const std::size_t x_size = time_steps * batch_size * input_size;

    std::vector<QuantT> x_quant(x_size);
    quantification(x, x_quant.data(), x_size, quant_parms.exp2_inv_x_,
                   quant_parms.zp_x_);

    dev::vector<QuantT> x_quant_dev(x_quant);

    const std::size_t h_size = (time_steps + 1) * batch_size * hidden_size;
    dev::vector<QuantT> h_quant_dev(h_size, quant_parms.zp_h_);

    dev::vector<int32_t> tmp_Wx_dev(time_steps * batch_size * hidden_size *
                                    3);// 用于存放W * x的中间结果
    dev::vector<int32_t> tmp_Rh_dev(batch_size * hidden_size *
                                    3);// 用于存放R * h的中间结果

    {
        gru::ForwardPassQuant<int8_t> forward = gru::ForwardPassQuant<int8_t>(
            false,// training
            batch_size, input_size, hidden_size, g_blas_handle);

        // 得到量化GRU中使用的rescale参数
        forward.setRescaleParam(quant_parms);

        ScopeTimer t("Inference Quant:");
        forward.Run(time_steps, W_dev.data(), R_dev.data(), bx_dev.data(),
                    br_dev.data(), x_quant_dev.data(), h_quant_dev.data(),
                    nullptr, tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f,
                    nullptr);
    }

    d2h(h_quant_out, h_quant_dev.data(), h_quant_dev.size());
}

void GruInference(const int time_steps,
                  const int batch_size,
                  const int input_size,
                  const int hidden_size,
                  const float *W, const float *R, const float *bx,
                  const float *br, const float *x, float *h) {

    // Copy weights over to GPU.
    dev::vector<float> W_dev(W, hidden_size * 3 * input_size);
    dev::vector<float> R_dev(R, hidden_size * 3 * hidden_size);
    dev::vector<float> bx_dev(bx, hidden_size * 3);
    dev::vector<float> br_dev(br, hidden_size * 3);
    dev::vector<float> x_dev(x, time_steps * input_size * batch_size);

    dev::vector<float> h_dev(hidden_size * batch_size * (time_steps + 1));
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size *
                                  3);// 用于存放W * x的中间结果
    dev::vector<float> tmp_Rh_dev(batch_size * hidden_size *
                                  3);// 用于存放R * h的中间结果

    h_dev.zero();// h初始化为0

    {
        ScopeTimer t("Inference:");

        gru::ForwardPass<float> forward = gru::ForwardPass<float>(
            false,// training
            batch_size, input_size, hidden_size, g_blas_handle);

        forward.Run(time_steps, W_dev.data(), R_dev.data(), bx_dev.data(),
                    br_dev.data(), x_dev.data(), h_dev.data(), nullptr,
                    tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);
    }

    d2h(h, h_dev.data(), h_dev.size());
}

// 梯度输出结构体
struct GRUTrainGradients {
    std::vector<float> dx; // 输入序列梯度 [time_steps * batch_size * input_size]
    std::vector<float> dW; // 对输入权重的梯度 [input_size * hidden_size * 3]
    std::vector<float> dR; // 对循环权重的梯度 [hidden_size * hidden_size * 3]
    std::vector<float> dbx;// 对输入偏置的梯度 [hidden_size * 3]
    std::vector<float> dbr;// 对循环偏置的梯度 [hidden_size * 3]
    std::vector<float> dh; // 对最后隐藏状态的梯度 [batch_size * hidden_size]
    std::vector<float> v;  // V中间值 [time_steps * batch_size * hidden_size * 4]
    std::vector<float> h;  // 隐藏状态 [time_steps * batch_size * hidden_size] (不包含初始状态)
};

template<typename QuantT>
GRUTrainGradients GruTrainQuant(const int time_steps,
                                const int batch_size,
                                const int input_size,
                                const int hidden_size,
                                const std::vector<float> &W,    // 输入到隐藏层的权重矩阵. [input_size,
                                                                // hidden_size * 3] 对应三个门
                                const std::vector<float> &R,    // 隐藏层到隐藏层的循环权重矩阵
                                const std::vector<float> &bx,   // 输入偏置项（input bias），来自输入路径
                                const std::vector<float> &br,   // 循环偏置项（recurrent bias），来自循环路径
                                const std::vector<float> &x,    // 输入序列张量
                                const std::vector<float> &dh_new// 来自上层网络或损失函数的反向梯度.
                                                                // [hidden_size, batch_size, time_steps]

) {
    // 步骤1: 先校验出量化参数
    GRUQuantitativeParameters quant_parms;
    {
        ScopeTimer t("Calibrate quant params:");
        calibrateGruScales(false, time_steps, batch_size, input_size, hidden_size,
                           W, R, bx, br, x,
                           g_blas_handle, quant_parms);
    }

    dev::vector<float> W_dev(W);  // 输入到隐藏层的权重矩阵. [input_size,
                                  // hidden_size * 3] 对应三个门
    dev::vector<float> R_dev(R);  // 隐藏层到隐藏层的循环权重矩阵
    dev::vector<float> bx_dev(bx);// 输入偏置项（input bias），来自输入路径
    dev::vector<float> br_dev(br);// 循环偏置项（recurrent bias），来自循环路径
    dev::vector<float> x_dev(x);

    // 步骤2: 将权重量化和x量化
    const int channel_size = hidden_size * 3;
    dev::vector<QuantT> W_quant_dev(input_size * hidden_size * 3);
    dev::vector<QuantT> R_quant_dev(hidden_size * hidden_size * 3);
    dev::vector<int32_t> bx_quant_dev(hidden_size * 3);
    dev::vector<int32_t> br_quant_dev(hidden_size * 3);
    const std::size_t x_size = time_steps * batch_size * input_size;
    dev::vector<QuantT> x_quant_dev(x_size);

    // 显式创建dev::vector以避免临时对象问题
    dev::vector<int32_t> exp2_inv_W_dev(quant_parms.exp2_inv_W_);
    dev::vector<int32_t> exp2_inv_R_dev(quant_parms.exp2_inv_R_);
    dev::vector<int32_t> exp2_inv_bx_dev(quant_parms.exp2_inv_bx_);
    dev::vector<int32_t> exp2_inv_br_dev(quant_parms.exp2_inv_br_);

    {
        ScopeTimer t("Quantize weights and x:");
        // 权重量化 (per-channel)
        dev::quantificationPerChannel(W_dev.data(), W_quant_dev.data(), input_size, channel_size,
                                      exp2_inv_W_dev);
        dev::quantificationPerChannel(R_dev.data(), R_quant_dev.data(), hidden_size, channel_size,
                                      exp2_inv_R_dev);
        // 偏置量化 (per-channel)
        dev::quantificationPerChannel(bx_dev.data(), bx_quant_dev.data(), 1, channel_size,
                                      exp2_inv_bx_dev);
        dev::quantificationPerChannel(br_dev.data(), br_quant_dev.data(), 1, channel_size,
                                      exp2_inv_br_dev);
        // x量化 (全局)
        dev::quantification(x_dev.data(), x_quant_dev.data(), x_size, quant_parms.exp2_inv_x_,
                            quant_parms.zp_x_);
    }

    // 生成LUT表
    generate_int8_lut_from_exp2_inv(
        quant_parms.exp2_inv_z_pre_, quant_parms.zp_z_pre_,
        quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_,
        quant_parms.exp2_inv_r_pre_, quant_parms.zp_r_pre_,
        quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_,
        quant_parms.exp2_inv_g_pre_, quant_parms.zp_g_pre_,
        quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_);

    const std::size_t h_size = (time_steps + 1) * batch_size * hidden_size;
    dev::vector<QuantT> h_quant_dev(h_size, quant_parms.zp_h_);
    dev::vector<QuantT> v_quant_dev(time_steps * batch_size * hidden_size * 4);

    dev::vector<int32_t> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<int32_t> tmp_Rh_dev(batch_size * hidden_size * 3);

    // 步骤3: 运行量化GRU (training模式)
    {
        ScopeTimer t("Train forward quant:");
        gru::ForwardPassQuant<QuantT> forward = gru::ForwardPassQuant<QuantT>(
            true,// training
            batch_size, input_size, hidden_size, g_blas_handle);

        forward.setRescaleParam(quant_parms);

        forward.Run(time_steps, W_quant_dev.data(), R_quant_dev.data(),
                    bx_quant_dev.data(), br_quant_dev.data(), x_quant_dev.data(),
                    h_quant_dev.data(), v_quant_dev.data(), tmp_Wx_dev.data(),
                    tmp_Rh_dev.data(), 0.0f, nullptr);
    }

    // 步骤4: 将所有量化值反量化
    dev::vector<float> W_dequant_dev(input_size * hidden_size * 3);
    dev::vector<float> R_dequant_dev(hidden_size * hidden_size * 3);
    dev::vector<float> bx_dequant_dev(hidden_size * 3);
    dev::vector<float> br_dequant_dev(hidden_size * 3);
    dev::vector<float> x_dequant_dev(x_size);
    dev::vector<float> h_dequant_dev(h_size);
    dev::vector<float> v_dequant_dev(time_steps * batch_size * hidden_size * 4);

    {
        ScopeTimer t("Dequantize all values:");
        // 反量化权重 (per-channel)
        dev::dequantificationPerChannel(W_quant_dev.data(), W_dequant_dev.data(),
                                        input_size, channel_size,
                                        quant_parms.exp2_inv_W_);
        dev::dequantificationPerChannel(R_quant_dev.data(), R_dequant_dev.data(),
                                        hidden_size, channel_size,
                                        quant_parms.exp2_inv_R_);
        // 反量化偏置 (per-channel)
        dev::dequantificationPerChannel(bx_quant_dev.data(), bx_dequant_dev.data(),
                                        1, channel_size,
                                        quant_parms.exp2_inv_bx_);
        dev::dequantificationPerChannel(br_quant_dev.data(), br_dequant_dev.data(),
                                        1, channel_size,
                                        quant_parms.exp2_inv_br_);
        // 反量化x (全局)
        dev::dequantification(x_quant_dev.data(), x_dequant_dev.data(), x_size,
                              quant_parms.exp2_inv_x_, quant_parms.zp_x_);
        // 反量化h (全局，但h的量化参数可能随时间步变化，这里使用固定参数)
        dev::dequantification(h_quant_dev.data(), h_dequant_dev.data(), h_size,
                              quant_parms.exp2_inv_h_, quant_parms.zp_h_);
        // 反量化v (v包含4个部分，每个部分使用不同的量化参数)
        dev::dequantificationV(v_quant_dev.data(), v_dequant_dev.data(),
                               time_steps, batch_size, hidden_size,
                               quant_parms.exp2_inv_z_out_, quant_parms.zp_z_out_,
                               quant_parms.exp2_inv_r_out_, quant_parms.zp_r_out_,
                               quant_parms.exp2_inv_g_out_, quant_parms.zp_g_out_,
                               quant_parms.exp2_inv_Rh_add_br_, quant_parms.zp_Rh_add_br_);
    }

    // Copy dh_new到GPU
    dev::vector<float> dh_new_dev(dh_new);

    // 步骤5: 反量化后传入BackwardPass<float>进行反向传播
    dev::vector<float> dx_dev(time_steps * batch_size * input_size);
    dev::vector<float> dW_dev(input_size * hidden_size * 3);
    dev::vector<float> dR_dev(hidden_size * hidden_size * 3);
    dev::vector<float> dbx_dev(hidden_size * 3);
    dev::vector<float> dbr_dev(hidden_size * 3);
    dev::vector<float> dh_dev(batch_size * hidden_size);
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size * 3);

    {
        ScopeTimer t("Train backward:");
        gru::BackwardPass<float> backward(batch_size, input_size, hidden_size,
                                          g_blas_handle);

        backward.Run(time_steps, W_dequant_dev.data(), R_dequant_dev.data(),
                     bx_dequant_dev.data(), br_dequant_dev.data(),
                     x_dequant_dev.data(), h_dequant_dev.data(), v_dequant_dev.data(),
                     dh_new_dev.data(), dx_dev.data(), dW_dev.data(), dR_dev.data(),
                     dbx_dev.data(), dbr_dev.data(), dh_dev.data(), dp_dev.data(),
                     dq_dev.data(), nullptr);
    }

    // 将梯度从GPU复制到CPU
    GRUTrainGradients gradients;

    d2h(gradients.dx, dx_dev);
    d2h(gradients.dW, dW_dev);
    d2h(gradients.dR, dR_dev);
    d2h(gradients.dbx, dbx_dev);
    d2h(gradients.dbr, dbr_dev);
    d2h(gradients.dh, dh_dev);

    // 将反量化后的V复制到CPU
    d2h(gradients.v, v_dequant_dev);

    // 将反量化后的h复制到CPU（跳过初始状态，只复制time_steps个时间步）
    const int h_output_size = time_steps * batch_size * hidden_size;
    gradients.h.resize(h_output_size);
    d2h(gradients.h.data(), h_dequant_dev.data() + batch_size * hidden_size, h_output_size);

    return gradients;
}

GRUTrainGradients GruTrain(const int time_steps,
                           const int batch_size,
                           const int input_size,
                           const int hidden_size,
                           const std::vector<float> &W,    // 输入到隐藏层的权重矩阵. [input_size,
                                                           // hidden_size * 3] 对应三个门
                           const std::vector<float> &R,    // 隐藏层到隐藏层的循环权重矩阵
                           const std::vector<float> &bx,   // 输入偏置项（input bias），来自输入路径
                           const std::vector<float> &br,   // 循环偏置项（recurrent bias），来自循环路径
                           const std::vector<float> &x,    // 输入序列张量
                           const std::vector<float> &dh_new// 来自上层网络或损失函数的反向梯度.
                                                           // [hidden_size, batch_size, time_steps]

) {

    // Copy weights over to GPU.
    dev::vector<float> W_dev(W);
    dev::vector<float> R_dev(R);
    dev::vector<float> bx_dev(bx);
    dev::vector<float> br_dev(br);
    dev::vector<float> x_dev(x);

    dev::vector<float> dh_new_dev(dh_new);

    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    {
        ScopeTimer t("Train forward:");
        gru::ForwardPass<float> forward = gru::ForwardPass<float>(
            true,// training
            batch_size, input_size, hidden_size, g_blas_handle);

        forward.Run(time_steps, W_dev.data(), R_dev.data(), bx_dev.data(),
                    br_dev.data(), x_dev.data(), h_dev.data(), v_dev.data(),
                    tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);
    }

    dev::vector<float> dx_dev(time_steps * batch_size *
                              input_size);// 输入序列梯度
    dev::vector<float> dW_dev(input_size * hidden_size *
                              3);// 对输入权重的梯度
    dev::vector<float> dR_dev(hidden_size * hidden_size *
                              3);               // 对循环权重的梯度
    dev::vector<float> dbx_dev(hidden_size * 3);// 对输入偏置的梯度
    dev::vector<float> dbr_dev(hidden_size * 3);// 对循环偏置的梯度
    dev::vector<float> dh_dev(batch_size *
                              hidden_size);// 对最后隐藏状态的梯度
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size *
                              3);// 临时缓存梯度（内部结构用）
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size *
                              3);// 临时缓存梯度（内部结构用）

    {
        ScopeTimer t("Train backward:");
        gru::BackwardPass<float> backward(batch_size, input_size, hidden_size,
                                          g_blas_handle);

        backward.Run(time_steps, W_dev.data(), R_dev.data(), bx_dev.data(),
                     br_dev.data(), x_dev.data(), h_dev.data(), v_dev.data(),
                     dh_new_dev.data(), dx_dev.data(), dW_dev.data(), dR_dev.data(),
                     dbx_dev.data(), dbr_dev.data(), dh_dev.data(), dp_dev.data(),
                     dq_dev.data(), nullptr);
    }

    // 将梯度从GPU复制到CPU
    GRUTrainGradients gradients;
    gradients.dx.resize(time_steps * batch_size * input_size);
    gradients.dW.resize(input_size * hidden_size * 3);
    gradients.dR.resize(hidden_size * hidden_size * 3);
    gradients.dbx.resize(hidden_size * 3);
    gradients.dbr.resize(hidden_size * 3);
    gradients.dh.resize(batch_size * hidden_size);

    d2h(gradients.dx.data(), dx_dev.data(), dx_dev.size());
    d2h(gradients.dW.data(), dW_dev.data(), dW_dev.size());
    d2h(gradients.dR.data(), dR_dev.data(), dR_dev.size());
    d2h(gradients.dbx.data(), dbx_dev.data(), dbx_dev.size());
    d2h(gradients.dbr.data(), dbr_dev.data(), dbr_dev.size());
    d2h(gradients.dh.data(), dh_dev.data(), dh_dev.size());

    // 将V从GPU复制到CPU
    d2h(gradients.v, v_dev);

    // 将h从GPU复制到CPU（跳过初始状态，只复制time_steps个时间步）
    const int h_output_size = time_steps * batch_size * hidden_size;
    gradients.h.resize(h_output_size);
    d2h(gradients.h.data(), h_dev.data() + batch_size * hidden_size, h_output_size);

    return gradients;
}

// 比较浮点和量化版本的V中间值
void compareVIntermediateValues(
    const std::vector<float> &v_float,
    const std::vector<float> &v_quant_dequant,
    int time_steps,
    int batch_size,
    int hidden_size,
    const std::string &prefix = "") {
    printf("\n========== %s V Intermediate Values Comparison ==========\n", prefix.c_str());

    const int v_size_per_step = batch_size * hidden_size * 4;// 4个部分：z_out, r_out, g_out, Rh_add_br
    const int v_size_per_part = batch_size * hidden_size;    // 每个部分的大小

    // 验证大小
    if (v_float.size() != static_cast<size_t>(time_steps * v_size_per_step)) {
        printf("[Error] v_float size mismatch: expected %d, got %zu\n",
               time_steps * v_size_per_step, v_float.size());
        return;
    }
    if (v_quant_dequant.size() != static_cast<size_t>(time_steps * v_size_per_step)) {
        printf("[Error] v_quant_dequant size mismatch: expected %d, got %zu\n",
               time_steps * v_size_per_step, v_quant_dequant.size());
        return;
    }

    // V的4个部分名称
    const char *part_names[] = {"z_out", "r_out", "g_out", "Rh_add_br"};

    // 整体比较
    {
        const float mse = computeMSE(v_float, v_quant_dequant);
        const float cos_sim = computeCosineSimilarity(v_float, v_quant_dequant);
        printf("Overall V: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 按部分比较（所有时间步）
    for (int part = 0; part < 4; ++part) {
        std::vector<float> v_float_part(time_steps * v_size_per_part);
        std::vector<float> v_quant_part(time_steps * v_size_per_part);

        for (int t = 0; t < time_steps; ++t) {
            const int t_offset = t * v_size_per_step;
            const int part_offset = part * hidden_size;

            for (int b = 0; b < batch_size; ++b) {
                const int b_offset = b * hidden_size;
                for (int h = 0; h < hidden_size; ++h) {
                    const int src_idx = t_offset + b_offset + part_offset + h;
                    const int dst_idx = t * v_size_per_part + b_offset + h;
                    v_float_part[dst_idx] = v_float[src_idx];
                    v_quant_part[dst_idx] = v_quant_dequant[src_idx];
                }
            }
        }

        const float mse = computeMSE(v_float_part, v_quant_part);
        const float cos_sim = computeCosineSimilarity(v_float_part, v_quant_part);
        printf("%s: MSE = %e, Cosine Similarity = %f\n", part_names[part], mse, cos_sim);
    }

    // 按时间步比较（所有部分）
    printf("\nPer time step comparison:\n");
    for (int t = 0; t < time_steps && t < 10; ++t) {// 只显示前10个时间步
        const int t_offset = t * v_size_per_step;
        std::vector<float> v_float_step(v_size_per_step);
        std::vector<float> v_quant_step(v_size_per_step);

        for (int i = 0; i < v_size_per_step; ++i) {
            v_float_step[i] = v_float[t_offset + i];
            v_quant_step[i] = v_quant_dequant[t_offset + i];
        }

        const float mse = computeMSE(v_float_step, v_quant_step);
        const float cos_sim = computeCosineSimilarity(v_float_step, v_quant_step);
        printf("  Time step %d: MSE = %e, Cosine Similarity = %f\n", t, mse, cos_sim);
    }

    printf("===========================================================\n\n");
}

// 比较浮点和量化版本的h隐藏状态
void compareHValues(
    const std::vector<float> &h_float,
    const std::vector<float> &h_quant_dequant,
    int time_steps,
    int batch_size,
    int hidden_size,
    const std::string &prefix = "") {
    printf("\n========== %s H Hidden States Comparison ==========\n", prefix.c_str());

    const int h_size_per_step = batch_size * hidden_size;// 每个时间步的大小

    // 验证大小
    if (h_float.size() != static_cast<size_t>(time_steps * h_size_per_step)) {
        printf("[Error] h_float size mismatch: expected %d, got %zu\n",
               time_steps * h_size_per_step, h_float.size());
        return;
    }
    if (h_quant_dequant.size() != static_cast<size_t>(time_steps * h_size_per_step)) {
        printf("[Error] h_quant_dequant size mismatch: expected %d, got %zu\n",
               time_steps * h_size_per_step, h_quant_dequant.size());
        return;
    }

    // 整体比较
    {
        const float mse = computeMSE(h_float, h_quant_dequant);
        const float cos_sim = computeCosineSimilarity(h_float, h_quant_dequant);
        printf("Overall H: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 按时间步比较
    printf("\nPer time step comparison:\n");
    for (int t = 0; t < time_steps && t < 10; ++t) {// 只显示前10个时间步
        const int t_offset = t * h_size_per_step;
        std::vector<float> h_float_step(h_size_per_step);
        std::vector<float> h_quant_step(h_size_per_step);

        for (int i = 0; i < h_size_per_step; ++i) {
            h_float_step[i] = h_float[t_offset + i];
            h_quant_step[i] = h_quant_dequant[t_offset + i];
        }

        const float mse = computeMSE(h_float_step, h_quant_step);
        const float cos_sim = computeCosineSimilarity(h_float_step, h_quant_step);
        printf("  Time step %d: MSE = %e, Cosine Similarity = %f\n", t, mse, cos_sim);
    }

    // 按批次比较（所有时间步）
    printf("\nPer batch comparison:\n");
    for (int b = 0; b < batch_size && b < 5; ++b) {// 只显示前5个批次
        std::vector<float> h_float_batch(time_steps * hidden_size);
        std::vector<float> h_quant_batch(time_steps * hidden_size);

        for (int t = 0; t < time_steps; ++t) {
            const int t_offset = t * h_size_per_step;
            const int b_offset = b * hidden_size;

            for (int h = 0; h < hidden_size; ++h) {
                const int src_idx = t_offset + b_offset + h;
                const int dst_idx = t * hidden_size + h;
                h_float_batch[dst_idx] = h_float[src_idx];
                h_quant_batch[dst_idx] = h_quant_dequant[src_idx];
            }
        }

        const float mse = computeMSE(h_float_batch, h_quant_batch);
        const float cos_sim = computeCosineSimilarity(h_float_batch, h_quant_batch);
        printf("  Batch %d: MSE = %e, Cosine Similarity = %f\n", b, mse, cos_sim);
    }

    printf("===========================================================\n\n");
}

// 比较两个GRU训练梯度的差异
void compareGRUTrainGradients(const GRUTrainGradients &gradients_float,
                              const GRUTrainGradients &gradients_quant,
                              const std::string &prefix = "") {
    printf("\n========== %s GRU Train Gradients Comparison ==========\n", prefix.c_str());

    // 比较 dx
    {
        const float mse = computeMSE(gradients_float.dx, gradients_quant.dx);
        const float cos_sim = computeCosineSimilarity(gradients_float.dx, gradients_quant.dx);
        printf("dx: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dW
    {
        const float mse = computeMSE(gradients_float.dW, gradients_quant.dW);
        const float cos_sim = computeCosineSimilarity(gradients_float.dW, gradients_quant.dW);
        printf("dW: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dR
    {
        const float mse = computeMSE(gradients_float.dR, gradients_quant.dR);
        const float cos_sim = computeCosineSimilarity(gradients_float.dR, gradients_quant.dR);
        printf("dR: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dbx
    {
        const float mse = computeMSE(gradients_float.dbx, gradients_quant.dbx);
        const float cos_sim = computeCosineSimilarity(gradients_float.dbx, gradients_quant.dbx);
        printf("dbx: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dbr
    {
        const float mse = computeMSE(gradients_float.dbr, gradients_quant.dbr);
        const float cos_sim = computeCosineSimilarity(gradients_float.dbr, gradients_quant.dbr);
        printf("dbr: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    // 比较 dh
    {
        const float mse = computeMSE(gradients_float.dh, gradients_quant.dh);
        const float cos_sim = computeCosineSimilarity(gradients_float.dh, gradients_quant.dh);
        printf("dh: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
    }

    printf("===========================================================\n\n");
}

void checkHQuantizationWithCosine(
    const std::vector<float> &h_inference,// 浮点 h, size = (time_steps+1) *
    // batch_size * hidden_size
    const std::vector<int8_t> &h_quant_inference,// 量化 h, size 同上
    int time_steps, int batch_size, int hidden_size,
    const GRUQuantitativeParameters &scaleParam) {
    const int size_per_step = batch_size * hidden_size;

    // 验证输入数据大小
    if (h_inference.size() !=
        static_cast<size_t>((time_steps + 1) * size_per_step)) {
        printf("[Error] h_inference size mismatch: expected %d, got %zu\n",
               (time_steps + 1) * size_per_step, h_inference.size());
        return;
    }
    if (h_quant_inference.size() !=
        static_cast<size_t>((time_steps + 1) * size_per_step)) {
        printf(
            "[Error] h_quant_inference size mismatch: expected %d, got %zu\n",
            (time_steps + 1) * size_per_step, h_quant_inference.size());
        return;
    }

    printf(
        "checkHQuantizationWithCosine: time_steps=%d, batch_size=%d, "
        "hidden_size=%d\n",
        time_steps, batch_size, hidden_size);
    printf("  exp2_inv_h_=%d, zp_h_=%d\n", scaleParam.exp2_inv_h_,
           scaleParam.zp_h_);

    // 检查前几个数据点的值
    printf("  Sample data check:\n");
    printf("    h_inference size: %zu, expected: %d\n", h_inference.size(),
           (time_steps + 1) * size_per_step);
    printf("    h_quant_inference size: %zu, expected: %d\n",
           h_quant_inference.size(), (time_steps + 1) * size_per_step);

    // 检查初始状态（t=0）的数据
    printf("    h_inference[t=0, first 5]: ");
    for (int i = 0; i < 5 && i < size_per_step; ++i) {
        printf("%f ", h_inference[i]);
    }
    printf("\n");
    printf("    h_quant_inference[t=0, first 5]: ");
    for (int i = 0; i < 5 && i < size_per_step; ++i) {
        printf("%d ", static_cast<int>(h_quant_inference[i]));
    }
    printf("\n");

    // 检查第一个时间步（t=1）的数据
    const int t1_offset = size_per_step;
    printf("    h_inference[t=1, first 5]: ");
    for (int i = 0; i < 5 && i < size_per_step; ++i) {
        printf("%f ", h_inference[t1_offset + i]);
    }
    printf("\n");
    printf("    h_quant_inference[t=1, first 5]: ");
    for (int i = 0; i < 5 && i < size_per_step; ++i) {
        printf("%d ", static_cast<int>(h_quant_inference[t1_offset + i]));
    }
    printf("\n");

    std::vector<float> h_float_step(size_per_step);
    std::vector<float> h_quant_step(size_per_step);

    for (int t = 1; t <= time_steps; ++t) {
        // ForwardPass 存储 h 的方式：h + t * (batch_size * hidden_size)
        // 每个时间步内部：按 [batch0_h0, batch0_h1, ..., batch0_hH-1,
        // batch1_h0, ..., batchN-1_hH-1] 顺序 即：t * (N*H) + n * H + h

        const size_t t_offset = static_cast<size_t>(t) * size_per_step;

        // 直接拷贝和反量化
        for (int idx = 0; idx < size_per_step; ++idx) {
            h_float_step[idx] = h_inference[t_offset + idx];

            const int8_t quant_val = h_quant_inference[t_offset + idx];
            h_quant_step[idx] = dequantize<int8_t>(
                quant_val, scaleParam.exp2_inv_h_, scaleParam.zp_h_);
        }
        const float mse = computeMSE(h_float_step, h_quant_step);
        const float cos_sim =
            computeCosineSimilarity(h_float_step, h_quant_step);

        printf("Time step %d: mse = %f, cosine_sim = %f\n", t, mse, cos_sim);
    }
}

void checkQuantize(
    const std::vector<float> &W,
    const std::vector<float> &R,
    const std::vector<float> &bx,
    const std::vector<float> &br,
    const std::vector<float> &x,
    const GRUQuantitativeParameters &quant_parms,
    int time_steps,
    int batch_size,
    int input_size,
    int hidden_size) {
    // ========== 验证CPU和GPU量化结果一致性 ==========
    printf("\n========== 验证CPU和GPU量化结果一致性 ==========\n");

    const int channel_size = hidden_size * 3;
    const std::size_t x_size = time_steps * batch_size * input_size;

    // CPU版本量化结果
    std::vector<int8_t> W_quant_cpu(input_size * hidden_size * 3);
    std::vector<int8_t> R_quant_cpu(hidden_size * hidden_size * 3);
    std::vector<int32_t> bx_quant_cpu(hidden_size * 3);
    std::vector<int32_t> br_quant_cpu(hidden_size * 3);
    std::vector<int8_t> x_quant_cpu(x_size);

    {
        ScopeTimer t("CPU量化:");
        quantificationPerChannel(W.data(), W_quant_cpu.data(), input_size, channel_size,
                                 quant_parms.exp2_inv_W_);
        quantificationPerChannel(R.data(), R_quant_cpu.data(), hidden_size, channel_size,
                                 quant_parms.exp2_inv_R_);
        quantificationPerChannel(bx.data(), bx_quant_cpu.data(), 1, channel_size,
                                 quant_parms.exp2_inv_bx_);
        quantificationPerChannel(br.data(), br_quant_cpu.data(), 1, channel_size,
                                 quant_parms.exp2_inv_br_);
        quantification(x.data(), x_quant_cpu.data(), x_size, quant_parms.exp2_inv_x_,
                       quant_parms.zp_x_);
    }

    // GPU版本量化结果
    dev::vector<float> W_dev(W);
    dev::vector<float> R_dev(R);
    dev::vector<float> bx_dev(bx);
    dev::vector<float> br_dev(br);
    dev::vector<float> x_dev(x);

    dev::vector<int8_t> W_quant_gpu(input_size * hidden_size * 3);
    dev::vector<int8_t> R_quant_gpu(hidden_size * hidden_size * 3);
    dev::vector<int32_t> bx_quant_gpu(hidden_size * 3);
    dev::vector<int32_t> br_quant_gpu(hidden_size * 3);
    dev::vector<int8_t> x_quant_gpu(x_size);

    dev::vector<int32_t> exp2_inv_W_dev(quant_parms.exp2_inv_W_);
    dev::vector<int32_t> exp2_inv_R_dev(quant_parms.exp2_inv_R_);
    dev::vector<int32_t> exp2_inv_bx_dev(quant_parms.exp2_inv_bx_);
    dev::vector<int32_t> exp2_inv_br_dev(quant_parms.exp2_inv_br_);

    {
        ScopeTimer t("GPU量化:");
        dev::quantificationPerChannel(W_dev.data(), W_quant_gpu.data(), input_size, channel_size,
                                      exp2_inv_W_dev);
        dev::quantificationPerChannel(R_dev.data(), R_quant_gpu.data(), hidden_size, channel_size,
                                      exp2_inv_R_dev);
        dev::quantificationPerChannel(bx_dev.data(), bx_quant_gpu.data(), 1, channel_size,
                                      exp2_inv_bx_dev);
        dev::quantificationPerChannel(br_dev.data(), br_quant_gpu.data(), 1, channel_size,
                                      exp2_inv_br_dev);
        dev::quantification(x_dev.data(), x_quant_gpu.data(), x_size, quant_parms.exp2_inv_x_,
                            quant_parms.zp_x_);
    }

    // 将GPU结果复制到CPU进行比较
    std::vector<int8_t> W_quant_gpu_host = d2h(W_quant_gpu);
    std::vector<int8_t> R_quant_gpu_host = d2h(R_quant_gpu);
    std::vector<int32_t> bx_quant_gpu_host = d2h(bx_quant_gpu);
    std::vector<int32_t> br_quant_gpu_host = d2h(br_quant_gpu);
    std::vector<int8_t> x_quant_gpu_host = d2h(x_quant_gpu);

    // 比较结果
    int mismatch_count = 0;
    const int max_show_mismatches = 10;

    // 比较W
    printf("\n比较W量化结果 (大小: %zu):\n", W_quant_cpu.size());
    for (size_t i = 0; i < W_quant_cpu.size(); ++i) {
        if (W_quant_cpu[i] != W_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  W[%zu]: CPU=%d, GPU=%d\n", i,
                       static_cast<int>(W_quant_cpu[i]),
                       static_cast<int>(W_quant_gpu_host[i]));
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ W量化结果完全一致\n");
    } else {
        printf("  ✗ W量化结果有 %d 个不匹配\n", mismatch_count);
    }

    // 比较R
    mismatch_count = 0;
    printf("\n比较R量化结果 (大小: %zu):\n", R_quant_cpu.size());
    for (size_t i = 0; i < R_quant_cpu.size(); ++i) {
        if (R_quant_cpu[i] != R_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  R[%zu]: CPU=%d, GPU=%d\n", i,
                       static_cast<int>(R_quant_cpu[i]),
                       static_cast<int>(R_quant_gpu_host[i]));
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ R量化结果完全一致\n");
    } else {
        printf("  ✗ R量化结果有 %d 个不匹配\n", mismatch_count);
    }

    // 比较bx
    mismatch_count = 0;
    printf("\n比较bx量化结果 (大小: %zu):\n", bx_quant_cpu.size());
    for (size_t i = 0; i < bx_quant_cpu.size(); ++i) {
        if (bx_quant_cpu[i] != bx_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  bx[%zu]: CPU=%d, GPU=%d\n", i,
                       bx_quant_cpu[i], bx_quant_gpu_host[i]);
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ bx量化结果完全一致\n");
    } else {
        printf("  ✗ bx量化结果有 %d 个不匹配\n", mismatch_count);
    }

    // 比较br
    mismatch_count = 0;
    printf("\n比较br量化结果 (大小: %zu):\n", br_quant_cpu.size());
    for (size_t i = 0; i < br_quant_cpu.size(); ++i) {
        if (br_quant_cpu[i] != br_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  br[%zu]: CPU=%d, GPU=%d\n", i,
                       br_quant_cpu[i], br_quant_gpu_host[i]);
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ br量化结果完全一致\n");
    } else {
        printf("  ✗ br量化结果有 %d 个不匹配\n", mismatch_count);
    }

    // 比较x
    mismatch_count = 0;
    printf("\n比较x量化结果 (大小: %zu):\n", x_quant_cpu.size());
    for (size_t i = 0; i < x_quant_cpu.size(); ++i) {
        if (x_quant_cpu[i] != x_quant_gpu_host[i]) {
            if (mismatch_count < max_show_mismatches) {
                printf("  x[%zu]: CPU=%d, GPU=%d\n", i,
                       static_cast<int>(x_quant_cpu[i]),
                       static_cast<int>(x_quant_gpu_host[i]));
            }
            mismatch_count++;
        }
    }
    if (mismatch_count == 0) {
        printf("  ✓ x量化结果完全一致\n");
    } else {
        printf("  ✗ x量化结果有 %d 个不匹配\n", mismatch_count);
    }

    printf("\n===========================================================\n\n");
}

int main() {
    srand(time(0));

    init_gru_cublas();// 使用初始化函数

    // Weights.
    std::vector<float> W(HIDDEN_DIMS * 3 * INPUT_DIMS); // 对应W_z/W_r/W_h的合并
    std::vector<float> R(HIDDEN_DIMS * 3 * HIDDEN_DIMS);// 对应R_z/R_r/R_h的合并
    std::vector<float> bx(HIDDEN_DIMS * 3);             // 对应b_z/b_r/b_h的合并. bx 负责给 "输入 x_t
    // 到门控的线性变换" 加偏置
    std::vector<float> br(HIDDEN_DIMS * 3);// br: 3H(部分实现中偏置分输出\隐藏层. br 负责给"隐藏状态
    // h_{t-1} 到门控的线性变换" 加偏置

    // Input.
    std::vector<float> x(INPUT_DIMS * BATCH_SIZE * SEQUENCE_LEN);

    // Gradients from upstream layers.
    std::vector<float> dh(HIDDEN_DIMS * BATCH_SIZE * (SEQUENCE_LEN + 1));

    // W: 输入权重矩阵，使用 Xavier/Glorot 均匀初始化
    // 范围: U(-k, k)，其中 k = sqrt(6 / (input_size + hidden_size * 3))
    // 这确保前向和反向传播的方差保持稳定
    fillVectorWithNormalDistribution(W, -1, 1);
    for (int i = 0; i < W.size(); ++i) {
        W[i] = W[i] * 0.1f;
        W[i] *= 0.01f;
    }

    fillVectorWithNormalDistribution(R, -1, 1);
    for (int i = 0; i < R.size(); ++i) {
        R[i] = R[i] * 0.5f;
        R[i] *= 0.01f;
    }

    fillVectorWithNormalDistribution(bx, -1, 1);
    for (int i = 0; i < bx.size(); ++i) {
        bx[i] = bx[i] * 0.15f;
    }
    fillVectorWithNormalDistribution(br, -1, 1);
    for (int i = 0; i < br.size(); ++i) {
        br[i] = br[i] * 0.15f;
    }

    fillVectorWithNormalDistribution(x, -1, 1);
    for (int i = 0; i < x.size(); ++i) {
        x[i] = x[i] * 0.8f;
        x[i] += 0.1f;
    }

    fillVectorWithNormalDistribution(dh, -1, 1);
    for (int i = 0; i < dh.size(); ++i) {
        dh[i] = dh[i] * 0.5f;
    }

    const int time_steps = SEQUENCE_LEN;
    const int batch_size = BATCH_SIZE;
    const int input_size = INPUT_DIMS;
    const int hidden_size = HIDDEN_DIMS;

    // 效验得到固定量化参数
    GRUQuantitativeParameters quant_parms;
    calibrateGruScales(false, time_steps, batch_size, input_size, hidden_size,
                       W, R, bx, br, x,
                       g_blas_handle, quant_parms);

    // Quant
    std::vector<int8_t> W_quant(HIDDEN_DIMS * 3 * INPUT_DIMS); // 对应W_z/W_r/W_h的合并
    std::vector<int8_t> R_quant(HIDDEN_DIMS * 3 * HIDDEN_DIMS);// 对应R_z/R_r/R_h的合并
    std::vector<int32_t> bx_quant(HIDDEN_DIMS * 3);            // 对应b_z/b_r/b_h的合并. bx 负责给
    // “输入 x_t 到门控的线性变换” 加偏置
    std::vector<int32_t> br_quant(HIDDEN_DIMS * 3);// br: 3H(部分实现中偏置分输出\隐藏层. br
    // 负责给“隐藏状态 h_{t-1} 到门控的线性变换” 加偏置
    std::vector<int8_t> x_quant(INPUT_DIMS * BATCH_SIZE * SEQUENCE_LEN);
    std::vector<int8_t> dh_new_quant(HIDDEN_DIMS * BATCH_SIZE * (SEQUENCE_LEN + 1));

    // 使用固定量化参数将输入量化
    GruQuantInit(time_steps, batch_size, input_size, hidden_size, W.data(),
                 R.data(), bx.data(), br.data(), x.data(), dh.data(),
                 W_quant.data(), R_quant.data(), bx_quant.data(),
                 br_quant.data(), x_quant.data(), dh_new_quant.data(),
                 quant_parms);

    Quantized_unit_testing<int8_t> quantized_unit_testing(
        W.data(), R.data(), bx.data(), br.data(), x.data(), dh.data(),
        W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data(),
        x_quant.data(), dh_new_quant.data(), hidden_size, input_size,
        batch_size, time_steps, g_blas_handle, quant_parms);
    quantized_unit_testing.printGRUQuantitativeParameters();
    //    quantized_unit_testing.checkQuantParameters();

    std::vector<int8_t> h_quant_inference(hidden_size * batch_size * (time_steps + 1));
    // 运行量化GRU得到量化结果2
    GruInferenceQuant(time_steps, batch_size, input_size, hidden_size,
                      W_quant.data(), R_quant.data(), bx_quant.data(),
                      br_quant.data(), x.data(), quant_parms,
                      h_quant_inference.data());

    // 运行浮点GRU得到结果1
    std::vector<float> h_inference(hidden_size * batch_size * (time_steps + 1));
    GruInference(time_steps,
                 batch_size,
                 input_size,
                 hidden_size,
                 W.data(),
                 R.data(),
                 bx.data(),
                 br.data(),
                 x.data(),
                 h_inference.data());

    printf("cudaError(GruInference finish): %s\n",
           cudaGetErrorString(cudaGetLastError()));

    if (true) {
        // Test
        std::vector<float> h_inference_tmp(
            h_inference.data(), h_inference.data() + h_inference.size());
        std::vector<int8_t> h_quant_inference_tmp(
            h_quant_inference.data(),
            h_quant_inference.data() + h_quant_inference.size());

        checkHQuantizationWithCosine(h_inference_tmp, h_quant_inference_tmp,
                                     time_steps, batch_size, hidden_size,
                                     quant_parms);
    }

    printf("cudaError(GruInferenceQuant finish): %s\n",
           cudaGetErrorString(cudaGetLastError()));

    // 运行浮点训练
    printf("\n========== Running Float GRU Training ==========\n");
    GRUTrainGradients gradients_float = GruTrain(time_steps, batch_size, input_size, hidden_size,
                                                 W, R, bx, br, x, dh);

    printf("cudaError(GruTrain finish): %s\n",
           cudaGetErrorString(cudaGetLastError()));

    // 运行量化训练
    printf("\n========== Running Quantized GRU Training ==========\n");
    GRUTrainGradients gradients_quant = GruTrainQuant<int8_t>(time_steps, batch_size, input_size, hidden_size,
                                                              W, R, bx, br, x, dh);

    printf("cudaError(GruTrainQuant finish): %s\n",
           cudaGetErrorString(cudaGetLastError()));

    // 比较V中间值
    compareVIntermediateValues(gradients_float.v, gradients_quant.v, time_steps, batch_size, hidden_size,
                               "Float vs Quantized");

    // 比较h隐藏状态
    compareHValues(gradients_float.h, gradients_quant.h, time_steps, batch_size, hidden_size,
                   "Float vs Quantized");

    // 比较两个训练的输出
    compareGRUTrainGradients(gradients_float, gradients_quant, "Float vs Quantized");

    // 验证CPU和GPU量化结果一致性
    checkQuantize(W, R, bx, br, x, quant_parms, time_steps, batch_size, input_size, hidden_size);

    cublasDestroy(g_blas_handle);

    return 0;
}
