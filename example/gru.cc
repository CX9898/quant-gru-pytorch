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

constexpr int BATCH_SIZE = 64;  // 批大小
constexpr int SEQUENCE_LEN = 50;// 序列长度(T), 每个样本有T个时间步
constexpr int HIDDEN_DIMS = 256;// 隐藏层维度(H), h_t的维度
constexpr int INPUT_DIMS = 256; // 输入维度(I), x_t的维度

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
    const int hidden_size, const std::vector<float> &W, const std::vector<float> &R, const std::vector<float> &bx,
    const std::vector<float> &br, const std::vector<float> &x,
    const GRUQuantitativeParameters &quant_parms,
    std::vector<float> &h_out// (time_steps + 1) * batch_size * hidden_size
) {
    dev::vector<float> x_dev(x);

    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);

    dev::vector<QuantT> W_quant_dev(W.size());
    dev::vector<QuantT> R_quant_dev(R.size());
    dev::vector<int32_t> bx_quant_dev(bx.size());
    dev::vector<int32_t> br_quant_dev(br.size());
    {
        dev::vector<float> W_dev(W);
        dev::vector<float> R_dev(R);
        dev::vector<float> bx_dev(bx);
        dev::vector<float> br_dev(br);
        quantitativeWeight<QuantT>(input_size, hidden_size,
                                   W_dev.data(), R_dev.data(), bx_dev.data(), br_dev.data(), quant_parms,
                                   W_quant_dev.data(), R_quant_dev.data(), bx_quant_dev.data(), br_quant_dev.data());
    }
    {
        bool is_int16 = std::is_same_v<QuantT, int16_t> ? true : false;
        initialize_quantization_lut(quant_parms, is_int16);
    }
    {
        ScopeTimer t("GruInferenceQuant:");
        quantGRUForward<QuantT>(false, time_steps, batch_size, input_size, hidden_size,
                                W_quant_dev.data(), R_quant_dev.data(), bx_quant_dev.data(), br_quant_dev.data(),
                                x_dev.data(), nullptr, quant_parms, g_blas_handle, h_dev.data(), nullptr);
    }
    d2h(h_out, h_dev);
}

void GruInference(const int time_steps,
                  const int batch_size,
                  const int input_size,
                  const int hidden_size,
                  const std::vector<float> &W,
                  const std::vector<float> &R,
                  const std::vector<float> &bx,
                  const std::vector<float> &br,
                  const std::vector<float> &x,
                  std::vector<float> &h) {
    dev::vector<float> W_dev(W);
    dev::vector<float> R_dev(R);
    dev::vector<float> bx_dev(bx);
    dev::vector<float> br_dev(br);
    dev::vector<float> x_dev(x);
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);

    // 调用hasteGRUForward进行推理
    ScopeTimer t("GruInference (float):");
    hasteGRUForward(false,
                    time_steps,
                    batch_size,
                    input_size,
                    hidden_size,
                    W_dev.data(),
                    R_dev.data(),
                    bx_dev.data(),
                    br_dev.data(),
                    x_dev.data(),
                    nullptr,// h0设为nullptr
                    g_blas_handle,
                    h_dev.data(),
                    nullptr// reserve设为nullptr
    );
    d2h(h, h_dev);
}


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
    {
        bool is_int16 = std::is_same_v<QuantT, int16_t> ? true : false;
        initialize_quantization_lut(quant_parms, is_int16);
    }

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
    // 1. 使用dev::vector拷贝数据到GPU
    dev::vector<float> W_dev(W);
    dev::vector<float> R_dev(R);
    dev::vector<float> bx_dev(bx);
    dev::vector<float> br_dev(br);
    dev::vector<float> x_dev(x);
    dev::vector<float> dh_new_dev(dh_new);

    // 2. 创建必要的dev::vector缓存
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    // 3. 前向传播: 调用hasteGRUForward
    {
        ScopeTimer t("hasteGRUForward (train):");
        hasteGRUForward(
            true,// training模式
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            W_dev.data(),
            R_dev.data(),
            bx_dev.data(),
            br_dev.data(),
            x_dev.data(),
            nullptr,// h0
            g_blas_handle,
            h_dev.data(),
            v_dev.data()// reserve
        );
    }

    // 4. 创建梯度参数, 用dev::vector分配空间
    dev::vector<float> dx_dev(time_steps * batch_size * input_size);
    dev::vector<float> dW_dev(input_size * hidden_size * 3);
    dev::vector<float> dR_dev(hidden_size * hidden_size * 3);
    dev::vector<float> dbx_dev(hidden_size * 3);
    dev::vector<float> dbr_dev(hidden_size * 3);
    dev::vector<float> dh_dev(batch_size * hidden_size);

    // 5. 反向传播: 调用hasteGRUbackward
    {
        ScopeTimer t("hasteGRUbackward:");
        hasteGRUBackward(
            time_steps,
            batch_size,
            input_size,
            hidden_size,
            W_dev.data(),
            R_dev.data(),
            bx_dev.data(),
            br_dev.data(),
            x_dev.data(),
            dh_new_dev.data(),
            h_dev.data(),
            v_dev.data(),
            g_blas_handle,
            dx_dev.data(),
            dW_dev.data(),
            dR_dev.data(),
            dbx_dev.data(),
            dbr_dev.data(),
            dh_dev.data());
    }

    // 6. 拷贝结果回CPU
    GRUTrainGradients gradients;
    gradients.dx.resize(dx_dev.size());
    gradients.dW.resize(dW_dev.size());
    gradients.dR.resize(dR_dev.size());
    gradients.dbx.resize(dbx_dev.size());
    gradients.dbr.resize(dbr_dev.size());
    gradients.dh.resize(dh_dev.size());

    d2h(gradients.dx.data(), dx_dev.data(), dx_dev.size());
    d2h(gradients.dW.data(), dW_dev.data(), dW_dev.size());
    d2h(gradients.dR.data(), dR_dev.data(), dR_dev.size());
    d2h(gradients.dbx.data(), dbx_dev.data(), dbx_dev.size());
    d2h(gradients.dbr.data(), dbr_dev.data(), dbr_dev.size());
    d2h(gradients.dh.data(), dh_dev.data(), dh_dev.size());

    // h需要跳过初始状态h0，只返回time_steps个h
    const int h_output_size = time_steps * batch_size * hidden_size;
    gradients.h.resize(h_output_size);
    d2h(gradients.h.data(), h_dev.data() + batch_size * hidden_size, h_output_size);

    return gradients;
}


int main() {
    srand(time(0));

    init_gru_cublas();// 使用初始化函数

    // Weights.
    std::vector<float> W(INPUT_DIMS * HIDDEN_DIMS * 3); // 对应W_z/W_r/W_h的合并
    std::vector<float> R(HIDDEN_DIMS * HIDDEN_DIMS * 3);// 对应R_z/R_r/R_h的合并
    std::vector<float> bx(HIDDEN_DIMS * 3);             // 对应b_z/b_r/b_h的合并. bx 负责给 "输入 x_t
    // 到门控的线性变换" 加偏置
    std::vector<float> br(HIDDEN_DIMS * 3);// br: 3H(部分实现中偏置分输出\隐藏层. br 负责给"隐藏状态
    // h_{t-1} 到门控的线性变换" 加偏置

    // Input.
    std::vector<float> x(SEQUENCE_LEN * BATCH_SIZE * INPUT_DIMS);

    // Gradients from upstream layers.
    std::vector<float> dh((SEQUENCE_LEN + 1) * BATCH_SIZE * HIDDEN_DIMS);

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
    std::vector<int8_t> W_quant(W.size());   // 对应W_z/W_r/W_h的合并
    std::vector<int8_t> R_quant(R.size());   // 对应R_z/R_r/R_h的合并
    std::vector<int32_t> bx_quant(bx.size());// 对应b_z/b_r/b_h的合并. bx 负责给
    // “输入 x_t 到门控的线性变换” 加偏置
    std::vector<int32_t> br_quant(br.size());// br: 3H(部分实现中偏置分输出\隐藏层. br
    // 负责给“隐藏状态 h_{t-1} 到门控的线性变换” 加偏置
    std::vector<int8_t> x_quant(x.size());

    // 使用固定量化参数将输入量化
    GruQuantInit(time_steps, batch_size, input_size, hidden_size, W.data(),
                 R.data(), bx.data(), br.data(), x.data(),
                 W_quant.data(), R_quant.data(), bx_quant.data(),
                 br_quant.data(), x_quant.data(),
                 quant_parms);

    Quantized_unit_testing<int8_t> quantized_unit_testing(
        W.data(), R.data(), bx.data(), br.data(), x.data(), dh.data(),
        W_quant.data(), R_quant.data(), bx_quant.data(), br_quant.data(),
        x_quant.data(), hidden_size, input_size,
        batch_size, time_steps, g_blas_handle, quant_parms);
    quantized_unit_testing.printGRUQuantitativeParameters();
    //    quantized_unit_testing.checkQuantParameters();

    std::vector<float> h_dequant_int8_inference((time_steps + 1) * batch_size * hidden_size);
    // 运行量化GRU得到量化结果2
    GruInferenceQuant<int8_t>(time_steps, batch_size, input_size, hidden_size,
                              W, R, bx, br, x, quant_parms,
                              h_dequant_int8_inference);

    printf("cudaError(GruInferenceQuant finish): %s\n",
           cudaGetErrorString(cudaGetLastError()));

    // 运行浮点GRU得到结果1
    std::vector<float> h_inference((time_steps + 1) * batch_size * hidden_size);
    GruInference(time_steps,
                 batch_size,
                 input_size,
                 hidden_size,
                 W,
                 R,
                 bx,
                 br,
                 x,
                 h_inference);

    printf("cudaError(GruInference finish): %s\n",
           cudaGetErrorString(cudaGetLastError()));

    compareHValues(h_inference, h_dequant_int8_inference, time_steps, batch_size, hidden_size,
                   "Inference: Float vs Quantized");

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
    checkQuantificationHostAndDevice(W, R, bx, br, x, quant_parms, time_steps, batch_size, input_size, hidden_size);

    cublasDestroy(g_blas_handle);

    return 0;
}
