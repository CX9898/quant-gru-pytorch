#include "gru.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "devVector.h"
#include "gru_quant.h"
#include "quantized_unit_testing.h"

constexpr int BATCH_SIZE = 64;     // 批大小
constexpr int SEQUENCE_LEN = 500;  // 序列长度(T), 每个样本有T个时间步
constexpr int HIDDEN_DIMS = 256;   // 隐藏层维度(H), h_t的维度
constexpr int INPUT_DIMS = 256;    // 输入维度(I), x_t的维度

cublasHandle_t g_blas_handle;  // 改为非static以便在wrapper中访问

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
    int8_t *h_quant_out  // (time_steps + 1) * batch_size * hidden_size
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
                                    3);  // 用于存放W * x的中间结果
    dev::vector<int32_t> tmp_Rh_dev(batch_size * hidden_size *
                                    3);  // 用于存放R * h的中间结果

    {
        gru::ForwardPassQuant<int8_t> forward = gru::ForwardPassQuant<int8_t>(
            false,  // training
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
                                  3);  // 用于存放W * x的中间结果
    dev::vector<float> tmp_Rh_dev(batch_size * hidden_size *
                                  3);  // 用于存放R * h的中间结果

    h_dev.zero();  // h初始化为0

    {
        ScopeTimer t("Inference:");

        gru::ForwardPass<float> forward = gru::ForwardPass<float>(
            false,  // training
            batch_size, input_size, hidden_size, g_blas_handle);

        forward.Run(time_steps, W_dev.data(), R_dev.data(), bx_dev.data(),
                    br_dev.data(), x_dev.data(), h_dev.data(), nullptr,
                    tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);
    }

    d2h(h, h_dev.data(), h_dev.size());
}

void GruTrain(const int time_steps,
              const int batch_size,
              const int input_size,
              const int hidden_size,
              const float *W,  // 输入到隐藏层的权重矩阵. [input_size,
                                               // hidden_size * 3] 对应三个门
              const float *R,   // 隐藏层到隐藏层的循环权重矩阵
              const float *bx,  // 输入偏置项（input bias），来自输入路径
              const float *br,  // 循环偏置项（recurrent bias），来自循环路径
              const float *x,   // 输入序列张量
              const float *dh_new,  // 来自上层网络或损失函数的反向梯度.
                                               // [hidden_size, batch_size, time_steps]
              bool enable_quantitative = false,  // 是否启用量化推理模式
              bool use_int16 = false             // 控制量化精度位宽
) {

    // Copy weights over to GPU.
    dev::vector<float> W_dev(W, hidden_size * 3 * input_size);
    dev::vector<float> R_dev(R, hidden_size * 3 * hidden_size);
    dev::vector<float> bx_dev(bx, hidden_size * 3);
    dev::vector<float> br_dev(br, hidden_size * 3);
    dev::vector<float> x_dev(x, time_steps * input_size * batch_size);

    dev::vector<float> dh_new_dev(dh_new, HIDDEN_DIMS * BATCH_SIZE * (SEQUENCE_LEN + 1));

    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
    dev::vector<float> tmp_Rh_dev(batch_size * hidden_size * 3);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    h_dev.zero();

    {
        ScopeTimer t("Train forward:");
        gru::ForwardPass<float> forward = gru::ForwardPass<float>(
            true,  // training
            batch_size, input_size, hidden_size, g_blas_handle);

        forward.Run(time_steps, W_dev.data(), R_dev.data(), bx_dev.data(),
                    br_dev.data(), x_dev.data(), h_dev.data(), v_dev.data(),
                    tmp_Wx_dev.data(), tmp_Rh_dev.data(), 0.0f, nullptr);
    }

    dev::vector<float> dx_dev(time_steps * batch_size *
                              input_size);  // 输入序列梯度
    dev::vector<float> dW_dev(input_size * hidden_size *
                              3);  // 对输入权重的梯度
    dev::vector<float> dR_dev(hidden_size * hidden_size *
                              3);                 // 对循环权重的梯度
    dev::vector<float> dbx_dev(hidden_size * 3);  // 对输入偏置的梯度
    dev::vector<float> dbr_dev(hidden_size * 3);  // 对循环偏置的梯度
    dev::vector<float> dh_dev(batch_size *
                              hidden_size);  // 对最后隐藏状态的梯度
    dev::vector<float> dp_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）
    dev::vector<float> dq_dev(time_steps * batch_size * hidden_size *
                              3);  // 临时缓存梯度（内部结构用）

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
}

void checkHQuantizationWithCosine(
    const std::vector<float> &h_inference,  // 浮点 h, size = (time_steps+1) *
    // batch_size * hidden_size
    const std::vector<int8_t> &h_quant_inference,  // 量化 h, size 同上
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

int main() {
    srand(time(0));

    init_gru_cublas();  // 使用初始化函数

    // Weights.
    std::vector<float> W(HIDDEN_DIMS * 3 * INPUT_DIMS);   // 对应W_z/W_r/W_h的合并
    std::vector<float> R(HIDDEN_DIMS * 3 * HIDDEN_DIMS);  // 对应R_z/R_r/R_h的合并
    std::vector<float> bx(HIDDEN_DIMS * 3);  // 对应b_z/b_r/b_h的合并. bx 负责给 "输入 x_t
    // 到门控的线性变换" 加偏置
    std::vector<float> br(HIDDEN_DIMS * 3);  // br: 3H(部分实现中偏置分输出\隐藏层. br 负责给"隐藏状态
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
                       W.data(), R.data(), bx.data(), br.data(), x.data(),
                       g_blas_handle, quant_parms);

    // Quant
    std::vector<int8_t> W_quant(HIDDEN_DIMS * 3 * INPUT_DIMS);   // 对应W_z/W_r/W_h的合并
    std::vector<int8_t> R_quant(HIDDEN_DIMS * 3 * HIDDEN_DIMS);  // 对应R_z/R_r/R_h的合并
    std::vector<int32_t> bx_quant(HIDDEN_DIMS * 3);  // 对应b_z/b_r/b_h的合并. bx 负责给
    // “输入 x_t 到门控的线性变换” 加偏置
    std::vector<int32_t> br_quant(HIDDEN_DIMS * 3);  // br: 3H(部分实现中偏置分输出\隐藏层. br
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

    GruTrain(time_steps,
             batch_size,
             input_size,
             hidden_size, W.data(), R.data(), bx.data(), br.data(), x.data(), dh.data(), false, false);

    printf("cudaError(GruTrain finish): %s\n",
           cudaGetErrorString(cudaGetLastError()));

    cublasDestroy(g_blas_handle);

    return 0;
}
