#include "gru.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "checkData.hpp"
#include "devVector.h"
#include "gru_interface.hpp"
#include "histogram_collector.hpp"
#include "quantized_unit_testing.cuh"

// ==================== 矩阵转置工具函数 ====================

// 使用 cuBLAS 进行 2D 矩阵转置: [rows, cols] -> [cols, rows]
// A: 输入矩阵 [rows x cols]
// A_t: 输出矩阵 [cols x rows]
void transpose2D(cublasHandle_t handle, const float *A, float *A_t, int rows, int cols) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cublasSgeam: C = alpha * op(A) + beta * op(B)
    // 将 A [cols, rows] 转置为 A_t [rows, cols]
    // 
    // 输入 A: 原始矩阵形状 [cols, rows]（列优先存储，lda = cols）
    // 输出 A_t: 转置后矩阵形状 [rows, cols]（列优先存储，ldc = rows）
    // 
    // cublasSgeam 参数说明:
    //   transa = CUBLAS_OP_T: 对 A 进行转置
    //   transb = CUBLAS_OP_N: B 不转置（但 beta=0 所以 B 不会被使用）
    //   m = rows: 输出矩阵 C 的行数
    //   n = cols: 输出矩阵 C 的列数
    //   lda = cols: A 的 leading dimension（A 的行数）
    //   ldb = rows: B 的 leading dimension（需要 >= m，即使 beta=0 也要有效）
    //   ldc = rows: C 的 leading dimension
    cublasStatus_t status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                         rows, cols, 
                                         &alpha, A, cols,    // A: [cols, rows], lda = cols
                                         &beta, A_t, rows,   // B: 使用 A_t 作为占位符, ldb = rows (>= m)
                                         A_t, rows);         // C: [rows, cols], ldc = rows
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgeam failed with status %d\n", status);
    }
}

// 3D 张量 permute: [T, B, I] -> [I, T, B]
// 使用 CPU 实现，因为这个操作只在初始化时进行一次
void permute3D_TBI_to_ITB(const std::vector<float> &src, std::vector<float> &dst, int T, int B,
                          int I) {
    dst.resize(I * T * B);
    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < I; ++i) {
                // src[t, b, i] = src[t * B * I + b * I + i]
                // dst[i, t, b] = dst[i * T * B + t * B + b]
                dst[i * T * B + t * B + b] = src[t * B * I + b * I + i];
            }
        }
    }
}

constexpr int BATCH_SIZE = 64;    // 批大小
constexpr int SEQUENCE_LEN = 50;  // 序列长度(T), 每个样本有T个时间步
constexpr int HIDDEN_DIMS = 256;  // 隐藏层维度(H), h_t的维度
constexpr int INPUT_DIMS = 256;   // 输入维度(I), x_t的维度

cublasHandle_t g_blas_handle = nullptr;

class ScopeTimer {
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

// ==================== 推理接口 ====================

// 浮点 GRU 推理
void runFloatInference(const int time_steps, const int batch_size, const int input_size,
                       const int hidden_size, const float *W, const float *R, const float *bx,
                       const float *br, const float *x, float *h) {
    ScopeTimer t("FloatInference:");
    hasteGRUForward(false,  // inference mode
                    time_steps, batch_size, input_size, hidden_size, W, R, bx, br, x,
                    nullptr,  // h0
                    g_blas_handle, h, nullptr);
}

// 量化 GRU 推理（使用统一接口 forwardInterface）
void runQuantInference(const int time_steps, const int batch_size, const int input_size,
                       const int hidden_size, const float *W, const float *R, const float *bx,
                       const float *br, const float *x,
                       const GRUQuantitativeParameters &quant_params, float *h) {
    ScopeTimer t("QuantInference:");
    forwardInterface(false,  // inference mode
                     true,   // is_quant
                     time_steps, batch_size, input_size, hidden_size, W, R, bx, br, x,
                     nullptr,  // h0
                     quant_params, g_blas_handle, h, nullptr);
}

// ==================== 训练接口 ====================

// 浮点 GRU 训练
// 注意: W_t, R_t, x_t 是转置后的数据
GRUTrainGradients runFloatTraining(const int time_steps, const int batch_size, const int input_size,
                                   const int hidden_size, const float *W, const float *R,
                                   const float *bx, const float *br, const float *x,
                                   const float *W_t, const float *R_t, const float *x_t,
                                   const float *dh_new) {
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    // 前向传播
    {
        ScopeTimer t("FloatTraining Forward:");
        hasteGRUForward(true,  // training mode
                        time_steps, batch_size, input_size, hidden_size, W, R, bx, br, x,
                        nullptr,  // h0
                        g_blas_handle, h_dev.data(), v_dev.data());
    }

    // 创建梯度缓存（必须初始化为零，因为反向传播是累加梯度）
    dev::vector<float> dx_dev(time_steps * batch_size * input_size);
    dev::vector<float> dW_dev(input_size * hidden_size * 3);
    dev::vector<float> dR_dev(hidden_size * hidden_size * 3);
    dev::vector<float> dbx_dev(hidden_size * 3);
    dev::vector<float> dbr_dev(hidden_size * 3);
    dev::vector<float> dh_dev(batch_size * hidden_size);
    dx_dev.zero();
    dW_dev.zero();
    dR_dev.zero();
    dbx_dev.zero();
    dbr_dev.zero();
    dh_dev.zero();

    // 反向传播
    // 注意：反向传播需要转置后的数据
    // W_t: [H*3, C], R_t: [H*3, H], x_t: [I, T, B]
    {
        ScopeTimer t("FloatTraining Backward:");
        hasteGRUBackward(time_steps, batch_size, input_size, hidden_size, W_t, R_t, bx, br, x_t,
                         dh_new, h_dev.data(), v_dev.data(), g_blas_handle, dx_dev.data(),
                         dW_dev.data(), dR_dev.data(), dbx_dev.data(), dbr_dev.data(),
                         dh_dev.data());
    }

    // 拷贝结果回 CPU
    GRUTrainGradients gradients;
    d2h(gradients.dx, dx_dev);
    d2h(gradients.dW, dW_dev);
    d2h(gradients.dR, dR_dev);
    d2h(gradients.dbx, dbx_dev);
    d2h(gradients.dbr, dbr_dev);
    d2h(gradients.dh, dh_dev);

    // h 跳过初始状态，只返回 time_steps 个 h
    const int h_output_size = time_steps * batch_size * hidden_size;
    gradients.h.resize(h_output_size);
    d2h(gradients.h.data(), h_dev.data() + batch_size * hidden_size, h_output_size);

    // v 中间值
    d2h(gradients.v, v_dev);

    return gradients;
}

// 量化 GRU 训练（使用统一接口 forwardInterface 进行前向传播）
// 注意: W_t, R_t, x_t 是转置后的数据
GRUTrainGradients runQuantTraining(const int time_steps, const int batch_size, const int input_size,
                                   const int hidden_size, const float *W, const float *R,
                                   const float *bx, const float *br, const float *x,
                                   const float *W_t, const float *R_t, const float *x_t,
                                   const float *dh_new,
                                   const GRUQuantitativeParameters &quant_params) {
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);

    // 前向传播（使用统一接口 forwardInterface）
    {
        ScopeTimer t("QuantTraining Forward:");
        forwardInterface(true,  // training mode
                         true,  // is_quant
                         time_steps, batch_size, input_size, hidden_size, W, R, bx, br, x,
                         nullptr,  // h0
                         quant_params, g_blas_handle, h_dev.data(), v_dev.data());
    }

    // 反向传播（使用反量化后的 h 和 v）
    // 创建梯度缓存（必须初始化为零，因为反向传播是累加梯度）
    dev::vector<float> dx_dev(time_steps * batch_size * input_size);
    dev::vector<float> dW_dev(input_size * hidden_size * 3);
    dev::vector<float> dR_dev(hidden_size * hidden_size * 3);
    dev::vector<float> dbx_dev(hidden_size * 3);
    dev::vector<float> dbr_dev(hidden_size * 3);
    dev::vector<float> dh_dev(batch_size * hidden_size);
    dx_dev.zero();
    dW_dev.zero();
    dR_dev.zero();
    dbx_dev.zero();
    dbr_dev.zero();
    dh_dev.zero();

    // 反向传播
    // 注意：反向传播需要转置后的数据
    // W_t: [H*3, C], R_t: [H*3, H], x_t: [I, T, B]
    {
        ScopeTimer t("QuantTraining Backward:");
        hasteGRUBackward(time_steps, batch_size, input_size, hidden_size, W_t, R_t, bx, br, x_t,
                         dh_new, h_dev.data(), v_dev.data(), g_blas_handle, dx_dev.data(),
                         dW_dev.data(), dR_dev.data(), dbx_dev.data(), dbr_dev.data(),
                         dh_dev.data());
    }

    // 拷贝结果回 CPU
    GRUTrainGradients gradients;
    d2h(gradients.dx, dx_dev);
    d2h(gradients.dW, dW_dev);
    d2h(gradients.dR, dR_dev);
    d2h(gradients.dbx, dbx_dev);
    d2h(gradients.dbr, dbr_dev);
    d2h(gradients.dh, dh_dev);

    // h 跳过初始状态，只返回 time_steps 个 h
    const int h_output_size = time_steps * batch_size * hidden_size;
    gradients.h.resize(h_output_size);
    d2h(gradients.h.data(), h_dev.data() + batch_size * hidden_size, h_output_size);

    // v 中间值
    d2h(gradients.v, v_dev);

    return gradients;
}

// ==================== 主函数 ====================

int main() {
    // 使用固定随机种子，确保结果可复现
    srand(42);
    setGlobalRandomSeed(42);  // C++ 随机数引擎的种子

    // 设置 CUDA 确定性模式
    cudaDeviceSetLimit(cudaLimitStackSize, 1024);  // 避免栈内存的随机性

    // ========== 1. 初始化 cuBLAS ==========
    init_gru_cublas(g_blas_handle);
    
    // 设置 cuBLAS 为确定性模式
    cublasSetMathMode(g_blas_handle, CUBLAS_DEFAULT_MATH);

    // ========== 2. 初始化权重和输入 ==========
    std::vector<float> W(INPUT_DIMS * HIDDEN_DIMS * 3);
    std::vector<float> R(HIDDEN_DIMS * HIDDEN_DIMS * 3);
    std::vector<float> bx(HIDDEN_DIMS * 3);
    std::vector<float> br(HIDDEN_DIMS * 3);
    std::vector<float> x(SEQUENCE_LEN * BATCH_SIZE * INPUT_DIMS);
    std::vector<float> dh((SEQUENCE_LEN + 1) * BATCH_SIZE * HIDDEN_DIMS);

    // W: 输入权重矩阵
    fillVectorWithNormalDistribution(W, -0.001f,  0.001f);

    // R: 循环权重矩阵
    fillVectorWithNormalDistribution(R, -0.005f,  0.005f);

    // bx, br: 偏置
    fillVectorWithNormalDistribution(bx, -0.15, 0.15);
    fillVectorWithNormalDistribution(br, -0.15, 0.15);

    // x: 输入序列
    fillVectorWithNormalDistribution(x, -3, 3.5);

    // dh: 上游梯度
    fillVectorWithNormalDistribution(dh, -0.5, 0.5);

    const int time_steps = SEQUENCE_LEN;
    const int batch_size = BATCH_SIZE;
    const int input_size = INPUT_DIMS;
    const int hidden_size = HIDDEN_DIMS;

    // ========== 3. 拷贝数据到 GPU ==========
    dev::vector<float> W_dev(W);
    dev::vector<float> R_dev(R);
    dev::vector<float> bx_dev(bx);
    dev::vector<float> br_dev(br);
    dev::vector<float> x_dev(x);
    dev::vector<float> dh_dev(dh);

    // ========== 4. 校准量化参数并初始化 LUT（只做一次）==========
    printf("\n========== Calibrating Quantization Parameters ==========\n");
    OperatorQuantConfig bitwidth_config;
    GRUQuantitativeParameters quant_params;
    {
        ScopeTimer t("CalibrateAndInitLut:");
        
        // 步骤 1: 收集直方图
        GRUHistogramCollectors hist_collectors(hidden_size);
        calibrateGruHistograms(
            time_steps, batch_size, input_size, hidden_size, W_dev.data(), R_dev.data(),
            bx_dev.data(), br_dev.data(), x_dev.data(), g_blas_handle, hist_collectors);
        
        // 步骤 2: 从直方图计算量化参数
        quant_params = calculateGRUQuantitativeParametersFromHistograms(
            hist_collectors, bitwidth_config, true);
        
        // 步骤 3: 初始化 LUT 表
        initialize_quantization_lut(quant_params);
    }
    printf("Calibration completed.\n");

    printParms(quant_params);

    // ========== 5. 推理测试 ==========
    printf("\n========== Running Inference Tests ==========\n");

    // 浮点推理
    dev::vector<float> h_float_dev((time_steps + 1) * batch_size * hidden_size);
    runFloatInference(time_steps, batch_size, input_size, hidden_size, W_dev.data(), R_dev.data(),
                      bx_dev.data(), br_dev.data(), x_dev.data(), h_float_dev.data());

    // 量化推理
    dev::vector<float> h_quant_dev((time_steps + 1) * batch_size * hidden_size);
    runQuantInference(time_steps, batch_size, input_size, hidden_size, W_dev.data(), R_dev.data(),
                      bx_dev.data(), br_dev.data(), x_dev.data(), quant_params, h_quant_dev.data());

    // 比较推理结果
    std::vector<float> h_float, h_quant;
    d2h(h_float, h_float_dev);
    d2h(h_quant, h_quant_dev);
    compareHValues(h_float, h_quant, time_steps, batch_size, hidden_size,
                   "Inference: Float vs Quant");

    printf("cudaError(Inference): %s\n", cudaGetErrorString(cudaGetLastError()));

#if 0  // 暂时注释掉训练测试，专注调试推理
    // ========== 6. 训练测试 ==========
    printf("\n========== Running Training Tests ==========\n");

    // ========== 6.1 准备反向传播所需的转置数据 ==========
    // 根据 gru.h 中的注释，反向传播需要转置后的数据：
    // W_t: [H*3, C] (原 W 是 [C, H*3])
    // R_t: [H*3, H] (原 R 是 [H, H*3])
    // x_t: [I, T, B] (原 x 是 [T, B, I])
    printf("\n----- Preparing Transposed Data for Backward -----\n");

    // 转置 W: [C, H*3] -> [H*3, C]
    dev::vector<float> W_t_dev(input_size * hidden_size * 3);
    transpose2D(g_blas_handle, W_dev.data(), W_t_dev.data(), hidden_size * 3, input_size);

    // 转置 R: [H, H*3] -> [H*3, H]
    dev::vector<float> R_t_dev(hidden_size * hidden_size * 3);
    transpose2D(g_blas_handle, R_dev.data(), R_t_dev.data(), hidden_size * 3, hidden_size);

    // 转置 x: [T, B, I] -> [I, T, B]
    std::vector<float> x_t;
    permute3D_TBI_to_ITB(x, x_t, time_steps, batch_size, input_size);
    dev::vector<float> x_t_dev(x_t);

    cudaDeviceSynchronize();
    printf("Transposed data prepared.\n");

    // 浮点训练
    printf("\n----- Float Training -----\n");
    GRUTrainGradients gradients_float =
        runFloatTraining(time_steps, batch_size, input_size, hidden_size, W_dev.data(),
                         R_dev.data(), bx_dev.data(), br_dev.data(), x_dev.data(), W_t_dev.data(),
                         R_t_dev.data(), x_t_dev.data(), dh_dev.data());

    printf("cudaError(FloatTraining): %s\n", cudaGetErrorString(cudaGetLastError()));

    // 量化训练
    printf("\n----- Quant Training -----\n");
    GRUTrainGradients gradients_quant =
        runQuantTraining(time_steps, batch_size, input_size, hidden_size, W_dev.data(),
                         R_dev.data(), bx_dev.data(), br_dev.data(), x_dev.data(), W_t_dev.data(),
                         R_t_dev.data(), x_t_dev.data(), dh_dev.data(), quant_params);

    printf("cudaError(QuantTraining): %s\n", cudaGetErrorString(cudaGetLastError()));

    // ========== 7. 比较训练结果 ==========
    printf("\n========== Comparing Training Results ==========\n");

    // 比较 V 中间值
    compareVIntermediateValues(gradients_float.v, gradients_quant.v, time_steps, batch_size,
                               hidden_size, "Float vs Quant");

    // 比较 h 隐藏状态
    compareHValues(gradients_float.h, gradients_quant.h, time_steps, batch_size, hidden_size,
                   "Training H: Float vs Quant");

    // 比较梯度
    compareGRUTrainGradients(gradients_float, gradients_quant, "Float vs Quant");
#endif

    // ========== 8. 清理 ==========
    cublasDestroy(g_blas_handle);

    printf("\n========== All Tests Completed ==========\n");

    return 0;
}
