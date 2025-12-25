#include "gru.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "check_data.h"
#include "dev_vector.h"
#include "gru_interface.h"
#include "histogram_collector.h"
#include "quantized_unit_testing.cuh"
#include "tensor_utils.h"

// ==================== 校准方式选择 ====================
enum class CalibrationMethod {
    MIN_MAX,   // 使用 min/max 范围校准（简单快速）
    HISTOGRAM  // 使用直方图校准（SQNR 优化，更精确）
};

// 全局配置：选择校准方式
constexpr CalibrationMethod CALIBRATION_METHOD = CalibrationMethod::MIN_MAX;

// 默认配置（可通过命令行参数覆盖）
int g_batch_size = 64;    // 批大小 (B)
int g_sequence_len = 50;  // 序列长度 (T), 每个样本有T个时间步
int g_hidden_dims = 256;  // 隐藏层维度 (H), h_t的维度
int g_input_dims = 256;   // 输入维度 (C), x_t的维度

void printUsage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -T <value>  Sequence length (time steps), default: %d\n", g_sequence_len);
    printf("  -C <value>  Input dimension, default: %d\n", g_input_dims);
    printf("  -B <value>  Batch size, default: %d\n", g_batch_size);
    printf("  -H <value>  Hidden dimension, default: %d\n", g_hidden_dims);
    printf("  -h          Show this help message\n");
    printf("\nExample: %s -T 10 -C 128 -B 32 -H 64\n", program_name);
}

void parseArgs(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "-T" && i + 1 < argc) {
            g_sequence_len = std::atoi(argv[++i]);
        } else if (arg == "-C" && i + 1 < argc) {
            g_input_dims = std::atoi(argv[++i]);
        } else if (arg == "-B" && i + 1 < argc) {
            g_batch_size = std::atoi(argv[++i]);
        } else if (arg == "-H" && i + 1 < argc) {
            g_hidden_dims = std::atoi(argv[++i]);
        }
    }
}

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

int main(int argc, char *argv[]) {
    // 解析命令行参数
    parseArgs(argc, argv);

    // 使用解析后的参数
    const int BATCH_SIZE = g_batch_size;
    const int SEQUENCE_LEN = g_sequence_len;
    const int HIDDEN_DIMS = g_hidden_dims;
    const int INPUT_DIMS = g_input_dims;

    printf("\n========== Configuration ==========\n");
    printf("T (Sequence Length): %d\n", SEQUENCE_LEN);
    printf("C (Input Dims):      %d\n", INPUT_DIMS);
    printf("B (Batch Size):      %d\n", BATCH_SIZE);
    printf("H (Hidden Dims):     %d\n", HIDDEN_DIMS);
    printf("====================================\n");

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
    fillVectorWithNormalDistribution(W, -0.001f, 0.001f);

    // R: 循环权重矩阵
    fillVectorWithNormalDistribution(R, -0.005f, 0.005f);

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
    printf("Calibration method: %s\n", CALIBRATION_METHOD == CalibrationMethod::HISTOGRAM
                                           ? "HISTOGRAM (SQNR优化)"
                                           : "MIN_MAX (简单快速)");

    OperatorQuantConfig bitwidth_config;
    // bitwidth_config.setAllBitWidths(16);
    GRUQuantitativeParameters quant_params;
    {
        ScopeTimer t("CalibrateAndInitLut:");

        if constexpr (CALIBRATION_METHOD == CalibrationMethod::HISTOGRAM) {
            // ==================== 方式一：直方图校准（SQNR 优化）====================
            // 优点：更精确，使用 SQNR 优化选择最佳量化范围
            // 缺点：计算开销稍大

            // 步骤 1: 收集直方图
            GRUHistogramCollectors hist_collectors(hidden_size);
            calibrateGruHistograms(time_steps, batch_size, input_size, hidden_size, W_dev.data(),
                                   R_dev.data(), bx_dev.data(), br_dev.data(), x_dev.data(),
                                   g_blas_handle, hist_collectors);

            // 步骤 2: 从直方图计算量化参数
            quant_params = calculateGRUQuantitativeParametersFromHistograms(hist_collectors,
                                                                            bitwidth_config, true);
        } else {
            // ==================== 方式二：Min/Max 范围校准 ====================
            // 优点：简单快速
            // 缺点：对异常值敏感，量化范围可能不够紧凑

            // 步骤 1: 收集 min/max 范围
            GRUQuantizationRanges quant_ranges(hidden_size);
            calibrateGruRanges(time_steps, batch_size, input_size, hidden_size, W_dev.data(),
                               R_dev.data(), bx_dev.data(), br_dev.data(), x_dev.data(),
                               g_blas_handle, quant_ranges);

            // 步骤 2: 从 min/max 范围计算量化参数
            quant_params = calculateGRUQuantitativeParameters(quant_ranges, bitwidth_config);
        }

        // 步骤 3: 初始化 LUT 表（两种方式共用）
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
