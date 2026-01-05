#include "gru.h"

#include <cuda_runtime_api.h>

#include <chrono>
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
#include "calibration_gpu.cuh"
#include "quantized_unit_testing.cuh"
#include "tensor_utils.h"

// ==================== 配置 ====================

// 全局配置：选择校准方式 (SQNR 或 PERCENTILE)
// enum class CalibrationMethod : int8_t {
//     NONE = 0,        // 不校准，正常 forward
//     MINMAX = 1,      // 收集 min/max 范围
//     SQNR = 2,        // 收集直方图，使用 SQNR 优化
//     PERCENTILE = 3   // 收集直方图，使用百分位裁剪
// };
constexpr CalibrationMethod CALIBRATION_METHOD = CalibrationMethod::MINMAX;

// 默认参数（可通过命令行覆盖）
int g_batch_size = 64;
int g_sequence_len = 50;
int g_hidden_dims = 256;
int g_input_dims = 256;

cublasHandle_t g_blas_handle = nullptr;

// ==================== 工具类 ====================

/**
 * @brief CUDA 事件计时器 (RAII)
 */
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        cudaDeviceSynchronize();
        cudaEventRecord(start_);
    }
    
    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_, stop_);
        return elapsed_ms;
    }

private:
    cudaEvent_t start_, stop_;
};

/**
 * @brief 作用域计时器 - 自动打印执行时间
 */
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
        printf("%s %.3f ms\n", msg_.c_str(), elapsed_ms);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    std::string msg_;
    cudaEvent_t start_, stop_;
};

/**
 * @brief CPU 高精度计时器
 */
class CpuTimer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    
    double stop_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// ==================== 命令行解析 ====================

void printUsage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -T <value>  Sequence length (time steps), default: %d\n", g_sequence_len);
    printf("  -C <value>  Input dimension, default: %d\n", g_input_dims);
    printf("  -B <value>  Batch size, default: %d\n", g_batch_size);
    printf("  -H <value>  Hidden dimension, default: %d\n", g_hidden_dims);
    printf("  -h          Show this help message\n");
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

// ==================== 验证工具函数 ====================

/**
 * @brief 验证标量参数是否匹配
 */
struct VerifyResult {
    int exp_match = 0;
    int zp_match = 0;
    int total = 0;
};

void verifyScalarParam(const char* name, int8_t cpu_exp, int8_t gpu_exp,
                       int32_t cpu_zp, int32_t gpu_zp, VerifyResult& result) {
    bool exp_ok = (cpu_exp == gpu_exp);
    bool zp_ok = (cpu_zp == gpu_zp);
    result.total++;
    if (exp_ok) result.exp_match++;
    if (zp_ok) result.zp_match++;
    
    printf("    %s: exp=%d%s, zp=%d%s\n", name, 
           cpu_exp, exp_ok ? "" : (std::string("(GPU=") + std::to_string(gpu_exp) + ")").c_str(),
           cpu_zp, zp_ok ? "" : (std::string("(GPU=") + std::to_string(gpu_zp) + ")").c_str());
}

/**
 * @brief 验证 per-channel 参数匹配情况
 */
std::pair<int, int> countPerChannelMatches(const std::vector<int8_t>& cpu, 
                                            const std::vector<int8_t>& gpu) {
    int match = 0;
    size_t n = std::min(cpu.size(), gpu.size());
    for (size_t i = 0; i < n; ++i) {
        if (cpu[i] == gpu[i]) match++;
    }
    return {match, (int)n};
}

/**
 * @brief 比较直方图统计信息
 */
void compareHistogramStats(const char* name, const Histogram& cpu, const Histogram& gpu) {
    printf("  %s: CPU(min=%.4f, max=%.4f, cnt=%ld) vs GPU(min=%.4f, max=%.4f, cnt=%ld)\n",
           name, cpu.min_val, cpu.max_val, cpu.total_count,
           gpu.min_val, gpu.max_val, gpu.total_count);
    
    float min_diff = std::abs(cpu.min_val - gpu.min_val);
    float max_diff = std::abs(cpu.max_val - gpu.max_val);
    if (min_diff > 0.01f || max_diff > 0.01f) {
        printf("    WARNING: Range mismatch! min_diff=%.6f, max_diff=%.6f\n", min_diff, max_diff);
    }
}

// ==================== 推理函数 ====================

void runFloatInference(int time_steps, int batch_size, int input_size, int hidden_size,
                       const float *W, const float *R, const float *bx, const float *br,
                       const float *x, float *h) {
    ScopeTimer t("FloatInference:");
    hasteGRUForward(false, time_steps, batch_size, input_size, hidden_size, 
                    W, R, bx, br, x, nullptr, g_blas_handle, h, nullptr);
}

void runQuantInference(int time_steps, int batch_size, int input_size, int hidden_size,
                       const float *W, const float *R, const float *bx, const float *br,
                       const float *x, const GRUQuantitativeParameters &quant_params, float *h) {
    ScopeTimer t("QuantInference (GPU):");
    forwardInterface(false, true, time_steps, batch_size, input_size, hidden_size,
                     W, R, bx, br, x, nullptr, quant_params, g_blas_handle, h, nullptr);
}

// ==================== CPU 量化推理 ====================

void runQuantInferenceCPU(int time_steps, int batch_size, int input_size, int hidden_size,
                          const float *W, const float *R, const float *bx, const float *br,
                          const float *x, const GRUQuantitativeParameters &quant_params,
                          float *h) {
    ScopeTimer t("QuantInference (CPU):");
    // 使用浮点权重版本的接口，内部会自动量化
    quantGRUForwardCPU(false, time_steps, batch_size, input_size, hidden_size,
                       W, R, bx, br, x, nullptr, quant_params, h, nullptr);
}

// ==================== 直方图收集性能比较 ====================

void compareHistogramCollectionPerformance(int time_steps, int batch_size, int input_size,
                                            int hidden_size, const float *W_dev, const float *R_dev,
                                            const float *bx_dev, const float *br_dev,
                                            const float *x_dev, int num_runs = 5) {
    printf("\n========== Histogram Collection Performance (CPU vs GPU) ==========\n");
    printf("Config: T=%d, B=%d, I=%d, H=%d, runs=%d\n",
           time_steps, batch_size, input_size, hidden_size, num_runs);

    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);
    CudaTimer timer;

    // CPU 直方图收集（前向传播在 GPU，直方图收集在 CPU）
    printf("\n----- CPU Histogram Collection -----\n");
    float cpu_total_ms = 0.0f;
    for (int run = 0; run < num_runs; ++run) {
        GRUHistogramCollectors cpu_hist(hidden_size);
        h_dev.zero();
        
        timer.start();
        forwardWithHistogramCPU(true, time_steps, batch_size, input_size, hidden_size,
                                W_dev, R_dev, bx_dev, br_dev, x_dev, nullptr, g_blas_handle,
                                &cpu_hist, h_dev.data(), v_dev.data());
        float elapsed = timer.stop();
        cpu_total_ms += elapsed;
        if (run == 0) printf("  Run 1: %.3f ms\n", elapsed);
    }
    float cpu_avg = cpu_total_ms / num_runs;
    printf("  Average: %.3f ms\n", cpu_avg);

    // GPU 直方图收集
    printf("\n----- GPU Histogram Collection -----\n");
    float gpu_total_ms = 0.0f;
    for (int run = 0; run < num_runs; ++run) {
        GRUGPUHistogramCollectors gpu_hist(hidden_size);
        h_dev.zero();
        
        timer.start();
        forwardWithCalibrationGPU(true, time_steps, batch_size, input_size, hidden_size,
                                  W_dev, R_dev, bx_dev, br_dev, x_dev, nullptr, g_blas_handle,
                                  CalibrationMethod::SQNR, nullptr, &gpu_hist,
                                  h_dev.data(), v_dev.data());
        float elapsed = timer.stop();
        gpu_total_ms += elapsed;
        if (run == 0) printf("  Run 1: %.3f ms\n", elapsed);
    }
    float gpu_avg = gpu_total_ms / num_runs;
    printf("  Average: %.3f ms\n", gpu_avg);

    // 结果对比
    printf("\n----- Summary -----\n");
    printf("  CPU: %.3f ms, GPU: %.3f ms, Speedup: %.2fx\n", 
           cpu_avg, gpu_avg, cpu_avg / gpu_avg);

    // 验证一致性
    printf("\n----- Verifying Consistency -----\n");
    GRUHistogramCollectors cpu_hist(hidden_size);
    GRUGPUHistogramCollectors gpu_hist(hidden_size);
    
    h_dev.zero();
    forwardWithHistogramCPU(true, time_steps, batch_size, input_size, hidden_size,
                            W_dev, R_dev, bx_dev, br_dev, x_dev, nullptr, g_blas_handle,
                            &cpu_hist, h_dev.data(), v_dev.data());
    
    h_dev.zero();
    forwardWithCalibrationGPU(true, time_steps, batch_size, input_size, hidden_size,
                              W_dev, R_dev, bx_dev, br_dev, x_dev, nullptr, g_blas_handle,
                              CalibrationMethod::SQNR, nullptr, &gpu_hist, h_dev.data(), v_dev.data());
    
    GRUHistogramCollectors gpu_hist_cpu = convertGPUHistogramsToCPU(gpu_hist);
    
    compareHistogramStats("x", cpu_hist.x_hist.histogram(), gpu_hist_cpu.x_hist.histogram());
    compareHistogramStats("h", cpu_hist.h_hist.histogram(), gpu_hist_cpu.h_hist.histogram());
    compareHistogramStats("Wx", cpu_hist.Wx_hist.histogram(), gpu_hist_cpu.Wx_hist.histogram());
    compareHistogramStats("Rh", cpu_hist.Rh_hist.histogram(), gpu_hist_cpu.Rh_hist.histogram());
    compareHistogramStats("z_out", cpu_hist.z_out_hist.histogram(), gpu_hist_cpu.z_out_hist.histogram());
    compareHistogramStats("r_out", cpu_hist.r_out_hist.histogram(), gpu_hist_cpu.r_out_hist.histogram());
    compareHistogramStats("g_out", cpu_hist.g_out_hist.histogram(), gpu_hist_cpu.g_out_hist.histogram());
}

// ==================== 校准函数 ====================

/**
 * @brief 执行 SQNR/Percentile 校准并返回量化参数
 */
GRUQuantitativeParameters calibrateWithHistogram(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W_dev, const float *R_dev, const float *bx_dev, const float *br_dev,
    const float *x_dev, const OperatorQuantConfig &bitwidth_config, bool use_percentile) {
    
    CudaTimer timer;
    CpuTimer cpu_timer;
    
    // Step 1: 内存分配
    timer.start();
    GRUGPUHistogramCollectors gpu_hist(hidden_size);
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);
    h_dev.zero();
    printf("  [Step 1] Memory allocation: %.3f ms\n", timer.stop());

    // Step 2: GPU 前向传播 + 直方图收集
    timer.start();
    forwardWithCalibrationGPU(true, time_steps, batch_size, input_size, hidden_size,
                              W_dev, R_dev, bx_dev, br_dev, x_dev, nullptr, g_blas_handle,
                              CalibrationMethod::SQNR, nullptr, &gpu_hist,
                              h_dev.data(), v_dev.data());
    printf("  [Step 2] Forward + GPU histogram: %.3f ms\n", timer.stop());

    // Step 3: GPU→CPU 直方图转换
    timer.start();
    GRUHistogramCollectors hist_cpu = convertGPUHistogramsToCPU(gpu_hist);
    printf("  [Step 3] GPU→CPU histogram convert: %.3f ms\n", timer.stop());

    // Step 4: SQNR 计算对比
    cpu_timer.start();
    GRUQuantitativeParameters params_cpu = calculateGRUQuantitativeParametersFromHistograms(
        hist_cpu, bitwidth_config, false, false);
    double sqnr_cpu_ms = cpu_timer.stop_ms();
    printf("  [Step 4a] SQNR param (CPU): %.3f ms\n", sqnr_cpu_ms);

    timer.start();
    GRUQuantitativeParameters params_gpu = calculateGRUQuantitativeParametersFromGPUHistograms(
        gpu_hist, bitwidth_config, false);
    float sqnr_gpu_ms = timer.stop();
    printf("  [Step 4b] SQNR param (GPU): %.3f ms (%.1fx speedup)\n", 
           sqnr_gpu_ms, sqnr_cpu_ms / sqnr_gpu_ms);

    // 验证 GPU vs CPU SQNR 结果
    printf("\n  ----- SQNR Verification -----\n");
    VerifyResult result;
    verifyScalarParam("x", params_cpu.exp2_inv_x_, params_gpu.exp2_inv_x_,
                      params_cpu.zp_x_, params_gpu.zp_x_, result);
    verifyScalarParam("h", params_cpu.exp2_inv_h_, params_gpu.exp2_inv_h_,
                      params_cpu.zp_h_, params_gpu.zp_h_, result);
    verifyScalarParam("Wx", params_cpu.exp2_inv_Wx_, params_gpu.exp2_inv_Wx_,
                      params_cpu.zp_Wx_, params_gpu.zp_Wx_, result);
    verifyScalarParam("Rh", params_cpu.exp2_inv_Rh_, params_gpu.exp2_inv_Rh_,
                      params_cpu.zp_Rh_, params_gpu.zp_Rh_, result);
    verifyScalarParam("z_out", params_cpu.exp2_inv_z_out_, params_gpu.exp2_inv_z_out_,
                      params_cpu.zp_z_out_, params_gpu.zp_z_out_, result);
    verifyScalarParam("r_out", params_cpu.exp2_inv_r_out_, params_gpu.exp2_inv_r_out_,
                      params_cpu.zp_r_out_, params_gpu.zp_r_out_, result);
    verifyScalarParam("g_out", params_cpu.exp2_inv_g_out_, params_gpu.exp2_inv_g_out_,
                      params_cpu.zp_g_out_, params_gpu.zp_g_out_, result);
    printf("    Scalar: exp=%d/%d, zp=%d/%d\n", 
           result.exp_match, result.total, result.zp_match, result.total);

    // Per-channel 验证
    auto [w_m, w_t] = countPerChannelMatches(params_cpu.exp2_inv_W_, params_gpu.exp2_inv_W_);
    auto [r_m, r_t] = countPerChannelMatches(params_cpu.exp2_inv_R_, params_gpu.exp2_inv_R_);
    auto [bx_m, bx_t] = countPerChannelMatches(params_cpu.exp2_inv_bx_, params_gpu.exp2_inv_bx_);
    auto [br_m, br_t] = countPerChannelMatches(params_cpu.exp2_inv_br_, params_gpu.exp2_inv_br_);
    printf("    Per-channel: W=%d/%d, R=%d/%d, bx=%d/%d, br=%d/%d\n",
           w_m, w_t, r_m, r_t, bx_m, bx_t, br_m, br_t);

    // Step 5: Percentile 计算 (如果需要)
    GRUQuantitativeParameters params_percentile;
    if (use_percentile) {
        const int PERC_RUNS = 10;
        double percentile_total = 0.0;
        for (int run = 0; run < PERC_RUNS; ++run) {
            cpu_timer.start();
            params_percentile = calculateGRUQuantitativeParametersFromHistograms(
                hist_cpu, bitwidth_config, false, true, 99.99f);
            percentile_total += cpu_timer.stop_ms();
        }
        printf("  [Step 5] Percentile param (OpenMP 4T, avg of %d): %.3f ms\n", 
               PERC_RUNS, percentile_total / PERC_RUNS);
    }

    // 返回选定的参数
    printf("\n  ----- Using %s parameters -----\n", use_percentile ? "Percentile" : "GPU SQNR");
    return use_percentile ? params_percentile : params_gpu;
}

/**
 * @brief 执行 MinMax 校准并返回量化参数
 */
GRUQuantitativeParameters calibrateWithMinMax(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W_dev, const float *R_dev, const float *bx_dev, const float *br_dev,
    const float *x_dev, const OperatorQuantConfig &bitwidth_config) {
    
    GRUQuantizationRanges ranges(hidden_size);
    dev::vector<float> h_dev((time_steps + 1) * batch_size * hidden_size);
    dev::vector<float> v_dev(time_steps * batch_size * hidden_size * 4);
    h_dev.zero();
    
    forwardWithCalibrationGPU(true, time_steps, batch_size, input_size, hidden_size,
                              W_dev, R_dev, bx_dev, br_dev, x_dev, nullptr, g_blas_handle,
                              CalibrationMethod::MINMAX, &ranges, nullptr,
                              h_dev.data(), v_dev.data());
    
    return calculateGRUQuantitativeParameters(ranges, bitwidth_config);
}

// ==================== 主函数 ====================

int main(int argc, char *argv[]) {
    parseArgs(argc, argv);

    const int T = g_sequence_len;
    const int B = g_batch_size;
    const int I = g_input_dims;
    const int H = g_hidden_dims;

    printf("\n========== Configuration ==========\n");
    printf("T (Sequence):  %d\n", T);
    printf("I (Input):     %d\n", I);
    printf("B (Batch):     %d\n", B);
    printf("H (Hidden):    %d\n", H);
    const char* method_name = 
        CALIBRATION_METHOD == CalibrationMethod::SQNR ? "SQNR" :
        CALIBRATION_METHOD == CalibrationMethod::PERCENTILE ? "PERCENTILE" :
        CALIBRATION_METHOD == CalibrationMethod::MINMAX ? "MINMAX" : "NONE";
    printf("Method:        %s\n", method_name);
    printf("====================================\n");

    // 初始化随机种子
    srand(42);
    setGlobalRandomSeed(42);

    // 初始化 cuBLAS
    init_gru_cublas(g_blas_handle);
    cublasSetMathMode(g_blas_handle, CUBLAS_DEFAULT_MATH);

    // 初始化数据
    std::vector<float> W(I * H * 3), R(H * H * 3), bx(H * 3), br(H * 3);
    std::vector<float> x(T * B * I);
    
    fillVectorWithNormalDistribution(W, -0.001f, 0.001f);
    fillVectorWithNormalDistribution(R, -0.005f, 0.005f);
    fillVectorWithNormalDistribution(bx, -0.15f, 0.15f);
    fillVectorWithNormalDistribution(br, -0.15f, 0.15f);
    fillVectorWithNormalDistribution(x, -3.0f, 3.5f);

    // 拷贝到 GPU
    dev::vector<float> W_dev(W), R_dev(R), bx_dev(bx), br_dev(br), x_dev(x);

    // 直方图收集性能对比
    compareHistogramCollectionPerformance(T, B, I, H, W_dev.data(), R_dev.data(),
                                          bx_dev.data(), br_dev.data(), x_dev.data(), 5);

    // 校准
    printf("\n========== Calibration ==========\n");
    OperatorQuantConfig bitwidth_config;
    GRUQuantitativeParameters quant_params;
    
    {
        ScopeTimer t("Total calibration time:");
        
        if constexpr (CALIBRATION_METHOD == CalibrationMethod::SQNR || 
                      CALIBRATION_METHOD == CalibrationMethod::PERCENTILE) {
            bool use_percentile = (CALIBRATION_METHOD == CalibrationMethod::PERCENTILE);
            quant_params = calibrateWithHistogram(T, B, I, H, W_dev.data(), R_dev.data(),
                                                   bx_dev.data(), br_dev.data(), x_dev.data(),
                                                   bitwidth_config, use_percentile);
        } else {
            quant_params = calibrateWithMinMax(T, B, I, H, W_dev.data(), R_dev.data(),
                                                bx_dev.data(), br_dev.data(), x_dev.data(),
                                                bitwidth_config);
        }
        // LUT 已在校准函数中自动生成
    }
    
    printParms(quant_params);

    // 推理测试
    printf("\n========== Inference Tests ==========\n");
    
    dev::vector<float> h_float((T + 1) * B * H);
    dev::vector<float> h_quant_gpu((T + 1) * B * H);

    runFloatInference(T, B, I, H, W_dev.data(), R_dev.data(), bx_dev.data(), br_dev.data(),
                      x_dev.data(), h_float.data());
    
    runQuantInference(T, B, I, H, W_dev.data(), R_dev.data(), bx_dev.data(), br_dev.data(),
                      x_dev.data(), quant_params, h_quant_gpu.data());
    
    std::vector<float> h_quant_cpu_vec((T + 1) * B * H);
    runQuantInferenceCPU(T, B, I, H, W.data(), R.data(), bx.data(), br.data(),
                         x.data(), quant_params, h_quant_cpu_vec.data());

    // 比较结果
    std::vector<float> h_float_cpu, h_quant_gpu_cpu;
    d2h(h_float_cpu, h_float);
    d2h(h_quant_gpu_cpu, h_quant_gpu);

    printf("\n----- Comparison Results -----\n");
    compareHValues(h_float_cpu, h_quant_gpu_cpu, T, B, H, "Float vs Quant(GPU)");
    compareHValues(h_float_cpu, h_quant_cpu_vec, T, B, H, "Float vs Quant(CPU)");
    compareHValues(h_quant_gpu_cpu, h_quant_cpu_vec, T, B, H, "Quant(GPU) vs Quant(CPU)");

    printf("CUDA Error: %s\n", cudaGetErrorString(cudaGetLastError()));

#if 0  // 训练测试（暂时禁用，需要时改为 #if 1）
    // ========== 训练测试 ==========
    printf("\n========== Running Training Tests ==========\n");

    // 准备上游梯度
    std::vector<float> dh((T + 1) * B * H);
    fillVectorWithNormalDistribution(dh, -0.5f, 0.5f);
    dev::vector<float> dh_dev(dh);

    // 准备反向传播所需的转置数据
    // W_t: [H*3, I] (原 W 是 [I, H*3])
    // R_t: [H*3, H] (原 R 是 [H, H*3])
    // x_t: [I, T, B] (原 x 是 [T, B, I])
    printf("\n----- Preparing Transposed Data for Backward -----\n");

    dev::vector<float> W_t_dev(I * H * 3);
    transpose2D(g_blas_handle, W_dev.data(), W_t_dev.data(), H * 3, I);

    dev::vector<float> R_t_dev(H * H * 3);
    transpose2D(g_blas_handle, R_dev.data(), R_t_dev.data(), H * 3, H);

    std::vector<float> x_t;
    permute3D_TBI_to_ITB(x, x_t, T, B, I);
    dev::vector<float> x_t_dev(x_t);

    cudaDeviceSynchronize();
    printf("Transposed data prepared.\n");

    // 浮点训练
    printf("\n----- Float Training -----\n");
    GRUTrainGradients gradients_float;
    {
        dev::vector<float> h_train((T + 1) * B * H);
        dev::vector<float> v_train(T * B * H * 4);

        {
            ScopeTimer t("FloatTraining Forward:");
            hasteGRUForward(true, T, B, I, H, W_dev.data(), R_dev.data(), bx_dev.data(),
                            br_dev.data(), x_dev.data(), nullptr, g_blas_handle,
                            h_train.data(), v_train.data());
        }

        dev::vector<float> dx_dev(T * B * I);
        dev::vector<float> dW_dev(I * H * 3);
        dev::vector<float> dR_dev(H * H * 3);
        dev::vector<float> dbx_dev(H * 3);
        dev::vector<float> dbr_dev(H * 3);
        dev::vector<float> dh_out_dev(B * H);
        dx_dev.zero(); dW_dev.zero(); dR_dev.zero();
        dbx_dev.zero(); dbr_dev.zero(); dh_out_dev.zero();

        {
            ScopeTimer t("FloatTraining Backward:");
            hasteGRUBackward(T, B, I, H, W_t_dev.data(), R_t_dev.data(), bx_dev.data(),
                             br_dev.data(), x_t_dev.data(), dh_dev.data(), h_train.data(),
                             v_train.data(), g_blas_handle, dx_dev.data(), dW_dev.data(),
                             dR_dev.data(), dbx_dev.data(), dbr_dev.data(), dh_out_dev.data());
        }

        d2h(gradients_float.dx, dx_dev);
        d2h(gradients_float.dW, dW_dev);
        d2h(gradients_float.dR, dR_dev);
        d2h(gradients_float.dbx, dbx_dev);
        d2h(gradients_float.dbr, dbr_dev);
        d2h(gradients_float.dh, dh_out_dev);
        gradients_float.h.resize(T * B * H);
        d2h(gradients_float.h.data(), h_train.data() + B * H, T * B * H);
        d2h(gradients_float.v, v_train);
    }
    printf("CUDA Error (FloatTraining): %s\n", cudaGetErrorString(cudaGetLastError()));

    // 量化训练
    printf("\n----- Quant Training -----\n");
    GRUTrainGradients gradients_quant;
    {
        dev::vector<float> h_train((T + 1) * B * H);
        dev::vector<float> v_train(T * B * H * 4);

        {
            ScopeTimer t("QuantTraining Forward:");
            forwardInterface(true, true, T, B, I, H, W_dev.data(), R_dev.data(), bx_dev.data(),
                             br_dev.data(), x_dev.data(), nullptr, quant_params, g_blas_handle,
                             h_train.data(), v_train.data());
        }

        dev::vector<float> dx_dev(T * B * I);
        dev::vector<float> dW_dev(I * H * 3);
        dev::vector<float> dR_dev(H * H * 3);
        dev::vector<float> dbx_dev(H * 3);
        dev::vector<float> dbr_dev(H * 3);
        dev::vector<float> dh_out_dev(B * H);
        dx_dev.zero(); dW_dev.zero(); dR_dev.zero();
        dbx_dev.zero(); dbr_dev.zero(); dh_out_dev.zero();

        {
            ScopeTimer t("QuantTraining Backward:");
            hasteGRUBackward(T, B, I, H, W_t_dev.data(), R_t_dev.data(), bx_dev.data(),
                             br_dev.data(), x_t_dev.data(), dh_dev.data(), h_train.data(),
                             v_train.data(), g_blas_handle, dx_dev.data(), dW_dev.data(),
                             dR_dev.data(), dbx_dev.data(), dbr_dev.data(), dh_out_dev.data());
        }

        d2h(gradients_quant.dx, dx_dev);
        d2h(gradients_quant.dW, dW_dev);
        d2h(gradients_quant.dR, dR_dev);
        d2h(gradients_quant.dbx, dbx_dev);
        d2h(gradients_quant.dbr, dbr_dev);
        d2h(gradients_quant.dh, dh_out_dev);
        gradients_quant.h.resize(T * B * H);
        d2h(gradients_quant.h.data(), h_train.data() + B * H, T * B * H);
        d2h(gradients_quant.v, v_train);
    }
    printf("CUDA Error (QuantTraining): %s\n", cudaGetErrorString(cudaGetLastError()));

    // 比较训练结果
    printf("\n========== Comparing Training Results ==========\n");
    compareVIntermediateValues(gradients_float.v, gradients_quant.v, T, B, H, "Float vs Quant");
    compareHValues(gradients_float.h, gradients_quant.h, T, B, H, "Training H: Float vs Quant");
    compareGRUTrainGradients(gradients_float, gradients_quant, "Float vs Quant");
#endif  // 训练测试

    // 清理
    cublasDestroy(g_blas_handle);
    printf("\n========== All Tests Completed ==========\n");

    return 0;
}
