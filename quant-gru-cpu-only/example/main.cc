// ============================================================================
// main.cc - CPU 定点 GRU 示例程序
// ============================================================================

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "gru_quant_cpu.h"

int main(int argc, char* argv[]) {
    // 默认参数
    int input_size = 128;
    int hidden_size = 256;
    int batch_size = 64;
    int seq_len = 50;
    int bitwidth = 8;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--input-size" && i + 1 < argc) {
            input_size = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--hidden-size" && i + 1 < argc) {
            hidden_size = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--batch-size" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--seq-len" && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--bitwidth" && i + 1 < argc) {
            bitwidth = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--help") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --input-size N    Input dimension (default: 128)\n");
            printf("  --hidden-size N   Hidden dimension (default: 256)\n");
            printf("  --batch-size N    Batch size (default: 64)\n");
            printf("  --seq-len N       Sequence length (default: 50)\n");
            printf("  --bitwidth N      Quantization bitwidth (8 or 16, default: 8)\n");
            return 0;
        }
    }

    printf("===============================================================\n");
    printf("CPU Fixed-Point GRU Example\n");
    printf("===============================================================\n");
    printf("Configuration:\n");
    printf("  input_size:  %d\n", input_size);
    printf("  hidden_size: %d\n", hidden_size);
    printf("  batch_size:  %d\n", batch_size);
    printf("  seq_len:     %d\n", seq_len);
    printf("  bitwidth:    %d\n", bitwidth);
    printf("---------------------------------------------------------------\n");

    // 随机数生成器
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.7f, 0.8f);

    // 分配内存
    const int hidden3 = hidden_size * 3;

    // 浮点数据
    std::vector<float> x_float(seq_len * batch_size * input_size);
    std::vector<float> W_float(input_size * hidden3);
    std::vector<float> R_float(hidden_size * hidden3);
    std::vector<float> bx_float(hidden3);
    std::vector<float> br_float(hidden3);

    // 随机初始化
    for (auto& v : x_float) v = dist(gen);
    for (auto& v : W_float) v = dist(gen) * 0.1f;
    for (auto& v : R_float) v = dist(gen) * 0.1f;
    for (auto& v : bx_float) v = dist(gen) * 0.1f;
    for (auto& v : br_float) v = dist(gen) * 0.1f;

    // 量化参数设置（示例值）
    GRUQuantitativeParameters quant_params;
    quant_params.hidden_ = hidden_size;
    quant_params.bitwidth_config_ = OperatorQuantConfig::create(bitwidth);

    // 设置量化参数（简化的对称量化示例）
    quant_params.exp2_inv_x_ = 7;
    quant_params.zp_x_ = 0;
    quant_params.exp2_inv_h_ = 8;
    quant_params.zp_h_ = 0;
    quant_params.exp2_inv_Wx_ = 7;
    quant_params.zp_Wx_ = 0;
    quant_params.exp2_inv_Rh_ = 8;
    quant_params.zp_Rh_ = 0;
    quant_params.exp2_inv_z_pre_ = 7;
    quant_params.zp_z_pre_ = 0;
    quant_params.exp2_inv_z_out_ = 8;
    quant_params.zp_z_out_ = 0;
    quant_params.exp2_inv_r_pre_ = 7;
    quant_params.zp_r_pre_ = 0;
    quant_params.exp2_inv_r_out_ = 8;
    quant_params.zp_r_out_ = 0;
    quant_params.exp2_inv_g_pre_ = 7;
    quant_params.zp_g_pre_ = 0;
    quant_params.exp2_inv_g_out_ = 7;
    quant_params.zp_g_out_ = 0;
    quant_params.exp2_inv_Rh_add_br_ = 8;
    quant_params.zp_Rh_add_br_ = 0;
    quant_params.exp2_inv_rRh_ = 8;
    quant_params.zp_rRh_ = 0;
    quant_params.exp2_inv_new_contrib_ = 8;
    quant_params.zp_new_contrib_ = 0;
    quant_params.exp2_inv_old_contrib_ = 8;
    quant_params.zp_old_contrib_ = 0;

    // Per-channel 参数
    quant_params.exp2_inv_W_.resize(hidden3, 10);
    quant_params.exp2_inv_R_.resize(hidden3, 10);
    quant_params.exp2_inv_bx_.resize(hidden3, 10);
    quant_params.exp2_inv_br_.resize(hidden3, 10);

    // 量化输入数据
    std::vector<int8_t> x_quant(seq_len * batch_size * input_size);
    std::vector<int8_t> h_quant((seq_len + 1) * batch_size * hidden_size, 0);  // 初始化为0
    std::vector<int8_t> W_quant(input_size * hidden3);
    std::vector<int8_t> R_quant(hidden_size * hidden3);
    std::vector<int32_t> bx_quant(hidden3);
    std::vector<int32_t> br_quant(hidden3);

    // 量化
    quantification(x_float.data(), x_quant.data(), x_float.size(),
                   quant_params.exp2_inv_x_, quant_params.zp_x_);
    quantificationPerChannel(W_float.data(), W_quant.data(), input_size, hidden3,
                             quant_params.exp2_inv_W_);
    quantificationPerChannel(R_float.data(), R_quant.data(), hidden_size, hidden3,
                             quant_params.exp2_inv_R_);

    for (int i = 0; i < hidden3; i++) {
        bx_quant[i] = quantize<int32_t>(bx_float[i], quant_params.exp2_inv_bx_[i], 0);
        br_quant[i] = quantize<int32_t>(br_float[i], quant_params.exp2_inv_br_[i], 0);
    }

    printf("Quantization complete.\n");
    printf("  x_quant range: [%d, %d]\n",
           *std::min_element(x_quant.begin(), x_quant.end()),
           *std::max_element(x_quant.begin(), x_quant.end()));
    printf("  W_quant range: [%d, %d]\n",
           *std::min_element(W_quant.begin(), W_quant.end()),
           *std::max_element(W_quant.begin(), W_quant.end()));

    // 创建 CPU 前向传播
    cpu::ForwardPassQuantCPU<int8_t, int8_t, int8_t, int8_t> forward_pass(
        false, batch_size, input_size, hidden_size);
    forward_pass.setRescaleParam(quant_params);

    printf("Running fixed-point forward pass...\n");

    auto start = std::chrono::high_resolution_clock::now();

    forward_pass.Run(seq_len, W_quant.data(), R_quant.data(),
                     bx_quant.data(), br_quant.data(),
                     x_quant.data(), h_quant.data(),
                     nullptr, 0.0f, nullptr);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("---------------------------------------------------------------\n");
    printf("Results:\n");
    printf("  h_quant range: [%d, %d]\n",
           *std::min_element(h_quant.begin(), h_quant.end()),
           *std::max_element(h_quant.begin(), h_quant.end()));
    printf("  Execution time: %lld ms\n", static_cast<long long>(duration.count()));
    printf("===============================================================\n");

    return 0;
}

