#pragma once

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

// GRU 量化范围结构体：存储GRU网络中每个算子计算 scale 前的 min/max 值
// 用于校准（calibration）阶段记录各算子的数值范围，便于后续分析和调试
//
// 命名约定（与 optimized_quantizable_gru_2.md 文档对齐）：
//   - weight_ih_linear: W*x + bx 的输出
//   - weight_hh_linear: R*h + br 的输出
//   - reset_gate_input/output: reset gate 的输入/输出
//   - update_gate_input/output: update gate 的输入/输出
//   - new_gate_input/output: new gate 的输入/输出
//   - mul_reset_hidden: r * h_n 的输出
//   - mul_new_contribution: (1-u) * n 的输出
//   - mul_old_contribution: u * h 的输出
struct GRUQuantizationRanges {
    // 默认构造函数：自动初始化所有范围为无效值
    GRUQuantizationRanges() : hidden_(0) { reset(); }

    // 带 hidden 参数的构造函数：初始化并设置 per-channel 向量大小
    explicit GRUQuantizationRanges(int hidden) : hidden_(hidden) { reset(); }

    int hidden_;  // channel = hidden * 3

    // 输入和隐藏状态
    float min_x_, max_x_;
    float min_h_, max_h_;

    // 权重矩阵（per-channel，每个输出通道一个范围）
    std::vector<float> min_W_, max_W_;  // size = hidden * 3
    std::vector<float> min_R_, max_R_;  // size = hidden * 3

    // 矩阵乘法结果
    float min_Wx_, max_Wx_;
    float min_Rh_, max_Rh_;

    // 偏置（per-channel）
    std::vector<float> min_bx_, max_bx_;  // size = hidden * 3
    std::vector<float> min_br_, max_br_;  // size = hidden * 3

    // 门的预激活值（sigmoid/tanh 输入）
    float min_update_gate_input_, max_update_gate_input_;
    float min_reset_gate_input_, max_reset_gate_input_;
    float min_new_gate_input_, max_new_gate_input_;

    // 门的输出值（sigmoid/tanh 输出）
    float min_update_gate_output_, max_update_gate_output_;
    float min_reset_gate_output_, max_reset_gate_output_;
    float min_new_gate_output_, max_new_gate_output_;

    // 中间计算结果
    float min_Rh_add_br_g_, max_Rh_add_br_g_;
    float min_mul_reset_hidden_, max_mul_reset_hidden_;

    // 最终输出计算
    float min_mul_new_contribution_, max_mul_new_contribution_;
    float min_mul_old_contribution_, max_mul_old_contribution_;

    // 重置所有范围为无效值
    // 如果传入 hidden > 0，则更新 hidden_ 并重新分配 per-channel 向量
    // 如果不传参数，则使用当前的 hidden_ 值
    void reset(int hidden = -1);

    // 打印所有范围信息
    void print() const;
};

// ==================== 方法实现 ====================

inline void GRUQuantizationRanges::reset(int hidden) {
    // 如果传入有效的 hidden 值，则更新 hidden_
    if (hidden > 0) {
        hidden_ = hidden;
    }

    // 重置所有标量范围
    min_x_ = std::numeric_limits<float>::max();
    max_x_ = std::numeric_limits<float>::lowest();
    min_h_ = std::numeric_limits<float>::max();
    max_h_ = std::numeric_limits<float>::lowest();
    min_Wx_ = std::numeric_limits<float>::max();
    max_Wx_ = std::numeric_limits<float>::lowest();
    min_Rh_ = std::numeric_limits<float>::max();
    max_Rh_ = std::numeric_limits<float>::lowest();
    min_update_gate_input_ = std::numeric_limits<float>::max();
    max_update_gate_input_ = std::numeric_limits<float>::lowest();
    min_reset_gate_input_ = std::numeric_limits<float>::max();
    max_reset_gate_input_ = std::numeric_limits<float>::lowest();
    min_new_gate_input_ = std::numeric_limits<float>::max();
    max_new_gate_input_ = std::numeric_limits<float>::lowest();
    min_update_gate_output_ = std::numeric_limits<float>::max();
    max_update_gate_output_ = std::numeric_limits<float>::lowest();
    min_reset_gate_output_ = std::numeric_limits<float>::max();
    max_reset_gate_output_ = std::numeric_limits<float>::lowest();
    min_new_gate_output_ = std::numeric_limits<float>::max();
    max_new_gate_output_ = std::numeric_limits<float>::lowest();
    min_Rh_add_br_g_ = std::numeric_limits<float>::max();
    max_Rh_add_br_g_ = std::numeric_limits<float>::lowest();
    min_mul_reset_hidden_ = std::numeric_limits<float>::max();
    max_mul_reset_hidden_ = std::numeric_limits<float>::lowest();
    min_mul_new_contribution_ = std::numeric_limits<float>::max();
    max_mul_new_contribution_ = std::numeric_limits<float>::lowest();
    min_mul_old_contribution_ = std::numeric_limits<float>::max();
    max_mul_old_contribution_ = std::numeric_limits<float>::lowest();

    // 重置 per-channel 向量
    if (hidden_ > 0) {
        const int channel_size = hidden_ * 3;
        min_W_.assign(channel_size, std::numeric_limits<float>::max());
        max_W_.assign(channel_size, std::numeric_limits<float>::lowest());
        min_R_.assign(channel_size, std::numeric_limits<float>::max());
        max_R_.assign(channel_size, std::numeric_limits<float>::lowest());
        min_bx_.assign(channel_size, std::numeric_limits<float>::max());
        max_bx_.assign(channel_size, std::numeric_limits<float>::lowest());
        min_br_.assign(channel_size, std::numeric_limits<float>::max());
        max_br_.assign(channel_size, std::numeric_limits<float>::lowest());
    }
}

inline void GRUQuantizationRanges::print() const {
    printf("GRUQuantizationRanges (量化范围):\n");
    printf("  hidden_ = %d\n", hidden_);
    printf("  x: [%f, %f]\n", min_x_, max_x_);
    printf("  h: [%f, %f]\n", min_h_, max_h_);
    printf("  Wx: [%f, %f]\n", min_Wx_, max_Wx_);
    printf("  Rh: [%f, %f]\n", min_Rh_, max_Rh_);
    printf("  update_gate_input: [%f, %f]\n", min_update_gate_input_, max_update_gate_input_);
    printf("  reset_gate_input: [%f, %f]\n", min_reset_gate_input_, max_reset_gate_input_);
    printf("  new_gate_input: [%f, %f]\n", min_new_gate_input_, max_new_gate_input_);
    printf("  update_gate_output: [%f, %f]\n", min_update_gate_output_, max_update_gate_output_);
    printf("  reset_gate_output: [%f, %f]\n", min_reset_gate_output_, max_reset_gate_output_);
    printf("  new_gate_output: [%f, %f]\n", min_new_gate_output_, max_new_gate_output_);
    printf("  Rh_add_br_g: [%f, %f]\n", min_Rh_add_br_g_, max_Rh_add_br_g_);
    printf("  mul_reset_hidden: [%f, %f]\n", min_mul_reset_hidden_, max_mul_reset_hidden_);
    printf("  mul_new_contribution: [%f, %f]\n", min_mul_new_contribution_, max_mul_new_contribution_);
    printf("  mul_old_contribution: [%f, %f]\n", min_mul_old_contribution_, max_mul_old_contribution_);

    // 打印 per-channel 向量的前几个值
    if (!min_W_.empty()) {
        printf("  W (per-channel, first 5): ");
        for (size_t i = 0; i < std::min(size_t(5), min_W_.size()); ++i) {
            printf("[%f,%f] ", min_W_[i], max_W_[i]);
        }
        if (min_W_.size() > 5) printf("...");
        printf("\n");
    }
    if (!min_R_.empty()) {
        printf("  R (per-channel, first 5): ");
        for (size_t i = 0; i < std::min(size_t(5), min_R_.size()); ++i) {
            printf("[%f,%f] ", min_R_[i], max_R_[i]);
        }
        if (min_R_.size() > 5) printf("...");
        printf("\n");
    }
}
