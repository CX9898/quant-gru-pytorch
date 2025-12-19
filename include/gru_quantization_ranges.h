#pragma once

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

// GRU 量化范围结构体：存储GRU网络中每个算子计算 scale 前的 min/max 值
// 用于校准（calibration）阶段记录各算子的数值范围，便于后续分析和调试
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
    float min_z_pre_, max_z_pre_;
    float min_r_pre_, max_r_pre_;
    float min_g_pre_, max_g_pre_;

    // 门的输出值（sigmoid/tanh 输出）
    float min_z_out_, max_z_out_;
    float min_r_out_, max_r_out_;
    float min_g_out_, max_g_out_;

    // 中间计算结果
    float min_Rh_add_br_g_, max_Rh_add_br_g_;
    float min_rRh_, max_rRh_;

    // 最终输出计算
    float min_new_contrib_, max_new_contrib_;
    float min_old_contrib_, max_old_contrib_;

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
    min_z_pre_ = std::numeric_limits<float>::max();
    max_z_pre_ = std::numeric_limits<float>::lowest();
    min_r_pre_ = std::numeric_limits<float>::max();
    max_r_pre_ = std::numeric_limits<float>::lowest();
    min_g_pre_ = std::numeric_limits<float>::max();
    max_g_pre_ = std::numeric_limits<float>::lowest();
    min_z_out_ = std::numeric_limits<float>::max();
    max_z_out_ = std::numeric_limits<float>::lowest();
    min_r_out_ = std::numeric_limits<float>::max();
    max_r_out_ = std::numeric_limits<float>::lowest();
    min_g_out_ = std::numeric_limits<float>::max();
    max_g_out_ = std::numeric_limits<float>::lowest();
    min_Rh_add_br_g_ = std::numeric_limits<float>::max();
    max_Rh_add_br_g_ = std::numeric_limits<float>::lowest();
    min_rRh_ = std::numeric_limits<float>::max();
    max_rRh_ = std::numeric_limits<float>::lowest();
    min_new_contrib_ = std::numeric_limits<float>::max();
    max_new_contrib_ = std::numeric_limits<float>::lowest();
    min_old_contrib_ = std::numeric_limits<float>::max();
    max_old_contrib_ = std::numeric_limits<float>::lowest();

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
    printf("  z_pre: [%f, %f]\n", min_z_pre_, max_z_pre_);
    printf("  r_pre: [%f, %f]\n", min_r_pre_, max_r_pre_);
    printf("  g_pre: [%f, %f]\n", min_g_pre_, max_g_pre_);
    printf("  z_out: [%f, %f]\n", min_z_out_, max_z_out_);
    printf("  r_out: [%f, %f]\n", min_r_out_, max_r_out_);
    printf("  g_out: [%f, %f]\n", min_g_out_, max_g_out_);
    printf("  Rh_add_br_g: [%f, %f]\n", min_Rh_add_br_g_, max_Rh_add_br_g_);
    printf("  rRh: [%f, %f]\n", min_rRh_, max_rRh_);
    printf("  new_contrib: [%f, %f]\n", min_new_contrib_, max_new_contrib_);
    printf("  old_contrib: [%f, %f]\n", min_old_contrib_, max_old_contrib_);

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
