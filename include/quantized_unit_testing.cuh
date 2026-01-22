#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "check_data.h"

// 梯度输出结构体
struct GRUTrainGradients {
    std::vector<float> dx;   // 输入序列梯度 [time_steps * batch_size * input_size]
    std::vector<float> dW;   // 对输入权重的梯度 [input_size * hidden_size * 3]
    std::vector<float> dR;   // 对循环权重的梯度 [hidden_size * hidden_size * 3]
    std::vector<float> dbw;  // 对输入偏置的梯度 [hidden_size * 3]
    std::vector<float> dbr;  // 对循环偏置的梯度 [hidden_size * 3]
    std::vector<float> dh;   // 对最后隐藏状态的梯度 [batch_size * hidden_size]
    std::vector<float> v;    // V中间值 [time_steps * batch_size * hidden_size * 4]
    std::vector<float> h;    // 隐藏状态 [time_steps * batch_size * hidden_size] (不包含初始状态)
};

/**
 * @brief 比较浮点和量化版本的V中间值
 * @param v_float 浮点版本的V中间值
 * @param v_quant_dequant 量化后反量化的V中间值
 * @param time_steps 时间步数
 * @param batch_size 批次大小
 * @param hidden_size 隐藏层维度
 * @param prefix 输出前缀（可选）
 */
inline void compareVIntermediateValues(const std::vector<float> &v_float,
                                       const std::vector<float> &v_quant_dequant, int time_steps,
                                       int batch_size, int hidden_size,
                                       const std::string &prefix = "") {
    printf("\n========== %s V Intermediate Values Comparison ==========\n", prefix.c_str());

    const int v_size_per_step =
        batch_size * hidden_size * 4;  // 4个部分：z_out, r_out, g_out, Rh_add_br
    const int v_size_per_part = batch_size * hidden_size;  // 每个部分的大小

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
    for (int t = 0; t < time_steps && t < 10; ++t) {  // 只显示前10个时间步
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

/**
 * @brief 比较浮点和量化版本的h隐藏状态（不包含初始状态）
 * @param h_float 浮点版本的h隐藏状态，size = time_steps * batch_size *
 * hidden_size（不包含初始状态t=0）
 * @param h_quant_dequant 量化后反量化的h隐藏状态，size同上
 * @param time_steps 时间步数
 * @param batch_size 批次大小
 * @param hidden_size 隐藏层维度
 * @param prefix 输出前缀（可选）
 */
inline void compareHValues(const std::vector<float> &h_float,
                           const std::vector<float> &h_quant_dequant, int time_steps,
                           int batch_size, int hidden_size, const std::string &prefix = "") {
    printf("\n========== %s H Hidden States Comparison ==========\n", prefix.c_str());

    const int h_size_per_step = batch_size * hidden_size;  // 每个时间步的大小

    // 验证大小
    if (h_quant_dequant.size() != h_float.size()) {
        printf(
            "[Error] h_float and h_quant_dequant size mismatch: h_float_size = %zu, "
            "h_quant_dequant_size = %zu\n",
            h_float.size(), h_quant_dequant.size());
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
    for (int t = 0; t < time_steps; ++t) {
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
    for (int b = 0; b < batch_size && b < 5; ++b) {  // 只显示前5个批次
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

/**
 * @brief 比较两个GRU训练梯度的差异
 * @param gradients_float 浮点版本的梯度
 * @param gradients_quant 量化版本的梯度
 * @param prefix 输出前缀（可选）
 */
inline void compareGRUTrainGradients(const GRUTrainGradients &gradients_float,
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

    // 比较 dbw
    {
        const float mse = computeMSE(gradients_float.dbw, gradients_quant.dbw);
        const float cos_sim = computeCosineSimilarity(gradients_float.dbw, gradients_quant.dbw);
        printf("dbw: MSE = %e, Cosine Similarity = %f\n", mse, cos_sim);
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
