// =====================================================================
// 校准辅助函数（calibration_utils.h）
// =====================================================================
// 提供量化校准所需的统计辅助函数，包括：
// - min/max 计算
// - EMA 平滑更新
// - per-channel 统计
// - 从中间值 v 提取各算子范围
// =====================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "dev_vector.h"
#include "gru_quantization_ranges.h"

// =====================================================================
// 范围处理函数
// =====================================================================

/// 确保范围不小于最小阈值，避免范围过窄导致量化精度问题
inline void ensureMinRange(float &min_val, float &max_val, float min_range_threshold = 0.1f,
                           const char *name = nullptr) {
    float range = max_val - min_val;
    if (range < min_range_threshold) {
        float center = (min_val + max_val) / 2.0f;
        [[maybe_unused]] float old_min = min_val;
        [[maybe_unused]] float old_max = max_val;
        min_val = center - min_range_threshold / 2.0f;
        max_val = center + min_range_threshold / 2.0f;
#ifdef DEBUG
        if (name) {
            printf(
                "[ensureMinRange] %s: range %.4f < %.4f, expanded [%.4f, %.4f] -> [%.4f, %.4f]\n",
                name, range, min_range_threshold, old_min, old_max, min_val, max_val);
        }
#endif
    }
}

// =====================================================================
// 基础统计函数
// =====================================================================

/// 计算向量的 min/max
template <typename T>
inline std::pair<T, T> computeMinMax(const std::vector<T> &data) {
    T min_val = data[0];
    T max_val = data[0];
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
    for (size_t i = 1; i < data.size(); ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    return {min_val, max_val};
}

/// 更新范围（取并集）
inline void updateRange(float &min_out, float &max_out, float min_val, float max_val) {
    min_out = std::min(min_out, min_val);
    max_out = std::max(max_out, max_val);
}

/// 检查范围是否已初始化
inline bool isRangeUninitialized(float min_val, float max_val) {
    return min_val == std::numeric_limits<float>::max() &&
           max_val == std::numeric_limits<float>::lowest();
}

/// 平滑更新范围（EMA）
/// 如果未初始化，直接使用当前值；否则使用 decay% 旧值 + (1-decay)% 新值
inline void updateRangeEMA(float &min_out, float &max_out, float min_val, float max_val,
                           float decay = 0.9f) {
    if (isRangeUninitialized(min_out, max_out)) {
        // 第一次更新，直接赋值
        min_out = min_val;
        max_out = max_val;
    } else {
        // 平滑更新
        min_out = decay * min_out + (1.0f - decay) * min_val;
        max_out = decay * max_out + (1.0f - decay) * max_val;
    }
}

// =====================================================================
// Host 端统计函数
// =====================================================================

/// 对 host 端数据分时间步计算 min/max 并使用 EMA 更新
template <typename T>
inline void computeMinMaxPerStepEMAHost(const std::vector<T> &data, int steps, int step_size,
                                        float &min_out, float &max_out, float decay = 0.9f) {
    for (int t = 0; t < steps; ++t) {
        const T *step_data = data.data() + t * step_size;
        T min_val = step_data[0];
        T max_val = step_data[0];
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
        for (int i = 1; i < step_size; ++i) {
            min_val = std::min(min_val, step_data[i]);
            max_val = std::max(max_val, step_data[i]);
        }
        updateRangeEMA(min_out, max_out, static_cast<float>(min_val), static_cast<float>(max_val),
                       decay);
    }
}

// =====================================================================
// Device 端统计函数（需要 CUDA）
// =====================================================================

/// 计算设备端数据的 min/max
template <typename T>
inline std::pair<T, T> computeMinMaxDev(const T *data_dev, size_t size) {
    std::vector<T> data_host = d2h(data_dev, size);
    return computeMinMax(data_host);
}

/// 分时间步计算设备端数据的 min/max 并使用 EMA 更新范围
template <typename T>
inline void computeMinMaxPerStepEMA(const T *data_dev, int steps, int step_size, float &min_out,
                                    float &max_out, float decay = 0.9f) {
    // 一次性拷贝所有数据
    std::vector<T> data_host = d2h(data_dev, steps * step_size);

    // 分时间步计算 min/max 并使用 EMA 更新
    for (int t = 0; t < steps; ++t) {
        const T *step_data = data_host.data() + t * step_size;
        T min_val = step_data[0];
        T max_val = step_data[0];
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
        for (int i = 1; i < step_size; ++i) {
            min_val = std::min(min_val, step_data[i]);
            max_val = std::max(max_val, step_data[i]);
        }
        updateRangeEMA(min_out, max_out, static_cast<float>(min_val), static_cast<float>(max_val),
                       decay);
    }
}

/// 计算 per-channel 的 min/max
template <typename T>
inline void computeMinMaxPerChannel(const T *data_dev, size_t input_size, size_t channel_size,
                                    std::vector<float> &min_out, std::vector<float> &max_out) {
    std::vector<T> data_host = d2h(data_dev, input_size * channel_size);

#pragma omp parallel for
    for (size_t c = 0; c < channel_size; ++c) {
        T min_val = data_host[c];
        T max_val = data_host[c];
        for (size_t i = 1; i < input_size; ++i) {
            const T val = data_host[i * channel_size + c];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        min_out[c] = std::min(min_out[c], static_cast<float>(min_val));
        max_out[c] = std::max(max_out[c], static_cast<float>(max_val));
    }
}

// =====================================================================
// 从中间值 v 更新范围
// =====================================================================

/// 从 v 中提取各算子的中间结果并更新量化范围
/// v 布局: [T, B, H*4] = [z, r, g, Rh_add_br_g]
template <typename T>
void updateRangesFromV(const std::vector<T> &h_host, const T *v_dev, size_t steps,
                       size_t hidden_size, size_t batch_size, GRUQuantizationRanges &quant_ranges) {
    std::vector<T> v_host = d2h(v_dev, steps * batch_size * hidden_size * 4);
    const size_t output_size = steps * batch_size * hidden_size;

    std::vector<T> z_out(output_size);
    std::vector<T> r_out(output_size);
    std::vector<T> g_out(output_size);
    std::vector<T> Rh_add_br_g(output_size);
    std::vector<T> rRh_g(output_size);
    std::vector<T> new_contrib(output_size);
    std::vector<T> old_contrib(output_size);

#pragma omp parallel for
    for (size_t t = 0; t < steps; ++t) {
        const size_t offset_v_per_step = t * batch_size * hidden_size * 4;
        for (size_t b = 0; b < batch_size; ++b) {
            const size_t offset_v_per_batch = b * hidden_size * 4;
            const size_t offset_v = offset_v_per_step + offset_v_per_batch;
            for (size_t h = 0; h < hidden_size; ++h) {
                const T z_val = v_host[offset_v + hidden_size * 0 + h];
                const T r_val = v_host[offset_v + hidden_size * 1 + h];
                const T g_val = v_host[offset_v + hidden_size * 2 + h];
                const T Rh_add_br_g_val = v_host[offset_v + hidden_size * 3 + h];
                const T rRh_g_val = r_val * Rh_add_br_g_val;
                const T one_minus_update_val = 1 - z_val;
                const T new_contrib_val = one_minus_update_val * g_val;

                const size_t offset_h = t * batch_size * hidden_size + b * hidden_size + h;
                const T h_old = h_host[offset_h];
                const T old_contrib_val = z_val * h_old;

                z_out[offset_h] = z_val;
                r_out[offset_h] = r_val;
                g_out[offset_h] = g_val;
                Rh_add_br_g[offset_h] = Rh_add_br_g_val;
                rRh_g[offset_h] = rRh_g_val;
                new_contrib[offset_h] = new_contrib_val;
                old_contrib[offset_h] = old_contrib_val;
            }
        }
    }

    // 计算并更新各中间结果的范围
#ifdef USE_EMA
    const int step_size = batch_size * hidden_size;
    computeMinMaxPerStepEMAHost(z_out, steps, step_size, quant_ranges.min_z_out_,
                                quant_ranges.max_z_out_);
    computeMinMaxPerStepEMAHost(r_out, steps, step_size, quant_ranges.min_r_out_,
                                quant_ranges.max_r_out_);
    computeMinMaxPerStepEMAHost(g_out, steps, step_size, quant_ranges.min_g_out_,
                                quant_ranges.max_g_out_);
    computeMinMaxPerStepEMAHost(Rh_add_br_g, steps, step_size, quant_ranges.min_Rh_add_br_g_,
                                quant_ranges.max_Rh_add_br_g_);
    computeMinMaxPerStepEMAHost(rRh_g, steps, step_size, quant_ranges.min_rRh_,
                                quant_ranges.max_rRh_);
    // 注意: one_minus_update 不再单独记录范围，直接复用 z_out 的 scale
    computeMinMaxPerStepEMAHost(new_contrib, steps, step_size, quant_ranges.min_new_contrib_,
                                quant_ranges.max_new_contrib_);
    computeMinMaxPerStepEMAHost(old_contrib, steps, step_size, quant_ranges.min_old_contrib_,
                                quant_ranges.max_old_contrib_);
#else
    auto [min_z, max_z] = computeMinMax(z_out);
    updateRange(quant_ranges.min_z_out_, quant_ranges.max_z_out_, min_z, max_z);

    auto [min_r, max_r] = computeMinMax(r_out);
    updateRange(quant_ranges.min_r_out_, quant_ranges.max_r_out_, min_r, max_r);

    auto [min_g, max_g] = computeMinMax(g_out);
    updateRange(quant_ranges.min_g_out_, quant_ranges.max_g_out_, min_g, max_g);

    auto [min_Rh_add_br_g, max_Rh_add_br_g] = computeMinMax(Rh_add_br_g);
    updateRange(quant_ranges.min_Rh_add_br_g_, quant_ranges.max_Rh_add_br_g_, min_Rh_add_br_g,
                max_Rh_add_br_g);

    auto [min_rRh, max_rRh] = computeMinMax(rRh_g);
    updateRange(quant_ranges.min_rRh_, quant_ranges.max_rRh_, min_rRh, max_rRh);

    // 注意: one_minus_update 不再单独记录范围，直接复用 z_out 的 scale

    auto [min_new_contrib, max_new_contrib] = computeMinMax(new_contrib);
    updateRange(quant_ranges.min_new_contrib_, quant_ranges.max_new_contrib_, min_new_contrib,
                max_new_contrib);

    auto [min_old_contrib, max_old_contrib] = computeMinMax(old_contrib);
    updateRange(quant_ranges.min_old_contrib_, quant_ranges.max_old_contrib_, min_old_contrib,
                max_old_contrib);
#endif
}

// =====================================================================
// MINMAX 量化范围更新函数（CPU 版本，需要 GPU→CPU 数据传输）
// =====================================================================

/// 根据前向传播的中间数据更新量化范围（CPU 版本）
/// 注意：此函数会将 GPU 数据拷贝到 CPU 进行计算，可能有性能开销
/// 推荐使用 GPU 版本 updateGRUQuantizationRangesGPU()
/// 
/// 输入:
///   time_steps, batch_size, input_size, hidden_size: 维度参数
///   W, R, bx, br: 权重和偏置（GPU 端）
///   x: [T, B, I] 输入序列（GPU 端）
///   h: [(T+1), B, H] 隐藏状态（GPU 端）
///   v: [T, B, H*4] 中间值（GPU 端）
///   tmp_Wx: [T, B, H*3] Wx 计算结果（GPU 端）
///   tmp_Rh: [T, B, H*3] Rh 计算结果（GPU 端）
///   z_pres, r_pres, g_pres: [T*B*H] 预激活值（GPU 端）
///   pres_size: 预激活值数组大小
/// 输出:
///   quant_ranges: 更新后的量化范围（原地更新）
inline void updateGRUQuantizationRanges(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *W, const float *R, const float *bx, const float *br,
    const float *x, const float *h, const float *v,
    const float *tmp_Wx, const float *tmp_Rh,
    const float *z_pres, const float *r_pres, const float *g_pres,
    size_t pres_size,
    GRUQuantizationRanges &quant_ranges) {
    
    const int NH = batch_size * hidden_size;
    const int NI = batch_size * input_size;

    // 设置 hidden_ 维度
    quant_ranges.hidden_ = hidden_size;

    // 输入 x 的范围（一次拷贝，分时间步平滑更新）
    computeMinMaxPerStepEMA(x, time_steps, NI, quant_ranges.min_x_, quant_ranges.max_x_);

    // 隐藏状态 h 的范围（跳过初始状态 h0，一次拷贝，分时间步平滑更新）
    computeMinMaxPerStepEMA(h + NH, time_steps, NH, quant_ranges.min_h_, quant_ranges.max_h_);

    // 权重 W 的范围（per-channel）
    computeMinMaxPerChannel(W, input_size, hidden_size * 3, quant_ranges.min_W_, quant_ranges.max_W_);

    // 权重 R 的范围（per-channel）
    computeMinMaxPerChannel(R, hidden_size, hidden_size * 3, quant_ranges.min_R_, quant_ranges.max_R_);

    // 偏置 bx 的范围（per-channel）
    computeMinMaxPerChannel(bx, 1, hidden_size * 3, quant_ranges.min_bx_, quant_ranges.max_bx_);

    // 偏置 br 的范围（per-channel）
    computeMinMaxPerChannel(br, 1, hidden_size * 3, quant_ranges.min_br_, quant_ranges.max_br_);

#ifdef USE_EMA
    // Wx 结果的范围（分时间步平滑更新）
    computeMinMaxPerStepEMA(tmp_Wx, time_steps, NH * 3, quant_ranges.min_Wx_, quant_ranges.max_Wx_);
    // Rh 结果的范围（分时间步平滑更新）
    computeMinMaxPerStepEMA(tmp_Rh, time_steps, NH * 3, quant_ranges.min_Rh_, quant_ranges.max_Rh_);
    // z 门输入的范围（分时间步平滑更新）
    computeMinMaxPerStepEMA(z_pres, time_steps, NH, quant_ranges.min_z_pre_, quant_ranges.max_z_pre_);
    // r 门输入的范围（分时间步平滑更新）
    computeMinMaxPerStepEMA(r_pres, time_steps, NH, quant_ranges.min_r_pre_, quant_ranges.max_r_pre_);
    // g 门输入的范围（分时间步平滑更新）
    computeMinMaxPerStepEMA(g_pres, time_steps, NH, quant_ranges.min_g_pre_, quant_ranges.max_g_pre_);
#else
    // Wx 结果的范围
    auto [min_Wx, max_Wx] = computeMinMaxDev(tmp_Wx, time_steps * NH * 3);
    updateRange(quant_ranges.min_Wx_, quant_ranges.max_Wx_, min_Wx, max_Wx);

    // Rh 结果的范围
    auto [min_Rh, max_Rh] = computeMinMaxDev(tmp_Rh, time_steps * NH * 3);
    updateRange(quant_ranges.min_Rh_, quant_ranges.max_Rh_, min_Rh, max_Rh);

    // z 门输入的范围
    auto [min_z_pre, max_z_pre] = computeMinMaxDev(z_pres, time_steps * NH);
    updateRange(quant_ranges.min_z_pre_, quant_ranges.max_z_pre_, min_z_pre, max_z_pre);

    // r 门输入的范围
    auto [min_r_pre, max_r_pre] = computeMinMaxDev(r_pres, time_steps * NH);
    updateRange(quant_ranges.min_r_pre_, quant_ranges.max_r_pre_, min_r_pre, max_r_pre);

    // g 门输入的范围
    auto [min_g_pre, max_g_pre] = computeMinMaxDev(g_pres, time_steps * NH);
    updateRange(quant_ranges.min_g_pre_, quant_ranges.max_g_pre_, min_g_pre, max_g_pre);
#endif

    // 从 v 中计算其他中间结果的范围
    std::vector<float> h_host = d2h(h, NH * (time_steps + 1));
    updateRangesFromV<float>(h_host, v, time_steps, hidden_size, batch_size, quant_ranges);
}

