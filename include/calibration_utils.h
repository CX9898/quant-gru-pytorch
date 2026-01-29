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


