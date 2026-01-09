#pragma once

/**
 * POT 量化校准模块
 * 
 * 四个独立模块（均为独立函数，无状态类）：
 * 1. 直方图收集 - 在 histogram_collector.h
 * 2. Percentile 校准 - calibratePercentile()
 * 3. SQNR 校准 - calibrateSqnr()
 * 4. POT 转换 - convertToPot(), roundScaleToPowerOfTwo(), applyBitwidthConstraint()
 * 
 * 调用流程：
 * Histogram → [calibratePercentile/calibrateSqnr] → ContinuousScaleResult → [convertToPot] → PotScaleResult
 * 
 * 统一入口：calibrateQuantParamsFromHistogram()
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "histogram_collector.h"
#include "quantize_bitwidth_config.h"  // for QuantBitWidth

// ============================================================================
// 公共数据结构
// ============================================================================

/**
 * 连续 scale 校准结果（模块 2/3 的输出）
 */
struct ContinuousScaleResult {
    float scale;      // 连续 scale
    float min;        // 量化范围最小值
    float max;        // 量化范围最大值
    float noise;      // 估计噪声（仅 SQNR 使用）
};

/**
 * POT scale 结果（模块 4 的输出）
 */
struct PotScaleResult {
    int8_t exp2_inv;   // POT 指数 (scale = 2^(-exp2_inv))
    float po2_scale;   // POT scale 值
    int32_t zero_point; // 零点
};

// ============================================================================
// 模块 2: Percentile 校准
// ============================================================================

/**
 * Percentile 校准配置
 */
struct PercentileConfig {
    float percentile = 99.99f;  // 百分位数 (如 99.99 表示保留 99.99% 数据)
};

/**
 * Percentile 校准：基于百分位数裁剪计算连续 scale
 * 
 * 与 AIMET PercentileEncodingAnalyzer.compute_encodings_from_stats + adjust_min_max 完全一致
 * 
 * @param hist 直方图数据
 * @param num_steps 量化级数 (quant_max - quant_min)
 * @param is_symmetric 是否对称量化
 * @param config 配置参数
 * @param is_unsigned 是否无符号量化（UINT）
 * @return ContinuousScaleResult
 */
inline ContinuousScaleResult calibratePercentile(
    const Histogram& hist,
    int64_t num_steps,
    bool is_symmetric,
    const PercentileConfig& config = PercentileConfig(),
    bool is_unsigned = false) {
    
    if (!hist.is_valid()) {
        throw std::runtime_error("Histogram is invalid in calibratePercentile");
    }
    
    // 与 AIMET 一致：检查 num_steps
    if (num_steps <= 0) {
        throw std::runtime_error("num_steps must be > 0 in calibratePercentile");
    }
    
    // 与 AIMET _get_minimum_scale 一致
    const float minimum_scale = get_minimum_scale(static_cast<int>(num_steps));
    
    // 获取百分位数范围
    float clip_ratio = (100.0f - config.percentile) / 100.0f;
    auto [pmin, pmax] = hist.getPercentileRange(clip_ratio);
    
    // 与 AIMET adjust_min_max 一致：确保范围包含 0
    pmin = std::min(pmin, 0.0f);
    pmax = std::max(pmax, 0.0f);
    
    // 与 AIMET adjust_min_max 一致：确保 finite（clamp 到合理范围）
    constexpr float float_max = std::numeric_limits<float>::max();
    constexpr float float_min = std::numeric_limits<float>::lowest();
    pmin = std::max(float_min, std::min(pmin, 0.0f));
    pmax = std::min(float_max, std::max(pmax, 0.0f));
    
    // 与 AIMET adjust_min_max 一致：确保范围不太小（使用 minimum_scale）
    float tensor_threshold = (pmax - pmin) / static_cast<float>(num_steps);
    if (tensor_threshold < minimum_scale) {
        if (is_symmetric && !is_unsigned) {
            // INT 对称量化：两边扩展
            int64_t num_neg_steps = (num_steps + 1) / 2;  // ceil
            int64_t num_pos_steps = num_steps / 2;        // floor
            pmin -= minimum_scale * static_cast<float>(num_neg_steps);
            pmax += minimum_scale * static_cast<float>(num_pos_steps);
        } else {
            // 非对称量化 或 UINT 对称量化：只扩展 max
            pmax += minimum_scale * static_cast<float>(num_steps);
        }
    }
    
    ContinuousScaleResult result;
    
    if (is_symmetric) {
        if (is_unsigned) {
            // UINT + symmetric: 数据范围 [0, max]，量化范围 [0, num_steps]
            float data_max = std::max(pmax, minimum_scale);
            result.scale = data_max / static_cast<float>(num_steps);
            result.scale = std::max(result.scale, minimum_scale);
            result.min = 0.0f;
            result.max = result.scale * static_cast<float>(num_steps);
        } else {
            // INT + symmetric: 与 AIMET adjust_min_max 对称量化处理完全一致
            int64_t num_pos_steps = num_steps / 2;        // floor
            int64_t num_neg_steps = (num_steps + 1) / 2;  // ceil
            
            // 边缘情况：num_steps=1 时 num_pos_steps=0，需要防止除零
            // AIMET 在这种情况下会得到 inf，但最终会被 minimum_scale 约束
            float delta_from_max = (num_pos_steps > 0) 
                ? pmax / static_cast<float>(num_pos_steps) 
                : std::numeric_limits<float>::max();
            float delta_from_min = (num_neg_steps > 0) 
                ? -pmin / static_cast<float>(num_neg_steps) 
                : std::numeric_limits<float>::max();
            
            result.scale = std::max(delta_from_max, delta_from_min);
            result.min = -static_cast<float>(num_neg_steps) * result.scale;
            result.max = static_cast<float>(num_pos_steps) * result.scale;
        }
    } else {
        result.scale = (pmax - pmin) / static_cast<float>(num_steps);
        result.min = pmin;
        result.max = pmax;
    }
    
    // 最终保护：确保 scale 不小于 minimum_scale（与 AIMET 一致）
    result.scale = std::max(result.scale, minimum_scale);
    result.noise = 0.0f;  // Percentile 不计算噪声
    
    return result;
}

// ============================================================================
// 模块 3: SQNR 校准
// ============================================================================

/**
 * SQNR 校准配置
 */
struct SqnrConfig {
    int symmetric_delta_candidates = 101;   // AIMET 默认 101
    int asymmetric_delta_candidates = 17;   // AIMET 默认 17
    int offset_candidates = 21;             // AIMET 默认 21
    float gamma = 3.0f;                     // AIMET 默认 3.0
    float p = 2.0f;                         // Lp 范数 (p=2 = MSE)
};

namespace sqnr_detail {

/**
 * 估算量化噪声（与 AIMET _estimate_clip_and_quant_noise 完全一致）
 */
inline float estimateNoise(const Histogram& hist, float delta, float offset,
                           int64_t num_steps, float gamma, float p) {
    if (delta <= 0) return std::numeric_limits<float>::max();
    
    float bin_width = hist.bin_width();
    float total_noise = 0.0f;
    
    for (int i = 0; i < hist.num_bins; ++i) {
        float count = hist.counts[i];
        if (count < 1e-6f) continue;
        
        float x = hist.min_val + (i + 0.5f) * bin_width;
        
        // AIMET 公式：q = round(x / delta - offset)
        float q = std::round(x / delta - offset);
        
        bool clipped = (q < 0) || (q > static_cast<float>(num_steps));
        q = std::max(0.0f, std::min(static_cast<float>(num_steps), q));
        float x_recon = (q + offset) * delta;
        
        float error = std::pow(std::abs(x_recon - x), p);
        if (clipped && gamma != 1.0f) error *= gamma;
        
        total_noise += error * count;
    }
    return total_noise;
}

/**
 * 对称量化搜索（与 AIMET _pick_test_candidates_symmetric 完全一致）
 * 
 * @param is_unsigned 是否 UINT 类型
 */
inline ContinuousScaleResult searchSymmetric(
    const Histogram& hist,
    float min_val, float max_val,
    int64_t num_steps,
    const SqnrConfig& config,
    bool is_unsigned = false) {
    
    const float minimum_scale = get_minimum_scale(static_cast<int>(num_steps));
    
    ContinuousScaleResult best{0, 0, 0, std::numeric_limits<float>::max()};
    
    float max_delta;
    float offset;
    
    if (is_unsigned) {
        // UINT + symmetric: 数据范围 [0, max]，量化范围 [0, num_steps]
        // offset = 0 (zp = 0)
        max_delta = max_val / static_cast<float>(num_steps);
        offset = 0.0f;
    } else {
        // INT + symmetric: 对称量化，offset = (-num_steps) // 2
        max_delta = 2.0f * std::max(max_val, -min_val) / static_cast<float>(num_steps);
        offset = -static_cast<float>((num_steps + 1) / 2);
    }
    
    // 边缘保护：确保 max_delta 不为 0（避免 SQNR 搜索退化）
    max_delta = std::max(max_delta, minimum_scale);
    
    // 边缘保护：确保除数不为 0
    const int divisor = std::max(config.symmetric_delta_candidates - 1, 1);
    
    for (int d = 1; d <= config.symmetric_delta_candidates; ++d) {
        float delta = max_delta * d / divisor;
        delta = std::max(delta, minimum_scale);
        
        float noise = estimateNoise(hist, delta, offset, num_steps, config.gamma, config.p);
        
        if (noise < best.noise) {
            best.noise = noise;
            best.scale = delta;
            best.min = offset * delta;
            best.max = best.min + num_steps * delta;
        }
    }
    
    if (best.noise == std::numeric_limits<float>::max()) {
        throw std::runtime_error("calibrateSqnr: searchSymmetric failed to find valid scale");
    }
    return best;
}

/**
 * 非对称量化搜索（与 AIMET _pick_test_candidates_asymmetric 完全一致）
 */
inline ContinuousScaleResult searchAsymmetric(
    const Histogram& hist,
    float min_val, float max_val,
    int64_t num_steps,
    const SqnrConfig& config) {
    
    float max_delta = (max_val - min_val) / static_cast<float>(num_steps);
    
    // 计算 observed_min/observed_max（量化对齐的范围）
    float observed_offset = std::round(min_val / max_delta);
    float observed_min = max_delta * observed_offset;
    float observed_max = observed_min + max_delta * static_cast<float>(num_steps);
    
    const float minimum_scale = get_minimum_scale(static_cast<int>(num_steps));
    
    // 生成 offset 候选值
    const int num_offsets = std::min(static_cast<int>(num_steps + 2), config.offset_candidates);
    std::vector<float> offsets(num_offsets);
    float offset_step = static_cast<float>(num_steps) / (num_offsets - 2);
    for (int o = 0; o < num_offsets - 1; ++o) {
        offsets[o] = std::round(-static_cast<float>(num_steps) + o * offset_step);
    }
    offsets[num_offsets - 1] = observed_offset;
    
    ContinuousScaleResult best{0, 0, 0, std::numeric_limits<float>::max()};
    
    for (int d = 1; d <= config.asymmetric_delta_candidates; ++d) {
        float delta = max_delta * d / (config.asymmetric_delta_candidates - 1);
        delta = std::max(delta, minimum_scale);
        
        for (int o = 0; o < num_offsets; ++o) {
            float off = offsets[o];
            
            // 与 AIMET _clamp_delta_offset_values 完全一致
            float test_min = delta * off;
            float test_max = test_min + delta * static_cast<float>(num_steps);
            test_min = std::max(observed_min, test_min);
            test_max = std::min(observed_max, test_max);
            float clamped_delta = (test_max - test_min) / static_cast<float>(num_steps);
            clamped_delta = std::max(clamped_delta, minimum_scale);
            float clamped_offset = std::round(test_min / clamped_delta);
            
            float noise = estimateNoise(hist, clamped_delta, clamped_offset, num_steps,
                                       config.gamma, config.p);
            
            if (noise < best.noise) {
                best.noise = noise;
                best.scale = clamped_delta;
                best.min = clamped_offset * clamped_delta;
                best.max = best.min + num_steps * clamped_delta;
            }
        }
    }
    
    if (best.noise == std::numeric_limits<float>::max()) {
        throw std::runtime_error("calibrateSqnr: searchAsymmetric failed to find valid scale");
    }
    return best;
}

}  // namespace sqnr_detail

/**
 * SQNR 校准：基于 AIMET SqnrEncodingAnalyzer 的 SQNR 优化搜索
 * 
 * @param hist 直方图数据
 * @param num_steps 量化级数 (quant_max - quant_min)
 * @param is_symmetric 是否对称量化
 * @param config 配置参数
 * @param is_unsigned 是否无符号量化（UINT）
 * @return ContinuousScaleResult
 */
inline ContinuousScaleResult calibrateSqnr(
    const Histogram& hist,
    int64_t num_steps,
    bool is_symmetric,
    const SqnrConfig& config = SqnrConfig(),
    bool is_unsigned = false) {
    
    if (!hist.is_valid()) {
        throw std::runtime_error("Histogram is invalid in calibrateSqnr");
    }
    
    // 与 AIMET _pick_test_candidates 完全一致的范围预处理
    const float minimum_scale = get_minimum_scale(static_cast<int>(num_steps));
    
    // 确保范围包含 0
    float min_val = std::min(hist.min_val, 0.0f);
    float max_val = std::max(hist.max_val, 0.0f);
    
    // 确保范围有效（与 AIMET 一致）
    float min_range_limit = min_val + minimum_scale * static_cast<float>(num_steps);
    max_val = (max_val > min_range_limit) ? max_val : min_range_limit;
    
    if (is_symmetric) {
        return sqnr_detail::searchSymmetric(hist, min_val, max_val, num_steps, config, is_unsigned);
    } else {
        return sqnr_detail::searchAsymmetric(hist, min_val, max_val, num_steps, config);
    }
}

// ============================================================================
// 模块 4: POT 转换工具函数
// ============================================================================

/**
 * 转换连续 scale 为 POT（AIMET find_closest_power_of_2_scale）
 * 
 * @param scale 连续 scale
 * @return {po2_scale, exp2_inv} 其中 po2_scale = 2^(-exp2_inv)
 */
inline std::pair<float, int8_t> roundScaleToPowerOfTwo(float scale) {
    if (scale <= 0) {
        throw std::runtime_error("Invalid scale <= 0 in roundScaleToPowerOfTwo");
    }
    float n = -std::log2(scale);
    int8_t n_rounded = static_cast<int8_t>(std::round(n));
    return {std::pow(2.0f, -static_cast<float>(n_rounded)), n_rounded};
}

/**
 * 计算 zero-point
 * 
 * @param continuous_min 量化范围最小值
 * @param po2_scale POT scale
 * @param quant_min 量化最小值
 * @param is_symmetric 是否对称量化
 * @return zero-point
 */
inline int32_t computeZeroPoint(float continuous_min, float po2_scale, 
                                int64_t quant_min, bool is_symmetric) {
    if (is_symmetric) {
        return 0;
    } else {
        float zp_fp = static_cast<float>(quant_min) - continuous_min / po2_scale;
        return static_cast<int32_t>(std::round(zp_fp));
    }
}

/**
 * 转换连续 scale 为 POT 形式（与 AIMET 一致）
 * 
 * 策略：
 * 1. 四舍五入 scale 到最近的 2^n
 * 2. 保持 qmin, qmax 不变
 * 3. 计算 zero-point
 * 
 * 注意：与 AIMET 一致，不做位宽约束。如果 POT scale 变小，
 * 硬件需要支持饱和处理（saturate）或调用方需要调整 rmin/rmax。
 * 
 * @param continuous_scale 连续 scale
 * @param continuous_min 连续 min（用于计算 zero-point）
 * @param bw 量化位宽配置
 * @param is_symmetric 是否对称量化
 * @return PotScaleResult
 */
inline PotScaleResult convertToPot(
    float continuous_scale,
    float continuous_min,
    QuantBitWidth bw,
    bool is_symmetric) {
    
    const int64_t quant_min = bw.qmin();
    
    // 步骤 1: 转换到 POT（AIMET find_closest_power_of_2_scale）
    auto [po2_scale, n] = roundScaleToPowerOfTwo(continuous_scale);
    
    // 步骤 2: 计算 zero-point
    int32_t zp = computeZeroPoint(continuous_min, po2_scale, quant_min, is_symmetric);
    
    return PotScaleResult{n, po2_scale, zp};
}

// ============================================================================
// 统一入口函数
// ============================================================================

/**
 * 从直方图计算 POT 量化参数
 * 
 * 调用流程: Histogram → [calibratePercentile/calibrateSqnr] → [convertToPot] → 最终参数
 * 
 * @param hist 直方图数据
 * @param bw 量化位宽配置（包含 is_unsigned_ 信息）
 * @param is_symmetric 是否对称量化
 * @param exp2_inv [out] POT 指数
 * @param zp [out] 零点
 * @param name 调试名称（可选）
 * @param use_percentile 使用 Percentile 校准（false = SQNR）
 * @param percentile 百分位数（仅当 use_percentile=true 时有效）
 */
inline void calibrateQuantParamsFromHistogram(
    const Histogram& hist,
    QuantBitWidth bw,
    bool is_symmetric,
    int8_t& exp2_inv,
    int32_t& zp,
    const char* name = nullptr,
    bool use_percentile = false,
    float percentile = 99.99f) {
    
    if (!hist.is_valid()) {
        throw std::runtime_error("Histogram is invalid in calibrateQuantParamsFromHistogram");
    }
    
    const int64_t num_steps = bw.qmax() - bw.qmin();
    const bool is_unsigned = bw.is_unsigned_;
    
    // 步骤 1: 使用对应的校准方法计算连续 scale
    ContinuousScaleResult continuous_result;
    
    if (use_percentile) {
        PercentileConfig config;
        config.percentile = percentile;
        continuous_result = calibratePercentile(hist, num_steps, is_symmetric, config, is_unsigned);
    } else {
        SqnrConfig config;
        continuous_result = calibrateSqnr(hist, num_steps, is_symmetric, config, is_unsigned);
    }
    
    // 步骤 2: 转换为 POT（与 AIMET 一致，无位宽约束）
    PotScaleResult pot_result = convertToPot(
        continuous_result.scale,
        continuous_result.min,
        bw,
        is_symmetric);
    
    exp2_inv = pot_result.exp2_inv;
    zp = pot_result.zero_point;
    
#ifdef DEBUG
    if (name && name[0]) {
        const char* scheme_name = use_percentile ? "PERC" : "SQNR";
        printf("[%s][%s] unsigned=%d range=[%.4f,%.4f] cont_scale=%.6f po2=%.6f(1/2^%d) zp=%d\n",
               scheme_name, name, is_unsigned, hist.min_val, hist.max_val, 
               continuous_result.scale, pot_result.po2_scale, pot_result.exp2_inv, pot_result.zero_point);
    }
#endif
}

