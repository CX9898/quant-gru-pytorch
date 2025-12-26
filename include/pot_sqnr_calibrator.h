#pragma once

/**
 * AIMET-Style POT-SQNR Calibrator
 * 
 * 基于 AIMET SqnrEncodingAnalyzer 的三阶段量化校准：
 * 
 * - 阶段 1：SQNR 优化找到最优连续 scale（与 AIMET 一致）
 *   - 搜索 delta/offset 候选值，最小化量化噪声
 *   - gamma 参数权衡 clipping noise
 * 
 * - 阶段 2：转换到 POT（Power-of-Two）
 *   - AIMET 使用 round(-log2(scale))
 *   - 本实现使用 floor(-log2(scale))，确保 po2_scale >= optimal_scale，避免 clipping
 * 
 * - 阶段 3：计算 zero-point
 *   - AIMET 使用 optimal_min
 *   - 本实现使用 hist.min_val，确保覆盖所有数据
 * 
 * 核心参数：
 * - symmetric_delta_candidates = 201
 * - asymmetric_delta_candidates = 35
 * - offset_candidates = 31
 * - gamma = 3.0
 * - num_bins = 2048
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "histogram_collector.h"

/**
 * 校准方案枚举
 */
enum class CalibrationScheme {
    SQNR,        // AIMET tf_enhanced 风格：SQNR 优化搜索最优 scale
    PERCENTILE   // AIMET percentile 风格：百分位数裁剪
};

/**
 * 直方图校准配置（统一配置，支持 SQNR 和 Percentile 两种方案）
 */
struct HistogramCalibrationConfig {
    // ========== 通用配置 ==========
    int num_bins = 2048;                    // 直方图 bin 数量
    CalibrationScheme scheme = CalibrationScheme::SQNR;  // 校准方案
    
    // ========== SQNR 方案配置 ==========
    int symmetric_delta_candidates = 201;   // 对称量化搜索精度
    int asymmetric_delta_candidates = 35;   // 非对称量化搜索精度
    int offset_candidates = 31;             // offset 搜索精度
    float gamma = 3.0f;                     // clipping noise 权重 (AIMET 默认 3.0)
    float p = 2.0f;                         // Lp 范数 (p=2 = MSE)
    
    // ========== Percentile 方案配置 ==========
    float percentile = 99.99f;              // 百分位数 (如 99.99 表示保留 99.99% 数据)
};

/**
 * 连续 scale 校准结果
 */
struct ContinuousCalibrationResult {
    float optimal_scale;
    float optimal_offset;
    float optimal_min;
    float optimal_max;
    float best_noise;
};

/**
 * AIMET 风格的 POT-SQNR 校准器
 */
class AimetPotSqnrCalibrator {
   public:
    explicit AimetPotSqnrCalibrator(HistogramCalibrationConfig config = HistogramCalibrationConfig())
        : config_(config), collector_(config.num_bins) {}

    void update(const float* data, size_t size) {
        if (size > 0) collector_.collect(data, size);
    }

    void merge(const Histogram& other) { collector_.merge(other); }

    void reset() { collector_.reset(); }

    bool isValid() const { return collector_.is_valid(); }

    const Histogram& getHistogram() const { return collector_.histogram(); }

    std::pair<float, float> getRange() const {
        const Histogram& hist = collector_.histogram();
        return {hist.min_val, hist.max_val};
    }

    /**
     * 计算最优 POT 量化参数（两阶段方法）
     */
    template <typename QuantT>
    void computeOptimalParams(bool is_symmetric, int8_t& out_exp2_inv, int32_t& out_zp,
                              const char* name = nullptr) {
        computeOptimalParamsFromHistogram<QuantT>(collector_.histogram(), is_symmetric,
                                                   out_exp2_inv, out_zp, name, config_);
    }

    /**
     * 统一的 POT 量化参数计算（支持 SQNR 和 Percentile 两种方案）
     * 
     * 流程：
     * 1. 根据 scheme 选择计算最优连续 scale 和 min 的方式
     *    - SQNR: 搜索最优 delta/offset，最小化量化噪声
     *    - PERCENTILE: 直接从百分位数范围计算
     * 2. round 到最近的 POT（AIMET find_closest_power_of_2_scale）
     * 3. 计算 zero-point
     */
    template <typename QuantT>
    static void computeOptimalParamsFromHistogram(const Histogram& hist, bool is_symmetric,
                                                  int8_t& out_exp2_inv, int32_t& out_zp,
                                                  const char* name = nullptr,
                                                  const HistogramCalibrationConfig& config = HistogramCalibrationConfig()) {
        if (!hist.is_valid()) {
            throw std::runtime_error("Histogram is invalid in computeOptimalParamsFromHistogram");
        }

        const int64_t quant_min = static_cast<int64_t>(std::numeric_limits<QuantT>::min());
        const int64_t quant_max = static_cast<int64_t>(std::numeric_limits<QuantT>::max());
        const int64_t num_steps = quant_max - quant_min;

        float optimal_scale, optimal_min;

        if (config.scheme == CalibrationScheme::PERCENTILE) {
            // ========== Percentile 方案 ==========
            // 从直方图获取百分位数范围
            float clip_ratio = (100.0f - config.percentile) / 100.0f;
            auto [pmin, pmax] = hist.getPercentileRange(clip_ratio);
            
            // 确保范围包含 0（与 AIMET 一致）
            pmin = std::min(pmin, 0.0f);
            pmax = std::max(pmax, 0.0f);
            
            // 确保范围有效
            if (pmax <= pmin) {
                pmax = pmin + 1e-6f;
            }
            
            if (is_symmetric) {
                float abs_max = std::max(std::abs(pmin), std::abs(pmax));
                optimal_scale = 2.0f * abs_max / static_cast<float>(num_steps);
                optimal_min = -abs_max;
            } else {
                optimal_scale = (pmax - pmin) / static_cast<float>(num_steps);
                optimal_min = pmin;
            }
            
            // 确保 scale 有效
            optimal_scale = std::max(optimal_scale, 1e-8f);
        } else {
            // ========== SQNR 方案（默认）==========
            auto result = computeOptimalContinuousScale<QuantT>(hist, is_symmetric, config);
            optimal_scale = result.optimal_scale;
            optimal_min = result.optimal_min;
        }

        // 阶段 2：round 到最近的 POT（AIMET 方式）
        auto [po2_scale, n] = roundToPowerOfTwo(optimal_scale);
        
        out_exp2_inv = n;
        
        // 阶段 3：计算 zp
        if (is_symmetric) {
            out_zp = 0;
        } else {
            float zp_fp = static_cast<float>(quant_min) - optimal_min / po2_scale;
            out_zp = static_cast<int32_t>(std::round(zp_fp));
        }

#ifdef DEBUG
        if (name && name[0]) {
            const char* scheme_name = (config.scheme == CalibrationScheme::PERCENTILE) ? "PERC" : "SQNR";
            printf("[AIMET-POT][%s][%s] range=[%.4f,%.4f] opt_min=%.4f cont_scale=%.6f po2=%.6f(1/2^%d) zp=%d\n",
                   scheme_name, name, hist.min_val, hist.max_val, optimal_min, optimal_scale, po2_scale, n, out_zp);
        }
#endif
    }

    /**
     * 阶段 1：AIMET 风格连续 scale 搜索
     */
    template <typename QuantT>
    static ContinuousCalibrationResult computeOptimalContinuousScale(
        const Histogram& hist, bool is_symmetric, const HistogramCalibrationConfig& config = HistogramCalibrationConfig()) {
        
        const int64_t quant_min = static_cast<int64_t>(std::numeric_limits<QuantT>::min());
        const int64_t quant_max = static_cast<int64_t>(std::numeric_limits<QuantT>::max());
        const int64_t num_steps = quant_max - quant_min;

        // 确保范围包含 0
        float min_val = std::min(hist.min_val, 0.0f);
        float max_val = std::max(hist.max_val, 0.0f);
        max_val = std::max(max_val, min_val + 1e-8f * num_steps);
        
        float max_delta = is_symmetric 
            ? 2.0f * std::max(max_val, -min_val) / num_steps
            : (max_val - min_val) / num_steps;
        
        return is_symmetric 
            ? searchSymmetric(hist, max_delta, num_steps, config)
            : searchAsymmetric(hist, min_val, max_val, max_delta, num_steps, config);
    }

    /**
     * 阶段 2：转换到 POT（完全 AIMET 一致）
     * 
     * AIMET find_closest_power_of_2_scale 使用 round：
     *   n = -log2(scale)
     *   n_rounded = round(n)
     *   new_scale = 2^(-n_rounded)
     */
    static std::pair<float, int8_t> roundToPowerOfTwo(float scale) {
        if (scale <= 0) {
            throw std::runtime_error("Invalid scale <= 0 in roundToPowerOfTwo");
        }
        float n = -std::log2(scale);
        // AIMET 使用 round，四舍五入到最近的 POT
        int8_t n_rounded = static_cast<int8_t>(std::round(n));
        return {std::pow(2.0f, -static_cast<float>(n_rounded)), n_rounded};
    }

   private:
    HistogramCalibrationConfig config_;
    HistogramCollector collector_;

    /**
     * 对称量化搜索
     */
    static ContinuousCalibrationResult searchSymmetric(
        const Histogram& hist, float max_delta, int64_t num_steps, const HistogramCalibrationConfig& config) {
        
        ContinuousCalibrationResult result{0, 0, 0, 0, std::numeric_limits<float>::max()};
        // 对称量化：offset = -num_steps // 2（整数除法，与 AIMET 一致）
        const float offset = -static_cast<float>(num_steps / 2);
        
        for (int d = 1; d <= config.symmetric_delta_candidates; ++d) {
            float delta = max_delta * d / (config.symmetric_delta_candidates - 1);
            delta = std::max(delta, 1e-8f);
            
            float noise = estimateNoise(hist, delta, offset, num_steps, config.gamma, config.p);
            
            if (noise < result.best_noise) {
                result.best_noise = noise;
                result.optimal_scale = delta;
                result.optimal_offset = offset;
                result.optimal_min = offset * delta;
                result.optimal_max = result.optimal_min + num_steps * delta;
            }
        }
        
        if (result.best_noise == std::numeric_limits<float>::max()) {
            throw std::runtime_error("searchSymmetric failed to find valid scale");
        }
        return result;
    }

    /**
     * 非对称量化搜索
     */
    static ContinuousCalibrationResult searchAsymmetric(
        const Histogram& hist, float min_val, float max_val, float max_delta,
        int64_t num_steps, const HistogramCalibrationConfig& config) {
        
        ContinuousCalibrationResult result{0, 0, 0, 0, std::numeric_limits<float>::max()};
        
        const int num_offsets = std::min(static_cast<int>(num_steps + 2), config.offset_candidates);
        std::vector<float> offsets(num_offsets);
        float offset_step = static_cast<float>(num_steps) / (num_offsets - 2);
        for (int o = 0; o < num_offsets - 1; ++o) {
            offsets[o] = std::round(-static_cast<float>(num_steps) + o * offset_step);
        }
        offsets[num_offsets - 1] = std::round(min_val / max_delta);  // observed offset
        
        for (int d = 1; d <= config.asymmetric_delta_candidates; ++d) {
            float delta = max_delta * d / (config.asymmetric_delta_candidates - 1);
            delta = std::max(delta, 1e-8f);
            
            for (int o = 0; o < num_offsets; ++o) {
                float offset = offsets[o];
                
                // Clamp to observed range
                float test_min = std::max(min_val, delta * offset);
                float test_max = std::min(max_val, test_min + delta * num_steps);
                float clamped_delta = std::max((test_max - test_min) / num_steps, 1e-8f);
                float clamped_offset = std::round(test_min / clamped_delta);
                
                float noise = estimateNoise(hist, clamped_delta, clamped_offset, num_steps,
                                             config.gamma, config.p);
                
                if (noise < result.best_noise) {
                    result.best_noise = noise;
                    result.optimal_scale = clamped_delta;
                    result.optimal_offset = clamped_offset;
                    result.optimal_min = clamped_offset * clamped_delta;
                    result.optimal_max = result.optimal_min + num_steps * clamped_delta;
                }
            }
        }
        
        if (result.best_noise == std::numeric_limits<float>::max()) {
            throw std::runtime_error("searchAsymmetric failed to find valid scale");
        }
        return result;
    }

    /**
     * 估算量化噪声（与 AIMET _estimate_clip_and_quant_noise 完全一致）
     * 
     * AIMET 量化公式：
     *   q = round(x / delta - offset)     // 先减后 round
     *   q = clamp(q, 0, num_steps)
     *   x_recon = (q + offset) * delta
     */
    static float estimateNoise(const Histogram& hist, float delta, float offset,
                                int64_t num_steps, float gamma, float p) {
        if (delta <= 0) return std::numeric_limits<float>::max();
        
        float bin_width = hist.bin_width();
        float total_noise = 0.0f;
        
        for (int i = 0; i < hist.num_bins; ++i) {
            float count = hist.counts[i];
            if (count < 1e-6f) continue;
            
            float x = hist.min_val + (i + 0.5f) * bin_width;
            
            // AIMET 公式：q = round(x / delta - offset)（先减后 round）
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
};

/**
 * 从直方图计算 POT 量化参数（便捷函数）
 * 
 * @param hist 直方图数据
 * @param is_symmetric 是否对称量化
 * @param exp2_inv 输出：POT 指数 (scale = 2^(-exp2_inv))
 * @param zp 输出：零点
 * @param name 调试名称（可选）
 * @param scheme 校准方案：SQNR 或 PERCENTILE
 * @param percentile 百分位数（仅 PERCENTILE 方案使用）
 */
template <typename QuantT>
inline void calibrateQuantParamsFromHistogram(const Histogram& hist, bool is_symmetric,
                                              int8_t& exp2_inv, int32_t& zp,
                                              const char* name = nullptr,
                                              CalibrationScheme scheme = CalibrationScheme::SQNR,
                                              float percentile = 99.99f) {
    HistogramCalibrationConfig config;
    config.scheme = scheme;
    config.percentile = percentile;
    AimetPotSqnrCalibrator::computeOptimalParamsFromHistogram<QuantT>(
        hist, is_symmetric, exp2_inv, zp, name, config);
}

// 向后兼容别名
using POTSqnrCalibrator = AimetPotSqnrCalibrator;
