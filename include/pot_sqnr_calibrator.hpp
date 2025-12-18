#pragma once

/**
 * POT-SQNR Calibrator (Power-of-Two SQNR Calibrator)
 *
 * 简化版 TfEnhanced 校准器：
 * - 保持 POT (Power-of-Two) scale 约束: scale = 2^(-exp2_inv)
 * - 使用直方图 + SQNR 优化选择最优 exp2_inv
 * - 比纯 MinMax 方法更好地处理异常值
 *
 * 核心思想：
 * 1. 收集数据分布的直方图
 * 2. 对于每个候选 exp2_inv，估算量化噪声
 * 3. 选择噪声最小的 exp2_inv
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "histogram_collector.hpp"

/**
 * POT-SQNR 校准器类
 *
 * 使用方法：
 *   POTSqnrCalibrator calibrator;
 *   calibrator.update(data1, size1);
 *   calibrator.update(data2, size2);
 *   auto [exp2_inv, zp] = calibrator.computeOptimalParams<int8_t>(is_symmetric);
 */
class POTSqnrCalibrator {
   public:
    // 配置参数
    struct Config {
        int num_bins;              // 直方图 bin 数量
        int exp2_inv_search_range; // 搜索范围：基准 ± range
        float gamma;               // 剪切噪声权重（越大越不容易剪切）
        float ema_decay;           // EMA 衰减系数（用于多 batch）
        bool use_ema;              // 是否使用 EMA 平滑直方图

        Config()
            : num_bins(2048), exp2_inv_search_range(8), gamma(3.0f), ema_decay(0.9f), use_ema(false) {}
    };

    POTSqnrCalibrator(Config config = Config()) : config_(config), collector_(config.num_bins) {}

    /**
     * 更新统计信息（收集数据到直方图）
     * @param data 输入数据指针
     * @param size 数据大小
     */
    void update(const float* data, size_t size) {
        if (size == 0) return;
        collector_.collect(data, size);
    }

    /**
     * 合并另一个直方图
     */
    void merge(const Histogram& other) { collector_.merge(other); }

    /**
     * 计算最优的 POT 量化参数
     * @tparam QuantT 量化类型 (int8_t, int16_t 等)
     * @param is_symmetric 是否对称量化
     * @param out_exp2_inv 输出的 exp2_inv
     * @param out_zp 输出的 zero-point
     * @param name 调试名称
     */
    template <typename QuantT>
    void computeOptimalParams(bool is_symmetric, int8_t& out_exp2_inv, int32_t& out_zp,
                              const char* name = nullptr) {
        const Histogram& hist = collector_.histogram();
        computeOptimalParamsFromHistogram<QuantT>(hist, is_symmetric, out_exp2_inv, out_zp, name,
                                                  config_.exp2_inv_search_range, config_.gamma);
    }

    /**
     * 从外部直方图计算最优的 POT 量化参数（静态方法）
     * @tparam QuantT 量化类型
     * @param hist 直方图
     * @param is_symmetric 是否对称量化
     * @param out_exp2_inv 输出的 exp2_inv
     * @param out_zp 输出的 zero-point
     * @param name 调试名称
     * @param search_range 搜索范围
     * @param gamma 剪切惩罚系数
     */
    template <typename QuantT>
    static void computeOptimalParamsFromHistogram(const Histogram& hist, bool is_symmetric,
                                                  int8_t& out_exp2_inv, int32_t& out_zp,
                                                  const char* name = nullptr, int search_range = 8,
                                                  float gamma = 3.0f) {
        if (!hist.is_valid()) {
            // 没有数据，使用默认值
            out_exp2_inv = 7;
            out_zp = 0;
            return;
        }

        const int64_t quant_min = static_cast<int64_t>(std::numeric_limits<QuantT>::min());
        const int64_t quant_max = static_cast<int64_t>(std::numeric_limits<QuantT>::max());
        const int64_t num_steps = quant_max - quant_min;

        // 1. 计算基准 exp2_inv（基于 MinMax）
        float range = hist.max_val - hist.min_val;
        if (is_symmetric) {
            range = 2.0f * std::max(std::abs(hist.min_val), std::abs(hist.max_val));
        }
        range = std::max(range, 1e-9f);

        float raw_scale = range / num_steps;
        int8_t base_exp2_inv = static_cast<int8_t>(std::floor(std::log2(1.0f / raw_scale)));

        // 2. 搜索最优 exp2_inv
        float best_noise = std::numeric_limits<float>::max();
        int8_t best_exp2_inv = base_exp2_inv;
        int32_t best_zp = 0;

        for (int delta = -search_range; delta <= search_range; ++delta) {
            int8_t candidate_exp2_inv = base_exp2_inv + delta;

            // 计算对应的 scale 和 zp
            float scale = std::pow(2.0f, -static_cast<float>(candidate_exp2_inv));
            int32_t zp = 0;

            if (!is_symmetric) {
                // 非对称量化：计算 zero-point
                float zp_fp = quant_min - hist.min_val / scale;
                zp = static_cast<int32_t>(std::round(zp_fp));
            }

            // 估算量化噪声
            float noise = estimateQuantizationNoiseFromHistogram<QuantT>(hist, candidate_exp2_inv,
                                                                         zp, is_symmetric, gamma);

            if (noise < best_noise) {
                best_noise = noise;
                best_exp2_inv = candidate_exp2_inv;
                best_zp = zp;
            }
        }

        out_exp2_inv = best_exp2_inv;
        out_zp = best_zp;

        // 调试输出
        if (name != nullptr && name[0] != '\0') {
            float best_scale = std::pow(2.0f, -static_cast<float>(best_exp2_inv));
            float base_scale = std::pow(2.0f, -static_cast<float>(base_exp2_inv));
            printf(
                "[POT-SQNR][%s] range=[%.4f, %.4f], base_exp2_inv=%d (scale=%.6f), "
                "best_exp2_inv=%d (scale=%.6f), zp=%d, noise=%.4f\n",
                name, hist.min_val, hist.max_val, base_exp2_inv, base_scale, best_exp2_inv,
                best_scale, best_zp, best_noise);
        }
    }

    /**
     * 从外部直方图估算量化噪声（静态方法）
     */
    template <typename QuantT>
    static float estimateQuantizationNoiseFromHistogram(const Histogram& hist, int8_t exp2_inv,
                                                        int32_t zp, bool is_symmetric,
                                                        float gamma = 3.0f) {
        const int64_t quant_min = static_cast<int64_t>(std::numeric_limits<QuantT>::min());
        const int64_t quant_max = static_cast<int64_t>(std::numeric_limits<QuantT>::max());

        float scale = std::pow(2.0f, -static_cast<float>(exp2_inv));
        float scale_inv = std::pow(2.0f, static_cast<float>(exp2_inv));

        float total_error = 0.0f;
        float total_count = 0.0f;

        float bin_width = hist.bin_width();

        for (int i = 0; i < hist.num_bins; ++i) {
            float count = hist.counts[i];
            if (count < 1e-6f) continue;

            // bin 中心值
            float x = hist.min_val + (i + 0.5f) * bin_width;

            // 量化
            int64_t q = static_cast<int64_t>(std::round(x * scale_inv)) + zp;

            // 检测剪切
            bool clipped = (q < quant_min) || (q > quant_max);

            // 剪切到有效范围
            q = std::max(quant_min, std::min(quant_max, q));

            // 反量化
            float x_recon = static_cast<float>(q - zp) * scale;

            // 计算误差 (MSE)
            float error = (x_recon - x) * (x_recon - x);

            // 对剪切的值应用 gamma 惩罚
            if (clipped) {
                error *= gamma;
            }

            total_error += error * count;
            total_count += count;
        }

        if (total_count < 1e-6f) return std::numeric_limits<float>::max();
        return total_error / total_count;  // 归一化误差
    }

    /**
     * 获取当前的 min/max 范围
     */
    std::pair<float, float> getRange() const {
        const Histogram& hist = collector_.histogram();
        return {hist.min_val, hist.max_val};
    }

    /**
     * 重置校准器
     */
    void reset() { collector_.reset(); }

    /**
     * 获取直方图（用于可视化/调试）
     */
    const Histogram& getHistogram() const { return collector_.histogram(); }

    /**
     * 检查是否有有效数据
     */
    bool isValid() const { return collector_.is_valid(); }

   private:
    Config config_;
    HistogramCollector collector_;
};

/**
 * 使用 POT-SQNR 优化的量化参数计算函数
 * 替代原有的 calibrateQuantParams，使用 SQNR 优化选择最优 exp2_inv
 */
template <typename T, typename QuantT>
inline void calibrateQuantParamsSQNR(const std::vector<T>& data, const bool is_symmetric,
                                     T& aligned_min, T& aligned_max, int8_t& exp2_inv, int32_t& zp,
                                     const std::string& name = "", bool verbose = false) {
    if (data.empty()) {
        exp2_inv = 7;
        zp = 0;
        aligned_min = -1;
        aligned_max = 1;
        return;
    }

    // 创建校准器并收集数据
    POTSqnrCalibrator::Config config;
    config.num_bins = 2048;
    config.exp2_inv_search_range = 6;
    config.gamma = 3.0f;
    config.use_ema = false;  // 单次调用不需要 EMA

    POTSqnrCalibrator calibrator(config);
    calibrator.update(data.data(), data.size());

    // 计算最优参数
    calibrator.computeOptimalParams<QuantT>(is_symmetric, exp2_inv, zp,
                                            verbose ? name.c_str() : nullptr);

    // 计算对齐后的范围
    float scale = std::pow(2.0f, -static_cast<float>(exp2_inv));
    const int32_t quant_min = std::numeric_limits<QuantT>::min();
    const int32_t quant_max = std::numeric_limits<QuantT>::max();

    if (is_symmetric) {
        aligned_max = static_cast<T>(scale * quant_max);
        aligned_min = -aligned_max;
    } else {
        aligned_min = static_cast<T>((quant_min - zp) * scale);
        aligned_max = static_cast<T>((quant_max - zp) * scale);
    }
}

/**
 * 从直方图计算 POT-SQNR 量化参数
 * 这是真正的 AIMET 风格校准方法
 * 
 * @param percentile_clip 百分位数裁剪 (例如 0.001 表示裁剪 0.1% 的极值)
 *                        设为 0 表示不裁剪
 */
template <typename QuantT>
inline void calibrateQuantParamsFromHistogram(const Histogram& hist, bool is_symmetric,
                                              int8_t& exp2_inv, int32_t& zp,
                                              const char* name = nullptr, int search_range = 6,
                                              float gamma = 3.0f, float percentile_clip = 0.001f) {
    // 如果需要百分位数裁剪，创建一个裁剪后的直方图副本
    if (percentile_clip > 0.0f && hist.is_valid()) {
        auto [pmin, pmax] = hist.getPercentileRange(percentile_clip);
        
        // 创建裁剪后的直方图
        Histogram clipped_hist = hist;
        clipped_hist.min_val = pmin;
        clipped_hist.max_val = pmax;
        
        POTSqnrCalibrator::computeOptimalParamsFromHistogram<QuantT>(clipped_hist, is_symmetric, exp2_inv, zp,
                                                                     name, search_range, gamma);
    } else {
        POTSqnrCalibrator::computeOptimalParamsFromHistogram<QuantT>(hist, is_symmetric, exp2_inv, zp,
                                                                     name, search_range, gamma);
    }
}

/**
 * Per-Channel POT-SQNR 校准
 * 用于权重的 per-channel 量化
 */
template <typename T, typename QuantT>
inline std::vector<int8_t> calibratePerChannelSQNR(const T* data, int input_size, int channel_size,
                                                   bool verbose = false,
                                                   const std::string& name = "") {
    std::vector<int8_t> exp2_invs(channel_size);

    POTSqnrCalibrator::Config config;
    config.num_bins = 256;  // per-channel 用较少的 bins
    config.exp2_inv_search_range = 4;
    config.gamma = 3.0f;
    config.use_ema = false;

#pragma omp parallel for
    for (int c = 0; c < channel_size; ++c) {
        // 提取该 channel 的数据
        std::vector<float> channel_data(input_size);
        for (int i = 0; i < input_size; ++i) {
            channel_data[i] = static_cast<float>(data[i * channel_size + c]);
        }

        // 校准
        POTSqnrCalibrator calibrator(config);
        calibrator.update(channel_data.data(), channel_data.size());

        int32_t zp_tmp;
        calibrator.computeOptimalParams<QuantT>(true,  // per-channel 权重通常用对称量化
                                                exp2_invs[c], zp_tmp,
                                                verbose ? (name + "_ch" + std::to_string(c)).c_str()
                                                        : nullptr);
    }

    return exp2_invs;
}

// POT_SQNR_CALIBRATOR_HPP
