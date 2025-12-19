// ============================================================================
// histogram_collector.hpp - AIMET 风格的直方图收集器
// ============================================================================
//
// 实现类似 AIMET _HistogramObserver 的功能：
//   1. 收集数据到直方图
//   2. 支持多批次合并（带范围扩展）
//   3. 支持 SQNR 优化计算量化参数
//
// ============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

/**
 * 单个直方图结构体
 * 存储一个张量的直方图数据
 */
struct Histogram {
    std::vector<float> counts;  // 每个 bin 的计数
    float min_val;              // 直方图覆盖的最小值
    float max_val;              // 直方图覆盖的最大值
    int num_bins;               // bin 数量
    int64_t total_count;        // 总采样数

    Histogram() : min_val(0), max_val(0), num_bins(0), total_count(0) {}

    Histogram(int bins)
        : counts(bins, 0.0f),
          min_val(std::numeric_limits<float>::max()),
          max_val(std::numeric_limits<float>::lowest()),
          num_bins(bins),
          total_count(0) {}

    bool is_valid() const { return num_bins > 0 && total_count > 0; }

    float bin_width() const {
        if (num_bins <= 0 || max_val <= min_val) return 1.0f;
        return (max_val - min_val) / num_bins;
    }

    void reset(int bins = 0) {
        if (bins > 0) {
            num_bins = bins;
            counts.assign(bins, 0.0f);
        } else if (num_bins > 0) {
            counts.assign(num_bins, 0.0f);
        }
        min_val = std::numeric_limits<float>::max();
        max_val = std::numeric_limits<float>::lowest();
        total_count = 0;
    }

    void print(const char* name = nullptr) const {
        if (name) printf("[Histogram %s] ", name);
        printf("bins=%d, range=[%.6f, %.6f], total=%ld\n", num_bins, min_val, max_val, total_count);
    }

    /**
     * 获取百分位数范围
     * @param percentile 裁剪百分位数 (例如 0.001 表示 0.1% ~ 99.9%)
     * @return (min, max) 对应的范围
     */
    std::pair<float, float> getPercentileRange(float percentile = 0.001f) const {
        if (!is_valid() || total_count == 0) {
            return {min_val, max_val};
        }

        float cumsum = 0.0f;
        float total = 0.0f;
        for (int i = 0; i < num_bins; ++i) {
            total += counts[i];
        }

        if (total < 1e-6f) {
            return {min_val, max_val};
        }

        float lower_threshold = total * percentile;
        float upper_threshold = total * (1.0f - percentile);

        float bw = bin_width();
        float pmin = min_val;
        float pmax = max_val;

        // 找下限
        cumsum = 0.0f;
        for (int i = 0; i < num_bins; ++i) {
            cumsum += counts[i];
            if (cumsum >= lower_threshold) {
                pmin = min_val + i * bw;
                break;
            }
        }

        // 找上限
        cumsum = 0.0f;
        for (int i = num_bins - 1; i >= 0; --i) {
            cumsum += counts[i];
            if (cumsum >= total * percentile) {
                pmax = min_val + (i + 1) * bw;
                break;
            }
        }

        return {pmin, pmax};
    }
};

/**
 * 直方图收集器
 * 类似 AIMET 的 _HistogramObserver
 */
class HistogramCollector {
   public:
    struct Config {
        int num_bins = 2048;  // 直方图 bin 数量（AIMET 默认 2048）
        float ema_decay = 0.0f;  // EMA 衰减系数，0 表示不使用 EMA
    };

   private:
    Config config_;
    Histogram hist_;

   public:
    HistogramCollector() : config_(), hist_() {}
    explicit HistogramCollector(const Config& config) : config_(config), hist_(config.num_bins) {}
    explicit HistogramCollector(int num_bins) : config_(), hist_(num_bins) { config_.num_bins = num_bins; }

    /**
     * 收集数据到直方图（首次收集）
     * 自动确定范围
     */
    void collect(const float* data, size_t size) {
        if (size == 0) return;

        // 计算数据范围
        float data_min = data[0];
        float data_max = data[0];
        for (size_t i = 1; i < size; ++i) {
            data_min = std::min(data_min, data[i]);
            data_max = std::max(data_max, data[i]);
        }

        // 确保范围有效
        if (data_max <= data_min) {
            data_max = data_min + 1e-6f;
        }

        if (!hist_.is_valid()) {
            // 首次收集：初始化直方图
            hist_.reset(config_.num_bins);
            hist_.min_val = data_min;
            hist_.max_val = data_max;
            _add_to_histogram(data, size);
        } else {
            // 后续收集：可能需要扩展范围并合并
            if (data_min >= hist_.min_val && data_max <= hist_.max_val) {
                // 新数据在已有范围内，直接添加
                _add_to_histogram(data, size);
            } else {
                // 需要扩展范围
                _merge_with_extended_range(data, size, data_min, data_max);
            }
        }
    }

    /**
     * 合并另一个直方图（类似 AIMET 的 merge_stats）
     */
    void merge(const Histogram& other) {
        if (!other.is_valid()) return;

        if (!hist_.is_valid()) {
            // 直接复制
            hist_ = other;
            return;
        }

        // 计算合并后的范围
        float new_min = std::min(hist_.min_val, other.min_val);
        float new_max = std::max(hist_.max_val, other.max_val);

        if (new_min == hist_.min_val && new_max == hist_.max_val) {
            // 范围相同，直接合并
            _merge_same_range(other);
        } else {
            // 范围不同，需要重新分配
            _merge_different_range(other, new_min, new_max);
        }
    }

    /**
     * 获取当前直方图
     */
    const Histogram& histogram() const { return hist_; }
    Histogram& histogram() { return hist_; }

    /**
     * 重置直方图
     */
    void reset() { hist_.reset(config_.num_bins); }

    /**
     * 是否有有效数据
     */
    bool is_valid() const { return hist_.is_valid(); }

   private:
    /**
     * 将数据添加到当前直方图（假设范围已设置）
     */
    void _add_to_histogram(const float* data, size_t size) {
        float bin_width = hist_.bin_width();
        for (size_t i = 0; i < size; ++i) {
            int bin_idx = static_cast<int>((data[i] - hist_.min_val) / bin_width);
            bin_idx = std::max(0, std::min(bin_idx, hist_.num_bins - 1));
            hist_.counts[bin_idx] += 1.0f;
        }
        hist_.total_count += size;
    }

    /**
     * 扩展范围并合并新数据
     * 类似 AIMET 的 _adjust_min_max_and_update_histogram
     */
    void _merge_with_extended_range(const float* data, size_t size, float data_min, float data_max) {
        float new_min = std::min(hist_.min_val, data_min);
        float new_max = std::max(hist_.max_val, data_max);

        // 创建新直方图
        std::vector<float> new_counts(config_.num_bins, 0.0f);
        float new_bin_width = (new_max - new_min) / config_.num_bins;

        // 重新分配旧直方图的计数
        float old_bin_width = hist_.bin_width();
        for (int old_idx = 0; old_idx < hist_.num_bins; ++old_idx) {
            if (hist_.counts[old_idx] <= 0) continue;

            // 旧 bin 的范围
            float old_bin_start = hist_.min_val + old_idx * old_bin_width;
            float old_bin_end = old_bin_start + old_bin_width;

            // 映射到新直方图
            int new_start_idx = static_cast<int>((old_bin_start - new_min) / new_bin_width);
            int new_end_idx = static_cast<int>((old_bin_end - new_min) / new_bin_width);

            new_start_idx = std::max(0, std::min(new_start_idx, config_.num_bins - 1));
            new_end_idx = std::max(0, std::min(new_end_idx, config_.num_bins - 1));

            if (new_start_idx == new_end_idx) {
                new_counts[new_start_idx] += hist_.counts[old_idx];
            } else {
                // 跨多个 bin，按比例分配
                float count_per_bin = hist_.counts[old_idx] / (new_end_idx - new_start_idx + 1);
                for (int j = new_start_idx; j <= new_end_idx; ++j) {
                    new_counts[j] += count_per_bin;
                }
            }
        }

        // 添加新数据
        for (size_t i = 0; i < size; ++i) {
            int bin_idx = static_cast<int>((data[i] - new_min) / new_bin_width);
            bin_idx = std::max(0, std::min(bin_idx, config_.num_bins - 1));
            new_counts[bin_idx] += 1.0f;
        }

        // 更新直方图
        hist_.counts = std::move(new_counts);
        hist_.min_val = new_min;
        hist_.max_val = new_max;
        hist_.total_count += size;
    }

    /**
     * 合并范围相同的直方图
     */
    void _merge_same_range(const Histogram& other) {
        for (int i = 0; i < hist_.num_bins; ++i) {
            hist_.counts[i] += other.counts[i];
        }
        hist_.total_count += other.total_count;
    }

    /**
     * 合并范围不同的直方图
     */
    void _merge_different_range(const Histogram& other, float new_min, float new_max) {
        std::vector<float> new_counts(config_.num_bins, 0.0f);
        float new_bin_width = (new_max - new_min) / config_.num_bins;

        // 重新分配当前直方图
        _redistribute_histogram(hist_, new_counts, new_min, new_bin_width);

        // 重新分配另一个直方图
        _redistribute_histogram(other, new_counts, new_min, new_bin_width);

        // 更新
        hist_.counts = std::move(new_counts);
        hist_.min_val = new_min;
        hist_.max_val = new_max;
        hist_.total_count += other.total_count;
    }

    /**
     * 将直方图重新分配到新的 bin
     */
    void _redistribute_histogram(const Histogram& src, std::vector<float>& dst, float new_min,
                                 float new_bin_width) {
        float src_bin_width = src.bin_width();
        for (int src_idx = 0; src_idx < src.num_bins; ++src_idx) {
            if (src.counts[src_idx] <= 0) continue;

            float src_bin_start = src.min_val + src_idx * src_bin_width;
            float src_bin_end = src_bin_start + src_bin_width;

            int dst_start_idx = static_cast<int>((src_bin_start - new_min) / new_bin_width);
            int dst_end_idx = static_cast<int>((src_bin_end - new_min) / new_bin_width);

            dst_start_idx = std::max(0, std::min(dst_start_idx, config_.num_bins - 1));
            dst_end_idx = std::max(0, std::min(dst_end_idx, config_.num_bins - 1));

            if (dst_start_idx == dst_end_idx) {
                dst[dst_start_idx] += src.counts[src_idx];
            } else {
                float count_per_bin = src.counts[src_idx] / (dst_end_idx - dst_start_idx + 1);
                for (int j = dst_start_idx; j <= dst_end_idx; ++j) {
                    dst[j] += count_per_bin;
                }
            }
        }
    }
};

/**
 * GRU 直方图收集器
 * 为 GRU 的每个中间张量维护一个直方图
 */
struct GRUHistogramCollectors {
    int hidden_ = 0;
    int num_bins_ = 2048;

    // 输入和隐藏状态
    HistogramCollector x_hist;
    HistogramCollector h_hist;

    // GEMM 结果
    HistogramCollector Wx_hist;
    HistogramCollector Rh_hist;

    // 门的预激活值
    HistogramCollector z_pre_hist;
    HistogramCollector r_pre_hist;
    HistogramCollector g_pre_hist;

    // 门的输出值
    HistogramCollector z_out_hist;
    HistogramCollector r_out_hist;
    HistogramCollector g_out_hist;

    // 中间计算结果
    HistogramCollector Rh_add_br_g_hist;
    HistogramCollector rRh_hist;
    HistogramCollector new_contrib_hist;
    HistogramCollector old_contrib_hist;

    // 权重（per-channel，每个 channel 一个直方图）
    std::vector<HistogramCollector> W_hist;
    std::vector<HistogramCollector> R_hist;
    std::vector<HistogramCollector> bx_hist;
    std::vector<HistogramCollector> br_hist;

    GRUHistogramCollectors() = default;

    explicit GRUHistogramCollectors(int hidden, int num_bins = 2048)
        : hidden_(hidden), num_bins_(num_bins) {
        reset(hidden, num_bins);
    }

    void reset(int hidden = -1, int num_bins = -1) {
        if (hidden > 0) hidden_ = hidden;
        if (num_bins > 0) num_bins_ = num_bins;

        HistogramCollector::Config cfg;
        cfg.num_bins = num_bins_;

        x_hist = HistogramCollector(cfg);
        h_hist = HistogramCollector(cfg);
        Wx_hist = HistogramCollector(cfg);
        Rh_hist = HistogramCollector(cfg);
        z_pre_hist = HistogramCollector(cfg);
        r_pre_hist = HistogramCollector(cfg);
        g_pre_hist = HistogramCollector(cfg);
        z_out_hist = HistogramCollector(cfg);
        r_out_hist = HistogramCollector(cfg);
        g_out_hist = HistogramCollector(cfg);
        Rh_add_br_g_hist = HistogramCollector(cfg);
        rRh_hist = HistogramCollector(cfg);
        new_contrib_hist = HistogramCollector(cfg);
        old_contrib_hist = HistogramCollector(cfg);

        // Per-channel 直方图
        int channel_size = hidden_ * 3;
        W_hist.assign(channel_size, HistogramCollector(cfg));
        R_hist.assign(channel_size, HistogramCollector(cfg));
        bx_hist.assign(channel_size, HistogramCollector(cfg));
        br_hist.assign(channel_size, HistogramCollector(cfg));
    }

    bool is_valid() const { return hidden_ > 0 && x_hist.is_valid(); }

    void print() const {
        printf("GRUHistogramCollectors (hidden=%d, num_bins=%d):\n", hidden_, num_bins_);
        x_hist.histogram().print("x");
        h_hist.histogram().print("h");
        Wx_hist.histogram().print("Wx");
        Rh_hist.histogram().print("Rh");
        z_pre_hist.histogram().print("z_pre");
        r_pre_hist.histogram().print("r_pre");
        g_pre_hist.histogram().print("g_pre");
        z_out_hist.histogram().print("z_out");
        r_out_hist.histogram().print("r_out");
        g_out_hist.histogram().print("g_out");
    }
};

