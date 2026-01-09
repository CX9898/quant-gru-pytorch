// ============================================================================
// calibration_gpu.cuh - GPU 加速的直方图收集器
// ============================================================================
//
// 在 GPU 上直接构建直方图，避免大量 GPU->CPU 数据传输
// 使用 Thrust 进行 min/max 计算，使用 atomicAdd 构建直方图
//
// ============================================================================

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "dev_vector.h"
#include "parallel_algorithm.h"  // for dev::fill_n
#include "pot_sqnr_calibrator.h"  // for SqnrConfig, QuantBitWidth

// ============================================================================
// GPU 直方图结构体
// ============================================================================

/**
 * @brief GPU 端直方图数据结构
 *
 * 直方图的 counts 存储在 GPU 显存中，min/max 等元数据在 Host 端
 */
struct GPUHistogram {
    dev::vector<float> counts;  // GPU 端的 bin 计数
    float min_val;              // 直方图覆盖的最小值
    float max_val;              // 直方图覆盖的最大值
    int num_bins;               // bin 数量
    int64_t total_count;        // 总采样数

    GPUHistogram() : min_val(0), max_val(0), num_bins(0), total_count(0) {}

    explicit GPUHistogram(int bins)
        : counts(bins),
          min_val(0),
          max_val(0),
          num_bins(bins),
          total_count(0) {
        if (bins > 0) {
            dev::fill_n(counts.data(), bins, 0.0f);
        }
    }

    bool is_valid() const { return num_bins > 0 && total_count > 0; }

    float bin_width() const {
        if (num_bins <= 0 || max_val <= min_val) return 1.0f;
        return (max_val - min_val) / num_bins;
    }

    void reset(int bins = 0) {
        if (bins > 0) {
            num_bins = bins;
            counts.resize(bins);
        }
        if (num_bins > 0) {
            dev::fill_n(counts.data(), num_bins, 0.0f);
        }
        min_val = 0;
        max_val = 0;
        total_count = 0;
    }

    // 将 GPU 直方图拷贝到 Host 端 vector
    std::vector<float> to_host() const {
        std::vector<float> host_counts(num_bins);
        if (num_bins > 0) {
            cudaMemcpy(host_counts.data(), counts.data(), num_bins * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
        return host_counts;
    }
};

// ============================================================================
// GPU 直方图收集器
// ============================================================================

/**
 * @brief GPU 加速的直方图收集器
 *
 * 与 HistogramCollector 接口类似，但所有计算在 GPU 上完成
 */
class GPUHistogramCollector {
   public:
    struct Config {
        int num_bins = 2048;       // 直方图 bin 数量
        float growth_limit = 1e9f; // 范围扩展限制
    };

   private:
    Config config_;
    GPUHistogram hist_;

    // 范围限制（首次收集后设置）
    bool range_limit_set_ = false;
    float range_limit_min_ = 0.0f;
    float range_limit_max_ = 0.0f;

   public:
    GPUHistogramCollector() : config_(), hist_() {}
    explicit GPUHistogramCollector(const Config& config)
        : config_(config), hist_(config.num_bins) {}
    explicit GPUHistogramCollector(int num_bins) : config_(), hist_(num_bins) {
        config_.num_bins = num_bins;
    }

    /**
     * @brief 从 GPU 数据收集直方图（核心函数）
     *
     * @param data_dev GPU 端数据指针
     * @param size 数据元素数量
     * @param stream CUDA 流（可选）
     */
    void collect(const float* data_dev, size_t size, cudaStream_t stream = 0);

    /**
     * @brief 使用已知范围收集直方图（跳过 minmax 计算，更快）
     *
     * @param data_dev GPU 端数据指针
     * @param size 数据元素数量
     * @param known_min 已知的数据最小值
     * @param known_max 已知的数据最大值
     * @param stream CUDA 流（可选）
     */
    void collectWithKnownRange(const float* data_dev, size_t size, float known_min, float known_max,
                               cudaStream_t stream = 0);

    /**
     * @brief 合并另一个 GPU 直方图
     */
    void merge(const GPUHistogram& other);

    /**
     * @brief 获取当前直方图（GPU 端）
     */
    const GPUHistogram& histogram() const { return hist_; }
    GPUHistogram& histogram() { return hist_; }

    /**
     * @brief 重置直方图
     */
    void reset() {
        hist_.reset(config_.num_bins);
        range_limit_set_ = false;
    }

    bool is_valid() const { return hist_.is_valid(); }

   private:
    void _add_to_histogram_gpu(const float* data_dev, size_t size, cudaStream_t stream);
    void _merge_with_extended_range_gpu(const float* data_dev, size_t size, float new_min,
                                        float new_max, cudaStream_t stream);
};

// ============================================================================
// 批量 Per-Channel 直方图结构（共享连续内存）
// ============================================================================

/**
 * @brief 批量 per-channel 直方图，所有 channel 共享一个连续 GPU 缓冲区
 *
 * 布局: counts[channel_size * num_bins]（连续内存）
 *       mins[channel_size], maxs[channel_size]
 *
 * 优势：避免 N 次 D2D 拷贝，只需一次批量操作
 */
struct PerChannelHistogramBatch {
    dev::vector<float> counts;      // [channel_size * num_bins] 连续内存
    std::vector<float> mins;        // [channel_size] CPU 端元数据（可能被范围扩展修改）
    std::vector<float> maxs;        // [channel_size] CPU 端元数据（可能被范围扩展修改）
    std::vector<float> original_mins;  // [channel_size] 原始 min（未被范围扩展修改）
    std::vector<float> original_maxs;  // [channel_size] 原始 max（未被范围扩展修改）
    int channel_size = 0;
    int num_bins = 0;
    int64_t per_channel_count = 0;  // 每个 channel 的样本数

    PerChannelHistogramBatch() = default;

    void reset(int channels, int bins) {
        channel_size = channels;
        num_bins = bins;
        per_channel_count = 0;
        counts.resize(channels * bins);
        counts.zero();
        mins.assign(channels, 0.0f);
        maxs.assign(channels, 0.0f);
        original_mins.assign(channels, 0.0f);
        original_maxs.assign(channels, 0.0f);
    }

    bool is_valid() const { return channel_size > 0 && per_channel_count > 0; }

    // 获取指定 channel 的 counts 指针（GPU）
    float* channel_counts(int c) { return counts.data() + c * num_bins; }
    const float* channel_counts(int c) const { return counts.data() + c * num_bins; }

    // 一次性拷贝所有 counts 到 CPU
    std::vector<float> all_counts_to_host() const {
        std::vector<float> host(channel_size * num_bins);
        if (channel_size > 0 && num_bins > 0) {
            cudaMemcpy(host.data(), counts.data(), 
                      channel_size * num_bins * sizeof(float),
                      cudaMemcpyDeviceToHost);
        }
        return host;
    }
};

// ============================================================================
// GRU GPU 直方图收集器组
// ============================================================================

/**
 * @brief GRU 所有中间张量的 GPU 直方图收集器
 */
struct GRUGPUHistogramCollectors {
    int hidden_ = 0;
    int num_bins_ = 2048;

    // 输入和隐藏状态
    GPUHistogramCollector x_hist;
    GPUHistogramCollector h_hist;

    // GEMM 结果
    GPUHistogramCollector Wx_hist;
    GPUHistogramCollector Rh_hist;

    // 门的预激活值
    GPUHistogramCollector z_pre_hist;
    GPUHistogramCollector r_pre_hist;
    GPUHistogramCollector g_pre_hist;

    // 门的输出值
    GPUHistogramCollector z_out_hist;
    GPUHistogramCollector r_out_hist;
    GPUHistogramCollector g_out_hist;

    // 中间计算结果
    GPUHistogramCollector Rh_add_br_g_hist;
    GPUHistogramCollector rRh_hist;
    GPUHistogramCollector new_contrib_hist;
    GPUHistogramCollector old_contrib_hist;

    // 权重（per-channel）- 使用批量结构，共享连续内存
    PerChannelHistogramBatch W_batch;
    PerChannelHistogramBatch R_batch;
    PerChannelHistogramBatch bx_batch;
    PerChannelHistogramBatch br_batch;

    GRUGPUHistogramCollectors() = default;

    explicit GRUGPUHistogramCollectors(int hidden, int num_bins = 2048)
        : hidden_(hidden), num_bins_(num_bins) {
        reset(hidden, num_bins);
    }

    void reset(int hidden = -1, int num_bins = -1) {
        if (hidden > 0) hidden_ = hidden;
        if (num_bins > 0) num_bins_ = num_bins;

        GPUHistogramCollector::Config cfg;
        cfg.num_bins = num_bins_;

        x_hist = GPUHistogramCollector(cfg);
        h_hist = GPUHistogramCollector(cfg);
        Wx_hist = GPUHistogramCollector(cfg);
        Rh_hist = GPUHistogramCollector(cfg);
        z_pre_hist = GPUHistogramCollector(cfg);
        r_pre_hist = GPUHistogramCollector(cfg);
        g_pre_hist = GPUHistogramCollector(cfg);
        z_out_hist = GPUHistogramCollector(cfg);
        r_out_hist = GPUHistogramCollector(cfg);
        g_out_hist = GPUHistogramCollector(cfg);
        Rh_add_br_g_hist = GPUHistogramCollector(cfg);
        rRh_hist = GPUHistogramCollector(cfg);
        new_contrib_hist = GPUHistogramCollector(cfg);
        old_contrib_hist = GPUHistogramCollector(cfg);

        int channel_size = hidden_ * 3;
        W_batch.reset(channel_size, num_bins_);
        R_batch.reset(channel_size, num_bins_);
        bx_batch.reset(channel_size, num_bins_);
        br_batch.reset(channel_size, num_bins_);
    }

    bool is_valid() const { return hidden_ > 0 && x_hist.is_valid(); }
};

// ============================================================================
// GPU 直方图辅助函数声明
// ============================================================================

namespace gpu_hist {

/**
 * @brief 在 GPU 上计算 float 数组的 min/max
 *
 * 使用 Thrust 实现高效并行归约（同步操作，使用默认 stream）
 */
void compute_minmax(const float* data_dev, size_t size, float& min_val, float& max_val);

/**
 * @brief 在 GPU 上构建直方图
 *
 * @param data_dev 输入数据（GPU）
 * @param size 数据大小
 * @param counts_dev 输出直方图计数（GPU，需预先分配）
 * @param min_val 直方图最小值
 * @param max_val 直方图最大值
 * @param num_bins bin 数量
 * @param stream CUDA 流
 */
void build_histogram(const float* data_dev, size_t size, float* counts_dev, float min_val,
                     float max_val, int num_bins, cudaStream_t stream = 0);

/**
 * @brief 重新分配直方图到新范围（GPU 版本）
 *
 * @param src_counts_dev 源直方图计数（GPU）
 * @param src_min 源最小值
 * @param src_max 源最大值
 * @param dst_counts_dev 目标直方图计数（GPU，累加）
 * @param dst_min 目标最小值
 * @param dst_max 目标最大值
 * @param num_bins bin 数量
 * @param stream CUDA 流
 */
void redistribute_histogram(const float* src_counts_dev, float src_min, float src_max,
                            float* dst_counts_dev, float dst_min, float dst_max, int num_bins,
                            cudaStream_t stream = 0);

/**
 * @brief 从 v 张量提取并计算门输出值的直方图
 *
 * v 布局: [T, B, H*4] = [z, r, g, Rh_add_br]
 * 同时计算: rRh = r * Rh_add_br, new_contrib = (1-z)*g, old_contrib = z*h
 */
void collect_gate_histograms(GRUGPUHistogramCollectors& collectors, const float* v_dev,
                             const float* h_dev, int time_steps, int batch_size, int hidden_size,
                             cudaStream_t stream = 0);

/**
 * @brief 收集 per-channel 直方图（GPU 版本，写入独立 collectors）
 *
 * 数据布局: [input_size, channel_size]（行主序）
 * 注意：需要 N 次 D2D 拷贝到各 collector
 */
void collect_per_channel_histograms(std::vector<GPUHistogramCollector>& collectors,
                                    const float* data_dev, int input_size, int channel_size,
                                    cudaStream_t stream = 0);

/**
 * @brief 收集 per-channel 直方图（GPU 版本，写入共享批量结构）
 *
 * 数据布局: [input_size, channel_size]（行主序）
 * 
 * 零拷贝优化：直接把结果写入 batch.counts，无需任何 D2D
 * - 1 次 kernel: 批量 minmax
 * - 1 次 kernel: 批量直方图（结果直接在 batch.counts）
 * - 1 次 D2H: min/max 元数据
 */
void collect_per_channel_histograms_batch(PerChannelHistogramBatch& batch,
                                           const float* data_dev, int input_size,
                                           cudaStream_t stream = 0);

// ============================================================================
// GPU SQNR 搜索（只返回连续 scale，POT 转换统一在 CPU）
// ============================================================================

/**
 * @brief GPU 加速的 SQNR 连续 scale 搜索
 *
 * 只做计算密集的 SQNR 搜索，返回连续 scale 结果
 * POT 转换统一使用 CPU 的 convertToPot() 函数
 *
 * @param counts_dev GPU 端直方图 counts
 * @param hist_min 直方图最小值
 * @param hist_max 直方图最大值
 * @param num_bins bin 数量
 * @param num_steps 量化级数 (quant_max - quant_min)
 * @param is_symmetric 是否对称量化
 * @param config SQNR 搜索配置
 * @param is_unsigned 是否无符号量化（UINT）
 * @param stream CUDA 流
 * @return ContinuousScaleResult 连续 scale 结果
 */
ContinuousScaleResult searchSqnrGpu(
    const float* counts_dev,
    float hist_min, float hist_max,
    int num_bins, int64_t num_steps,
    bool is_symmetric,
    const SqnrConfig& config = SqnrConfig(),
    bool is_unsigned = false,
    cudaStream_t stream = 0);

/**
 * @brief 批量 GPU SQNR 搜索
 */
void searchSqnrBatchGpu(
    const std::vector<const float*>& counts_ptrs,
    const std::vector<float>& mins,
    const std::vector<float>& maxs,
    int num_bins, int64_t num_steps,
    const std::vector<bool>& is_symmetric,
    std::vector<ContinuousScaleResult>& out_results,
    const SqnrConfig& config = SqnrConfig(),
    const std::vector<bool>& is_unsigned = {},
    cudaStream_t stream = 0);

/**
 * @brief 从 PerChannelHistogramBatch 进行 per-channel SQNR 搜索
 */
void searchSqnrPerChannelGpu(
    const PerChannelHistogramBatch& batch,
    int64_t num_steps, bool is_symmetric,
    std::vector<ContinuousScaleResult>& out_results,
    const SqnrConfig& config = SqnrConfig(),
    bool is_unsigned = false,
    cudaStream_t stream = 0);

// ============================================================================
// GPU MINMAX 量化范围计算
// ============================================================================

/**
 * @brief 计算设备端数据的 min/max
 *
 * 使用 Thrust 实现，完全在 GPU 上计算
 */
void compute_minmax_dev(const float* data_dev, size_t size, float& min_out, float& max_out,
                        cudaStream_t stream = 0);

/**
 * @brief 分时间步计算 min/max 并使用 EMA 更新（GPU 版本）
 *
 * 适用于输入/输出数据，EMA 可以平滑掉噪声异常值
 * 使用批量 kernel 一次计算所有时间步的 min/max，然后 CPU 端做 EMA 融合
 */
void compute_minmax_per_step_ema_gpu(const float* data_dev, int steps, int step_size,
                                      float& min_out, float& max_out, float decay = 0.9f,
                                      cudaStream_t stream = 0);

/**
 * @brief 分时间步计算 min/max 并使用全局极值累积（GPU 版本）
 *
 * 适用于中间计算结果，使用全局极值更稳定，与 AIMET MinMax 一致
 * 使用批量 kernel 一次计算所有时间步的 min/max，然后 CPU 端取全局极值
 */
void compute_minmax_per_step_gpu(const float* data_dev, int steps, int step_size,
                                  float& min_out, float& max_out,
                                  cudaStream_t stream = 0);

/**
 * @brief 计算 per-channel 的 min/max（GPU 版本）
 *
 * 数据布局: [input_size, channel_size]（行主序）
 * 使用 GPU kernel 批量计算所有 channel 的 min/max
 */
void compute_minmax_per_channel_gpu(const float* data_dev, size_t input_size, size_t channel_size,
                                     std::vector<float>& min_out, std::vector<float>& max_out,
                                     cudaStream_t stream = 0);

/**
 * @brief 从 v 张量提取中间值并计算范围（GPU 版本）
 *
 * v 布局: [T, B, H*4] = [z, r, g, Rh_add_br]
 * 在 GPU 上提取并计算 7 个派生量的 min/max，避免大量数据传输
 */
void update_ranges_from_v_gpu(const float* h_dev, const float* v_dev, size_t steps,
                               size_t hidden_size, size_t batch_size,
                               float& min_z_out, float& max_z_out,
                               float& min_r_out, float& max_r_out,
                               float& min_g_out, float& max_g_out,
                               float& min_Rh_add_br, float& max_Rh_add_br,
                               float& min_rRh, float& max_rRh,
                               float& min_new_contrib, float& max_new_contrib,
                               float& min_old_contrib, float& max_old_contrib,
                               cudaStream_t stream = 0);

}  // namespace gpu_hist

// ============================================================================
// GPU MINMAX 量化范围更新函数
// ============================================================================

struct GRUQuantizationRanges;  // 前向声明

/**
 * @brief GPU 加速的 MINMAX 量化范围更新
 *
 * 完全在 GPU 上计算 min/max，避免大量 GPU→CPU 数据传输
 * 仅传输最终的 min/max 结果（几十个 float）
 *
 * 性能对比（T=100, B=32, H=256, I=128）：
 *   - CPU 版本（updateGRUQuantizationRanges）：~15-20 ms（主要是 D2H 传输）
 *   - GPU 版本（updateGRUQuantizationRangesGPU）：~1-2 ms
 *   - 加速比：8-15x
 *
 * @param time_steps 时间步数
 * @param batch_size 批大小
 * @param input_size 输入维度
 * @param hidden_size 隐藏层维度
 * @param W 输入权重 [I, H*3]（GPU 端）
 * @param R 隐藏权重 [H, H*3]（GPU 端）
 * @param bx 输入偏置 [H*3]（GPU 端）
 * @param br 隐藏偏置 [H*3]（GPU 端）
 * @param x 输入序列 [T, B, I]（GPU 端）
 * @param h 隐藏状态 [(T+1), B, H]（GPU 端）
 * @param v 中间值 [T, B, H*4]（GPU 端）
 * @param tmp_Wx Wx 计算结果 [T, B, H*3]（GPU 端）
 * @param tmp_Rh Rh 计算结果 [T, B, H*3]（GPU 端）
 * @param z_pres z 门预激活值 [T*B*H]（GPU 端）
 * @param r_pres r 门预激活值 [T*B*H]（GPU 端）
 * @param g_pres g 门预激活值 [T*B*H]（GPU 端）
 * @param pres_size 预激活值数组大小
 * @param quant_ranges 输出量化范围（原地更新）
 * @param stream CUDA 流
 */
void updateGRUQuantizationRangesGPU(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float* W, const float* R, const float* bx, const float* br,
    const float* x, const float* h, const float* v,
    const float* tmp_Wx, const float* tmp_Rh,
    const float* z_pres, const float* r_pres, const float* g_pres,
    size_t pres_size,
    GRUQuantizationRanges& quant_ranges,
    cudaStream_t stream = 0);

// ============================================================================
// 转换函数：GPU 直方图 -> CPU Histogram
// ============================================================================

struct Histogram;  // 前向声明（来自 histogram_collector.h）

/**
 * @brief 将 GPU 直方图转换为 CPU Histogram 结构
 */
Histogram gpu_histogram_to_cpu(const GPUHistogram& gpu_hist);


