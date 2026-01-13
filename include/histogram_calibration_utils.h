// =====================================================================
// 直方图校准辅助函数（histogram_calibration_utils.h）
// =====================================================================
// 提供直方图收集相关的辅助函数，包括：
// - 从 GPU 数据收集直方图
// - 分时间步收集直方图
// - per-channel 直方图收集
// - CPU/GPU 版本的全量直方图收集
// =====================================================================

#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "histogram_collector.h"
#include "calibration_gpu.cuh"

// =====================================================================
// 模板辅助函数（从 GPU 数据收集直方图）
// =====================================================================

/// 从 GPU 数据收集直方图
template <typename T>
inline void collectHistogramFromDevice(HistogramCollector &collector, const T *data_dev,
                                       size_t size) {
    if (size == 0) return;

    // 拷贝到 host
    std::vector<T> data_host(size);
    cudaMemcpy(data_host.data(), data_dev, size * sizeof(T), cudaMemcpyDeviceToHost);

    // 转换为 float 并收集直方图
    std::vector<float> data_float(size);
    for (size_t i = 0; i < size; ++i) {
        data_float[i] = static_cast<float>(data_host[i]);
    }

    collector.collect(data_float.data(), data_float.size());
}

/// 分时间步收集直方图（用于时序数据）
template <typename T>
inline void collectHistogramPerStep(HistogramCollector &collector, const T *data_dev, int steps,
                                    int step_size) {
    std::vector<T> data_host(steps * step_size);
    cudaMemcpy(data_host.data(), data_dev, steps * step_size * sizeof(T), cudaMemcpyDeviceToHost);

    // 分时间步收集（模拟多批次累积）
    std::vector<float> step_float(step_size);
    for (int t = 0; t < steps; ++t) {
        const T *step_data = data_host.data() + t * step_size;
        for (int i = 0; i < step_size; ++i) {
            step_float[i] = static_cast<float>(step_data[i]);
        }
        collector.collect(step_float.data(), step_float.size());
    }
}

/// per-channel 直方图收集
template <typename T>
inline void collectPerChannelHistograms(std::vector<HistogramCollector> &collectors,
                                        const T *data_dev, int input_size, int channel_size) {
    std::vector<T> data_host(input_size * channel_size);
    cudaMemcpy(data_host.data(), data_dev, input_size * channel_size * sizeof(T),
               cudaMemcpyDeviceToHost);

    // 为每个 channel 收集直方图
    for (int c = 0; c < channel_size; ++c) {
        std::vector<float> channel_data(input_size);
        for (int i = 0; i < input_size; ++i) {
            channel_data[i] = static_cast<float>(data_host[i * channel_size + c]);
        }
        collectors[c].collect(channel_data.data(), channel_data.size());
    }
}

// =====================================================================
// CPU 直方图收集函数声明
// =====================================================================

/// 收集所有 GRU 中间值的直方图（CPU 版本）
/// 需要将 GPU 数据拷贝到 CPU 进行收集
void collectAllHistograms(
    GRUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *Wx_add_bw, const float *Rh_add_br,
    const float *W, const float *R, const float *bw, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    // 预激活值（z_pre, r_pre, g_pre）- 可选，传 nullptr 则跳过
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size);

// =====================================================================
// GPU 直方图收集函数声明
// =====================================================================

/// 使用 GPU 收集所有直方图（高性能版本）
/// 所有直方图计算都在 GPU 上完成，避免大量 GPU->CPU 数据传输
/// 使用 CUDA streams 并行收集多个直方图
void collectAllHistogramsGPU(
    GRUGPUHistogramCollectors &hist_collectors,
    const float *x, const float *h, const float *v,
    const float *Wx_add_bw, const float *Rh_add_br,
    const float *W, const float *R, const float *bw, const float *br,
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float *z_pres, const float *r_pres, const float *g_pres, size_t pres_size,
    cudaStream_t stream = 0);


