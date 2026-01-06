// ============================================================================
// histogram_gpu.cu - GPU 加速直方图实现
// ============================================================================

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cfloat>  // for FLT_MAX
#include <cmath>
#include <limits>

#include "calibration_gpu.cuh"
#include "histogram_collector.h"  // for Histogram struct and get_minimum_scale
#include "parallel_algorithm.h"   // for dev::fill_n
#include "gru_quantization_ranges.h"  // for GRUQuantizationRanges

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief 直方图构建核心 Kernel（与 AIMET 完全一致）
 *
 * 使用 atomicAdd 累加到全局直方图
 * 与 AIMET torch.histc + 边界外统计完全一致：
 * - inf/NaN 被忽略
 * - 边界外的值加到边界 bin
 */
__global__ void histogram_kernel(const float* __restrict__ data, size_t size,
                                  float* __restrict__ counts, float min_val,
                                  float max_val, float inv_bin_width, int num_bins) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = data[idx];
    
    // 跳过 inf/NaN（与 AIMET torch.histc 行为一致）
    if (!isfinite(val)) return;
    
    // 与 AIMET 一致：边界外的值加到边界 bin
    if (val < min_val) {
        atomicAdd(&counts[0], 1.0f);  // histogram[0] += sum(input < bin_edges[0])
    } else if (val > max_val) {
        atomicAdd(&counts[num_bins - 1], 1.0f);  // histogram[-1] += sum(input > bin_edges[-1])
    } else {
        // 范围内的值：与 AIMET _get_bin_num 一致（只有上界 clamp）
        int bin_idx = static_cast<int>((val - min_val) * inv_bin_width);
        bin_idx = min(bin_idx, num_bins - 1);
        atomicAdd(&counts[bin_idx], 1.0f);
    }
}

/**
 * @brief 使用共享内存优化的直方图 Kernel（与 AIMET 完全一致）
 *
 * 每个 block 先在共享内存中累加，最后再 atomicAdd 到全局
 * 当 num_bins <= 2048 时效率更高
 * 与 AIMET torch.histc + 边界外统计完全一致
 */
__global__ void histogram_kernel_shared(const float* __restrict__ data, size_t size,
                                         float* __restrict__ counts, float min_val,
                                         float max_val, float inv_bin_width, int num_bins) {
    extern __shared__ float shared_hist[];

    // 初始化共享内存
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0.0f;
    }
    __syncthreads();

    // 每个线程处理多个元素（与 AIMET 完全一致）
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += stride) {
        float val = data[idx];
        
        // 跳过 inf/NaN（与 AIMET torch.histc 行为一致）
        if (!isfinite(val)) continue;
        
        // 与 AIMET 一致：边界外的值加到边界 bin
        if (val < min_val) {
            atomicAdd(&shared_hist[0], 1.0f);
        } else if (val > max_val) {
            atomicAdd(&shared_hist[num_bins - 1], 1.0f);
        } else {
            int bin_idx = static_cast<int>((val - min_val) * inv_bin_width);
            bin_idx = min(bin_idx, num_bins - 1);
            atomicAdd(&shared_hist[bin_idx], 1.0f);
        }
    }
    __syncthreads();

    // 将共享内存结果累加到全局
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&counts[i], shared_hist[i]);
        }
    }
}

/**
 * @brief 直方图重分配 Kernel（与 AIMET _HistogramObserver.merge_stats 完全一致）
 *
 * 将源直方图按比例分配到新范围的目标直方图
 * 
 * 算法：对于每个源 bin，计算其落入目标 bin 的比例：
 *   - split_ratio = (dest_bin_end - src_bin_start) / src_bin_width
 *   - 第一个目标 bin 获得 round(split_ratio * count)
 *   - 剩余部分分配到下一个目标 bin
 */
__global__ void redistribute_histogram_kernel(const float* __restrict__ src_counts, float src_min,
                                               float src_bin_width, float* __restrict__ dst_counts,
                                               float dst_min, float dst_bin_width,
                                               int num_bins) {
    const int src_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (src_idx >= num_bins) return;

    float count = src_counts[src_idx];
    if (count <= 0) return;

    // 计算 inv_bin_width 用于索引计算
    float dst_inv_bin_width = 1.0f / dst_bin_width;

    // 源 bin 的起始位置
    float src_bin_start = src_min + src_idx * src_bin_width;

    // 计算落入的目标 bin 索引（与 AIMET _get_bin_num 完全一致：只有上界 clamp）
    int dst_idx = static_cast<int>((src_bin_start - dst_min) * dst_inv_bin_width);
    dst_idx = min(dst_idx, num_bins - 1);

    // 目标 bin 的结束位置
    float dst_bin_end = dst_min + dst_bin_width * (dst_idx + 1);

    // 计算分割比例（与 AIMET split_hist_value 完全一致：不对 ratio clamp）
    float split_hist_value = roundf(
        ((dst_bin_end - src_bin_start) / src_bin_width) * count
    );
    float first_bin_count = fminf(split_hist_value, count);

    // 添加到第一个目标 bin
    atomicAdd(&dst_counts[dst_idx], first_bin_count);

    // 剩余部分添加到下一个 bin（与 AIMET other_bin_updates 完全一致）
    float remaining = count - first_bin_count;
    if (remaining > 0) {
        // 与 AIMET 一致：使用 _get_bin_num 计算 other_bin_index
        int other_bin_idx = static_cast<int>((src_bin_start + dst_bin_width - dst_min) * dst_inv_bin_width);
        other_bin_idx = min(other_bin_idx, num_bins - 1);
        atomicAdd(&dst_counts[other_bin_idx], remaining);
    }
}

/**
 * @brief 从 v 张量提取门值并计算派生量的 Kernel
 *
 * v 布局: [T, B, H*4] = [z, r, g, Rh_add_br]
 * 输出: z_out, r_out, g_out, Rh_add_br, rRh, new_contrib, old_contrib
 */
__global__ void extract_gate_values_kernel(const float* __restrict__ v,
                                            const float* __restrict__ h,
                                            float* __restrict__ z_out,
                                            float* __restrict__ r_out,
                                            float* __restrict__ g_out,
                                            float* __restrict__ Rh_add_br,
                                            float* __restrict__ rRh,
                                            float* __restrict__ new_contrib,
                                            float* __restrict__ old_contrib,
                                            int time_steps, int batch_size, int hidden_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = time_steps * batch_size * hidden_size;
    if (idx >= total) return;

    // 计算 t, b, hh 索引
    const int hh = idx % hidden_size;
    const int b = (idx / hidden_size) % batch_size;
    const int t = idx / (batch_size * hidden_size);

    // v 索引: [t, b, h*4 + offset]
    const int v_base = t * batch_size * hidden_size * 4 + b * hidden_size * 4;
    const float z_val = v[v_base + 0 * hidden_size + hh];
    const float r_val = v[v_base + 1 * hidden_size + hh];
    const float g_val = v[v_base + 2 * hidden_size + hh];
    const float Rh_add_br_val = v[v_base + 3 * hidden_size + hh];

    // h 索引: [t, b, h]（h_old 是当前时间步的输入，即 h[t] 而非 h[t+1]）
    const int h_base = t * batch_size * hidden_size + b * hidden_size;
    const float h_old = h[h_base + hh];

    // 输出
    z_out[idx] = z_val;
    r_out[idx] = r_val;
    g_out[idx] = g_val;
    Rh_add_br[idx] = Rh_add_br_val;
    rRh[idx] = r_val * Rh_add_br_val;
    new_contrib[idx] = (1.0f - z_val) * g_val;
    old_contrib[idx] = z_val * h_old;
}

/**
 * @brief 提取 per-channel 数据 Kernel
 *
 * 将 [input_size, channel_size] 的数据按 channel 分离出来
 */
__global__ void extract_channel_kernel(const float* __restrict__ data,
                                        float* __restrict__ channel_data,
                                        int input_size, int channel_size, int target_channel) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input_size) return;

    channel_data[i] = data[i * channel_size + target_channel];
}

/**
 * @brief 批量计算所有 channel 的 min/max（一个 kernel 搞定）
 *
 * 数据布局: [input_size, channel_size]（行主序）
 * 输出: mins[channel_size], maxs[channel_size]
 * 
 * 使用 shared memory 做 per-block reduction，然后 atomic 更新全局结果
 * 与 AIMET _get_min_max 一致：过滤 inf/NaN 值
 */
__global__ void compute_per_channel_minmax_kernel(
    const float* __restrict__ data,
    float* __restrict__ mins,
    float* __restrict__ maxs,
    int input_size,
    int channel_size) {
    
    // 每个 block 处理一个 channel
    const int channel = blockIdx.x;
    if (channel >= channel_size) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* s_min = shared_mem;
    float* s_max = shared_mem + blockDim.x;
    
    // 初始化
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    // 每个线程处理多个元素（与 AIMET _get_min_max 一致：过滤 inf/NaN）
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        float val = data[i * channel_size + channel];
        if (isfinite(val)) {  // 过滤 inf 和 NaN
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
    }
    
    // 存入 shared memory
    s_min[threadIdx.x] = local_min;
    s_max[threadIdx.x] = local_max;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = fminf(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    // 写入全局结果（如果所有值都是 inf/NaN，min > max，后续会处理）
    if (threadIdx.x == 0) {
        mins[channel] = s_min[0];
        maxs[channel] = s_max[0];
    }
}

/**
 * @brief 批量计算多个独立数组的 min/max（一个 kernel 搞定）
 *
 * 每个 block 处理一个数组，使用 shared memory reduction
 * 用于 gate histograms 等场景（7 个独立数组）
 * 与 AIMET _get_min_max 一致：过滤 inf/NaN 值
 */
__global__ void compute_batch_minmax_kernel(
    const float* const* __restrict__ data_ptrs,  // 数组指针数组
    const size_t* __restrict__ sizes,            // 各数组大小
    float* __restrict__ mins,                    // 输出 min
    float* __restrict__ maxs,                    // 输出 max
    int num_arrays) {
    
    const int arr_idx = blockIdx.x;
    if (arr_idx >= num_arrays) return;
    
    const float* data = data_ptrs[arr_idx];
    const size_t size = sizes[arr_idx];
    
    extern __shared__ float shared_mem[];
    float* s_min = shared_mem;
    float* s_max = shared_mem + blockDim.x;
    
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    // 与 AIMET _get_min_max 一致：过滤 inf/NaN
    for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
        float val = data[i];
        if (isfinite(val)) {  // 过滤 inf 和 NaN
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
    }
    
    s_min[threadIdx.x] = local_min;
    s_max[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = fminf(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        mins[arr_idx] = s_min[0];
        maxs[arr_idx] = s_max[0];
    }
}

/**
 * @brief 批量计算连续存储的多个数组的 min/max
 *
 * 数据布局: data[num_arrays * array_size]（连续存储）
 * 每个 block 处理一个数组
 * 用于 per-step EMA 场景
 * 与 AIMET _get_min_max 一致：过滤 inf/NaN 值
 */
__global__ void compute_contiguous_batch_minmax_kernel(
    const float* __restrict__ data,    // 连续数据 [num_arrays, array_size]
    float* __restrict__ mins,          // 输出 min [num_arrays]
    float* __restrict__ maxs,          // 输出 max [num_arrays]
    int array_size,                    // 每个数组的大小
    int num_arrays) {                  // 数组数量
    
    const int arr_idx = blockIdx.x;
    if (arr_idx >= num_arrays) return;
    
    // 定位到当前数组的起始位置
    const float* my_data = data + arr_idx * array_size;
    
    extern __shared__ float shared_mem[];
    float* s_min = shared_mem;
    float* s_max = shared_mem + blockDim.x;
    
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    // 每个线程处理多个元素（与 AIMET _get_min_max 一致：过滤 inf/NaN）
    for (int i = threadIdx.x; i < array_size; i += blockDim.x) {
        float val = my_data[i];
        if (isfinite(val)) {  // 过滤 inf 和 NaN
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
    }
    
    s_min[threadIdx.x] = local_min;
    s_max[threadIdx.x] = local_max;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = fminf(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        mins[arr_idx] = s_min[0];
        maxs[arr_idx] = s_max[0];
    }
}

/**
 * @brief 批量构建所有 channel 的直方图（一个 kernel 搞定）
 *
 * 数据布局: [input_size, channel_size]
 * 输出: counts[channel_size * num_bins]（每个 channel 一个直方图）
 */
__global__ void build_per_channel_histogram_kernel(
    const float* __restrict__ data,
    float* __restrict__ counts,  // [channel_size, num_bins]
    const float* __restrict__ mins,
    const float* __restrict__ maxs,
    int input_size,
    int channel_size,
    int num_bins) {
    
    // 每个 block 处理一个 channel
    const int channel = blockIdx.x;
    if (channel >= channel_size) return;
    
    float min_val = mins[channel];
    float max_val = maxs[channel];
    float bin_width = (max_val - min_val) / num_bins;
    if (bin_width < 1e-9f) bin_width = 1e-9f;
    float inv_bin_width = 1.0f / bin_width;
    
    // 该 channel 的直方图起始位置
    float* my_counts = counts + channel * num_bins;
    
    // 使用 shared memory 累加（避免 global atomic 竞争）
    extern __shared__ float s_hist[];
    
    // 初始化 shared histogram
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0.0f;
    }
    __syncthreads();
    
    // 每个线程处理多个元素（与 AIMET 完全一致）
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        float val = data[i * channel_size + channel];
        
        // 跳过 inf/NaN（与 AIMET torch.histc 行为一致）
        if (!isfinite(val)) continue;
        
        // 与 AIMET 一致：边界外的值加到边界 bin
        if (val < min_val) {
            atomicAdd(&s_hist[0], 1.0f);
        } else if (val > max_val) {
            atomicAdd(&s_hist[num_bins - 1], 1.0f);
        } else {
            int bin_idx = static_cast<int>((val - min_val) * inv_bin_width);
            bin_idx = min(bin_idx, num_bins - 1);
            atomicAdd(&s_hist[bin_idx], 1.0f);
        }
    }
    __syncthreads();
    
    // 写回 global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        my_counts[i] = s_hist[i];
    }
}

// ============================================================================
// GPU 直方图辅助函数实现
// ============================================================================

namespace gpu_hist {

// 用于过滤 inf/NaN 的自定义 min/max 二元操作（与 AIMET _get_min_max 一致）
struct finite_min_op {
    __host__ __device__ float operator()(float a, float b) const {
        bool a_finite = isfinite(a);
        bool b_finite = isfinite(b);
        if (!a_finite && !b_finite) return FLT_MAX;  // 都无效，返回最大值
        if (!a_finite) return b;
        if (!b_finite) return a;
        return fminf(a, b);
    }
};

struct finite_max_op {
    __host__ __device__ float operator()(float a, float b) const {
        bool a_finite = isfinite(a);
        bool b_finite = isfinite(b);
        if (!a_finite && !b_finite) return -FLT_MAX;  // 都无效，返回最小值
        if (!a_finite) return b;
        if (!b_finite) return a;
        return fmaxf(a, b);
    }
};

void compute_minmax(const float* data_dev, size_t size, float& min_val, float& max_val) {
    if (size == 0) {
        min_val = 0;
        max_val = 0;
        return;
    }

    thrust::device_ptr<const float> data_ptr(data_dev);
    
    // 使用 Thrust reduce 计算 min/max，过滤 inf/NaN（与 AIMET _get_min_max 一致）
    min_val = thrust::reduce(thrust::device, data_ptr, data_ptr + size, FLT_MAX, finite_min_op());
    max_val = thrust::reduce(thrust::device, data_ptr, data_ptr + size, -FLT_MAX, finite_max_op());
    
    // 如果所有值都是 inf/NaN，使用默认范围
    if (min_val > max_val) {
        min_val = 0.0f;
        max_val = 0.0f;
    }
}

void build_histogram(const float* data_dev, size_t size, float* counts_dev, float min_val,
                     float max_val, int num_bins, cudaStream_t stream) {
    if (size == 0 || num_bins <= 0) return;

    float bin_width = (max_val - min_val) / num_bins;
    if (bin_width < 1e-9f) bin_width = 1e-9f;
    float inv_bin_width = 1.0f / bin_width;

    const int threads = 256;

    // 选择合适的 kernel（与 AIMET 完全一致：包含边界外值处理）
    if (num_bins <= 2048) {
        // 使用共享内存优化版本
        const int blocks = std::min(static_cast<int>((size + threads - 1) / threads), 256);
        const size_t shared_mem_size = num_bins * sizeof(float);
        histogram_kernel_shared<<<blocks, threads, shared_mem_size, stream>>>(
            data_dev, size, counts_dev, min_val, max_val, inv_bin_width, num_bins);
    } else {
        // 使用简单的全局 atomic 版本
        const int blocks = (size + threads - 1) / threads;
        histogram_kernel<<<blocks, threads, 0, stream>>>(data_dev, size, counts_dev, min_val,
                                                          max_val, inv_bin_width, num_bins);
    }
}

void redistribute_histogram(const float* src_counts_dev, float src_min, float src_max,
                            float* dst_counts_dev, float dst_min, float dst_max, int num_bins,
                            cudaStream_t stream) {
    if (num_bins <= 0) return;

    float src_bin_width = (src_max - src_min) / num_bins;
    float dst_bin_width = (dst_max - dst_min) / num_bins;
    if (dst_bin_width < 1e-9f) dst_bin_width = 1e-9f;

    const int threads = 256;
    const int blocks = (num_bins + threads - 1) / threads;

    redistribute_histogram_kernel<<<blocks, threads, 0, stream>>>(
        src_counts_dev, src_min, src_bin_width, dst_counts_dev, dst_min, dst_bin_width,
        num_bins);
}

void collect_gate_histograms(GRUGPUHistogramCollectors& collectors, const float* v_dev,
                             const float* h_dev, int time_steps, int batch_size, int hidden_size,
                             cudaStream_t stream) {
    const size_t total_size = time_steps * batch_size * hidden_size;

    // 分配临时 GPU 缓冲区
    dev::vector<float> z_out_dev(total_size);
    dev::vector<float> r_out_dev(total_size);
    dev::vector<float> g_out_dev(total_size);
    dev::vector<float> Rh_add_br_dev(total_size);
    dev::vector<float> rRh_dev(total_size);
    dev::vector<float> new_contrib_dev(total_size);
    dev::vector<float> old_contrib_dev(total_size);

    // 在 GPU 上提取并计算所有门值
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    extract_gate_values_kernel<<<blocks, threads, 0, stream>>>(
        v_dev, h_dev, z_out_dev.data(), r_out_dev.data(), g_out_dev.data(), Rh_add_br_dev.data(),
        rRh_dev.data(), new_contrib_dev.data(), old_contrib_dev.data(), time_steps, batch_size,
        hidden_size);

    // 等待 extract kernel 完成
    cudaStreamSynchronize(stream);

    // 收集直方图（使用 collect 函数，与 AIMET 行为一致：支持多批次范围扩展）
    collectors.z_out_hist.collect(z_out_dev.data(), total_size, stream);
    collectors.r_out_hist.collect(r_out_dev.data(), total_size, stream);
    collectors.g_out_hist.collect(g_out_dev.data(), total_size, stream);
    collectors.Rh_add_br_g_hist.collect(Rh_add_br_dev.data(), total_size, stream);
    collectors.rRh_hist.collect(rRh_dev.data(), total_size, stream);
    collectors.new_contrib_hist.collect(new_contrib_dev.data(), total_size, stream);
    collectors.old_contrib_hist.collect(old_contrib_dev.data(), total_size, stream);
}

void collect_per_channel_histograms(std::vector<GPUHistogramCollector>& collectors,
                                    const float* data_dev, int input_size, int channel_size,
                                    cudaStream_t stream) {
    if (channel_size == 0 || input_size == 0) return;
    
    const int num_bins = collectors[0].histogram().num_bins;
    
    // 1. 分配批量 GPU 缓冲区
    dev::vector<float> all_mins(channel_size);
    dev::vector<float> all_maxs(channel_size);
    dev::vector<float> all_counts(channel_size * num_bins);
    all_counts.zero();
    
    // 2. 批量计算所有 channel 的 min/max（已过滤 inf/NaN）
    {
        const int threads = 256;
        const int blocks = channel_size;
        const size_t shared_mem = 2 * threads * sizeof(float);
        compute_per_channel_minmax_kernel<<<blocks, threads, shared_mem, stream>>>(
            data_dev, all_mins.data(), all_maxs.data(), input_size, channel_size);
    }
    
    // 3. 拷贝 min/max 到 CPU 进行范围预处理（与 AIMET 一致）
    std::vector<float> h_mins(channel_size);
    std::vector<float> h_maxs(channel_size);
    cudaMemcpyAsync(h_mins.data(), all_mins.data(), channel_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_maxs.data(), all_maxs.data(), channel_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 4. 范围预处理：与 CPU HistogramCollector::collect() 和 AIMET 完全一致
    const float minimum_scale = get_minimum_scale(num_bins);
    const float minimum_range = minimum_scale * num_bins;
    for (int c = 0; c < channel_size; ++c) {
        // 处理所有值都是 inf/NaN 的情况（min > max）
        if (h_mins[c] > h_maxs[c]) {
            h_mins[c] = 0.0f;
            h_maxs[c] = 0.0f;
        }
        
        // 与 AIMET _create_bin_edges 一致：如果 min == max，使用 ±0.5 扩展
        if (h_mins[c] == h_maxs[c]) {
            h_mins[c] = h_mins[c] - 0.5f;
            h_maxs[c] = h_maxs[c] + 0.5f;
        }
        
        float input_range = h_maxs[c] - h_mins[c];
        if (input_range < minimum_range || std::isnan(input_range) || std::isinf(input_range)) {
            // 确保 0 在范围内（与 CPU 一致）
            h_mins[c] = std::min(h_mins[c], 0.0f);
            h_maxs[c] = std::max(h_maxs[c], 0.0f);
            // 基于 min 扩展范围
            h_maxs[c] = h_mins[c] + minimum_range;
        }
    }
    
    // 5. 将预处理后的范围拷贝回 GPU
    cudaMemcpyAsync(all_mins.data(), h_mins.data(), channel_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(all_maxs.data(), h_maxs.data(), channel_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    
    // 6. 使用预处理后的范围构建直方图
    {
        const int threads = 256;
        const int blocks = channel_size;
        const size_t shared_mem = num_bins * sizeof(float);
        build_per_channel_histogram_kernel<<<blocks, threads, shared_mem, stream>>>(
            data_dev, all_counts.data(), all_mins.data(), all_maxs.data(),
            input_size, channel_size, num_bins);
    }
    cudaStreamSynchronize(stream);
    
    // 7. 设置各 collector 的直方图元数据和计数
    for (int c = 0; c < channel_size; ++c) {
        GPUHistogram& hist = collectors[c].histogram();
        hist.min_val = h_mins[c];
        hist.max_val = h_maxs[c];
        hist.total_count = input_size;
        
        // D2D: 完全在 GPU 上
        cudaMemcpyAsync(hist.counts.data(), all_counts.data() + c * num_bins,
                       num_bins * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);
}

/**
 * @brief 批量收集 per-channel 直方图（直接写入 PerChannelHistogramBatch）
 *
 * 优化版本：直接把结果写入共享缓冲区，零拷贝
 */
void collect_per_channel_histograms_batch(PerChannelHistogramBatch& batch,
                                           const float* data_dev, int input_size,
                                           cudaStream_t stream) {
    if (batch.channel_size == 0 || input_size == 0) return;
    
    const int channel_size = batch.channel_size;
    const int num_bins = batch.num_bins;
    
    // 临时 GPU 缓冲区存储 min/max
    dev::vector<float> d_mins(channel_size);
    dev::vector<float> d_maxs(channel_size);
    
    // 1. 批量计算 min/max
    {
        const int threads = 256;
        const int blocks = channel_size;
        const size_t shared_mem = 2 * threads * sizeof(float);
        compute_per_channel_minmax_kernel<<<blocks, threads, shared_mem, stream>>>(
            data_dev, d_mins.data(), d_maxs.data(), input_size, channel_size);
    }
    
    // 2. 拷贝 min/max 到 CPU 进行范围扩展
    cudaMemcpyAsync(batch.mins.data(), d_mins.data(), channel_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(batch.maxs.data(), d_maxs.data(), channel_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 保存原始 min/max（用于位宽约束检查）
    batch.original_mins = batch.mins;
    batch.original_maxs = batch.maxs;
    
    // 3. 范围扩展：与 CPU HistogramCollector::collect() 和 AIMET 完全一致
    // 必须在构建直方图之前进行，以确保 bin 分布一致
    const float minimum_scale = get_minimum_scale(num_bins);
    const float minimum_range = minimum_scale * num_bins;
    for (int c = 0; c < channel_size; ++c) {
        // 处理所有值都是 inf/NaN 的情况（min > max）
        if (batch.mins[c] > batch.maxs[c]) {
            batch.mins[c] = 0.0f;
            batch.maxs[c] = 0.0f;
        }
        
        // 与 AIMET _create_bin_edges 一致：如果 min == max，使用 ±0.5 扩展
        if (batch.mins[c] == batch.maxs[c]) {
            batch.mins[c] = batch.mins[c] - 0.5f;
            batch.maxs[c] = batch.maxs[c] + 0.5f;
        }
        
        float input_range = batch.maxs[c] - batch.mins[c];
        if (input_range < minimum_range || std::isnan(input_range) || std::isinf(input_range)) {
            // 确保 0 在范围内（与 CPU 一致）
            batch.mins[c] = std::min(batch.mins[c], 0.0f);
            batch.maxs[c] = std::max(batch.maxs[c], 0.0f);
            // 基于 min 扩展范围
            batch.maxs[c] = batch.mins[c] + minimum_range;
        }
    }
    
    // 4. 将扩展后的范围拷贝回 GPU
    cudaMemcpyAsync(d_mins.data(), batch.mins.data(), channel_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_maxs.data(), batch.maxs.data(), channel_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    
    // 5. 使用扩展后的范围构建直方图
    batch.counts.zero();
    {
        const int threads = 256;
        const int blocks = channel_size;
        const size_t shared_mem = num_bins * sizeof(float);
        build_per_channel_histogram_kernel<<<blocks, threads, shared_mem, stream>>>(
            data_dev, batch.counts.data(), d_mins.data(), d_maxs.data(),
            input_size, channel_size, num_bins);
    }
    cudaStreamSynchronize(stream);
    
    batch.per_channel_count = input_size;
}

}  // namespace gpu_hist

// ============================================================================
// GPUHistogramCollector 方法实现
// ============================================================================

void GPUHistogramCollector::collect(const float* data_dev, size_t size, cudaStream_t stream) {
    if (size == 0) return;

    // 在 GPU 上计算 min/max（过滤 inf/NaN）
    float data_min, data_max;
    gpu_hist::compute_minmax(data_dev, size, data_min, data_max);

    // 与 AIMET _create_bin_edges 一致：如果 min == max，使用 ±0.5 扩展
    // 这是为了兼容 PyTorch 的 torch.histc 实现
    if (data_min == data_max) {
        data_min = data_min - 0.5f;
        data_max = data_max + 0.5f;
    }

    // 处理特殊情况
    float minimum_scale = get_minimum_scale(config_.num_bins);
    float minimum_range = minimum_scale * config_.num_bins;
    float input_range = data_max - data_min;

    // 如果范围太小，使用最小范围并确保 0 在范围内
    // 与 AIMET merge_stats 中的 zero_range_mask 处理一致
    if (input_range < minimum_range || std::isnan(input_range) || std::isinf(input_range)) {
        data_min = std::min(data_min, 0.0f);
        data_max = std::max(data_max, 0.0f);
        input_range = minimum_range;
        data_max = data_min + minimum_range;
    }

    if (!hist_.is_valid()) {
        // 首次收集：初始化直方图
        hist_.reset(config_.num_bins);
        hist_.min_val = data_min;
        hist_.max_val = data_max;
        _add_to_histogram_gpu(data_dev, size, stream);

        // 设置范围限制
        range_limit_min_ = data_min - input_range * config_.growth_limit / 2.0f;
        range_limit_max_ = data_max + input_range * config_.growth_limit / 2.0f;
        range_limit_set_ = true;
    } else {
        // 后续收集：应用范围限制
        float updated_min = hist_.min_val;
        float updated_max = hist_.max_val;

        if (range_limit_set_) {
            updated_min = std::max(range_limit_min_, std::min(data_min, hist_.min_val));
            updated_max = std::min(range_limit_max_, std::max(data_max, hist_.max_val));
        } else {
            updated_min = std::min(data_min, hist_.min_val);
            updated_max = std::max(data_max, hist_.max_val);
        }

        if (updated_min == hist_.min_val && updated_max == hist_.max_val) {
            // 范围不变，直接添加
            _add_to_histogram_gpu(data_dev, size, stream);
        } else {
            // 需要扩展范围
            _merge_with_extended_range_gpu(data_dev, size, updated_min, updated_max, stream);
        }
    }
}

void GPUHistogramCollector::collectWithKnownRange(const float* data_dev, size_t size,
                                                   float known_min, float known_max,
                                                   cudaStream_t stream) {
    if (size == 0) return;

    // 与 AIMET _create_bin_edges 一致：如果 min == max，使用 ±0.5 扩展
    if (known_min == known_max) {
        known_min = known_min - 0.5f;
        known_max = known_max + 0.5f;
    }

    // 使用已知范围，跳过 minmax 计算
    float minimum_scale = get_minimum_scale(config_.num_bins);
    float minimum_range = minimum_scale * config_.num_bins;
    float input_range = known_max - known_min;

    if (input_range < minimum_range) {
        known_min = std::min(known_min, 0.0f);
        known_max = std::max(known_max, 0.0f);
        input_range = minimum_range;
        known_max = known_min + minimum_range;
    }

    if (!hist_.is_valid()) {
        // 首次收集：初始化直方图
        hist_.reset(config_.num_bins);
        hist_.min_val = known_min;
        hist_.max_val = known_max;
        _add_to_histogram_gpu(data_dev, size, stream);

        // 设置范围限制（与 collect 一致）
        range_limit_min_ = known_min - input_range * config_.growth_limit / 2.0f;
        range_limit_max_ = known_max + input_range * config_.growth_limit / 2.0f;
        range_limit_set_ = true;
    } else {
        // 后续收集：与 collect 函数逻辑一致，支持范围扩展
        float updated_min = hist_.min_val;
        float updated_max = hist_.max_val;

        if (range_limit_set_) {
            // 应用范围限制（与 AIMET clamp 逻辑一致）
            updated_min = std::max(range_limit_min_, std::min(known_min, hist_.min_val));
            updated_max = std::min(range_limit_max_, std::max(known_max, hist_.max_val));
        } else {
            updated_min = std::min(known_min, hist_.min_val);
            updated_max = std::max(known_max, hist_.max_val);
        }

        if (updated_min == hist_.min_val && updated_max == hist_.max_val) {
            // 范围不变，直接添加
            _add_to_histogram_gpu(data_dev, size, stream);
        } else {
            // 需要扩展范围
            _merge_with_extended_range_gpu(data_dev, size, updated_min, updated_max, stream);
        }
    }
}

void GPUHistogramCollector::_add_to_histogram_gpu(const float* data_dev, size_t size,
                                                   cudaStream_t stream) {
    gpu_hist::build_histogram(data_dev, size, hist_.counts.data(), hist_.min_val, hist_.max_val,
                              hist_.num_bins, stream);
    hist_.total_count += size;
}

void GPUHistogramCollector::_merge_with_extended_range_gpu(const float* data_dev, size_t size,
                                                           float new_min, float new_max,
                                                           cudaStream_t stream) {
    // 创建新的直方图计数并清零
    dev::vector<float> new_counts(config_.num_bins);
    dev::fill_n(new_counts.data(), config_.num_bins, 0.0f);

    // 检查旧直方图是否有效
    float src_bin_width = hist_.bin_width();
    float minimum_scale = get_minimum_scale(config_.num_bins);

    if (std::abs(src_bin_width) >= minimum_scale && !std::isnan(src_bin_width) &&
        !std::isinf(src_bin_width)) {
        // 重新分配旧直方图
        gpu_hist::redistribute_histogram(hist_.counts.data(), hist_.min_val, hist_.max_val,
                                         new_counts.data(), new_min, new_max, config_.num_bins,
                                         stream);
    }

    // 添加新数据
    gpu_hist::build_histogram(data_dev, size, new_counts.data(), new_min, new_max, config_.num_bins,
                              stream);

    // 更新直方图
    hist_.counts = std::move(new_counts);
    hist_.min_val = new_min;
    hist_.max_val = new_max;
    hist_.total_count += size;
}

void GPUHistogramCollector::merge(const GPUHistogram& other) {
    if (!other.is_valid()) return;

    if (!hist_.is_valid()) {
        // 首次合并：复制直方图并设置范围限制（与 CPU 版本和 AIMET 一致）
        hist_ = other;
        
        // 设置范围限制（与 collect 一致）
        float input_range = other.max_val - other.min_val;
        
        // 处理零范围情况
        float minimum_scale = get_minimum_scale(config_.num_bins);
        float minimum_range = minimum_scale * config_.num_bins;
        if (input_range < minimum_range) {
            input_range = minimum_range;
        }
        
        range_limit_min_ = other.min_val - input_range * config_.growth_limit / 2.0f;
        range_limit_max_ = other.max_val + input_range * config_.growth_limit / 2.0f;
        range_limit_set_ = true;
        return;
    }

    // 后续合并：应用范围限制（与 CPU 版本和 AIMET 一致）
    float updated_min = hist_.min_val;
    float updated_max = hist_.max_val;
    
    if (range_limit_set_) {
        // 应用范围限制（与 AIMET clamp 逻辑一致）
        updated_min = std::max(range_limit_min_, std::min(other.min_val, hist_.min_val));
        updated_max = std::min(range_limit_max_, std::max(other.max_val, hist_.max_val));
    } else {
        updated_min = std::min(other.min_val, hist_.min_val);
        updated_max = std::max(other.max_val, hist_.max_val);
    }

    if (updated_min == hist_.min_val && updated_max == hist_.max_val) {
        // 范围相同，直接累加
        // 使用 Thrust 进行向量加法
        thrust::device_ptr<float> dst_ptr(hist_.counts.data());
        thrust::device_ptr<const float> src_ptr(other.counts.data());
        thrust::transform(thrust::device, dst_ptr, dst_ptr + hist_.num_bins, src_ptr, dst_ptr,
                          thrust::plus<float>());
        hist_.total_count += other.total_count;
    } else {
        // 范围不同，需要重新分配
        dev::vector<float> new_counts(config_.num_bins);
        dev::fill_n(new_counts.data(), config_.num_bins, 0.0f);

        // 重新分配当前直方图
        gpu_hist::redistribute_histogram(hist_.counts.data(), hist_.min_val, hist_.max_val,
                                         new_counts.data(), updated_min, updated_max, config_.num_bins, 0);

        // 重新分配另一个直方图
        gpu_hist::redistribute_histogram(other.counts.data(), other.min_val, other.max_val,
                                         new_counts.data(), updated_min, updated_max, config_.num_bins, 0);

        hist_.counts = std::move(new_counts);
        hist_.min_val = updated_min;
        hist_.max_val = updated_max;
        hist_.total_count += other.total_count;
    }
}

// ============================================================================
// 转换函数实现
// ============================================================================

Histogram gpu_histogram_to_cpu(const GPUHistogram& gpu_hist) {
    Histogram cpu_hist(gpu_hist.num_bins);
    cpu_hist.min_val = gpu_hist.min_val;
    cpu_hist.max_val = gpu_hist.max_val;
    cpu_hist.total_count = gpu_hist.total_count;
    cpu_hist.counts = gpu_hist.to_host();
    return cpu_hist;
}

// ============================================================================
// GPU SQNR 量化参数计算
// ============================================================================

namespace {

// SQNR 配置现在从 GPUSqnrConfig 参数传入

/**
 * @brief SQNR 噪声计算 Kernel（对称量化，带 clamp 机制）
 *
 * 每个 block 处理一个 delta 候选
 * block 内线程并行计算各 bin 的噪声，然后 reduction
 * 
 * clamp 机制（类似 AIMET _clamp_delta_offset_values）：
 * 如果当前 delta 无法覆盖观察范围，则调整到能覆盖的最小值
 */
__global__ void sqnr_noise_symmetric_kernel(
    const float* __restrict__ counts,
    float min_val, float bin_width, int num_bins,
    float max_delta, int num_delta_candidates,
    int64_t num_steps, float offset,
    float observed_max_abs,  // 观察到的绝对值最大范围
    float gamma, float p,
    float* __restrict__ noise_out)  // [num_delta_candidates]
{
    const int delta_idx = blockIdx.x;
    if (delta_idx >= num_delta_candidates) return;
    
    // 计算当前 delta
    float delta = max_delta * (delta_idx + 1) / (num_delta_candidates - 1);
    delta = fmaxf(delta, 1e-8f);
    
    // ===== clamp 机制：确保 delta 能覆盖观察范围 =====
    const float num_pos_steps = static_cast<float>(num_steps / 2);
    float quant_max_abs = delta * num_pos_steps;
    if (quant_max_abs < observed_max_abs) {
        delta = observed_max_abs / num_pos_steps;
    }
    // =================================================
    
    // Shared memory for reduction
    extern __shared__ float shared_noise[];
    
    float local_noise = 0.0f;
    
    // 每个线程处理多个 bin
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        float count = counts[i];
        if (count < 1e-6f) continue;
        
        float x = min_val + (i + 0.5f) * bin_width;
        
        // AIMET: q = round(x / delta - offset)
        float q = roundf(x / delta - offset);
        
        bool clipped = (q < 0) || (q > static_cast<float>(num_steps));
        q = fmaxf(0.0f, fminf(static_cast<float>(num_steps), q));
        float x_recon = (q + offset) * delta;
        
        float error = powf(fabsf(x_recon - x), p);
        if (clipped) error *= gamma;
        
        local_noise += error * count;
    }
    
    shared_noise[threadIdx.x] = local_noise;
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_noise[threadIdx.x] += shared_noise[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        noise_out[delta_idx] = shared_noise[0];
    }
}

/**
 * @brief SQNR 噪声计算 Kernel（非对称量化）
 *
 * 每个 block 处理一个 (delta_idx, offset_idx) 组合
 * 
 * 参数说明（与 AIMET 一致）：
 * - hist_min: 直方图原始 min，用于计算 bin 中点
 * - search_min/search_max: 搜索范围（已扩展确保包含 0），用于 clamp 操作
 */
__global__ void sqnr_noise_asymmetric_kernel(
    const float* __restrict__ counts,
    float hist_min,        // 直方图原始 min（用于 bin 中点计算）
    float search_min,      // 搜索范围 min（用于 clamp，与 AIMET observed_min 类似）
    float search_max,      // 搜索范围 max（用于 clamp，与 AIMET observed_max 类似）
    float bin_width, int num_bins,
    float max_delta, int num_delta_candidates, int num_offset_candidates,
    int64_t num_steps,
    const float* __restrict__ offsets,  // [num_offset_candidates]
    float gamma, float p,
    float* __restrict__ noise_out)  // [num_delta_candidates * num_offset_candidates]
{
    const int combo_idx = blockIdx.x;
    const int total_combos = num_delta_candidates * num_offset_candidates;
    if (combo_idx >= total_combos) return;
    
    const int delta_idx = combo_idx / num_offset_candidates;
    const int offset_idx = combo_idx % num_offset_candidates;
    
    // 计算 delta
    float delta = max_delta * (delta_idx + 1) / (num_delta_candidates - 1);
    delta = fmaxf(delta, 1e-8f);
    
    // 获取 offset 并 clamp（使用搜索范围，与 AIMET _clamp_delta_offset_values 一致）
    float offset = offsets[offset_idx];
    float test_min = fmaxf(search_min, delta * offset);
    float test_max = fminf(search_max, test_min + delta * num_steps);
    float clamped_delta = fmaxf((test_max - test_min) / num_steps, 1e-8f);
    float clamped_offset = roundf(test_min / clamped_delta);
    
    // Shared memory for reduction
    extern __shared__ float shared_noise[];
    
    float local_noise = 0.0f;
    
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        float count = counts[i];
        if (count < 1e-6f) continue;
        
        // bin 中点使用原始直方图范围（与 AIMET stat.bin_edges 一致）
        float x = hist_min + (i + 0.5f) * bin_width;
        
        float q = roundf(x / clamped_delta - clamped_offset);
        bool clipped = (q < 0) || (q > static_cast<float>(num_steps));
        q = fmaxf(0.0f, fminf(static_cast<float>(num_steps), q));
        float x_recon = (q + clamped_offset) * clamped_delta;
        
        float error = powf(fabsf(x_recon - x), p);
        if (clipped) error *= gamma;
        
        local_noise += error * count;
    }
    
    shared_noise[threadIdx.x] = local_noise;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_noise[threadIdx.x] += shared_noise[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        noise_out[combo_idx] = shared_noise[0];
    }
}

/**
 * @brief 找最小噪声索引的 Kernel
 */
__global__ void find_min_noise_kernel(
    const float* __restrict__ noise, int n,
    int* __restrict__ min_idx, float* __restrict__ min_val)
{
    extern __shared__ float shared_data[];
    float* shared_vals = shared_data;
    int* shared_idxs = (int*)(shared_data + blockDim.x);
    
    float local_min = FLT_MAX;
    int local_idx = 0;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (noise[i] < local_min) {
            local_min = noise[i];
            local_idx = i;
        }
    }
    
    shared_vals[threadIdx.x] = local_min;
    shared_idxs[threadIdx.x] = local_idx;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_vals[threadIdx.x + stride] < shared_vals[threadIdx.x]) {
                shared_vals[threadIdx.x] = shared_vals[threadIdx.x + stride];
                shared_idxs[threadIdx.x] = shared_idxs[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        *min_idx = shared_idxs[0];
        *min_val = shared_vals[0];
    }
}

}  // anonymous namespace

namespace gpu_hist {

void compute_sqnr_params_gpu(const float* counts_dev, float min_val, float max_val,
                              int num_bins, int64_t total_count,
                              bool is_symmetric, QuantBitWidth bw,
                              int8_t& out_exp2_inv, int32_t& out_zp,
                              const GPUSqnrConfig& config,
                              cudaStream_t stream) {
    
    // 从 QuantBitWidth 获取量化范围（支持任意位宽）
    const int64_t quant_min = bw.qmin();
    const int64_t quant_max = bw.qmax();
    const int64_t num_steps = quant_max - quant_min;
    
    // ===== 关键修复：保存原始直方图范围用于计算 bin_width =====
    // bin_width 必须基于直方图收集时的原始范围，而不是扩展后的范围
    // 这与 AIMET 的 _estimate_clip_and_quant_noise 中使用 stat.bin_edges 一致
    const float hist_min = min_val;
    const float hist_max = max_val;
    float bin_width = (hist_max - hist_min) / num_bins;
    if (bin_width < 1e-9f) bin_width = 1e-9f;
    
    // 确保搜索范围包含 0（用于 SQNR 搜索，与 AIMET _pick_test_candidates 一致）
    min_val = (min_val < 0.0f) ? min_val : 0.0f;
    max_val = (max_val > 0.0f) ? max_val : 0.0f;
    float min_range_limit = min_val + 1e-8f * static_cast<float>(num_steps);
    max_val = (max_val > min_range_limit) ? max_val : min_range_limit;
    
    float optimal_scale, optimal_min;
    
    // 观察到的绝对值最大范围（用于对称量化的 clamp 机制）
    // 注意：必须使用原始直方图范围，而不是扩展后的 min_val/max_val
    float observed_max_abs = std::max(std::abs(hist_min), std::abs(hist_max));
    
    if (is_symmetric) {
        // 对称量化
        float abs_max_for_delta = (max_val > -min_val) ? max_val : -min_val;
        float max_delta = 2.0f * abs_max_for_delta / static_cast<float>(num_steps);
        float offset = -static_cast<float>((num_steps + 1) / 2);
        const float num_pos_steps = static_cast<float>(num_steps / 2);
        
        const int num_candidates = config.symmetric_delta_candidates;
        
        // 分配噪声缓冲区
        dev::vector<float> noise_dev(num_candidates);
        
        // 启动 kernel（带 clamp 机制）
        const int threads = 256;
        const int blocks = num_candidates;
        size_t shared_mem = threads * sizeof(float);
        
        // 传入原始 hist_min 和 bin_width（与 CPU estimateNoise 和 AIMET 一致）
        sqnr_noise_symmetric_kernel<<<blocks, threads, shared_mem, stream>>>(
            counts_dev, hist_min, bin_width, num_bins,
            max_delta, num_candidates,
            num_steps, offset, observed_max_abs, config.gamma, config.p, noise_dev.data());
        
        // 找最小噪声
        dev::vector<int> min_idx_dev(1);
        dev::vector<float> min_val_dev(1);
        
        find_min_noise_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int)), stream>>>(
            noise_dev.data(), num_candidates,
            min_idx_dev.data(), min_val_dev.data());
        
        int best_idx;
        cudaMemcpyAsync(&best_idx, min_idx_dev.data(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // 计算 optimal_scale，需要应用与 kernel 相同的 clamp 机制
        optimal_scale = max_delta * (best_idx + 1) / (num_candidates - 1);
        optimal_scale = std::max(optimal_scale, 1e-8f);
        // clamp 机制：确保能覆盖观察范围
        float quant_max_abs = optimal_scale * num_pos_steps;
        if (quant_max_abs < observed_max_abs) {
            optimal_scale = observed_max_abs / num_pos_steps;
        }
        optimal_min = offset * optimal_scale;
        
    } else {
        // 非对称量化
        float max_delta = (max_val - min_val) / static_cast<float>(num_steps);
        
        const int num_delta_candidates = config.asymmetric_delta_candidates;
        
        // 生成 offset 候选
        int num_offsets_limit = static_cast<int>(num_steps + 2);
        const int num_offsets = (num_offsets_limit < config.offset_candidates) ? num_offsets_limit : config.offset_candidates;
        std::vector<float> h_offsets(num_offsets);
        float offset_step = static_cast<float>(num_steps) / (num_offsets - 2);
        for (int o = 0; o < num_offsets - 1; ++o) {
            h_offsets[o] = std::round(-static_cast<float>(num_steps) + o * offset_step);
        }
        h_offsets[num_offsets - 1] = std::round(min_val / max_delta);
        
        dev::vector<float> offsets_dev(num_offsets);
        cudaMemcpyAsync(offsets_dev.data(), h_offsets.data(), num_offsets * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        
        // 分配噪声缓冲区
        int total_combos = num_delta_candidates * num_offsets;
        dev::vector<float> noise_dev(total_combos);
        
        // 启动 kernel
        const int threads = 256;
        const int blocks = total_combos;
        size_t shared_mem = threads * sizeof(float);
        
        // 传入原始 hist_min 和 bin_width（与 CPU estimateNoise 和 AIMET 一致）
        // search_min/search_max 用于 clamp 操作（已扩展确保包含 0）
        sqnr_noise_asymmetric_kernel<<<blocks, threads, shared_mem, stream>>>(
            counts_dev, hist_min, min_val, max_val, bin_width, num_bins,
            max_delta, num_delta_candidates, num_offsets,
            num_steps, offsets_dev.data(), config.gamma, config.p, noise_dev.data());
        
        // 找最小噪声
        dev::vector<int> min_idx_dev(1);
        dev::vector<float> min_val_dev(1);
        
        find_min_noise_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int)), stream>>>(
            noise_dev.data(), total_combos, min_idx_dev.data(), min_val_dev.data());
        
        int best_idx;
        cudaMemcpyAsync(&best_idx, min_idx_dev.data(), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        int best_delta_idx = best_idx / num_offsets;
        int best_offset_idx = best_idx % num_offsets;
        
        float delta = max_delta * (best_delta_idx + 1) / (num_delta_candidates - 1);
        delta = (delta > 1e-8f) ? delta : 1e-8f;
        float offset = h_offsets[best_offset_idx];
        
        float test_min_calc = delta * offset;
        float test_min = (min_val > test_min_calc) ? min_val : test_min_calc;
        float test_max_calc = test_min + delta * static_cast<float>(num_steps);
        float test_max = (max_val < test_max_calc) ? max_val : test_max_calc;
        float scale_calc = (test_max - test_min) / static_cast<float>(num_steps);
        optimal_scale = (scale_calc > 1e-8f) ? scale_calc : 1e-8f;
        optimal_min = std::round(test_min / optimal_scale) * optimal_scale;
    }
    
    // 转换到 POT
    float n = -std::log2(optimal_scale);
    int8_t n_rounded = static_cast<int8_t>(std::round(n));
    
    // 位宽约束：确保量化值不超出范围
    // max(|val|) / scale <= qmax => exp2_inv <= log2(qmax / max(|val|))
    // 注意：使用原始直方图范围进行约束，而不是扩展后的范围
    float max_abs = std::max(std::abs(hist_min), std::abs(hist_max));
    if (max_abs > 1e-10f) {
        int8_t max_exp2_inv = static_cast<int8_t>(std::floor(
            std::log2(static_cast<float>(quant_max) / max_abs)));
        n_rounded = (n_rounded < max_exp2_inv) ? n_rounded : max_exp2_inv;
    }
    
    float po2_scale = std::pow(2.0f, -static_cast<float>(n_rounded));
    
    out_exp2_inv = n_rounded;
    
    // 计算 zp
    if (is_symmetric) {
        out_zp = 0;
    } else {
        float zp_fp = static_cast<float>(quant_min) - optimal_min / po2_scale;
        out_zp = static_cast<int32_t>(std::round(zp_fp));
    }
}

void compute_sqnr_params_batch_gpu(
    const std::vector<const float*>& counts_ptrs,
    const std::vector<float>& mins,
    const std::vector<float>& maxs,
    int num_bins,
    const std::vector<int64_t>& total_counts,
    const std::vector<bool>& is_symmetric,
    QuantBitWidth bw,
    std::vector<int8_t>& out_exp2_inv,
    std::vector<int32_t>& out_zp,
    cudaStream_t stream) {
    
    const int n = counts_ptrs.size();
    out_exp2_inv.resize(n);
    out_zp.resize(n);
    
    GPUSqnrConfig config;  // 使用默认配置
    
    // 简单实现：串行处理每个直方图
    // TODO: 可以进一步优化为真正的批量并行
    for (int i = 0; i < n; ++i) {
        compute_sqnr_params_gpu(
            counts_ptrs[i], mins[i], maxs[i], num_bins, total_counts[i],
            is_symmetric[i], bw, out_exp2_inv[i], out_zp[i], config, stream);
    }
}

void compute_sqnr_per_channel_gpu(
    const PerChannelHistogramBatch& batch,
    bool is_symmetric, QuantBitWidth bw,
    std::vector<int8_t>& out_exp2_inv,
    const GPUSqnrConfig& config,
    cudaStream_t stream) {
    
    const int n = batch.channel_size;
    if (n == 0 || !batch.is_valid()) {
        out_exp2_inv.clear();
        return;
    }
    
    out_exp2_inv.resize(n);
    
    // 使用多个 CUDA stream 并行计算
    constexpr int NUM_STREAMS = 8;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 从配置获取候选数量
    const int sym_candidates = config.symmetric_delta_candidates;
    const int asym_delta_candidates = config.asymmetric_delta_candidates;
    const int offset_candidates = config.offset_candidates;
    
    // 分配缓冲区给每个 stream
    std::vector<dev::vector<float>> noise_buffers(NUM_STREAMS);
    std::vector<dev::vector<int>> idx_buffers(NUM_STREAMS);
    std::vector<dev::vector<float>> val_buffers(NUM_STREAMS);
    
    const int max_candidates = is_symmetric ? sym_candidates 
                                            : asym_delta_candidates * offset_candidates;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        noise_buffers[i].resize(max_candidates);
        idx_buffers[i].resize(1);
        val_buffers[i].resize(1);
    }
    
    // 从 QuantBitWidth 获取量化范围（支持任意位宽）
    const int64_t quant_min = bw.qmin();
    const int64_t quant_max = bw.qmax();
    const int64_t num_steps = quant_max - quant_min;
    
    std::vector<std::vector<float>> h_offsets_all;
    std::vector<dev::vector<float>> offsets_dev_all;
    
    if (!is_symmetric) {
        h_offsets_all.resize(n);
        offsets_dev_all.resize(n);
        
        for (int c = 0; c < n; ++c) {
            float max_delta = (batch.maxs[c] - batch.mins[c]) / static_cast<float>(num_steps);
            if (max_delta < 1e-8f) max_delta = 1e-8f;
            
            int num_offsets_limit = static_cast<int>(num_steps + 2);
            const int num_offsets = (num_offsets_limit < offset_candidates) ? num_offsets_limit : offset_candidates;
            h_offsets_all[c].resize(num_offsets);
            float offset_step = static_cast<float>(num_steps) / (num_offsets - 2);
            for (int o = 0; o < num_offsets - 1; ++o) {
                h_offsets_all[c][o] = std::round(-static_cast<float>(num_steps) + o * offset_step);
            }
            h_offsets_all[c][num_offsets - 1] = std::round(batch.mins[c] / max_delta);
            
            offsets_dev_all[c].resize(num_offsets);
            cudaMemcpyAsync(offsets_dev_all[c].data(), h_offsets_all[c].data(), 
                           num_offsets * sizeof(float), cudaMemcpyHostToDevice, streams[c % NUM_STREAMS]);
        }
    }
    
    // 并行处理所有 channel
    std::vector<int> best_idx_host(n);
    
    for (int c = 0; c < n; ++c) {
        int stream_id = c % NUM_STREAMS;
        cudaStream_t s = streams[stream_id];
        
        // 使用原始直方图的 min/max 计算 bin_width（与 CPU 一致）
        float hist_min = batch.mins[c];
        float hist_max = batch.maxs[c];
        float bin_width = (hist_max - hist_min) / batch.num_bins;
        // 注意：不对 bin_width 做 clamp，保持与 CPU 一致（允许为 0）
        
        // 计算搜索用的 min_val/max_val（确保包含 0）
        float min_val = (hist_min < 0.0f) ? hist_min : 0.0f;
        float max_val = (hist_max > 0.0f) ? hist_max : 0.0f;
        float min_range = min_val + 1e-8f * static_cast<float>(num_steps);
        max_val = (max_val > min_range) ? max_val : min_range;
        
        const float* counts_ptr = batch.channel_counts(c);
        
        if (is_symmetric) {
            float abs_max_val = (max_val > -min_val) ? max_val : -min_val;
            float max_delta = 2.0f * abs_max_val / static_cast<float>(num_steps);
            float offset = -static_cast<float>((num_steps + 1) / 2);
            
            // 观察到的绝对值最大范围（用于 clamp 机制）
            // 使用原始直方图范围，不是扩展后的 min_val/max_val
            float abs_orig_min = std::abs(batch.original_mins[c]);
            float abs_orig_max = std::abs(batch.original_maxs[c]);
            float observed_max_abs = (abs_orig_min > abs_orig_max) ? abs_orig_min : abs_orig_max;
            
            const int threads = 256;
            size_t shared_mem = threads * sizeof(float);
            
            // 传递原始 hist_min 和 bin_width（与 CPU estimateNoise 一致）
            sqnr_noise_symmetric_kernel<<<sym_candidates, threads, shared_mem, s>>>(
                counts_ptr, hist_min, bin_width, batch.num_bins,
                max_delta, sym_candidates,
                num_steps, offset, observed_max_abs, config.gamma, config.p, noise_buffers[stream_id].data());
            
            find_min_noise_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int)), s>>>(
                noise_buffers[stream_id].data(), sym_candidates,
                idx_buffers[stream_id].data(), val_buffers[stream_id].data());
            
        } else {
            float max_delta = (max_val - min_val) / num_steps;
            max_delta = std::max(max_delta, 1e-8f);
            
            const int num_offsets = h_offsets_all[c].size();
            const int total_combos = asym_delta_candidates * num_offsets;
            
            const int threads = 256;
            size_t shared_mem = threads * sizeof(float);
            
            // 传递原始 hist_min 和 bin_width（与 CPU estimateNoise 一致）
            // search_min/search_max 用于 clamp 操作（已扩展确保包含 0）
            sqnr_noise_asymmetric_kernel<<<total_combos, threads, shared_mem, s>>>(
                counts_ptr, hist_min, min_val, max_val, bin_width, batch.num_bins,
                max_delta, asym_delta_candidates, num_offsets,
                num_steps, offsets_dev_all[c].data(), config.gamma, config.p,
                noise_buffers[stream_id].data());
            
            find_min_noise_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int)), s>>>(
                noise_buffers[stream_id].data(), total_combos,
                idx_buffers[stream_id].data(), val_buffers[stream_id].data());
        }
    }
    
    // 同步并收集结果
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // 拷贝最小索引到 host 并计算最终参数
    for (int c = 0; c < n; ++c) {
        int stream_id = c % NUM_STREAMS;
        int best_idx;
        cudaMemcpy(&best_idx, idx_buffers[stream_id].data(), sizeof(int), cudaMemcpyDeviceToHost);
        
        float min_val = (batch.mins[c] < 0.0f) ? batch.mins[c] : 0.0f;
        float max_val = (batch.maxs[c] > 0.0f) ? batch.maxs[c] : 0.0f;
        float min_range = min_val + 1e-8f * static_cast<float>(num_steps);
        max_val = (max_val > min_range) ? max_val : min_range;
        
        float optimal_scale;
        
        if (is_symmetric) {
            float abs_max_val = (max_val > -min_val) ? max_val : -min_val;
            float max_delta = 2.0f * abs_max_val / static_cast<float>(num_steps);
            const float num_pos_steps = static_cast<float>(num_steps / 2);
            
            optimal_scale = max_delta * (best_idx + 1) / (sym_candidates - 1);
            optimal_scale = (optimal_scale > 1e-8f) ? optimal_scale : 1e-8f;
            
            // clamp 机制：确保能覆盖观察范围（使用原始 min/max）
            float abs_orig_min = std::abs(batch.original_mins[c]);
            float abs_orig_max = std::abs(batch.original_maxs[c]);
            float observed_max_abs = (abs_orig_min > abs_orig_max) ? abs_orig_min : abs_orig_max;
            float quant_max_abs = optimal_scale * num_pos_steps;
            if (quant_max_abs < observed_max_abs) {
                optimal_scale = observed_max_abs / num_pos_steps;
            }
        } else {
            float max_delta = (max_val - min_val) / static_cast<float>(num_steps);
            max_delta = std::max(max_delta, 1e-8f);
            
            const int num_offsets = h_offsets_all[c].size();
            int best_delta_idx = best_idx / num_offsets;
            int best_offset_idx = best_idx % num_offsets;
            
            float delta = max_delta * (best_delta_idx + 1) / (asym_delta_candidates - 1);
            delta = std::max(delta, 1e-8f);
            float offset = h_offsets_all[c][best_offset_idx];
            
            float test_min_calc = delta * offset;
            float test_min = (min_val > test_min_calc) ? min_val : test_min_calc;
            float test_max_calc = test_min + delta * static_cast<float>(num_steps);
            float test_max = (max_val < test_max_calc) ? max_val : test_max_calc;
            float scale_calc = (test_max - test_min) / static_cast<float>(num_steps);
            optimal_scale = (scale_calc > 1e-8f) ? scale_calc : 1e-8f;
        }
        
        // 转换到 POT
        float n_val = -std::log2(optimal_scale);
        int8_t n_rounded = static_cast<int8_t>(std::round(n_val));
        
        // 位宽约束：使用原始 min/max（未被范围扩展修改的值）
        // max(|val|) / scale <= qmax => exp2_inv <= log2(qmax / max(|val|))
        float orig_min = batch.original_mins[c];
        float orig_max = batch.original_maxs[c];
        float max_abs = std::max(std::abs(orig_min), std::abs(orig_max));
        if (max_abs > 1e-10f) {
            int8_t max_exp2_inv = static_cast<int8_t>(std::floor(
                std::log2(static_cast<float>(quant_max) / max_abs)));
            if (max_exp2_inv < n_rounded) {
                n_rounded = max_exp2_inv;
            }
        }
        out_exp2_inv[c] = n_rounded;
    }
    
    // 清理 streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

// ============================================================================
// GPU MINMAX 量化范围计算实现
// ============================================================================

void compute_minmax_dev(const float* data_dev, size_t size, float& min_out, float& max_out,
                        cudaStream_t stream) {
    if (size == 0) {
        min_out = 0.0f;
        max_out = 0.0f;
        return;
    }
    
    // 直接使用已有的 compute_minmax 函数
    compute_minmax(data_dev, size, min_out, max_out);
}

void compute_minmax_per_step_ema_gpu(const float* data_dev, int steps, int step_size,
                                      float& min_out, float& max_out, float decay,
                                      cudaStream_t stream) {
    if (steps <= 0 || step_size <= 0) return;
    
    // 分配 GPU 端 min/max 数组
    dev::vector<float> all_mins(steps);
    dev::vector<float> all_maxs(steps);
    
    // 一次 kernel 调用计算所有时间步的 min/max
    {
        const int threads = 256;
        const int blocks = steps;
        const size_t shared_mem = 2 * threads * sizeof(float);
        compute_contiguous_batch_minmax_kernel<<<blocks, threads, shared_mem, stream>>>(
            data_dev, all_mins.data(), all_maxs.data(), step_size, steps);
    }
    cudaStreamSynchronize(stream);
    
    // 拷贝到 CPU
    std::vector<float> h_mins(steps), h_maxs(steps);
    cudaMemcpy(h_mins.data(), all_mins.data(), steps * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxs.data(), all_maxs.data(), steps * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU 端做 EMA 融合（计算量很小）
    bool initialized = !(min_out == std::numeric_limits<float>::max() && 
                         max_out == std::numeric_limits<float>::lowest());
    
    for (int t = 0; t < steps; ++t) {
        if (!initialized) {
            min_out = h_mins[t];
            max_out = h_maxs[t];
            initialized = true;
        } else {
            min_out = decay * min_out + (1.0f - decay) * h_mins[t];
            max_out = decay * max_out + (1.0f - decay) * h_maxs[t];
        }
    }
}

void compute_minmax_per_channel_gpu(const float* data_dev, size_t input_size, size_t channel_size,
                                     std::vector<float>& min_out, std::vector<float>& max_out,
                                     cudaStream_t stream) {
    if (input_size == 0 || channel_size == 0) return;
    
    // 分配 GPU 端缓冲区
    dev::vector<float> d_mins(channel_size);
    dev::vector<float> d_maxs(channel_size);
    
    // 批量计算所有 channel 的 min/max
    {
        const int threads = 256;
        const int blocks = static_cast<int>(channel_size);
        const size_t shared_mem = 2 * threads * sizeof(float);
        compute_per_channel_minmax_kernel<<<blocks, threads, shared_mem, stream>>>(
            data_dev, d_mins.data(), d_maxs.data(), static_cast<int>(input_size),
            static_cast<int>(channel_size));
    }
    cudaStreamSynchronize(stream);
    
    // 拷贝到 CPU
    std::vector<float> h_mins(channel_size), h_maxs(channel_size);
    cudaMemcpy(h_mins.data(), d_mins.data(), channel_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxs.data(), d_maxs.data(), channel_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 更新输出（取并集）
#pragma omp parallel for
    for (size_t c = 0; c < channel_size; ++c) {
        min_out[c] = std::min(min_out[c], h_mins[c]);
        max_out[c] = std::max(max_out[c], h_maxs[c]);
    }
}

void update_ranges_from_v_gpu(const float* h_dev, const float* v_dev, size_t steps,
                               size_t hidden_size, size_t batch_size,
                               float& min_z_out, float& max_z_out,
                               float& min_r_out, float& max_r_out,
                               float& min_g_out, float& max_g_out,
                               float& min_Rh_add_br, float& max_Rh_add_br,
                               float& min_rRh, float& max_rRh,
                               float& min_new_contrib, float& max_new_contrib,
                               float& min_old_contrib, float& max_old_contrib,
                               cudaStream_t stream) {
    const size_t total_size = steps * batch_size * hidden_size;
    if (total_size == 0) return;
    
    // 分配临时 GPU 缓冲区
    dev::vector<float> z_out_dev(total_size);
    dev::vector<float> r_out_dev(total_size);
    dev::vector<float> g_out_dev(total_size);
    dev::vector<float> Rh_add_br_dev(total_size);
    dev::vector<float> rRh_dev(total_size);
    dev::vector<float> new_contrib_dev(total_size);
    dev::vector<float> old_contrib_dev(total_size);
    
    // 在 GPU 上提取并计算所有门值
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    
    extract_gate_values_kernel<<<blocks, threads, 0, stream>>>(
        v_dev, h_dev, z_out_dev.data(), r_out_dev.data(), g_out_dev.data(), 
        Rh_add_br_dev.data(), rRh_dev.data(), new_contrib_dev.data(), 
        old_contrib_dev.data(), 
        static_cast<int>(steps), static_cast<int>(batch_size), 
        static_cast<int>(hidden_size));
    
    // 等待 extract kernel 完成
    cudaStreamSynchronize(stream);
    
    // 计算 minmax 并更新范围
    float new_min, new_max;
    
    compute_minmax(z_out_dev.data(), total_size, new_min, new_max);
    min_z_out = std::min(min_z_out, new_min);
    max_z_out = std::max(max_z_out, new_max);
    
    compute_minmax(r_out_dev.data(), total_size, new_min, new_max);
    min_r_out = std::min(min_r_out, new_min);
    max_r_out = std::max(max_r_out, new_max);
    
    compute_minmax(g_out_dev.data(), total_size, new_min, new_max);
    min_g_out = std::min(min_g_out, new_min);
    max_g_out = std::max(max_g_out, new_max);
    
    compute_minmax(Rh_add_br_dev.data(), total_size, new_min, new_max);
    min_Rh_add_br = std::min(min_Rh_add_br, new_min);
    max_Rh_add_br = std::max(max_Rh_add_br, new_max);
    
    compute_minmax(rRh_dev.data(), total_size, new_min, new_max);
    min_rRh = std::min(min_rRh, new_min);
    max_rRh = std::max(max_rRh, new_max);
    
    compute_minmax(new_contrib_dev.data(), total_size, new_min, new_max);
    min_new_contrib = std::min(min_new_contrib, new_min);
    max_new_contrib = std::max(max_new_contrib, new_max);
    
    compute_minmax(old_contrib_dev.data(), total_size, new_min, new_max);
    min_old_contrib = std::min(min_old_contrib, new_min);
    max_old_contrib = std::max(max_old_contrib, new_max);
}

}  // namespace gpu_hist

// ============================================================================
// GPU MINMAX 量化范围更新实现
// ============================================================================

void updateGRUQuantizationRangesGPU(
    int time_steps, int batch_size, int input_size, int hidden_size,
    const float* W, const float* R, const float* bx, const float* br,
    const float* x, const float* h, const float* v,
    const float* tmp_Wx, const float* tmp_Rh,
    const float* z_pres, const float* r_pres, const float* g_pres,
    size_t pres_size,
    GRUQuantizationRanges& quant_ranges,
    cudaStream_t stream) {
    
    const int NH = batch_size * hidden_size;
    const int NI = batch_size * input_size;
    const int channel_size = hidden_size * 3;
    
    // 设置 hidden_ 维度（确保 per-channel 向量已分配）
    if (quant_ranges.hidden_ != hidden_size) {
        quant_ranges.reset(hidden_size);
    }
    
    // =====================================================================
    // 1. 标量范围：使用 GPU minmax + EMA
    // =====================================================================
    
    // 输入 x 的范围（分时间步 EMA）
    gpu_hist::compute_minmax_per_step_ema_gpu(
        x, time_steps, NI, quant_ranges.min_x_, quant_ranges.max_x_, 0.9f, stream);
    
    // 隐藏状态 h 的范围（跳过 h0）
    gpu_hist::compute_minmax_per_step_ema_gpu(
        h + NH, time_steps, NH, quant_ranges.min_h_, quant_ranges.max_h_, 0.9f, stream);
    
    // Wx 结果的范围
    gpu_hist::compute_minmax_per_step_ema_gpu(
        tmp_Wx, time_steps, NH * 3, quant_ranges.min_Wx_, quant_ranges.max_Wx_, 0.9f, stream);
    
    // Rh 结果的范围
    gpu_hist::compute_minmax_per_step_ema_gpu(
        tmp_Rh, time_steps, NH * 3, quant_ranges.min_Rh_, quant_ranges.max_Rh_, 0.9f, stream);
    
    // z 门预激活值
    gpu_hist::compute_minmax_per_step_ema_gpu(
        z_pres, time_steps, NH, quant_ranges.min_z_pre_, quant_ranges.max_z_pre_, 0.9f, stream);
    
    // r 门预激活值
    gpu_hist::compute_minmax_per_step_ema_gpu(
        r_pres, time_steps, NH, quant_ranges.min_r_pre_, quant_ranges.max_r_pre_, 0.9f, stream);
    
    // g 门预激活值
    gpu_hist::compute_minmax_per_step_ema_gpu(
        g_pres, time_steps, NH, quant_ranges.min_g_pre_, quant_ranges.max_g_pre_, 0.9f, stream);
    
    // =====================================================================
    // 2. Per-channel 范围：使用 GPU 批量 kernel
    // =====================================================================
    
    // 权重 W [I, H*3]
    gpu_hist::compute_minmax_per_channel_gpu(
        W, input_size, channel_size, quant_ranges.min_W_, quant_ranges.max_W_, stream);
    
    // 权重 R [H, H*3]
    gpu_hist::compute_minmax_per_channel_gpu(
        R, hidden_size, channel_size, quant_ranges.min_R_, quant_ranges.max_R_, stream);
    
    // 偏置 bx [1, H*3]
    gpu_hist::compute_minmax_per_channel_gpu(
        bx, 1, channel_size, quant_ranges.min_bx_, quant_ranges.max_bx_, stream);
    
    // 偏置 br [1, H*3]
    gpu_hist::compute_minmax_per_channel_gpu(
        br, 1, channel_size, quant_ranges.min_br_, quant_ranges.max_br_, stream);
    
    // =====================================================================
    // 3. 从 v 提取中间值范围：使用 GPU kernel
    // =====================================================================
    
    gpu_hist::update_ranges_from_v_gpu(
        h, v, time_steps, hidden_size, batch_size,
        quant_ranges.min_z_out_, quant_ranges.max_z_out_,
        quant_ranges.min_r_out_, quant_ranges.max_r_out_,
        quant_ranges.min_g_out_, quant_ranges.max_g_out_,
        quant_ranges.min_Rh_add_br_g_, quant_ranges.max_Rh_add_br_g_,
        quant_ranges.min_rRh_, quant_ranges.max_rRh_,
        quant_ranges.min_new_contrib_, quant_ranges.max_new_contrib_,
        quant_ranges.min_old_contrib_, quant_ranges.max_old_contrib_,
        stream);
}


