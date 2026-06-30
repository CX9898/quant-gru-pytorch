#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <cstdio>

#include "dev_vector.h"
#include "quantize_ops_helper.h"

// LUT 生成函数声明已在 quantize_lut_types.h 中（通过 quantize_ops_helper.h 包含）

// 生成分段线性量化表并存储到参数中
// 在 finalize_calibration 时调用一次，然后在每次 forward 时从参数复制到 QuantGRUReScale
void generate_piecewise_linear_lut_to_params(GRUQuantParams &params) {
    const auto &config = params.bitwidth_config_;
    // 直接使用单一权威 scale（仿射=连续校准值，POT2=2^-shift），无需 fixed/shift 往返
    const float update_in_scale = params.update_gate_input_.scale;
    const float update_out_scale = params.update_gate_output_.scale;
    const float reset_in_scale = params.reset_gate_input_.scale;
    const float reset_out_scale = params.reset_gate_output_.scale;
    const float new_in_scale = params.new_gate_input_.scale;
    const float new_out_scale = params.new_gate_output_.scale;

    // update gate Sigmoid
    params.sigmoid_update_gate_lut_ = generate_sigmoid_lut(
        update_in_scale, params.update_gate_input_.zero_point,
        update_out_scale, params.update_gate_output_.zero_point,
        config.update_gate_input_, config.update_gate_output_);

    // reset gate Sigmoid
    params.sigmoid_reset_gate_lut_ = generate_sigmoid_lut(
        reset_in_scale, params.reset_gate_input_.zero_point,
        reset_out_scale, params.reset_gate_output_.zero_point,
        config.reset_gate_input_, config.reset_gate_output_);

    // new gate Tanh
    params.tanh_new_gate_lut_ = generate_tanh_lut(
        new_in_scale, params.new_gate_input_.zero_point,
        new_out_scale, params.new_gate_output_.zero_point,
        config.new_gate_input_, config.new_gate_output_);

#ifdef DEBUG
    printf("[DEBUG] generate_piecewise_linear_lut_to_params: LUTs stored in params\n");
#endif
}

namespace kernel {

// ★★★ 修复：使用 int64_t 存储以避免 16 位量化时的溢出 ★★★
template <typename T>
__global__ void computeWeightSumMulZP(
    const T *__restrict__ W_q,         // [out_dim, in_dim] 权重量化矩阵, 列主序储存
    int64_t *__restrict__ weight_sum,  // [out_dim] 输出数组（改为 int64_t）
    int x_zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,  // 输出通道数 (M)
    int in_dim    // 输入通道数 (K)
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) {
        return;
    }

    // 使用 int64_t 进行整个计算，避免溢出
    int64_t sum_i64 = 0;
#pragma unroll
    for (int j = 0; j < in_dim; ++j) {
        sum_i64 += static_cast<int64_t>(W_q[row + j * out_dim]);
    }

    // 乘以 x_zp（使用 int64_t 避免溢出）
    sum_i64 *= static_cast<int64_t>(x_zp);

#ifdef DEBUG
    // 调试输出
    if (row == 0) {
        printf("[DEBUG] computeWeightSumMulZP: row=0, in_dim=%d, x_zp=%d, result=%lld\n", in_dim,
               x_zp, (long long)sum_i64);
    }
#endif

    // 使用 int64_t 存储完整结果
    weight_sum[row] = sum_i64;
}

// int32_t 版本：适用于 8 位量化，累加结果不会溢出 int32
template <typename T>
__global__ void computeWeightSumMulZP_i32(
    const T *__restrict__ W_q,         // [out_dim, in_dim] 权重量化矩阵, 列主序储存
    int32_t *__restrict__ weight_sum,  // [out_dim] 输出数组
    int x_zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,  // 输出通道数 (M)
    int in_dim    // 输入通道数 (K)
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) {
        return;
    }

    int32_t sum = 0;
#pragma unroll
    for (int j = 0; j < in_dim; ++j) {
        sum += static_cast<int32_t>(W_q[row + j * out_dim]);
    }
    sum *= x_zp;
    weight_sum[row] = sum;
}


template <typename T, typename QuantT>
__global__ void dequantification(const QuantT *quant_data, T *data, size_t size, FixedPointScale exp2_inv,
                                 int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    data[idx] = dequantize(static_cast<int32_t>(quant_data[idx]), exp2_inv, zp);
}

// ============================================================================
// 浮点存储版量化 kernel（用于 GPU-FP 实现）
// ============================================================================

// ============================================================================
// 量化 kernel（统一模板版本，支持可选 mask 输出）
// ============================================================================

// 量化到 float 存储（值仍是定点整数）
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training = false>
__global__ void quantificationFP(const float *data, float *quant_data, uint8_t *mask,
                                  size_t size, FixedPointScale exp2_inv, int32_t zp, QuantBitWidth bw) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    uint8_t keep_gradient;
    quant_data[idx] = quantize_f<Training>(data[idx], exp2_inv, zp, bw, Training ? &keep_gradient : nullptr);
    if constexpr (Training) {
        mask[idx] = keep_gradient;
    }
}

// Per-channel 量化到 float 存储
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training = false>
__global__ void quantificationPerChannelFP(const float *src, float *quant_data, uint8_t *mask,
                                            size_t input_size, size_t channel_size,
                                            const FixedPointScale *__restrict__ exp2_invs,
                                            QuantBitWidth bw) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (channel_idx >= channel_size || input_idx >= input_size) return;
    
    const size_t idx = input_idx * channel_size + channel_idx;
    uint8_t keep_gradient;
    quant_data[idx] = quantize_f<Training>(src[idx], exp2_invs[channel_idx], 0, bw, Training ? &keep_gradient : nullptr);
    if constexpr (Training) {
        mask[idx] = keep_gradient;
    }
}

// Per-gate 量化到 float 存储（用于 GRU 权重）
// 数据布局: [input_size, hidden_size * 3]，每个 gate 有 hidden_size 个通道
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training = false>
__global__ void quantificationPerGateFP(const float *src, float *quant_data, uint8_t *mask,
                                         size_t input_size, size_t hidden_size,
                                         FixedPointScale exp2_inv_z, FixedPointScale exp2_inv_r, FixedPointScale exp2_inv_g,
                                         QuantBitWidth bw) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    const size_t channel_size = hidden_size * 3;
    if (channel_idx >= channel_size || input_idx >= input_size) return;
    
    // 计算 gate 索引：0=z, 1=r, 2=g
    const size_t gate_idx = channel_idx / hidden_size;
    FixedPointScale exp2_inv;
    if (gate_idx == 0) {
        exp2_inv = exp2_inv_z;
    } else if (gate_idx == 1) {
        exp2_inv = exp2_inv_r;
    } else {
        exp2_inv = exp2_inv_g;
    }
    
    const size_t idx = input_idx * channel_size + channel_idx;
    uint8_t keep_gradient;
    quant_data[idx] = quantize_f<Training>(src[idx], exp2_inv, 0, bw, Training ? &keep_gradient : nullptr);
    if constexpr (Training) {
        mask[idx] = keep_gradient;
    }
}

// 从 float 存储的量化值反量化
__global__ void dequantificationFP(const float *quant_data, float *data, size_t size,
                                    FixedPointScale exp2_inv, int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    data[idx] = dequantize_f(quant_data[idx], exp2_inv, zp);
}

// 从 float 存储的量化值原地反量化（in-place）
__global__ void dequantificationFPInplace(float *data, size_t size,
                                          FixedPointScale exp2_inv, int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    data[idx] = dequantize_f(data[idx], exp2_inv, zp);
}

// 从 float 存储的量化值进行 per-channel 反量化
__global__ void dequantificationPerChannelFP(const float *quant_data, float *data,
                                             size_t input_size, size_t channel_size,
                                             const FixedPointScale *__restrict__ exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const FixedPointScale exp2_inv = exp2_invs[channel_idx];
    
    // 内存布局：idx = input_idx * channel_size + channel_idx
    // 与 quantificationPerChannelFP 保持一致
    const size_t idx = input_idx * channel_size + channel_idx;
    
    // 使用 dequantize_f 进行反量化（无零点，因为权重通常是对称量化）
    data[idx] = dequantize_f(quant_data[idx], exp2_inv, 0);
}

// 从 float 存储的量化值进行 per-channel 原地反量化（in-place）
__global__ void dequantificationPerChannelFPInplace(float *data,
                                                    size_t input_size, size_t channel_size,
                                                    const FixedPointScale *__restrict__ exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const FixedPointScale exp2_inv = exp2_invs[channel_idx];
    
    // 内存布局：idx = input_idx * channel_size + channel_idx
    // 与 quantificationPerChannelFP 保持一致
    const size_t idx = input_idx * channel_size + channel_idx;
    
    // 原地反量化：直接修改 data[idx]
    data[idx] = dequantize_f(data[idx], exp2_inv, 0);
}

// Per-gate 原地反量化（用于 GRU 权重）
// 数据布局: [input_size, hidden_size * 3]，每个 gate 有 hidden_size 个通道
__global__ void dequantificationPerGateFPInplace(float *data,
                                                 size_t input_size, size_t hidden_size,
                                                 FixedPointScale exp2_inv_z, FixedPointScale exp2_inv_r, FixedPointScale exp2_inv_g) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    const size_t channel_size = hidden_size * 3;
    if (channel_idx >= channel_size || input_idx >= input_size) return;
    
    // 计算 gate 索引：0=z, 1=r, 2=g
    const size_t gate_idx = channel_idx / hidden_size;
    FixedPointScale exp2_inv;
    if (gate_idx == 0) {
        exp2_inv = exp2_inv_z;
    } else if (gate_idx == 1) {
        exp2_inv = exp2_inv_r;
    } else {
        exp2_inv = exp2_inv_g;
    }
    
    const size_t idx = input_idx * channel_size + channel_idx;
    
    // 原地反量化：直接修改 data[idx]（无零点，因为权重通常是对称量化）
    data[idx] = dequantize_f(data[idx], exp2_inv, 0);
}

// 量化到 int32 存储
// @tparam Training 是否训练模式（决定是否使用 mask）
template <typename T, bool Training = false>
__global__ void quantificationBitwidth(const T *data, int32_t *quant_data, uint8_t *mask,
                                        size_t size, FixedPointScale exp2_inv, int32_t zp, QuantBitWidth bw) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    uint8_t keep_gradient;
    quant_data[idx] = ::quantize<Training>(data[idx], exp2_inv, zp, bw, Training ? &keep_gradient : nullptr);
    if constexpr (Training) {
        mask[idx] = keep_gradient;
    }
}

// Per-channel 量化到 int32 存储
// @tparam Training 是否训练模式（决定是否使用 mask）
template <typename T, bool Training = false>
__global__ void quantificationPerChannelBitwidth(const T *src, int32_t *quant_data, uint8_t *mask,
                                                  size_t input_size, size_t channel_size, 
                                                  const FixedPointScale *exp2_invs, QuantBitWidth bw) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const FixedPointScale exp2_inv = exp2_invs[channel_idx];
    const size_t idx = input_idx * channel_size + channel_idx;
    uint8_t keep_gradient;
    quant_data[idx] = ::quantize<Training>(src[idx], exp2_inv, 0, bw, Training ? &keep_gradient : nullptr);
    if constexpr (Training) {
        mask[idx] = keep_gradient;
    }
}

// 从 float 存储的量化值反量化 V 向量
// V 布局: [time_steps, batch_size, hidden_size * 4]
// 4个部分: [z_out, r_out, g_out, weight_hh_linear_g]
__global__ void dequantificationVFP(const float *quant_data, float *data, int time_steps,
                                     int batch_size, int hidden_size,
                                     FixedPointScale shift_z, int32_t zp_z,
                                     FixedPointScale shift_r, int32_t zp_r,
                                     FixedPointScale shift_g, int32_t zp_g,
                                     FixedPointScale shift_hh, int32_t zp_hh) {
    const int t = blockIdx.x;
    const int b = blockIdx.y;
    const int h = threadIdx.x;

    if (t >= time_steps || b >= batch_size || h >= hidden_size) return;

    const int base_idx = t * (batch_size * hidden_size * 4) + b * (hidden_size * 4);

    // 反量化 z_out (第0部分)
    const int z_idx = base_idx + 0 * hidden_size + h;
    data[z_idx] = dequantize_f(quant_data[z_idx], shift_z, zp_z);

    // 反量化 r_out (第1部分)
    const int r_idx = base_idx + 1 * hidden_size + h;
    data[r_idx] = dequantize_f(quant_data[r_idx], shift_r, zp_r);

    // 反量化 g_out (第2部分)
    const int g_idx = base_idx + 2 * hidden_size + h;
    data[g_idx] = dequantize_f(quant_data[g_idx], shift_g, zp_g);

    // 反量化 weight_hh_linear_g (第3部分)
    const int hh_idx = base_idx + 3 * hidden_size + h;
    data[hh_idx] = dequantize_f(quant_data[hh_idx], shift_hh, zp_hh);
}

// 从 float 存储的量化值原地反量化 V 向量（in-place）
// V 布局: [time_steps, batch_size, hidden_size * 4]
// 4个部分: [z_out, r_out, g_out, weight_hh_linear_g]
__global__ void dequantificationVFPInplace(float *data, int time_steps,
                                            int batch_size, int hidden_size,
                                            FixedPointScale shift_z, int32_t zp_z,
                                            FixedPointScale shift_r, int32_t zp_r,
                                            FixedPointScale shift_g, int32_t zp_g,
                                            FixedPointScale shift_hh, int32_t zp_hh) {
    const int t = blockIdx.x;
    const int b = blockIdx.y;
    const int h = threadIdx.x;

    if (t >= time_steps || b >= batch_size || h >= hidden_size) return;

    const int base_idx = t * (batch_size * hidden_size * 4) + b * (hidden_size * 4);

    // 反量化 z_out (第0部分)
    const int z_idx = base_idx + 0 * hidden_size + h;
    data[z_idx] = dequantize_f(data[z_idx], shift_z, zp_z);

    // 反量化 r_out (第1部分)
    const int r_idx = base_idx + 1 * hidden_size + h;
    data[r_idx] = dequantize_f(data[r_idx], shift_r, zp_r);

    // 反量化 g_out (第2部分)
    const int g_idx = base_idx + 2 * hidden_size + h;
    data[g_idx] = dequantize_f(data[g_idx], shift_g, zp_g);

    // 反量化 weight_hh_linear_g (第3部分)
    const int hh_idx = base_idx + 3 * hidden_size + h;
    data[hh_idx] = dequantize_f(data[hh_idx], shift_hh, zp_hh);
}

// ============================================================================
// Bias 特殊量化 kernel（使用 round(bias / scale / 128) * 128）
// ============================================================================

/**
 * @brief Bias 特殊量化到 float 存储
 * 
 * 量化公式: q = clamp(round((bias / scale) / 128) * 128, qmin, qmax)
 * 这是为了与 PyTorch 的量化行为保持一致
 */
/**
 * @brief Bias 特殊量化到 float 存储
 * @tparam Training 是否训练模式（决定是否使用 mask）
 */
template <bool Training = false>
__global__ void quantificationBiasFP(const float *src, float *quant_data, uint8_t *mask,
                                      size_t channel_size,
                                      const FixedPointScale *__restrict__ exp2_invs,
                                      QuantBitWidth bw) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (channel_idx >= channel_size) return;
    
    float scale = decode_scale(exp2_invs[channel_idx]);
    // 特殊量化: round((bias / scale) / 128) * 128
    float normalized = src[channel_idx] / scale;
    float q = round_f(normalized / 128.0f) * 128.0f;
    uint8_t keep_gradient;
    quant_data[channel_idx] = clamp_f<Training>(q, bw, Training ? &keep_gradient : nullptr);
    if constexpr (Training) {
        mask[channel_idx] = keep_gradient;
    }
}

// v 使用 int32_t 存储，但内部各部分使用不同的量化参数:
// - z: 使用 shift_z, zp_z (update_gate_output)
// - r: 使用 shift_r, zp_r (reset_gate_output)
// - g: 使用 shift_g, zp_g (new_gate_output)
// - weight_hh_linear_g: 使用 shift_hh, zp_hh (weight_hh_linear)
template <typename T>
__global__ void dequantificationV(const int32_t *quant_data, T *data, int time_steps,
                                  int batch_size, int hidden_size, FixedPointScale shift_z, int32_t zp_z,
                                  FixedPointScale shift_r, int32_t zp_r, FixedPointScale shift_g, int32_t zp_g,
                                  FixedPointScale shift_hh, int32_t zp_hh) {
    // 计算当前线程处理的索引
    // blockIdx.x: time_step
    // blockIdx.y: batch
    // threadIdx.x: hidden_unit
    const int t = blockIdx.x;
    const int b = blockIdx.y;
    const int h = threadIdx.x;

    if (t >= time_steps || b >= batch_size || h >= hidden_size) {
        return;
    }

    // v的布局: [time_steps, batch_size, hidden_size * 4]
    // 每个时间步内: [batch_size, hidden_size * 4]
    // 每个batch内: [hidden_size * 4]
    // 4个部分: [z_out, r_out, g_out, weight_hh_linear_g]，每个部分大小为 hidden_size

    const int base_idx = t * (batch_size * hidden_size * 4) + b * (hidden_size * 4);

    // 反量化 z_out (第0部分) - 从 int32_t 反量化
    const int z_idx = base_idx + 0 * hidden_size + h;
    data[z_idx] = dequantize<int32_t>(quant_data[z_idx], shift_z, zp_z);

    // 反量化 r_out (第1部分) - 从 int32_t 反量化
    const int r_idx = base_idx + 1 * hidden_size + h;
    data[r_idx] = dequantize<int32_t>(quant_data[r_idx], shift_r, zp_r);

    // 反量化 g_out (第2部分) - 从 int32_t 反量化
    const int g_idx = base_idx + 2 * hidden_size + h;
    data[g_idx] = dequantize<int32_t>(quant_data[g_idx], shift_g, zp_g);

    // 反量化 weight_hh_linear_g (第3部分) - 从 int32_t 反量化
    const int hh_idx = base_idx + 3 * hidden_size + h;
    data[hh_idx] = dequantize<int32_t>(quant_data[hh_idx], shift_hh, zp_hh);
}


template <typename T, typename QuantT>
__global__ void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                           size_t channel_size, const FixedPointScale *exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const FixedPointScale exp2_inv = exp2_invs[channel_idx];

    const size_t idx = input_idx * channel_size + channel_idx;
    data[idx] = dequantize<QuantT>(quant_data[idx], exp2_inv, 0);
}

}  // namespace kernel

// int64_t 版本：用于 16 位量化，避免溢出
template <typename T>
void computeWeightSumMulzp(
    const T *W_q,         // [out_dim, in_dim] 权重量化矩阵
    int64_t *weight_sum,  // [out_dim] 输出数组（int64_t）
    int x_zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,  // 输出通道数 (M)
    int in_dim,   // 输入通道数 (K)
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;
    kernel::computeWeightSumMulZP<<<blocks, threads, 0, stream>>>(W_q, weight_sum, x_zp, n, out_dim,
                                                                  in_dim);
}

// int32_t 版本：用于 8 位量化，不会溢出
template <typename T>
void computeWeightSumMulzp(
    const T *W_q,         // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,  // [out_dim] 输出数组（int32_t）
    int x_zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,  // 输出通道数 (M)
    int in_dim,   // 输入通道数 (K)
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;
    kernel::computeWeightSumMulZP_i32<<<blocks, threads, 0, stream>>>(W_q, weight_sum, x_zp, n,
                                                                      out_dim, in_dim);
}

// int64_t 版本显式实例化
template void computeWeightSumMulzp<int8_t>(const int8_t *W_q, int64_t *weight_sum, int x_zp,
                                            const int8_t *__restrict__ n, int out_dim, int in_dim,
                                            cudaStream_t stream);

template void computeWeightSumMulzp<int16_t>(const int16_t *W_q, int64_t *weight_sum, int x_zp,
                                             const int8_t *__restrict__ n, int out_dim, int in_dim,
                                             cudaStream_t stream);

// int32_t 版本显式实例化
template void computeWeightSumMulzp<int8_t>(const int8_t *W_q, int32_t *weight_sum, int x_zp,
                                            const int8_t *__restrict__ n, int out_dim, int in_dim,
                                            cudaStream_t stream);

template void computeWeightSumMulzp<int16_t>(const int16_t *W_q, int32_t *weight_sum, int x_zp,
                                             const int8_t *__restrict__ n, int out_dim, int in_dim,
                                             cudaStream_t stream);

// int32_t 权重版本显式实例化（统一 int32 存储方案）
template void computeWeightSumMulzp<int32_t>(const int32_t *W_q, int64_t *weight_sum, int x_zp,
                                             const int8_t *__restrict__ n, int out_dim, int in_dim,
                                             cudaStream_t stream);

template void computeWeightSumMulzp<int32_t>(const int32_t *W_q, int32_t *weight_sum, int x_zp,
                                             const int8_t *__restrict__ n, int out_dim, int in_dim,
                                             cudaStream_t stream);

namespace dev {

// ============================================================================
// 量化函数（统一接口，支持可选 mask 输出）
// ============================================================================

// ============================================================================
// 量化函数（统一接口，支持可选 mask 输出）
// ============================================================================

// ============================================================================
// 量化函数（模板版本，使用模板参数控制 mask 行为）
// ============================================================================

// 统一 int32_t 输出，使用位宽配置进行 clamp
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training>
void quantificationBitwidth(const float *data, int32_t *quant_data, uint8_t *mask,
                             size_t size, FixedPointScale exp2_inv, int32_t zp, QuantBitWidth bw) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::quantificationBitwidth<float, Training><<<grid, block>>>(data, quant_data, mask, size, exp2_inv, zp, bw);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// 浮点存储版量化函数（用于 GPU-FP 实现）
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training>
void quantificationFP(const float *data, float *quant_data, uint8_t *mask,
                      size_t size, FixedPointScale exp2_inv, int32_t zp, QuantBitWidth bw) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::quantificationFP<Training><<<grid, block>>>(data, quant_data, mask, size, exp2_inv, zp, bw);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("quantificationFP kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training>
void quantificationPerChannelFP(const float *src, float *quant_data, uint8_t *mask,
                                size_t input_size, size_t channel_size, 
                                const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);
    kernel::quantificationPerChannelFP<Training><<<gridDim, blockDim>>>(src, quant_data, mask, input_size,
                                                                         channel_size, exp2_invs.data(), bw);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("quantificationPerChannelFP kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// 统一 int32_t 输出，使用位宽配置进行 clamp
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training>
void quantificationPerChannelBitwidth(const float *src, int32_t *quant_data, uint8_t *mask,
                                       size_t input_size, size_t channel_size, 
                                       const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);
    kernel::quantificationPerChannelBitwidth<float, Training><<<gridDim, blockDim>>>(src, quant_data, mask,
                                                                                      input_size, channel_size,
                                                                                      exp2_invs.data(), bw);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("quantificationPerChannelBitwidth kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// 显式实例化模板函数
template void quantificationBitwidth<false>(const float *data, int32_t *quant_data, uint8_t *mask,
                                            size_t size, FixedPointScale exp2_inv, int32_t zp, QuantBitWidth bw);
template void quantificationBitwidth<true>(const float *data, int32_t *quant_data, uint8_t *mask,
                                           size_t size, FixedPointScale exp2_inv, int32_t zp, QuantBitWidth bw);

template void quantificationFP<false>(const float *data, float *quant_data, uint8_t *mask,
                                     size_t size, FixedPointScale exp2_inv, int32_t zp, QuantBitWidth bw);
template void quantificationFP<true>(const float *data, float *quant_data, uint8_t *mask,
                                    size_t size, FixedPointScale exp2_inv, int32_t zp, QuantBitWidth bw);

template void quantificationPerChannelFP<false>(const float *src, float *quant_data, uint8_t *mask,
                                                size_t input_size, size_t channel_size,
                                                const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw);
template void quantificationPerChannelFP<true>(const float *src, float *quant_data, uint8_t *mask,
                                               size_t input_size, size_t channel_size,
                                               const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw);

// Per-gate 量化 wrapper（用于 GRU 权重）
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training>
void quantificationPerGateFP(const float *src, float *quant_data, uint8_t *mask,
                             size_t input_size, size_t hidden_size,
                             FixedPointScale exp2_inv_z, FixedPointScale exp2_inv_r, FixedPointScale exp2_inv_g,
                             QuantBitWidth bw) {
    const size_t channel_size = hidden_size * 3;
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);
    kernel::quantificationPerGateFP<Training><<<gridDim, blockDim>>>(src, quant_data, mask, input_size,
                                                                     hidden_size, exp2_inv_z, exp2_inv_r, exp2_inv_g, bw);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("quantificationPerGateFP kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template void quantificationPerGateFP<false>(const float *src, float *quant_data, uint8_t *mask,
                                            size_t input_size, size_t hidden_size,
                                            FixedPointScale exp2_inv_z, FixedPointScale exp2_inv_r, FixedPointScale exp2_inv_g,
                                            QuantBitWidth bw);
template void quantificationPerGateFP<true>(const float *src, float *quant_data, uint8_t *mask,
                                            size_t input_size, size_t hidden_size,
                                            FixedPointScale exp2_inv_z, FixedPointScale exp2_inv_r, FixedPointScale exp2_inv_g,
                                            QuantBitWidth bw);

// 通用权重量化函数（根据 granularity 自动选择）
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training>
void quantificationWeightFP(const float *src, float *quant_data, uint8_t *mask,
                            size_t input_size, size_t hidden_size,
                            OperatorQuantConfig::QuantizationGranularity granularity,
                            FixedPointScale shift_tensor,
                            const std::array<FixedPointScale, 3> &shift_gate,
                            const dev::vector<FixedPointScale> &shift_channel,
                            QuantBitWidth bw) {
    const size_t channel_size = hidden_size * 3;
    const size_t total_size = input_size * channel_size;
    
    if (granularity == OperatorQuantConfig::PER_TENSOR) {
        // Per-tensor: 直接调用 quantificationFP
        quantificationFP<Training>(src, quant_data, mask, total_size, shift_tensor, 0, bw);
    } else if (granularity == OperatorQuantConfig::PER_GATE) {
        // Per-gate: 调用 quantificationPerGateFP
        quantificationPerGateFP<Training>(src, quant_data, mask, input_size, hidden_size,
                                          shift_gate[0], shift_gate[1], shift_gate[2], bw);
    } else if (granularity == OperatorQuantConfig::PER_CHANNEL) {
        // Per-channel: 调用 quantificationPerChannelFP
        // 检查 shift_channel 是否有效
        if (shift_channel.size() == 0) {
            printf("quantificationWeightFP: shift_channel is empty for PER_CHANNEL granularity (expected size: %zu)\n", channel_size);
            return;
        }
        if (shift_channel.size() != channel_size) {
            printf("quantificationWeightFP: shift_channel size mismatch for PER_CHANNEL granularity (got %zu, expected %zu)\n", 
                   shift_channel.size(), channel_size);
            return;
        }
        quantificationPerChannelFP<Training>(src, quant_data, mask, input_size, channel_size,
                                             shift_channel, bw);
    }
}

template void quantificationWeightFP<false>(const float *src, float *quant_data, uint8_t *mask,
                                            size_t input_size, size_t hidden_size,
                                            OperatorQuantConfig::QuantizationGranularity granularity,
                                            FixedPointScale shift_tensor,
                                            const std::array<FixedPointScale, 3> &shift_gate,
                                            const dev::vector<FixedPointScale> &shift_channel,
                                            QuantBitWidth bw);
template void quantificationWeightFP<true>(const float *src, float *quant_data, uint8_t *mask,
                                           size_t input_size, size_t hidden_size,
                                           OperatorQuantConfig::QuantizationGranularity granularity,
                                           FixedPointScale shift_tensor,
                                           const std::array<FixedPointScale, 3> &shift_gate,
                                           const dev::vector<FixedPointScale> &shift_channel,
                                           QuantBitWidth bw);

template void quantificationPerChannelBitwidth<false>(const float *src, int32_t *quant_data, uint8_t *mask,
                                                      size_t input_size, size_t channel_size,
                                                      const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw);
template void quantificationPerChannelBitwidth<true>(const float *src, int32_t *quant_data, uint8_t *mask,
                                                     size_t input_size, size_t channel_size,
                                                     const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw);

// Bias 特殊量化函数（使用 round(bias / scale / 128) * 128）
// @tparam Training 是否训练模式（决定是否使用 mask）
template <bool Training>
void quantificationBiasFP(const float *src, float *quant_data, uint8_t *mask,
                          size_t channel_size,
                          const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw) {
    size_t block = 256;
    size_t grid = (channel_size + block - 1) / block;
    kernel::quantificationBiasFP<Training><<<grid, block>>>(src, quant_data, mask, channel_size, exp2_invs.data(), bw);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("quantificationBiasFP kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// 显式实例化模板函数
template void quantificationBiasFP<false>(const float *src, float *quant_data, uint8_t *mask,
                                          size_t channel_size,
                                          const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw);
template void quantificationBiasFP<true>(const float *src, float *quant_data, uint8_t *mask,
                                         size_t channel_size,
                                         const dev::vector<FixedPointScale> &exp2_invs, QuantBitWidth bw);

void dequantificationFP(const float *quant_data, float *data, size_t size,
                        FixedPointScale exp2_inv, int32_t zp) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::dequantificationFP<<<grid, block>>>(quant_data, data, size, exp2_inv, zp);
    cudaDeviceSynchronize();
}

void dequantificationFPInplace(float *data, size_t size,
                               FixedPointScale exp2_inv, int32_t zp) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::dequantificationFPInplace<<<grid, block>>>(data, size, exp2_inv, zp);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in dequantificationFPInplace: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in dequantificationFPInplace: ") + err_str);
    }
}

void dequantificationVFP(const float *quant_data, float *data, int time_steps, int batch_size,
                         int hidden_size, FixedPointScale shift_z, int32_t zp_z, FixedPointScale shift_r,
                         int32_t zp_r, FixedPointScale shift_g, int32_t zp_g,
                         FixedPointScale shift_hh, int32_t zp_hh) {
    const dim3 blockDim(hidden_size);
    const dim3 gridDim(time_steps, batch_size);

    kernel::dequantificationVFP<<<gridDim, blockDim>>>(
        quant_data, data, time_steps, batch_size, hidden_size,
        shift_z, zp_z, shift_r, zp_r, shift_g, zp_g, shift_hh, zp_hh);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("dequantificationVFP kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void dequantificationVFPInplace(float *data, int time_steps, int batch_size,
                                 int hidden_size, FixedPointScale shift_z, int32_t zp_z, FixedPointScale shift_r,
                                 int32_t zp_r, FixedPointScale shift_g, int32_t zp_g,
                                 FixedPointScale shift_hh, int32_t zp_hh) {
    const dim3 blockDim(hidden_size);
    const dim3 gridDim(time_steps, batch_size);

    kernel::dequantificationVFPInplace<<<gridDim, blockDim>>>(
        data, time_steps, batch_size, hidden_size,
        shift_z, zp_z, shift_r, zp_r, shift_g, zp_g, shift_hh, zp_hh);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in dequantificationVFPInplace: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in dequantificationVFPInplace: ") + err_str);
    }
    cudaDeviceSynchronize();
}

void dequantificationPerChannelFP(const float *quant_data, float *data,
                                  size_t input_size, size_t channel_size,
                                  const dev::vector<FixedPointScale> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::dequantificationPerChannelFP<<<gridDim, blockDim>>>(
        quant_data, data, input_size, channel_size, exp2_invs.data());
    
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in dequantificationPerChannelFP: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in dequantificationPerChannelFP: ") + err_str);
    }
}

void dequantificationPerChannelFPInplace(float *data,
                                        size_t input_size, size_t channel_size,
                                        const dev::vector<FixedPointScale> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::dequantificationPerChannelFPInplace<<<gridDim, blockDim>>>(
        data, input_size, channel_size, exp2_invs.data());
    
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in dequantificationPerChannelFPInplace: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in dequantificationPerChannelFPInplace: ") + err_str);
    }
}

void dequantificationPerGateFPInplace(float *data,
                                     size_t input_size, size_t hidden_size,
                                     FixedPointScale exp2_inv_z, FixedPointScale exp2_inv_r, FixedPointScale exp2_inv_g) {
    const dim3 blockDim(32, 16);
    const size_t channel_size = hidden_size * 3;
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::dequantificationPerGateFPInplace<<<gridDim, blockDim>>>(
        data, input_size, hidden_size, exp2_inv_z, exp2_inv_r, exp2_inv_g);
    
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        const char *err_str = cudaGetErrorString(err);
        fprintf(stderr, "CUDA error in dequantificationPerGateFPInplace: %s\n", err_str);
        throw std::runtime_error(std::string("CUDA error in dequantificationPerGateFPInplace: ") + err_str);
    }
}

// 统一的反量化接口：根据 granularity 自动选择 per-tensor、per-gate 或 per-channel 反量化
void dequantificationWeightFPInplace(float *data,
                                     size_t input_size, size_t hidden_size,
                                     OperatorQuantConfig::QuantizationGranularity granularity,
                                     FixedPointScale shift_tensor,
                                     const std::array<FixedPointScale, 3> &shift_gate,
                                     const dev::vector<FixedPointScale> &shift_channel) {
    const size_t channel_size = hidden_size * 3;
    const size_t total_size = input_size * channel_size;
    
    if (granularity == OperatorQuantConfig::PER_TENSOR) {
        // Per-tensor: 直接调用 dequantificationFPInplace
        dequantificationFPInplace(data, total_size, shift_tensor, 0);
    } else if (granularity == OperatorQuantConfig::PER_GATE) {
        // Per-gate: 调用 dequantificationPerGateFPInplace
        dequantificationPerGateFPInplace(data, input_size, hidden_size,
                                        shift_gate[0], shift_gate[1], shift_gate[2]);
    } else if (granularity == OperatorQuantConfig::PER_CHANNEL) {
        // Per-channel: 调用 dequantificationPerChannelFPInplace
        // 检查 shift_channel 是否有效
        if (shift_channel.size() == 0) {
            printf("dequantificationWeightFPInplace: shift_channel is empty for PER_CHANNEL granularity (expected size: %zu)\n", channel_size);
            return;
        }
        if (shift_channel.size() != channel_size) {
            printf("dequantificationWeightFPInplace: shift_channel size mismatch for PER_CHANNEL granularity (got %zu, expected %zu)\n", 
                   shift_channel.size(), channel_size);
            return;
        }
        dequantificationPerChannelFPInplace(data, input_size, channel_size, shift_channel);
    }
}

template <typename T, typename QuantT>
void dequantification(const QuantT *quant_data, T *data, size_t size, FixedPointScale exp2_inv, int32_t zp) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::dequantification<<<grid, block>>>(quant_data, data, size, exp2_inv, zp);
    cudaDeviceSynchronize();
}

template void dequantification<float, int8_t>(const int8_t *quant_data, float *data, size_t size,
                                              FixedPointScale exp2_inv, int32_t zp);
template void dequantification<float, int16_t>(const int16_t *quant_data, float *data, size_t size,
                                               FixedPointScale exp2_inv, int32_t zp);
template void dequantification<float, int32_t>(const int32_t *quant_data, float *data, size_t size,
                                               FixedPointScale exp2_inv, int32_t zp);

// v 统一使用 int32_t 存储
// V 向量布局: [z_out, r_out, g_out, weight_hh_linear_g]
template <typename T>
void dequantificationV(const int32_t *quant_data, T *data, int time_steps, int batch_size,
                       int hidden_size, FixedPointScale shift_z, int32_t zp_z, FixedPointScale shift_r,
                       int32_t zp_r, FixedPointScale shift_g, int32_t zp_g, 
                       FixedPointScale shift_hh, int32_t zp_hh) {
    // Launch configuration: 每个block处理一个时间步和一个batch的所有hidden单元
    // blockDim.x = hidden_size (每个线程处理一个hidden单元)
    // gridDim.x = time_steps
    // gridDim.y = batch_size
    const dim3 blockDim(hidden_size);
    const dim3 gridDim(time_steps, batch_size);

    kernel::dequantificationV<<<gridDim, blockDim>>>(
        quant_data, data, time_steps, batch_size, hidden_size, shift_z, zp_z, shift_r, zp_r,
        shift_g, zp_g, shift_hh, zp_hh);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("dequantificationV kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template void dequantificationV<float>(const int32_t *quant_data, float *data, int time_steps,
                                       int batch_size, int hidden_size, FixedPointScale shift_z,
                                       int32_t zp_z, FixedPointScale shift_r, int32_t zp_r,
                                       FixedPointScale shift_g, int32_t zp_g, 
                                       FixedPointScale shift_hh, int32_t zp_hh);


template <typename T, typename QuantT>
void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                size_t channel_size, const dev::vector<FixedPointScale> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::dequantificationPerChannel<<<gridDim, blockDim>>>(quant_data, data, input_size,
                                                              channel_size, exp2_invs.data());
    cudaDeviceSynchronize();
}

template void dequantificationPerChannel<float, int8_t>(const int8_t *quant_data, float *data,
                                                        size_t input_size, size_t channel_size,
                                                        const dev::vector<FixedPointScale> &exp2_invs);
template void dequantificationPerChannel<float, int16_t>(const int16_t *quant_data, float *data,
                                                         size_t input_size, size_t channel_size,
                                                         const dev::vector<FixedPointScale> &exp2_invs);
template void dequantificationPerChannel<float, int32_t>(const int32_t *quant_data, float *data,
                                                         size_t input_size, size_t channel_size,
                                                         const dev::vector<FixedPointScale> &exp2_invs);
}  // namespace dev

// ==================== 分段线性量化参数生成函数 ====================

// 线性拟合函数（最小二乘法）
inline void linear_fit(const std::vector<float> &x, const std::vector<float> &y, float &b,
                       float &c) {
    int n = x.size();
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;

    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
    }

    float denom = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denom) < 1e-9f) {
        b = 0.0f;
        c = sum_y / n;
        return;
    }

    b = (n * sum_xy - sum_x * sum_y) / denom;
    c = (sum_y - b * sum_x) / n;
}

// 自适应分段（Sigmoid/Tanh 专用）
// 🔥 基于导数的权重分配，与 Python 参考 (bc_ds_U8.py) 保持一致
// 关键：中心区域固定在 x = 0 附近（sigmoid/tanh 的特性），不是输入范围的中心
std::vector<float> adaptive_segmentation_sigmoid(float x_min, float x_max, int num_segments) {
    // Sigmoid/Tanh 的权重配置（与 Python 参考一致）
    // centerWeight: 中心区域的权重倍数
    // centerRange: 中心区域的半宽度
    const float centerWeight = 5.0f;  // sigmoid: 5.0, tanh: 4.0
    const float centerRange = 2.0f;   // |x| < 2.0 的区域权重增加

    // 1. 在输入范围内均匀采样，计算权重
    const int numSamples = 1000;
    std::vector<float> xSamples(numSamples);
    std::vector<float> weights(numSamples - 1);

    for (int i = 0; i < numSamples; i++) {
        xSamples[i] = x_min + (x_max - x_min) * static_cast<float>(i) / (numSamples - 1);
    }

    // 2. 计算导数（斜率）和权重
    for (int i = 0; i < numSamples - 1; i++) {
        float x = xSamples[i];
        float x_next = xSamples[i + 1];

        // 计算 sigmoid 的导数 y' = y * (1 - y)，其中 y = sigmoid(x)
        float y = 1.0f / (1.0f + std::exp(-x));
        float y_next = 1.0f / (1.0f + std::exp(-x_next));
        float slope = std::abs(y_next - y) / (x_next - x + 1e-9f);

        // 距离 x = 0 的距离（与 Python 参考一致）
        float distToCenter = std::abs(x);

        // 计算权重
        if (distToCenter < centerRange) {
            // 中心区域：权重随距离线性递减
            weights[i] = centerWeight * (1.0f - distToCenter / centerRange) + 1.0f;
        } else {
            // 外侧区域：基于斜率的权重
            weights[i] = 1.0f + slope * 0.5f;
        }
    }

    // 3. 归一化权重
    float sumWeights = 0.0f;
    for (int i = 0; i < numSamples - 1; i++) {
        sumWeights += weights[i];
    }
    for (int i = 0; i < numSamples - 1; i++) {
        weights[i] /= sumWeights;
    }

    // 4. 计算累积权重
    std::vector<float> cumWeights(numSamples - 1);
    cumWeights[0] = weights[0];
    for (int i = 1; i < numSamples - 1; i++) {
        cumWeights[i] = cumWeights[i - 1] + weights[i];
    }

    // 5. 根据累积权重生成分段点
    std::vector<float> points;
    points.push_back(x_min);

    for (int i = 1; i < num_segments; i++) {
        float target = static_cast<float>(i) / num_segments;

        // 二分查找目标累积权重对应的 x 值
        auto it = std::lower_bound(cumWeights.begin(), cumWeights.end(), target);
        int idx = static_cast<int>(std::distance(cumWeights.begin(), it));
        if (idx >= numSamples - 1) idx = numSamples - 2;
        if (idx < 0) idx = 0;

        points.push_back(xSamples[idx]);
    }

    points.push_back(x_max);

    // 6. 确保点单调递增且无重复
    std::sort(points.begin(), points.end());
    auto last = std::unique(points.begin(), points.end(),
                            [](float a, float b) { return std::abs(a - b) < 1e-9f; });
    points.erase(last, points.end());

    // 如果去重后点数不够，在最大间隔处插入点
    while (static_cast<int>(points.size()) < num_segments + 1) {
        float max_gap = 0.0f;
        size_t max_gap_idx = 0;
        for (size_t i = 0; i < points.size() - 1; i++) {
            float gap = points[i + 1] - points[i];
            if (gap > max_gap) {
                max_gap = gap;
                max_gap_idx = i;
            }
        }
        float new_point = (points[max_gap_idx] + points[max_gap_idx + 1]) / 2.0f;
        points.insert(points.begin() + max_gap_idx + 1, new_point);
    }

    return points;
}

// ==================== 统一的 LUT 生成函数 ====================
//
// 【设计原则】
//   - 所有位宽配置使用统一的 SigmoidLUT 结构
//   - q_b 使用 int32_t 避免溢出（tanh 斜率 1.0 需要此精度）
//   - 根据 input_bw 自动确定输入范围
//
// =========================================================================

/**
 * @brief 统一的 Sigmoid LUT 生成函数
 * @param shift_bits_x 输入量化 shift bits
 * @param zp_x 输入 zero-point
 * @param shift_bits_y 输出量化 shift bits
 * @param zp_y 输出 zero-point
 * @param input_bw 输入位宽（决定输入范围）
 * @param output_bw 输出位宽（决定 shift_bits_b 精度）
 */
static inline std::pair<int32_t, int8_t> quantize_multiplier_signed(float value, QuantBitWidth output_bw) {
    if (std::abs(value) < 1e-12f) {
        return {0, 0};
    }
    const int32_t max_q = output_bw.qmax();
    int8_t n = static_cast<int8_t>(std::floor(std::log2(static_cast<double>(max_q) / std::abs(value))));
    int64_t q = round_to_int64(static_cast<double>(value) * std::ldexp(1.0, n));
    q = std::max<int64_t>(-max_q, std::min<int64_t>(max_q, q));
    return {static_cast<int32_t>(q), n};
}

SigmoidLUT generate_sigmoid_lut(float effective_scale_x, int32_t zp_x, float effective_scale_y,
                                 int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    // 根据输入位宽确定量化范围（任意位宽支持）
    int32_t quant_min = input_bw.qmin();
    int32_t quant_max = input_bw.qmax();

    float scale_x = effective_scale_x;
    float x_min = static_cast<float>(quant_min - zp_x) * scale_x;
    float x_max = static_cast<float>(quant_max - zp_x) * scale_x;

    // Sigmoid 有效范围限制
    constexpr float SIGMOID_EFFECTIVE_RANGE = 8.0f;
    x_min = std::max(x_min, -SIGMOID_EFFECTIVE_RANGE);
    x_max = std::min(x_max, SIGMOID_EFFECTIVE_RANGE);

#ifdef DEBUG
    printf("[DEBUG] generate_sigmoid_lut: input_bw=%d, eff_scale_x=%.8f, zp_x=%d, x_range=[%.4f, %.4f]\n",
           static_cast<int>(input_bw), scale_x, zp_x, x_min, x_max);
#endif

    SigmoidLUT lut{};
    lut.shift_bits_x = 0;
    lut.zp_x = zp_x;
    lut.shift_bits_y = 0;
    lut.zp_y = zp_y;
    lut.effective_scale_x = effective_scale_x;
    lut.effective_scale_y = effective_scale_y;

    // 生成分段点
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    // 第一遍扫描：拟合所有分段
    struct SegmentCoeffs { float x_start, x_end, b, c; };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples), y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));  // Sigmoid
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);
        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    const float scale_y = effective_scale_y;

    // 第三遍扫描：量化每段
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto &coeff = all_coeffs[i];
        const float m_real = coeff.b * (scale_x / scale_y);
        auto [q_b, n_BX_total] = quantize_multiplier_signed(m_real, output_bw);
        int32_t term_c_precomputed = round_to_int(coeff.c / scale_y + static_cast<float>(zp_y));

        // threshold 量化（任意位宽支持，存储为 int32_t）
        int32_t threshold = round_to_int(coeff.x_end / scale_x + zp_x);
        threshold = clamp_by_bitwidth(threshold, input_bw);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    return generate_sigmoid_lut(exp2_scale(shift_bits_x), zp_x, exp2_scale(shift_bits_y), zp_y, input_bw, output_bw);
}

/**
 * @brief 统一的 Tanh LUT 生成函数
 * @param shift_bits_x 输入量化 shift bits
 * @param zp_x 输入 zero-point
 * @param shift_bits_y 输出量化 shift bits
 * @param zp_y 输出 zero-point
 * @param input_bw 输入位宽（决定输入范围）
 * @param output_bw 输出位宽（决定 shift_bits_b 精度）
 */
SigmoidLUT generate_tanh_lut(float effective_scale_x, int32_t zp_x, float effective_scale_y,
                              int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    // 根据输入位宽确定量化范围（任意位宽支持）
    int32_t quant_min = input_bw.qmin();
    int32_t quant_max = input_bw.qmax();

    float scale_x = effective_scale_x;
    float x_min = static_cast<float>(quant_min - zp_x) * scale_x;
    float x_max = static_cast<float>(quant_max - zp_x) * scale_x;

    // Tanh 有效范围限制
    constexpr float TANH_EFFECTIVE_RANGE = 4.0f;
    x_min = std::max(x_min, -TANH_EFFECTIVE_RANGE);
    x_max = std::min(x_max, TANH_EFFECTIVE_RANGE);

#ifdef DEBUG
    printf("[DEBUG] generate_tanh_lut: input_bw=%d, eff_scale_x=%.8f, zp_x=%d, x_range=[%.4f, %.4f]\n",
           static_cast<int>(input_bw), scale_x, zp_x, x_min, x_max);
#endif

    SigmoidLUT lut{};
    lut.shift_bits_x = 0;
    lut.zp_x = zp_x;
    lut.shift_bits_y = 0;
    lut.zp_y = zp_y;
    lut.effective_scale_x = effective_scale_x;
    lut.effective_scale_y = effective_scale_y;

    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    struct SegmentCoeffs { float x_start, x_end, b, c; };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples), y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = std::tanh(x_val);  // Tanh
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);
        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    const float scale_y = effective_scale_y;

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto &coeff = all_coeffs[i];
        const float m_real = coeff.b * (scale_x / scale_y);
        auto [q_b, n_BX_total] = quantize_multiplier_signed(m_real, output_bw);
        int32_t term_c_precomputed = round_to_int(coeff.c / scale_y + static_cast<float>(zp_y));
        
        // threshold 量化（任意位宽支持，存储为 int32_t）
        int32_t threshold = round_to_int(coeff.x_end / scale_x + zp_x);
        threshold = clamp_by_bitwidth(threshold, input_bw);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                             int32_t zp_y, QuantBitWidth input_bw, QuantBitWidth output_bw) {
    return generate_tanh_lut(exp2_scale(shift_bits_x), zp_x, exp2_scale(shift_bits_y), zp_y, input_bw, output_bw);
}
