#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "dev_vector.h"
#include "quantize_ops.cuh"
#include "quantize_ops_helper.h"

// 调试开关：取消注释以启用调试输出
#define DEBUG_QUANT

// 统一的 LUT 生成函数（前向声明）
SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, 
                                 int32_t zp_y, QuantBitWidth input_bw);
SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, 
                              int32_t zp_y, QuantBitWidth input_bw);

// 分段线性量化常量内存（统一结构）
__constant__ SigmoidLUT d_sigmoid_z_lut;  // z 门的 Sigmoid LUT
__constant__ SigmoidLUT d_sigmoid_r_lut;  // r 门的 Sigmoid LUT
__constant__ SigmoidLUT d_tanh_lut;       // g 门的 Tanh LUT

// sigmoid 输出使用 uint8_t，因为 sigmoid ∈ [0, 1] 没有负数
std::vector<uint8_t> generate_sigmoid_int8_lut(float scale_z_pre, int32_t zp_z_pre, float scale_z,
                                               int32_t zp_z) {
    std::vector<uint8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;

        const float x_fp = static_cast<float>(x_i8 - zp_z_pre) * scale_z_pre;
        const float y_fp = 1.f / (1.f + std::exp(-x_fp));

        // 输出使用 uint8_t 范围 [0, 255]
        int y_u8 = static_cast<int>(std::round(y_fp / scale_z + zp_z));
        if (y_u8 < 0) y_u8 = 0;
        if (y_u8 > 255) y_u8 = 255;

        lut[i] = static_cast<uint8_t>(y_u8);
    }
    return lut;
}

std::vector<int8_t> generate_tanh_int8_lut(float scale_pre, int32_t zp_pre, float scale_out,
                                           int32_t zp_out) {
    std::vector<int8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;

        float x_fp = (x_i8 - zp_pre) * scale_pre;
        float y_fp = std::tanh(x_fp);

        int y_i8 = static_cast<int>(std::round(y_fp / scale_out + zp_out));
        if (y_i8 < -128) y_i8 = -128;
        if (y_i8 > 127) y_i8 = 127;

        lut[i] = static_cast<int8_t>(y_i8);
    }
    return lut;
}

// sigmoid 输出使用 uint8_t，因为 sigmoid ∈ [0, 1] 没有负数
std::vector<uint8_t> generate_sigmoid_int8_lut_exp2(int8_t exp2_inv_z_pre, int32_t zp_z_pre,
                                                    int8_t exp2_inv_z, int32_t zp_z) {
    std::vector<uint8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;

        // （1）反量化 x
        float x_fp = dequantize(x_i8, exp2_inv_z_pre, zp_z_pre);

        // （2）计算 sigmoid
        float y_fp = 1.f / (1.f + std::exp(-x_fp));

        // （3）量化 y 到 uint8_t 范围 [0, 255]
        int y_u8 = quantize<uint8_t>(y_fp, exp2_inv_z, zp_z);

        lut[i] = static_cast<uint8_t>(y_u8);
    }

    return lut;
}

std::vector<int8_t> generate_tanh_int8_lut_exp2(int8_t exp2_inv_pre, int32_t zp_pre,
                                                int8_t exp2_inv_out, int32_t zp_out) {
    std::vector<int8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;

        // （1）反量化 x
        float x_fp = dequantize(x_i8, exp2_inv_pre, zp_pre);

        // （2）tanh
        float y_fp = std::tanh(x_fp);

        // （3）量化 y
        int y_i8 = quantize<int8_t>(y_fp, exp2_inv_out, zp_out);

        lut[i] = static_cast<int8_t>(y_i8);
    }

    return lut;
}

// 生成分段线性量化表（基于GRUQuantitativeParameters，根据bitwidth_config_中的实际位宽配置）
// 统一的 LUT 生成与初始化函数
void generate_piecewise_linear_lut(const GRUQuantitativeParameters &params) {
    const auto &config = params.bitwidth_config_;

    // z 门 Sigmoid
    SigmoidLUT z_lut = generate_sigmoid_lut(
        params.exp2_inv_z_pre_, params.zp_z_pre_,
        params.exp2_inv_z_out_, params.zp_z_out_,
        config.z_pre_);
    cudaMemcpyToSymbol(d_sigmoid_z_lut, &z_lut, sizeof(SigmoidLUT));

    // r 门 Sigmoid
    SigmoidLUT r_lut = generate_sigmoid_lut(
        params.exp2_inv_r_pre_, params.zp_r_pre_,
        params.exp2_inv_r_out_, params.zp_r_out_,
        config.r_pre_);
    cudaMemcpyToSymbol(d_sigmoid_r_lut, &r_lut, sizeof(SigmoidLUT));

    // g 门 Tanh
    SigmoidLUT g_lut = generate_tanh_lut(
        params.exp2_inv_g_pre_, params.zp_g_pre_,
        params.exp2_inv_g_out_, params.zp_g_out_,
        config.g_pre_);
    cudaMemcpyToSymbol(d_tanh_lut, &g_lut, sizeof(SigmoidLUT));

#ifdef DEBUG_QUANT
    printf("[DEBUG] generate_piecewise_linear_lut: z/r/g LUTs initialized\n");
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

#ifdef DEBUG_QUANT
    // 调试输出
    if (row == 0) {
        printf("[DEBUG] computeWeightSumMulZP: row=0, in_dim=%d, x_zp=%d, result=%lld\n", in_dim,
               x_zp, (long long)sum_i64);
    }
#endif

    // 使用 int64_t 存储完整结果
    weight_sum[row] = sum_i64;
}

// 兼容旧版本：int32_t 输出（用于 8 位量化，不会溢出）
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
__global__ void quantification(const T *data, QuantT *quant_data, size_t size, int8_t exp2_inv,
                               int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    quant_data[idx] = dev::quantize<QuantT>(data[idx], exp2_inv, zp);
}

template <typename T, typename QuantT>
__global__ void dequantification(const QuantT *quant_data, T *data, size_t size, int8_t exp2_inv,
                                 int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    data[idx] = dequantize<QuantT>(quant_data[idx], exp2_inv, zp);
}

}  // namespace kernel

namespace kernel {

// v 使用 int32_t 存储，但内部各部分使用不同的量化参数:
// - z: 使用 exp2_inv_z, zp_z
// - r: 使用 exp2_inv_r, zp_r
// - g: 使用 exp2_inv_g, zp_g
// - Rh_add_br_g: 使用 exp2_inv_Rh_add_br, zp_Rh_add_br
template <typename T>
__global__ void dequantificationV(const int32_t *quant_data, T *data, int time_steps,
                                  int batch_size, int hidden_size, int8_t exp2_inv_z, int32_t zp_z,
                                  int8_t exp2_inv_r, int32_t zp_r, int8_t exp2_inv_g, int32_t zp_g,
                                  int8_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br) {
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
    // 4个部分: [z_out, r_out, g_out, Rh_add_br_g]，每个部分大小为 hidden_size

    const int base_idx = t * (batch_size * hidden_size * 4) + b * (hidden_size * 4);

    // 反量化 z_out (第0部分) - 从 int32_t 反量化
    const int z_idx = base_idx + 0 * hidden_size + h;
    data[z_idx] = dequantize<int32_t>(quant_data[z_idx], exp2_inv_z, zp_z);

    // 反量化 r_out (第1部分) - 从 int32_t 反量化
    const int r_idx = base_idx + 1 * hidden_size + h;
    data[r_idx] = dequantize<int32_t>(quant_data[r_idx], exp2_inv_r, zp_r);

    // 反量化 g_out (第2部分) - 从 int32_t 反量化
    const int g_idx = base_idx + 2 * hidden_size + h;
    data[g_idx] = dequantize<int32_t>(quant_data[g_idx], exp2_inv_g, zp_g);

    // 反量化 Rh_add_br_g (第3部分) - 从 int32_t 反量化
    const int rh_idx = base_idx + 3 * hidden_size + h;
    data[rh_idx] = dequantize<int32_t>(quant_data[rh_idx], exp2_inv_Rh_add_br, zp_Rh_add_br);
}

template <typename T, typename QuantT>
__global__ void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                                         size_t channel_size, const int8_t *exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const int8_t exp2_inv = exp2_invs[channel_idx];

    const size_t idx = input_idx * channel_size + channel_idx;
    quant_data[idx] = dev::quantize<QuantT>(src[idx], exp2_inv, 0);
}

template <typename T, typename QuantT>
__global__ void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                           size_t channel_size, const int8_t *exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const int8_t exp2_inv = exp2_invs[channel_idx];

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

namespace dev {

template <typename T, typename QuantT>
void quantification(const T *data, QuantT *quant_data, size_t size, int8_t exp2_inv, int32_t zp) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::quantification<<<grid, block>>>(data, quant_data, size, exp2_inv, zp);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template void quantification<float, int8_t>(const float *data, int8_t *quant_data, size_t size,
                                            int8_t exp2_inv, int32_t zp);
template void quantification<float, int16_t>(const float *data, int16_t *quant_data, size_t size,
                                             int8_t exp2_inv, int32_t zp);
template void quantification<float, int32_t>(const float *data, int32_t *quant_data, size_t size,
                                             int8_t exp2_inv, int32_t zp);

template <typename T, typename QuantT>
void dequantification(const QuantT *quant_data, T *data, size_t size, int8_t exp2_inv, int32_t zp) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::dequantification<<<grid, block>>>(quant_data, data, size, exp2_inv, zp);
    cudaDeviceSynchronize();
}

template void dequantification<float, int8_t>(const int8_t *quant_data, float *data, size_t size,
                                              int8_t exp2_inv, int32_t zp);
template void dequantification<float, int16_t>(const int16_t *quant_data, float *data, size_t size,
                                               int8_t exp2_inv, int32_t zp);
template void dequantification<float, int32_t>(const int32_t *quant_data, float *data, size_t size,
                                               int8_t exp2_inv, int32_t zp);

// v 统一使用 int32_t 存储
template <typename T>
void dequantificationV(const int32_t *quant_data, T *data, int time_steps, int batch_size,
                       int hidden_size, int8_t exp2_inv_z, int32_t zp_z, int8_t exp2_inv_r,
                       int32_t zp_r, int8_t exp2_inv_g, int32_t zp_g, int8_t exp2_inv_Rh_add_br,
                       int32_t zp_Rh_add_br) {
    // Launch configuration: 每个block处理一个时间步和一个batch的所有hidden单元
    // blockDim.x = hidden_size (每个线程处理一个hidden单元)
    // gridDim.x = time_steps
    // gridDim.y = batch_size
    const dim3 blockDim(hidden_size);
    const dim3 gridDim(time_steps, batch_size);

    kernel::dequantificationV<<<gridDim, blockDim>>>(
        quant_data, data, time_steps, batch_size, hidden_size, exp2_inv_z, zp_z, exp2_inv_r, zp_r,
        exp2_inv_g, zp_g, exp2_inv_Rh_add_br, zp_Rh_add_br);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("dequantificationV kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template void dequantificationV<float>(const int32_t *quant_data, float *data, int time_steps,
                                       int batch_size, int hidden_size, int8_t exp2_inv_z,
                                       int32_t zp_z, int8_t exp2_inv_r, int32_t zp_r,
                                       int8_t exp2_inv_g, int32_t zp_g, int8_t exp2_inv_Rh_add_br,
                                       int32_t zp_Rh_add_br);

template <typename T, typename QuantT>
void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                              size_t channel_size, const dev::vector<int8_t> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::quantificationPerChannel<<<gridDim, blockDim>>>(src, quant_data, input_size,
                                                            channel_size, exp2_invs.data());
    cudaDeviceSynchronize();
}

template void quantificationPerChannel<float, int8_t>(const float *src, int8_t *quant_data,
                                                      size_t input_size, size_t channel_size,
                                                      const dev::vector<int8_t> &exp2_invs);

template void quantificationPerChannel<float, int16_t>(const float *src, int16_t *quant_data,
                                                       size_t input_size, size_t channel_size,
                                                       const dev::vector<int8_t> &exp2_invs);
template void quantificationPerChannel<float, int32_t>(const float *src, int32_t *quant_data,
                                                       size_t input_size, size_t channel_size,
                                                       const dev::vector<int8_t> &exp2_invs);

template <typename T, typename QuantT>
void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                size_t channel_size, const dev::vector<int8_t> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::dequantificationPerChannel<<<gridDim, blockDim>>>(quant_data, data, input_size,
                                                              channel_size, exp2_invs.data());
    cudaDeviceSynchronize();
}

template void dequantificationPerChannel<float, int8_t>(const int8_t *quant_data, float *data,
                                                        size_t input_size, size_t channel_size,
                                                        const dev::vector<int8_t> &exp2_invs);
template void dequantificationPerChannel<float, int16_t>(const int16_t *quant_data, float *data,
                                                         size_t input_size, size_t channel_size,
                                                         const dev::vector<int8_t> &exp2_invs);
template void dequantificationPerChannel<float, int32_t>(const int32_t *quant_data, float *data,
                                                         size_t input_size, size_t channel_size,
                                                         const dev::vector<int8_t> &exp2_invs);
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
 */
SigmoidLUT generate_sigmoid_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                 int32_t zp_y, QuantBitWidth input_bw) {
    // 根据输入位宽确定量化范围
    int32_t quant_min, quant_max;
    if (input_bw == QuantBitWidth::INT16) {
        quant_min = -32768;
        quant_max = 32767;
    } else {  // INT8 或其他
        quant_min = -128;
        quant_max = 127;
    }

    float scale_x = std::pow(2.0f, -static_cast<float>(shift_bits_x));
    float x_min = static_cast<float>(quant_min - zp_x) * scale_x;
    float x_max = static_cast<float>(quant_max - zp_x) * scale_x;

    // Sigmoid 有效范围限制
    constexpr float SIGMOID_EFFECTIVE_RANGE = 8.0f;
    x_min = std::max(x_min, -SIGMOID_EFFECTIVE_RANGE);
    x_max = std::min(x_max, SIGMOID_EFFECTIVE_RANGE);

#ifdef DEBUG_QUANT
    printf("[DEBUG] generate_sigmoid_lut: input_bw=%d, shift_x=%d, zp_x=%d, x_range=[%.4f, %.4f]\n",
           static_cast<int>(input_bw), shift_bits_x, zp_x, x_min, x_max);
#endif

    SigmoidLUT lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

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

    // 第二遍扫描：统一量化参数
    float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
    float zp_y_offset = static_cast<float>(zp_y) * scale_y;

    float b_abs_max = 0.0f, c_abs_max = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        b_abs_max = std::max(b_abs_max, std::abs(all_coeffs[i].b));
        float c_adjusted = all_coeffs[i].c + zp_y_offset;
        c_abs_max = std::max(c_abs_max, std::abs(c_adjusted));
    }

    if (b_abs_max < 1e-9f) b_abs_max = 1e-9f;
    if (c_abs_max < 1e-9f) c_abs_max = 1e-9f;

    // 使用 INT16 精度的 shift_bits_b，避免 n_BX_total 过大导致精度损失
    // q_b 用 INT32 存储不会溢出，但 shift_bits_b 要控制在合理范围
    int8_t shift_bits_b = determine_shift_bits_int16(b_abs_max);
    int8_t shift_bits_c = determine_shift_bits_int16(c_abs_max);

    // 第三遍扫描：量化每段
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto &coeff = all_coeffs[i];
        float c_adjusted = coeff.c + zp_y_offset;

        int32_t q_b = quantize_coefficient_int32(coeff.b, shift_bits_b);
        int32_t q_c = quantize_coefficient_int32(c_adjusted, shift_bits_c);

        int8_t n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        int32_t term_c_precomputed = (n_yc >= 0) ? (q_c >> n_yc) : (q_c << (-n_yc));

        // threshold 根据输入位宽量化，存储为 int16_t
        int16_t threshold;
        if (input_bw == QuantBitWidth::INT16) {
            threshold = quantize_input_int16(coeff.x_end, shift_bits_x, zp_x);
        } else {
            // INT8 输入：clamp 到 [-128, 127]
            threshold = static_cast<int16_t>(quantize_input_int8(coeff.x_end, shift_bits_x, zp_x));
        }

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

/**
 * @brief 统一的 Tanh LUT 生成函数
 */
SigmoidLUT generate_tanh_lut(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                              int32_t zp_y, QuantBitWidth input_bw) {
    // 根据输入位宽确定量化范围
    int32_t quant_min, quant_max;
    if (input_bw == QuantBitWidth::INT16) {
        quant_min = -32768;
        quant_max = 32767;
    } else {
        quant_min = -128;
        quant_max = 127;
    }

    float scale_x = std::pow(2.0f, -static_cast<float>(shift_bits_x));
    float x_min = static_cast<float>(quant_min - zp_x) * scale_x;
    float x_max = static_cast<float>(quant_max - zp_x) * scale_x;

    // Tanh 有效范围限制
    constexpr float TANH_EFFECTIVE_RANGE = 4.0f;
    x_min = std::max(x_min, -TANH_EFFECTIVE_RANGE);
    x_max = std::min(x_max, TANH_EFFECTIVE_RANGE);

#ifdef DEBUG_QUANT
    printf("[DEBUG] generate_tanh_lut: input_bw=%d, shift_x=%d, zp_x=%d, x_range=[%.4f, %.4f]\n",
           static_cast<int>(input_bw), shift_bits_x, zp_x, x_min, x_max);
#endif

    SigmoidLUT lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

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

    float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
    float zp_y_offset = static_cast<float>(zp_y) * scale_y;

    float b_abs_max = 0.0f, c_abs_max = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        b_abs_max = std::max(b_abs_max, std::abs(all_coeffs[i].b));
        float c_adjusted = all_coeffs[i].c + zp_y_offset;
        c_abs_max = std::max(c_abs_max, std::abs(c_adjusted));
    }

    if (b_abs_max < 1e-9f) b_abs_max = 1e-9f;
    if (c_abs_max < 1e-9f) c_abs_max = 1e-9f;

    // 使用 INT16 精度的 shift_bits_b，避免 n_BX_total 过大
    // tanh 斜率最大 1.0，用 INT16 精度足够（ceil 后 shift_bits_b = 15）
    int8_t shift_bits_b = determine_shift_bits_int16(b_abs_max);
    int8_t shift_bits_c = determine_shift_bits_int16(c_abs_max);

#ifdef DEBUG_QUANT
    printf("[DEBUG] generate_tanh_lut: b_abs_max=%.6f, shift_bits_b=%d\n", b_abs_max, shift_bits_b);
#endif

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto &coeff = all_coeffs[i];
        float c_adjusted = coeff.c + zp_y_offset;

        int32_t q_b = quantize_coefficient_int32(coeff.b, shift_bits_b);
        int32_t q_c = quantize_coefficient_int32(c_adjusted, shift_bits_c);

        int8_t n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        int32_t term_c_precomputed = (n_yc >= 0) ? (q_c >> n_yc) : (q_c << (-n_yc));
        
        // threshold 根据输入位宽量化
        int16_t threshold;
        if (input_bw == QuantBitWidth::INT16) {
            threshold = quantize_input_int16(coeff.x_end, shift_bits_x, zp_x);
        } else {
            threshold = static_cast<int16_t>(quantize_input_int8(coeff.x_end, shift_bits_x, zp_x));
        }

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}
