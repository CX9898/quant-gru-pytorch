#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "devVector.h"
#include "quantize_ops.cuh"
#include "quantize_ops_helper.hpp"

// 前向声明
struct SigmoidLUT_INT16;
struct SigmoidLUT_INT8;

__constant__ uint8_t d_sigmoid_int8_z_lut[256];  // sigmoid 输出 [0,1] 使用无符号
__constant__ uint8_t d_sigmoid_int8_r_lut[256];  // sigmoid 输出 [0,1] 使用无符号
__constant__ int8_t d_tanh_int8_g_lut[256];      // tanh 输出 [-1,1] 仍使用有符号

// 分段线性量化常量内存
__constant__ SigmoidLUT_INT16 d_sigmoid_z_lut_int16;  // z 门的 Sigmoid LUT
__constant__ SigmoidLUT_INT16 d_sigmoid_r_lut_int16;  // r 门的 Sigmoid LUT
__constant__ SigmoidLUT_INT16 d_tanh_lut_int16;
__constant__ SigmoidLUT_INT8 d_sigmoid_z_lut_int8;  // z 门的 Sigmoid LUT
__constant__ SigmoidLUT_INT8 d_sigmoid_r_lut_int8;  // r 门的 Sigmoid LUT
__constant__ SigmoidLUT_INT8 d_tanh_lut_int8;

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

void generate_int8_lut(float scale_z_pre, int32_t zp_z_pre, float scale_z_out, int32_t zp_z_out,
                       float scale_r_pre, int32_t zp_r_pre, float scale_r_out, int32_t zp_r_out,
                       float scale_g_pre, int32_t zp_g_pre, float scale_g_out, int32_t zp_g_out) {
    // sigmoid LUT 使用 uint8_t（输出 [0, 255]）
    std::vector<uint8_t> sigmoid_z_lut =
        generate_sigmoid_int8_lut(scale_z_pre, zp_z_pre, scale_z_out, zp_z_out);
    std::vector<uint8_t> sigmoid_r_lut =
        generate_sigmoid_int8_lut(scale_r_pre, zp_r_pre, scale_r_out, zp_r_out);
    // tanh LUT 仍使用 int8_t（输出 [-128, 127]）
    std::vector<int8_t> tanh_int8_lut =
        generate_tanh_int8_lut(scale_g_pre, zp_g_pre, scale_g_out, zp_g_out);

    cudaMemcpyToSymbol(d_sigmoid_int8_z_lut, sigmoid_z_lut.data(),
                       sizeof(uint8_t) * 256);  // 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(d_sigmoid_int8_r_lut, sigmoid_r_lut.data(),
                       sizeof(uint8_t) * 256);  // 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(d_tanh_int8_g_lut, tanh_int8_lut.data(),
                       sizeof(int8_t) * 256);   // 从host端拷贝到device端中编译期固定的地址
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

void generate_int8_lut_from_exp2_inv(int8_t exp2_inv_z_pre, int32_t zp_z_pre, int8_t exp2_inv_z_out,
                                     int32_t zp_z_out, int8_t exp2_inv_r_pre, int32_t zp_r_pre,
                                     int8_t exp2_inv_r_out, int32_t zp_r_out, int8_t exp2_inv_g_pre,
                                     int32_t zp_g_pre, int8_t exp2_inv_g_out, int32_t zp_g_out) {
    // sigmoid LUT 使用 uint8_t（输出 [0, 255]）
    std::vector<uint8_t> sigmoid_z_lut =
        generate_sigmoid_int8_lut_exp2(exp2_inv_z_pre, zp_z_pre, exp2_inv_z_out, zp_z_out);
    std::vector<uint8_t> sigmoid_r_lut =
        generate_sigmoid_int8_lut_exp2(exp2_inv_r_pre, zp_r_pre, exp2_inv_r_out, zp_r_out);
    // tanh LUT 仍使用 int8_t
    std::vector<int8_t> tanh_int8_lut =
        generate_tanh_int8_lut_exp2(exp2_inv_g_pre, zp_g_pre, exp2_inv_g_out, zp_g_out);

    cudaMemcpyToSymbol(d_sigmoid_int8_z_lut, sigmoid_z_lut.data(), sizeof(uint8_t) * 256);
    cudaMemcpyToSymbol(d_sigmoid_int8_r_lut, sigmoid_r_lut.data(), sizeof(uint8_t) * 256);
    cudaMemcpyToSymbol(d_tanh_int8_g_lut, tanh_int8_lut.data(), sizeof(int8_t) * 256);
}

// 生成分段线性量化表（基于exp2_inv参数，支持模板类型）
// exp2_inv 就是 shift_bits（因为 scale = 2^(-exp2_inv) = 2^(-shift_bits)）
template <typename QuantT>
void generate_piecewise_linear_lut_from_exp2_inv(int8_t exp2_inv_z_pre, int32_t zp_z_pre,
                                                 int8_t exp2_inv_z_out, int32_t zp_z_out,
                                                 int8_t exp2_inv_r_pre, int32_t zp_r_pre,
                                                 int8_t exp2_inv_r_out, int32_t zp_r_out,
                                                 int8_t exp2_inv_g_pre, int32_t zp_g_pre,
                                                 int8_t exp2_inv_g_out, int32_t zp_g_out) {
    // 从量化参数计算 min 和 max
    // scale = 2^(-exp2_inv) = 1.0f / (1 << exp2_inv)
    auto calculate_scale = [](int8_t exp2_inv) -> float {
        if (exp2_inv >= 0) {
            return 1.0f / static_cast<float>(1 << exp2_inv);
        } else {
            return static_cast<float>(1 << (-exp2_inv));
        }
    };

    // 🔥 关键修正：C++ 实现中，sigmoid/tanh 的输入是有符号整数（来自 clamp<int8_t/int16_t>）
    // 所以应该使用有符号整数范围：[-128, 127] 或 [-32768, 32767]
    // 注意：这与 Python 参考不同，Python 参考使用无符号整数范围
    int32_t quant_min, quant_max;
    if constexpr (std::is_same_v<QuantT, int8_t>) {
        // 对于 int8_t，输入使用有符号范围 [-128, 127]
        quant_min = -128;
        quant_max = 127;
    } else if constexpr (std::is_same_v<QuantT, int16_t>) {
        // 对于 int16_t，输入使用有符号范围 [-32768, 32767]
        quant_min = -32768;
        quant_max = 32767;
    } else {
        // 默认情况（不应该到达这里）
        quant_min = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
        quant_max = static_cast<int32_t>(std::numeric_limits<QuantT>::max());
    }

    // 计算每个门的输入范围（使用 pre 的量化参数）
    // 公式：x = (q - zp) * scale，其中 q ∈ [quant_min, quant_max]
    float scale_z_pre = calculate_scale(exp2_inv_z_pre);
    float x_min_z = static_cast<float>(quant_min - zp_z_pre) * scale_z_pre;
    float x_max_z = static_cast<float>(quant_max - zp_z_pre) * scale_z_pre;

    float scale_r_pre = calculate_scale(exp2_inv_r_pre);
    float x_min_r = static_cast<float>(quant_min - zp_r_pre) * scale_r_pre;
    float x_max_r = static_cast<float>(quant_max - zp_r_pre) * scale_r_pre;

    float scale_g_pre = calculate_scale(exp2_inv_g_pre);
    float x_min_g = static_cast<float>(quant_min - zp_g_pre) * scale_g_pre;
    float x_max_g = static_cast<float>(quant_max - zp_g_pre) * scale_g_pre;

    // 根据 QuantT 类型选择相应的初始化函数
    if constexpr (std::is_same_v<QuantT, int8_t>) {
        // INT8 版本
        init_sigmoid_z_lut_int8(exp2_inv_z_pre, zp_z_pre, exp2_inv_z_out, zp_z_out, x_min_z,
                                x_max_z);

        init_sigmoid_r_lut_int8(exp2_inv_r_pre, zp_r_pre, exp2_inv_r_out, zp_r_out, x_min_r,
                                x_max_r);

        init_tanh_lut_int8(exp2_inv_g_pre, zp_g_pre, exp2_inv_g_out, zp_g_out, x_min_g, x_max_g);
    } else if constexpr (std::is_same_v<QuantT, int16_t>) {
        // INT16 版本
        init_sigmoid_z_lut_int16(exp2_inv_z_pre, zp_z_pre, exp2_inv_z_out, zp_z_out, x_min_z,
                                 x_max_z);

        init_sigmoid_r_lut_int16(exp2_inv_r_pre, zp_r_pre, exp2_inv_r_out, zp_r_out, x_min_r,
                                 x_max_r);

        init_tanh_lut_int16(exp2_inv_g_pre, zp_g_pre, exp2_inv_g_out, zp_g_out, x_min_g, x_max_g);
    } else {
        static_assert(std::is_same_v<QuantT, int8_t> || std::is_same_v<QuantT, int16_t>,
                      "QuantT must be int8_t or int16_t");
    }
}

namespace kernel {

template <typename T>
__global__ void computeWeightSumMulZP(
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
    //    sum = rshift_round(sum, n[row]);
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
__global__ void dequantificationV(const int32_t *quant_data, T *data, int time_steps, int batch_size,
                                  int hidden_size, int8_t exp2_inv_z, int32_t zp_z,
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

template <typename T>
void computeWeightSumMulzp(
    const T *W_q,         // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,  // [out_dim] 输出数组
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

template void computeWeightSumMulzp<int8_t>(
    const int8_t *W_q,    // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,  // [out_dim] 输出数组
    int x_zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,  // 输出通道数 (M)
    int in_dim,   // 输入通道数 (K)
    cudaStream_t stream);

template void computeWeightSumMulzp<int16_t>(
    const int16_t *W_q,   // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,  // [out_dim] 输出数组
    int x_zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,  // 输出通道数 (M)
    int in_dim,   // 输入通道数 (K)
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

template void dequantificationV<float>(const int32_t *quant_data, float *data,
                                       int time_steps, int batch_size, int hidden_size,
                                       int8_t exp2_inv_z, int32_t zp_z, int8_t exp2_inv_r,
                                       int32_t zp_r, int8_t exp2_inv_g, int32_t zp_g,
                                       int8_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);

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

// ==================== INT16 版本的分段线性量化参数生成函数 ====================
//
// 【生成流程】三遍扫描（与 INT8 版本相同，仅位宽不同）
//   Pass 1: 线性拟合每段 → 浮点系数 (b_fp, c_fp)
//   Pass 2: 统计最大值 → 全局量化参数 (shift_bits_b, shift_bits_c)
//   Pass 3: 量化系数 → (q_b, term_c_precomputed, n_BX_total)
//
// 【最终公式】q_y = (q_b * (q_x - zp_x)) >> n_BX_total + term_c_precomputed
//
// 【与 INT8 的区别】
//   - q_b: int16_t（范围 [-32768, 32767]）
//   - term_c_precomputed: int32_t（INT8 版本为 int16_t）
//   - threshold: int16_t
//
// =========================================================================

/**
 * @brief 生成 Sigmoid 分段线性拟合 LUT（INT16 版本）
 */
SigmoidLUT_INT16 generate_sigmoid_lut_int16(int8_t shift_bits_x, int32_t zp_x,
                                            int8_t shift_bits_y, int32_t zp_y,
                                            float x_min, float x_max) {
    SigmoidLUT_INT16 lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // 1. 生成分段点（自适应分段）
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    // ===== 第一遍扫描：拟合所有分段，收集所有系数 =====
    struct SegmentCoeffs {
        float x_start, x_end;
        float b, c;
    };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        // 生成该段的训练数据
        const int num_samples = 100;
        std::vector<float> x_seg(num_samples);
        std::vector<float> y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));  // Sigmoid
        }

        // 线性拟合: y = b*x + c
        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);

        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    // ===== 第二遍扫描：统一量化系数 =====
    // 计算输出 zero-point 偏移，烘焙到 c 中
    float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
    float zp_y_offset = static_cast<float>(zp_y) * scale_y;

    // 收集所有 b 和调整后的 c
    float b_abs_max = 0.0f;
    float c_abs_max = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        b_abs_max = std::max(b_abs_max, std::abs(all_coeffs[i].b));
        float c_adjusted = all_coeffs[i].c + zp_y_offset;
        c_abs_max = std::max(c_abs_max, std::abs(c_adjusted));
    }

    // 为所有段创建统一的量化参数
    if (b_abs_max < 1e-9f) b_abs_max = 1e-9f;
    if (c_abs_max < 1e-9f) c_abs_max = 1e-9f;

    int8_t shift_bits_b = determine_shift_bits_int16(b_abs_max);
    int8_t shift_bits_c = determine_shift_bits_int16(c_abs_max);

    // ===== 第三遍扫描：量化每段并计算移位 =====
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto& coeff = all_coeffs[i];
        float c_adjusted = coeff.c + zp_y_offset;

        // 使用统一的量化器量化系数
        int16_t q_b = quantize_coefficient_int16(coeff.b, shift_bits_b);
        int16_t q_c = quantize_coefficient_int16(c_adjusted, shift_bits_c);

        // 计算融合移位位数
        // n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y
        // （简化：省略中间 bx 量化步骤，直接融合）
        int8_t n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y;

        // 计算 n_yc
        int8_t n_yc = shift_bits_c - shift_bits_y;

        // 预计算 term_c
        int32_t term_c_precomputed;
        if (n_yc >= 0) {
            term_c_precomputed = static_cast<int32_t>(q_c) >> n_yc;
        } else {
            term_c_precomputed = static_cast<int32_t>(q_c) << (-n_yc);
        }

        // 量化阈值（使用有符号量化 INT16）
        int16_t threshold = quantize_input_int16(coeff.x_end, shift_bits_x, zp_x);

        // 保存段参数
        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

/**
 * @brief 生成 Tanh 分段线性拟合 LUT（INT16 版本）
 * @note Tanh 输出范围 [-1, 1]，设备端返回 int16_t
 */
SigmoidLUT_INT16 generate_tanh_lut_int16(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y,
                                         int32_t zp_y, float x_min, float x_max) {
    SigmoidLUT_INT16 lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // 1. 生成分段点
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    // ===== 第一遍扫描：拟合所有分段，收集所有系数 =====
    struct SegmentCoeffs {
        float x_start, x_end;
        float b, c;
    };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples);
        std::vector<float> y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = std::tanh(x_val);  // Tanh
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);

        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    // ===== 第二遍扫描：统一量化系数 =====
    float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
    float zp_y_offset = static_cast<float>(zp_y) * scale_y;

    float b_abs_max = 0.0f;
    float c_abs_max = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        b_abs_max = std::max(b_abs_max, std::abs(all_coeffs[i].b));
        float c_adjusted = all_coeffs[i].c + zp_y_offset;
        c_abs_max = std::max(c_abs_max, std::abs(c_adjusted));
    }

    if (b_abs_max < 1e-9f) b_abs_max = 1e-9f;
    if (c_abs_max < 1e-9f) c_abs_max = 1e-9f;

    int8_t shift_bits_b = determine_shift_bits_int16(b_abs_max);
    int8_t shift_bits_c = determine_shift_bits_int16(c_abs_max);

    // ===== 第三遍扫描：量化每段并计算移位 =====
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto& coeff = all_coeffs[i];
        float c_adjusted = coeff.c + zp_y_offset;

        int16_t q_b = quantize_coefficient_int16(coeff.b, shift_bits_b);
        int16_t q_c = quantize_coefficient_int16(c_adjusted, shift_bits_c);

        int8_t n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        int32_t term_c_precomputed;
        if (n_yc >= 0) {
            term_c_precomputed = static_cast<int32_t>(q_c) >> n_yc;
        } else {
            term_c_precomputed = static_cast<int32_t>(q_c) << (-n_yc);
        }

        int16_t threshold = quantize_input_int16(coeff.x_end, shift_bits_x, zp_x);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - z 门）
void init_sigmoid_z_lut_int16(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                              float x_min, float x_max) {
    SigmoidLUT_INT16 lut =
        generate_sigmoid_lut_int16(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_sigmoid_z_lut_int16, &lut, sizeof(SigmoidLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid z LUT to constant memory: %s\n", cudaGetErrorString(err));
    }
}

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - r 门）
void init_sigmoid_r_lut_int16(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                              float x_min, float x_max) {
    SigmoidLUT_INT16 lut =
        generate_sigmoid_lut_int16(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_sigmoid_r_lut_int16, &lut, sizeof(SigmoidLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid r LUT to constant memory: %s\n", cudaGetErrorString(err));
    }
}

void init_tanh_lut_int16(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                         float x_min, float x_max) {
    SigmoidLUT_INT16 lut =
        generate_tanh_lut_int16(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_tanh_lut_int16, &lut, sizeof(SigmoidLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy tanh LUT to constant memory: %s\n", cudaGetErrorString(err));
    }
}

// ==================== INT8 版本的分段线性量化参数生成函数 ====================
//
// 【生成流程】三遍扫描
//   Pass 1: 线性拟合每段 → 浮点系数 (b_fp, c_fp)
//   Pass 2: 统计最大值 → 全局量化参数 (shift_bits_b, shift_bits_c)
//   Pass 3: 量化系数 → (q_b, term_c_precomputed, n_BX_total)
//
// 【量化公式推导】
//   浮点:  y_fp = b_fp * x_fp + c_fp
//   
//   量化:  x_fp = (q_x - zp_x) * scale_x     其中 scale_x = 2^(-shift_bits_x)
//          y_fp = (q_y - zp_y) * scale_y     其中 scale_y = 2^(-shift_bits_y)
//          b_fp = q_b * scale_b              其中 scale_b = 2^(-shift_bits_b)
//          c_fp = q_c * scale_c              其中 scale_c = 2^(-shift_bits_c)
//   
//   代入:  (q_y - zp_y) * scale_y = q_b * scale_b * (q_x - zp_x) * scale_x + q_c * scale_c
//   
//   整理:  q_y = q_b * (q_x - zp_x) * (scale_b * scale_x / scale_y) + q_c * (scale_c / scale_y) + zp_y
//             = q_b * (q_x - zp_x) >> (shift_bits_b + shift_bits_x - shift_bits_y)
//               + q_c >> (shift_bits_c - shift_bits_y) + zp_y
//   
//   优化:  将 zp_y 烘焙到 c 中: c_adjusted = c_fp + zp_y * scale_y
//          n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y
//          term_c_precomputed = q_c >> (shift_bits_c - shift_bits_y)
//   
//   最终:  q_y = (q_b * (q_x - zp_x)) >> n_BX_total + term_c_precomputed
//
// =========================================================================

/**
 * @brief 生成 Sigmoid 分段线性拟合 LUT（INT8 版本）
 */
SigmoidLUT_INT8 generate_sigmoid_lut_int8(int8_t shift_bits_x, int32_t zp_x,
                                          int8_t shift_bits_y, int32_t zp_y,
                                          float x_min, float x_max) {
    SigmoidLUT_INT8 lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // ===== Pass 1: 生成分段点 + 线性拟合 =====
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    struct SegmentCoeffs {
        float x_start, x_end;
        float b, c;  // y_fp = b * x_fp + c
    };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        // 采样并拟合: sigmoid(x) = 1 / (1 + exp(-x))
        const int num_samples = 100;
        std::vector<float> x_seg(num_samples);
        std::vector<float> y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);
        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    // ===== Pass 2: 确定全局量化参数 =====
    // 公式: c_adjusted = c_fp + zp_y * scale_y  (将输出零点烘焙到 c)
    float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
    float zp_y_offset = static_cast<float>(zp_y) * scale_y;

    // 统计 |b| 和 |c_adjusted| 的最大值，用于确定 shift_bits
    float b_abs_max = 0.0f, c_abs_max = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        b_abs_max = std::max(b_abs_max, std::abs(all_coeffs[i].b));
        c_abs_max = std::max(c_abs_max, std::abs(all_coeffs[i].c + zp_y_offset));
    }
    if (b_abs_max < 1e-9f) b_abs_max = 1e-9f;
    if (c_abs_max < 1e-9f) c_abs_max = 1e-9f;

    // 公式: scale_b = 2^(-shift_bits_b), 使得 |q_b| <= 127
    int8_t shift_bits_b = determine_shift_bits_int8(b_abs_max);
    int8_t shift_bits_c = determine_shift_bits_int8(c_abs_max);

    // ===== Pass 3: 量化系数并计算预计算项 =====
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto& coeff = all_coeffs[i];

        // 公式: c_adjusted = c_fp + zp_y * scale_y
        float c_adjusted = coeff.c + zp_y_offset;

        // 公式: q_b = round(b_fp / scale_b), q_c = round(c_adjusted / scale_c)
        int8_t q_b = quantize_coefficient_int8(coeff.b, shift_bits_b);
        int16_t q_c = quantize_coefficient_int16(c_adjusted, shift_bits_c);

        // 公式: n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y
        int8_t n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y;

        // 公式: n_yc = shift_bits_c - shift_bits_y
        int8_t n_yc = shift_bits_c - shift_bits_y;

        // 公式: term_c_precomputed = q_c >> n_yc (或 << 如果 n_yc < 0)
        int16_t term_c_precomputed = (n_yc >= 0) ? static_cast<int16_t>(q_c >> n_yc)
                                                 : static_cast<int16_t>(q_c << (-n_yc));
        term_c_precomputed = std::max<int16_t>(-32768, std::min<int16_t>(32767, term_c_precomputed));

        // 公式: threshold = round(x_end / scale_x) + zp_x
        int8_t threshold = quantize_input_int8(coeff.x_end, shift_bits_x, zp_x);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

/**
 * @brief 生成 Tanh 分段线性拟合 LUT（INT8 版本）
 * @note Tanh 输出范围 [-1, 1]，使用有符号输出
 */
SigmoidLUT_INT8 generate_tanh_lut_int8(int8_t shift_bits_x, int32_t zp_x,
                                       int8_t shift_bits_y, int32_t zp_y,
                                       float x_min, float x_max) {
    SigmoidLUT_INT8 lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // 1. 生成分段点
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    // ===== 第一遍扫描：拟合所有分段，收集所有系数 =====
    struct SegmentCoeffs {
        float x_start, x_end;
        float b, c;
    };
    std::vector<SegmentCoeffs> all_coeffs(NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples);
        std::vector<float> y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = std::tanh(x_val);  // Tanh
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);

        all_coeffs[i] = {x_start, x_end, b_fp, c_fp};
    }

    // ===== 第二遍扫描：统一量化系数 =====
    float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
    float zp_y_offset = static_cast<float>(zp_y) * scale_y;

    float b_abs_max = 0.0f;
    float c_abs_max = 0.0f;
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        b_abs_max = std::max(b_abs_max, std::abs(all_coeffs[i].b));
        float c_adjusted = all_coeffs[i].c + zp_y_offset;
        c_abs_max = std::max(c_abs_max, std::abs(c_adjusted));
    }

    if (b_abs_max < 1e-9f) b_abs_max = 1e-9f;
    if (c_abs_max < 1e-9f) c_abs_max = 1e-9f;

    int8_t shift_bits_b = determine_shift_bits_int8(b_abs_max);
    int8_t shift_bits_c = determine_shift_bits_int8(c_abs_max);

    // ===== 第三遍扫描：量化每段并计算移位 =====
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        const auto& coeff = all_coeffs[i];
        float c_adjusted = coeff.c + zp_y_offset;

        int8_t q_b = quantize_coefficient_int8(coeff.b, shift_bits_b);
        int16_t q_c = quantize_coefficient_int16(c_adjusted, shift_bits_c);

        int8_t n_BX_total = shift_bits_b + shift_bits_x - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        int16_t term_c_precomputed;
        if (n_yc >= 0) {
            term_c_precomputed = static_cast<int16_t>(q_c >> n_yc);
        } else {
            term_c_precomputed = static_cast<int16_t>(q_c << (-n_yc));
        }
        term_c_precomputed =
            std::max(static_cast<int16_t>(-32768), std::min(static_cast<int16_t>(32767), term_c_precomputed));

        int8_t threshold = quantize_input_int8(coeff.x_end, shift_bits_x, zp_x);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

// ==================== INT8 版本的 LUT 初始化函数 ====================
// 生成 LUT 并复制到 CUDA 常量内存

/// @brief 初始化 z 门的 Sigmoid LUT（INT8 版本）
void init_sigmoid_z_lut_int8(int8_t shift_bits_x, int32_t zp_x,
                             int8_t shift_bits_y, int32_t zp_y,
                             float x_min, float x_max) {
    SigmoidLUT_INT8 lut = generate_sigmoid_lut_int8(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);
    cudaError_t err = cudaMemcpyToSymbol(d_sigmoid_z_lut_int8, &lut, sizeof(SigmoidLUT_INT8));
    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid z LUT (INT8) to constant memory: %s\n", cudaGetErrorString(err));
    }
}

/// @brief 初始化 r 门的 Sigmoid LUT（INT8 版本）
void init_sigmoid_r_lut_int8(int8_t shift_bits_x, int32_t zp_x,
                             int8_t shift_bits_y, int32_t zp_y,
                             float x_min, float x_max) {
    SigmoidLUT_INT8 lut = generate_sigmoid_lut_int8(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);
    cudaError_t err = cudaMemcpyToSymbol(d_sigmoid_r_lut_int8, &lut, sizeof(SigmoidLUT_INT8));
    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid r LUT (INT8) to constant memory: %s\n", cudaGetErrorString(err));
    }
}

/// @brief 初始化 g 门的 Tanh LUT（INT8 版本）
void init_tanh_lut_int8(int8_t shift_bits_x, int32_t zp_x,
                        int8_t shift_bits_y, int32_t zp_y,
                        float x_min, float x_max) {
    SigmoidLUT_INT8 lut = generate_tanh_lut_int8(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);
    cudaError_t err = cudaMemcpyToSymbol(d_tanh_lut_int8, &lut, sizeof(SigmoidLUT_INT8));
    if (err != cudaSuccess) {
        printf("Failed to copy tanh LUT (INT8) to constant memory: %s\n", cudaGetErrorString(err));
    }
}

// 显式实例化 generate_piecewise_linear_lut_from_exp2_inv 模板函数
template void generate_piecewise_linear_lut_from_exp2_inv<int8_t>(
    int8_t exp2_inv_z_pre, int32_t zp_z_pre, int8_t exp2_inv_z_out, int32_t zp_z_out,
    int8_t exp2_inv_r_pre, int32_t zp_r_pre, int8_t exp2_inv_r_out, int32_t zp_r_out,
    int8_t exp2_inv_g_pre, int32_t zp_g_pre, int8_t exp2_inv_g_out, int32_t zp_g_out);

template void generate_piecewise_linear_lut_from_exp2_inv<int16_t>(
    int8_t exp2_inv_z_pre, int32_t zp_z_pre, int8_t exp2_inv_z_out, int32_t zp_z_out,
    int8_t exp2_inv_r_pre, int32_t zp_r_pre, int8_t exp2_inv_r_out, int32_t zp_r_out,
    int8_t exp2_inv_g_pre, int32_t zp_g_pre, int8_t exp2_inv_g_out, int32_t zp_g_out);
