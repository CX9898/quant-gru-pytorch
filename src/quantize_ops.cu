#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <limits>

#include "devVector.h"
#include "quantize_ops.cuh"
#include "quantize_ops_helper.hpp"

__constant__ int8_t d_sigmoid_int8_z_lut[256];
__constant__ int8_t d_sigmoid_int8_r_lut[256];
__constant__ int8_t d_tanh_int8_g_lut[256];

std::vector<int8_t> generate_sigmoid_int8_lut(float scale_z_pre, int zp_z_pre,
                                              float scale_z, int zp_z) {
    std::vector<int8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;

        const float x_fp = static_cast<float>(x_i8 - zp_z_pre) * scale_z_pre;
        const float y_fp = 1.f / (1.f + std::exp(-x_fp));

        int y_i8 = static_cast<int>(std::round(y_fp / scale_z + zp_z));
        if (y_i8 < -128) y_i8 = -128;
        if (y_i8 > 127) y_i8 = 127;

        lut[i] = static_cast<int8_t>(y_i8);
    }
    return lut;
}

std::vector<int8_t> generate_tanh_int8_lut(float scale_pre, int zp_pre,
                                           float scale_out, int zp_out) {
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

void generate_int8_lut(float scale_z_pre, int32_t zp_z_pre, float scale_z_out,
                       int32_t zp_z_out, float scale_r_pre, int32_t zp_r_pre,
                       float scale_r_out, int32_t zp_r_out, float scale_g_pre,
                       int32_t zp_g_pre, float scale_g_out, int32_t zp_g_out) {
    std::vector<int8_t> sigmoid_z_lut =
        generate_sigmoid_int8_lut(scale_z_pre, zp_z_pre, scale_z_out, zp_z_out);
    //    printf("scale_z_pre = %.15f, zp_z_pre = %d, scale_z_out = %.15f,
    //    zp_z_out = %d\n",
    //           scale_z_pre,
    //           zp_z_pre,
    //           scale_z_out,
    //           zp_z_out);
    std::vector<int8_t> sigmoid_r_lut =
        generate_sigmoid_int8_lut(scale_r_pre, zp_r_pre, scale_r_out, zp_r_out);
    //    printf("scale_r_pre = %.15f, zp_r_pre = %d, scale_r_out = %.15f,
    //    zp_r_out = %d\n",
    //           scale_r_pre,
    //           zp_r_pre,
    //           scale_r_out,
    //           zp_r_out);
    std::vector<int8_t> tanh_int8_lut =
        generate_tanh_int8_lut(scale_g_pre, zp_g_pre, scale_g_out, zp_g_out);
    //    printf("scale_g_pre = %.15f, zp_g_pre = %d, scale_g_out = %.15f,
    //    zp_g_out = %d\n",
    //           scale_g_pre,
    //           zp_g_pre,
    //           scale_g_out,
    //           zp_g_out);

    cudaMemcpyToSymbol(
        d_sigmoid_int8_z_lut, sigmoid_z_lut.data(),
        sizeof(int8_t) * 256);// 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(
        d_sigmoid_int8_r_lut, sigmoid_r_lut.data(),
        sizeof(int8_t) * 256);// 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(
        d_tanh_int8_g_lut, tanh_int8_lut.data(),
        sizeof(int8_t) * 256);// 从host端拷贝到device端中编译期固定的地址
}

std::vector<int8_t> generate_sigmoid_int8_lut_exp2(int32_t exp2_inv_z_pre,
                                                   int zp_z_pre,
                                                   int32_t exp2_inv_z,
                                                   int zp_z) {
    std::vector<int8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;

        // （1）反量化 x
        float x_fp = dequantize(x_i8, exp2_inv_z_pre, zp_z_pre);

        // （2）计算 sigmoid
        float y_fp = 1.f / (1.f + std::exp(-x_fp));

        // （3）量化 y
        int y_i8 = quantize<int8_t>(y_fp, exp2_inv_z, zp_z);

        lut[i] = static_cast<int8_t>(y_i8);
    }

    return lut;
}

std::vector<int8_t> generate_tanh_int8_lut_exp2(int32_t exp2_inv_pre,
                                                int zp_pre,
                                                int32_t exp2_inv_out,
                                                int zp_out) {
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

void generate_int8_lut_from_exp2_inv(int32_t exp2_inv_z_pre, int32_t zp_z_pre,
                                     int32_t exp2_inv_z_out, int32_t zp_z_out,
                                     int32_t exp2_inv_r_pre, int32_t zp_r_pre,
                                     int32_t exp2_inv_r_out, int32_t zp_r_out,
                                     int32_t exp2_inv_g_pre, int32_t zp_g_pre,
                                     int32_t exp2_inv_g_out, int32_t zp_g_out) {
    std::vector<int8_t> sigmoid_z_lut = generate_sigmoid_int8_lut_exp2(
        exp2_inv_z_pre, zp_z_pre, exp2_inv_z_out, zp_z_out);
    std::vector<int8_t> sigmoid_r_lut = generate_sigmoid_int8_lut_exp2(
        exp2_inv_r_pre, zp_r_pre, exp2_inv_r_out, zp_r_out);
    std::vector<int8_t> tanh_int8_lut = generate_tanh_int8_lut_exp2(
        exp2_inv_g_pre, zp_g_pre, exp2_inv_g_out, zp_g_out);

    cudaMemcpyToSymbol(d_sigmoid_int8_z_lut, sigmoid_z_lut.data(),
                       sizeof(int8_t) * 256);
    cudaMemcpyToSymbol(d_sigmoid_int8_r_lut, sigmoid_r_lut.data(),
                       sizeof(int8_t) * 256);
    cudaMemcpyToSymbol(d_tanh_int8_g_lut, tanh_int8_lut.data(),
                       sizeof(int8_t) * 256);
}

namespace kernel {

template<typename T>
__global__ void computeWeightSumMulZP(
    const T *__restrict__ W_q,       // [out_dim, in_dim] 权重量化矩阵, 列主序储存
    int32_t *__restrict__ weight_sum,// [out_dim] 输出数组
    int x_zp,
    const int32_t *__restrict__ n,// n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,// 输出通道数 (M)
    int in_dim  // 输入通道数 (K)
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

template<typename T, typename QuantT>
__global__ void quantification(const T *data, QuantT *quant_data, size_t size,
                               int32_t exp2_inv, int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    quant_data[idx] = dev::quantize<QuantT>(data[idx], exp2_inv, zp);
}

template<typename T, typename QuantT>
__global__ void dequantification(const QuantT *quant_data, T *data, size_t size,
                                 int32_t exp2_inv, int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    data[idx] = dequantize<QuantT>(quant_data[idx], exp2_inv, zp);
}

}// namespace kernel

namespace kernel {

template<typename T, typename QuantT>
__global__ void dequantificationV(const QuantT *quant_data, T *data,
                                  int time_steps, int batch_size, int hidden_size,
                                  int32_t exp2_inv_z, int32_t zp_z,
                                  int32_t exp2_inv_r, int32_t zp_r,
                                  int32_t exp2_inv_g, int32_t zp_g,
                                  int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br) {
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

    // 反量化 z_out (第0部分)
    const int z_idx = base_idx + 0 * hidden_size + h;
    data[z_idx] = dequantize<QuantT>(quant_data[z_idx], exp2_inv_z, zp_z);

    // 反量化 r_out (第1部分)
    const int r_idx = base_idx + 1 * hidden_size + h;
    data[r_idx] = dequantize<QuantT>(quant_data[r_idx], exp2_inv_r, zp_r);

    // 反量化 g_out (第2部分，对称量化，zp=0)
    const int g_idx = base_idx + 2 * hidden_size + h;
    data[g_idx] = dequantize<QuantT>(quant_data[g_idx], exp2_inv_g, zp_g);

    // 反量化 Rh_add_br_g (第3部分)
    const int rh_idx = base_idx + 3 * hidden_size + h;
    data[rh_idx] = dequantize<QuantT>(quant_data[rh_idx], exp2_inv_Rh_add_br, zp_Rh_add_br);
}

template<typename T, typename QuantT>
__global__ void quantificationPerChannel(const T *src, QuantT *quant_data,
                                         size_t input_size, size_t channel_size,
                                         const int32_t *exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const int32_t exp2_inv = exp2_invs[channel_idx];

    const size_t idx = input_idx * channel_size + channel_idx;
    quant_data[idx] = dev::quantize<QuantT>(src[idx], exp2_inv, 0);
}

template<typename T, typename QuantT>
__global__ void dequantificationPerChannel(const QuantT *quant_data, T *data,
                                           size_t input_size, size_t channel_size,
                                           const int32_t *exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const int32_t exp2_inv = exp2_invs[channel_idx];

    const size_t idx = input_idx * channel_size + channel_idx;
    data[idx] = dequantize<QuantT>(quant_data[idx], exp2_inv, 0);
}

}// namespace kernel

template<typename T>
void computeWeightSumMulzp(
    const T *W_q,       // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int x_zp,
    const int32_t *__restrict__ n,// n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,// 输出通道数 (M)
    int in_dim, // 输入通道数 (K)
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;
    kernel::computeWeightSumMulZP<<<blocks, threads, 0, stream>>>(
        W_q, weight_sum, x_zp, n, out_dim, in_dim);
}

template void computeWeightSumMulzp<int8_t>(
    const int8_t *W_q,  // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int x_zp,
    const int32_t *__restrict__ n,// n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,// 输出通道数 (M)
    int in_dim, // 输入通道数 (K)
    cudaStream_t stream);

template void computeWeightSumMulzp<int16_t>(
    const int16_t *W_q, // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int x_zp,
    const int32_t *__restrict__ n,// n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,// 输出通道数 (M)
    int in_dim, // 输入通道数 (K)
    cudaStream_t stream);

namespace dev {

template<typename T, typename QuantT>
void quantification(const T *data, QuantT *quant_data, size_t size,
                    int32_t exp2_inv, int32_t zp) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::quantification<<<grid, block>>>(data, quant_data, size, exp2_inv, zp);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();
}

template void quantification<float, int8_t>(const float *data, int8_t *quant_data, size_t size,
                                            int32_t exp2_inv, int32_t zp);
template void quantification<float, int16_t>(const float *data, int16_t *quant_data, size_t size,
                                             int32_t exp2_inv, int32_t zp);
template void quantification<float, int32_t>(const float *data, int32_t *quant_data, size_t size,
                                             int32_t exp2_inv, int32_t zp);

template<typename T, typename QuantT>
void dequantification(const QuantT *quant_data, T *data, size_t size,
                      int32_t exp2_inv, int32_t zp) {
    size_t block = 256;
    size_t grid = (size + block - 1) / block;
    kernel::dequantification<<<grid, block>>>(quant_data, data, size, exp2_inv, zp);
    cudaDeviceSynchronize();
}

template void dequantification<float, int8_t>(const int8_t *quant_data, float *data, size_t size,
                                              int32_t exp2_inv, int32_t zp);
template void dequantification<float, int16_t>(const int16_t *quant_data, float *data, size_t size,
                                               int32_t exp2_inv, int32_t zp);
template void dequantification<float, int32_t>(const int32_t *quant_data, float *data, size_t size,
                                               int32_t exp2_inv, int32_t zp);

template<typename T, typename QuantT>
void dequantificationV(const QuantT *quant_data, T *data,
                       int time_steps, int batch_size, int hidden_size,
                       int32_t exp2_inv_z, int32_t zp_z,
                       int32_t exp2_inv_r, int32_t zp_r,
                       int32_t exp2_inv_g, int32_t zp_g,
                       int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br) {
    // Launch configuration: 每个block处理一个时间步和一个batch的所有hidden单元
    // blockDim.x = hidden_size (每个线程处理一个hidden单元)
    // gridDim.x = time_steps
    // gridDim.y = batch_size
    const dim3 blockDim(hidden_size);
    const dim3 gridDim(time_steps, batch_size);

    kernel::dequantificationV<<<gridDim, blockDim>>>(
        quant_data, data, time_steps, batch_size, hidden_size,
        exp2_inv_z, zp_z, exp2_inv_r, zp_r, exp2_inv_g, zp_g,
        exp2_inv_Rh_add_br, zp_Rh_add_br);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("dequantificationV kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template void dequantificationV<float, int8_t>(const int8_t *quant_data, float *data,
                                               int time_steps, int batch_size, int hidden_size,
                                               int32_t exp2_inv_z, int32_t zp_z,
                                               int32_t exp2_inv_r, int32_t zp_r,
                                               int32_t exp2_inv_g, int32_t zp_g,
                                               int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);
template void dequantificationV<float, int16_t>(const int16_t *quant_data, float *data,
                                                int time_steps, int batch_size, int hidden_size,
                                                int32_t exp2_inv_z, int32_t zp_z,
                                                int32_t exp2_inv_r, int32_t zp_r,
                                                int32_t exp2_inv_g, int32_t zp_g,
                                                int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);


template<typename T, typename QuantT>
void quantificationPerChannel(const T *src, QuantT *quant_data,
                              size_t input_size, size_t channel_size,
                              const dev::vector<int32_t> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::quantificationPerChannel<<<gridDim, blockDim>>>(
        src, quant_data, input_size, channel_size, exp2_invs.data());
    cudaDeviceSynchronize();
}

template void quantificationPerChannel<float, int8_t>(const float *src, int8_t *quant_data,
                                                      size_t input_size, size_t channel_size,
                                                      const dev::vector<int32_t> &exp2_invs);

template void quantificationPerChannel<float, int16_t>(const float *src, int16_t *quant_data,
                                                       size_t input_size, size_t channel_size,
                                                       const dev::vector<int32_t> &exp2_invs);
template void quantificationPerChannel<float, int32_t>(const float *src, int32_t *quant_data,
                                                       size_t input_size, size_t channel_size,
                                                       const dev::vector<int32_t> &exp2_invs);

template<typename T, typename QuantT>
void dequantificationPerChannel(const QuantT *quant_data, T *data,
                                size_t input_size, size_t channel_size,
                                const dev::vector<int32_t> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::dequantificationPerChannel<<<gridDim, blockDim>>>(
        quant_data, data, input_size, channel_size, exp2_invs.data());
    cudaDeviceSynchronize();
}

template void dequantificationPerChannel<float, int8_t>(const int8_t *quant_data, float *data,
                                                        size_t input_size, size_t channel_size,
                                                        const dev::vector<int32_t> &exp2_invs);
template void dequantificationPerChannel<float, int16_t>(const int16_t *quant_data, float *data,
                                                         size_t input_size, size_t channel_size,
                                                         const dev::vector<int32_t> &exp2_invs);
template void dequantificationPerChannel<float, int32_t>(const int32_t *quant_data, float *data,
                                                         size_t input_size, size_t channel_size,
                                                         const dev::vector<int32_t> &exp2_invs);
}// namespace dev
