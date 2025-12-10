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

__constant__ int8_t d_sigmoid_int8_z_lut[256];
__constant__ int8_t d_sigmoid_int8_r_lut[256];
__constant__ int8_t d_tanh_int8_g_lut[256];

// 分段线性量化常量内存
__constant__ SigmoidLUT_INT16 d_sigmoid_z_lut_int16;// z 门的 Sigmoid LUT
__constant__ SigmoidLUT_INT16 d_sigmoid_r_lut_int16;// r 门的 Sigmoid LUT
__constant__ SigmoidLUT_INT16 d_tanh_lut_int16;
__constant__ SigmoidLUT_INT8 d_sigmoid_z_lut_int8;// z 门的 Sigmoid LUT
__constant__ SigmoidLUT_INT8 d_sigmoid_r_lut_int8;// r 门的 Sigmoid LUT
__constant__ SigmoidLUT_INT8 d_tanh_lut_int8;

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

// 生成分段线性量化表（基于exp2_inv参数，支持模板类型）
// exp2_inv 就是 shift_bits（因为 scale = 2^(-exp2_inv) = 2^(-shift_bits)）
template<typename QuantT>
void generate_piecewise_linear_lut_from_exp2_inv(int32_t exp2_inv_z_pre,
                                                 int32_t zp_z_pre,
                                                 int32_t exp2_inv_z_out,
                                                 int32_t zp_z_out,
                                                 int32_t exp2_inv_r_pre,
                                                 int32_t zp_r_pre,
                                                 int32_t exp2_inv_r_out,
                                                 int32_t zp_r_out,
                                                 int32_t exp2_inv_g_pre,
                                                 int32_t zp_g_pre,
                                                 int32_t exp2_inv_g_out,
                                                 int32_t zp_g_out) {
    // 从量化参数计算 min 和 max
    // scale = 2^(-exp2_inv) = 1.0f / (1 << exp2_inv)
    auto calculate_scale = [](int32_t exp2_inv) -> float {
        if (exp2_inv >= 0) {
            return 1.0f / static_cast<float>(1 << exp2_inv);
        } else {
            return static_cast<float>(1 << (-exp2_inv));
        }
    };

    // 🔥 关键修正：根据 Python 参考（u8.py, u16.py），非对称量化使用无符号整数范围
    // 输入和输出使用无符号量化：[0, 2^bit_width - 1]
    // 即使 QuantT 是 int8_t/int16_t，在计算输入/输出范围时也应使用对应的无符号范围
    int32_t quant_min, quant_max;
    if constexpr (std::is_same_v<QuantT, int8_t>) {
        // 对于 int8_t，输入/输出使用 uint8_t 范围 [0, 255]
        quant_min = 0;
        quant_max = 255;
    } else if constexpr (std::is_same_v<QuantT, int16_t>) {
        // 对于 int16_t，输入/输出使用 uint16_t 范围 [0, 65535]
        quant_min = 0;
        quant_max = 65535;
    } else {
        // 默认情况（不应该到达这里）
        quant_min = 0;
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

    // 将 exp2_inv 转换为 shift_bits（它们实际上是相同的）
    // shift_bits 始终是 int8_t 类型
    int8_t shift_bits_z_pre = static_cast<int8_t>(std::max(0, std::min(127, static_cast<int>(exp2_inv_z_pre))));
    int8_t shift_bits_z_out = static_cast<int8_t>(std::max(0, std::min(127, static_cast<int>(exp2_inv_z_out))));
    int8_t shift_bits_r_pre = static_cast<int8_t>(std::max(0, std::min(127, static_cast<int>(exp2_inv_r_pre))));
    int8_t shift_bits_r_out = static_cast<int8_t>(std::max(0, std::min(127, static_cast<int>(exp2_inv_r_out))));
    int8_t shift_bits_g_pre = static_cast<int8_t>(std::max(0, std::min(127, static_cast<int>(exp2_inv_g_pre))));
    int8_t shift_bits_g_out = static_cast<int8_t>(std::max(0, std::min(127, static_cast<int>(exp2_inv_g_out))));

    // 根据 QuantT 类型选择相应的 zp 类型和初始化函数
    if constexpr (std::is_same_v<QuantT, int8_t>) {
        // INT8 版本
        int8_t zp_z_pre_quant = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(zp_z_pre))));
        int8_t zp_z_out_quant = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(zp_z_out))));
        int8_t zp_r_pre_quant = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(zp_r_pre))));
        int8_t zp_r_out_quant = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(zp_r_out))));
        int8_t zp_g_pre_quant = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(zp_g_pre))));
        int8_t zp_g_out_quant = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(zp_g_out))));

        init_sigmoid_z_lut_int8(shift_bits_z_pre, zp_z_pre_quant,
                                shift_bits_z_out, zp_z_out_quant,
                                x_min_z, x_max_z);

        init_sigmoid_r_lut_int8(shift_bits_r_pre, zp_r_pre_quant,
                                shift_bits_r_out, zp_r_out_quant,
                                x_min_r, x_max_r);

        init_tanh_lut_int8(shift_bits_g_pre, zp_g_pre_quant,
                           shift_bits_g_out, zp_g_out_quant,
                           x_min_g, x_max_g);
    } else if constexpr (std::is_same_v<QuantT, int16_t>) {
        // INT16 版本
        int16_t zp_z_pre_quant = static_cast<int16_t>(std::max(-32768, std::min(32767, static_cast<int>(zp_z_pre))));
        int16_t zp_z_out_quant = static_cast<int16_t>(std::max(-32768, std::min(32767, static_cast<int>(zp_z_out))));
        int16_t zp_r_pre_quant = static_cast<int16_t>(std::max(-32768, std::min(32767, static_cast<int>(zp_r_pre))));
        int16_t zp_r_out_quant = static_cast<int16_t>(std::max(-32768, std::min(32767, static_cast<int>(zp_r_out))));
        int16_t zp_g_pre_quant = static_cast<int16_t>(std::max(-32768, std::min(32767, static_cast<int>(zp_g_pre))));
        int16_t zp_g_out_quant = static_cast<int16_t>(std::max(-32768, std::min(32767, static_cast<int>(zp_g_out))));

        init_sigmoid_z_lut_int16(shift_bits_z_pre, zp_z_pre_quant,
                                 shift_bits_z_out, zp_z_out_quant,
                                 x_min_z, x_max_z);

        init_sigmoid_r_lut_int16(shift_bits_r_pre, zp_r_pre_quant,
                                 shift_bits_r_out, zp_r_out_quant,
                                 x_min_r, x_max_r);

        init_tanh_lut_int16(shift_bits_g_pre, zp_g_pre_quant,
                            shift_bits_g_out, zp_g_out_quant,
                            x_min_g, x_max_g);
    } else {
        static_assert(std::is_same_v<QuantT, int8_t> || std::is_same_v<QuantT, int16_t>,
                      "QuantT must be int8_t or int16_t");
    }
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
__global__ void quantificationV(const T *data, QuantT *quant_data,
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

    // 量化 z_out (第0部分)
    const int z_idx = base_idx + 0 * hidden_size + h;
    quant_data[z_idx] = dev::quantize<QuantT>(data[z_idx], exp2_inv_z, zp_z);

    // 量化 r_out (第1部分)
    const int r_idx = base_idx + 1 * hidden_size + h;
    quant_data[r_idx] = dev::quantize<QuantT>(data[r_idx], exp2_inv_r, zp_r);

    // 量化 g_out (第2部分，对称量化，zp=0)
    const int g_idx = base_idx + 2 * hidden_size + h;
    quant_data[g_idx] = dev::quantize<QuantT>(data[g_idx], exp2_inv_g, zp_g);

    // 量化 Rh_add_br_g (第3部分)
    const int rh_idx = base_idx + 3 * hidden_size + h;
    quant_data[rh_idx] = dev::quantize<QuantT>(data[rh_idx], exp2_inv_Rh_add_br, zp_Rh_add_br);
}

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
void quantificationV(const T *data, QuantT *quant_data,
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

    kernel::quantificationV<<<gridDim, blockDim>>>(
        data, quant_data, time_steps, batch_size, hidden_size,
        exp2_inv_z, zp_z, exp2_inv_r, zp_r, exp2_inv_g, zp_g,
        exp2_inv_Rh_add_br, zp_Rh_add_br);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("quantificationV kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template void quantificationV<float, int8_t>(const float *data, int8_t *quant_data,
                                             int time_steps, int batch_size, int hidden_size,
                                             int32_t exp2_inv_z, int32_t zp_z,
                                             int32_t exp2_inv_r, int32_t zp_r,
                                             int32_t exp2_inv_g, int32_t zp_g,
                                             int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);
template void quantificationV<float, int16_t>(const float *data, int16_t *quant_data,
                                              int time_steps, int batch_size, int hidden_size,
                                              int32_t exp2_inv_z, int32_t zp_z,
                                              int32_t exp2_inv_r, int32_t zp_r,
                                              int32_t exp2_inv_g, int32_t zp_g,
                                              int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);

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

// ==================== 分段线性量化参数生成函数 ====================

// 线性拟合函数（最小二乘法）
inline void linear_fit(const std::vector<float> &x, const std::vector<float> &y,
                       float &b, float &c) {
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

// 自适应分段（Sigmoid 专用）
std::vector<float> adaptive_segmentation_sigmoid(float x_min, float x_max, int num_segments) {
    std::vector<float> segment_points(num_segments + 1);
    segment_points[0] = x_min;
    segment_points[num_segments] = x_max;

    // 在中心区域（x ≈ 0）密集分段
    float center_range = 2.0f;     // 中心区域范围 [-2, 2]
    int n_dense = num_segments / 2;// 一半段用于中心区域
    int n_sparse = num_segments - n_dense;

    // 稀疏分段（远离中心）
    if (x_min < -center_range) {
        float sparse_range = -center_range - x_min;
        for (int i = 1; i <= n_sparse; i++) {
            float ratio = static_cast<float>(i) / (n_sparse + 1);
            segment_points[i] = x_min + sparse_range * ratio;
        }
    }

    // 密集分段（中心区域）
    float dense_start = std::max(x_min, -center_range);
    float dense_end = std::min(x_max, center_range);
    float dense_range = dense_end - dense_start;
    for (int i = 0; i < n_dense; i++) {
        float ratio = static_cast<float>(i + 1) / (n_dense + 1);
        segment_points[n_sparse + i] = dense_start + dense_range * ratio;
    }

    // 稀疏分段（远离中心，右侧）
    if (x_max > center_range) {
        float sparse_range = x_max - center_range;
        for (int i = 0; i < n_sparse; i++) {
            float ratio = static_cast<float>(i + 1) / (n_sparse + 1);
            segment_points[n_sparse + n_dense + i] = center_range + sparse_range * ratio;
        }
    }

    // 排序确保单调递增
    std::sort(segment_points.begin(), segment_points.end());

    return segment_points;
}

// 生成 Sigmoid 分段线性拟合 LUT（主机端）
SigmoidLUT_INT16 generate_sigmoid_lut_int16(
    int8_t shift_bits_x,// 输入 shift_bits
    int16_t zp_x,       // 输入 zero-point
    int8_t shift_bits_y,// 输出 shift_bits
    int16_t zp_y,       // 输出 zero-point
    float x_min,        // 输入范围最小值
    float x_max         // 输入范围最大值
) {
    SigmoidLUT_INT16 lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // 1. 生成分段点（自适应分段）
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    // 2. 对每段进行线性拟合
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
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));// Sigmoid
        }

        // 线性拟合: y = b*x + c
        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);

        // 3. 量化系数 b（对称量化，zero-point=0）
        int8_t shift_bits_b = determine_shift_bits_int16(std::abs(b_fp));
        int16_t q_b = quantize_coefficient_int16(b_fp, shift_bits_b);

        // 4. 量化系数 c（需要烘焙 zero-point）
        // c_adjusted = c + zp_y * scale_y
        float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
        float c_adjusted = c_fp + static_cast<float>(zp_y) * scale_y;

        int8_t shift_bits_c = determine_shift_bits_int16(std::abs(c_adjusted));
        int16_t q_c = quantize_coefficient_int16(c_adjusted, shift_bits_c);

        // 5. 计算 shift_bits_bx（根据 bx 的实际范围）
        // bx = b * x，需要计算该段内 bx 的范围
        float scale_x = std::pow(2.0f, -static_cast<float>(shift_bits_x));

        // 计算该段内 x_offset 的范围（去零点后的范围）
        // x_offset = q_x - zp_x，对应的浮点范围是 x_start 到 x_end
        // 但实际计算时，x_offset 的范围需要考虑量化后的值
        // 简化：直接使用浮点范围计算 bx 的范围
        float bx_at_start = b_fp * x_start;
        float bx_at_end = b_fp * x_end;
        float bx_min = std::min(bx_at_start, bx_at_end);
        float bx_max = std::max(bx_at_start, bx_at_end);

        // 根据 bx 的范围确定 shift_bits_bx
        // 使用对称量化（因为 bx 可能跨越0）
        float bx_abs_max = std::max(std::abs(bx_min), std::abs(bx_max));
        if (bx_abs_max < 1e-9f) {
            bx_abs_max = 1e-9f;// 避免除零
        }

        // 计算 shift_bits_bx：使 scale_bx = 2^(-shift_bits_bx) 能够覆盖 bx 的范围
        // 🔥 修正：根据 Python 参考（u16.py），bx 使用非对称量化（无符号），范围 [0, 65535]
        // scale_bx >= bx_range / 65535 (UINT16 最大值)
        const float max_uint16 = 65535.0f;
        float bx_range = bx_max - bx_min;// bx 的实际范围（可能包含负值，通过 zero-point 处理）
        if (bx_range < 1e-9f) {
            bx_range = 1e-9f;// 避免除零
        }
        float raw_scale_bx = bx_range / max_uint16;
        int8_t shift_bits_bx = static_cast<int8_t>(std::ceil(-std::log2(raw_scale_bx)));
        shift_bits_bx = std::max(static_cast<int8_t>(0), shift_bits_bx);// 确保非负

        // 6. 计算移位位数（根据文档公式）
        int8_t n_bx = shift_bits_b + shift_bits_x - shift_bits_bx;
        int8_t n_yb = shift_bits_bx - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        // 融合移位
        int8_t n_BX_total = n_bx + n_yb;

        // 7. 预计算 term_c
        int32_t term_c_precomputed;
        if (n_yc >= 0) {
            term_c_precomputed = static_cast<int32_t>(q_c) >> n_yc;
        } else {
            term_c_precomputed = static_cast<int32_t>(q_c) << (-n_yc);
        }

        // 8. 量化阈值
        uint16_t threshold = quantize_input_uint16(x_end, shift_bits_x, zp_x);

        // 保存段参数
        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

// 生成 Tanh 分段线性拟合 LUT（主机端）
SigmoidLUT_INT16 generate_tanh_lut_int16(
    int8_t shift_bits_x,
    int16_t zp_x,
    int8_t shift_bits_y,
    int16_t zp_y,
    float x_min,
    float x_max) {
    SigmoidLUT_INT16 lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // 与 sigmoid 类似的实现，但使用 tanh 函数
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples);
        std::vector<float> y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = std::tanh(x_val);// Tanh
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);

        int8_t shift_bits_b = determine_shift_bits_int16(std::abs(b_fp));
        int16_t q_b = quantize_coefficient_int16(b_fp, shift_bits_b);

        float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
        float c_adjusted = c_fp + static_cast<float>(zp_y) * scale_y;

        int8_t shift_bits_c = determine_shift_bits_int16(std::abs(c_adjusted));
        int16_t q_c = quantize_coefficient_int16(c_adjusted, shift_bits_c);

        // 计算 shift_bits_bx（根据 bx 的实际范围）
        float bx_at_start = b_fp * x_start;
        float bx_at_end = b_fp * x_end;
        float bx_min = std::min(bx_at_start, bx_at_end);
        float bx_max = std::max(bx_at_start, bx_at_end);

        // 🔥 修正：根据 Python 参考（u16.py），bx 使用非对称量化（无符号），范围 [0, 65535]
        // scale_bx >= bx_range / 65535 (UINT16 最大值)
        const float max_uint16 = 65535.0f;
        float bx_range = bx_max - bx_min;// bx 的实际范围（可能包含负值，通过 zero-point 处理）
        if (bx_range < 1e-9f) {
            bx_range = 1e-9f;// 避免除零
        }
        float raw_scale_bx = bx_range / max_uint16;
        int8_t shift_bits_bx = static_cast<int8_t>(std::ceil(-std::log2(raw_scale_bx)));
        shift_bits_bx = std::max(static_cast<int8_t>(0), shift_bits_bx);

        int8_t n_bx = shift_bits_b + shift_bits_x - shift_bits_bx;
        int8_t n_yb = shift_bits_bx - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        int8_t n_BX_total = n_bx + n_yb;

        int32_t term_c_precomputed;
        if (n_yc >= 0) {
            term_c_precomputed = static_cast<int32_t>(q_c) >> n_yc;
        } else {
            term_c_precomputed = static_cast<int32_t>(q_c) << (-n_yc);
        }

        uint16_t threshold = quantize_input_uint16(x_end, shift_bits_x, zp_x);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - z 门）
void init_sigmoid_z_lut_int16(
    int8_t shift_bits_x,
    int16_t zp_x,
    int8_t shift_bits_y,
    int16_t zp_y,
    float x_min,
    float x_max) {
    SigmoidLUT_INT16 lut = generate_sigmoid_lut_int16(
        shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(
        d_sigmoid_z_lut_int16, &lut, sizeof(SigmoidLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid z LUT to constant memory: %s\n",
               cudaGetErrorString(err));
    }
}

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - r 门）
void init_sigmoid_r_lut_int16(
    int8_t shift_bits_x,
    int16_t zp_x,
    int8_t shift_bits_y,
    int16_t zp_y,
    float x_min,
    float x_max) {
    SigmoidLUT_INT16 lut = generate_sigmoid_lut_int16(
        shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(
        d_sigmoid_r_lut_int16, &lut, sizeof(SigmoidLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid r LUT to constant memory: %s\n",
               cudaGetErrorString(err));
    }
}

void init_tanh_lut_int16(
    int8_t shift_bits_x,
    int16_t zp_x,
    int8_t shift_bits_y,
    int16_t zp_y,
    float x_min,
    float x_max) {
    SigmoidLUT_INT16 lut = generate_tanh_lut_int16(
        shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(
        d_tanh_lut_int16, &lut, sizeof(SigmoidLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy tanh LUT to constant memory: %s\n",
               cudaGetErrorString(err));
    }
}

// ==================== INT8 版本的分段线性量化参数生成函数 ====================

// 生成 Sigmoid 分段线性拟合 LUT（INT8 版本）
SigmoidLUT_INT8 generate_sigmoid_lut_int8(
    int8_t shift_bits_x,// 输入 shift_bits
    int8_t zp_x,        // 输入 zero-point
    int8_t shift_bits_y,// 输出 shift_bits
    int8_t zp_y,        // 输出 zero-point
    float x_min,        // 输入范围最小值
    float x_max         // 输入范围最大值
) {
    SigmoidLUT_INT8 lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // 1. 生成分段点（自适应分段）
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    // 2. 对每段进行线性拟合
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
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));// Sigmoid
        }

        // 线性拟合: y = b*x + c
        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);

        // 3. 量化系数 b（对称量化，zero-point=0）
        int8_t shift_bits_b = determine_shift_bits_int8(std::abs(b_fp));
        int8_t q_b = quantize_coefficient_int8(b_fp, shift_bits_b);

        // 4. 量化系数 c（需要烘焙 zero-point）
        // c_adjusted = c + zp_y * scale_y
        float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
        float c_adjusted = c_fp + static_cast<float>(zp_y) * scale_y;

        int8_t shift_bits_c = determine_shift_bits_int8(std::abs(c_adjusted));
        int16_t q_c = quantize_coefficient_int16(c_adjusted, shift_bits_c);

        // 5. 计算 shift_bits_bx（根据 bx 的实际范围）
        float bx_at_start = b_fp * x_start;
        float bx_at_end = b_fp * x_end;
        float bx_min = std::min(bx_at_start, bx_at_end);
        float bx_max = std::max(bx_at_start, bx_at_end);

        // 根据 bx 的范围确定 shift_bits_bx
        float bx_abs_max = std::max(std::abs(bx_min), std::abs(bx_max));
        if (bx_abs_max < 1e-9f) {
            bx_abs_max = 1e-9f;// 避免除零
        }

        // 计算 shift_bits_bx：使 scale_bx = 2^(-shift_bits_bx) 能够覆盖 bx 的范围
        // 🔥 修正：根据 Python 参考（u8.py），bx 使用非对称量化（无符号），范围 [0, 255]
        // scale_bx >= bx_range / 255 (UINT8 最大值)
        const float max_uint8 = 255.0f;
        float bx_range = bx_max - bx_min;// bx 的实际范围（可能包含负值，通过 zero-point 处理）
        if (bx_range < 1e-9f) {
            bx_range = 1e-9f;// 避免除零
        }
        float raw_scale_bx = bx_range / max_uint8;
        int8_t shift_bits_bx = static_cast<int8_t>(std::ceil(-std::log2(raw_scale_bx)));
        shift_bits_bx = std::max(static_cast<int8_t>(0), shift_bits_bx);// 确保非负

        // 6. 计算移位位数（根据文档公式）
        int8_t n_bx = shift_bits_b + shift_bits_x - shift_bits_bx;
        int8_t n_yb = shift_bits_bx - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        // 融合移位
        int8_t n_BX_total = n_bx + n_yb;

        // 7. 预计算 term_c（INT16 存储）
        int16_t term_c_precomputed;
        if (n_yc >= 0) {
            term_c_precomputed = static_cast<int16_t>(q_c >> n_yc);
        } else {
            term_c_precomputed = static_cast<int16_t>(q_c << (-n_yc));
        }
        // 确保在 INT16 范围内
        term_c_precomputed = std::max(-32768, std::min(32767, static_cast<int32_t>(term_c_precomputed)));

        // 8. 量化阈值（使用无符号量化，直接使用 quantize_input_uint8）
        // 🔥 修正：根据 Python 参考，输入应使用无符号量化 [0, 255]
        uint8_t threshold = quantize_input_uint8(x_end, shift_bits_x, zp_x);

        // 保存段参数
        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

// 生成 Tanh 分段线性拟合 LUT（INT8 版本）
SigmoidLUT_INT8 generate_tanh_lut_int8(
    int8_t shift_bits_x,
    int8_t zp_x,
    int8_t shift_bits_y,
    int8_t zp_y,
    float x_min,
    float x_max) {
    SigmoidLUT_INT8 lut;
    lut.shift_bits_x = shift_bits_x;
    lut.zp_x = zp_x;
    lut.shift_bits_y = shift_bits_y;
    lut.zp_y = zp_y;

    // 与 sigmoid 类似的实现，但使用 tanh 函数
    std::vector<float> segment_points = adaptive_segmentation_sigmoid(x_min, x_max, NUM_SEGMENTS);

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        float x_start = segment_points[i];
        float x_end = segment_points[i + 1];

        const int num_samples = 100;
        std::vector<float> x_seg(num_samples);
        std::vector<float> y_seg(num_samples);

        for (int j = 0; j < num_samples; j++) {
            float x_val = x_start + (x_end - x_start) * static_cast<float>(j) / (num_samples - 1);
            x_seg[j] = x_val;
            y_seg[j] = std::tanh(x_val);// Tanh
        }

        float b_fp, c_fp;
        linear_fit(x_seg, y_seg, b_fp, c_fp);

        int8_t shift_bits_b = determine_shift_bits_int8(std::abs(b_fp));
        int8_t q_b = quantize_coefficient_int8(b_fp, shift_bits_b);

        float scale_y = std::pow(2.0f, -static_cast<float>(shift_bits_y));
        float c_adjusted = c_fp + static_cast<float>(zp_y) * scale_y;

        int8_t shift_bits_c = determine_shift_bits_int8(std::abs(c_adjusted));
        int16_t q_c = quantize_coefficient_int16(c_adjusted, shift_bits_c);

        // 计算 shift_bits_bx（根据 bx 的实际范围）
        float bx_at_start = b_fp * x_start;
        float bx_at_end = b_fp * x_end;
        float bx_min = std::min(bx_at_start, bx_at_end);
        float bx_max = std::max(bx_at_start, bx_at_end);

        // 🔥 修正：根据 Python 参考（u8.py），bx 使用非对称量化（无符号），范围 [0, 255]
        // scale_bx >= bx_range / 255 (UINT8 最大值)
        const float max_uint8 = 255.0f;
        float bx_range = bx_max - bx_min;// bx 的实际范围（可能包含负值，通过 zero-point 处理）
        if (bx_range < 1e-9f) {
            bx_range = 1e-9f;// 避免除零
        }
        float raw_scale_bx = bx_range / max_uint8;
        int8_t shift_bits_bx = static_cast<int8_t>(std::ceil(-std::log2(raw_scale_bx)));
        shift_bits_bx = std::max(static_cast<int8_t>(0), shift_bits_bx);

        int8_t n_bx = shift_bits_b + shift_bits_x - shift_bits_bx;
        int8_t n_yb = shift_bits_bx - shift_bits_y;
        int8_t n_yc = shift_bits_c - shift_bits_y;

        int8_t n_BX_total = n_bx + n_yb;

        int16_t term_c_precomputed;
        if (n_yc >= 0) {
            term_c_precomputed = static_cast<int16_t>(q_c >> n_yc);
        } else {
            term_c_precomputed = static_cast<int16_t>(q_c << (-n_yc));
        }
        term_c_precomputed = std::max(-32768, std::min(32767, static_cast<int32_t>(term_c_precomputed)));

        // 🔥 修正：根据 Python 参考，输入应使用无符号量化 [0, 255]
        uint8_t threshold = quantize_input_uint8(x_end, shift_bits_x, zp_x);

        lut.segments[i].q_b = q_b;
        lut.segments[i].n_BX_total = n_BX_total;
        lut.segments[i].term_c_precomputed = term_c_precomputed;
        lut.segments[i].threshold = threshold;
    }

    return lut;
}

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT8 版本 - z 门）
void init_sigmoid_z_lut_int8(
    int8_t shift_bits_x,
    int8_t zp_x,
    int8_t shift_bits_y,
    int8_t zp_y,
    float x_min,
    float x_max) {
    SigmoidLUT_INT8 lut = generate_sigmoid_lut_int8(
        shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(
        d_sigmoid_z_lut_int8, &lut, sizeof(SigmoidLUT_INT8));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid z LUT (INT8) to constant memory: %s\n",
               cudaGetErrorString(err));
    }
}

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT8 版本 - r 门）
void init_sigmoid_r_lut_int8(
    int8_t shift_bits_x,
    int8_t zp_x,
    int8_t shift_bits_y,
    int8_t zp_y,
    float x_min,
    float x_max) {
    SigmoidLUT_INT8 lut = generate_sigmoid_lut_int8(
        shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(
        d_sigmoid_r_lut_int8, &lut, sizeof(SigmoidLUT_INT8));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid r LUT (INT8) to constant memory: %s\n",
               cudaGetErrorString(err));
    }
}

void init_tanh_lut_int8(
    int8_t shift_bits_x,
    int8_t zp_x,
    int8_t shift_bits_y,
    int8_t zp_y,
    float x_min,
    float x_max) {
    SigmoidLUT_INT8 lut = generate_tanh_lut_int8(
        shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(
        d_tanh_lut_int8, &lut, sizeof(SigmoidLUT_INT8));

    if (err != cudaSuccess) {
        printf("Failed to copy tanh LUT (INT8) to constant memory: %s\n",
               cudaGetErrorString(err));
    }
}

// 显式实例化 generate_piecewise_linear_lut_from_exp2_inv 模板函数
template void generate_piecewise_linear_lut_from_exp2_inv<int8_t>(
    int32_t exp2_inv_z_pre, int32_t zp_z_pre,
    int32_t exp2_inv_z_out, int32_t zp_z_out,
    int32_t exp2_inv_r_pre, int32_t zp_r_pre,
    int32_t exp2_inv_r_out, int32_t zp_r_out,
    int32_t exp2_inv_g_pre, int32_t zp_g_pre,
    int32_t exp2_inv_g_out, int32_t zp_g_out);

template void generate_piecewise_linear_lut_from_exp2_inv<int16_t>(
    int32_t exp2_inv_z_pre, int32_t zp_z_pre,
    int32_t exp2_inv_z_out, int32_t zp_z_out,
    int32_t exp2_inv_r_pre, int32_t zp_r_pre,
    int32_t exp2_inv_r_out, int32_t zp_r_out,
    int32_t exp2_inv_g_pre, int32_t zp_g_pre,
    int32_t exp2_inv_g_out, int32_t zp_g_out);
