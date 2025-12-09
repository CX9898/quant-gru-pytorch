#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>

#include "devVector.h"
#include "quantize_ops.cuh"
#include "quantize_ops_helper.hpp"

// 前向声明
struct SigmoidLUT_INT16;
struct SigmoidLUT_INT8;
struct TanhLUT_INT16;
struct TanhLUT_INT8;

__constant__ int8_t d_sigmoid_int8_z_lut[256];
__constant__ int8_t d_sigmoid_int8_r_lut[256];
__constant__ int8_t d_tanh_int8_g_lut[256];

// uint8 版本的 sigmoid LUT（输出范围 [0, 255]，适用于 sigmoid 输出）
__constant__ uint8_t d_sigmoid_uint8_z_lut[256];
__constant__ uint8_t d_sigmoid_uint8_r_lut[256];

// 分段线性量化常量内存
// Sigmoid LUT（z/r 门）- 输入 int，输出 uint
__constant__ SigmoidLUT_INT16 d_sigmoid_z_lut_int16;
__constant__ SigmoidLUT_INT16 d_sigmoid_r_lut_int16;
__constant__ SigmoidLUT_INT8 d_sigmoid_z_lut_int8;
__constant__ SigmoidLUT_INT8 d_sigmoid_r_lut_int8;

// Tanh LUT（g 门）- 输入 int，输出 int
__constant__ TanhLUT_INT16 d_tanh_lut_int16;
__constant__ TanhLUT_INT8 d_tanh_lut_int8;

std::vector<int8_t> generate_sigmoid_int8_lut(float scale_z_pre, int zp_z_pre, float scale_z,
                                              int zp_z) {
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

// uint8 版本：sigmoid 输出范围 [0, 1] 映射到 [0, 255]
std::vector<uint8_t> generate_sigmoid_uint8_lut(float scale_z_pre, int zp_z_pre, float scale_z,
                                                int zp_z) {
    std::vector<uint8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;  // 输入仍然是 int8 范围

        const float x_fp = static_cast<float>(x_i8 - zp_z_pre) * scale_z_pre;
        const float y_fp = 1.f / (1.f + std::exp(-x_fp));

        // 输出量化到 uint8 [0, 255]
        int y_u8 = static_cast<int>(std::round(y_fp / scale_z + zp_z));
        if (y_u8 < 0) y_u8 = 0;
        if (y_u8 > 255) y_u8 = 255;

        lut[i] = static_cast<uint8_t>(y_u8);
    }
    return lut;
}

std::vector<int8_t> generate_tanh_int8_lut(float scale_pre, int zp_pre, float scale_out,
                                           int zp_out) {
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

    cudaMemcpyToSymbol(d_sigmoid_int8_z_lut, sigmoid_z_lut.data(),
                       sizeof(int8_t) * 256);  // 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(d_sigmoid_int8_r_lut, sigmoid_r_lut.data(),
                       sizeof(int8_t) * 256);  // 从host端拷贝到device端中编译期固定的地址
    cudaMemcpyToSymbol(d_tanh_int8_g_lut, tanh_int8_lut.data(),
                       sizeof(int8_t) * 256);  // 从host端拷贝到device端中编译期固定的地址
}

std::vector<int8_t> generate_sigmoid_int8_lut_exp2(int32_t exp2_inv_z_pre, int zp_z_pre,
                                                   int32_t exp2_inv_z, int zp_z) {
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

std::vector<int8_t> generate_tanh_int8_lut_exp2(int32_t exp2_inv_pre, int zp_pre,
                                                int32_t exp2_inv_out, int zp_out) {
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

// uint8 版本：sigmoid 输出量化到 [0, 255]
std::vector<uint8_t> generate_sigmoid_uint8_lut_exp2(int32_t exp2_inv_z_pre, int zp_z_pre,
                                                     int32_t exp2_inv_z, int zp_z) {
    std::vector<uint8_t> lut(256);

    for (int i = 0; i < 256; i++) {
        int x_i8 = i - 128;  // 输入仍然是 int8 范围

        // （1）反量化 x
        float x_fp = dequantize(x_i8, exp2_inv_z_pre, zp_z_pre);

        // （2）计算 sigmoid
        float y_fp = 1.f / (1.f + std::exp(-x_fp));

        // （3）量化 y 到 uint8
        int y_u8 = quantize<uint8_t>(y_fp, exp2_inv_z, zp_z);

        lut[i] = static_cast<uint8_t>(y_u8);
    }

    return lut;
}

void generate_uint8_lut_from_exp2_inv(int32_t exp2_inv_z_pre, int32_t zp_z_pre,
                                      int32_t exp2_inv_z_out, int32_t zp_z_out,
                                      int32_t exp2_inv_r_pre, int32_t zp_r_pre,
                                      int32_t exp2_inv_r_out, int32_t zp_r_out) {
    std::vector<uint8_t> sigmoid_z_lut =
        generate_sigmoid_uint8_lut_exp2(exp2_inv_z_pre, zp_z_pre, exp2_inv_z_out, zp_z_out);
    std::vector<uint8_t> sigmoid_r_lut =
        generate_sigmoid_uint8_lut_exp2(exp2_inv_r_pre, zp_r_pre, exp2_inv_r_out, zp_r_out);

    cudaMemcpyToSymbol(d_sigmoid_uint8_z_lut, sigmoid_z_lut.data(), sizeof(uint8_t) * 256);
    cudaMemcpyToSymbol(d_sigmoid_uint8_r_lut, sigmoid_r_lut.data(), sizeof(uint8_t) * 256);
}

void generate_int8_lut_from_exp2_inv(int32_t exp2_inv_z_pre, int32_t zp_z_pre,
                                     int32_t exp2_inv_z_out, int32_t zp_z_out,
                                     int32_t exp2_inv_r_pre, int32_t zp_r_pre,
                                     int32_t exp2_inv_r_out, int32_t zp_r_out,
                                     int32_t exp2_inv_g_pre, int32_t zp_g_pre,
                                     int32_t exp2_inv_g_out, int32_t zp_g_out) {
    std::vector<int8_t> sigmoid_z_lut =
        generate_sigmoid_int8_lut_exp2(exp2_inv_z_pre, zp_z_pre, exp2_inv_z_out, zp_z_out);
    std::vector<int8_t> sigmoid_r_lut =
        generate_sigmoid_int8_lut_exp2(exp2_inv_r_pre, zp_r_pre, exp2_inv_r_out, zp_r_out);
    std::vector<int8_t> tanh_int8_lut =
        generate_tanh_int8_lut_exp2(exp2_inv_g_pre, zp_g_pre, exp2_inv_g_out, zp_g_out);

    cudaMemcpyToSymbol(d_sigmoid_int8_z_lut, sigmoid_z_lut.data(), sizeof(int8_t) * 256);
    cudaMemcpyToSymbol(d_sigmoid_int8_r_lut, sigmoid_r_lut.data(), sizeof(int8_t) * 256);
    cudaMemcpyToSymbol(d_tanh_int8_g_lut, tanh_int8_lut.data(), sizeof(int8_t) * 256);
}

// 生成分段线性量化表（基于 GRUQuantitativeParameters）
// 根据 bitwidth_config_ 自动选择每个门的 LUT 类型
// - z/r 门（sigmoid）输出是 uint8 或 uint16
// - g 门（tanh）输出是 int8 或 int16
void generate_piecewise_linear_lut_from_exp2_inv(const GRUQuantitativeParameters &params) {
    const auto &cfg = params.bitwidth_config_;

    // 辅助函数：判断是否为16位量化
    auto is_16bit = [](QuantBitWidth bw) {
        return bw == QuantBitWidth::INT16 || bw == QuantBitWidth::UINT16;
    };

    // 从量化参数计算 scale
    auto calculate_scale = [](int32_t exp2_inv) -> float {
        if (exp2_inv >= 0) {
            return 1.0f / static_cast<float>(1 << exp2_inv);
        } else {
            return static_cast<float>(1 << (-exp2_inv));
        }
    };

    // 判断每个门使用 8 位还是 16 位
    bool z_use_16bit = is_16bit(cfg.z_out_bitwidth);
    bool r_use_16bit = is_16bit(cfg.r_out_bitwidth);
    bool g_use_16bit = is_16bit(cfg.g_out_bitwidth);

    // 获取每个门的量化范围（sigmoid 输出是 uint，tanh 输出是 int）
    // z 门：uint8 [0, 255] 或 uint16 [0, 65535]
    int32_t z_quant_min = 0;
    int32_t z_quant_max = z_use_16bit ? 65535 : 255;

    // r 门：uint8 [0, 255] 或 uint16 [0, 65535]
    int32_t r_quant_min = 0;
    int32_t r_quant_max = r_use_16bit ? 65535 : 255;

    // g 门（tanh）：int8 [-128, 127] 或 int16 [-32768, 32767]
    int32_t g_quant_min = g_use_16bit ? -32768 : -128;
    int32_t g_quant_max = g_use_16bit ? 32767 : 127;

    // 计算每个门的输入范围
    float scale_z_pre = calculate_scale(params.exp2_inv_z_pre_);
    float x_min_z = static_cast<float>(z_quant_min - params.zp_z_pre_) * scale_z_pre;
    float x_max_z = static_cast<float>(z_quant_max - params.zp_z_pre_) * scale_z_pre;

    float scale_r_pre = calculate_scale(params.exp2_inv_r_pre_);
    float x_min_r = static_cast<float>(r_quant_min - params.zp_r_pre_) * scale_r_pre;
    float x_max_r = static_cast<float>(r_quant_max - params.zp_r_pre_) * scale_r_pre;

    float scale_g_pre = calculate_scale(params.exp2_inv_g_pre_);
    float x_min_g = static_cast<float>(g_quant_min - params.zp_g_pre_) * scale_g_pre;
    float x_max_g = static_cast<float>(g_quant_max - params.zp_g_pre_) * scale_g_pre;

    // 转换为 shift_bits
    int8_t shift_bits_z_pre =
        static_cast<int8_t>(std::max(0, std::min(127, params.exp2_inv_z_pre_)));
    int8_t shift_bits_z_out =
        static_cast<int8_t>(std::max(0, std::min(127, params.exp2_inv_z_out_)));
    int8_t shift_bits_r_pre =
        static_cast<int8_t>(std::max(0, std::min(127, params.exp2_inv_r_pre_)));
    int8_t shift_bits_r_out =
        static_cast<int8_t>(std::max(0, std::min(127, params.exp2_inv_r_out_)));
    int8_t shift_bits_g_pre =
        static_cast<int8_t>(std::max(0, std::min(127, params.exp2_inv_g_pre_)));
    int8_t shift_bits_g_out =
        static_cast<int8_t>(std::max(0, std::min(127, params.exp2_inv_g_out_)));

    // ========== Z 门 LUT 初始化（sigmoid，输出 uint）==========
    if (z_use_16bit) {
        uint16_t zp_z_pre_quant =
            static_cast<uint16_t>(std::max(0, std::min(65535, params.zp_z_pre_)));
        uint16_t zp_z_out_quant =
            static_cast<uint16_t>(std::max(0, std::min(65535, params.zp_z_out_)));
        init_sigmoid_z_lut_int16(shift_bits_z_pre, static_cast<int16_t>(zp_z_pre_quant),
                                 shift_bits_z_out, static_cast<int16_t>(zp_z_out_quant), x_min_z,
                                 x_max_z);
    } else {
        uint8_t zp_z_pre_quant = static_cast<uint8_t>(std::max(0, std::min(255, params.zp_z_pre_)));
        uint8_t zp_z_out_quant = static_cast<uint8_t>(std::max(0, std::min(255, params.zp_z_out_)));
        init_sigmoid_z_lut_int8(shift_bits_z_pre, static_cast<int8_t>(zp_z_pre_quant),
                                shift_bits_z_out, static_cast<int8_t>(zp_z_out_quant), x_min_z,
                                x_max_z);
    }

    // ========== R 门 LUT 初始化（sigmoid，输出 uint）==========
    if (r_use_16bit) {
        uint16_t zp_r_pre_quant =
            static_cast<uint16_t>(std::max(0, std::min(65535, params.zp_r_pre_)));
        uint16_t zp_r_out_quant =
            static_cast<uint16_t>(std::max(0, std::min(65535, params.zp_r_out_)));
        init_sigmoid_r_lut_int16(shift_bits_r_pre, static_cast<int16_t>(zp_r_pre_quant),
                                 shift_bits_r_out, static_cast<int16_t>(zp_r_out_quant), x_min_r,
                                 x_max_r);
    } else {
        uint8_t zp_r_pre_quant = static_cast<uint8_t>(std::max(0, std::min(255, params.zp_r_pre_)));
        uint8_t zp_r_out_quant = static_cast<uint8_t>(std::max(0, std::min(255, params.zp_r_out_)));
        init_sigmoid_r_lut_int8(shift_bits_r_pre, static_cast<int8_t>(zp_r_pre_quant),
                                shift_bits_r_out, static_cast<int8_t>(zp_r_out_quant), x_min_r,
                                x_max_r);
    }

    // ========== G 门 LUT 初始化（tanh，输出 int）==========
    if (g_use_16bit) {
        int16_t zp_g_pre_quant =
            static_cast<int16_t>(std::max(-32768, std::min(32767, params.zp_g_pre_)));
        int16_t zp_g_out_quant =
            static_cast<int16_t>(std::max(-32768, std::min(32767, params.zp_g_out_)));
        init_tanh_lut_int16(shift_bits_g_pre, zp_g_pre_quant, shift_bits_g_out, zp_g_out_quant,
                            x_min_g, x_max_g);
    } else {
        int8_t zp_g_pre_quant =
            static_cast<int8_t>(std::max(-128, std::min(127, params.zp_g_pre_)));
        int8_t zp_g_out_quant =
            static_cast<int8_t>(std::max(-128, std::min(127, params.zp_g_out_)));
        init_tanh_lut_int8(shift_bits_g_pre, zp_g_pre_quant, shift_bits_g_out, zp_g_out_quant,
                           x_min_g, x_max_g);
    }
}

namespace kernel {

template <typename T>
__global__ void computeWeightSumMulZP(
    const T *__restrict__ W_q,         // [out_dim, in_dim] 权重量化矩阵, 列主序储存
    int32_t *__restrict__ weight_sum,  // [out_dim] 输出数组
    int x_zp,
    const int32_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
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
__global__ void quantification(const T *data, QuantT *quant_data, size_t size, int32_t exp2_inv,
                               int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    quant_data[idx] = dev::quantize<QuantT>(data[idx], exp2_inv, zp);
}

template <typename T, typename QuantT>
__global__ void dequantification(const QuantT *quant_data, T *data, size_t size, int32_t exp2_inv,
                                 int32_t zp) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    data[idx] = dequantize<QuantT>(quant_data[idx], exp2_inv, zp);
}

}  // namespace kernel

namespace kernel {

template <typename T, typename QuantT>
__global__ void quantificationV(const T *data, QuantT *quant_data, int time_steps, int batch_size,
                                int hidden_size, int32_t exp2_inv_z, int32_t zp_z,
                                int32_t exp2_inv_r, int32_t zp_r, int32_t exp2_inv_g, int32_t zp_g,
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

template <typename T, typename QuantT>
__global__ void dequantificationV(const QuantT *quant_data, T *data, int time_steps, int batch_size,
                                  int hidden_size, int32_t exp2_inv_z, int32_t zp_z,
                                  int32_t exp2_inv_r, int32_t zp_r, int32_t exp2_inv_g,
                                  int32_t zp_g, int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br) {
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

template <typename T, typename QuantT>
__global__ void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                                         size_t channel_size, const int32_t *exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const int32_t exp2_inv = exp2_invs[channel_idx];

    const size_t idx = input_idx * channel_size + channel_idx;
    quant_data[idx] = dev::quantize<QuantT>(src[idx], exp2_inv, 0);
}

template <typename T, typename QuantT>
__global__ void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                           size_t channel_size, const int32_t *exp2_invs) {
    const size_t channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel_idx >= channel_size || input_idx >= input_size) {
        return;
    }

    const int32_t exp2_inv = exp2_invs[channel_idx];

    const size_t idx = input_idx * channel_size + channel_idx;
    data[idx] = dequantize<QuantT>(quant_data[idx], exp2_inv, 0);
}

}  // namespace kernel

template <typename T>
void computeWeightSumMulzp(
    const T *W_q,         // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,  // [out_dim] 输出数组
    int x_zp,
    const int32_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
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
    const int32_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,  // 输出通道数 (M)
    int in_dim,   // 输入通道数 (K)
    cudaStream_t stream);

template void computeWeightSumMulzp<int16_t>(
    const int16_t *W_q,   // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,  // [out_dim] 输出数组
    int x_zp,
    const int32_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n.
    // per-channel
    int out_dim,  // 输出通道数 (M)
    int in_dim,   // 输入通道数 (K)
    cudaStream_t stream);

namespace dev {

template <typename T, typename QuantT>
void quantification(const T *data, QuantT *quant_data, size_t size, int32_t exp2_inv, int32_t zp) {
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

template <typename T, typename QuantT>
void dequantification(const QuantT *quant_data, T *data, size_t size, int32_t exp2_inv,
                      int32_t zp) {
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

template <typename T, typename QuantT>
void quantificationV(const T *data, QuantT *quant_data, int time_steps, int batch_size,
                     int hidden_size, int32_t exp2_inv_z, int32_t zp_z, int32_t exp2_inv_r,
                     int32_t zp_r, int32_t exp2_inv_g, int32_t zp_g, int32_t exp2_inv_Rh_add_br,
                     int32_t zp_Rh_add_br) {
    // Launch configuration: 每个block处理一个时间步和一个batch的所有hidden单元
    // blockDim.x = hidden_size (每个线程处理一个hidden单元)
    // gridDim.x = time_steps
    // gridDim.y = batch_size
    const dim3 blockDim(hidden_size);
    const dim3 gridDim(time_steps, batch_size);

    kernel::quantificationV<<<gridDim, blockDim>>>(
        data, quant_data, time_steps, batch_size, hidden_size, exp2_inv_z, zp_z, exp2_inv_r, zp_r,
        exp2_inv_g, zp_g, exp2_inv_Rh_add_br, zp_Rh_add_br);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("quantificationV kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template void quantificationV<float, int8_t>(const float *data, int8_t *quant_data, int time_steps,
                                             int batch_size, int hidden_size, int32_t exp2_inv_z,
                                             int32_t zp_z, int32_t exp2_inv_r, int32_t zp_r,
                                             int32_t exp2_inv_g, int32_t zp_g,
                                             int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);
template void quantificationV<float, int16_t>(const float *data, int16_t *quant_data,
                                              int time_steps, int batch_size, int hidden_size,
                                              int32_t exp2_inv_z, int32_t zp_z, int32_t exp2_inv_r,
                                              int32_t zp_r, int32_t exp2_inv_g, int32_t zp_g,
                                              int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);

template <typename T, typename QuantT>
void dequantificationV(const QuantT *quant_data, T *data, int time_steps, int batch_size,
                       int hidden_size, int32_t exp2_inv_z, int32_t zp_z, int32_t exp2_inv_r,
                       int32_t zp_r, int32_t exp2_inv_g, int32_t zp_g, int32_t exp2_inv_Rh_add_br,
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

template void dequantificationV<float, int8_t>(const int8_t *quant_data, float *data,
                                               int time_steps, int batch_size, int hidden_size,
                                               int32_t exp2_inv_z, int32_t zp_z, int32_t exp2_inv_r,
                                               int32_t zp_r, int32_t exp2_inv_g, int32_t zp_g,
                                               int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);
template void dequantificationV<float, int16_t>(const int16_t *quant_data, float *data,
                                                int time_steps, int batch_size, int hidden_size,
                                                int32_t exp2_inv_z, int32_t zp_z,
                                                int32_t exp2_inv_r, int32_t zp_r,
                                                int32_t exp2_inv_g, int32_t zp_g,
                                                int32_t exp2_inv_Rh_add_br, int32_t zp_Rh_add_br);

template <typename T, typename QuantT>
void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                              size_t channel_size, const dev::vector<int32_t> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::quantificationPerChannel<<<gridDim, blockDim>>>(src, quant_data, input_size,
                                                            channel_size, exp2_invs.data());
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

template <typename T, typename QuantT>
void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                size_t channel_size, const dev::vector<int32_t> &exp2_invs) {
    const dim3 blockDim(32, 16);
    const dim3 gridDim((channel_size + blockDim.x - 1) / blockDim.x,
                       (input_size + blockDim.y - 1) / blockDim.y);

    kernel::dequantificationPerChannel<<<gridDim, blockDim>>>(quant_data, data, input_size,
                                                              channel_size, exp2_invs.data());
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

// 辅助函数：计算 shift_bits 并进行边界调整（与 Python 实现一致）
// 返回 (shift_bits, adjusted_max, zero_point)
inline std::tuple<int8_t, float, int32_t> calculate_shift_bits_and_adjust_range(float f_min,
                                                                                float f_max,
                                                                                int32_t quant_max) {
    // 步骤1: 计算原始 scale
    float s_original_scale = (f_max - f_min) / static_cast<float>(quant_max);

    // 步骤2: 转换为2的幂次方 scale = 1 / 2^n
    // 使用 round() 而不是 ceil()，与 Python 实现一致
    int8_t shift_bits = 0;
    if (s_original_scale > 0) {
        float n_shift = -std::log2(s_original_scale);
        shift_bits = static_cast<int8_t>(std::round(n_shift));
        shift_bits = std::max(static_cast<int8_t>(0), shift_bits);
    }

    // 步骤3: 计算 scale 和 zero_point
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t zero_point = static_cast<int32_t>(std::round(-f_min / scale));

    // 步骤4: 验证边界并调整 f_max（与 Python 实现一致）
    int32_t q_max_from_f_max = static_cast<int32_t>(std::round(f_max / scale)) + zero_point;
    float adjusted_f_max = f_max;
    if (q_max_from_f_max > quant_max) {
        // 调整 f_max，使得 q_max = Q_max，保持 zero_point 不变
        adjusted_f_max = (quant_max - zero_point) * scale;
    }

    return std::make_tuple(shift_bits, adjusted_f_max, zero_point);
}

// 自适应分段（Sigmoid 专用）
std::vector<float> adaptive_segmentation_sigmoid(float x_min, float x_max, int num_segments) {
    std::vector<float> segment_points(num_segments + 1);
    segment_points[0] = x_min;
    segment_points[num_segments] = x_max;

    // 在中心区域（x ≈ 0）密集分段
    float center_range = 2.0f;       // 中心区域范围 [-2, 2]
    int n_dense = num_segments / 2;  // 一半段用于中心区域
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
SigmoidLUT_INT16 generate_sigmoid_lut_int16(int8_t shift_bits_x,  // 输入 shift_bits
                                            int16_t zp_x,         // 输入 zero-point
                                            int8_t shift_bits_y,  // 输出 shift_bits
                                            int16_t zp_y,         // 输出 zero-point
                                            float x_min,          // 输入范围最小值
                                            float x_max           // 输入范围最大值
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
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));  // Sigmoid
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
            bx_abs_max = 1e-9f;  // 避免除零
        }

        // 计算 shift_bits_bx：使 scale_bx = 2^(-shift_bits_bx) 能够覆盖 bx 的范围
        // 🔥 修正：根据 Python 参考（u16.py），bx 使用非对称量化（无符号），范围 [0, 65535]
        // 使用与 Python 一致的 round() 方法，并添加边界调整机制
        const int32_t max_uint16 = 65535;
        if (bx_max - bx_min < 1e-9f) {
            bx_max = bx_min + 1e-9f;  // 避免除零
        }
        auto [shift_bits_bx, adjusted_bx_max, zp_bx] =
            calculate_shift_bits_and_adjust_range(bx_min, bx_max, max_uint16);
        // 注意：adjusted_bx_max 和 zp_bx 目前不直接使用，但保持与 Python 实现一致

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
// Tanh 输入 int16，输出 int16（有符号）
TanhLUT_INT16 generate_tanh_lut_int16(int8_t shift_bits_x, int16_t zp_x, int8_t shift_bits_y,
                                      int16_t zp_y, float x_min, float x_max) {
    TanhLUT_INT16 lut;
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
            y_seg[j] = std::tanh(x_val);  // Tanh
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
        // 使用与 Python 一致的 round() 方法，并添加边界调整机制
        const int32_t max_uint16 = 65535;
        if (bx_max - bx_min < 1e-9f) {
            bx_max = bx_min + 1e-9f;  // 避免除零
        }
        auto [shift_bits_bx, adjusted_bx_max, zp_bx] =
            calculate_shift_bits_and_adjust_range(bx_min, bx_max, max_uint16);
        // 注意：adjusted_bx_max 和 zp_bx 目前不直接使用，但保持与 Python 实现一致

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
void init_sigmoid_z_lut_int16(int8_t shift_bits_x, int16_t zp_x, int8_t shift_bits_y, int16_t zp_y,
                              float x_min, float x_max) {
    SigmoidLUT_INT16 lut =
        generate_sigmoid_lut_int16(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_sigmoid_z_lut_int16, &lut, sizeof(SigmoidLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid z LUT to constant memory: %s\n", cudaGetErrorString(err));
    }
}

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - r 门）
void init_sigmoid_r_lut_int16(int8_t shift_bits_x, int16_t zp_x, int8_t shift_bits_y, int16_t zp_y,
                              float x_min, float x_max) {
    SigmoidLUT_INT16 lut =
        generate_sigmoid_lut_int16(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_sigmoid_r_lut_int16, &lut, sizeof(SigmoidLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid r LUT to constant memory: %s\n", cudaGetErrorString(err));
    }
}

void init_tanh_lut_int16(int8_t shift_bits_x, int16_t zp_x, int8_t shift_bits_y, int16_t zp_y,
                         float x_min, float x_max) {
    TanhLUT_INT16 lut =
        generate_tanh_lut_int16(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_tanh_lut_int16, &lut, sizeof(TanhLUT_INT16));

    if (err != cudaSuccess) {
        printf("Failed to copy tanh LUT to constant memory: %s\n", cudaGetErrorString(err));
    }
}

// ==================== INT8 版本的分段线性量化参数生成函数 ====================

// 生成 Sigmoid 分段线性拟合 LUT（INT8 版本）
SigmoidLUT_INT8 generate_sigmoid_lut_int8(int8_t shift_bits_x,  // 输入 shift_bits
                                          int8_t zp_x,          // 输入 zero-point
                                          int8_t shift_bits_y,  // 输出 shift_bits
                                          int8_t zp_y,          // 输出 zero-point
                                          float x_min,          // 输入范围最小值
                                          float x_max           // 输入范围最大值
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
            y_seg[j] = 1.0f / (1.0f + std::exp(-x_val));  // Sigmoid
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
            bx_abs_max = 1e-9f;  // 避免除零
        }

        // 计算 shift_bits_bx：使 scale_bx = 2^(-shift_bits_bx) 能够覆盖 bx 的范围
        // 🔥 修正：根据 Python 参考（u8.py），bx 使用非对称量化（无符号），范围 [0, 255]
        // 使用与 Python 一致的 round() 方法，并添加边界调整机制
        const int32_t max_uint8 = 255;
        if (bx_max - bx_min < 1e-9f) {
            bx_max = bx_min + 1e-9f;  // 避免除零
        }
        auto [shift_bits_bx, adjusted_bx_max, zp_bx] =
            calculate_shift_bits_and_adjust_range(bx_min, bx_max, max_uint8);
        // 注意：adjusted_bx_max 和 zp_bx 目前不直接使用，但保持与 Python 实现一致

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
        term_c_precomputed =
            std::max(-32768, std::min(32767, static_cast<int32_t>(term_c_precomputed)));

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
// Tanh 输入 int8，输出 int8（有符号）
TanhLUT_INT8 generate_tanh_lut_int8(int8_t shift_bits_x, int8_t zp_x, int8_t shift_bits_y,
                                    int8_t zp_y, float x_min, float x_max) {
    TanhLUT_INT8 lut;
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
            y_seg[j] = std::tanh(x_val);  // Tanh
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
        // 使用与 Python 一致的 round() 方法，并添加边界调整机制
        const int32_t max_uint8 = 255;
        if (bx_max - bx_min < 1e-9f) {
            bx_max = bx_min + 1e-9f;  // 避免除零
        }
        auto [shift_bits_bx, adjusted_bx_max, zp_bx] =
            calculate_shift_bits_and_adjust_range(bx_min, bx_max, max_uint8);
        // 注意：adjusted_bx_max 和 zp_bx 目前不直接使用，但保持与 Python 实现一致

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
        term_c_precomputed =
            std::max(-32768, std::min(32767, static_cast<int32_t>(term_c_precomputed)));

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
void init_sigmoid_z_lut_int8(int8_t shift_bits_x, int8_t zp_x, int8_t shift_bits_y, int8_t zp_y,
                             float x_min, float x_max) {
    SigmoidLUT_INT8 lut =
        generate_sigmoid_lut_int8(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_sigmoid_z_lut_int8, &lut, sizeof(SigmoidLUT_INT8));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid z LUT (INT8) to constant memory: %s\n",
               cudaGetErrorString(err));
    }
}

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT8 版本 - r 门）
void init_sigmoid_r_lut_int8(int8_t shift_bits_x, int8_t zp_x, int8_t shift_bits_y, int8_t zp_y,
                             float x_min, float x_max) {
    SigmoidLUT_INT8 lut =
        generate_sigmoid_lut_int8(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_sigmoid_r_lut_int8, &lut, sizeof(SigmoidLUT_INT8));

    if (err != cudaSuccess) {
        printf("Failed to copy sigmoid r LUT (INT8) to constant memory: %s\n",
               cudaGetErrorString(err));
    }
}

void init_tanh_lut_int8(int8_t shift_bits_x, int8_t zp_x, int8_t shift_bits_y, int8_t zp_y,
                        float x_min, float x_max) {
    TanhLUT_INT8 lut = generate_tanh_lut_int8(shift_bits_x, zp_x, shift_bits_y, zp_y, x_min, x_max);

    cudaError_t err = cudaMemcpyToSymbol(d_tanh_lut_int8, &lut, sizeof(TanhLUT_INT8));

    if (err != cudaSuccess) {
        printf("Failed to copy tanh LUT (INT8) to constant memory: %s\n", cudaGetErrorString(err));
    }
}

template <typename T, typename QuantT>
void calculateScale(const std::vector<T> &data_host, const bool use_symmetric, int32_t &exp2_inv,
                    int32_t &zp, const std::string &name) {
    T min_val = data_host[0];
    T max_val = data_host[0];
#pragma omp parallel for reduction(min : min_val, max : max_val)
    for (int i = 1; i < data_host.size(); ++i) {
        const T val = data_host[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    T min_new = min_val;
    T max_new = max_val;
    calibrateQuantParams<T, QuantT>(min_val, max_val, use_symmetric, min_new, max_new, exp2_inv, zp,
                                    name);
};

template <typename T, typename QuantT>
void calculateScale(const T *data_dev, const size_t size, const bool use_symmetric,
                    int32_t &exp2_inv, int32_t &zp, const std::string &name) {
    std::vector<T> data_host = d2h(data_dev, size);
    calculateScale<T, QuantT>(data_host, use_symmetric, exp2_inv, zp, name);
};

template <typename T, typename QuantT>
std::vector<int32_t> calculateScalesPerChannels(const T *W_dev, int channel_size, int input_size,
                                                const std::string &name) {
    // 列主序排列

    std::vector<T> W_host = d2h(W_dev, channel_size * input_size);

    std::vector<int32_t> exp2_inv_per_channels(channel_size);
    std::vector<T> min(channel_size);
    std::vector<T> max(channel_size);

#pragma omp parallel for
    for (int i = 0; i < channel_size; ++i) {
        min[i] = W_host[i];
        max[i] = W_host[i];
        for (int j = 1; j < input_size; ++j) {
            min[i] = std::min(min[i], W_host[j * channel_size + i]);
            max[i] = std::max(max[i], W_host[j * channel_size + i]);
        }
    }

    std::vector<int32_t> zp_tmp(channel_size);
#pragma omp parallel for
    for (int i = 0; i < channel_size; ++i) {
        if (min[i] == max[i]) {
            const float half = std::abs(min[i]);
            min[i] = -half;
            max[i] = half;
        }
        calibrateQuantParams<T, QuantT>(min[i], max[i], true, min[i], max[i],
                                        exp2_inv_per_channels[i], zp_tmp[i], name);
    }
    return exp2_inv_per_channels;
};

/**
 * 通用(仅host)scale/zp 计算函数
 * @param x_dev  -- 设备端输入数据指针
 * @param size_per_step -- 每步输入长度
 * @param steps -- 步数
 * @param use_symmetric -- 是否对称量化
 * @param name -- 调试信息
 */
template <typename T, typename QuantT>
void calculateScalePerSteps(const T *x_dev, const int size_per_step, const int steps,
                            const bool use_symmetric, int32_t &exp2_inv, int32_t &zp,
                            const std::string &name) {
    if (size_per_step == 0 || steps == 0) {
        printf("Warning! %s input size = 0\n", name.c_str());
        return;
    }
    std::vector<T> x_host = d2h(x_dev, steps * size_per_step);
    std::vector<T> min(steps);
    std::vector<T> max(steps);

#pragma omp parallel for
    for (int t = 0; t < steps; ++t) {
        const int offset = t * size_per_step;
        min[t] = x_host[offset];
        max[t] = x_host[offset];
        for (int i = 1; i < size_per_step; ++i) {
            min[t] = std::min(min[t], x_host[offset + i]);
            max[t] = std::max(max[t], x_host[offset + i]);
        }
    }

    // 使用中位数初始化，避免第一个时间步的异常值影响全局
    // 创建临时副本用于排序
    std::vector<T> min_sorted = min;
    std::vector<T> max_sorted = max;
    std::sort(min_sorted.begin(), min_sorted.end());
    std::sort(max_sorted.begin(), max_sorted.end());

    // 使用中位数初始化（如果steps为偶数，使用中间两个值的平均）
    T res_min = (steps % 2 == 1) ? min_sorted[steps / 2]
                                 : T(0.5) * (min_sorted[steps / 2 - 1] + min_sorted[steps / 2]);
    T res_max = (steps % 2 == 1) ? max_sorted[steps / 2]
                                 : T(0.5) * (max_sorted[steps / 2 - 1] + max_sorted[steps / 2]);

    for (int t = 0; t < steps; ++t) {
#ifdef DEBUG
        // 增加调试信息: 输出res_min, res_max, min[t], max[t]
        printf(
            "[DEBUG][%s][Step %d] res_min = %.8f, res_max = %.8f, min[t] = %.8f, max[t] = %.8f\n",
            name.c_str(), t, static_cast<double>(res_min), static_cast<double>(res_max),
            static_cast<double>(min[t]), static_cast<double>(max[t]));
#endif
        // 平滑更新
        res_min = 0.9 * res_min + 0.1 * min[t];
        res_max = 0.9 * res_max + 0.1 * max[t];

        //        res_min = std::min(res_min, min[t]);
        //        res_max = std::max(res_max, max[t]);
    }

    calibrateQuantParams<T, QuantT>(res_min, res_max, use_symmetric, res_min, res_max, exp2_inv, zp,
                                    name);
};

// ==================== 显式实例化 calculateScale ====================
// 版本1: std::vector 输入
template void calculateScale<float, int8_t>(const std::vector<float> &data_host,
                                            const bool use_symmetric, int32_t &exp2_inv,
                                            int32_t &zp, const std::string &name);
template void calculateScale<float, int16_t>(const std::vector<float> &data_host,
                                             const bool use_symmetric, int32_t &exp2_inv,
                                             int32_t &zp, const std::string &name);
template void calculateScale<float, int32_t>(const std::vector<float> &data_host,
                                             const bool use_symmetric, int32_t &exp2_inv,
                                             int32_t &zp, const std::string &name);
template void calculateScale<float, uint8_t>(const std::vector<float> &data_host,
                                             const bool use_symmetric, int32_t &exp2_inv,
                                             int32_t &zp, const std::string &name);
template void calculateScale<float, uint16_t>(const std::vector<float> &data_host,
                                              const bool use_symmetric, int32_t &exp2_inv,
                                              int32_t &zp, const std::string &name);

// 版本2: 设备指针输入
template void calculateScale<float, int8_t>(const float *data_dev, const size_t size,
                                            const bool use_symmetric, int32_t &exp2_inv,
                                            int32_t &zp, const std::string &name);
template void calculateScale<float, int16_t>(const float *data_dev, const size_t size,
                                             const bool use_symmetric, int32_t &exp2_inv,
                                             int32_t &zp, const std::string &name);
template void calculateScale<float, int32_t>(const float *data_dev, const size_t size,
                                             const bool use_symmetric, int32_t &exp2_inv,
                                             int32_t &zp, const std::string &name);
template void calculateScale<float, uint8_t>(const float *data_dev, const size_t size,
                                             const bool use_symmetric, int32_t &exp2_inv,
                                             int32_t &zp, const std::string &name);
template void calculateScale<float, uint16_t>(const float *data_dev, const size_t size,
                                              const bool use_symmetric, int32_t &exp2_inv,
                                              int32_t &zp, const std::string &name);

// ==================== 显式实例化 calculateScalesPerChannels ====================
template std::vector<int32_t> calculateScalesPerChannels<float, int8_t>(const float *W_dev,
                                                                        int channel_size,
                                                                        int input_size,
                                                                        const std::string &name);
template std::vector<int32_t> calculateScalesPerChannels<float, int16_t>(const float *W_dev,
                                                                         int channel_size,
                                                                         int input_size,
                                                                         const std::string &name);
template std::vector<int32_t> calculateScalesPerChannels<float, int32_t>(const float *W_dev,
                                                                         int channel_size,
                                                                         int input_size,
                                                                         const std::string &name);
template std::vector<int32_t> calculateScalesPerChannels<float, uint8_t>(const float *W_dev,
                                                                         int channel_size,
                                                                         int input_size,
                                                                         const std::string &name);
template std::vector<int32_t> calculateScalesPerChannels<float, uint16_t>(const float *W_dev,
                                                                          int channel_size,
                                                                          int input_size,
                                                                          const std::string &name);

// ==================== 显式实例化 calculateScalePerSteps ====================
template void calculateScalePerSteps<float, int8_t>(const float *x_dev, const int size_per_step,
                                                    const int steps, const bool use_symmetric,
                                                    int32_t &exp2_inv, int32_t &zp,
                                                    const std::string &name);
template void calculateScalePerSteps<float, int16_t>(const float *x_dev, const int size_per_step,
                                                     const int steps, const bool use_symmetric,
                                                     int32_t &exp2_inv, int32_t &zp,
                                                     const std::string &name);
template void calculateScalePerSteps<float, int32_t>(const float *x_dev, const int size_per_step,
                                                     const int steps, const bool use_symmetric,
                                                     int32_t &exp2_inv, int32_t &zp,
                                                     const std::string &name);
template void calculateScalePerSteps<float, uint8_t>(const float *x_dev, const int size_per_step,
                                                     const int steps, const bool use_symmetric,
                                                     int32_t &exp2_inv, int32_t &zp,
                                                     const std::string &name);
template void calculateScalePerSteps<float, uint16_t>(const float *x_dev, const int size_per_step,
                                                      const int steps, const bool use_symmetric,
                                                      int32_t &exp2_inv, int32_t &zp,
                                                      const std::string &name);

// ==================== double 类型的显式实例化 ====================
// calculateScale - std::vector 输入
template void calculateScale<double, int8_t>(const std::vector<double> &data_host,
                                             const bool use_symmetric, int32_t &exp2_inv,
                                             int32_t &zp, const std::string &name);
template void calculateScale<double, int16_t>(const std::vector<double> &data_host,
                                              const bool use_symmetric, int32_t &exp2_inv,
                                              int32_t &zp, const std::string &name);
template void calculateScale<double, int32_t>(const std::vector<double> &data_host,
                                              const bool use_symmetric, int32_t &exp2_inv,
                                              int32_t &zp, const std::string &name);
template void calculateScale<double, uint8_t>(const std::vector<double> &data_host,
                                              const bool use_symmetric, int32_t &exp2_inv,
                                              int32_t &zp, const std::string &name);
template void calculateScale<double, uint16_t>(const std::vector<double> &data_host,
                                               const bool use_symmetric, int32_t &exp2_inv,
                                               int32_t &zp, const std::string &name);

// calculateScale - 设备指针输入
template void calculateScale<double, int8_t>(const double *data_dev, const size_t size,
                                             const bool use_symmetric, int32_t &exp2_inv,
                                             int32_t &zp, const std::string &name);
template void calculateScale<double, int16_t>(const double *data_dev, const size_t size,
                                              const bool use_symmetric, int32_t &exp2_inv,
                                              int32_t &zp, const std::string &name);
template void calculateScale<double, int32_t>(const double *data_dev, const size_t size,
                                              const bool use_symmetric, int32_t &exp2_inv,
                                              int32_t &zp, const std::string &name);
template void calculateScale<double, uint8_t>(const double *data_dev, const size_t size,
                                              const bool use_symmetric, int32_t &exp2_inv,
                                              int32_t &zp, const std::string &name);
template void calculateScale<double, uint16_t>(const double *data_dev, const size_t size,
                                               const bool use_symmetric, int32_t &exp2_inv,
                                               int32_t &zp, const std::string &name);

// calculateScalesPerChannels
template std::vector<int32_t> calculateScalesPerChannels<double, int8_t>(const double *W_dev,
                                                                         int channel_size,
                                                                         int input_size,
                                                                         const std::string &name);
template std::vector<int32_t> calculateScalesPerChannels<double, int16_t>(const double *W_dev,
                                                                          int channel_size,
                                                                          int input_size,
                                                                          const std::string &name);
template std::vector<int32_t> calculateScalesPerChannels<double, int32_t>(const double *W_dev,
                                                                          int channel_size,
                                                                          int input_size,
                                                                          const std::string &name);
template std::vector<int32_t> calculateScalesPerChannels<double, uint8_t>(const double *W_dev,
                                                                          int channel_size,
                                                                          int input_size,
                                                                          const std::string &name);
template std::vector<int32_t> calculateScalesPerChannels<double, uint16_t>(const double *W_dev,
                                                                           int channel_size,
                                                                           int input_size,
                                                                           const std::string &name);

// calculateScalePerSteps
template void calculateScalePerSteps<double, int8_t>(const double *x_dev, const int size_per_step,
                                                     const int steps, const bool use_symmetric,
                                                     int32_t &exp2_inv, int32_t &zp,
                                                     const std::string &name);
template void calculateScalePerSteps<double, int16_t>(const double *x_dev, const int size_per_step,
                                                      const int steps, const bool use_symmetric,
                                                      int32_t &exp2_inv, int32_t &zp,
                                                      const std::string &name);
template void calculateScalePerSteps<double, int32_t>(const double *x_dev, const int size_per_step,
                                                      const int steps, const bool use_symmetric,
                                                      int32_t &exp2_inv, int32_t &zp,
                                                      const std::string &name);
template void calculateScalePerSteps<double, uint8_t>(const double *x_dev, const int size_per_step,
                                                      const int steps, const bool use_symmetric,
                                                      int32_t &exp2_inv, int32_t &zp,
                                                      const std::string &name);
template void calculateScalePerSteps<double, uint16_t>(const double *x_dev, const int size_per_step,
                                                       const int steps, const bool use_symmetric,
                                                       int32_t &exp2_inv, int32_t &zp,
                                                       const std::string &name);
