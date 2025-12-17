#pragma once

#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "devVector.h"
#include "gru_quantization_ranges.hpp"
#include "quantize_bitwidth_config.hpp"

// #define DEBUG

// GRU 量化参数结构体：存储GRU网络量化过程中所有定点化/反量化所需的参数
// 核心约束：所有缩放因子均以「2的负n次方」形式存储，exp2_inv_xxx 表示缩放因子 scale =
// 2^(-exp2_inv_xxx) zp_xxx 表示量化零点（zero point），用于浮点数与整数的映射：量化值 q = round(x /
// scale + zp)，反量化 x = (q - zp) * scale
struct GRUQuantitativeParameters {
    // 为每个算子独立配置量化位宽
    OperatorQuantConfig bitwidth_config_;

    int hidden_;  // channel = hidden * 3
    int8_t exp2_inv_x_;
    int32_t zp_x_;
    int8_t exp2_inv_h_;
    int32_t zp_h_;

    std::vector<int8_t> exp2_inv_W_;  // size = hidden * 3. per-channel
                                      // (每个输出通道一个scale，即W的每一列一个scale)
    std::vector<int8_t> exp2_inv_R_;  // size = hidden * 3. per-channel
                                      // (每个输出通道一个scale，即R的每一列一个scale)

    int8_t exp2_inv_Wx_;
    int32_t zp_Wx_;
    int8_t exp2_inv_Rh_;
    int32_t zp_Rh_;

    std::vector<int8_t> exp2_inv_bx_;
    std::vector<int8_t> exp2_inv_br_;

    int8_t exp2_inv_z_pre_;
    int32_t zp_z_pre_;
    int8_t exp2_inv_r_pre_;
    int32_t zp_r_pre_;
    int8_t exp2_inv_g_pre_;
    int32_t zp_g_pre_;

    int8_t exp2_inv_z_out_;
    int32_t zp_z_out_;
    int8_t exp2_inv_r_out_;
    int32_t zp_r_out_;
    int8_t exp2_inv_g_out_;
    int32_t zp_g_out_;

    int8_t exp2_inv_Rh_add_br_;
    int32_t zp_Rh_add_br_;
    int8_t exp2_inv_rRh_;
    int32_t zp_rRh_;

    int8_t exp2_inv_new_contrib_;
    int32_t zp_new_contrib_;
    int8_t exp2_inv_old_contrib_;
    int32_t zp_old_contrib_;
};

struct QuantGRUReScale {
    int32_t zp_x_;
    int32_t zp_h_;

    dev::vector<int8_t> n_W_mul_x_div_Wx_;  // size = hidden * 3
    int32_t zp_Wx_;
    dev::vector<int8_t> n_R_mul_h_div_Rh_;  // size = hidden * 3
    int32_t zp_Rh_;

    // z门
    int32_t zp_z_pre_;
    int32_t zp_z_out_;
    int8_t exp2_inv_Wx_div_z_pre_;
    int8_t exp2_inv_Wx_div_z_;
    int8_t exp2_inv_Rh_div_z_pre_;
    int8_t exp2_inv_Rh_div_z_;
    dev::vector<int8_t> n_bx_div_z_;
    dev::vector<int8_t> n_br_div_z_;

    // r门
    int32_t zp_r_pre_;
    int32_t zp_r_out_;
    int8_t exp2_inv_Wx_div_r_pre_;  // n5
    int8_t exp2_inv_Rh_div_r_pre_;  // n6
    dev::vector<int8_t> n_bx_div_r_;
    dev::vector<int8_t> n_br_div_r_;

    // New Gate
    int32_t zp_g_pre_;
    int32_t zp_g_out_;
    int8_t n_Rh_div_Rh_add_br_;
    int8_t exp2_inv_Rh_div_Rh_add_br_;
    dev::vector<int8_t> n_br_div_Rh_add_br_;  // br 是 per-channel
    int32_t zp_Rh_add_br_;
    int8_t n_r_mul_Rh_add_br_div_rRh_;     // n9
    int8_t exp2_inv_r_out_mul_h_div_rRh_;  // S9
    int32_t zp_rRh_;
    int8_t n_Wx_div_g_pre_;          // n10
    int8_t exp2_inv_Wx_div_g_pre_;   // S10
    int8_t n_rRh_div_g_pre_;         // n11
    int8_t exp2_inv_rRh_div_g_pre_;  // S11
    dev::vector<int8_t> exp2_inv_bx_div_g_pre_;

    // h_new
    // 1-z 直接复用 z_out 的 scale，将常数1对齐到 z_out 的量化空间
    int32_t one_in_z_scale_;  // 1 对应的量化值: round(1.0 / scale_z_out) + zp_z_out

    int32_t zp_new_contrib_;
    int8_t n_z_out_mul_g_div_new_contrib_;  // (1-z)*g 计算时的 rescale 参数
    int32_t zp_old_contrib_;
    int8_t n_z_mul_h_div_old_contrib_;         // n14
    int8_t exp2_inv_z_mul_h_div_old_contrib_;  // S14
    int8_t n_new_contrib_div_h_;               // n15
    int8_t exp2_inv_new_contrib_div_h_;        // S15
    int8_t n_old_contrib_div_h_;               // n16
    int8_t exp2_inv_old_contrib_div_h_;        // S16

    // device 可访问的 bias scale (从 GRUQuantitativeParameters 拷贝)
    dev::vector<int8_t> exp2_inv_bx_dev_;  // size = hidden * 3
    dev::vector<int8_t> exp2_inv_br_dev_;  // size = hidden * 3

    // 位宽配置（从 GRUQuantitativeParameters 中复制，用于运行时选择正确的 kernel 实例）
    OperatorQuantConfig bitwidth_config_;

    // 调试用：保存完整的量化参数
    GRUQuantitativeParameters test;
};

// 生成分段线性量化表（基于exp2_inv参数，支持模板类型）
// x_min 和 x_max 从量化参数（exp2_inv_pre 和 zp_pre）自动计算：
//   - scale = 2^(-exp2_inv_pre) = 1.0f / (1 << exp2_inv_pre)
//   - x_min = (quant_min - zp_pre) * scale
//   - x_max = (quant_max - zp_pre) * scale
// 生成分段线性量化表（根据 GRUQuantitativeParameters 中的 bitwidth_config_ 决定各门的位宽）
void generate_piecewise_linear_lut(const GRUQuantitativeParameters &params);

__host__ __device__ __forceinline__ int32_t rshift_round(int32_t x, int8_t n) {
    if (n <= 0) return x << (-n);

    const int32_t offset = 1 << (n - 1);
    if (x >= 0) {
        return (x + offset) >> n;
    } else {
        // 对负数要改成向零舍入：
        return -((-x + offset) >> n);
    }
}

// int64_t 版本：用于处理 16 位量化时可能超出 int32 范围的乘积
__host__ __device__ __forceinline__ int64_t rshift_round(int64_t x, int8_t n) {
    if (n <= 0) return x << (-n);

    const int64_t offset = static_cast<int64_t>(1) << (n - 1);
    if (x >= 0) {
        return (x + offset) >> n;
    } else {
        // 对负数要改成向零舍入：
        return -((-x + offset) >> n);
    }
}

// int64_t 版本：用于 16 位量化，避免溢出
template <typename T>
void computeWeightSumMulzp(
    const T *W_q,         // [out_dim, in_dim] 权重量化矩阵
    int64_t *weight_sum,  // [out_dim] 输出数组（int64_t）
    int32_t zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n. per-channel
    int out_dim,                   // 输出通道数 (M)
    int in_dim,                    // 输入通道数 (K)
    cudaStream_t stream = 0);

// int32_t 版本：用于 8 位量化，不会溢出
template <typename T>
void computeWeightSumMulzp(
    const T *W_q,         // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,  // [out_dim] 输出数组（int32_t）
    int32_t zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n. per-channel
    int out_dim,                   // 输出通道数 (M)
    int in_dim,                    // 输入通道数 (K)
    cudaStream_t stream = 0);

void applyZeroPointCompensation2D(int32_t *Y_int32, const int32_t *weight_sum, const int32_t *x_zp,
                                  int out_dim, int batch_size, cudaStream_t stream = 0);

/**
 * @brief
 * 模板化量化参数计算函数：支持任意量化类型（int8/int6等）和输入范围类型，对齐2的负n次方缩放因子
 * @tparam T 输入范围数据类型（如float、double，需支持算术运算和std::log2）
 * @tparam QuantT 量化目标类型（如int8_t、int6_t，必须是有符号整数类型）
 * @param[in] orig_min 原始数据最小值（输入，类型T）
 * @param[in] orig_max 原始数据最大值（输入，类型T）
 * @param[in] is_symmetric 是否使用对称量化（true=对称，false=非对称）
 * @param[out] exp2_inv 缩放因子指数（scale = 2^(-exp2_inv)），非负int32_t
 * @param[out] aligned_min 对齐后的最小值（输出，类型T）
 * @param[out] aligned_max 对齐后的最大值（输出，类型T）
 * @param[out] zp 量化零点（zero point），类型与QuantT一致，对称量化时固定为0
 * @note 1.
 * 模板约束：QuantT必须是有符号整数类型（如int8_t、int6_t），T必须是浮点类型（float/double）；
 *       2. 缩放因子严格为2的负n次方（scale ∈ (0, 1]），exp2_inv ≥ 0；
 *       3. 对称量化：zp=0，对齐范围尽可能关于原点对称，覆盖原始min/max；
 *       4. 非对称量化：zp为QuantT类型整数，对齐范围覆盖原始min/max，满足 (aligned_max -
 * aligned_min) = scale × (quant_max - quant_min)；
 *       5. 自动适配量化范围：通过std::numeric_limits<QuantT>获取quant_min/quant_max，无需手动配置；
 *       6. 异常处理：原始min ≥ orig_max、QuantT非有符号整数、T非浮点类型时抛出异常。
 */
template <typename T, typename QuantT>
inline void calibrateQuantParams(const T orig_min, const T orig_max, const bool is_symmetric,
                                 T &aligned_min, T &aligned_max, int8_t &exp2_inv, int32_t &zp,
                                 const std::string &name = "") {
    static_assert(std::is_floating_point<T>::value, "T must be float or double");

    // 量化类型的范围
    const int32_t quant_min = std::numeric_limits<QuantT>::min();
    const int32_t quant_max = std::numeric_limits<QuantT>::max();

    float scale;
    if (is_symmetric) {
        // 对称量化，zero point 固定为 0
        zp = 0;

        // 取绝对值范围，保证对称
        T abs_max = std::max(std::abs(orig_min), std::abs(orig_max));
        abs_max = std::max(abs_max, static_cast<T>(1e-9));  // 避免除零

        // scale = abs_max / quant_max => 对齐到 2^-n
        T raw_scale = abs_max / quant_max;
        // scale >= raw_scale
        exp2_inv =
            static_cast<int32_t>(std::floor(std::log2(1.0 / raw_scale)));  // floor instead of ceil
        scale = std::pow(2.0, -exp2_inv);
        aligned_max = scale * quant_max;
        aligned_min = -aligned_max;
    } else {
        // 非对称量化
        T range = orig_max - orig_min;
        range = std::max(range, static_cast<T>(1e-9));

        // 使用浮点数计算避免 int32_t 溢出（当 QuantT=int32_t 时，quant_max - quant_min 会溢出）
        T raw_scale = range / (static_cast<T>(quant_max) - static_cast<T>(quant_min));

        // scale >= raw_scale 对齐到 2^-n
        exp2_inv = static_cast<int32_t>(std::floor(std::log2(1.0 / raw_scale)));
        scale = std::pow(2.0, -exp2_inv);  // 取2的负exp2_inv次方

        aligned_min = std::floor(orig_min / scale) * scale;
        aligned_max = std::ceil(orig_max / scale) * scale;

        // 计算 zero-point
        T zp_fp = quant_min - aligned_min / scale;
        zp = std::round(zp_fp);
        //        zp = std::clamp(zp, quant_min, quant_max);
    }

    // 可选调试打印
#ifdef DEBUG
    if (!name.empty() &&
        (name == "scale_z_out" || name == "scale_r_out" || name == "scale_g_out")) {
        printf(
            "[DEBUG][QuantParam][%s] "
            "orig_min=%f, orig_max=%f, "
            "aligned_min=%f, aligned_max=%f, scale=%f, "
            "exp2_inv=%d, zp=%d, is_symmetric=%d\n",
            name.c_str(), static_cast<double>(orig_min), static_cast<double>(orig_max),
            static_cast<double>(aligned_min), static_cast<double>(aligned_max),
            static_cast<double>(scale), static_cast<int>(exp2_inv), static_cast<int>(zp),
            static_cast<int>(is_symmetric));
    }
#endif
}

template <typename QuantT>
inline QuantT quantize(float src, int8_t exp2_inv, int32_t zp) {
    // Host code: 与GPU版本保持一致，使用位运算
    // 量化公式：q = round(src / scale + zp)
    float scale;
    if (exp2_inv >= 0) {
        // scale = 2^(-exp2) = 1 / (1 << exp2)
        scale = 1.0f / static_cast<float>(1 << exp2_inv);
    } else {
        // scale = 2^(-(-x)) = 2^x = (1 << -exp2_inv)
        scale = static_cast<float>(1 << (-exp2_inv));
    }
    // 正确的量化流程：先计算 src/scale + zp，然后四舍五入
    float shifted = src / scale + static_cast<float>(zp);
    int32_t q = static_cast<int32_t>(std::round(shifted));

    constexpr int32_t qmin = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    constexpr int32_t qmax = static_cast<int32_t>(std::numeric_limits<QuantT>::max());
    q = std::clamp(q, qmin, qmax);

    return static_cast<QuantT>(q);
}

template <typename QuantT>
inline __host__ __device__ float dequantize(QuantT q, int8_t exp2_inv, int32_t zp) {
    // Host code: 与GPU版本保持一致
    int32_t v = static_cast<int32_t>(q) - zp;

    if (exp2_inv >= 0) {
        // scale = 2^(-exp2) = 1 / (1 << exp2)
        return static_cast<float>(v) / static_cast<float>(1 << exp2_inv);
    } else {
        // scale = 2^(-(-x)) = 2^x = (1 << -exp2_inv)
        return static_cast<float>(v) * static_cast<float>(1 << (-exp2_inv));
    }
}

template <typename T, typename QuantT>
inline void quantification(const T *data, QuantT *quant_data, size_t size, int8_t exp2_inv,
                           int32_t zp) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        quant_data[i] = quantize<QuantT>(data[i], exp2_inv, zp);
    }
}

template <typename T, typename QuantT>
inline void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                                     size_t channel_size, const std::vector<int8_t> &exp2_invs) {
#pragma omp parallel for
    for (int i = 0; i < channel_size; ++i) {
        // i: [0, H*3)
        const int8_t exp2_inv = exp2_invs[i];
        for (int j = 0; j < input_size; ++j) {
            // j: [0, input_size)
            const int idx = j * channel_size + i;
            // 对称量化到int8：clip到[-128,127]
            quant_data[idx] = quantize<QuantT>(src[idx], exp2_inv, 0);
        }
    }
}

namespace dev {

template <typename T, typename QuantT>
void quantification(const T *data, QuantT *quant_data, size_t size, int8_t exp2_inv, int32_t zp);

template <typename T, typename QuantT>
void dequantification(const QuantT *quant_data, T *data, size_t size, int8_t exp2_inv, int32_t zp);

// v 统一使用 int32_t 存储，内部各部分使用不同量化参数
template <typename T>
void dequantificationV(const int32_t *quant_data, T *data, int time_steps, int batch_size,
                       int hidden_size, int8_t exp2_inv_z, int32_t zp_z, int8_t exp2_inv_r,
                       int32_t zp_r, int8_t exp2_inv_g, int32_t zp_g, int8_t exp2_inv_Rh_add_br,
                       int32_t zp_Rh_add_br);

template <typename T, typename QuantT>
void quantificationPerChannel(const T *src, QuantT *quant_data, size_t input_size,
                              size_t channel_size, const dev::vector<int8_t> &exp2_invs);

template <typename T, typename QuantT>
void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                size_t channel_size, const dev::vector<int8_t> &exp2_invs);
}  // namespace dev

#include <limits>
#include <random>

// 全局随机数生成器（使用固定种子确保可复现）
inline std::mt19937 &getGlobalRng() {
    static std::mt19937 gen(42);  // 固定种子
    return gen;
}

// 设置全局随机种子
inline void setGlobalRandomSeed(unsigned int seed) { getGlobalRng().seed(seed); }

/**
 * @brief Fill a vector with random values from a normal distribution, and clamp to range.
 *
 * @param data [in/out]     The vector to fill with random values.
 * @param min_value [in]    Minimum allowed value.
 * @param max_value [in]    Maximum allowed value.
 */
inline void fillVectorWithNormalDistribution(std::vector<float> &data, float min_value,
                                             float max_value) {
    float mean = (min_value + max_value) / 2.0f;
    float stddev = (max_value - min_value) / 6.0f;  // 3σ 刚好覆盖范围

    std::mt19937 &gen = getGlobalRng();
    std::normal_distribution<float> dist(mean, stddev);

    for (auto &value : data) {
        float sample;
        // 截断采样：直到落入范围
        do {
            sample = dist(gen);
        } while (sample < min_value || sample > max_value);

        value = sample;
    }
}

// 辅助函数：量化浮点数为 INT16（对称量化）
inline int16_t quantize_coefficient_int16(float val_fp, int8_t shift_bits) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale));
    q = std::max(-32768, std::min(32767, q));
    return static_cast<int16_t>(q);
}

// 辅助函数：量化输入为 UINT16（非对称量化）
inline uint16_t quantize_input_uint16(float val_fp, int8_t shift_bits, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    q = std::max(0, std::min(65535, q));
    return static_cast<uint16_t>(q);
}

// 辅助函数：量化输入为 INT16（非对称量化，有符号版本）
inline int16_t quantize_input_int16(float val_fp, int8_t shift_bits, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    q = std::max(-32768, std::min(32767, q));
    return static_cast<int16_t>(q);
}

// 辅助函数：确定 shift_bits（根据最大值）
inline int8_t determine_shift_bits_int16(float max_val) {
    const float max_q = 32767.0f;
    if (max_val < 1e-9f) return 0;
    float scale = max_val / max_q;
    int8_t shift_bits = static_cast<int8_t>(std::ceil(-std::log2(scale)));
    return std::max(static_cast<int8_t>(0), shift_bits);
}

// 辅助函数：量化浮点数为 INT8（对称量化）
inline int8_t quantize_coefficient_int8(float val_fp, int8_t shift_bits) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale));
    q = std::max(-128, std::min(127, q));
    return static_cast<int8_t>(q);
}

// 辅助函数：量化输入为 UINT8（非对称量化）
inline uint8_t quantize_input_uint8(float val_fp, int8_t shift_bits, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    q = std::max(0, std::min(255, q));
    return static_cast<uint8_t>(q);
}

// 辅助函数：量化输入为 INT8（非对称量化，有符号版本）
inline int8_t quantize_input_int8(float val_fp, int8_t shift_bits, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    q = std::max(-128, std::min(127, q));
    return static_cast<int8_t>(q);
}

// 辅助函数：确定 shift_bits（根据最大值，INT8 版本）
inline int8_t determine_shift_bits_int8(float max_val) {
    const float max_q = 127.0f;
    if (max_val < 1e-9f) return 0;
    float scale = max_val / max_q;
    int8_t shift_bits = static_cast<int8_t>(std::ceil(-std::log2(scale)));
    return std::max(static_cast<int8_t>(0), shift_bits);
}

void init_sigmoid_z_lut_int8(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                             float x_min = -6.0f, float x_max = 6.0f);

void init_sigmoid_r_lut_int8(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                             float x_min = -6.0f, float x_max = 6.0f);

void init_tanh_lut_int8(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                        float x_min = -6.0f, float x_max = 6.0f);

void init_tanh_lut_int16(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                         float x_min = -6.0f, float x_max = 6.0f);

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - r 门）
void init_sigmoid_r_lut_int16(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                              float x_min = -6.0f, float x_max = 6.0f);

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - z 门）
void init_sigmoid_z_lut_int16(int8_t shift_bits_x, int32_t zp_x, int8_t shift_bits_y, int32_t zp_y,
                              float x_min = -6.0f, float x_max = 6.0f);

/**
 * 通用(仅host)scale/zp 计算函数
 * @param x_dev  -- 设备端输入数据指针
 * @param size_per_step -- 每步输入长度
 * @param steps -- 步数
 * @param use_symmetric -- 是否对称量化
 * @param name -- 调试信息
 */
template <typename T, typename QuantT>
inline void calculateScalePerSteps(const T *x_dev, const int size_per_step, const int steps,
                                   const bool use_symmetric, int8_t &exp2_inv, int32_t &zp,
                                   const std::string &name = "") {
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

    T res_min = min[0];
    T res_max = max[0];
    for (int t = 1; t < steps; ++t) {
        //        // TODO: 修改为原来的方法
        //        res_min = 0.9 * res_min + 0.1 * min[t];
        //        res_max = 0.9 * res_max + 0.1 * max[t];
        res_min = std::min(res_min, min[t]);
        res_max = std::max(res_max, max[t]);
    }

    calibrateQuantParams<T, QuantT>(res_min, res_max, use_symmetric, res_min, res_max, exp2_inv, zp,
                                    name);
}

template <typename T, typename QuantT>
inline std::vector<int8_t> calculateScalesPerChannels(const T *W_dev, int channel_size,
                                                      int input_size,
                                                      const std::string &name = "") {
    // 列主序排列

    std::vector<T> W_host = d2h(W_dev, channel_size * input_size);

    std::vector<int8_t> exp2_inv_per_channels(channel_size);
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
}

template <typename T, typename QuantT>
inline void calculateScale(const std::vector<T> &data_host, const bool use_symmetric,
                           int8_t &exp2_inv, int32_t &zp, const std::string &name = "") {
    T min_val = data_host[0];
    T max_val = data_host[0];
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
    for (int i = 1; i < data_host.size(); ++i) {
        const T val = data_host[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    T min_new = min_val;
    T max_new = max_val;
    calibrateQuantParams<T, QuantT>(min_val, max_val, use_symmetric, min_new, max_new, exp2_inv, zp,
                                    name);
}

template <typename T, typename QuantT>
inline void calculateScale(const T *data_dev, const size_t size, const bool use_symmetric,
                           int8_t &exp2_inv, int32_t &zp, const std::string &name = "") {
    std::vector<T> data_host = d2h(data_dev, size);
    calculateScale<T, QuantT>(data_host, use_symmetric, exp2_inv, zp, name);
}

inline void printParms(const GRUQuantitativeParameters &quant_parms) {
    printf("GRUQuantitativeParameters (量化参数):\n");
    printf("  hidden_ = %d\n", quant_parms.hidden_);
    printf("  exp2_inv_x_ = %d, zp_x_ = %d\n", static_cast<int>(quant_parms.exp2_inv_x_),
           quant_parms.zp_x_);
    printf("  exp2_inv_h_ = %d, zp_h_ = %d\n", static_cast<int>(quant_parms.exp2_inv_h_),
           quant_parms.zp_h_);

    printf("  exp2_inv_W_ (size %zu): ", quant_parms.exp2_inv_W_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_W_.size() && i < 5; ++i) {
        printf("%d ", static_cast<int>(quant_parms.exp2_inv_W_[i]));
    }
    if (quant_parms.exp2_inv_W_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_R_ (size %zu): ", quant_parms.exp2_inv_R_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_R_.size() && i < 5; ++i) {
        printf("%d ", static_cast<int>(quant_parms.exp2_inv_R_[i]));
    }
    if (quant_parms.exp2_inv_R_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_bx_ (size %zu): ", quant_parms.exp2_inv_bx_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_bx_.size() && i < 5; ++i) {
        printf("%d ", static_cast<int>(quant_parms.exp2_inv_bx_[i]));
    }
    if (quant_parms.exp2_inv_bx_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_br_ (size %zu): ", quant_parms.exp2_inv_br_.size());
    for (size_t i = 0; i < quant_parms.exp2_inv_br_.size() && i < 5; ++i) {
        printf("%d ", static_cast<int>(quant_parms.exp2_inv_br_[i]));
    }
    if (quant_parms.exp2_inv_br_.size() > 8) printf("...");
    printf("\n");

    printf("  exp2_inv_Wx_ = %d, zp_Wx_ = %d \n", static_cast<int>(quant_parms.exp2_inv_Wx_),
           quant_parms.zp_Wx_);
    printf("  exp2_inv_Rh_ = %d, zp_Rh_ = %d \n", static_cast<int>(quant_parms.exp2_inv_Rh_),
           quant_parms.zp_Rh_);
    printf("  exp2_inv_z_pre_ = %d, zp_z_pre_ = %d \n",
           static_cast<int>(quant_parms.exp2_inv_z_pre_), quant_parms.zp_z_pre_);
    printf("  exp2_inv_r_pre_ = %d, zp_r_pre_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_r_pre_), quant_parms.zp_r_pre_);
    printf("  exp2_inv_g_pre_ = %d, zp_g_pre_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_g_pre_), quant_parms.zp_g_pre_);
    printf("  exp2_inv_z_out_ = %d, zp_z_out_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_z_out_), quant_parms.zp_z_out_);
    printf("  exp2_inv_r_out_ = %d, zp_r_out_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_r_out_), quant_parms.zp_r_out_);
    printf("  exp2_inv_g_out_ = %d, zp_g_out_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_g_out_), quant_parms.zp_g_out_);
    printf("  exp2_inv_Rh_add_br_ = %d, zp_Rh_add_br_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_Rh_add_br_), quant_parms.zp_Rh_add_br_);
    printf("  exp2_inv_rRh_ = %d, zp_rRh_ = %d\n", static_cast<int>(quant_parms.exp2_inv_rRh_),
           quant_parms.zp_rRh_);
    printf("  exp2_inv_new_contrib_ = %d, zp_new_contrib_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_new_contrib_), quant_parms.zp_new_contrib_);
    printf("  exp2_inv_old_contrib_ = %d, zp_old_contrib_ = %d\n",
           static_cast<int>(quant_parms.exp2_inv_old_contrib_), quant_parms.zp_old_contrib_);
}
