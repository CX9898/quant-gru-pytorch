#pragma once

#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "devVector.h"

// GRU 量化参数结构体：存储GRU网络量化过程中所有定点化/反量化所需的参数
// 核心约束：所有缩放因子均以「2的负n次方」形式存储，exp2_inv_xxx 表示缩放因子 scale =
// 2^(-exp2_inv_xxx) zp_xxx 表示量化零点（zero point），用于浮点数与整数的映射：量化值 q = round(x /
// scale + zp)，反量化 x = (q - zp) * scale
struct GRUQuantitativeParameters {
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

    int8_t exp2_inv_one_minus_update_;
    int32_t zp_one_minus_update_;
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
    int32_t one_div_one_minus_update_;
    int8_t n_z_out_div_one_minus_update_;         // n12
    int8_t exp2_inv_z_out_div_one_minus_update_;  // S12
    int32_t zp_one_minus_update_;

    int32_t zp_new_contrib_;
    int8_t n_one_minus_update_mul_g_div_new_contrib_;         // n13
    int8_t exp2_inv_one_minus_update_mul_g_div_new_contrib_;  // S13
    int32_t zp_old_contrib_;
    int8_t n_z_mul_h_div_old_contrib_;         // n14
    int8_t exp2_inv_z_mul_h_div_old_contrib_;  // S14
    int8_t n_new_contrib_div_h_;               // n15
    int8_t exp2_inv_new_contrib_div_h_;        // S15
    int8_t n_old_contrib_div_h_;               // n16
    int8_t exp2_inv_old_contrib_div_h_;        // S16

    // test
    GRUQuantitativeParameters test;
};

template <typename QuantT>
void GruQuantInit(
    const int time_steps, const int batch_size, const int input_size, const int hidden_size,
    const float *W,   // 输入到隐藏层的权重矩阵. [input_size, hidden_size * 3] 对应三个门
    const float *R,   // 隐藏层到隐藏层的循环权重矩阵
    const float *bx,  // 输入偏置项（input bias），来自输入路径
    const float *br,  // 循环偏置项（recurrent bias），来自循环路径
    const float *x,   // 输入序列张量
    QuantT *W_quant, QuantT *R_quant, int32_t *bx_quant, int32_t *br_quant, QuantT *x_quant,
    const GRUQuantitativeParameters &gruRescaleParams);

void generate_int8_lut_from_exp2_inv(int8_t exp2_inv_z_pre, int32_t zp_z_pre, int8_t exp2_inv_z_out,
                                     int32_t zp_z_out, int8_t exp2_inv_r_pre, int32_t zp_r_pre,
                                     int8_t exp2_inv_r_out, int32_t zp_r_out, int8_t exp2_inv_g_pre,
                                     int32_t zp_g_pre, int8_t exp2_inv_g_out, int32_t zp_g_out);

// 生成分段线性量化表（基于exp2_inv参数，支持模板类型）
// x_min 和 x_max 从量化参数（exp2_inv_pre 和 zp_pre）自动计算：
//   - scale = 2^(-exp2_inv_pre) = 1.0f / (1 << exp2_inv_pre)
//   - x_min = (quant_min - zp_pre) * scale
//   - x_max = (quant_max - zp_pre) * scale
// 其中 quant_min 和 quant_max 由量化类型 QuantT 决定
template <typename QuantT>
void generate_piecewise_linear_lut_from_exp2_inv(int8_t exp2_inv_z_pre, int32_t zp_z_pre,
                                                 int8_t exp2_inv_z_out, int32_t zp_z_out,
                                                 int8_t exp2_inv_r_pre, int32_t zp_r_pre,
                                                 int8_t exp2_inv_r_out, int32_t zp_r_out,
                                                 int8_t exp2_inv_g_pre, int32_t zp_g_pre,
                                                 int8_t exp2_inv_g_out, int32_t zp_g_out);

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

template <typename T>
void computeWeightSumMulzp(
    const T *W_q,         // [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,  // [out_dim] 输出数组
    int zp,
    const int8_t *__restrict__ n,  // n为: scale_W * scale_x / scale_Wx ≈ 2^-n. per-channel
    int out_dim,                   // 输出通道数 (M)
    int in_dim,                    // 输入通道数 (K)
    cudaStream_t stream = 0);

void applyZeroPointCompensation2D(int32_t *Y_int32, const int32_t *weight_sum, const int32_t *x_zp,
                                  int out_dim, int batch_size, cudaStream_t stream = 0);

/**
 * @brief 从浮点数据计算量化参数 scale 和 zero_point（通用函数）
 *
 * 量化公式：
 * - 对称量化：scale = max(|min_val|, |max_val|) / qmax, zero_point = 0
 * - 非对称量化：scale = (max_val - min_val) / (qmax - qmin),
 *               zero_point = round(qmin - min_val / scale)
 *
 * 量化过程：q = round(x / scale + zero_point)
 * 反量化过程：x = (q - zero_point) * scale
 *
 * @tparam QuantT         目标量化类型（int8_t 或 int16_t）
 * @param data            [in] 输入浮点数据指针（CPU内存）
 * @param size            [in] 数据元素数量
 * @param scale           [out] 输出的量化scale参数
 * @param zero_point      [out] 输出的量化zero_point参数
 * @param symmetric       [in] 是否使用对称量化（默认true）
 *                          - true: 对称量化，zero_point固定为0，scale基于绝对值最大值计算
 *                          - false: 非对称量化，zero_point可非零，scale基于实际范围计算
 * @param min_val         [in/out] 可选，输入时指定最小值（跳过计算），输出时返回计算的最小值
 * @param max_val         [in/out] 可选，输入时指定最大值（跳过计算），输出时返回计算的最大值
 *
 * @note 如果 min_val 和 max_val 都有效（min_val <
 * max_val），则跳过数据扫描，直接使用提供的值计算scale和zero_point
 */
template <typename QuantT>
void calculateScaleZeroPoint(const float *data, size_t size, float &scale, int32_t &zero_point,
                             bool symmetric = true, float *min_val = nullptr,
                             float *max_val = nullptr);

/**
 * @brief 从GPU上的浮点数据计算量化参数 scale 和 zero_point
 *
 * 功能与 calculateScaleZeroPoint 相同，但数据位于GPU内存中。
 * 函数会自动将数据拷贝到CPU进行最大最小值计算。
 *
 * @tparam QuantT         目标量化类型（int8_t 或 int16_t）
 * @param data_dev        [in] 输入浮点数据指针（GPU内存）
 * @param size            [in] 数据元素数量
 * @param scale           [out] 输出的量化scale参数
 * @param zero_point      [out] 输出的量化zero_point参数
 * @param symmetric       [in] 是否使用对称量化（默认true）
 * @param stream          [in] CUDA stream，用于异步拷贝（默认0）
 * @param min_val         [in/out] 可选，输入时指定最小值，输出时返回计算的最小值
 * @param max_val         [in/out] 可选，输入时指定最大值，输出时返回计算的最大值
 */
template <typename QuantT>
void calculateScaleZeroPointFromFloatDevice(const float *data_dev, size_t size, float &scale,
                                            int32_t &zero_point, bool symmetric = true,
                                            cudaStream_t stream = 0, float *min_val = nullptr,
                                            float *max_val = nullptr);

// template<typename T>
// T findMaxValueFromDev(const T *dev_data, size_t size);
//
// template<typename T>
// T findMinValueFromDev(const T *dev_data, size_t size);

/**
 * @brief 计算缩放因子 S 对应的右移位数 n（使 S ≈ 2^(-n)）
 * @param S 待转换的浮点数缩放因子（必须 > 0）
 * @param min_n 允许的最小右移位数（默认 0，避免左移溢出）
 * @param max_n 允许的最大右移位数（默认 30，适配 32 位整数运算）
 * @return 最优右移位数 n（int8_t 类型，在 [min_n, max_n] 范围内）
 * @throws std::invalid_argument 若 S ≤ 0，抛出异常
 */
inline int8_t calculate_right_shift_bits(float S, const std::string &name = "", int8_t min_n = 0,
                                         int8_t max_n = 30) {
    // 输入合法性检查：缩放因子必须为正
    if (S <= 0.0f) {
        throw std::invalid_argument(
            name + ": Scale factor S must be greater than 0. Current value: " + std::to_string(S));
    }

    // 步骤1：计算理论最优n（n = -log2(S)）
    // log2f 是单精度浮点数对数函数，效率高于 log2（double 类型）
    float log2_S = log2f(S);
    float n_theory = -log2_S;

    //    // 步骤2：四舍五入到最近整数（C++11+ 支持 roundf 单精度 rounding）
    //    int32_t n_candidate = static_cast<int32_t>(roundf(n_theory));

    // 步骤2：改进的四舍五入策略 - 考虑相对误差
    int8_t n_candidate;
    if (n_theory >= 0) {
        n_candidate = static_cast<int8_t>(n_theory + 0.5f);  // 标准四舍五入
    } else {
        n_candidate = static_cast<int8_t>(n_theory - 0.5f);  // 负数的四舍五入
    }

    // 步骤3：边界裁剪，确保n在合理范围（避免移位溢出或数值归零）
    int8_t n = std::max(min_n, std::min(n_candidate, max_n));

    //    // （可选）验证近似效果（调试用，发布时可注释）
    //    float S_approx = powf(2.0f, -static_cast<float>(n));
    //    float relative_error = (fabsf(S - S_approx) / S) * 100.0f;
    //    std::cout << "[DEBUG] S: " << S << " | Approx S=2^(-" << n << "): " << S_approx
    //              << " | Relative Error: " << relative_error << "%" << std::endl;

    return n;
}

inline void test_basic_cases() {
    std::cout << "=== 基础测试 ===" << std::endl;

    // 测试 1: S = 1.0，应该返回 n=0 (2^0 = 1)
    assert(calculate_right_shift_bits(1.0f) == 0);
    std::cout << "S=1.0 -> n=0 ✓" << std::endl;

    // 测试 2: S = 0.5，应该返回 n=1 (2^-1 = 0.5)
    assert(calculate_right_shift_bits(0.5f) == 1);
    std::cout << "S=0.5 -> n=1 ✓" << std::endl;

    // 测试 3: S = 0.25，应该返回 n=2 (2^-2 = 0.25)
    assert(calculate_right_shift_bits(0.25f) == 2);
    std::cout << "S=0.25 -> n=2 ✓" << std::endl;

    // 测试 4: S = 0.125，应该返回 n=3 (2^-3 = 0.125)
    assert(calculate_right_shift_bits(0.125f) == 3);
    std::cout << "S=0.125 -> n=3 ✓" << std::endl;

    std::cout << "\n=== 边界约束测试 ===" << std::endl;

    // 测试最小边界
    assert(calculate_right_shift_bits(8.0f, "", static_cast<int8_t>(3), static_cast<int8_t>(10)) ==
           3);  // 理论值-3，被裁剪到3
    std::cout << "min_n=3约束生效 ✓" << std::endl;

    // 测试最大边界
    assert(calculate_right_shift_bits(0.0001f, "", static_cast<int8_t>(0),
                                      static_cast<int8_t>(10)) == 10);  // 理论值~13，被裁剪到10
    std::cout << "max_n=10约束生效 ✓" << std::endl;

    // 测试四舍五入
    assert(calculate_right_shift_bits(0.375f) == 1);  // 理论值1.415 → 四舍五入到1
    assert(calculate_right_shift_bits(0.625f) == 1);  // 理论值0.678 → 四舍五入到1
    std::cout << "四舍五入逻辑正确 ✓" << std::endl;

    std::cout << "\n=== 改进的精度验证测试 ===" << std::endl;

    struct TestCase {
        float S;
        float max_acceptable_error;  // 可接受的最大相对误差百分比
    };

    std::vector<TestCase> test_cases = {
        {1.0f, 1.0f},     // 精确匹配
        {0.5f, 1.0f},     // 精确匹配
        {0.25f, 1.0f},    // 精确匹配
        {0.75f, 35.0f},   // 0.75 ≈ 2^0=1, 误差33% 在可接受范围
        {0.375f, 35.0f},  // 0.375 ≈ 2^-1=0.5, 误差33% 在可接受范围
        {0.625f, 25.0f},  // 0.625 ≈ 2^-1=0.5, 误差20% 在可接受范围
        {0.999f, 1.0f},   // 非常接近1，应该误差很小
        {1.001f, 1.0f},   // 非常接近1，应该误差很小
        {0.1f, 50.0f},    // 0.1 ≈ 2^-3=0.125, 误差25% 或 2^-4=0.0625, 误差37.5%
        {0.9f, 12.0f},    // 0.9 ≈ 2^0=1, 误差11%
    };

    int passed = 0;
    int total = 0;

    for (const auto &tc : test_cases) {
        int8_t n = calculate_right_shift_bits(tc.S);
        float S_approx = powf(2.0f, -static_cast<float>(n));
        float relative_error = (fabsf(tc.S - S_approx) / tc.S) * 100.0f;

        std::cout << "S=" << tc.S << " -> n=" << static_cast<int>(n) << " (近似S=" << S_approx
                  << ")"
                  << " 误差=" << relative_error << "%";

        if (relative_error <= tc.max_acceptable_error) {
            std::cout << " ✓" << std::endl;
            passed++;
        } else {
            std::cout << " ✗ 最大可接受误差=" << tc.max_acceptable_error << "%" << std::endl;
        }
        total++;
    }

    std::cout << "精度测试通过率: " << passed << "/" << total << " (" << (passed * 100 / total)
              << "%)" << std::endl;
}

/**
 * @brief 计算 1/S 的近似值，用整数右移位实现
 * @param S 输入缩放因子 (必须 > 0)
 * @return 近似计算 1/S 的整数值
 */
inline int32_t calculate_one_over_S(float S) {
    if (S <= 0.0f) {
        throw std::invalid_argument("S must be positive! Current value: " + std::to_string(S));
    }

    // 计算理论右移位数：n = -log2(S)
    float n_theory = -log2f(S);

    // 四舍五入到最近整数
    int32_t n = static_cast<int32_t>(roundf(n_theory));

    // 边界检查：确保 n 在合理范围内
    // n < 0 意味着 S > 1.0，此时 1/S < 1，不适合用右移位
    // n ≥ 32 会导致未定义行为（对于 32 位整数）
    if (n < 0) {
        n = 0;  // 最小右移 0 位（即不移位）
    } else if (n > 30) {
        n = 30;  // 最大右移 30 位（避免溢出）
    }

    // 计算 1/S ≈ 2^n
    // 注意：这里返回的是 2^n，不是右移位数 n
    int32_t result = 1 << n;

    return result;
}

inline int32_t calculate_one_over_Somu(float S_omu, float tolerance = 1e-6f) {
    if (S_omu <= 0.0f) {
        throw std::invalid_argument("S_omu must be positive!");
    }

    // 直接计算 1/S_omu 并四舍五入到整数
    float one_over = 1.0f / S_omu;
    int32_t result = static_cast<int32_t>(roundf(one_over));

    // 可选：检查结果是否在 int32_t 范围内
    if (result <= 0) {
        throw std::overflow_error("1/S_omu is too large or invalid");
    }

    return result;
}

///**
// * @brief 验证 S_omu 是否为 2的负幂次浮点数，并计算 1/S_omu（整数结果）
// * @param S_omu 输入的缩放因子（浮点数，需符合量化工程约定：S_omu = 2^(-m), m为整数）
// * @param tolerance 浮点数精度容差（默认 1e-6，应对浮点存储微小误差）
// * @return 1/S_omu 的整数结果（int32_t 类型，因量化场景中 m 通常≤30，2^30≈1e9，适配32位整数）
// * @throws std::invalid_argument 若 S_omu 不是 2的负幂次，或 S_omu ≤0
// */
// inline int32_t calculate_one_over_Somu(float S_omu, float tolerance = 1e-6f) {
//    // 1. 基础合法性检查：S_omu 必须为正
//    if (S_omu <= 0.0f) {
//        throw std::invalid_argument("S_omu must be positive! Current value: " +
//        std::to_string(S_omu));
//    }
//
//    // 2. 计算理论 m 值：由 S_omu = 2^(-m) → m = -log2(S_omu)
//    float log2_Somu = log2f(S_omu);
//    float m_theory = -log2_Somu;
//
//    // 3. 验证 m 是否为整数（核心：S_omu 必须是 2的负幂次）
//    int32_t m = static_cast<int32_t>(roundf(m_theory)); // 四舍五入到最近整数
//    float S_omu_approx = powf(2.0f, -static_cast<float>(m)); // 由 m 反推理论 S_omu
//
//    // 4. 检查实际 S_omu 与理论值的误差是否在容差内
//    float absolute_error = fabsf(S_omu - S_omu_approx);
//    if (absolute_error > tolerance) {
//        throw std::invalid_argument(
//            "S_omu is not a negative power of 2! "
//            "Current S_omu: " + std::to_string(S_omu) + ", "
//                                                        "Nearest 2^(-m): " +
//                                                        std::to_string(S_omu_approx) + ", "
//                                                                                                            "Error: " +
//            std::to_string(absolute_error)
//        );
//    }
//
//    // 5. 计算 1/S_omu = 2^m（整数结果），并检查溢出
//    if (m < 0) {
//        throw std::invalid_argument("m = " + std::to_string(m) + " is negative! S_omu is too large
//        (exceeds 1.0f)");
//    }
//    if (m > 30) { // 2^30 = 1073741824，2^31 会溢出 int32_t（最大值 2147483647）
//        throw std::overflow_error("m = " + std::to_string(m) + " is too large! 2^m exceeds int32_t
//        limit");
//    }
//
//    int32_t one_over_Somu = 1 << m; // 等价于 2^m，用左移实现整数乘法（无浮点运算）
//    return one_over_Somu;
//}

template <typename T>
inline void calculateNewRangeForExp2(const T orig_min, const T orig_max, const T quant_min,
                                     const T quant_max, T &new_min, T &new_max) {
    const T range = orig_max - orig_min;
    const T quant_range = quant_max - quant_min;
    const T scale = range / quant_range;
    // Force scale to the nearest 2's negative power, i.e., scale = 2^-n
    // Calculate n = round(-log2(scale))
    T log2_scale = std::log2(scale);
    int32_t n = static_cast<int32_t>(std::ceil(-log2_scale));
}

inline int8_t selectBestExp2InvSym(const float orig_min, const float orig_max, int32_t quant_max) {
    const float half_range = std::max(std::abs(orig_min), std::abs(orig_max));
    const float safe_half = std::max(half_range, 1e-9f);

    int8_t exp2_inv0 = static_cast<int8_t>(std::ceil(std::log2(quant_max / safe_half)));
    exp2_inv0 = std::max(exp2_inv0, static_cast<int8_t>(0));

    float best_mse = std::numeric_limits<float>::max();
    int8_t best_exp2 = exp2_inv0;

    for (int8_t candidate = exp2_inv0 - 1; candidate <= exp2_inv0 + 1; ++candidate) {
        if (candidate < 0) continue;
        float scale = std::pow(2.f, -static_cast<float>(candidate));
        float aligned_max = scale * quant_max;
        float aligned_min = -aligned_max;

        // 简单估算 MSE：用 orig_min 和 orig_max 量化 + 反量化
        float qmin = std::round(orig_min / scale);
        float qmax = std::round(orig_max / scale);
        float deq_min = qmin * scale;
        float deq_max = qmax * scale;

        float mse = ((deq_min - orig_min) * (deq_min - orig_min) +
                     (deq_max - orig_max) * (deq_max - orig_max)) *
                    0.5f;

        if (mse < best_mse) {
            best_mse = mse;
            best_exp2 = candidate;
        }
    }

    return best_exp2;
}

// #define DEBUG true

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
    static_assert(std::is_signed<QuantT>::value, "QuantT must be a signed integer type");

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

        T raw_scale = range / (quant_max - quant_min);

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
    if (!name.empty() && name == "scale_x") {
        std::cout << "[QuantParam][" << name << "] "
                  << "orig_min=" << orig_min << ", orig_max=" << orig_max
                  << ", aligned_min=" << aligned_min << ", aligned_max=" << aligned_max
                  << ", scale=" << scale << ", exp2_inv=" << exp2_inv << ", zp=" << zp
                  << ", is_symmetric=" << is_symmetric << std::endl;
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

template <typename T, typename QuantT>
void quantificationV(const T *data, QuantT *quant_data, int time_steps, int batch_size,
                     int hidden_size, int8_t exp2_inv_z, int32_t zp_z, int8_t exp2_inv_r,
                     int32_t zp_r, int8_t exp2_inv_g, int32_t zp_g, int8_t exp2_inv_Rh_add_br,
                     int32_t zp_Rh_add_br);

template <typename T, typename QuantT>
void dequantificationV(const QuantT *quant_data, T *data, int time_steps, int batch_size,
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

    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
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
inline uint16_t quantize_input_uint16(float val_fp, int8_t shift_bits, int16_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    q = std::max(0, std::min(65535, q));
    return static_cast<uint16_t>(q);
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
inline uint8_t quantize_input_uint8(float val_fp, int8_t shift_bits, int8_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(shift_bits));
    int32_t q = static_cast<int32_t>(std::round(val_fp / scale + static_cast<float>(zp)));
    q = std::max(0, std::min(255, q));
    return static_cast<uint8_t>(q);
}

// 辅助函数：确定 shift_bits（根据最大值，INT8 版本）
inline int8_t determine_shift_bits_int8(float max_val) {
    const float max_q = 127.0f;
    if (max_val < 1e-9f) return 0;
    float scale = max_val / max_q;
    int8_t shift_bits = static_cast<int8_t>(std::ceil(-std::log2(scale)));
    return std::max(static_cast<int8_t>(0), shift_bits);
}

void init_sigmoid_z_lut_int8(int8_t shift_bits_x, int8_t zp_x, int8_t shift_bits_y, int8_t zp_y,
                             float x_min = -6.0f, float x_max = 6.0f);

void init_sigmoid_r_lut_int8(int8_t shift_bits_x, int8_t zp_x, int8_t shift_bits_y, int8_t zp_y,
                             float x_min = -6.0f, float x_max = 6.0f);

void init_tanh_lut_int8(int8_t shift_bits_x, int8_t zp_x, int8_t shift_bits_y, int8_t zp_y,
                        float x_min = -6.0f, float x_max = 6.0f);

void init_tanh_lut_int16(int8_t shift_bits_x, int16_t zp_x, int8_t shift_bits_y, int16_t zp_y,
                         float x_min = -6.0f, float x_max = 6.0f);

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - r 门）
void init_sigmoid_r_lut_int16(int8_t shift_bits_x, int16_t zp_x, int8_t shift_bits_y, int16_t zp_y,
                              float x_min = -6.0f, float x_max = 6.0f);

// 初始化 LUT（将数据复制到 CUDA 常量内存，INT16 版本 - z 门）
void init_sigmoid_z_lut_int16(int8_t shift_bits_x, int16_t zp_x, int8_t shift_bits_y, int16_t zp_y,
                              float x_min = -6.0f, float x_max = 6.0f);
