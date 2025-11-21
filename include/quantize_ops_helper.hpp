#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>

#include "devVector.h"

template<typename T>
struct GRUQuantitativeParametersInCalibration {
  dev::vector<T> z_pres_;
  dev::vector<T> r_pres_;
  dev::vector<T> g_pres_;
  dev::vector<T> one_minus_update_;
  dev::vector<T> new_contrib_;
  dev::vector<T> old_contrib_;
};

// GRU 量化参数结构体：存储GRU网络量化过程中所有定点化/反量化所需的参数
// 核心约束：所有缩放因子均以「2的负n次方」形式存储，exp2_inv_xxx 表示缩放因子 scale = 2^(-exp2_inv_xxx)
// zp_xxx 表示量化零点（zero point），用于浮点数与整数的映射：量化值 q = round(x / scale + zp)，反量化 x = (q - zp) * scale
struct GRUQuantitativeParameters {
  int hidden_; // channel = hidden * 3
  int32_t exp2_inv_x_;
  int32_t zp_x_;
  int32_t exp2_inv_h_;
  int32_t zp_h_;

  std::vector<int32_t> exp2_inv_W_; // size = hidden * 3. per-channel (每个输出通道一个scale，即W的每一行一个scale)
  std::vector<int32_t> exp2_inv_R_; // size = hidden * 3. per-channel (每个输出通道一个scale，即R的每一行一个scale)

  int32_t exp2_inv_Wx_;
  int32_t zp_Wx_;
  int32_t exp2_inv_Rh_;
  int32_t zp_Rh_;

  std::vector<int32_t> exp2_inv_bx_;
  std::vector<int32_t> exp2_inv_br_;
  dev::vector<int32_t> exp2_inv_bx_dev_;
  dev::vector<int32_t> exp2_inv_br_dev_;

  int32_t exp2_inv_z_pre_;
  int32_t zp_z_pre_;
  int32_t exp2_inv_r_pre_;
  int32_t zp_r_pre_;
  int32_t exp2_inv_g_pre_;
  int32_t zp_g_pre_;

  int32_t exp2_inv_z_out_;
  int32_t zp_z_out_;
  int32_t exp2_inv_r_out_;
  int32_t zp_r_out_;
  int32_t exp2_inv_g_out_;
  int32_t zp_g_out_;

  int32_t exp2_inv_Rh_add_br_;
  int32_t zp_Rh_add_br_;
  int32_t exp2_inv_rRh_;
  int32_t zp_rRh_;

  int32_t exp2_inv_one_minus_update_;
  int32_t zp_one_minus_update_;
  int32_t exp2_inv_new_contrib_;
  int32_t zp_new_contrib_;
  int32_t exp2_inv_old_contrib_;
  int32_t zp_old_contrib_;
};

struct QuantGRUReScale {
  int32_t zp_x_;
  int32_t zp_h_;

  dev::vector<int32_t> n_W_mul_x_div_Wx_; // size = hidden * 3
  dev::vector<float> scale_W_mul_x_div_Wx_;
  int32_t zp_Wx_;
  dev::vector<int32_t> n_R_mul_h_div_Rh_; // size = hidden * 3
  dev::vector<float> scale_R_mul_h_div_Rh_;
  int32_t zp_Rh_;

  // z门
  int32_t zp_z_pre_;
  int32_t zp_z_out_;
  int32_t exp2_inv_Wx_div_z_pre_;
  int32_t exp2_inv_Wx_div_z_;
  int32_t exp2_inv_Rh_div_z_pre_;
  int32_t exp2_inv_Rh_div_z_;
  dev::vector<int32_t> n_bx_div_z_;
  dev::vector<float> scale_bx_div_z_;
  dev::vector<int32_t> n_br_div_z_;
  dev::vector<float> scale_br_div_z_;

  // r门
  int32_t zp_r_pre_;
  int32_t zp_r_out_;
  int32_t exp2_inv_Wx_div_r_pre_; // n5
  int32_t exp2_inv_Rh_div_r_pre_; // n6
  dev::vector<int32_t> n_bx_div_r_;
  dev::vector<float> scale_bx_div_r_;
  dev::vector<int32_t> n_br_div_r_;
  dev::vector<float> scale_br_div_r_;

  // New Gate
  int32_t zp_g_pre_;
  int32_t zp_g_out_;
  int32_t n_Rh_div_Rh_add_br_;
  int32_t exp2_inv_Rh_div_Rh_add_br_;
  dev::vector<int32_t> n_br_div_Rh_add_br_; // br 是 per-channel
  dev::vector<float> scale_br_div_Rh_add_br_;
  int32_t zp_Rh_add_br_;
  int32_t n_r_mul_Rh_add_br_div_rRh_; // n9
  int32_t exp2_inv_r_out_mul_h_div_rRh_; // S9
  int32_t zp_rRh_;
  int32_t n_Wx_div_g_pre_; // n10
  int32_t exp2_inv_Wx_div_g_pre_; // S10
  int32_t n_rRh_div_g_pre_; // n11
  int32_t exp2_inv_rRh_div_g_pre_; // S11
  dev::vector<int32_t> exp2_inv_bx_div_g_pre_;
  dev::vector<float> scale_bx_div_g_pre_;

  // h_new
  int32_t one_div_one_minus_update_;
  int32_t n_z_out_div_one_minus_update_; // n12
  int32_t exp2_inv_z_out_div_one_minus_update_; // S12
  int32_t zp_one_minus_update_;

  int32_t zp_new_contrib_;
  int32_t n_one_minus_update_mul_g_div_new_contrib_; // n13
  int32_t exp2_inv_one_minus_update_mul_g_div_new_contrib_; // S13
  int32_t zp_old_contrib_;
  int32_t n_z_mul_h_div_old_contrib_; // n14
  int32_t exp2_inv_z_mul_h_div_old_contrib_; // S14
  int32_t n_new_contrib_div_h_; // n15
  int32_t exp2_inv_new_contrib_div_h_; // S15
  int32_t n_old_contrib_div_h_; // n16
  int32_t exp2_inv_old_contrib_div_h_; // S16


  //test
  GRUQuantitativeParameters test;
};
//
//struct QuantGRUScales {
//  int hidden_;
//  int32_t exp2_inv_x_;
//  int32_t zp_x_;
//  int32_t exp2_inv_h_;
//  int32_t zp_h_;
//  int32_t exp2_inv_bx_;
//  int32_t exp2_inv_br_;
//  std::vector<float> Wx; // size = hidden. per-channel
//  std::vector<float> Rh; // size = hidden
//
//  std::vector<float> Wx_add_bx; // size = hidden
//  std::vector<float> Rh_add_br; // size = hidden
//
//  std::vector<float> z_pre; // size = hidden
//  std::vector<float> r_pre; // size = hidden
//  std::vector<float> g_pre; // size = hidden
//
//  std::vector<float> z_out; // size = hidden
//  std::vector<float> r_out; // size = hidden
//  std::vector<float> g_out; // size = hidden
//};

inline __host__ __device__ float dequant_from_exp2(int q, int32_t exp2_inv, int zp) {
    int32_t v = q - zp;

    if (exp2_inv >= 0) {
        // scale = 2^(-exp2) = 1 / (1 << exp2)
        return static_cast<float>(v) / static_cast<float>(1 << exp2_inv);
    } else {
        // scale = 2^(-(-x)) = 2^x = (1 << -exp2_inv)
        return static_cast<float>(v) * static_cast<float>(1 << (-exp2_inv));
    }
}

template<typename T, typename QuantT>
inline QuantT quant_from_exp2(T src, int32_t exp2_inv, int zp) {
    float scaled;
    if (exp2_inv >= 0) {
        scaled = src * static_cast<float>(1 << exp2_inv);
    } else {
        scaled = src / static_cast<float>(1 << (-exp2_inv));
    }

    int32_t q = static_cast<int32_t>(std::round(scaled)) + zp;
    constexpr int32_t qmin = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    constexpr int32_t qmax = static_cast<int32_t>(std::numeric_limits<QuantT>::max());
    q = std::clamp(q, qmin, qmax);
    return static_cast<QuantT>(q);
}

void generate_int8_lut_from_exp2_inv(int32_t exp2_inv_z_pre,
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
                                     int32_t zp_g_out);


__host__ __device__ __forceinline__ int32_t rshift_round(int32_t x, int n) {
    if (n <= 0) return x << (-n);

    const int32_t offset = 1 << (n - 1);
    if (x >= 0) {
        return (x + offset) >> n;
    } else {
        // 对负数要改成向零舍入：
        return -((-x + offset) >> n);
    }
}

/**
 * @brief 在 GPU 上将 float 数据量化为 int8
 * @tparam QuantT       目标量化类型（int8_t 或 int16_t）
 * @tparam use_inv_scale 是否使用 inv_scale（乘法而非除法）
 * @tparam symmetric    是否使用对称量化（zero_point=0）
 * @tparam clamp    是否使用饱和处理 (对bias不处理)
 * @param src_dev    输入 float 指针（GPU 内存）
 * @param dst_dev    输出 int8 指针（GPU 内存）
 * @param size       元素数量
 * @param scale      量化 scale
 * @param zero_point 量化 zero_point（非对称量化有效）
 */
template<typename QuantT, bool use_inv_scale, bool symmetric, bool clamp = true>
void quantizeFloatToInt(const float *src_dev,
                        QuantT *dst_dev,
                        uint32_t size,
                        float scale,
                        int32_t zero_point = 0);

/**
 * @brief 在 GPU 上将 float 数据量化为 int8/int16（支持每个时间步独立 scale）
 * @tparam QuantT       目标量化类型（int8_t 或 int16_t）
 * @tparam use_inv_scale 是否使用 inv_scale（乘法而非除法）
 * @tparam symmetric    是否使用对称量化（zero_point=0）
 * @tparam clamp        是否使用饱和处理
 * @param src_dev       输入 float 指针（GPU 内存）
 * @param dst_dev       输出 int8/int16 指针（GPU 内存）
 * @param size          总元素数量
 * @param scale_per_t   每个时间步的量化 scale 数组（GPU 内存，长度为 time_steps）
 * @param zero_point    每个时间步的量化 zero_point（非对称量化有效）
 * @param time_step_size 每个时间步的元素数（例如 batch_size * input_dim）
 */
template<typename QuantT, bool use_inv_scale, bool symmetric, bool clamp = true>
void quantizeFloatToIntPerStep(const float *src_dev,
                               QuantT *dst_dev,
                               size_t size,
                               const float *scale_per_t,
                               const int32_t *zero_point_per_t,
                               int time_step_size);

template<typename T>
void computeWeightSumMulzp(
    const T *W_q,// [out_dim, in_dim] 权重量化矩阵
    int32_t *weight_sum,// [out_dim] 输出数组
    int zp,
    const int32_t *__restrict__ n, // n为: scale_W * scale_x / scale_Wx ≈ 2^-n. per-channel
    int out_dim,// 输出通道数 (M)
    int in_dim,// 输入通道数 (K)
    cudaStream_t stream = 0);

void applyZeroPointCompensation2D(
    int32_t *Y_int32,
    const int32_t *weight_sum,
    const int32_t *x_zp,
    int out_dim,
    int batch_size,
    cudaStream_t stream = 0);

/**
 * @brief 从 GPU 上的量化数据计算 scale（使用最大最小值）
 *
 * @tparam QuantT         量化类型（int8_t 或 int16_t）
 * @param h_dev           [in] GPU 上的量化数据指针
 * @param size            [in] 数据元素数量
 * @param scale           [out] 输出的 scale
 * @param zero_point      [out] 输出的 zero_point
 * @param symmetric       [in] 是否使用对称量化
 * @param stream          [in] CUDA stream
 */
template<typename QuantT>
void calculateScaleZeroPointFromDevice(
    const QuantT *h_dev,
    size_t size,
    float &scale,
    int32_t &zero_point,
    bool symmetric = true,
    cudaStream_t stream = 0);

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
 * @note 如果 min_val 和 max_val 都有效（min_val < max_val），则跳过数据扫描，直接使用提供的值计算scale和zero_point
 */
template<typename QuantT>
void calculateScaleZeroPoint(
    const float *data,
    size_t size,
    float &scale,
    int32_t &zero_point,
    bool symmetric = true,
    float *min_val = nullptr,
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
template<typename QuantT>
void calculateScaleZeroPointFromFloatDevice(
    const float *data_dev,
    size_t size,
    float &scale,
    int32_t &zero_point,
    bool symmetric = true,
    cudaStream_t stream = 0,
    float *min_val = nullptr,
    float *max_val = nullptr);

//template<typename T>
//T findMaxValueFromDev(const T *dev_data, size_t size);
//
//template<typename T>
//T findMinValueFromDev(const T *dev_data, size_t size);

/**
 * @brief 使用 (M, shift) 参数将量化值反量化为浮点数
 * @tparam QuantT      量化类型（int8_t / int16_t / int32_t）
 * @param quant_data   输入量化数据指针
 * @param size         数据元素数量
 * @param M            定点缩放系数（整数）
 * @param shift        缩放右移位数
 * @param dequant_data 输出反量化后的 float 数组
 */
template<typename QuantT>
void dequantizeTensorFixedPoint(const QuantT *quant_data,
                                size_t size,
                                int32_t M,
                                int shift,
                                float *dequant_data) {
    // 计算等效的scale（float），只在CPU上调试时使用
    const float scale = static_cast<float>(M) / static_cast<float>(1 << shift);

    for (size_t i = 0; i < size; ++i) {
        const int32_t q = static_cast<int32_t>(quant_data[i]);
        dequant_data[i] = q * scale;
    }
}

/**
 * @brief 完整的反量化函数（支持对称和非对称量化）
 * @param quant_data 量化后的int8数据
 * @param dequant_data 反量化后的float数据输出
 * @param size 数据数量
 * @param scale 缩放因子
 * @param zero_point 零点（对称量化为0，非对称量化通常不为0）
 */
template<typename QuantT>
inline void dequantizeTensor(const QuantT *quant_data, float *dequant_data,
                             int size, float scale, int32_t zero_point) {
    for (int i = 0; i < size; ++i) {
        dequant_data[i] = static_cast<float>(quant_data[i] - zero_point) * scale;
    }
}

// 定义常量
constexpr int32_t Q15_ONE = 32768;
constexpr int32_t ALPHA_Q15 = 29491; // 0.9 * 32768
constexpr int32_t INV_QMAX = (1 << 15) / 127; // 257 in Q15

// 输入: 上一步scale参数 (M_prev, shift_prev)
// 输入: 当前步隐藏态整数张量 h_t_int[]
// 输出: 更新后的scale参数 (M_new, shift_new)
inline void updateHScaleInt8(const int8_t *h_t, size_t size,
                             int32_t &M_prev, int &shift_prev) {
    // 1. 求当前步最大值
    int max_abs = 0;
    for (size_t i = 0; i < size; ++i)
        max_abs = std::max(max_abs, abs((int) h_t[i]));

    // 2. ratio 定点化 (Q15)
    int32_t ratio_q15 = (max_abs * INV_QMAX); // Q15 格式

    // 3. EMA 更新 (Q15)
    static int32_t s_prev_q15 = Q15_ONE; // 初始scale比例=1.0
    int32_t s_new_q15 = (ALPHA_Q15 * s_prev_q15 +
                         (Q15_ONE - ALPHA_Q15) * ratio_q15 + (1 << 14)) >> 15;
    s_prev_q15 = s_new_q15;

    // 4. 更新 M (scale整数因子)
    M_prev = (M_prev * s_new_q15 + (1 << 14)) >> 15;

    // shift_prev 可视范围动态调整（或保持不变）
}

/**
 * @brief 计算缩放因子 S 对应的右移位数 n（使 S ≈ 2^(-n)）
 * @param S 待转换的浮点数缩放因子（必须 > 0）
 * @param min_n 允许的最小右移位数（默认 0，避免左移溢出）
 * @param max_n 允许的最大右移位数（默认 30，适配 32 位整数运算）
 * @return 最优右移位数 n（int 类型，在 [min_n, max_n] 范围内）
 * @throws std::invalid_argument 若 S ≤ 0，抛出异常
 */
inline int32_t calculate_right_shift_bits(float S,
                                          const std::string &name = "",
                                          int32_t min_n = 0,
                                          int32_t max_n = 30) {
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
    int32_t n_candidate;
    if (n_theory >= 0) {
        n_candidate = static_cast<int32_t>(n_theory + 0.5f);  // 标准四舍五入
    } else {
        n_candidate = static_cast<int32_t>(n_theory - 0.5f);  // 负数的四舍五入
    }


    // 步骤3：边界裁剪，确保n在合理范围（避免移位溢出或数值归零）
    int32_t n = std::max(min_n, std::min(n_candidate, max_n));

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
    assert(calculate_right_shift_bits(8.0f, "", 3, 10) == 3);  // 理论值-3，被裁剪到3
    std::cout << "min_n=3约束生效 ✓" << std::endl;

    // 测试最大边界
    assert(calculate_right_shift_bits(0.0001f, "", 0, 10) == 10);  // 理论值~13，被裁剪到10
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
        {1.0f, 1.0f},      // 精确匹配
        {0.5f, 1.0f},      // 精确匹配
        {0.25f, 1.0f},     // 精确匹配
        {0.75f, 35.0f},    // 0.75 ≈ 2^0=1, 误差33% 在可接受范围
        {0.375f, 35.0f},   // 0.375 ≈ 2^-1=0.5, 误差33% 在可接受范围
        {0.625f, 25.0f},   // 0.625 ≈ 2^-1=0.5, 误差20% 在可接受范围
        {0.999f, 1.0f},    // 非常接近1，应该误差很小
        {1.001f, 1.0f},    // 非常接近1，应该误差很小
        {0.1f, 50.0f},     // 0.1 ≈ 2^-3=0.125, 误差25% 或 2^-4=0.0625, 误差37.5%
        {0.9f, 12.0f},     // 0.9 ≈ 2^0=1, 误差11%
    };

    int passed = 0;
    int total = 0;

    for (const auto &tc : test_cases) {
        int n = calculate_right_shift_bits(tc.S);
        float S_approx = powf(2.0f, -static_cast<float>(n));
        float relative_error = (fabsf(tc.S - S_approx) / tc.S) * 100.0f;

        std::cout << "S=" << tc.S << " -> n=" << n
                  << " (近似S=" << S_approx << ")"
                  << " 误差=" << relative_error << "%";

        if (relative_error <= tc.max_acceptable_error) {
            std::cout << " ✓" << std::endl;
            passed++;
        } else {
            std::cout << " ✗ 最大可接受误差=" << tc.max_acceptable_error << "%" << std::endl;
        }
        total++;
    }

    std::cout << "精度测试通过率: " << passed << "/" << total << " ("
              << (passed * 100 / total) << "%)" << std::endl;
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
        n = 30; // 最大右移 30 位（避免溢出）
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
//inline int32_t calculate_one_over_Somu(float S_omu, float tolerance = 1e-6f) {
//    // 1. 基础合法性检查：S_omu 必须为正
//    if (S_omu <= 0.0f) {
//        throw std::invalid_argument("S_omu must be positive! Current value: " + std::to_string(S_omu));
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
//                                                        "Nearest 2^(-m): " + std::to_string(S_omu_approx) + ", "
//                                                                                                            "Error: " +
//            std::to_string(absolute_error)
//        );
//    }
//
//    // 5. 计算 1/S_omu = 2^m（整数结果），并检查溢出
//    if (m < 0) {
//        throw std::invalid_argument("m = " + std::to_string(m) + " is negative! S_omu is too large (exceeds 1.0f)");
//    }
//    if (m > 30) { // 2^30 = 1073741824，2^31 会溢出 int32_t（最大值 2147483647）
//        throw std::overflow_error("m = " + std::to_string(m) + " is too large! 2^m exceeds int32_t limit");
//    }
//
//    int32_t one_over_Somu = 1 << m; // 等价于 2^m，用左移实现整数乘法（无浮点运算）
//    return one_over_Somu;
//}

inline int32_t selectBestExp2InvSym(const float orig_min, const float orig_max,
                                    int32_t quant_max) {
    const float half_range = std::max(std::abs(orig_min), std::abs(orig_max));
    const float safe_half = std::max(half_range, 1e-9f);

    int32_t exp2_inv0 = static_cast<int32_t>(std::ceil(std::log2(quant_max / safe_half)));
    exp2_inv0 = std::max(exp2_inv0, 0);

    float best_mse = std::numeric_limits<float>::max();
    int32_t best_exp2 = exp2_inv0;

    for (int32_t candidate = exp2_inv0 - 1; candidate <= exp2_inv0 + 1; ++candidate) {
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
                     (deq_max - orig_max) * (deq_max - orig_max)) * 0.5f;

        if (mse < best_mse) {
            best_mse = mse;
            best_exp2 = candidate;
        }
    }

    return best_exp2;
}

inline int32_t selectBestExp2InvAsym(const float orig_min,
                                     const float orig_max,
                                     int32_t quant_min,
                                     int32_t quant_max) {
    const float EPS = 1e-12f;
    const float orig_range = std::max(orig_max - orig_min, EPS);
    const int32_t quant_range = quant_max - quant_min;

    // 初始候选
    int32_t exp2_inv0 = static_cast<int32_t>(std::ceil(std::log2(orig_range / quant_range)));
    exp2_inv0 = std::max(exp2_inv0, 0);

    float best_mse = std::numeric_limits<float>::max();
    int32_t best_exp2 = exp2_inv0;

    for (int32_t candidate = exp2_inv0 - 1; candidate <= exp2_inv0 + 1; ++candidate) {
        if (candidate < 0) continue;

        const float scale = std::pow(2.f, -static_cast<float>(candidate));
        float aligned_min = std::floor(orig_min / scale) * scale;
        float aligned_max = aligned_min + scale * quant_range;

        // 估算零点
        float zp_float = (-aligned_min) / scale + quant_min;
        int32_t zp = static_cast<int32_t>(std::llround(zp_float));
        zp = std::clamp(zp, quant_min, quant_max);

        // 简单估算 MSE: 取 min/max 两点
        float deq_min = (quant_min - zp) * scale + aligned_min;
        float deq_max = (quant_max - zp) * scale + aligned_min;
        float mse = ((deq_min - orig_min) * (deq_min - orig_min) +
                     (deq_max - orig_max) * (deq_max - orig_max)) * 0.5f;

        if (mse < best_mse) {
            best_mse = mse;
            best_exp2 = candidate;
        }
    }
    return best_exp2;
}

#define DEBUG true

/**
 * @brief 模板化量化参数计算函数：支持任意量化类型（int8/int6等）和输入范围类型，对齐2的负n次方缩放因子
 * @tparam T 输入范围数据类型（如float、double，需支持算术运算和std::log2）
 * @tparam QuantT 量化目标类型（如int8_t、int6_t，必须是有符号整数类型）
 * @param[in] orig_min 原始数据最小值（输入，类型T）
 * @param[in] orig_max 原始数据最大值（输入，类型T）
 * @param[in] is_symmetric 是否使用对称量化（true=对称，false=非对称）
 * @param[out] exp2_inv 缩放因子指数（scale = 2^(-exp2_inv)），非负int32_t
 * @param[out] aligned_min 对齐后的最小值（输出，类型T）
 * @param[out] aligned_max 对齐后的最大值（输出，类型T）
 * @param[out] zp 量化零点（zero point），类型与QuantT一致，对称量化时固定为0
 * @note 1. 模板约束：QuantT必须是有符号整数类型（如int8_t、int6_t），T必须是浮点类型（float/double）；
 *       2. 缩放因子严格为2的负n次方（scale ∈ (0, 1]），exp2_inv ≥ 0；
 *       3. 对称量化：zp=0，对齐范围尽可能关于原点对称，覆盖原始min/max；
 *       4. 非对称量化：zp为QuantT类型整数，对齐范围覆盖原始min/max，满足 (aligned_max - aligned_min) = scale × (quant_max - quant_min)；
 *       5. 自动适配量化范围：通过std::numeric_limits<QuantT>获取quant_min/quant_max，无需手动配置；
 *       6. 异常处理：原始min ≥ orig_max、QuantT非有符号整数、T非浮点类型时抛出异常。
 */
template<typename T, typename QuantT>
inline void calibrateQuantParams(
    const T orig_min,
    const T orig_max,
    const bool is_symmetric,
    T &aligned_min,
    T &aligned_max,
    int32_t &exp2_inv,
    int32_t &zp,
    const std::string &name = "") {

    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(std::is_signed_v<QuantT> && std::is_integral_v<QuantT>, "QuantT must be signed integer");

    constexpr int32_t quant_min = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    constexpr int32_t quant_max = static_cast<int32_t>(std::numeric_limits<QuantT>::max());
    // difference (quant_max - quant_min)
    constexpr int32_t quant_range = quant_max - quant_min;

    if (orig_min >= orig_max - static_cast<T>(1e-12)) {
//        throw std::invalid_argument(
//            name +
//            "Invalid original range: orig_min (" + std::to_string(orig_min) +
//            ") must be less than orig_max (" + std::to_string(orig_max) + ")");
        return;
    }

    // small epsilon for zero-checks
    const T EPS = static_cast<T>(1e-12);
    T aligned_scale = static_cast<T>(1.0);

    if (is_symmetric) {
        // symmetric quantization: zp == 0
        zp = static_cast<QuantT>(0);

        // half_range = max(|min|, |max|)
        const T half_range = std::max(std::abs(orig_min), std::abs(orig_max));
        const T safe_half_range = (half_range <= EPS) ? static_cast<T>(1.0) : half_range;

//        // Want scale = 2^-n such that scale * quant_max >= half_range
//        // => 2^-n >= half_range / quant_max  => n <= log2(quant_max / half_range)
//        // choose largest scale (smallest n) satisfying => n = floor(log2(quant_max / half_range))
        {
            const T ratio = static_cast<T>(quant_max) / safe_half_range;
            const T log2_val = std::log2(std::max(ratio, static_cast<T>(1e-30))); // guard
            int32_t n = static_cast<int32_t>(std::floor(log2_val));
            n = std::max(n, static_cast<int32_t>(0));
            exp2_inv = n;
        }

//        exp2_inv = selectBestExp2InvSym(orig_min, orig_max, quant_max);

        aligned_scale = std::pow(static_cast<T>(2.0), -static_cast<T>(exp2_inv));
        // preserve symmetry: aligned_max >= half_range, aligned_min = -aligned_max
        aligned_max = aligned_scale * static_cast<T>(quant_max);
        if (aligned_max < safe_half_range) {
            // should not happen because of floor logic, but guard anyway by reducing exp2_inv
            // if happens, decrease exp2_inv until coverage satisfied (rare)
            while (exp2_inv > 0 && (aligned_scale * static_cast<T>(quant_max) < safe_half_range)) {
                --exp2_inv;
                aligned_scale = std::pow(static_cast<T>(2.0), -static_cast<T>(exp2_inv));
                aligned_max = aligned_scale * static_cast<T>(quant_max);
            }
        }
        // final symmetric aligned range
        aligned_max = std::max(aligned_max, safe_half_range);
        aligned_min = -aligned_max;

        // Finally ensure we cover original min/max (preserve symmetry)
        if (aligned_min > orig_min) aligned_min = -aligned_max; // keep symmetric
        if (aligned_max < orig_max) aligned_max = aligned_max; // already symmetric
    } else {
        // asymmetric quantization
        T orig_range = orig_max - orig_min;
        orig_range = (orig_range <= EPS) ? static_cast<T>(1.0) : orig_range;

        // Want scale = 2^-n such that scale * quant_range >= orig_range
        // => 2^-n >= orig_range / quant_range => n <= log2(quant_range / orig_range)
        // choose largest scale satisfying => n = floor(log2(quant_range / orig_range))
        {
            const T ratio = static_cast<T>(quant_range) / orig_range;
            const T log2_val = std::log2(std::max(ratio, static_cast<T>(1e-30)));
            int32_t n = static_cast<int32_t>(std::floor(log2_val));
            n = std::max(n, static_cast<int32_t>(0));
            exp2_inv = n;
        }
//        exp2_inv = selectBestExp2InvAsym(orig_min, orig_max, quant_min, quant_max);

        aligned_scale = std::pow(static_cast<T>(2.0), -static_cast<T>(exp2_inv));
        // align min down to multiple of scale
        aligned_min = std::floor(orig_min / aligned_scale) * aligned_scale;
        aligned_max = aligned_min + aligned_scale * static_cast<T>(quant_range);

        // if due to numerical error we still don't cover orig_max, expand by one step
        if (aligned_max < orig_max - EPS) {
            aligned_min -= aligned_scale;
            aligned_max += aligned_scale;
        }

        // zero point: quant_min + round( (-aligned_min) / scale )
        const T zp_float = (-aligned_min) / aligned_scale;
        int32_t zp_int = static_cast<int32_t>(std::llround(zp_float)); // round to nearest
        // convert to quant space by offsetting quant_min
        zp_int = quant_min + zp_int;
        // clamp
        zp_int = std::clamp(zp_int, quant_min, quant_max);
        zp = static_cast<QuantT>(zp_int);

        // small numerical cleanup
        aligned_min = std::round(aligned_min * static_cast<T>(1e9)) / static_cast<T>(1e9);
        aligned_max = std::round(aligned_max * static_cast<T>(1e9)) / static_cast<T>(1e9);
    }

    if (DEBUG && name != "exp2_inv_W" && name != "exp2_inv_R" && name != "scale_bx" && name != "scale_br") {
        printf("%s : min_val = %.15f, max_val = %.15f, min_new = %.15f, max_new = %.15f, exp2_inv = %d, zp = %d\n",
               name.c_str(),
               orig_min,
               orig_max,
               aligned_min,
               aligned_max,
               exp2_inv,
               zp);
    }
}

//template<typename T, typename QuantT>
//inline void calibrateQuantParams(
//    const T orig_min,
//    const T orig_max,
//    const bool is_symmetric,
//    T &aligned_min,
//    T &aligned_max,
//    int32_t &exp2_inv,
//    int32_t &zp,
//    const std::string &name = "")
//{
//    // 1. 确定量化范围
//    int32_t quant_max = static_cast<int32_t>(std::numeric_limits<QuantT>::max());
////    if constexpr (std::is_same<QuantT, int8_t>::value) {
////        quant_max = 127;
////    } else if constexpr (std::is_same<QuantT, int6_t>::value) {
////        quant_max = 31;
////    } else {
////        static_assert(std::is_same<QuantT, int8_t>::value || std::is_same<QuantT, int6_t>::value,
////                      "Unsupported QuantT type!");
////    }
//
//    T min_val = orig_min;
//    T max_val = orig_max;
//
//    // 对称量化处理
//    if (is_symmetric) {
//        T abs_max = std::max(std::abs(min_val), std::abs(max_val));
//        min_val = -abs_max;
//        max_val = abs_max;
//    }
//
//    // 防止极小范围
//    T range = std::max(max_val - min_val, static_cast<T>(1e-9));
//
//    // 2. 找到最接近 2^-n 的步长
//    exp2_inv = static_cast<int32_t>(std::round(std::log2(range / quant_max)));
//    T scale = std::pow(2.0, exp2_inv);
//
//    // 3. 尝试向内和向外对齐，选择误差最小的
//    T aligned_min_in = std::ceil(min_val / scale) * scale;
//    T aligned_max_in = std::floor(max_val / scale) * scale;
//    T error_in = (aligned_max_in - aligned_min_in - range);
//
//    T aligned_min_out = std::floor(min_val / scale) * scale;
//    T aligned_max_out = std::ceil(max_val / scale) * scale;
//    T error_out = (aligned_max_out - aligned_min_out - range);
//
//    if (std::abs(error_in) < std::abs(error_out)) {
//        aligned_min = aligned_min_in;
//        aligned_max = aligned_max_in;
//    } else {
//        aligned_min = aligned_min_out;
//        aligned_max = aligned_max_out;
//    }
//
//    // 4. 计算 zero point
//    if (is_symmetric) {
//        zp = 0;
//    } else {
//        zp = static_cast<int32_t>(std::round(-aligned_min / scale));
//        zp = std::min(std::max(0, zp), quant_max);
//    }
//
////    if (!name.empty()) {
////        std::cout << "[" << name << "] "
////                  << "aligned_min=" << aligned_min
////                  << ", aligned_max=" << aligned_max
////                  << ", exp2_inv=" << exp2_inv
////                  << ", zp=" << zp << "\n";
////    }
//}


//template<typename QuantT>
//inline QuantT quantize(
//    float src,
//    int32_t exp2_inv,  // scale = 2^(-exp2_inv)
//    int32_t zp         // zero point
//) {
//    // 使用 exp2_inv 重建 scale：
//    // src / scale = src * (1 << exp2_inv) if exp2_inv >= 0
//    // src / scale = src / (1 << -exp2_inv) if exp2_inv < 0
//
//    float scaled;
//
//    if (exp2_inv >= 0) {
//        scaled = src * static_cast<float>(1 << exp2_inv);
//    } else {
//        scaled = src / static_cast<float>(1 << (-exp2_inv));
//    }
//
//    // q = round(src/scale) + zp
//    int32_t q = static_cast<int32_t>(std::round(scaled)) + zp;
//
//    // clamp to quant type range
//    constexpr int32_t qmin = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
//    constexpr int32_t qmax = static_cast<int32_t>(std::numeric_limits<QuantT>::max());
//    q = std::clamp(q, qmin, qmax);
//
//    return static_cast<QuantT>(q);
//}
//
//template<typename QuantT>
//inline float dequantize(
//    QuantT q,
//    int32_t exp2_inv,  // scale = 2^(-exp2_inv)
//    int32_t zp         // zero point
//) {
//    // remove zp first
//    int32_t q_int = static_cast<int32_t>(q) - zp;
//
//    float scale;
//
//    // build scale value
//    if (exp2_inv >= 0) {
//        // scale = 1 / (2^exp)
//        scale = 1.0f / static_cast<float>(1 << exp2_inv);
//    } else {
//        // scale = 2^(-exp2_inv) = (1 << -exp2_inv)
//        scale = static_cast<float>(1 << (-exp2_inv));
//    }
//
//    return q_int * scale;
//}

template<typename QuantT>
inline QuantT quantize(float src, int32_t exp2_inv, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(exp2_inv));
    int32_t q = static_cast<int32_t>(std::round(src / scale)) + zp;

    constexpr int32_t qmin = static_cast<int32_t>(std::numeric_limits<QuantT>::min());
    constexpr int32_t qmax = static_cast<int32_t>(std::numeric_limits<QuantT>::max());
    q = std::clamp(q, qmin, qmax);

    return static_cast<QuantT>(q);
}

template<typename QuantT>
inline float dequantize(QuantT q, int32_t exp2_inv, int32_t zp) {
    float scale = std::pow(2.0f, -static_cast<float>(exp2_inv));
    return (static_cast<int32_t>(q) - zp) * scale;
}


namespace unit_testing {
// ==========================
// Helper
// ==========================
template<typename T>
bool almost_equal(T a, T b, T eps = static_cast<T>(1e-6)) {
    return std::abs(a - b) <= eps;
}

template<typename T, typename QuantT>
void run_one_test(
    const std::string &name,
    T orig_min,
    T orig_max,
    bool is_symmetric) {
    int32_t exp2_inv = -1;
    T aligned_min = 0, aligned_max = 0;
    int32_t zp = 0;

    std::cout << "-------------------------------------------------\n";
    std::cout << "Test: " << name << "\n";
    std::cout << "orig_min=" << orig_min << ", orig_max=" << orig_max
              << ", is_symmetric=" << is_symmetric << "\n";

    calibrateQuantParams<T, QuantT>(
        orig_min, orig_max, is_symmetric,
        aligned_min, aligned_max, exp2_inv, zp
    );

    const int32_t quant_min = std::numeric_limits<QuantT>::min();
    const int32_t quant_max = std::numeric_limits<QuantT>::max();
    const int32_t quant_range = quant_max - quant_min;

    T scale = std::pow((T) 2, (T) -exp2_inv);

    std::cout << "exp2_inv = " << exp2_inv << "\n";
    std::cout << "scale    = " << scale << "\n";
    std::cout << "aligned_min = " << aligned_min << "\n";
    std::cout << "aligned_max = " << aligned_max << "\n";
    std::cout << "zp = " << (int) zp << "\n";

    // -------------------------------
    // 自动断言
    // -------------------------------

    // 1. 必须覆盖原始范围
//    assert(aligned_min <= orig_min + 1e-6);
//    assert(aligned_max >= orig_max - 1e-6);

    // 2. scale 必须严格是 2^-n
    {
        T expected = std::pow((T) 2, (T) -exp2_inv);
        assert(almost_equal(scale, expected, (T) 1e-12));
    }

    // 3. 对称量化检查
    if (is_symmetric) {
        // zp 必须是 0
        assert(zp == 0);

        // 必须保持对称（允许极小浮点误差）
        assert(almost_equal(aligned_max, -aligned_min, (T) 1e-6));
    }

    // 4. 非对称量化检查
    if (!is_symmetric) {
        // zp 在范围内
        assert((int) zp >= quant_min && (int) zp <= quant_max);

        // (aligned_max - aligned_min) == scale * quant_range
        T expect_span = scale * quant_range;
//        assert(almost_equal(aligned_max - aligned_min, expect_span, (T) 1e-5));
    }

    std::cout << "✓ PASS\n";
    std::cout << "\n";
}

// ==========================
// Main Entry: 多组测试
// ==========================
inline void quantizationTest() {

    std::cout << "===== Running Quantization Tests =====\n";

    using T = float;

    // ---------- Symmetric int8 ----------
    run_one_test<T, int8_t>("Symm-int8: balanced range", -40.0f, 40.0f, true);
    run_one_test<T, int8_t>("Symm-int8: unbalanced range", -30.0f, 80.0f, true);
    run_one_test<T, int8_t>("Symm-int8: very small", -0.01f, 0.02f, true);

    // ---------- Asymmetric int8 ----------
    run_one_test<T, int8_t>("Asym-int8: general", -1.2f, 2.7f, false);
    run_one_test<T, int8_t>("Asym-int8: positive only", 0.1f, 8.5f, false);
    run_one_test<T, int8_t>("Asym-int8: small range", 0.001f, 0.005f, false);

    // ---------- Symmetric int16 ----------
    run_one_test<T, int16_t>("Symm-int16: wide", -5000.f, 9000.f, true);

    // ---------- Asymmetric int16 ----------
    run_one_test<T, int16_t>("Asym-int16: general", -100.f, 30000.f, false);

    std::cout << "===== All tests passed! =====\n";

}

inline void testCalibrateQuantParams() {
    std::cout << "Running unit tests for calibrateQuantParams...\n";

    // ====== 测试 1：对称量化，正负正常范围 ======
    {
        float min_val = -1.0f, max_val = 2.0f;
        float aligned_min, aligned_max;
        int32_t exp2_inv, zp;

        calibrateQuantParams<float, int8_t>(min_val, max_val, true,
                                            aligned_min, aligned_max,
                                            exp2_inv, zp, "test1");

        assert(aligned_min <= 0.0f);
        assert(aligned_max >= 0.0f);
        assert(zp == 0);
    }

    // ====== 测试 2：非对称量化，正负范围 ======
    {
        float min_val = -0.5f, max_val = 1.5f;
        float aligned_min, aligned_max;
        int32_t exp2_inv, zp;

        calibrateQuantParams<float, int8_t>(min_val, max_val, false,
                                            aligned_min, aligned_max,
                                            exp2_inv, zp, "test2");

        assert(aligned_min <= min_val + 1e-6);
        assert(aligned_max >= max_val - 1e-6);
        assert(zp >= 0 && zp <= 127);
    }

    // ====== 测试 3：非常小的范围 ======
    {
        float min_val = 1e-8f, max_val = 2e-8f;
        float aligned_min, aligned_max;
        int32_t exp2_inv, zp;

        calibrateQuantParams<float, int8_t>(min_val, max_val, true,
                                            aligned_min, aligned_max,
                                            exp2_inv, zp, "test3");

        assert(aligned_min <= 0.0f);
        assert(aligned_max >= 0.0f);
    }

    // ====== 测试 4：极大范围 ======
    {
        float min_val = -1e6f, max_val = 1e6f;
        float aligned_min, aligned_max;
        int32_t exp2_inv, zp;

        calibrateQuantParams<float, int8_t>(min_val, max_val, true,
                                            aligned_min, aligned_max,
                                            exp2_inv, zp, "test4");

        assert(aligned_min <= -1e6f);
        assert(aligned_max >= 1e6f);
        assert(zp == 0);
    }

    // ====== 测试 5：min=max ======
    {
        float min_val = 0.5f, max_val = 0.5f;
        float aligned_min, aligned_max;
        int32_t exp2_inv, zp;

        calibrateQuantParams<float, int8_t>(min_val, max_val, false,
                                            aligned_min, aligned_max,
                                            exp2_inv, zp, "test5");

        assert(aligned_max - aligned_min >= 0);
    }

    // ====== 测试 7：负 min，正 max，非常小 ======
    {
        float min_val = -1e-7f, max_val = 1e-7f;
        float aligned_min, aligned_max;
        int32_t exp2_inv, zp;

        calibrateQuantParams<float, int8_t>(min_val, max_val, false,
                                            aligned_min, aligned_max,
                                            exp2_inv, zp, "test7");

        assert(aligned_min <= min_val + 1e-9f);
        assert(aligned_max >= max_val - 1e-9f);
        assert(zp >= 0 && zp <= 127);
    }

    std::cout << "All tests passed!\n";
}
}
