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

struct GRUQuantitativeParameters {
  int hidden_;
  float scale_x_;
  int32_t zp_x_;
  float scale_h_;
  int32_t zp_h_;

  std::vector<float> scale_W_; // size = hidden * 3. per-channel (每个输出通道一个scale，即W的每一行一个scale)
  std::vector<float> scale_R_; // size = hidden * 3. per-channel (每个输出通道一个scale，即R的每一行一个scale)

  float scale_Wx_;
  int32_t zp_Wx_;
  float scale_Rh_;
  int32_t zp_Rh_;

  std::vector<float> scale_bx_;
  std::vector<float> scale_br_;

  float scale_z_pre_;
  int32_t zp_z_pre_;
  float scale_r_pre_;
  int32_t zp_r_pre_;
  float scale_g_pre_;
  int32_t zp_g_pre_;

  float scale_z_out_;
  int32_t zp_z_out_;
  float scale_r_out_;
  int32_t zp_r_out_;
  float scale_g_out_;
  int32_t zp_g_out_;

  float scale_Rh_add_br_;
  int32_t zp_Rh_add_br_;
  float scale_rRh_;
  int32_t zp_rRh_;

  float scale_one_minus_update_;
  int32_t zp_one_minus_update_;
  float scale_new_contrib_;
  int32_t zp_new_contrib_;
  float scale_old_contrib_;
  int32_t zp_old_contrib_;
};

struct QuantGRUReScale {
  int32_t zp_x_;
  int32_t zp_h_;

  dev::vector<int32_t> n_W_mul_x_div_Wx_; // size = hidden
  dev::vector<float> scale_W_mul_x_div_Wx_;
  int32_t zp_Wx_;
  dev::vector<int32_t> n_R_mul_h_div_Rh_; // size = hidden
  dev::vector<float> scale_R_mul_h_div_Rh_;
  int32_t zp_Rh_;

  // z门
  int32_t zp_z_pre_;
  int32_t zp_z_out_;
  int32_t n_Wx_div_z_; // size = hidden
  float scale_Wx_div_z_;
  int32_t n_Rh_div_z_;
  float scale_Rh_div_z_;
  dev::vector<int32_t> n_bx_div_z_;
  dev::vector<float> scale_bx_div_z_;
  dev::vector<int32_t> n_br_div_z_;
  dev::vector<float> scale_br_div_z_;

  // r门
  int32_t zp_r_pre_;
  int32_t zp_r_out_;
  int32_t n_Wx_div_r_;
  float scale_Wx_div_r_;
  int32_t n_Rh_div_r_;
  float scale_Rh_div_r_;
  dev::vector<int32_t> n_bx_div_r_;
  dev::vector<float> scale_bx_div_r_;
  dev::vector<int32_t> n_br_div_r_;
  dev::vector<float> scale_br_div_r_;

  // New Gate
  int32_t zp_g_pre_;
  int32_t zp_g_out_;
  int32_t n_Rh_div_Rh_add_br_;
  float scale_Rh_div_Rh_add_br_;
  dev::vector<int32_t> n_br_div_Rh_add_br_; // br 是 per-channel
  dev::vector<float> scale_br_div_Rh_add_br_;
  int32_t zp_Rh_add_br_;
  int32_t n_r_mul_h_div_rRh_; // n9
  float scale_r_out_mul_h_div_rRh_; // S9
  int32_t zp_rRh_;
  int32_t n_Wx_div_g_pre_; // n10
  float scale_Wx_div_g_pre_; // S10
  int32_t n_rRh_div_g_pre_; // n11
  float scale_rRh_div_g_pre_; // S11
  dev::vector<int32_t> n_bx_to_g_;
  dev::vector<float> scale_bx_div_g_pre_;

  // h_new
  int32_t n_z_out_div_one_minus_update__; // n12
  float scale_z_out_div_one_minus_update_; // S12
  int32_t c12_; // 不需要加zp_omu

  int32_t zp_new_contrib_;
  int32_t n_one_minus_update_mul_g_div_new_contrib_; // n13
  float scale_one_minus_update_mul_g_div_new_contrib_; // S13
  int32_t zp_old_contrib_;
  int32_t n_z_mul_h_div_old_contrib_; // n14
  float scale_z_mul_h_div_old_contrib_; // S14
  int32_t n_new_contrib_div_h_; // n15
  float scale_new_contrib_div_h_; // S15
  int32_t n_old_contrib_div_h_; // n16
  float scale_old_contrib_div_h_; // S16
  int32_t c15_; // c15

};
//
//struct QuantGRUScales {
//  int hidden_;
//  float scale_x_;
//  int32_t zp_x_;
//  float scale_h_;
//  int32_t zp_h_;
//  float scale_bx_;
//  float scale_br_;
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

void generate_int8_lut(float scale_z_pre, int32_t zp_z_pre, float scale_z_out, int32_t zp_z_out,
                       float scale_r_pre, int32_t zp_r_pre, float scale_r_out, int32_t zp_r_out,
                       float scale_g_pre, int32_t zp_g_pre, float scale_g_out, int32_t zp_g_out);


__host__ __device__ __forceinline__ int32_t rshift_round(int32_t x, int n) {
    if (n <= 0) return x << (-n);  // 可支持负 shift（左移）
    int32_t offset = 1 << (n - 1); // 四舍五入
    return (x >= 0 ? (x + offset) : (x - offset)) >> n;
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
