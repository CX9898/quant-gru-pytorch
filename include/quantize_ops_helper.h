#pragma once

// ============================================================================
// quantize_ops_helper.h - GRU 量化 CPU/GPU 共用函数
// ============================================================================
//
// 本文件包含：
//   1. CPU/GPU 共用的内联函数（__host__ __device__ __forceinline__）
//   2. 量化/反量化基础操作函数
//   3. 分段线性近似函数（Sigmoid/Tanh LUT）
//   4. 真实 Sigmoid/Tanh 函数（用于 QAT 训练）
//   5. GRU 门计算模板函数
//   6. 量化参数校准函数
//   7. 调试与工具函数
//
// 量化参数结构体定义已移至 quantize_param_types.h
//
// 设计原则：
//   - 所有缩放因子均为 2 的负 n 次方：scale = 2^(-exp2_inv)
//   - 支持对称量化（zp=0）和非对称量化（zp≠0）
//   - CPU/GPU 共用函数使用 __host__ __device__ __forceinline__ 标记，确保行为一致
//
// ============================================================================

// ============================================================================
// 激活函数模式开关
// ============================================================================
// 
// USE_REAL_ACTIVATION: 使用真实 sigmoid/tanh（反量化 → 浮点计算 → 量化）
//   - 优点：前向/反向一致，QAT 梯度更准确
//   - 缺点：比 LUT 慢（涉及 exp/tanh 浮点运算）
//
// 不定义 USE_REAL_ACTIVATION: 使用分段线性 LUT（默认）
//   - 优点：快速，纯整数运算
//   - 缺点：QAT 时前向（LUT）与反向（真实导数）不一致
//
// 使用方法：
//   - QAT 训练时：在 CMakeLists.txt 或编译命令中添加 -DUSE_REAL_ACTIVATION
//   - 推理部署时：不定义此宏，使用 LUT
//
// ============================================================================
#define USE_REAL_ACTIVATION  // 取消注释以启用真实激活函数

#include "cuda_compat.h"
#include "inline_ops.h"
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <vector>

// CUDA 编译时使用内置 min/max（有 __device__ 修饰）
// 纯 C++ 编译时使用 std::min/max
#ifndef __CUDACC__
using std::min;
using std::max;
#endif

#include "gru_quantization_ranges.h"
#include "histogram_collector.h"  // for get_minimum_scale
#include "quantize_param_types.h" // 量化参数结构体定义

// #define DEBUG

/**
 * @brief 生成分段线性量化查找表并存储到参数中
 *
 * 将 LUT 存储到 GRUQuantParams 中，避免全局 __constant__ 内存覆盖问题。
 * 在 finalize_calibration 时调用一次，然后在 setRescaleParam 时复制到 GateQuantParams。
 *
 * @param params GRU 量化参数，会被修改以存储生成的 LUT
 */
void generate_piecewise_linear_lut_to_params(GRUQuantParams &params);

// ============================================================================
// CPU/GPU 共用基础运算函数
// ============================================================================

/**
 * @brief 计算 2 的负 n 次方缩放因子（CPU/GPU 共用）
 *
 * scale = 2^(-exp2_inv)
 *
 * 使用 ldexpf 直接操作浮点指数，比除法更高效。
 *
 * @param exp2_inv 缩放因子指数
 * @return 缩放因子 scale = 2^(-exp2_inv)
 */
__host__ __device__ __forceinline__ float exp2_scale(int8_t exp2_inv) {
    return ldexpf(1.0f, -static_cast<int>(exp2_inv));
}

// ============================================================================
// 通用舍入函数（银行家舍入，与 rint / PyTorch round 一致）
// ============================================================================

/**
 * @brief 单精度浮点数银行家舍入（round half to even）
 *
 * 采用 round half to even 策略：1.5 → 2, 2.5 → 2, -1.5 → -2
 * 与 PyTorch 的 torch.round 行为一致。
 */
__host__ __device__ __forceinline__ float round_f(float x) {
    return rintf(x);
}

/**
 * @brief 双精度浮点数银行家舍入（round half to even）
 *
 * 采用 round half to even 策略：1.5 → 2, 2.5 → 2, -1.5 → -2
 * 与 PyTorch 的 torch.round 行为一致。
 */
__host__ __device__ __forceinline__ double round_d(double x) {
    return rint(x);
}

/**
 * @brief 浮点数舍入到 int32_t（银行家舍入）
 */
__host__ __device__ __forceinline__ int32_t round_to_int(float x) {
    return static_cast<int32_t>(round_f(x));
}

/**
 * @brief 双精度浮点数舍入到 int64_t（银行家舍入）
 */
__host__ __device__ __forceinline__ int64_t round_to_int64(double x) {
    return static_cast<int64_t>(round_d(x));
}

/**
 * @brief 带银行家舍入的右移操作（int32_t 版本，纯定点实现）
 *
 * 实现 round(x / 2^n) 的定点运算，支持正负移位。
 * 采用 round half to even（银行家舍入）策略。
 *
 * @param x 被移位的值
 * @param n 移位量（正数右移，负数或零左移）
 * @return 移位后的结果
 *
 * @note 纯定点实现，避免浮点转换的精度损失
 */
__host__ __device__ __forceinline__ int32_t rshift_round(int32_t x, int8_t n) {
    if (n <= 0) return x << (-n);
    
    // 处理负数：对绝对值舍入后取反
    const bool neg = (x < 0);
    const int32_t abs_x = neg ? -x : x;
    
    // 正数的银行家舍入
    const int32_t half = 1 << (n - 1);
    const int32_t mask = (1 << n) - 1;
    const int32_t q = abs_x >> n;      // 商（向下取整）
    const int32_t r = abs_x & mask;    // 余数
    
    int32_t result;
    if (r > half) {
        result = q + 1;                // 大于一半，进位
    } else if (r < half) {
        result = q;                    // 小于一半，舍去
    } else {
        // 正好一半：舍入到偶数
        result = (q & 1) ? (q + 1) : q;
    }
    
    return neg ? -result : result;
}

/**
 * @brief 带银行家舍入的右移操作（int64_t 版本，纯定点实现）
 *
 * 用于处理 16 位量化时可能超出 int32 范围的乘积。
 */
__host__ __device__ __forceinline__ int64_t rshift_round(int64_t x, int8_t n) {
    if (n <= 0) return x << (-n);
    
    // 处理负数
    const bool neg = (x < 0);
    const int64_t abs_x = neg ? -x : x;
    
    // 正数的银行家舍入
    const int64_t half = static_cast<int64_t>(1) << (n - 1);
    const int64_t mask = (static_cast<int64_t>(1) << n) - 1;
    const int64_t q = abs_x >> n;
    const int64_t r = abs_x & mask;
    
    int64_t result;
    if (r > half) {
        result = q + 1;
    } else if (r < half) {
        result = q;
    } else {
        result = (q & 1) ? (q + 1) : q;
    }
    
    return neg ? -result : result;
}

// ============================================================================
// CPU/GPU 共用饱和截断函数
// ============================================================================

/**
 * @brief 按任意位宽饱和截断
 *
 * 适用于位宽在运行时确定的场景，支持 1-31 位任意位宽。
 *
 * @param val 输入值
 * @param bw 目标位宽配置
 * @return 截断后的值（始终返回 int32_t，但值已在目标范围内）
 */
template <bool Training = false>
__host__ __device__ __forceinline__ 
int32_t clamp_by_bitwidth(int32_t val, QuantBitWidth bw, uint8_t* was_clamped = nullptr) {
    int32_t lo = bw.qmin();
    int32_t hi = bw.qmax();
    if constexpr (Training) {
        *was_clamped = (val < lo || val > hi) ? 1 : 0;
    }
    return max(lo, min(val, hi));
}

// ============================================================================
// CPU/GPU 共用量化/反量化函数
// ============================================================================

/**
 * @brief 单值反量化（CPU/GPU 共用）
 *
 * x = (q - zp) * scale
 *
 * 注：反量化不涉及截断，所以不需要 QuantBitWidth
 *
 * @tparam QuantT 量化类型（int8_t, int16_t, int32_t 等）
 * @param q 量化值
 * @param exp2_inv 缩放因子指数
 * @param zp 零点
 * @return 反量化浮点值
 */
template <typename QuantT>
__host__ __device__ __forceinline__ float dequantize(QuantT q, int8_t exp2_inv, int32_t zp) {
    int32_t v = static_cast<int32_t>(q) - zp;
    return static_cast<float>(v) * exp2_scale(exp2_inv);
}

// ============================================================================
// 浮点存储版量化/反量化函数（值是定点整数，用 float 存储）
// ============================================================================
// 用于 GPU-FP 实现，避免整数到浮点的频繁转换

/**
 * @brief 浮点版 clamp（值是定点整数，用 float 存储）
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param val 输入值
 * @param bw 位宽配置
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 * @return 截断后的值
 */
template <bool Training = false>
__host__ __device__ __forceinline__ 
float clamp_f(float val, QuantBitWidth bw, uint8_t* was_clamped = nullptr) {
    float lo = static_cast<float>(bw.qmin());
    float hi = static_cast<float>(bw.qmax());
    if constexpr (Training) {
        *was_clamped = (val < lo || val > hi) ? 1 : 0;
    }
#ifdef __CUDA_ARCH__
    return fmaxf(lo, fminf(val, hi));
#else
    return std::max(lo, std::min(val, hi));
#endif
}

/**
 * @brief 浮点版量化（输出 float 存储的定点值）
 * 
 * q = clamp(round(src / scale + zp), qmin, qmax)
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param src 浮点输入
 * @param exp2_inv 缩放因子指数
 * @param zp 零点
 * @param bw 位宽配置
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 */
template <bool Training = false>
__host__ __device__ __forceinline__ float quantize_f(float src, int8_t exp2_inv, int32_t zp,
                                                      QuantBitWidth bw, uint8_t* was_clamped = nullptr) {
    float scale = exp2_scale(exp2_inv);
    float q = round_f(src / scale + static_cast<float>(zp));
    return clamp_f<Training>(q, bw, was_clamped);
}

/**
 * @brief 浮点版量化（无零点，用于 per-channel 权重）
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 */
template <bool Training = false>
__host__ __device__ __forceinline__ float quantize_f(float src, int8_t exp2_inv, QuantBitWidth bw,
                                                      uint8_t* was_clamped = nullptr) {
    float scale = exp2_scale(exp2_inv);
    float q = round_f(src / scale);
    return clamp_f<Training>(q, bw, was_clamped);
}


/**
 * @brief 浮点版反量化
 * 
 * x = (q - zp) * scale
 */
__host__ __device__ __forceinline__ float dequantize_f(float q, int8_t exp2_inv, int32_t zp) {
    float scale = exp2_scale(exp2_inv);
    return (q - static_cast<float>(zp)) * scale;
}

/**
 * @brief 单值量化（任意位宽版本，CPU/GPU 共用）
 *
 * q = clamp(round(src / scale + zp), qmin, qmax)
 *
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param src 浮点输入
 * @param exp2_inv 缩放因子指数
 * @param zp 零点
 * @param bw 位宽配置
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 * @return 量化值（int32_t 存储）
 */
template <bool Training = false>
__host__ __device__ __forceinline__ int32_t quantize(float src, int8_t exp2_inv, int32_t zp,
                                                      QuantBitWidth bw, uint8_t* was_clamped = nullptr) {
    float q = round_f(src / exp2_scale(exp2_inv)) + static_cast<float>(zp);
    return clamp_by_bitwidth<Training>(static_cast<int32_t>(q), bw, was_clamped);
}


// ============================================================================
// 分段线性近似函数（CPU/GPU 共用）
// ============================================================================
//
// 【原理】将非线性函数（Sigmoid/Tanh）在每个分段内用线性函数 y = b*x + c 近似
//
// 【量化公式】q_y = (q_b * (q_x - zp_x)) >> n_BX_total + term_c_precomputed
//
// 【计算流程】
//   1. find_segment: 根据输入找到所属分段
//   2. x_offset = q_x - zp_x: 去零点
//   3. bw = q_b * x_offset: 乘以斜率（INT64 避免溢出）
//   4. term_bx = bw >> n_BX_total: 重缩放
//   5. q_y = term_bx + term_c_precomputed: 加上预计算的截距项
//
// ============================================================================

/**
 * @brief 查找输入所属的分段索引
 *
 * @param q_x 量化输入值
 * @param segments 分段参数数组（NUM_SEGMENTS 个元素）
 * @return 分段索引 [0, NUM_SEGMENTS-1]
 */
__host__ __device__ __forceinline__ int find_segment(int32_t q_x, const SegmentParams *segments) {
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        if (q_x < segments[i].threshold) {
            return i;
        }
    }
    return NUM_SEGMENTS - 1;
}

/**
 * @brief 分段线性近似核心函数（不做饱和截断）
 *
 * @param q_x 量化输入值
 * @param lut 查找表（包含分段参数和量化参数）
 * @return 近似结果（int32_t，未截断）
 */
__host__ __device__ __forceinline__ int32_t piecewise_linear_raw(int32_t q_x,
                                                                   const SigmoidLUT &lut) {
    int seg_id = find_segment(q_x, lut.segments);
    const SegmentParams &seg = lut.segments[seg_id];

    int32_t x_offset = q_x - lut.zp_x;
    int64_t bx_64 = static_cast<int64_t>(seg.q_b) * static_cast<int64_t>(x_offset);

    int32_t term_bx = (seg.n_BX_total >= 0)
                          ? static_cast<int32_t>(rshift_round(bx_64, seg.n_BX_total))
                          : static_cast<int32_t>(bx_64 << (-seg.n_BX_total));

    return term_bx + seg.term_c_precomputed;
}

/**
 * @brief 分段线性近似函数（带输入/输出饱和截断）
 *
 * @param q_x 量化输入值
 * @param lut 查找表
 * @param pre_bw 输入位宽（用于输入截断）
 * @param out_bw 输出位宽（用于输出截断）
 * @return 近似结果（已截断到输出范围）
 */
__host__ __device__ __forceinline__ int32_t piecewise_linear(int32_t q_x, const SigmoidLUT &lut,
                                                              QuantBitWidth pre_bw,
                                                              QuantBitWidth out_bw) {
    int32_t q_x_clamped = clamp_by_bitwidth(q_x, pre_bw);
    int32_t result = piecewise_linear_raw(q_x_clamped, lut);
    return clamp_by_bitwidth(result, out_bw);
}

// ============================================================================
// 真实 Sigmoid/Tanh 函数（反量化 → 浮点计算 → 量化）
// ============================================================================
// 用于替换 piecewise_linear，使前向/反向传播的激活函数一致
// 优点：QAT 时梯度更准确
// 缺点：比 LUT 慢（涉及浮点 exp 运算）
// 
// 激活函数 sigmoid<T>/tanh<T> 定义在 inline_ops.h 中

/**
 * @brief 真实 Sigmoid 函数（反量化 → sigmoid → 量化）
 *
 * 计算流程：
 *   1. 反量化：x_float = dequantize(q_x, shift_x, zp_x)
 *   2. Sigmoid：y_float = sigmoid<float>(x_float)
 *   3. 量化：  q_y = quantize(y_float, shift_y, zp_y, out_bw)
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 */
template <bool Training = false>
__host__ __device__ __forceinline__ int32_t real_sigmoid(int32_t q_x,
                                                          int8_t shift_x, int32_t zp_x,
                                                          int8_t shift_y, int32_t zp_y,
                                                          QuantBitWidth pre_bw,
                                                          QuantBitWidth out_bw,
                                                          uint8_t* was_clamped = nullptr) {
    int32_t q_x_clamped = clamp_by_bitwidth<false>(q_x, pre_bw);
    float x_float = dequantize(q_x_clamped, shift_x, zp_x);
    float y_float = sigmoid<float>(x_float);
    return quantize<Training>(y_float, shift_y, zp_y, out_bw, was_clamped);
}

/**
 * @brief 真实 Tanh 函数（反量化 → tanh → 量化）
 *
 * 计算流程：
 *   1. 反量化：x_float = dequantize(q_x, shift_x, zp_x)
 *   2. Tanh：  y_float = tanh<float>(x_float)
 *   3. 量化：  q_y = quantize(y_float, shift_y, zp_y, out_bw)
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 */
template <bool Training = false>
__host__ __device__ __forceinline__ int32_t real_tanh(int32_t q_x,
                                                       int8_t shift_x, int32_t zp_x,
                                                       int8_t shift_y, int32_t zp_y,
                                                       QuantBitWidth pre_bw,
                                                       QuantBitWidth out_bw,
                                                       uint8_t* was_clamped = nullptr) {
    int32_t q_x_clamped = clamp_by_bitwidth<false>(q_x, pre_bw);
    float x_float = dequantize(q_x_clamped, shift_x, zp_x);
    float y_float = tanh<float>(x_float);
    return quantize<Training>(y_float, shift_y, zp_y, out_bw, was_clamped);
}



// ============================================================================
// 浮点存储版激活函数（用于 GPU-FP 实现）
// ============================================================================
// 输入/输出均为 float 存储的定点值，scale 已预计算为 2^(-shift)

/**
 * @brief 浮点版真实 Sigmoid（反量化 → sigmoid → 量化）
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param q_x 量化输入（float 存储）
 * @param scale_x 输入反量化 scale = 2^(-shift_x)（已预计算）
 * @param zp_x 输入零点
 * @param scale_y 输出量化 scale = 2^(-shift_y)（已预计算）
 * @param zp_y 输出零点
 * @param out_bw 输出位宽配置
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 */
template <bool Training = false>
__host__ __device__ __forceinline__ 
float real_sigmoid_f(float q_x, float scale_x, float zp_x,
                     float scale_y, float zp_y, QuantBitWidth out_bw,
                     uint8_t* was_clamped = nullptr) {
    float x_fp = (q_x - zp_x) * scale_x;
    float y_fp = sigmoid<float>(x_fp);
    float q_y = round_f(y_fp / scale_y + zp_y);
    return clamp_f<Training>(q_y, out_bw, was_clamped);
}

/**
 * @brief 浮点版真实 Tanh（反量化 → tanh → 量化）
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param q_x 量化输入（float 存储）
 * @param scale_x 输入反量化 scale = 2^(-shift_x)（已预计算）
 * @param zp_x 输入零点
 * @param scale_y 输出量化 scale = 2^(-shift_y)（已预计算）
 * @param zp_y 输出零点
 * @param out_bw 输出位宽配置
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 */
template <bool Training = false>
__host__ __device__ __forceinline__ 
float real_tanh_f(float q_x, float scale_x, float zp_x,
                  float scale_y, float zp_y, QuantBitWidth out_bw,
                  uint8_t* was_clamped = nullptr) {
    float x_fp = (q_x - zp_x) * scale_x;
    float y_fp = tanh<float>(x_fp);
    float q_y = round_f(y_fp / scale_y + zp_y);
    return clamp_f<Training>(q_y, out_bw, was_clamped);
}

// ============================================================================
// GRU 门计算模板函数 (CPU/GPU 共用)
// ============================================================================

// 调试辅助函数（直接调用通用激活函数）
#if defined(DEBUG_QUANT) || defined(DEBUG_QUANT_DETAIL)
__host__ __device__ __forceinline__ float sigmoid_fp(float x) { return sigmoid<float>(x); }
__host__ __device__ __forceinline__ float tanh_fp(float x) { return tanh<float>(x); }
#endif  // DEBUG_QUANT || DEBUG_QUANT_DETAIL

/**
 * @brief 计算更新门 update_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 * 
 * @param weight_ih_linear  输入 Linear 变换结果 W*x + bw
 * @param weight_hh_linear  隐状态 Linear 变换结果 R*h + br
 * @param params    门计算参数
 * @param debug_idx 调试索引，-1 表示不输出调试信息
 */
/**
 * @brief 计算更新门 update_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param input_was_clamped 训练模式时保存输入 clamp mask，推理模式时可为 nullptr
 * @param output_was_clamped 训练模式时保存输出 clamp mask，推理模式时可为 nullptr
 */
template <bool Training>
__host__ __device__ __forceinline__ 
int32_t computeUpdateGate(int32_t weight_ih_linear, int32_t weight_hh_linear, const GateQuantParams &params,
                 uint8_t* input_was_clamped = nullptr,
                 uint8_t* output_was_clamped = nullptr,
                 [[maybe_unused]] int debug_idx = -1) {
    // 重缩放到 update_gate_input 空间
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, params.shift_weight_ih_linear_to_update_gate_input_);
    const int32_t hh_shifted = rshift_round(weight_hh_linear - params.zp_weight_hh_linear_, params.shift_weight_hh_linear_to_update_gate_input_);

    const int32_t update_gate_input = ih_shifted + hh_shifted + params.zp_update_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    const auto &lut = params.sigmoid_update_gate_lut_;
    
    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        // 对门输入进行位宽截断并记录 mask
        const int32_t clamped_input = clamp_by_bitwidth<true>(update_gate_input, bw_cfg.update_gate_input_, input_was_clamped);
        
#ifdef USE_REAL_ACTIVATION
        // 使用截断后的输入计算激活函数
        float x_float = dequantize(clamped_input, lut.shift_bits_x, lut.zp_x);
        float y_float = sigmoid<float>(x_float);
        const int32_t update_gate = quantize<true>(y_float, lut.shift_bits_y, lut.zp_y, 
                                                    bw_cfg.update_gate_output_, output_was_clamped);
#else
        // 非 real activation 时，使用分段线性
        int32_t result = piecewise_linear_raw(clamped_input, lut);
        const int32_t update_gate = clamp_by_bitwidth<true>(result, bw_cfg.update_gate_output_, output_was_clamped);
#endif
        return update_gate;
    } else {
#ifdef USE_REAL_ACTIVATION
        const int32_t update_gate = real_sigmoid(update_gate_input, 
                                                  lut.shift_bits_x, lut.zp_x,
                                                  lut.shift_bits_y, lut.zp_y,
                                                  bw_cfg.update_gate_input_, bw_cfg.update_gate_output_);
#else
        const int32_t update_gate = piecewise_linear(update_gate_input, lut, bw_cfg.update_gate_input_, bw_cfg.update_gate_output_);
#endif

#ifdef DEBUG_QUANT
        if (debug_idx == 0) {
            float update_gate_input_fp = (float)(update_gate_input - params.zp_update_gate_input_) /
                             (float)(1 << params.test.shift_update_gate_input_);
            float update_gate_fp = (float)(update_gate - params.zp_update_gate_output_) /
                         (float)(1 << params.test.shift_update_gate_output_);
            printf("[QUANT_I32] computeUpdateGate: update_gate_input_q=%d, update_gate_input_fp=%.6f, update_gate_q=%d, update_gate_fp=%.6f\n", 
                   update_gate_input, update_gate_input_fp, update_gate, update_gate_fp);
        }
#endif

#ifdef DEBUG_QUANT_DETAIL
        if (debug_idx >= 0 && debug_idx < 3) {
            float update_gate_input_fp = (float)(update_gate_input - params.zp_update_gate_input_) /
                             (float)(1 << params.test.shift_update_gate_input_);
            float update_gate_quant_fp = (float)(update_gate - params.zp_update_gate_output_) /
                               (float)(1 << params.test.shift_update_gate_output_);
            float update_gate_theory = sigmoid_fp(update_gate_input_fp);
            float error = update_gate_quant_fp - update_gate_theory;
            printf("[UpdateGate] idx=%d input_q=%d input_fp=%.4f | output_q=%d output_fp=%.4f | theory=%.4f | err=%.6f\n",
                   debug_idx, update_gate_input, update_gate_input_fp, update_gate, update_gate_quant_fp, update_gate_theory, error);
        }
#endif

        return update_gate;
    }
}

/**
 * @brief 计算重置门 reset_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param input_was_clamped 训练模式时保存输入 clamp mask，推理模式时可为 nullptr
 * @param output_was_clamped 训练模式时保存输出 clamp mask，推理模式时可为 nullptr
 */
template <bool Training>
__host__ __device__ __forceinline__ 
int32_t computeResetGate(int32_t weight_ih_linear, int32_t weight_hh_linear, const GateQuantParams &params,
                 uint8_t* input_was_clamped = nullptr,
                 uint8_t* output_was_clamped = nullptr,
                 [[maybe_unused]] int debug_idx = -1) {
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, params.shift_weight_ih_linear_to_reset_gate_input_);
    const int32_t hh_shifted = rshift_round(weight_hh_linear - params.zp_weight_hh_linear_, params.shift_weight_hh_linear_to_reset_gate_input_);

    const int32_t reset_gate_input = ih_shifted + hh_shifted + params.zp_reset_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    const auto &lut = params.sigmoid_reset_gate_lut_;
    
    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        // 对门输入进行位宽截断并记录 mask
        const int32_t clamped_input = clamp_by_bitwidth<true>(reset_gate_input, bw_cfg.reset_gate_input_, input_was_clamped);
        
#ifdef USE_REAL_ACTIVATION
        float x_float = dequantize(clamped_input, lut.shift_bits_x, lut.zp_x);
        float y_float = sigmoid<float>(x_float);
        const int32_t reset_gate = quantize<true>(y_float, lut.shift_bits_y, lut.zp_y,
                                                   bw_cfg.reset_gate_output_, output_was_clamped);
#else
        int32_t result = piecewise_linear_raw(clamped_input, lut);
        const int32_t reset_gate = clamp_by_bitwidth<true>(result, bw_cfg.reset_gate_output_, output_was_clamped);
#endif
        return reset_gate;
    } else {
#ifdef USE_REAL_ACTIVATION
        const int32_t reset_gate = real_sigmoid(reset_gate_input,
                                                 lut.shift_bits_x, lut.zp_x,
                                                 lut.shift_bits_y, lut.zp_y,
                                                 bw_cfg.reset_gate_input_, bw_cfg.reset_gate_output_);
#else
        const int32_t reset_gate = piecewise_linear(reset_gate_input, lut, bw_cfg.reset_gate_input_, bw_cfg.reset_gate_output_);
#endif

#ifdef DEBUG_QUANT
        if (debug_idx == 0) {
            float reset_gate_input_fp = (float)(reset_gate_input - params.zp_reset_gate_input_) /
                             (float)(1 << params.test.shift_reset_gate_input_);
            float reset_gate_fp = (float)(reset_gate - params.zp_reset_gate_output_) /
                         (float)(1 << params.test.shift_reset_gate_output_);
            printf("[QUANT_I32] computeResetGate: reset_gate_input_q=%d, reset_gate_input_fp=%.6f, reset_gate_q=%d, reset_gate_fp=%.6f\n", 
                   reset_gate_input, reset_gate_input_fp, reset_gate, reset_gate_fp);
        }
#endif

#ifdef DEBUG_QUANT_DETAIL
        if (debug_idx >= 0 && debug_idx < 3) {
            float reset_gate_input_fp = (float)(reset_gate_input - params.zp_reset_gate_input_) /
                             (float)(1 << params.test.shift_reset_gate_input_);
            float reset_gate_quant_fp = (float)(reset_gate - params.zp_reset_gate_output_) /
                               (float)(1 << params.test.shift_reset_gate_output_);
            float reset_gate_theory = sigmoid_fp(reset_gate_input_fp);
            float error = reset_gate_quant_fp - reset_gate_theory;
            printf("[ResetGate] idx=%d input_q=%d input_fp=%.4f | output_q=%d output_fp=%.4f | theory=%.4f | err=%.6f\n",
                   debug_idx, reset_gate_input, reset_gate_input_fp, reset_gate, reset_gate_quant_fp, reset_gate_theory, error);
        }
#endif

        return reset_gate;
    }
}

/**
 * @brief 计算候选门 new_gate = tanh(weight_ih_linear + reset_gate * weight_hh_linear)
 * 
 * 乘法scale融合：r * weight_hh_linear 的结果直接对齐到 new_gate_input，省略中间层
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param weight_ih_linear    输入 Linear 变换结果
 * @param weight_hh_linear    隐状态 Linear 变换结果（即 R*h + br，用于反向传播时直接保存到 v）
 * @param reset_gate          重置门输出
 * @param input_was_clamped 训练模式时保存输入 clamp mask，推理模式时可为 nullptr
 * @param output_was_clamped 训练模式时保存输出 clamp mask，推理模式时可为 nullptr
 */
template <bool Training>
__host__ __device__ __forceinline__ 
int32_t computeNewGate(int32_t weight_ih_linear, int32_t weight_hh_linear, int32_t reset_gate,
                 const GateQuantParams &params,
                 uint8_t* input_was_clamped = nullptr,
                 uint8_t* output_was_clamped = nullptr,
                 [[maybe_unused]] int debug_idx = -1) {
    // Linear 融合后，weight_hh_linear 就是 R*h + br
    // 计算 reset_gate * weight_hh_linear，直接对齐到 new_gate_input（融合中间层）
    const int64_t r_diff = static_cast<int64_t>(reset_gate) - params.zp_reset_gate_output_;
    const int64_t hh_diff = static_cast<int64_t>(weight_hh_linear) - params.zp_weight_hh_linear_;
    const int64_t reset_hidden_mul = r_diff * hh_diff;

    // 乘法结果直接 shift 到 new_gate_input 空间（融合后省略中间 zp）
    const int32_t rh_shifted = static_cast<int32_t>(rshift_round(reset_hidden_mul, params.shift_reset_mul_hh_to_new_gate_input_));

    // weight_ih_linear shift 到 new_gate_input 空间
    const int32_t ih_shifted = rshift_round(weight_ih_linear - params.zp_weight_ih_linear_, params.shift_weight_ih_linear_to_new_gate_input_);

    const int32_t new_gate_input = ih_shifted + rh_shifted + params.zp_new_gate_input_;

    const auto &bw_cfg = params.bitwidth_config_;
    const auto &lut = params.tanh_new_gate_lut_;
    
    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        // 对门输入进行位宽截断并记录 mask
        const int32_t clamped_input = clamp_by_bitwidth<true>(new_gate_input, bw_cfg.new_gate_input_, input_was_clamped);
        
#ifdef USE_REAL_ACTIVATION
        float x_float = dequantize(clamped_input, lut.shift_bits_x, lut.zp_x);
        float y_float = tanh<float>(x_float);
        const int32_t new_gate = quantize<true>(y_float, lut.shift_bits_y, lut.zp_y,
                                                 bw_cfg.new_gate_output_, output_was_clamped);
#else
        int32_t result = piecewise_linear_raw(clamped_input, lut);
        const int32_t new_gate = clamp_by_bitwidth<true>(result, bw_cfg.new_gate_output_, output_was_clamped);
#endif
        return new_gate;
    } else {
#ifdef USE_REAL_ACTIVATION
        const int32_t new_gate = real_tanh(new_gate_input,
                                            lut.shift_bits_x, lut.zp_x,
                                            lut.shift_bits_y, lut.zp_y,
                                            bw_cfg.new_gate_input_, bw_cfg.new_gate_output_);
#else
        const int32_t new_gate = piecewise_linear(new_gate_input, lut, bw_cfg.new_gate_input_, bw_cfg.new_gate_output_);
#endif

#ifdef DEBUG_QUANT
        if (debug_idx == 0) {
            float hh_fp = (float)(weight_hh_linear - params.zp_weight_hh_linear_) /
                          (float)(1 << params.test.shift_weight_hh_linear_);
            float new_gate_input_fp = (float)(new_gate_input - params.zp_new_gate_input_) /
                             (float)(1 << params.test.shift_new_gate_input_);
            float new_gate_fp = (float)(new_gate - params.zp_new_gate_output_) /
                         (float)(1 << params.test.shift_new_gate_output_);
            printf("[QUANT_I32] computeNewGate: hh_fp=%.6f, new_gate_input_fp=%.6f, new_gate_fp=%.6f\n",
                   hh_fp, new_gate_input_fp, new_gate_fp);
        }
#endif

#ifdef DEBUG_QUANT_DETAIL
        if (debug_idx >= 0 && debug_idx < 3) {
            float new_gate_input_fp = (float)(new_gate_input - params.zp_new_gate_input_) /
                             (float)(1 << params.test.shift_new_gate_input_);
            float new_gate_quant_fp = (float)(new_gate - params.zp_new_gate_output_) /
                               (float)(1 << params.test.shift_new_gate_output_);
            float new_gate_theory = tanh_fp(new_gate_input_fp);
            float error = new_gate_quant_fp - new_gate_theory;
            printf("[NewGate] idx=%d input_q=%d input_fp=%.4f | output_q=%d output_fp=%.4f | theory=%.4f | err=%.6f\n",
                   debug_idx, new_gate_input, new_gate_input_fp, new_gate, new_gate_quant_fp, new_gate_theory, error);
        }
#endif

        return new_gate;
    }
}

/**
 * @brief 计算隐藏状态 h_new = update_gate * h_old + (1 - update_gate) * new_gate
 * 
 * 乘法scale融合：u*h 和 (1-u)*n 的结果直接对齐到 h，省略中间层
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 */
template <bool Training>
__host__ __device__ __forceinline__ 
int32_t computeHiddenState(int32_t update_gate, int32_t new_gate, int32_t h_old, const GateQuantParams &params,
                 uint8_t* was_clamped = nullptr,
                 [[maybe_unused]] int debug_idx = -1) {
    // 计算 update_gate * h_old，直接对齐到 h（融合中间层）
    const int64_t u_diff = static_cast<int64_t>(update_gate) - params.zp_update_gate_output_;
    const int64_t h_diff = static_cast<int64_t>(h_old) - params.zp_h_;
    const int64_t old_contribution_mul = u_diff * h_diff;

    // 乘法结果直接 shift 到 h 空间（融合后省略中间 zp）
    const int32_t old_shifted = static_cast<int32_t>(rshift_round(old_contribution_mul, params.shift_update_old_to_h_));

    // 计算 (1 - update_gate) * new_gate，直接对齐到 h（融合中间层）
    // quant_one = 2^shift + zp，是常数 1 在 update_gate_output 量化空间的完整表示
    // one_minus_u = quant_one - update_gate = (2^shift + zp) - update_gate
    const int64_t one_minus_u = static_cast<int64_t>(params.quant_one_in_update_gate_scale_) - update_gate;
    const int64_t n_diff = static_cast<int64_t>(new_gate) - params.zp_new_gate_output_;
    const int64_t new_contribution_mul = one_minus_u * n_diff;

    // 乘法结果直接 shift 到 h 空间（融合后省略中间 zp）
    const int32_t new_shifted = static_cast<int32_t>(rshift_round(new_contribution_mul, params.shift_update_new_to_h_));

    // 计算 h_new = old_shifted + new_shifted + zp_h
    const int32_t h_new = old_shifted + new_shifted + params.zp_h_;

    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        return clamp_by_bitwidth<true>(h_new, params.bitwidth_config_.h_, was_clamped);
    } else {
        const int32_t h = clamp_by_bitwidth(h_new, params.bitwidth_config_.h_);

#ifdef DEBUG_QUANT
        if (debug_idx == 0) {
            float u_fp = (float)(update_gate - params.zp_update_gate_output_) /
                         (float)(1 << params.test.shift_update_gate_output_);
            float n_fp = (float)(new_gate - params.zp_new_gate_output_) /
                         (float)(1 << params.test.shift_new_gate_output_);
            float h_old_fp = (float)(h_old - params.zp_h_) / 
                             (float)(1 << params.test.shift_h_);
            float h_fp = (float)(h - params.zp_h_) / 
                         (float)(1 << params.test.shift_h_);
            printf("[QUANT_I32] computeHiddenState: u_fp=%.6f, n_fp=%.6f, h_old_fp=%.6f, h_new_fp=%.6f\n", 
                   u_fp, n_fp, h_old_fp, h_fp);
        }
#endif

#ifdef DEBUG_QUANT_DETAIL
        if (debug_idx >= 0 && debug_idx < 3) {
            float u_fp = (float)(update_gate - params.zp_update_gate_output_) /
                         (float)(1 << params.test.shift_update_gate_output_);
            float n_fp = (float)(new_gate - params.zp_new_gate_output_) /
                         (float)(1 << params.test.shift_new_gate_output_);
            float h_old_fp = (float)(h_old - params.zp_h_) / 
                             (float)(1 << params.test.shift_h_);
            float h_quant_fp = (float)(h - params.zp_h_) / 
                               (float)(1 << params.test.shift_h_);
            printf("[HiddenState] idx=%d u_fp=%.4f n_fp=%.4f h_old_fp=%.4f | h_q=%d h_fp=%.4f\n",
                   debug_idx, u_fp, n_fp, h_old_fp, h, h_quant_fp);
        }
#endif

        return h;
    }
}

// ============================================================================
// 带 mask 版本的 GRU 门计算函数（用于 QAT）
// ============================================================================
// 
// 分离输入和输出的 clamp mask：
//   - input_was_clamped: 门输入被截断时为 1
//   - output_was_clamped: 门输出被截断时为 1
//
// 反向传播时：
//   - gate_input_mask 影响传回 linear 输出的梯度 (dp, dq)
//   - gate_output_mask 影响传回下一层的梯度
// ============================================================================




// ============================================================================
// GPU Kernel 函数声明
// ============================================================================

/**
 * @brief 计算权重矩阵列和乘以零点（用于 GEMM 零点补偿）
 *
 * weight_sum[j] = sum_i(W_q[i,j]) * zp >> n[j]
 *
 * @tparam T 权重类型（int8_t/int16_t）
 * @param W_q 量化权重矩阵 [out_dim, in_dim]
 * @param weight_sum 输出数组 [out_dim]
 * @param zp 输入零点
 * @param n per-channel 重缩放移位
 * @param out_dim 输出维度 (M)
 * @param in_dim 输入维度 (K)
 * @param stream CUDA 流
 */
template <typename T>
void computeWeightSumMulzp(const T *W_q, int64_t *weight_sum, int32_t zp,
                           const int8_t *__restrict__ n, int out_dim, int in_dim,
                           cudaStream_t stream = 0);

/// @brief int32_t 输出版本（适用于 8 位量化，不会溢出）
template <typename T>
void computeWeightSumMulzp(const T *W_q, int32_t *weight_sum, int32_t zp,
                           const int8_t *__restrict__ n, int out_dim, int in_dim,
                           cudaStream_t stream = 0);

/**
 * @brief 应用零点补偿到 2D GEMM 输出
 *
 * Y[i,j] -= weight_sum[i] * x_zp[j]
 */
void applyZeroPointCompensation2D(int32_t *Y_int32, const int32_t *weight_sum, const int32_t *x_zp,
                                  int out_dim, int batch_size, cudaStream_t stream = 0);

// ============================================================================
// 量化参数校准与量化/反量化函数
// ============================================================================

/**
 * @brief 量化参数校准函数（与 AIMET MinMaxEncodingAnalyzer.compute_encodings_from_stats 完全一致）
 *
 * 根据数据范围和位宽配置计算量化参数，缩放因子对齐到 2 的负 n 次方。
 * 
 * 与 AIMET 的一致性:
 *   1. num_steps 检查
 *   2. 使用 get_minimum_scale(num_steps)
 *   3. 确保 0 在范围内
 *   4. 范围扩展逻辑（对称/非对称分别处理）
 *   5. 对称量化 delta 计算（区分 pos/neg steps）
 *   6. 2-bit 特殊处理
 *   7. Inf 保护
 *
 * @param[in] orig_min 原始数据最小值
 * @param[in] orig_max 原始数据最大值
 * @param[in] bw 位宽配置（含位数和符号）
 * @param[in] is_symmetric 是否对称量化
 * @param[out] aligned_min 对齐后的最小值
 * @param[out] aligned_max 对齐后的最大值
 * @param[out] exp2_inv 缩放因子指数，scale = 2^(-exp2_inv)
 * @param[out] zp 零点（对称量化时为 0）
 * @param[in] name 调试用名称（可选）
 */
inline void calibrateQuantParams(float orig_min, float orig_max, QuantBitWidth bw,
                                 bool is_symmetric, float &aligned_min, float &aligned_max,
                                 int8_t &exp2_inv, int32_t &zp, const std::string &name = "") {
    // 从位宽配置获取量化范围（使用 auto scale 版本用于计算 num_steps 和 scale）
    const int32_t quant_min = bw.qmin_auto_scale();
    const int32_t quant_max = bw.qmax_auto_scale();
    const int num_steps = quant_max - quant_min;  // 量化级数
    
    // 与 AIMET 一致: num_steps 检查
    if (num_steps <= 0) {
        throw std::runtime_error("calibrateQuantParams: num_steps must be > 0");
    }
    
    // 与 AIMET _get_minimum_scale 一致
    const float minimum_scale = get_minimum_scale(num_steps);
    
    // 与 AIMET 一致: 确保 0 在范围内
    // min_with_zero = clamp(min, max=0) -> min <= 0
    // max_with_zero = clamp(max, min=0) -> max >= 0
    float min_with_zero = std::min(orig_min, 0.0f);
    float max_with_zero = std::max(orig_max, 0.0f);
    
    // 与 AIMET 一致: 范围扩展（当 tensor_diff < minimum_scale 时）
    float tensor_diff = (max_with_zero - min_with_zero) / static_cast<float>(num_steps);
    float adjustment_step = (tensor_diff < minimum_scale) ? minimum_scale : 0.0f;
    
    float updated_min, updated_max;
    if (is_symmetric) {
        if (bw.is_unsigned_) {
            // UINT 对称量化：只扩展 max（数据范围 [0, max]）
            updated_max = max_with_zero + static_cast<float>(num_steps) * adjustment_step;
            updated_min = 0.0f;  // UINT 的 min 固定为 0
        } else {
            // INT 对称量化: 两边扩展
            updated_max = max_with_zero + std::floor(num_steps / 2.0f) * adjustment_step;
            updated_min = min_with_zero - std::ceil(num_steps / 2.0f) * adjustment_step;
        }
    } else {
        // 非对称量化: 只扩展 max
        updated_max = max_with_zero + static_cast<float>(num_steps) * adjustment_step;
        updated_min = min_with_zero;
    }
    
    float scale;
    if (is_symmetric) {
        zp = 0;
        
        // UINT + symmetric: 特殊处理（zp=0，范围 [0, qmax]）
        if (bw.is_unsigned_) {
            // UINT 对称量化：数据范围 [0, max]，量化范围 [0, qmax]
            // scale = max / qmax，确保 max 能被正确量化
            float data_max = std::max(updated_max, minimum_scale);
            float raw_scale = data_max / static_cast<float>(quant_max);
            raw_scale = std::max(raw_scale, minimum_scale);
            
            // POT 转换
            exp2_inv = static_cast<int8_t>(std::floor(std::log2(1.0f / raw_scale)));
            scale = exp2_scale(exp2_inv);
            aligned_min = 0.0f;
            aligned_max = scale * static_cast<float>(quant_max);
        } else {
            // INT 对称量化：与 AIMET 一致
            const int num_pos_steps = num_steps / 2;           // floor(N/2), 8-bit: 127
            const int num_neg_steps = (num_steps + 1) / 2;     // ceil(N/2),  8-bit: 128
            
            // 与 AIMET 一致: 2-bit 特殊处理
            // "For 2-bit quantization, using math.floor to compute num_pos_steps can result 
            //  in a wasted bin on the negative side given a symmetrically distributed weight."
            int additional_step_for_calibration = 0;
            if (num_steps == 3) {  // 2-bit strict symmetric
                additional_step_for_calibration = 1;
            }
            
            // 与 AIMET 一致: delta = max(max/(pos+additional), -min/neg)
            float delta_from_max = (num_pos_steps + additional_step_for_calibration > 0) 
                                 ? updated_max / (num_pos_steps + additional_step_for_calibration)
                                 : 0.0f;
            float delta_from_min = (num_neg_steps > 0) 
                                 ? -updated_min / num_neg_steps 
                                 : 0.0f;
            float delta = std::max(delta_from_max, delta_from_min);
            delta = std::max(delta, minimum_scale);  // 确保 delta >= minimum_scale
            
            // 与 AIMET 一致: 重新计算 min/max
            // offset = -num_neg_steps
            // updated_min = offset * delta = -num_neg_steps * delta
            // updated_max = num_pos_steps * delta
            updated_min = -static_cast<float>(num_neg_steps) * delta;
            updated_max = static_cast<float>(num_pos_steps) * delta;
            
            // POT 转换
            float raw_scale = delta;
            exp2_inv = static_cast<int8_t>(std::floor(std::log2(1.0f / raw_scale)));
            scale = exp2_scale(exp2_inv);
            aligned_max = scale * num_pos_steps;
            aligned_min = -scale * num_neg_steps;
        }
    } else {
        // 非对称量化
        float range = updated_max - updated_min;
        range = std::max(range, minimum_scale * static_cast<float>(num_steps));
        float raw_scale = range / static_cast<float>(num_steps);

        exp2_inv = static_cast<int8_t>(std::floor(std::log2(1.0f / raw_scale)));
        scale = exp2_scale(exp2_inv);

        aligned_min = std::floor(updated_min / scale) * scale;
        aligned_max = std::ceil(updated_max / scale) * scale;

        zp = round_to_int(quant_min - aligned_min / scale);
    }
    
    // 与 AIMET 一致: Inf 保护
    constexpr float float_max = std::numeric_limits<float>::max();
    constexpr float float_min = std::numeric_limits<float>::lowest();
    aligned_max = std::min(aligned_max, float_max);
    aligned_min = std::max(aligned_min, float_min);

#ifdef DEBUG
    if (!name.empty()) {
        printf("[DEBUG][QuantParam][%s] bits=%d, unsigned=%d, orig=[%.4f,%.4f], "
               "zero_included=[%.4f,%.4f], aligned=[%.4f,%.4f], scale=%.6f, exp2_inv=%d, zp=%d\n",
               name.c_str(), bw.bits_, bw.is_unsigned_, orig_min, orig_max,
               min_with_zero, max_with_zero, aligned_min, aligned_max, scale, exp2_inv, zp);
    }
#endif
}

/// @brief 批量量化（任意位宽版本，输出 int32_t）
inline void quantification(const float *data, int32_t *quant_data, size_t size, 
                           int8_t exp2_inv, int32_t zp, QuantBitWidth bw) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        quant_data[i] = quantize(data[i], exp2_inv, zp, bw);
    }
}

/// @brief Per-channel 批量量化（Host 端，统一 int32_t 输出，使用位宽配置）
inline void quantificationPerChannelBitwidth(const float *src, int32_t *quant_data, size_t input_size,
                                              size_t channel_size, const std::vector<int8_t> &exp2_invs,
                                              QuantBitWidth bw) {
#pragma omp parallel for
    for (size_t i = 0; i < channel_size; ++i) {
        const int8_t exp2_inv = exp2_invs[i];
        for (size_t j = 0; j < input_size; ++j) {
            const size_t idx = j * channel_size + i;
            quant_data[idx] = quantize(src[idx], exp2_inv, 0, bw);
        }
    }
}

// ============================================================================
// GPU 量化/反量化 Kernel 声明
// ============================================================================

namespace dev {

/// @brief GPU 批量反量化
template <typename T, typename QuantT>
void dequantification(const QuantT *quant_data, T *data, size_t size, int8_t exp2_inv, int32_t zp);

/// @brief GPU 反量化 V 向量（各部分使用不同量化参数）
/// V 向量布局: [z_out, r_out, g_out, weight_hh_linear_g]
template <typename T>
void dequantificationV(const int32_t *quant_data, T *data, int time_steps, int batch_size,
                       int hidden_size, int8_t shift_update_gate, int32_t zp_update_gate, int8_t shift_reset_gate,
                       int32_t zp_reset_gate, int8_t shift_new_gate, int32_t zp_new_gate, 
                       int8_t shift_weight_hh_linear, int32_t zp_weight_hh_linear);



// ============================================================================
// 浮点存储版量化函数（用于 GPU-FP 实现）
// ============================================================================



/// @brief GPU 反量化（float 输入的量化值）
void dequantificationFP(const float *quant_data, float *data, size_t size,
                        int8_t exp2_inv, int32_t zp);

/// @brief GPU 反量化 V 向量（float 输入的量化值）
/// V 布局: [time_steps, batch_size, hidden_size * 4]
/// 4个部分使用不同量化参数: [z_out, r_out, g_out, weight_hh_linear_g]
void dequantificationVFP(const float *quant_data, float *data, int time_steps, int batch_size,
                         int hidden_size, int8_t shift_z, int32_t zp_z, int8_t shift_r,
                         int32_t zp_r, int8_t shift_g, int32_t zp_g,
                         int8_t shift_hh, int32_t zp_hh);

/// @brief GPU 原地反量化 V 向量（float 输入的量化值）
/// V 布局: [time_steps, batch_size, hidden_size * 4]
/// 4个部分使用不同量化参数: [z_out, r_out, g_out, weight_hh_linear_g]
/// @param data 量化后的数据（输入），反量化后的数据（输出），原地修改
void dequantificationVFPInplace(float *data, int time_steps, int batch_size,
                                int hidden_size, int8_t shift_z, int32_t zp_z, int8_t shift_r,
                                int32_t zp_r, int8_t shift_g, int32_t zp_g,
                                int8_t shift_hh, int32_t zp_hh);

/// @brief GPU Per-channel 反量化（float 输入的量化值）
/// @param quant_data 量化后的数据（float 存储，实际是定点整数）
/// @param data 反量化后的输出数据
/// @param input_size 输入维度（对于 W 是 input_size，对于 R 是 hidden_size）
/// @param channel_size 通道数（hidden_size * 3）
/// @param exp2_invs per-channel 的缩放因子指数向量
void dequantificationPerChannelFP(const float *quant_data, float *data,
                                  size_t input_size, size_t channel_size,
                                  const dev::vector<int8_t> &exp2_invs);

/// @brief GPU Per-channel 原地反量化（float 输入的量化值）
/// @param data 量化后的数据（输入），反量化后的数据（输出），原地修改
/// @param input_size 输入维度（对于 W 是 input_size，对于 R 是 hidden_size）
/// @param channel_size 通道数（hidden_size * 3）
/// @param exp2_invs per-channel 的缩放因子指数向量
void dequantificationPerChannelFPInplace(float *data,
                                         size_t input_size, size_t channel_size,
                                         const dev::vector<int8_t> &exp2_invs);

/// @brief GPU 原地反量化（float 输入的量化值）
/// @param data 量化后的数据（输入），反量化后的数据（输出），原地修改
/// @param size 数据大小
/// @param exp2_inv 缩放因子指数
/// @param zp 零点
void dequantificationFPInplace(float *data, size_t size,
                               int8_t exp2_inv, int32_t zp);

// ============================================================================
// 带 mask 版本的量化函数（用于 QAT）
// ============================================================================

// ============================================================================
// 量化函数（统一接口，支持可选 mask 输出）
// ============================================================================

/// @brief GPU 量化（float 输出）
/// @tparam Training 是否训练模式（决定是否使用 mask）
/// @param mask 训练模式时保存 clamp mask，推理模式时可为 nullptr
template <bool Training = false>
void quantificationFP(const float *data, float *quant_data, uint8_t *mask,
                      size_t size, int8_t exp2_inv, int32_t zp, QuantBitWidth bw);

/// @brief GPU 量化（int32 输出）
/// @tparam Training 是否训练模式（决定是否使用 mask）
/// @param mask 训练模式时保存 clamp mask，推理模式时可为 nullptr
template <bool Training = false>
void quantificationBitwidth(const float *data, int32_t *quant_data, uint8_t *mask,
                            size_t size, int8_t exp2_inv, int32_t zp, QuantBitWidth bw);

/// @brief GPU Per-channel 量化（float 输出）
/// @tparam Training 是否训练模式（决定是否使用 mask）
/// @param mask 训练模式时保存 clamp mask，推理模式时可为 nullptr
template <bool Training = false>
void quantificationPerChannelFP(const float *src, float *quant_data, uint8_t *mask,
                                size_t input_size, size_t channel_size,
                                const dev::vector<int8_t> &exp2_invs, QuantBitWidth bw);

/// @brief GPU Per-gate 量化（float 输出，用于 GRU 权重）
/// @tparam Training 是否训练模式（决定是否使用 mask）
/// @param mask 训练模式时保存 clamp mask，推理模式时可为 nullptr
/// @param input_size 输入维度（对于 W 是 input_size，对于 R 是 hidden_size）
/// @param hidden_size 隐藏层维度
/// @param exp2_inv_z/r/g 三个 gate 的量化参数
template <bool Training = false>
void quantificationPerGateFP(const float *src, float *quant_data, uint8_t *mask,
                             size_t input_size, size_t hidden_size,
                             int8_t exp2_inv_z, int8_t exp2_inv_r, int8_t exp2_inv_g,
                             QuantBitWidth bw);

/// @brief GPU Per-channel 量化（int32 输出）
/// @tparam Training 是否训练模式（决定是否使用 mask）
/// @param mask 训练模式时保存 clamp mask，推理模式时可为 nullptr
template <bool Training = false>
void quantificationPerChannelBitwidth(const float *src, int32_t *quant_data, uint8_t *mask,
                                      size_t input_size, size_t channel_size,
                                      const dev::vector<int8_t> &exp2_invs, QuantBitWidth bw);

// ============================================================================
// Bias 特殊量化函数（使用 round(bias / scale / 128) * 128）
// ============================================================================

/// @brief GPU Bias 特殊量化（float 输出）
/// 量化公式: q = clamp(round((bias / scale) / 128) * 128, qmin, qmax)
/// @tparam Training 是否训练模式（决定是否使用 mask）
/// @param mask 训练模式时保存 clamp mask，推理模式时可为 nullptr
template <bool Training = false>
void quantificationBiasFP(const float *src, float *quant_data, uint8_t *mask,
                          size_t channel_size,
                          const dev::vector<int8_t> &exp2_invs, QuantBitWidth bw);

// ============================================================================
// 通用权重量化函数（根据 granularity 自动选择）
// ============================================================================

/// @brief GPU 权重量化（float 输出，根据 granularity 自动选择）
/// @tparam Training 是否训练模式（决定是否使用 mask）
/// @param src 输入权重数据
/// @param quant_data 输出量化数据
/// @param mask 训练模式时保存 clamp mask，推理模式时可为 nullptr
/// @param input_size 输入维度（对于 W 是 input_size，对于 R 是 hidden_size）
/// @param hidden_size 隐藏层维度
/// @param granularity 量化粒度（PER_TENSOR, PER_GATE, PER_CHANNEL）
/// @param shift_tensor per-tensor 的 shift 值（granularity == PER_TENSOR 时使用）
/// @param shift_gate per-gate 的 shift 数组 [z, r, g]（granularity == PER_GATE 时使用）
/// @param shift_channel per-channel 的 shift 数组（granularity == PER_CHANNEL 时使用，非 PER_CHANNEL 时可为空 vector）
/// @param bw 位宽配置
template <bool Training = false>
void quantificationWeightFP(const float *src, float *quant_data, uint8_t *mask,
                             size_t input_size, size_t hidden_size,
                             OperatorQuantConfig::QuantizationGranularity granularity,
                             int8_t shift_tensor,
                             const std::array<int8_t, 3> &shift_gate,
                             const dev::vector<int8_t> &shift_channel,
                             QuantBitWidth bw);

/// @brief GPU Per-channel 反量化
template <typename T, typename QuantT>
void dequantificationPerChannel(const QuantT *quant_data, T *data, size_t input_size,
                                size_t channel_size, const dev::vector<int8_t> &exp2_invs);

// ============================================================================
// GRU 权重量化统一接口（封装 W, R, bw, br）
// ============================================================================

/// @brief GRU 权重量化统一接口（根据 granularity 自动选择量化方式）
/// @tparam Training 是否训练模式（决定是否使用 mask）
/// @param W 输入权重 W [input_size, hidden_size * 3]
/// @param R 循环权重 R [hidden_size, hidden_size * 3]
/// @param bw 输入偏置 bw [hidden_size * 3]
/// @param br 循环偏置 br [hidden_size * 3]
/// @param W_q_out 输出量化权重 W（必须由外部分配内存）
/// @param R_q_out 输出量化权重 R（必须由外部分配内存）
/// @param bw_q_out 输出量化偏置 bw（必须由外部分配内存）
/// @param br_q_out 输出量化偏置 br（必须由外部分配内存）
/// @param W_mask 训练模式时保存 W 的 clamp mask，推理模式时可为 nullptr
/// @param R_mask 训练模式时保存 R 的 clamp mask，推理模式时可为 nullptr
/// @param bw_mask 训练模式时保存 bw 的 clamp mask，推理模式时可为 nullptr
/// @param br_mask 训练模式时保存 br 的 clamp mask，推理模式时可为 nullptr
/// @param input_size 输入维度
/// @param hidden_size 隐藏层维度
/// @param quant_params 量化参数（包含 granularity 配置和 shift 值）
template <bool Training = false>
void quantizeGRUWeights(const float *W, const float *R, const float *bw, const float *br,
                        float *W_q_out, float *R_q_out, float *bw_q_out, float *br_q_out,
                        uint8_t *W_mask, uint8_t *R_mask, uint8_t *bw_mask, uint8_t *br_mask,
                        size_t input_size, size_t hidden_size,
                        const GRUQuantParams &quant_params);

}  // namespace dev

// ============================================================================
// 工具函数
// ============================================================================

#include <limits>
#include <random>

/// @brief 获取全局随机数生成器（固定种子，确保可复现）
inline std::mt19937 &getGlobalRng() {
    static std::mt19937 gen(42);
    return gen;
}

/// @brief 设置全局随机种子
inline void setGlobalRandomSeed(unsigned int seed) { getGlobalRng().seed(seed); }

/**
 * @brief 用截断正态分布填充向量
 *
 * @param data 待填充向量
 * @param min_value 最小值
 * @param max_value 最大值
 *
 * @note 使用 3σ 覆盖范围，超出范围的值会重新采样
 */
inline void fillVectorWithNormalDistribution(std::vector<float> &data, float min_value,
                                             float max_value) {
    float mean = (min_value + max_value) / 2.0f;
    float stddev = (max_value - min_value) / 6.0f;

    std::mt19937 &gen = getGlobalRng();
    std::normal_distribution<float> dist(mean, stddev);

    for (auto &value : data) {
        float sample;
        do {
            sample = dist(gen);
        } while (sample < min_value || sample > max_value);
        value = sample;
    }
}

// ============================================================================
// LUT 系数量化辅助函数
// ============================================================================
// 这些函数用于 LUT 生成时量化分段线性近似的系数（斜率、截距等）

// -------------------- 系数量化（对称量化，zp=0）--------------------

/// @brief 量化系数为 INT32（用于 LUT 斜率 q_b，避免截断误差）
inline int32_t quantize_coefficient_int32(float val_fp, int8_t shift_bits) {
    float scale = exp2_scale(shift_bits);
    int64_t q = round_to_int64(static_cast<double>(val_fp / scale));
    q = std::max(static_cast<int64_t>(INT32_MIN), std::min(static_cast<int64_t>(INT32_MAX), q));
    return static_cast<int32_t>(q);
}

// -------------------- Shift bits 自动确定 --------------------

/// @brief 根据最大值和位宽配置自动确定 shift_bits
/// @param max_val 浮点数的最大绝对值
/// @param bw 目标量化位宽
/// @return 使量化值能充分利用目标范围的 shift_bits
inline int8_t determine_shift_bits(float max_val, QuantBitWidth bw) {
    if (max_val < 1e-9f) return 0;
    // 使用 qmax 作为量化范围上限（对称量化）
    float qmax = static_cast<float>(bw.qmax());
    float scale = max_val / qmax;
    int8_t shift_bits = static_cast<int8_t>(std::floor(-std::log2(scale)));
    return std::max(static_cast<int8_t>(0), shift_bits);
}

// ============================================================================
// 调试函数
// ============================================================================

/// @brief 打印 GRU 量化参数（调试用）
inline void printParms(const GRUQuantParams &quant_parms) {
    printf("GRUQuantParams:\n");
    printf("  hidden = %d\n", quant_parms.hidden_);

    // 输入/隐状态
    printf("  x:  exp2_inv=%2d, zp=%d\n", quant_parms.shift_x_, quant_parms.zp_x_);
    printf("  h:  exp2_inv=%2d, zp=%d\n", quant_parms.shift_h_, quant_parms.zp_h_);

    // Per-channel 权重
    auto print_vec = [](const char *name, const std::vector<int8_t> &vec) {
        printf("  %s (size %zu): ", name, vec.size());
        for (size_t i = 0; i < vec.size() && i < 5; ++i) printf("%d ", vec[i]);
        if (vec.size() > 5) printf("...");
        printf("\n");
    };
    print_vec("W ", quant_parms.shift_W_);
    print_vec("R ", quant_parms.shift_R_);
    print_vec("bw", quant_parms.shift_bw_);
    print_vec("br", quant_parms.shift_br_);

    // Linear 输出
    printf("  weight_ih_linear: shift=%2d, zp=%d\n", quant_parms.shift_weight_ih_linear_, quant_parms.zp_weight_ih_linear_);
    printf("  weight_hh_linear: shift=%2d, zp=%d\n", quant_parms.shift_weight_hh_linear_, quant_parms.zp_weight_hh_linear_);

    // 门参数
    printf("  update_gate_input:  exp2_inv=%2d, zp=%d\n", quant_parms.shift_update_gate_input_, quant_parms.zp_update_gate_input_);
    printf("  update_gate_output: exp2_inv=%2d, zp=%d\n", quant_parms.shift_update_gate_output_, quant_parms.zp_update_gate_output_);
    printf("  reset_gate_input:   exp2_inv=%2d, zp=%d\n", quant_parms.shift_reset_gate_input_, quant_parms.zp_reset_gate_input_);
    printf("  reset_gate_output:  exp2_inv=%2d, zp=%d\n", quant_parms.shift_reset_gate_output_, quant_parms.zp_reset_gate_output_);
    printf("  new_gate_input:     exp2_inv=%2d, zp=%d\n", quant_parms.shift_new_gate_input_, quant_parms.zp_new_gate_input_);
    printf("  new_gate_output:    exp2_inv=%2d, zp=%d\n", quant_parms.shift_new_gate_output_, quant_parms.zp_new_gate_output_);

    // 中间计算
    printf("  mul_reset_hidden:   exp2_inv=%2d, zp=%d\n", quant_parms.shift_mul_reset_hidden_, quant_parms.zp_mul_reset_hidden_);

    // 隐状态更新
    printf("  mul_new_contribution: exp2_inv=%2d, zp=%d\n", quant_parms.shift_mul_new_contribution_,
           quant_parms.zp_mul_new_contribution_);
    printf("  mul_old_contribution: exp2_inv=%2d, zp=%d\n", quant_parms.shift_mul_old_contribution_,
           quant_parms.zp_mul_old_contribution_);
}
