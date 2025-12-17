#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "blas.h"
#include "devVector.h"
#include "device_ptr.h"
#include "gru_quant.h"
#include "inline_ops.h"
#include "quantize_ops.cuh"
#include "quantize_ops_helper.hpp"

namespace kernel {

// 调试开关：取消注释以启用调试输出
// #define DEBUG_QUANT

// 调试统计变量（可选启用）
// #define DEBUG_OVERFLOW_STATS
#ifdef DEBUG_OVERFLOW_STATS
__device__ int64_t g_max_old_contrib_product = 0;
__device__ int64_t g_max_new_contrib_product = 0;
__device__ int64_t g_max_rRh_product = 0;
__device__ int g_overflow_count = 0;
#endif

// 将 int64 GEMM 结果减去补偿项并右移，存入 int32
// output[i] = (gemm_i64[i] - compensation[i % hidden3]) >> shift[i % hidden3] + zp
__global__ void rescaleGemmI64ToI32(
    const int64_t* __restrict__ gemm_i64,     // [hidden*3, batch*steps] GEMM 输出
    const int64_t* __restrict__ compensation, // [hidden*3] W_sum_mul_x_zp
    const int8_t* __restrict__ shift,         // [hidden*3] per-channel shift
    int32_t* __restrict__ output,             // [hidden*3, batch*steps] rescaled 输出
    int32_t zp,                               // zero point
    int hidden3,                              // hidden_size * 3
    int total_size,                           // hidden*3 * batch*steps
    bool debug = false                        // 调试开关
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    int channel = idx % hidden3;
    int64_t val = gemm_i64[idx] - compensation[channel];
    int8_t n = shift[channel];
    
#ifdef DEBUG_QUANT
    // 调试输出
    if (debug && idx == 0) {
        printf("[DEBUG] rescaleGemmI64ToI32: gemm_i64[0]=%lld, comp[0]=%lld, val=%lld, n=%d, zp=%d\n",
               (long long)gemm_i64[idx], (long long)compensation[channel], (long long)val, (int)n, zp);
    }
#endif
    
    // rshift_round for int64
    int64_t result;
    if (n <= 0) {
        result = val << (-n);
    } else {
        const int64_t offset = static_cast<int64_t>(1) << (n - 1);
        if (val >= 0) {
            result = (val + offset) >> n;
        } else {
            result = -((-val + offset) >> n);
        }
    }
    result += zp;
    
#ifdef DEBUG_QUANT
    if (debug && idx == 0) {
        printf("[DEBUG]   result after shift=%lld, final output=%lld\n", 
               result - zp, result);
    }
#endif
    
    // clamp to int32
    if (result > INT32_MAX) result = INT32_MAX;
    if (result < INT32_MIN) result = INT32_MIN;
    
    output[idx] = static_cast<int32_t>(result);
}

// computeZ: 更新门 z = sigmoid(...)
// sigmoid 输出 ∈ [0, 1]，使用无符号类型（UINT8 或 UINT16）
// QuantZ_Out: z 门输出的量化类型（uint8_t 或 uint16_t）
// 注意：Wx_val 已经在 rescaleGemmI64ToI32 中完成了 rescale，可以直接使用
//       Rh_val 仍需要在这里 rescale（因为 R@h 在每个时间步都会重新计算）
template <typename QuantZ_Out>
__device__ __forceinline__ QuantZ_Out computeZ(const int channel_idx,
                                               const int32_t Wx_val,  // 已经 rescale 后的 Wx
                                               const int32_t Rh_val,  // Rh 对应门的原始值
                                               const int64_t W_sum_mul_x_zp,  // 不再使用（Wx 已 rescale）
                                               const int64_t R_sum_mul_h_zp,  // Rh 的补偿项
                                               const int32_t bx_val,  // bx 对应门的bias
                                               const int32_t br_val,  // br 对应门的bias
                                               const QuantGRUReScale &rescale_params,
                                               const int debug_idx = -1) {
    // z = sigmoid(Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx]);

    // ★★★ Wx_val 和 Rh_val 都已经在 rescaleGemmI64ToI32 中完成了 rescale，直接使用 ★★★
    const int32_t Wx = Wx_val;  // 已经包含 zp_Wx_
    const int32_t Rh = Rh_val;  // 已经包含 zp_Rh_

    // scale_z_pre是通过效验阶段得到的;
    // 通过sigmoid函数入口前的各项相加:Wx_val+Rh_val+bx_val+br_val的结果的的最大最小值计算得到

    const int32_t Wx_shifted =
        rshift_round(Wx - rescale_params.zp_Wx_,
                     rescale_params.exp2_inv_Wx_div_z_pre_);  // n为: scale_Wx / scale_z_pre ≈ 2^-n
    const int32_t Rh_shifted =
        rshift_round(Rh - rescale_params.zp_Rh_,
                     rescale_params.exp2_inv_Rh_div_z_pre_);  // n为: scale_Rh / scale_z_pre ≈ 2^-n
    const int32_t bx_shifted = rshift_round(
        bx_val, rescale_params
                    .n_bx_div_z_[channel_idx]);  // n为: scale_bx / scale_z_pre ≈ 2^-n; bx为X的偏置
    const int32_t br_shifted = rshift_round(
        br_val, rescale_params
                    .n_br_div_z_[channel_idx]);  // n为: scale_br / scale_z_pre ≈ 2^-n; br为R的偏置

    const int32_t z_pre_i32 =
        Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale_params.zp_z_pre_;

#ifdef DEBUG_QUANT
    // 调试输出：Wx 已经 rescale，直接输出
    if (debug_idx == 0) {
        // 反量化后的 Wx 和 Rh
        float Wx_fp = (float)(Wx - rescale_params.zp_Wx_) / (float)(1 << rescale_params.test.exp2_inv_Wx_);
        float Rh_fp = (float)(Rh - rescale_params.zp_Rh_) / (float)(1 << rescale_params.test.exp2_inv_Rh_);
        // 反量化 bias (使用 device 可访问的 scale)
        float bx_fp = (float)bx_val / (float)(1 << rescale_params.exp2_inv_bx_dev_[channel_idx]);
        float br_fp = (float)br_val / (float)(1 << rescale_params.exp2_inv_br_dev_[channel_idx]);
        float z_pre_fp = (float)(z_pre_i32 - rescale_params.zp_z_pre_) / (float)(1 << rescale_params.test.exp2_inv_z_pre_);
        
        printf("[QUANT] computeZ: Wx_q=%d, Rh_q=%d, bx_q=%d, br_q=%d\n", Wx, Rh, bx_val, br_val);
        printf("[QUANT]   Wx_fp=%.6f, Rh_fp=%.6f, bx_fp=%.6f, br_fp=%.6f\n", Wx_fp, Rh_fp, bx_fp, br_fp);
        printf("[QUANT]   sum=%.6f, z_pre_q=%d, z_pre_fp=%.6f\n", 
               Wx_fp + Rh_fp + bx_fp + br_fp, z_pre_i32, z_pre_fp);
    }
#endif

    // 根据输出类型选择不同的 sigmoid 实现
    QuantZ_Out z;
    if constexpr (std::is_same_v<QuantZ_Out, uint16_t>) {
        const int16_t z_pre_i16 = dev::clamp<int16_t>(z_pre_i32);
        z = dev::sigmoid_piecewise_linear_int16(z_pre_i16, d_sigmoid_z_lut_int16);
    } else {
        const int8_t z_pre_i8 = dev::clamp<int8_t>(z_pre_i32);
        z = dev::sigmoid_piecewise_linear_int8(z_pre_i8, d_sigmoid_z_lut_int8);
    }
    
#ifdef DEBUG_QUANT
    // 调试输出：z 值
    if (debug_idx == 0) {
        float z_fp = (float)(z - rescale_params.zp_z_out_) / (float)(1 << rescale_params.test.exp2_inv_z_out_);
        printf("[QUANT]   z_q=%d, z_fp=%.6f\n", (int)z, z_fp);
    }
#endif

    return z;
}

// computeR: 重置门 r = sigmoid(...)
// sigmoid 输出 ∈ [0, 1]，使用无符号类型（UINT8 或 UINT16）
// QuantR_Out: r 门输出的量化类型（uint8_t 或 uint16_t）
// 注意：Wx_val 已经在 rescaleGemmI64ToI32 中完成了 rescale
template <typename QuantR_Out>
__device__ __forceinline__ QuantR_Out computeR(const int channel_idx,
                                               const int32_t Wx_val,  // 已经 rescale 后的 Wx
                                               const int32_t Rh_val,  // Rh 对应门的原始值
                                               const int64_t W_sum_mul_x_zp,  // 不再使用
                                               const int64_t R_sum_mul_h_zp,  // Rh 的补偿项
                                               const int32_t bx_val,  // bx 对应门的bias
                                               const int32_t br_val,  // br 对应门的bias
                                               const QuantGRUReScale &rescale_params,
                                               const int debug_idx = -1) {
    // r = sigmoid(Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx]);

    // ★★★ Wx_val 和 Rh_val 都已经在 rescaleGemmI64ToI32 中完成了 rescale，直接使用 ★★★
    const int32_t Wx = Wx_val;
    const int32_t Rh = Rh_val;

    const int32_t Wx_shifted =
        rshift_round(Wx - rescale_params.zp_Wx_,
                     rescale_params.exp2_inv_Wx_div_r_pre_);  // n为: scale_Wx / scale_r_pre ≈ 2^-n
    const int32_t Rh_shifted =
        rshift_round(Rh - rescale_params.zp_Rh_,
                     rescale_params.exp2_inv_Rh_div_r_pre_);  // n为: scale_Rh / scale_r_pre ≈ 2^-n
    const int32_t bx_shifted = rshift_round(
        bx_val, rescale_params
                    .n_bx_div_r_[channel_idx]);  // n为: scale_bx / scale_r_pre ≈ 2^-n; bx为X的偏置
    const int32_t br_shifted = rshift_round(
        br_val, rescale_params
                    .n_br_div_r_[channel_idx]);  // n为: scale_br / scale_r_pre ≈ 2^-n; br为R的偏置

    // scale_r_pre是通过校验阶段得到的;
    // 通过sigmoid函数入口前的各项相加:Wx_val+Rh_val+bx_val+br_val的结果的的最大最小值计算得到
    const int32_t r_pre_i32 =
        Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale_params.zp_r_pre_;

    // 根据输出类型选择不同的 sigmoid 实现
    QuantR_Out r;
    if constexpr (std::is_same_v<QuantR_Out, uint16_t>) {
        const int16_t r_pre_i16 = dev::clamp<int16_t>(r_pre_i32);
        r = dev::sigmoid_piecewise_linear_int16(r_pre_i16, d_sigmoid_r_lut_int16);
    } else {
        const int8_t r_pre_i8 = dev::clamp<int8_t>(r_pre_i32);  // clamp: 截断到int8的范围
        r = dev::sigmoid_piecewise_linear_int8(r_pre_i8, d_sigmoid_r_lut_int8);
    }
    
#ifdef DEBUG_QUANT
    // 调试输出
    if (debug_idx == 0) {
        float r_pre_fp = (float)(r_pre_i32 - rescale_params.zp_r_pre_) / (float)(1 << rescale_params.test.exp2_inv_r_pre_);
        float r_fp = (float)(r - rescale_params.zp_r_out_) / (float)(1 << rescale_params.test.exp2_inv_r_out_);
        printf("[QUANT] computeR: r_pre_q=%d, r_pre_fp=%.6f, r_q=%d, r_fp=%.6f\n", 
               r_pre_i32, r_pre_fp, (int)r, r_fp);
        // 输出 sigmoid r LUT 使用的参数
        if constexpr (std::is_same_v<QuantR_Out, uint16_t>) {
            printf("[QUANT] sigmoid_r LUT in use: zp_x=%d, zp_y=%d, shift_x=%d, shift_y=%d\n",
                   d_sigmoid_r_lut_int16.zp_x, d_sigmoid_r_lut_int16.zp_y,
                   d_sigmoid_r_lut_int16.shift_bits_x, d_sigmoid_r_lut_int16.shift_bits_y);
        }
    }
#endif

    //        printf("computeR: "
    //               "Wx_val = %d, "
    //               "W_sum_mul_x_zp = %d, "
    //               "Wx = %d, "
    //               "Rh_val = %d, "
    //               "R_sum_mul_h_zp = %d, "
    //               "Rh = %d, "
    //               "bx_val = %d, "
    //               "br_val = %d, "
    //               "Wx_shifted=%d, Rh_shifted=%d, bx_shifted=%d, br_shifted=%d, "
    //               "r_pre_i32 = %d, "
    //               "r_pre_i8 = %d, "
    //               "r = %d, "
    //               "r_pre_fp = %f, "
    //               "r_fp = %f"
    //               "\n",
    //               Wx_val, W_sum_mul_x_zp, Wx, Rh_val, R_sum_mul_h_zp, Rh, bx_val, br_val,
    //               Wx_shifted, Rh_shifted, bx_shifted, br_shifted, r_pre_i32, r_pre_i8, r,
    //               r_pre_fp, r_fp);
    //    }

    return r;
}

// 注意：Wx_val 已经在 rescaleGemmI64ToI32 中完成了 rescale
template <typename QuantG, typename QuantR>
__device__ __forceinline__ QuantG computeG(  // New Gate
    const int channel_idx,
    const int32_t Wx_val,  // 已经 rescale 后的 Wx
    const int32_t Rh_val,  // Rh 对应门的原始值
    const int64_t W_sum_mul_x_zp, const int64_t R_sum_mul_h_zp,  // W 的不再使用，R 的仍需要
    const int32_t bx_val,  // bx 对应门的bias
    const int32_t br_val,  // br 对应门的bias
    const QuantR r, const QuantGRUReScale &rescale_params, int32_t &Rh_add_br_g,
    const int debug_idx = -1) {
    //  g = tanh (Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx]);

    // ★★★ Wx_val 和 Rh_val 都已经在 rescaleGemmI64ToI32 中完成了 rescale，直接使用 ★★★
    const int32_t Wx = Wx_val;
    const int32_t Rh = Rh_val;
    
    Rh_add_br_g = rshift_round(Rh - rescale_params.zp_Rh_, rescale_params.n_Rh_div_Rh_add_br_) +
                  rshift_round(br_val, rescale_params.n_br_div_Rh_add_br_[channel_idx]) +
                  rescale_params.zp_Rh_add_br_;

    // 使用 int64_t 计算 r * (Rh + br) 避免溢出
    const int64_t r_diff = static_cast<int64_t>(r) - rescale_params.zp_r_out_;
    const int64_t Rh_add_br_diff = static_cast<int64_t>(Rh_add_br_g) - rescale_params.zp_Rh_add_br_;
    const int64_t rRh_mul_i64 = r_diff * Rh_add_br_diff;
    
    // ★★★ 始终使用 int64_t 计算，然后右移转回 int32 ★★★
    const int32_t rRh = static_cast<int32_t>(
        rshift_round(rRh_mul_i64, rescale_params.n_r_mul_Rh_add_br_div_rRh_)) +
        rescale_params.zp_rRh_;

    const int32_t Wx_shifted =
        rshift_round(Wx - rescale_params.zp_Wx_, rescale_params.n_Wx_div_g_pre_);
    const int32_t rRh_shifted =
        rshift_round(rRh - rescale_params.zp_rRh_, rescale_params.n_rRh_div_g_pre_);
    const int32_t bx_shifted =
        rshift_round(bx_val, rescale_params.exp2_inv_bx_div_g_pre_[channel_idx]);

    // 累加求和
    const int32_t g_pre_i32 = Wx_shifted + rRh_shifted + bx_shifted + rescale_params.zp_g_pre_;

    QuantG g;
    if constexpr (std::is_same_v<QuantG, int16_t>) {
        const int16_t g_pre_i16_linear = dev::clamp<int16_t>(g_pre_i32);
        g = static_cast<QuantG>(
            dev::tanh_piecewise_linear_int16(g_pre_i16_linear, d_tanh_lut_int16));
        
#ifdef DEBUG_QUANT
        // 调试 tanh LUT
        if (debug_idx == 0) {
            printf("[QUANT] tanh LUT debug: g_pre_i32=%d, g_pre_i16=%d, g_out=%d\n",
                   g_pre_i32, (int)g_pre_i16_linear, (int)g);
            printf("[QUANT] tanh LUT params: zp_x=%d, zp_y=%d, shift_x=%d, shift_y=%d\n",
                   d_tanh_lut_int16.zp_x, d_tanh_lut_int16.zp_y,
                   d_tanh_lut_int16.shift_bits_x, d_tanh_lut_int16.shift_bits_y);
        }
#endif
    } else {
        const int8_t g_pre_i8_linear = dev::clamp<int8_t>(g_pre_i32);
        g = static_cast<QuantG>(dev::tanh_piecewise_linear_int8(g_pre_i8_linear, d_tanh_lut_int8));
    }

#ifdef DEBUG_QUANT
    // 调试输出：使用 test 成员中的参数反量化
    if (debug_idx == 0) {
        float Rh_add_br_fp = (float)(Rh_add_br_g - rescale_params.zp_Rh_add_br_) / (float)(1 << rescale_params.test.exp2_inv_Rh_add_br_);
        float rRh_fp = (float)(rRh - rescale_params.zp_rRh_) / (float)(1 << rescale_params.test.exp2_inv_rRh_);
        float g_pre_fp = (float)(g_pre_i32 - rescale_params.zp_g_pre_) / (float)(1 << rescale_params.test.exp2_inv_g_pre_);
        float g_fp = (float)(g - rescale_params.zp_g_out_) / (float)(1 << rescale_params.test.exp2_inv_g_out_);
        printf("[QUANT] computeG: Rh_add_br_fp=%.6f, rRh_fp=%.6f, g_pre_fp=%.6f, g_fp=%.6f\n",
               Rh_add_br_fp, rRh_fp, g_pre_fp, g_fp);
    }
#endif

    //    // TODO: 分段线性量化
    //    QuantT g;
    //    if constexpr (std::is_same_v<QuantT, int16_t>) {
    //        // INT16 版本：使用分段线性拟合
    //        // g_pre_i32 已经包含了 zero-point，直接转换为 uint16_t
    //        uint16_t q_x = static_cast<uint16_t>(max(0, min(65535, g_pre_i32)));
    //        uint16_t q_y = dev::tanh_piecewise_linear_int16(q_x, d_tanh_lut_int16);
    //        // 将结果转换回 INT16
    //        g = static_cast<QuantT>(q_y);
    //    } else {
    //        // INT8 版本：使用分段线性拟合
    //        const int8_t g_pre_i8 = dev::clamp<int8_t>(g_pre_i32); // 截断到int8
    //        g = dev::tanh_piecewise_linear_int8(g_pre_i8, d_tanh_lut_int8);
    //    }

    //    const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
    //    const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本
    //    const int weight_idx = col * (rescale_params.test.hidden_ * 3) + row; // 用于访问 [Wx, Rh]
    //    的展开索引 if (weight_idx == 0) {
    //        float Wx_fp = dequant_from_exp2(Wx, rescale_params.test.exp2_inv_Wx_,
    //        rescale_params.zp_Wx_); float Rh_fp = dequant_from_exp2(Rh,
    //        rescale_params.test.exp2_inv_Rh_, rescale_params.zp_Rh_); float bx_fp =
    //        dequant_from_exp2(bx_val, rescale_params.test.exp2_inv_bx_dev_[channel_idx], 0); float
    //        br_fp = dequant_from_exp2(br_val, rescale_params.test.exp2_inv_br_dev_[channel_idx],
    //        0); float g_pre_fp = dequant_from_exp2(g_pre_i8, rescale_params.test.exp2_inv_g_pre_,
    //        rescale_params.zp_g_pre_); float g_fp = dequant_from_exp2(g,
    //        rescale_params.test.exp2_inv_g_out_, rescale_params.test.zp_g_out_); printf(
    //            "quant haste computeG: Wx_fp=%f, Rh_fp=%f, bx_fp=%f, br_fp=%f, g_pre_fp=%f, ",
    //            Wx_fp,
    //            Rh_fp,
    //            bx_fp,
    //            br_fp,
    //            g_pre_fp);
    //        printf(""
    //               "Wx_val = %d, "
    //               "W_sum_mul_x_zp = %d, "
    //               "Wx = %d, "
    //               "Rh_val = %d, "
    //               "R_sum_mul_h_zp = %d, "
    //               "Rh = %d, "
    //               "bx_val = %d, "
    //               "br_val = %d, "
    //               "Wx_shifted=%d, rRh_shifted=%d, bx_shifted=%d, "
    //               "g_pre_i32 = %d, "
    //               "g_pre_i8 = %d, "
    //               "g = %d, "
    //               "g_pre_fp = %f, "
    //               "g_fp = %f"
    //               "\n",
    //               Wx_val, W_sum_mul_x_zp, Wx, Rh_val, R_sum_mul_h_zp, Rh, bx_val, br_val,
    //               Wx_shifted, rRh_shifted, bx_shifted, g_pre_i32, g_pre_i8, g, g_pre_fp, g_fp);
    //    }

    return g;
}

template <typename QuantT, typename QuantZ, typename QuantG>
__device__ __forceinline__ QuantT computeH(  // 最终h
    const QuantZ z, const QuantG g, const QuantT h_old, const QuantGRUReScale &rescale_params,
    const int debug_idx = -1) {
    // cur_h_value = z * h[output_idx] + (1.0 - z) * g;

    // ★★★ 始终使用 int64_t 计算所有乘法，避免溢出 ★★★
    const int64_t z_diff = static_cast<int64_t>(z) - rescale_params.zp_z_out_;
    const int64_t h_diff = static_cast<int64_t>(h_old) - rescale_params.zp_h_;
    const int64_t old_contrib_mul_i64 = z_diff * h_diff;
    
    const int32_t old_contrib = static_cast<int32_t>(
        rshift_round(old_contrib_mul_i64, rescale_params.n_z_mul_h_div_old_contrib_)) +
        rescale_params.zp_old_contrib_;

    // 1-z 直接在 z_out 的量化空间计算
    const int32_t one_minus_update =
        rescale_params.one_in_z_scale_ - static_cast<int32_t>(z) + rescale_params.zp_z_out_;

    const int64_t one_minus_diff = static_cast<int64_t>(one_minus_update) - rescale_params.zp_z_out_;
    const int64_t g_diff = static_cast<int64_t>(g) - rescale_params.zp_g_out_;
    const int64_t new_contrib_mul_i64 = one_minus_diff * g_diff;
    
    const int32_t new_contrib = static_cast<int32_t>(
        rshift_round(new_contrib_mul_i64, rescale_params.n_z_out_mul_g_div_new_contrib_)) +
        rescale_params.zp_new_contrib_;
    
    const int32_t h_i32 = rshift_round(old_contrib - rescale_params.zp_old_contrib_,
                                       rescale_params.n_old_contrib_div_h_) +
                          rshift_round(new_contrib - rescale_params.zp_new_contrib_,
                                       rescale_params.n_new_contrib_div_h_) +
                          rescale_params.zp_h_;

    const QuantT h = dev::clamp<QuantT>(h_i32);

#ifdef DEBUG_QUANT
    // 调试输出：使用 test 成员中的参数反量化
    if (debug_idx == 0) {
        float z_fp = (float)(z - rescale_params.zp_z_out_) / (float)(1 << rescale_params.test.exp2_inv_z_out_);
        float g_fp = (float)(g - rescale_params.zp_g_out_) / (float)(1 << rescale_params.test.exp2_inv_g_out_);
        float h_old_fp = (float)(h_old - rescale_params.zp_h_) / (float)(1 << rescale_params.test.exp2_inv_h_);
        float old_contrib_fp = z_fp * h_old_fp;
        float one_minus_z_fp = 1.0f - z_fp;
        float new_contrib_fp = one_minus_z_fp * g_fp;
        float h_fp = (float)(h - rescale_params.zp_h_) / (float)(1 << rescale_params.test.exp2_inv_h_);
        
        printf("[QUANT] computeH: z_fp=%.6f, g_fp=%.6f, h_old_fp=%.6f\n", z_fp, g_fp, h_old_fp);
        printf("[QUANT]   old_contrib_fp=%.6f, one_minus_z_fp=%.6f, new_contrib_fp=%.6f, h_new_fp=%.6f\n",
               old_contrib_fp, one_minus_z_fp, new_contrib_fp, h_fp);
    }
#endif

    // const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
    // const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本
    // const int weight_idx = col * (rescale_params.test.hidden_ * 3) + row; // 用于访问 [Wx, Rh]
    // 的展开索引 if (weight_idx == 1) {
    //     float z_fp = dequant_from_exp2(z, rescale_params.test.exp2_inv_z_out_,
    //     rescale_params.zp_z_out_); float g_fp = dequant_from_exp2(g,
    //     rescale_params.test.exp2_inv_g_out_, rescale_params.zp_g_out_); float h_old_fp =
    //     dequant_from_exp2(h_old, rescale_params.test.exp2_inv_h_, rescale_params.zp_h_); float
    //     old_contrib_fp = dequant_from_exp2(old_contrib,
    //     rescale_params.test.exp2_inv_old_contrib_,
    //                                              rescale_params.test.zp_old_contrib_);
    //     float one_minus_update_fp = dequant_from_exp2(one_minus_update,
    //                                                   rescale_params.test.exp2_inv_one_minus_update_,
    //                                                   rescale_params.zp_one_minus_update_);
    //     float new_contrib_fp = dequant_from_exp2(new_contrib,
    //                                              rescale_params.test.exp2_inv_new_contrib_,
    //                                              rescale_params.test.zp_new_contrib_);
    //     float h_fp = dequant_from_exp2(h, rescale_params.test.exp2_inv_h_,
    //     rescale_params.test.zp_h_); printf("quant haste computeH: "
    //            "z_q = %d, "
    //            "g_q = %d, "
    //            "h_old_q = %d, "
    //            "old_contrib_q = %d, "
    //            "one_minus_update_q = %d, "
    //            "new_contrib_q = %d, "
    //            "h_q = %d, "
    //            " z_fp=%f, g_fp=%f, h_old_fp=%f, old_contrib_fp=%f, one_minus_update_fp=%f,
    //            new_contrib_fp=%f, h_fp=%f\n", z, g, h_old, old_contrib, one_minus_update,
    //            new_contrib, h, z_fp, g_fp, h_old_fp, old_contrib_fp, one_minus_update_fp,
    //            new_contrib_fp,
    //            h_fp);
    // }

    return h;
}

// x : 非对称量化, scale分时间步不同
// W : 对称量化, scale分为三个门, 分为
// R : 对称量化, scale分为三个门
// bx : 对称量化, scale分为三个门
// br : 对称量化, scale分为三个门
// h : 对称量化, scale分时间步不同
//
// C = input_size(输入维度), H = hidden_size(隐藏层维度),
// T = time_steps(时间步), N = batch_size(批量大小)
template <typename QuantT, typename QuantZ, typename QuantR, typename QuantG, bool Training,
          bool ApplyZoneout>
__global__ void PointwiseOperationsQuant(
    const int batch_dim,                     // 批量大小
    const int hidden_dim,                    // 隐藏单元数
    const int32_t *Wx,                       // 前向矩阵乘W * x, 包含Wz, Wr, Wh
    const int32_t *Rh,                       // 前向矩阵乘R * h, 包含Rz, Rr, Rh
    const int64_t *W_sum_mul_x_zp,           // hidden_size * 3（改为 int64_t）
    const int64_t *R_sum_mul_h_zp,           // hidden_size * 3（改为 int64_t）
    const int32_t *bx,                       // 输入偏置, 包含bz, br, bh
    const int32_t *br,                       // 隐藏偏置, 包含bz, br, bh
    const QuantT *h,                         // 上一时间步隐藏状态
    QuantT *h_out,                           // 当前时间步隐藏状态
    int32_t *v,                              // 保存内部分量用于反向传播 (32位存储)
    const QuantT zoneout_prob,               // Zoneout概率
    const QuantT *zoneout_mask,              // 训练模式用
    const QuantGRUReScale rescale_params) {  // Zoneout mask (only used if ApplyZoneout==true)

    /* 计算索引 */
    const int row = blockDim.x * blockIdx.x + threadIdx.x;  // 当前线程对应的隐藏单元
    const int col = blockDim.y * blockIdx.y + threadIdx.y;  // 当前线程对应的batch样本

    if (row >= hidden_dim || col >= batch_dim) return;  // 边缘判断

    const int weight_idx = col * (hidden_dim * 3) + row;  // 用于访问 [Wx, Rh] 的展开索引

    // Index into the `h` and `h_out` vectors (they have a stride of
    // `hidden_dim`).
    const int output_idx = col * hidden_dim + row;

    // Indicies into the Wx and Rh matrices (for each of the u, r, and e
    // components).
    const int z_idx = weight_idx + 0 * hidden_dim;
    const int r_idx = weight_idx + 1 * hidden_dim;
    const int g_idx = weight_idx + 2 * hidden_dim;

    // Indices into the bias vectors (for each of the u, r, and e components).
    const int b_z_idx = row + 0 * hidden_dim;  // 更新门对应索引
    const int b_r_idx = row + 1 * hidden_dim;  // 重置门对应索引
    const int b_g_idx = row + 2 * hidden_dim;  // 候选状态对应索引

    /* GRU前向计算 */

    // 用于调试输出的索引（只在第一个线程输出）
    const int debug_idx = (row == 0 && col == 0) ? 0 : -1;

    // z 和 r 门的 sigmoid 输出使用无符号类型（UINT8 或 UINT16），因为 sigmoid ∈ [0, 1]
    // QuantZ/QuantR 类型由 OperatorQuantConfig 配置决定
    const QuantZ z = computeZ<QuantZ>(b_z_idx, Wx[z_idx], Rh[z_idx], W_sum_mul_x_zp[b_z_idx],
                                      R_sum_mul_h_zp[b_z_idx], bx[b_z_idx], br[b_z_idx],
                                      rescale_params, debug_idx);  // 更新门z

    const QuantR r = computeR<QuantR>(b_r_idx, Wx[r_idx], Rh[r_idx], W_sum_mul_x_zp[b_r_idx],
                                      R_sum_mul_h_zp[b_r_idx], bx[b_r_idx], br[b_r_idx],
                                      rescale_params, debug_idx);  // 重置门r
    int32_t Rh_add_br_g;
    const QuantG g = computeG<QuantG, QuantR>(
        b_g_idx, Wx[g_idx], Rh[g_idx], W_sum_mul_x_zp[b_g_idx], R_sum_mul_h_zp[b_g_idx],
        bx[b_g_idx], br[b_g_idx], r, rescale_params, Rh_add_br_g, debug_idx);  // New Gate
    // 候选状态~ht

    /* 训练模式 */
    // Store internal activations if we're eventually going to backprop.
    // 注意: v 使用 int32_t 存储，但内部各部分原始类型可能不同
    // z, r, g 分别使用 QuantZ, QuantR, QuantG 类型，存储时直接转为 int32_t
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = static_cast<int32_t>(z);
        v[base_v_idx + 1 * hidden_dim] = static_cast<int32_t>(r);
        v[base_v_idx + 2 * hidden_dim] = static_cast<int32_t>(g);
        v[base_v_idx + 3 * hidden_dim] = Rh_add_br_g;
    }

    QuantT cur_h_value = computeH<QuantT, QuantZ, QuantG>(z, g, h[output_idx], rescale_params, debug_idx);

    /* 启用Zoneout, 对GRU 隐藏状态的随机保留 */
    // TODO: 支持量化
    //    if (ApplyZoneout) {
    //        if (Training) {
    //            cur_h_value = (cur_h_value - h[output_idx]) * zoneout_mask[output_idx] +
    //                          h[output_idx];
    //        } else {
    //            cur_h_value = (zoneout_prob * h[output_idx]) +
    //                          ((static_cast<T>(1.0) - zoneout_prob) * cur_h_value);
    //        }
    //    }

    /* 结果储存 */
    h_out[output_idx] = cur_h_value;
}

// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
//
// template<typename T, bool Training, bool ApplyZoneout>
//__global__ void PointwiseOperations(const int batch_dim, const int hidden_dim,
//                                     const half *Wx, const half *Rh,
//                                     const half *bx, const half *br,
//                                     const half *h, half *h_out, half *v,
//                                     const half zoneout_prob,
//                                     const half *zoneout_mask) {
//     device_assert_fail("FP16 is not supported on compute capability < 7.0.");
// }
//
// #endif

}  // namespace kernel

namespace gru {

template <typename XT, typename HT, typename WT, typename RT>
struct ForwardPassQuant<XT, HT, WT, RT>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[2];
    cudaEvent_t event;
    cudaStream_t sync_stream;
};

template <typename XT, typename HT, typename WT, typename RT>
ForwardPassQuant<XT, HT, WT, RT>::ForwardPassQuant(const bool training, const int batch_size,
                                                   const int input_size, const int hidden_size,
                                                   const cublasHandle_t &blas_handle,
                                                   const cudaStream_t &stream)
    : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->sync_stream = stream;
    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template <typename XT, typename HT, typename WT, typename RT>
ForwardPassQuant<XT, HT, WT, RT>::~ForwardPassQuant() {
    if (data_->sync_stream) {
        cudaEventRecord(data_->event, data_->stream[1]);
        cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
        cudaEventRecord(data_->event, data_->stream[0]);
        cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    } else {
        cudaStreamSynchronize(data_->stream[1]);
        cudaStreamSynchronize(data_->stream[0]);
    }
    cudaEventDestroy(data_->event);
    cudaStreamDestroy(data_->stream[1]);
    cudaStreamDestroy(data_->stream[0]);
    delete data_;
}

template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::Iterate(const WT *W,        // [C,H*3]
                                               const RT *R,        // [H,H*3]
                                               const int32_t *bx,  // [H*3]
                                               const int32_t *br,  // [H*3]
                                               const XT *x,        // [N,C]
                                               const HT *h,        // [N,H]
                                               HT *h_out,          // [N,H]
                                               int32_t *v,         // [N,H*4]
                                               int32_t *tmp_Wx,    // [N,H*3]
                                               int32_t *tmp_Rh,    // [N,H*3]
                                               const float zoneout_prob,
                                               const HT *zoneout_mask  // Zoneout mask [N,H]
) {
    // TODO : 支持量化
    //    using alpha_beta_t = std::conditional_t<
    //        std::is_same_v<HT, int8_t> || std::is_same_v<HT, int16_t>,
    //        int,
    //        HT>;
    //
    //    static const alpha_beta_t alpha = static_cast<alpha_beta_t>(1);
    //    static const alpha_beta_t beta = static_cast<alpha_beta_t>(0);
    //
    //    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);
    //
    //    const int batch_size = data_->batch_size;
    //    const int input_size = data_->input_size;
    //    const int hidden_size = data_->hidden_size;
    //    const cublasHandle_t blas_handle = data_->blas_handle;
    //    const cudaStream_t stream2 = data_->stream[1];
    //    const cudaEvent_t event = data_->event;
    //
    //    cudaStream_t save_stream;
    //    cublasGetStream(blas_handle, &save_stream);
    //
    //    cublasSetStream(blas_handle, stream2);
    //    blas<WT>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3,
    //                  batch_size, input_size, &alpha, W, hidden_size * 3, x,
    //                  input_size, &beta, tmp_Wx, hidden_size * 3);
    //    cudaEventRecord(event, stream2);
    //
    //    IterateInternal(R, bx, br, h, h_out, v, tmp_Wx, tmp_Rh, zoneout_prob,
    //                    zoneout_mask);
    //
    //    cublasSetStream(blas_handle, save_stream);
}

template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::IterateInternal(
    // C = input_size(输入维度), H = hidden_size(隐藏层维度),
    // T = time_steps(时间步), N = batch_size(批量大小)
    const RT *R,                  // [H,H*3]
    const int32_t *bx,            // [H*3]
    const int32_t *br,            // [H*3]
    const HT *h,                  // [N,H]
    HT *h_out,                    // [N,H]
    int32_t *v,                   // [N,H*4]
    const int32_t *tmp_Wx,        // [N,H*3]
    int32_t *tmp_Rh,              // [N,H*3]
    const int64_t *W_sum_mul_x_zp,  // hidden_size * 3（改为 int64_t）
    const int64_t *R_sum_mul_h_zp,  // hidden_size * 3（改为 int64_t）
    const float zoneout_prob,
    const HT *zoneout_mask  // Zoneout mask [N,H]
) {
    // Constants for GEMM
    static const int32_t alpha = static_cast<int32_t>(1);
    static const int32_t beta = static_cast<int32_t>(0);

    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    cublasSetStream(blas_handle, stream1);
    
    // 使用 int64 GEMM 避免 16 位量化时 int32 溢出
    static const int64_t alpha64 = 1;
    static const int64_t beta64 = 0;
    const int total_rh_size = hidden_size * 3 * batch_size;
    dev::vector<int64_t> tmp_Rh_i64(total_rh_size);
    blas<HT>::gemm_to_int64(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                            hidden_size * 3, batch_size, hidden_size,
                            &alpha64, R, hidden_size * 3, h, hidden_size, 
                            &beta64, tmp_Rh_i64.data(), hidden_size * 3);
    
    // 将 int64 GEMM 结果 rescale 为 int32：(Rh_i64 - R_sum_mul_h_zp) >> n + zp_Rh
    {
        int threads = 256;
        int blocks = (total_rh_size + threads - 1) / threads;
        static bool first_rh_call = true;
        kernel::rescaleGemmI64ToI32<<<blocks, threads, 0, stream1>>>(
            tmp_Rh_i64.data(), R_sum_mul_h_zp, 
            rescale_param_.n_R_mul_h_div_Rh_.data(),
            tmp_Rh, rescale_param_.zp_Rh_,
            hidden_size * 3, total_rh_size, first_rh_call);
        
#ifdef DEBUG_QUANT
        // 调试：输出量化 Rh GEMM 结果并反量化对比 (第一和第二时间步)
        static int rh_quant_debug_count = 0;
        if (rh_quant_debug_count < 2) {
            cudaDeviceSynchronize();
            HT h_host[5];
            int32_t tmp_Rh_host[5];
            cudaMemcpy(h_host, h, sizeof(HT) * 5, cudaMemcpyDeviceToHost);
            cudaMemcpy(tmp_Rh_host, tmp_Rh, sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);
            
            float scale_h = 1.0f / (1 << rescale_param_.test.exp2_inv_h_);
            int32_t zp_h = rescale_param_.zp_h_;
            float scale_Rh = 1.0f / (1 << rescale_param_.test.exp2_inv_Rh_);
            int32_t zp_Rh = rescale_param_.zp_Rh_;
            
            printf("[QUANT GEMM step=%d] h_q[0..4] = %d, %d, %d, %d, %d (zp=%d)\n", rh_quant_debug_count,
                   (int)h_host[0], (int)h_host[1], (int)h_host[2], (int)h_host[3], (int)h_host[4], zp_h);
            printf("[QUANT GEMM step=%d] h_fp[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n", rh_quant_debug_count,
                   (h_host[0] - zp_h) * scale_h, (h_host[1] - zp_h) * scale_h,
                   (h_host[2] - zp_h) * scale_h, (h_host[3] - zp_h) * scale_h,
                   (h_host[4] - zp_h) * scale_h);
            printf("[QUANT GEMM step=%d] Rh_q[0..4] = %d, %d, %d, %d, %d (zp=%d)\n", rh_quant_debug_count,
                   tmp_Rh_host[0], tmp_Rh_host[1], tmp_Rh_host[2], tmp_Rh_host[3], tmp_Rh_host[4], zp_Rh);
            printf("[QUANT GEMM step=%d] Rh_fp[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f\n", rh_quant_debug_count,
                   (tmp_Rh_host[0] - zp_Rh) * scale_Rh,
                   (tmp_Rh_host[1] - zp_Rh) * scale_Rh,
                   (tmp_Rh_host[2] - zp_Rh) * scale_Rh,
                   (tmp_Rh_host[3] - zp_Rh) * scale_Rh,
                   (tmp_Rh_host[4] - zp_Rh) * scale_Rh);
            rh_quant_debug_count++;
        }
#endif
        first_rh_call = false;
    }

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    // 根据 OperatorQuantConfig 中的 z_out_ 和 r_out_ 配置选择 kernel 实例
    // z_out 和 r_out 只有 UINT8 和 UINT16 两种可能（sigmoid 输出无负数）
    const auto &bw_config = rescale_param_.test.bitwidth_config_;
    const bool z_is_uint8 = (bw_config.z_out_ == QuantBitWidth::UINT8);
    const bool r_is_uint8 = (bw_config.r_out_ == QuantBitWidth::UINT8);

// 宏简化 kernel 调用（避免重复代码）
#define LAUNCH_KERNEL(QuantZ, QuantR, Training, ApplyZoneout)                                   \
    kernel::PointwiseOperationsQuant<HT, QuantZ, QuantR, HT, Training, ApplyZoneout>            \
        <<<gridDim, blockDim, 0, stream1>>>(                                                    \
            batch_size, hidden_size, tmp_Wx, tmp_Rh, W_sum_mul_x_zp, R_sum_mul_h_zp, bx, br, h, \
            h_out, Training ? v : nullptr, ApplyZoneout ? zoneout_prob : 0.0f,                  \
            ApplyZoneout ? zoneout_mask : nullptr, rescale_param_)

    // 根据配置分发到正确的 kernel 实例
    if (z_is_uint8 && r_is_uint8) {
        // z_out: UINT8, r_out: UINT8
        if (training) {
            if (zoneout_prob && zoneout_mask) {
                LAUNCH_KERNEL(uint8_t, uint8_t, true, true);
            } else {
                LAUNCH_KERNEL(uint8_t, uint8_t, true, false);
            }
        } else {
            if (zoneout_prob && zoneout_mask) {
                LAUNCH_KERNEL(uint8_t, uint8_t, false, true);
            } else {
                LAUNCH_KERNEL(uint8_t, uint8_t, false, false);
            }
        }
    } else if (z_is_uint8 && !r_is_uint8) {
        // z_out: UINT8, r_out: UINT16
        if (training) {
            if (zoneout_prob && zoneout_mask) {
                LAUNCH_KERNEL(uint8_t, uint16_t, true, true);
            } else {
                LAUNCH_KERNEL(uint8_t, uint16_t, true, false);
            }
        } else {
            if (zoneout_prob && zoneout_mask) {
                LAUNCH_KERNEL(uint8_t, uint16_t, false, true);
            } else {
                LAUNCH_KERNEL(uint8_t, uint16_t, false, false);
            }
        }
    } else if (!z_is_uint8 && r_is_uint8) {
        // z_out: UINT16, r_out: UINT8
        if (training) {
            if (zoneout_prob && zoneout_mask) {
                LAUNCH_KERNEL(uint16_t, uint8_t, true, true);
            } else {
                LAUNCH_KERNEL(uint16_t, uint8_t, true, false);
            }
        } else {
            if (zoneout_prob && zoneout_mask) {
                LAUNCH_KERNEL(uint16_t, uint8_t, false, true);
            } else {
                LAUNCH_KERNEL(uint16_t, uint8_t, false, false);
            }
        }
    } else {
        // z_out: UINT16, r_out: UINT16
        if (training) {
            if (zoneout_prob && zoneout_mask) {
                LAUNCH_KERNEL(uint16_t, uint16_t, true, true);
            } else {
                LAUNCH_KERNEL(uint16_t, uint16_t, true, false);
            }
        } else {
            if (zoneout_prob && zoneout_mask) {
                LAUNCH_KERNEL(uint16_t, uint16_t, false, true);
            } else {
                LAUNCH_KERNEL(uint16_t, uint16_t, false, false);
            }
        }
    }

#undef LAUNCH_KERNEL
}

template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::setRescaleParam(const GRUQuantitativeParameters &parms) {
    const int channel = parms.hidden_ * 3;

    std::vector<int8_t> n_W_mul_x_div_Wx(channel);
    std::vector<int8_t> n_R_mul_h_div_Rh(channel);

    // z门
    std::vector<int8_t> n_bx_to_z(channel);
    std::vector<int8_t> n_br_to_z(channel);

    // r门
    std::vector<int8_t> n_bx_to_r(channel);
    std::vector<int8_t> n_br_to_r(channel);

    // n门
    std::vector<int8_t> n_br_to_Rh_add_br(channel);
    std::vector<int8_t> n_bx_to_g(channel);

    for (int idx = 0; idx < channel; ++idx) {  // per-channel
        n_W_mul_x_div_Wx[idx] = (parms.exp2_inv_W_[idx] + parms.exp2_inv_x_) - parms.exp2_inv_Wx_;
        n_R_mul_h_div_Rh[idx] = (parms.exp2_inv_R_[idx] + parms.exp2_inv_h_) - parms.exp2_inv_Rh_;

        // z门
        n_bx_to_z[idx] = parms.exp2_inv_bx_[idx] - parms.exp2_inv_z_pre_;
        n_br_to_z[idx] = parms.exp2_inv_br_[idx] - parms.exp2_inv_z_pre_;

        // r门
        n_bx_to_r[idx] = parms.exp2_inv_bx_[idx] - parms.exp2_inv_r_pre_;
        n_br_to_r[idx] = parms.exp2_inv_br_[idx] - parms.exp2_inv_r_pre_;

        // n门
        n_br_to_Rh_add_br[idx] = parms.exp2_inv_br_[idx] - parms.exp2_inv_Rh_add_br_;
        n_bx_to_g[idx] = parms.exp2_inv_bx_[idx] - parms.exp2_inv_g_pre_;
    }

    /* init */

    rescale_param_.zp_x_ = parms.zp_x_;
    rescale_param_.zp_h_ = parms.zp_h_;
    h2d(rescale_param_.n_W_mul_x_div_Wx_, n_W_mul_x_div_Wx);
    rescale_param_.zp_Wx_ = parms.zp_Wx_;
    h2d(rescale_param_.n_R_mul_h_div_Rh_, n_R_mul_h_div_Rh);
    rescale_param_.zp_Rh_ = parms.zp_Rh_;

    // z门
    rescale_param_.zp_z_pre_ = parms.zp_z_pre_;
    rescale_param_.zp_z_out_ = parms.zp_z_out_;
    rescale_param_.exp2_inv_Wx_div_z_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_z_pre_;
    rescale_param_.exp2_inv_Rh_div_z_pre_ = parms.exp2_inv_Rh_ - parms.exp2_inv_z_pre_;
    h2d(rescale_param_.n_bx_div_z_, n_bx_to_z);
    h2d(rescale_param_.n_br_div_z_, n_br_to_z);

    // r门
    rescale_param_.zp_r_pre_ = parms.zp_r_pre_;
    rescale_param_.zp_r_out_ = parms.zp_r_out_;
    rescale_param_.exp2_inv_Wx_div_r_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_r_pre_;
    rescale_param_.exp2_inv_Rh_div_r_pre_ = parms.exp2_inv_Rh_ - parms.exp2_inv_r_pre_;
    h2d(rescale_param_.n_bx_div_r_, n_bx_to_r);
    h2d(rescale_param_.n_br_div_r_, n_br_to_r);

    // n门
    rescale_param_.zp_g_pre_ = parms.zp_g_pre_;
    rescale_param_.zp_g_out_ = parms.zp_g_out_;
    rescale_param_.n_Rh_div_Rh_add_br_ = parms.exp2_inv_Rh_ - parms.exp2_inv_Rh_add_br_;
    h2d(rescale_param_.n_br_div_Rh_add_br_, n_br_to_Rh_add_br);
    rescale_param_.zp_Rh_add_br_ = parms.zp_Rh_add_br_;
    rescale_param_.n_r_mul_Rh_add_br_div_rRh_ =
        (parms.exp2_inv_r_out_ + parms.exp2_inv_Rh_add_br_) - parms.exp2_inv_rRh_;
    rescale_param_.zp_rRh_ = parms.zp_rRh_;
    rescale_param_.n_Wx_div_g_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_g_pre_;
    rescale_param_.n_rRh_div_g_pre_ = parms.exp2_inv_rRh_ - parms.exp2_inv_g_pre_;
    h2d(rescale_param_.exp2_inv_bx_div_g_pre_, n_bx_to_g);

    // h_new
    // 1-z 直接复用 z_out 的 scale：将常数1对齐到 z_out 的量化空间
    // one_in_z_scale =
    //      round(1.0 / scale_z_out) + zp_z_out = round(1.0 * 2^exp2_inv_z_out) + zp_z_out
    rescale_param_.one_in_z_scale_ = rshift_round(1, -parms.exp2_inv_z_out_) + parms.zp_z_out_;
    rescale_param_.zp_new_contrib_ = parms.zp_new_contrib_;
    // n_z_out_mul_g_div_new_contrib = (exp2_inv_z_out + exp2_inv_g_out) - exp2_inv_new_contrib
    rescale_param_.n_z_out_mul_g_div_new_contrib_ =
        (parms.exp2_inv_z_out_ + parms.exp2_inv_g_out_) - parms.exp2_inv_new_contrib_;
    rescale_param_.zp_old_contrib_ = parms.zp_old_contrib_;
    rescale_param_.n_z_mul_h_div_old_contrib_ =
        (parms.exp2_inv_z_out_ + parms.exp2_inv_h_) - parms.exp2_inv_old_contrib_;
    rescale_param_.n_new_contrib_div_h_ = parms.exp2_inv_new_contrib_ - parms.exp2_inv_h_;
    rescale_param_.n_old_contrib_div_h_ = parms.exp2_inv_old_contrib_ - parms.exp2_inv_h_;

    // 将 bias 的 scale 拷贝到 device 可访问的 vector
    rescale_param_.exp2_inv_bx_dev_ = dev::vector<int8_t>(parms.exp2_inv_bx_);
    rescale_param_.exp2_inv_br_dev_ = dev::vector<int8_t>(parms.exp2_inv_br_);

    // test
    rescale_param_.test = parms;
}

// C = input_size(输入维度), H = hidden_size(隐藏层维度),
// T = time_steps(时间步), N = batch_size(批量大小)
template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::Run(
    const int steps,    // 时间步数, 序列长度T
    const WT *W,        // [C,H*3], 输入到隐藏状态的权重矩阵（Wx）, 对应 GRU 的三个门（z、r、h）。C
                        // 是输入特征维度，H 是隐藏状态维度, （行主序，计算 x @ W）
    const RT *R,        // [H,H*3], 隐状态到隐藏状态的权重矩阵（Rh），对应 GRU 的三个门（z、r、h）.
                        // （行主序，计算 h @ R）
    const int32_t *bx,  // [H*3], 输入偏置（bias for W），对应 z、r、h 门
    const int32_t *br,  // [H*3], 隐状态偏置（bias for R），对应 z、r、h 门
    const XT *x,        // [N,C], 输入序列，batch_size = N，特征维度 = C
    HT *h,              // [N,H], 输出隐藏状态，每个时间步保存的 GRU 隐状态
    int32_t *v,         // [N,H*4], 临时存储向量/中间计算值，通常保存 z, r, h_tilde, h_new
                        // 的中间值，用于后向传播或 zoneout (32位存储)
    int32_t *tmp_Wx,    // [N,H*3], W * x 的临时结果
    int32_t *tmp_Rh,    // [N,H*3], R * h 的临时结果
    const float zoneout_prob,  // Zoneout 概率，用于随机丢弃部分隐藏状态
    const HT
        *zoneout_mask  // Zoneout mask，0/1 矩阵，控制哪些隐藏单元被保留,  // Zoneout mask [N,H]
) {
    static const int32_t alpha = static_cast<int32_t>(1);
    static const int32_t beta = static_cast<int32_t>(0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    cublasSetStream(blas_handle, stream2);
    
    // 使用 int64 GEMM 避免 16 位量化时 int32 溢出
    static const int64_t alpha64 = 1;
    static const int64_t beta64 = 0;
    
    // 分配 int64 临时缓冲区
    const int total_wx_size = hidden_size * 3 * steps * batch_size;
    dev::vector<int64_t> tmp_Wx_i64(total_wx_size);
    blas<WT>::gemm_to_int64(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                            hidden_size * 3, steps * batch_size, input_size,
                            &alpha64, W, hidden_size * 3, x, input_size, 
                            &beta64, tmp_Wx_i64.data(), hidden_size * 3);

    // 计算W_sum_mul_zp用于补偿x_zp（使用 int64_t 避免 16 位量化时溢出）
    dev::vector<int64_t> W_sum_mul_x_zp(hidden_size * 3);
    computeWeightSumMulzp(W, W_sum_mul_x_zp.data(), rescale_param_.zp_x_,
                          rescale_param_.n_W_mul_x_div_Wx_.data(), W_sum_mul_x_zp.size(),
                          input_size, stream2);
    
    // 将 int64 GEMM 结果 rescale 为 int32：(Wx_i64 - W_sum_mul_x_zp) >> n + zp_Wx
    {
        int threads = 256;
        int blocks = (total_wx_size + threads - 1) / threads;
        kernel::rescaleGemmI64ToI32<<<blocks, threads, 0, stream2>>>(
            tmp_Wx_i64.data(), W_sum_mul_x_zp.data(), 
            rescale_param_.n_W_mul_x_div_Wx_.data(),
            tmp_Wx, rescale_param_.zp_Wx_,
            hidden_size * 3, total_wx_size, false);
        
#ifdef DEBUG_QUANT
        // 调试：输出量化 Wx GEMM 结果并反量化对比
        static bool first_wx_quant_debug = true;
        if (first_wx_quant_debug) {
            cudaDeviceSynchronize();
            int32_t tmp_Wx_host[5];
            cudaMemcpy(tmp_Wx_host, tmp_Wx, sizeof(int32_t) * 5, cudaMemcpyDeviceToHost);
            float scale_Wx = 1.0f / (1 << rescale_param_.test.exp2_inv_Wx_);
            int32_t zp_Wx = rescale_param_.zp_Wx_;
            printf("[QUANT GEMM] Wx_q[0..4] = %d, %d, %d, %d, %d (zp=%d, scale=2^-%d)\n",
                   tmp_Wx_host[0], tmp_Wx_host[1], tmp_Wx_host[2], tmp_Wx_host[3], tmp_Wx_host[4],
                   zp_Wx, rescale_param_.test.exp2_inv_Wx_);
            printf("[QUANT GEMM] Wx_fp[0..4] = %.6f, %.6f, %.6f, %.6f, %.6f (反量化)\n",
                   (tmp_Wx_host[0] - zp_Wx) * scale_Wx,
                   (tmp_Wx_host[1] - zp_Wx) * scale_Wx,
                   (tmp_Wx_host[2] - zp_Wx) * scale_Wx,
                   (tmp_Wx_host[3] - zp_Wx) * scale_Wx,
                   (tmp_Wx_host[4] - zp_Wx) * scale_Wx);
            first_wx_quant_debug = false;
        }
#endif
    }

    // Rh的gemm需要补偿h_zp, 所以提前计算 h_zp * R_sum * h_zp, stream1（使用 int64_t）
    dev::vector<int64_t> R_sum_mul_h_zp(hidden_size * 3);
    computeWeightSumMulzp(R, R_sum_mul_h_zp.data(), rescale_param_.zp_h_,
                          rescale_param_.n_R_mul_h_div_Rh_.data(), R_sum_mul_h_zp.size(),
                          hidden_size, stream2);

    // 同步Wx计算
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh, W_sum_mul_x_zp.data(), R_sum_mul_h_zp.data(),
                        zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr);
        //        if (i >= 2) { break; }
    }

    cublasSetStream(blas_handle, save_stream);
}

// 显式实例化：四个类型参数相同的情况
template struct ForwardPassQuant<int8_t, int8_t, int8_t, int8_t>;
template struct ForwardPassQuant<int16_t, int16_t, int16_t, int16_t>;

}  // namespace gru
