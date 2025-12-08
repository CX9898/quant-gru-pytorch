#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <vector>

#include "blas.h"
#include "devVector.h"
#include "device_ptr.h"
#include "gru_quant.h"
#include "quantize_bitwidth_config.hpp"
#include "quantize_ops.cuh"
#include "quantize_ops_helper.hpp"

namespace kernel {

//#define USE_Piecewise_linear_quantization

template<typename QuantT>
__device__ __forceinline__ QuantT computeZ(// 更新门z
    const int channel_idx,
    const int32_t Wx_val,// Wx 对应门的值
    const int32_t Rh_val,// Rh 对应门的值
    const int32_t W_sum_mul_x_zp,
    const int32_t R_sum_mul_h_zp,
    const int32_t bx_val,// bx 对应门的bias
    const int32_t br_val,// br 对应门的bias
    const QuantGRUReScale &rescale_params) {
    // z = sigmoid(Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx]);

    // TODO: 优化计算
    const int32_t Wx =
        rshift_round(Wx_val - W_sum_mul_x_zp, rescale_params.n_W_mul_x_div_Wx_[channel_idx]) + rescale_params.zp_Wx_;
    const int32_t Rh =
        rshift_round(Rh_val - R_sum_mul_h_zp, rescale_params.n_R_mul_h_div_Rh_[channel_idx]) + rescale_params.zp_Rh_;

    // scale_z_pre是通过效验阶段得到的; 通过sigmoid函数入口前的各项相加:Wx_val+Rh_val+bx_val+br_val的结果的的最大最小值计算得到

    const int32_t Wx_shifted = rshift_round(Wx - rescale_params.zp_Wx_,
                                            rescale_params.exp2_inv_Wx_div_z_pre_);// n为: scale_Wx / scale_z_pre ≈ 2^-n
    const int32_t Rh_shifted = rshift_round(Rh - rescale_params.zp_Rh_,
                                            rescale_params.exp2_inv_Rh_div_z_pre_);// n为: scale_Rh / scale_z_pre ≈ 2^-n
    const int32_t bx_shifted =
        rshift_round(bx_val, rescale_params.n_bx_div_z_[channel_idx]);// n为: scale_bx / scale_z_pre ≈ 2^-n; bx为X的偏置
    const int32_t br_shifted =
        rshift_round(br_val, rescale_params.n_br_div_z_[channel_idx]);// n为: scale_br / scale_z_pre ≈ 2^-n; br为R的偏置

    const int32_t z_pre_i32 = Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale_params.zp_z_pre_;

    const int8_t z_pre_i8 = dev::clamp<int8_t>(z_pre_i32);  // clamp: 截断到int8的范围（输入始终是int8）

    // 根据输出类型选择正确的 LUT
    QuantT z;
    if constexpr (std::is_same_v<QuantT, uint8_t>) {
        // uint8 输出：使用 uint8 sigmoid LUT
        z = dev::sigmoid_uint8_lut(z_pre_i8, d_sigmoid_uint8_z_lut);
    } else {
        // int8/int16 输出：使用 int8 sigmoid LUT
        z = static_cast<QuantT>(dev::sigmoid_int8_lut(z_pre_i8, d_sigmoid_int8_z_lut));
    }

#ifdef USE_Piecewise_linear_quantization
    // TODO: 分段线性量化
    if constexpr (std::is_same_v<QuantT, int16_t>) {
        // INT16 版本：使用分段线性拟合（z 门）
        // z_pre_i32 已经包含了 zero-point，是量化后的值（无符号域）
        uint16_t q_x = static_cast<uint16_t>(max(0, min(65535, z_pre_i32)));
        uint16_t q_y = dev::sigmoid_piecewise_linear_int16(q_x, d_sigmoid_z_lut_int16);
        // 输出 q_y 已经是 uint16_t，直接转换
        z = static_cast<QuantT>(q_y);
    } else {
        // INT8 版本：使用分段线性拟合（z 门）
        // z_pre_i32 已经包含了 zero-point，是量化后的值（无符号域）
        // 🔥 修复：直接转换为 uint8_t，而不是先转换为 int8_t
        uint8_t q_x = static_cast<uint8_t>(max(0, min(255, z_pre_i32)));
        uint8_t q_y = dev::sigmoid_piecewise_linear_int8(q_x, d_sigmoid_z_lut_int8);
        // 输出 q_y 已经是 uint8_t，直接转换
        z = static_cast<QuantT>(q_y);
    }
#endif

//     const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
//     const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本
//     const int weight_idx = col * (rescale_params.test.hidden_ * 3) + row; // 用于访问 [Wx, Rh] 的展开索引
//     if (weight_idx == 1) {
//         float Wx_fp = dequantize(Wx, rescale_params.test.exp2_inv_Wx_, rescale_params.zp_Wx_);
//         float Rh_fp = dequantize(Rh, rescale_params.test.exp2_inv_Rh_, rescale_params.zp_Rh_);
//         float bx_fp = dequantize(bx_val, rescale_params.test.exp2_inv_bx_dev_[channel_idx], 0);
//         float br_fp = dequantize(br_val, rescale_params.test.exp2_inv_br_dev_[channel_idx], 0);
//         float z_pre_fp = dequantize(z_pre_i8, rescale_params.test.exp2_inv_z_pre_, rescale_params.zp_z_pre_);
//         float Wx_shifted_fp = dequantize(Wx_shifted,
//                                                 rescale_params.exp2_inv_Wx_div_z_pre_,
//                                                 rescale_params.zp_Wx_);
//         float Rh_shifted_fp = dequantize(Rh_shifted,
//                                                 rescale_params.exp2_inv_Rh_div_z_pre_,
//                                                 rescale_params.zp_Rh_);
//         float bx_shifted_fp = dequantize(bx_shifted, rescale_params.n_bx_div_z_[channel_idx], 0);
//         float br_shifted_fp = dequantize(br_shifted, rescale_params.n_br_div_z_[channel_idx], 0);
//         float z_fp = dequantize(z, rescale_params.test.exp2_inv_z_out_, rescale_params.test.zp_z_out_);
//         printf("quant haste computeZ: "
//                "Wx_fp=%f, Rh_fp=%f, bx_fp=%f, br_fp=%f, z_pre_fp=%f, z_out_fp=%f "
//                "Wx_q = %d, "
//                "Rh_q = %d, "
//                "z_pre_i32_q = %d, "
//                "z_pre_i8_q = %d, "
//                "z_out_q = %d"
//                "\n",
//                Wx_fp, Rh_fp, bx_fp, br_fp, z_pre_fp, z_fp,
//                Wx, Rh, z_pre_i32, z_pre_i8, z);
//     }


    return z;
}

template<typename QuantT>
__device__ __forceinline__ QuantT computeR(// 重置门r
    const int channel_idx,
    const int32_t Wx_val,// Wx 对应门的值
    const int32_t Rh_val,// Rh 对应门的值
    const int32_t W_sum_mul_x_zp,
    const int32_t R_sum_mul_h_zp,
    const int32_t bx_val,// bx 对应门的bias
    const int32_t br_val,// br 对应门的bias
    const QuantGRUReScale &rescale_params) {
    // r = sigmoid(Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx]);

    // n为: (scale_W * scale_x) / scale_Wx ≈ 2^-n
    const int32_t Wx =
        rshift_round(Wx_val - W_sum_mul_x_zp, rescale_params.n_W_mul_x_div_Wx_[channel_idx]) + rescale_params.zp_Wx_;
    // n为: (scale_R * scale_h) / scale_Rh ≈ 2^-n
    const int32_t Rh =
        rshift_round(Rh_val - R_sum_mul_h_zp, rescale_params.n_R_mul_h_div_Rh_[channel_idx]) + rescale_params.zp_Rh_;

    const int32_t Wx_shifted = rshift_round(Wx - rescale_params.zp_Wx_,
                                            rescale_params.exp2_inv_Wx_div_r_pre_);// n为: scale_Wx / scale_r_pre ≈ 2^-n
    const int32_t Rh_shifted = rshift_round(Rh - rescale_params.zp_Rh_,
                                            rescale_params.exp2_inv_Rh_div_r_pre_);// n为: scale_Rh / scale_r_pre ≈ 2^-n
    const int32_t bx_shifted =
        rshift_round(bx_val, rescale_params.n_bx_div_r_[channel_idx]);// n为: scale_bx / scale_r_pre ≈ 2^-n; bx为X的偏置
    const int32_t br_shifted =
        rshift_round(br_val, rescale_params.n_br_div_r_[channel_idx]);// n为: scale_br / scale_r_pre ≈ 2^-n; br为R的偏置

    // scale_z_pre是通过效验阶段得到的; 通过sigmoid函数入口前的各项相加:Wx_val+Rh_val+bx_val+br_val的结果的的最大最小值计算得到
    const int32_t r_pre_i32 = Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale_params.zp_r_pre_;

    const int8_t r_pre_i8 = dev::clamp<int8_t>(r_pre_i32);  // clamp: 截断到int8的范围（输入始终是int8）

    // 根据输出类型选择正确的 LUT
    QuantT r;
    if constexpr (std::is_same_v<QuantT, uint8_t>) {
        // uint8 输出：使用 uint8 sigmoid LUT
        r = dev::sigmoid_uint8_lut(r_pre_i8, d_sigmoid_uint8_r_lut);
    } else {
        // int8/int16 输出：使用 int8 sigmoid LUT
        r = static_cast<QuantT>(dev::sigmoid_int8_lut(r_pre_i8, d_sigmoid_int8_r_lut));
    }

#ifdef USE_Piecewise_linear_quantization
    // TODO: 分段线性量化
    if constexpr (std::is_same_v<QuantT, int16_t>) {
        // INT16 版本：使用分段线性拟合（r 门）
        // r_pre_i32 已经包含了 zero-point，是量化后的值（无符号域）
        uint16_t q_x = static_cast<uint16_t>(max(0, min(65535, r_pre_i32)));
        uint16_t q_y = dev::sigmoid_piecewise_linear_int16(q_x, d_sigmoid_r_lut_int16);
        // 输出 q_y 已经是 uint16_t，直接转换
        r = static_cast<QuantT>(q_y);
    } else {
        // INT8 版本：使用分段线性拟合（r 门）
        // r_pre_i32 已经包含了 zero-point，是量化后的值（无符号域）
        // 🔥 修复：直接转换为 uint8_t，而不是先转换为 int8_t
        uint8_t q_x = static_cast<uint8_t>(max(0, min(255, r_pre_i32)));
        uint8_t q_y = dev::sigmoid_piecewise_linear_int8(q_x, d_sigmoid_r_lut_int8);
        // 输出 q_y 已经是 uint8_t，直接转换
        r = static_cast<QuantT>(q_y);
    }
#endif

//        const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
//        const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本
//        const int weight_idx = col * (rescale_params.test.hidden_ * 3) + row; // 用于访问 [Wx, Rh] 的展开索引
//        if (weight_idx == 0) {
//            float Wx_fp = dequantize(Wx, rescale_params.test.exp2_inv_Wx_, rescale_params.zp_Wx_);
//            float Rh_fp = dequantize(Rh, rescale_params.test.exp2_inv_Rh_, rescale_params.zp_Rh_);
//            float bx_fp = dequantize(bx_val, rescale_params.test.exp2_inv_bx_dev_[channel_idx], 0);
//            float br_fp = dequantize(br_val, rescale_params.test.exp2_inv_br_dev_[channel_idx], 0);
//            float r_pre_fp = dequantize(r_pre_i8, rescale_params.test.exp2_inv_r_pre_, rescale_params.zp_r_pre_);
//            float Wx_shifted_fp = dequantize(Wx_shifted,
//                                                    rescale_params.exp2_inv_Wx_div_r_pre_,
//                                                    rescale_params.zp_Wx_);
//            float Rh_shifted_fp = dequantize(Rh_shifted,
//                                                    rescale_params.exp2_inv_Rh_div_r_pre_,
//                                                    rescale_params.zp_Rh_);
//            float bx_shifted_fp = dequantize(bx_shifted, rescale_params.n_bx_div_r_[channel_idx], 0);
//            float br_shifted_fp = dequantize(br_shifted, rescale_params.n_br_div_r_[channel_idx], 0);
//            float r_fp = dequantize(r, rescale_params.test.exp2_inv_r_out_, rescale_params.test.zp_r_out_);
//            printf(
//                "quant haste compute R: Wx_fp=%f, Rh_fp=%f, bx_fp=%f, br_fp=%f, r_pre_fp=%f, r_fp=%f "
//                "Wx_shifted_fp=%f, Rh_shifted_fp=%f, bx_shifted_fp=%f, br_shifted_fp=%f\n",
//                Wx_fp,
//                Rh_fp,
//                bx_fp,
//                br_fp,
//                r_pre_fp,
//                r_fp,
//                Wx_shifted_fp,
//                Rh_shifted_fp,
//                bx_shifted_fp,
//                br_shifted_fp);
//        }

    return r;
}

template<typename GT,    // 候选状态 g 输出的类型
         typename RT>    // 重置门 r 的类型
__device__ __forceinline__ GT computeG(// New Gate
    const int channel_idx,
    const int32_t Wx_val,// Wx 对应门的值
    const int32_t Rh_val,// Rh 对应门的值
    const int32_t W_sum_mul_x_zp,
    const int32_t R_sum_mul_h_zp,
    const int32_t bx_val,// bx 对应门的bias
    const int32_t br_val,// br 对应门的bias
    const RT r,
    const QuantGRUReScale &rescale_params) {
    //  g = tanh (Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx]);

    const int32_t Wx =
        rshift_round(Wx_val - W_sum_mul_x_zp, rescale_params.n_W_mul_x_div_Wx_[channel_idx]) + rescale_params.zp_Wx_;
    const int32_t Rh =
        rshift_round(Rh_val - R_sum_mul_h_zp, rescale_params.n_R_mul_h_div_Rh_[channel_idx]) + rescale_params.zp_Rh_;
    const int32_t Rh_add_br_g = rshift_round(Rh - rescale_params.zp_Rh_, rescale_params.n_Rh_div_Rh_add_br_) +
                                rshift_round(br_val, rescale_params.n_br_div_Rh_add_br_[channel_idx]) +
                                rescale_params.zp_Rh_add_br_;

    const int32_t rRh = rshift_round(
                            (r - rescale_params.zp_r_out_) * (Rh_add_br_g - rescale_params.zp_Rh_add_br_),
                            rescale_params.n_r_mul_Rh_add_br_div_rRh_) +
                        rescale_params.zp_rRh_;

    const int32_t Wx_shifted = rshift_round(Wx - rescale_params.zp_Wx_, rescale_params.n_Wx_div_g_pre_);
    const int32_t rRh_shifted = rshift_round(rRh - rescale_params.zp_rRh_, rescale_params.n_rRh_div_g_pre_);
    const int32_t bx_shifted = rshift_round(bx_val, rescale_params.exp2_inv_bx_div_g_pre_[channel_idx]);

    // 累加求和
    const int32_t g_pre_i32 = Wx_shifted + rRh_shifted + bx_shifted + rescale_params.zp_g_pre_;

    const GT g_pre_i8 = dev::clamp<GT>(g_pre_i32);         // 截断到int8
    GT g = dev::tanh_int8_lut(g_pre_i8, d_tanh_int8_g_lut);// TODO: 支持int16量化

#ifdef USE_Piecewise_linear_quantization
    // TODO: 分段线性量化
    if constexpr (std::is_same_v<GT, int16_t>) {
        // INT16 版本：使用分段线性拟合
        // g_pre_i32 已经包含了 zero-point，是量化后的值（无符号域）
        uint16_t q_x = static_cast<uint16_t>(max(0, min(65535, g_pre_i32)));
        uint16_t q_y = dev::tanh_piecewise_linear_int16(q_x, d_tanh_lut_int16);
        // 输出 q_y 已经是 uint16_t，直接转换
        g = static_cast<GT>(q_y);
    } else {
        // INT8 版本：使用分段线性拟合
        // g_pre_i32 已经包含了 zero-point，是量化后的值（无符号域）
        // 🔥 修复：直接转换为 uint8_t，而不是先转换为 int8_t
        uint8_t q_x = static_cast<uint8_t>(max(0, min(255, g_pre_i32)));
        uint8_t q_y = dev::tanh_piecewise_linear_int8(q_x, d_tanh_lut_int8);
        // 输出 q_y 已经是 uint8_t，直接转换
        g = static_cast<GT>(q_y);
    }
#endif

    //    const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
    //    const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本
    //    const int weight_idx = col * (rescale_params.test.hidden_ * 3) + row; // 用于访问 [Wx, Rh] 的展开索引
    //    if (weight_idx == 0) {
    //        float Wx_fp = dequant_from_exp2(Wx, rescale_params.test.exp2_inv_Wx_, rescale_params.zp_Wx_);
    //        float Rh_fp = dequant_from_exp2(Rh, rescale_params.test.exp2_inv_Rh_, rescale_params.zp_Rh_);
    //        float bx_fp = dequant_from_exp2(bx_val, rescale_params.test.exp2_inv_bx_dev_[channel_idx], 0);
    //        float br_fp = dequant_from_exp2(br_val, rescale_params.test.exp2_inv_br_dev_[channel_idx], 0);
    //        float g_pre_fp = dequant_from_exp2(g_pre_i8, rescale_params.test.exp2_inv_g_pre_, rescale_params.zp_g_pre_);
    //        float g_fp = dequant_from_exp2(g, rescale_params.test.exp2_inv_g_out_, rescale_params.test.zp_g_out_);
    //        printf(
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

template<typename ZT,      // 更新门 z 的类型
         typename GT,      // 候选状态 g 的类型
         typename HT>      // 隐藏状态的类型（输入输出相同）
__device__ __forceinline__ HT computeH(// 最终h
    const ZT z,
    const GT g,
    const HT h_old,
    const QuantGRUReScale &rescale_params) {
    // cur_h_value = z * h[output_idx] + (1.0 - z) * g;

    const int32_t old_contrib = rshift_round((z - rescale_params.zp_z_out_) * (h_old - rescale_params.zp_h_),
                                             rescale_params.n_z_mul_h_div_old_contrib_) +
                                rescale_params.zp_old_contrib_;

    const int32_t one_minus_update = rescale_params.one_div_one_minus_update_ -
                                     rshift_round(z - rescale_params.zp_z_out_,
                                                  rescale_params.n_z_out_div_one_minus_update_) +
                                     rescale_params.zp_one_minus_update_;
    const int32_t new_contrib =
        rshift_round((one_minus_update - rescale_params.zp_one_minus_update_) * (g - rescale_params.zp_g_out_),
                     rescale_params.n_one_minus_update_mul_g_div_new_contrib_) +
        rescale_params.zp_new_contrib_;
    const int32_t h_i32 =
        rshift_round(old_contrib - rescale_params.zp_old_contrib_, rescale_params.n_old_contrib_div_h_) +
        rshift_round(new_contrib - rescale_params.zp_new_contrib_, rescale_params.n_new_contrib_div_h_) +
        rescale_params.zp_h_;

    const HT h = dev::clamp<HT>(h_i32);

    // const int row = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程对应的隐藏单元
    // const int col = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程对应的batch样本
    // const int weight_idx = col * (rescale_params.test.hidden_ * 3) + row; // 用于访问 [Wx, Rh] 的展开索引
    // if (weight_idx == 1) {
    //     float z_fp = dequant_from_exp2(z, rescale_params.test.exp2_inv_z_out_, rescale_params.zp_z_out_);
    //     float g_fp = dequant_from_exp2(g, rescale_params.test.exp2_inv_g_out_, rescale_params.zp_g_out_);
    //     float h_old_fp = dequant_from_exp2(h_old, rescale_params.test.exp2_inv_h_, rescale_params.zp_h_);
    //     float old_contrib_fp = dequant_from_exp2(old_contrib, rescale_params.test.exp2_inv_old_contrib_,
    //                                              rescale_params.test.zp_old_contrib_);
    //     float one_minus_update_fp = dequant_from_exp2(one_minus_update,
    //                                                   rescale_params.test.exp2_inv_one_minus_update_,
    //                                                   rescale_params.zp_one_minus_update_);
    //     float new_contrib_fp = dequant_from_exp2(new_contrib,
    //                                              rescale_params.test.exp2_inv_new_contrib_,
    //                                              rescale_params.test.zp_new_contrib_);
    //     float h_fp = dequant_from_exp2(h, rescale_params.test.exp2_inv_h_, rescale_params.test.zp_h_);
    //     printf("quant haste computeH: "
    //            "z_q = %d, "
    //            "g_q = %d, "
    //            "h_old_q = %d, "
    //            "old_contrib_q = %d, "
    //            "one_minus_update_q = %d, "
    //            "new_contrib_q = %d, "
    //            "h_q = %d, "
    //            " z_fp=%f, g_fp=%f, h_old_fp=%f, old_contrib_fp=%f, one_minus_update_fp=%f, new_contrib_fp=%f, h_fp=%f\n",
    //            z, g, h_old, old_contrib, one_minus_update, new_contrib, h,
    //            z_fp,
    //            g_fp,
    //            h_old_fp,
    //            old_contrib_fp,
    //            one_minus_update_fp,
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
template<typename ZT,           // 更新门 z 的类型
         typename RT,           // 重置门 r 的类型
         typename GT,           // 候选状态 g 的类型
         typename HT,           // 隐藏状态 h 的类型（输入和输出相同）
         typename VT,           // 内部分量 v 的类型
         bool Training, bool ApplyZoneout>
__global__ void PointwiseOperationsQuant(
    const int batch_dim,                   // 批量大小
    const int hidden_dim,                  // 隐藏单元数
    const int32_t *Wx,                     // 前向矩阵乘W * x, 包含Wz, Wr, Wh
    const int32_t *Rh,                     // 前向矩阵乘R * h, 包含Rz, Rr, Rh
    const int32_t *W_sum_mul_x_zp,         // hidden_size * 3
    const int32_t *R_sum_mul_h_zp,         // hidden_size * 3
    const int32_t *bx,                     // 输入偏置, 包含bz, br, bh
    const int32_t *br,                     // 隐藏偏置, 包含bz, br, bh
    const HT *h,                           // 上一时间步隐藏状态
    HT *h_out,                             // 当前时间步隐藏状态
    VT *v,                                 // 保存内部分量用于反向传播
    const float zoneout_prob,              // Zoneout概率
    const HT *zoneout_mask,                // Zoneout mask (only used if ApplyZoneout==true)
    const QuantGRUReScale rescale_params) {

    /* 计算索引 */
    const int row = blockDim.x * blockIdx.x + threadIdx.x;// 当前线程对应的隐藏单元
    const int col = blockDim.y * blockIdx.y + threadIdx.y;// 当前线程对应的batch样本

    if (row >= hidden_dim || col >= batch_dim) return;// 边缘判断

    const int weight_idx = col * (hidden_dim * 3) + row;// 用于访问 [Wx, Rh] 的展开索引

    // Index into the `h` and `h_out` vectors (they have a stride of
    // `hidden_dim`).
    const int output_idx = col * hidden_dim + row;

    // Indicies into the Wx and Rh matrices (for each of the u, r, and e
    // components).
    const int z_idx = weight_idx + 0 * hidden_dim;
    const int r_idx = weight_idx + 1 * hidden_dim;
    const int g_idx = weight_idx + 2 * hidden_dim;

    // Indices into the bias vectors (for each of the u, r, and e components).
    const int b_z_idx = row + 0 * hidden_dim;// 更新门对应索引
    const int b_r_idx = row + 1 * hidden_dim;// 重置门对应索引
    const int b_g_idx = row + 2 * hidden_dim;// 候选状态对应索引

    /* GRU前向计算 */

    const ZT z = computeZ<ZT>(b_z_idx,
                              Wx[z_idx],
                              Rh[z_idx],
                              W_sum_mul_x_zp[b_z_idx],
                              R_sum_mul_h_zp[b_z_idx],
                              bx[b_z_idx],
                              br[b_z_idx],
                              rescale_params);// 更新门z

    const RT r = computeR<RT>(b_r_idx,
                              Wx[r_idx],
                              Rh[r_idx],
                              W_sum_mul_x_zp[b_r_idx],
                              R_sum_mul_h_zp[b_r_idx],
                              bx[b_r_idx],
                              br[b_r_idx],
                              rescale_params);// 重置门r

    const GT g = computeG<GT, RT>(b_g_idx,
                                  Wx[g_idx],
                                  Rh[g_idx],
                                  W_sum_mul_x_zp[b_g_idx],
                                  R_sum_mul_h_zp[b_g_idx],
                                  bx[b_g_idx],
                                  br[b_g_idx],
                                  r,
                                  rescale_params);// New Gate
    // 候选状态~ht


    /* 训练模式 */
    // Store internal activations if we're eventually going to backprop.
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = static_cast<VT>(z);
        v[base_v_idx + 1 * hidden_dim] = static_cast<VT>(r);
        v[base_v_idx + 2 * hidden_dim] = static_cast<VT>(g);
        const VT Rh_add_br_g = rshift_round(Rh[g_idx] - rescale_params.zp_Rh_, rescale_params.n_Rh_div_Rh_add_br_) +
                               rshift_round(br[b_g_idx], rescale_params.n_br_div_Rh_add_br_[b_g_idx]) +
                               rescale_params.zp_Rh_add_br_;

        v[base_v_idx + 3 * hidden_dim] = Rh_add_br_g;
    }

    HT cur_h_value = computeH<ZT, GT, HT>(z, g, h[output_idx], rescale_params);

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

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
//
//template<typename T, bool Training, bool ApplyZoneout>
//__global__ void PointwiseOperations(const int batch_dim, const int hidden_dim,
//                                    const half *Wx, const half *Rh,
//                                    const half *bx, const half *br,
//                                    const half *h, half *h_out, half *v,
//                                    const half zoneout_prob,
//                                    const half *zoneout_mask) {
//    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
//}
//
//#endif

}// namespace kernel


namespace gru {

template<typename T>
struct ForwardPassQuant<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[2];
    cudaEvent_t event;
    cudaStream_t sync_stream;
};

template<typename T>
ForwardPassQuant<T>::ForwardPassQuant(const bool training, const int batch_size,
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

template<typename T>
ForwardPassQuant<T>::~ForwardPassQuant() {
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

template<typename T>
void ForwardPassQuant<T>::Iterate(const T *W,       // [C,H*3]
                                  const T *R,       // [H,H*3]
                                  const int32_t *bx,// [H*3]
                                  const int32_t *br,// [H*3]
                                  const T *x,       // [N,C]
                                  const T *h,       // [N,H]
                                  T *h_out,         // [N,H]
                                  T *v,             // [N,H*4]
                                  int32_t *tmp_Wx,  // [N,H*3]
                                  int32_t *tmp_Rh,  // [N,H*3]
                                  const float zoneout_prob,
                                  const T *zoneout_mask// Zoneout mask [N,H]
) {
    // TODO : 支持量化
    //    using alpha_beta_t = std::conditional_t<
    //        std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>,
    //        int,
    //        T>;
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
    //    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3,
    //                  batch_size, input_size, &alpha, W, hidden_size * 3, x,
    //                  input_size, &beta, tmp_Wx, hidden_size * 3);
    //    cudaEventRecord(event, stream2);
    //
    //    IterateInternal(R, bx, br, h, h_out, v, tmp_Wx, tmp_Rh, zoneout_prob,
    //                    zoneout_mask);
    //
    //    cublasSetStream(blas_handle, save_stream);
}

template<typename QuantT>
void ForwardPassQuant<QuantT>::IterateInternal(
    // C = input_size(输入维度), H = hidden_size(隐藏层维度),
    // T = time_steps(时间步), N = batch_size(批量大小)
    const QuantT *R,          // [H,H*3]
    const int32_t *bx,        // [H*3]
    const int32_t *br,        // [H*3]
    const QuantT *h,          // [N,H]
    QuantT *h_out,            // [N,H]
    QuantT *v,                // [N,H*4]
    const int32_t *tmp_Wx,    // [N,H*3]
    int32_t *tmp_Rh,          // [N,H*3]
    const int *W_sum_mul_x_zp,// hidden_size * 3
    const int *R_sum_mul_h_zp,// hidden_size * 3
    const float zoneout_prob,
    const QuantT *zoneout_mask// Zoneout mask [N,H]
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
    blas<QuantT>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3,
                       batch_size, hidden_size, &alpha, R, hidden_size * 3, h,
                       hidden_size, &beta, tmp_Rh, hidden_size * 3);

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    // 根据 bitwidth_config_ 选择 kernel 模板参数
    // 使用模式匹配而非完全展开，避免模板膨胀（5^4 × 4 = 2500 个实例）
    // 只支持常用配置：全 INT8、全 INT16、混合精度
    const auto& cfg = bitwidth_config_;

    // 辅助宏简化 kernel 调用（添加类型转换）
    // 模板参数: ZT, RT, GT, HT, VT, Training, ApplyZoneout
    #define LAUNCH_KERNEL(ZT, RT, GT, HT, TRAINING, ZONEOUT) \
        kernel::PointwiseOperationsQuant<ZT, RT, GT, HT, HT, TRAINING, ZONEOUT> \
            <<<gridDim, blockDim, 0, stream1>>>( \
            batch_size, hidden_size, tmp_Wx, tmp_Rh, W_sum_mul_x_zp, R_sum_mul_h_zp, bx, br, \
            reinterpret_cast<const HT*>(h), reinterpret_cast<HT*>(h_out), \
            (TRAINING) ? reinterpret_cast<HT*>(v) : nullptr, \
            (ZONEOUT) ? zoneout_prob : 0.0f, \
            (ZONEOUT) ? reinterpret_cast<const HT*>(zoneout_mask) : nullptr, \
            rescale_param_)

    // 检测配置模式
    auto isAllInt8 = [&cfg]() {
        return cfg.z_out_bitwidth == QuantBitWidth::INT8 &&
               cfg.r_out_bitwidth == QuantBitWidth::INT8 &&
               cfg.g_out_bitwidth == QuantBitWidth::INT8 &&
               cfg.h_bitwidth == QuantBitWidth::INT8;
    };

    auto isAllInt16 = [&cfg]() {
        return cfg.z_out_bitwidth == QuantBitWidth::INT16 &&
               cfg.r_out_bitwidth == QuantBitWidth::INT16 &&
               cfg.g_out_bitwidth == QuantBitWidth::INT16 &&
               cfg.h_bitwidth == QuantBitWidth::INT16;
    };

    // 混合精度：门用 INT8，候选状态用 INT16
    auto isMixedPrecision = [&cfg]() {
        return cfg.z_out_bitwidth == QuantBitWidth::INT8 &&
               cfg.r_out_bitwidth == QuantBitWidth::INT8 &&
               cfg.g_out_bitwidth == QuantBitWidth::INT16 &&
               cfg.h_bitwidth == QuantBitWidth::INT8;
    };

    // INT8 + UINT8 sigmoid：sigmoid 输出（z/r）用 UINT8，其他用 INT8
    // sigmoid 输出范围 [0,1]，使用 UINT8 [0,255] 更合适
    auto isInt8WithUint8Sigmoid = [&cfg]() {
        return cfg.z_out_bitwidth == QuantBitWidth::UINT8 &&
               cfg.r_out_bitwidth == QuantBitWidth::UINT8 &&
               cfg.g_out_bitwidth == QuantBitWidth::INT8 &&
               cfg.h_bitwidth == QuantBitWidth::INT8;
    };

    const bool useZoneout = zoneout_prob && zoneout_mask;

    if (isAllInt8()) {
        // 全 INT8 配置（最常用）
        if (training) {
            if (useZoneout) {
                LAUNCH_KERNEL(int8_t, int8_t, int8_t, int8_t, true, true);
            } else {
                LAUNCH_KERNEL(int8_t, int8_t, int8_t, int8_t, true, false);
            }
        } else {
            if (useZoneout) {
                LAUNCH_KERNEL(int8_t, int8_t, int8_t, int8_t, false, true);
            } else {
                LAUNCH_KERNEL(int8_t, int8_t, int8_t, int8_t, false, false);
            }
        }
    } else if (isAllInt16()) {
        // 全 INT16 配置
        if (training) {
            if (useZoneout) {
                LAUNCH_KERNEL(int16_t, int16_t, int16_t, int16_t, true, true);
            } else {
                LAUNCH_KERNEL(int16_t, int16_t, int16_t, int16_t, true, false);
            }
        } else {
            if (useZoneout) {
                LAUNCH_KERNEL(int16_t, int16_t, int16_t, int16_t, false, true);
            } else {
                LAUNCH_KERNEL(int16_t, int16_t, int16_t, int16_t, false, false);
            }
        }
    } else if (isMixedPrecision()) {
        // 混合精度：z/r 用 INT8，g 用 INT16
        if (training) {
            if (useZoneout) {
                LAUNCH_KERNEL(int8_t, int8_t, int16_t, int8_t, true, true);
            } else {
                LAUNCH_KERNEL(int8_t, int8_t, int16_t, int8_t, true, false);
            }
        } else {
            if (useZoneout) {
                LAUNCH_KERNEL(int8_t, int8_t, int16_t, int8_t, false, true);
            } else {
                LAUNCH_KERNEL(int8_t, int8_t, int16_t, int8_t, false, false);
            }
        }
    } else if (isInt8WithUint8Sigmoid()) {
        // INT8 + UINT8 sigmoid：sigmoid 输出（z/r）用 UINT8，tanh 和 h 用 INT8
        if (training) {
            if (useZoneout) {
                LAUNCH_KERNEL(uint8_t, uint8_t, int8_t, int8_t, true, true);
            } else {
                LAUNCH_KERNEL(uint8_t, uint8_t, int8_t, int8_t, true, false);
            }
        } else {
            if (useZoneout) {
                LAUNCH_KERNEL(uint8_t, uint8_t, int8_t, int8_t, false, true);
            } else {
                LAUNCH_KERNEL(uint8_t, uint8_t, int8_t, int8_t, false, false);
            }
        }
    } else {
        // 回退到默认配置（根据 QuantT 模板参数）
        if (training) {
            if (useZoneout) {
                LAUNCH_KERNEL(QuantT, QuantT, QuantT, QuantT, true, true);
            } else {
                LAUNCH_KERNEL(QuantT, QuantT, QuantT, QuantT, true, false);
            }
        } else {
            if (useZoneout) {
                LAUNCH_KERNEL(QuantT, QuantT, QuantT, QuantT, false, true);
            } else {
                LAUNCH_KERNEL(QuantT, QuantT, QuantT, QuantT, false, false);
            }
        }
    }

    #undef LAUNCH_KERNEL
}

template<typename T>
void ForwardPassQuant<T>::setRescaleParam(const GRUQuantitativeParameters &parms) {
    const int channel = parms.hidden_ * 3;

    std::vector<int32_t> n_W_mul_x_div_Wx(channel);
    std::vector<int32_t> n_R_mul_h_div_Rh(channel);

    // z门
    std::vector<int32_t> n_bx_to_z(channel);
    std::vector<int32_t> n_br_to_z(channel);

    // r门
    std::vector<int32_t> n_bx_to_r(channel);
    std::vector<int32_t> n_br_to_r(channel);

    // n门
    std::vector<int32_t> n_br_to_Rh_add_br(channel);
    std::vector<int32_t> n_bx_to_g(channel);

    for (int idx = 0; idx < channel; ++idx) {// per-channel
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
    rescale_param_.n_Rh_div_Rh_add_br_ =
        parms.exp2_inv_Rh_ - parms.exp2_inv_Rh_add_br_;
    h2d(rescale_param_.n_br_div_Rh_add_br_, n_br_to_Rh_add_br);
    rescale_param_.zp_Rh_add_br_ = parms.zp_Rh_add_br_;
    rescale_param_.n_r_mul_Rh_add_br_div_rRh_ =
        (parms.exp2_inv_r_out_ + parms.exp2_inv_Rh_add_br_) - parms.exp2_inv_rRh_;
    rescale_param_.zp_rRh_ = parms.zp_rRh_;
    rescale_param_.n_Wx_div_g_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_g_pre_;
    rescale_param_.n_rRh_div_g_pre_ = parms.exp2_inv_rRh_ - parms.exp2_inv_g_pre_;
    h2d(rescale_param_.exp2_inv_bx_div_g_pre_, n_bx_to_g);

    // h_new
    rescale_param_.one_div_one_minus_update_ = rshift_round(1, -parms.exp2_inv_one_minus_update_);
    rescale_param_.n_z_out_div_one_minus_update_ =
        parms.exp2_inv_z_out_ - parms.exp2_inv_one_minus_update_;
    rescale_param_.zp_one_minus_update_ = parms.zp_one_minus_update_;
    rescale_param_.zp_new_contrib_ = parms.zp_new_contrib_;
    rescale_param_.n_one_minus_update_mul_g_div_new_contrib_ =
        (parms.exp2_inv_one_minus_update_ + parms.exp2_inv_g_out_) - parms.exp2_inv_new_contrib_;
    rescale_param_.zp_old_contrib_ = parms.zp_old_contrib_;
    rescale_param_.n_z_mul_h_div_old_contrib_ =
        (parms.exp2_inv_z_out_ + parms.exp2_inv_h_) - parms.exp2_inv_old_contrib_;
    rescale_param_.n_new_contrib_div_h_ = parms.exp2_inv_new_contrib_ - parms.exp2_inv_h_;
    rescale_param_.n_old_contrib_div_h_ = parms.exp2_inv_old_contrib_ - parms.exp2_inv_h_;

    // TODO delete test
    rescale_param_.test = parms;
    h2d(rescale_param_.test.exp2_inv_bx_dev_, parms.exp2_inv_bx_);
    h2d(rescale_param_.test.exp2_inv_br_dev_, parms.exp2_inv_br_);

    // 保存位宽配置
    bitwidth_config_ = parms.bitwidth_config_;
}

// C = input_size(输入维度), H = hidden_size(隐藏层维度),
// T = time_steps(时间步), N = batch_size(批量大小)
template<typename QuantT>
void ForwardPassQuant<QuantT>::Run(const int steps,          // 时间步数, 序列长度T
                                   const QuantT *W,          // [C,H*3], 输入到隐藏状态的权重矩阵（Wx）, 对应 GRU 的三个门（z、r、h）。C 是输入特征维度，H 是隐藏状态维度, （行主序，计算 x @ W）
                                   const QuantT *R,          // [H,H*3], 隐状态到隐藏状态的权重矩阵（Rh），对应 GRU 的三个门（z、r、h）. （行主序，计算 h @ R）
                                   const int32_t *bx,        // [H*3], 输入偏置（bias for W），对应 z、r、h 门
                                   const int32_t *br,        // [H*3], 隐状态偏置（bias for R），对应 z、r、h 门
                                   const QuantT *x,          // [N,C], 输入序列，batch_size = N，特征维度 = C
                                   QuantT *h,                // [N,H], 输出隐藏状态，每个时间步保存的 GRU 隐状态
                                   QuantT *v,                // [N,H*4], 临时存储向量/中间计算值，通常保存 z, r, h_tilde, h_new 的中间值，用于后向传播或 zoneout
                                   int32_t *tmp_Wx,          // [N,H*3], W * x 的临时结果
                                   int32_t *tmp_Rh,          // [N,H*3], R * h 的临时结果
                                   const float zoneout_prob, // Zoneout 概率，用于随机丢弃部分隐藏状态
                                   const QuantT *zoneout_mask// Zoneout mask，0/1 矩阵，控制哪些隐藏单元被保留,  // Zoneout mask [N,H]
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
    blas<QuantT>::gemm(blas_handle,// 提前使用cuBlas计算W * x
                       CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, steps * batch_size,
                       input_size, &alpha, W, hidden_size * 3, x, input_size, &beta,
                       tmp_Wx, hidden_size * 3);

    // 计算W_sum_mul_zp用于补偿x_zp
    dev::vector<int32_t> W_sum_mul_x_zp(hidden_size * 3);
    computeWeightSumMulzp(W,
                          W_sum_mul_x_zp.data(),
                          rescale_param_.zp_x_,
                          rescale_param_.n_W_mul_x_div_Wx_.data(),
                          W_sum_mul_x_zp.size(),
                          input_size,
                          stream2);

    // Rh的gemm需要补偿h_zp, 所以提前计算 h_zp * R_sum * h_zp, stream1
    dev::vector<int32_t> R_sum_mul_h_zp(hidden_size * 3);
    computeWeightSumMulzp(R,
                          R_sum_mul_h_zp.data(),
                          rescale_param_.zp_h_,
                          rescale_param_.n_R_mul_h_div_Rh_.data(),
                          R_sum_mul_h_zp.size(),
                          hidden_size,
                          stream1);

    // 同步Wx计算
    cudaEventRecord(event, stream2);

    // 同步R_sum_mul_h_zp计算
    cudaEventRecord(event, stream1);

    const int NH = batch_size * hidden_size;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH, v + i * NH * 4,
                        tmp_Wx + i * NH * 3, tmp_Rh, W_sum_mul_x_zp.data(), R_sum_mul_h_zp.data(), zoneout_prob,
                        zoneout_mask ? zoneout_mask + i * NH : nullptr);
        //        if (i >= 2) { break; }
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPassQuant<int8_t>;
template struct ForwardPassQuant<int16_t>;

}// namespace gru
