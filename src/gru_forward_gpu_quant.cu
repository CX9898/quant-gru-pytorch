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

// ============================================================================
// 融合的 INT16 量化 GEMM kernel（边计算边减 zero-point）
// C[m,n] = rshift_round(sum_k(A[m,k] * (B[k,n] - zp_B)), shift[m]) + zp_out
// ============================================================================
constexpr int TILE_SIZE = 16;  // 每个线程块处理 16x16 的输出 tile

template <typename AT, typename BT>
__global__ void quantizedGemmInt16Fused(
    const AT* __restrict__ A,       // [M, K] 权重（W 或 R），行主序
    const BT* __restrict__ B,       // [K, N] 输入（x 或 h），列主序（cuBLAS 风格）
    int32_t* __restrict__ C,        // [M, N] 输出，列主序
    int M, int N, int K,
    int32_t zp_B,                   // 输入的 zero-point
    const int8_t* __restrict__ shift_per_row,  // [M] per-row shift
    int32_t zp_out                  // 输出的 zero-point
) {
    // 共享内存：用于 tiled 矩阵乘法
    __shared__ int32_t As[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict
    __shared__ int32_t Bs[TILE_SIZE][TILE_SIZE + 1];

    // 计算当前线程负责的输出位置
    // 注意：cuBLAS 使用列主序，所以 A 是 [M,K] 行主序，B 是 [K,N] 列主序
    // 这里 A 实际存储为 A[k*M + m]（列主序转置），B 存储为 B[n*K + k]
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // m in [0, M)
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // n in [0, N)

    int64_t acc = 0;  // 使用 int64 累加，避免溢出

    // 分 tile 计算
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 加载 A tile（A 是列主序：A[k*M + m]）
        const int aK = t * TILE_SIZE + threadIdx.x;
        if (row < M && aK < K) {
            As[threadIdx.y][threadIdx.x] = static_cast<int32_t>(A[aK * M + row]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // 加载 B tile 并减去 zp_B（B 是列主序：B[n*K + k]）
        const int bK = t * TILE_SIZE + threadIdx.y;
        if (col < N && bK < K) {
            // 核心：边加载边减 zero-point
            Bs[threadIdx.y][threadIdx.x] = static_cast<int32_t>(B[col * K + bK]) - zp_B;
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // 计算当前 tile 的贡献
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += static_cast<int64_t>(As[threadIdx.y][k]) * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回结果：rshift_round + zp_out
    if (row < M && col < N) {
        int8_t n = shift_per_row[row];
        int64_t result;

        // rshift_round for int64
        if (n <= 0) {
            result = acc << (-n);
        } else {
            const int64_t offset = static_cast<int64_t>(1) << (n - 1);
            if (acc >= 0) {
                result = (acc + offset) >> n;
            } else {
                result = -((-acc + offset) >> n);
            }
        }
        result += zp_out;

        // clamp to INT16 range
        if (result > 32767) result = 32767;
        if (result < -32768) result = -32768;

        // 输出是列主序：C[n*M + m]
        C[col * M + row] = static_cast<int32_t>(result);
    }
}

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

    // clamp to INT16 range（因为这是 INT16 专用 kernel）
    if (result > 32767) result = 32767;
    if (result < -32768) result = -32768;

    output[idx] = static_cast<int32_t>(result);
}

// INT8 专用：将 int32 GEMM 结果原地 rescale
// output[i] = clamp((gemm_i32[i] - compensation[i % hidden3]) >> shift[i % hidden3] + zp, INT8)
__global__ void rescaleGemmI32(
    int32_t* __restrict__ data,               // [hidden*3, batch*steps] GEMM 输出（原地修改）
    const int64_t* __restrict__ compensation, // [hidden*3] W_sum_mul_x_zp
    const int8_t* __restrict__ shift,         // [hidden*3] per-channel shift
    int32_t zp,                               // zero point
    int hidden3,                              // hidden_size * 3
    int total_size                            // hidden*3 * batch*steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int channel = idx % hidden3;
    int64_t val = static_cast<int64_t>(data[idx]) - compensation[channel];
    int8_t n = shift[channel];

    // rshift_round
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

    // clamp to INT8 range（因为这是 INT8 专用 kernel）
    if (result > 127) result = 127;
    if (result < -128) result = -128;

    data[idx] = static_cast<int32_t>(result);
}

// computeZ: 更新门 z = sigmoid(...)
// sigmoid 输出 ∈ [0, 1]，使用无符号类型（UINT8 或 UINT16）
// QuantZ_Out: z 门输出的量化类型（uint8_t 或 uint16_t）
// 注意：Wx_val 和 Rh_val 都已在 rescaleGemmI64ToI32 中完成 rescale 和补偿
template <typename QuantZ_Out>
__device__ __forceinline__ QuantZ_Out computeZ(const int channel_idx,
                                               const int32_t Wx_val,  // 已经 rescale 后的 Wx
                                               const int32_t Rh_val,  // 已经 rescale 后的 Rh
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
// 注意：Wx_val 和 Rh_val 都已在 rescaleGemmI64ToI32 中完成 rescale 和补偿
template <typename QuantR_Out>
__device__ __forceinline__ QuantR_Out computeR(const int channel_idx,
                                               const int32_t Wx_val,  // 已经 rescale 后的 Wx
                                               const int32_t Rh_val,  // 已经 rescale 后的 Rh
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

// 注意：Wx_val 和 Rh_val 都已在 rescaleGemmI64ToI32 中完成 rescale 和补偿
template <typename QuantG, typename QuantR>
__device__ __forceinline__ QuantG computeG(  // New Gate
    const int channel_idx,
    const int32_t Wx_val,  // 已经 rescale 后的 Wx
    const int32_t Rh_val,  // 已经 rescale 后的 Rh
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
    const int32_t *Wx,                       // 前向矩阵乘W * x, 包含Wz, Wr, Wh（已 rescale）
    const int32_t *Rh,                       // 前向矩阵乘R * h, 包含Rz, Rr, Rh（已 rescale）
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
    // 注意：Wx 和 Rh 已在 rescaleGemmI64ToI32 中完成 rescale 和补偿
    const QuantZ z = computeZ<QuantZ>(b_z_idx, Wx[z_idx], Rh[z_idx],
                                      bx[b_z_idx], br[b_z_idx],
                                      rescale_params, debug_idx);  // 更新门z

    const QuantR r = computeR<QuantR>(b_r_idx, Wx[r_idx], Rh[r_idx],
                                      bx[b_r_idx], br[b_r_idx],
                                      rescale_params, debug_idx);  // 重置门r
    int32_t Rh_add_br_g;
    const QuantG g = computeG<QuantG, QuantR>(
        b_g_idx, Wx[g_idx], Rh[g_idx],
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
    // dev::vector 自动管理内存，无需手动释放
}

template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::EnsureBuffersAllocated(int steps) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;

    // 如果已分配且足够大，直接返回
    if (steps <= max_steps_) {
        return;
    }

    // 使用 dev::vector::resize 自动管理内存
    // GEMM rescale 后的结果（int32）- 两种位宽都需要
    tmp_Wx_.resize(hidden3 * steps * batch_size);
    tmp_Rh_.resize(hidden3 * batch_size);

    if constexpr (sizeof(WT) == 1) {
        // INT8: 需要权重和常量用于 rescaleGemmI32
        // 注意：INT8 的 cuBLAS GEMM 直接输出 int32，不需要 int64 中间存储
        if (W_sum_mul_x_zp_.size() == 0) {
            W_sum_mul_x_zp_.resize(hidden3);
            R_sum_mul_h_zp_.resize(hidden3);
        }
    }
    // INT16: 使用融合 kernel，不需要权重和预计算

    max_steps_ = steps;
    weight_sums_computed_ = false;  // 需要重新计算
}

template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::PrecomputeWeightSums(const WT *W, const RT *R) {
    // INT16 使用融合 kernel，不需要预计算权重和
    if constexpr (sizeof(WT) != 1) {
        weight_sums_computed_ = true;
        return;
    }

    // INT8: 需要预计算 W_sum_mul_x_zp 和 R_sum_mul_h_zp
    // 如果权重变化，需要重新计算
    if (cached_W_ != W || cached_R_ != R) {
        weight_sums_computed_ = false;
        cached_W_ = W;
        cached_R_ = R;
    }

    if (weight_sums_computed_) return;

    const int hidden_size = data_->hidden_size;
    const int input_size = data_->input_size;
    const cudaStream_t stream = data_->stream[1];

    // 计算 W_sum_mul_x_zp
    computeWeightSumMulzp(W, W_sum_mul_x_zp_.data(), rescale_param_.zp_x_,
                          rescale_param_.n_W_mul_x_div_Wx_.data(), hidden_size * 3,
                          input_size, stream);

    // 计算 R_sum_mul_h_zp
    computeWeightSumMulzp(R, R_sum_mul_h_zp_.data(), rescale_param_.zp_h_,
                          rescale_param_.n_R_mul_h_div_Rh_.data(), hidden_size * 3,
                          hidden_size, stream);

    cudaStreamSynchronize(stream);
    weight_sums_computed_ = true;
}

template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::ComputeWx(const WT *W, const XT *x, int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream = data_->stream[1];
    const int total_size = hidden_size * 3 * steps * batch_size;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    if constexpr (sizeof(WT) == 1) {
        // INT8: 直接调用 cuBLAS GEMM 输出 INT32（不会溢出）
        static const int32_t alpha32 = 1;
        static const int32_t beta32 = 0;

        // GEMM: W @ x -> tmp_Wx_ (直接输出 int32)
        blas<WT>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       hidden_size * 3, steps * batch_size, input_size,
                       &alpha32, W, hidden_size * 3, x, input_size,
                       &beta32, tmp_Wx_.data(), hidden_size * 3);

        // Rescale: (Wx_i32 - W_sum_mul_x_zp) >> n + zp_Wx（原地操作）
        kernel::rescaleGemmI32<<<blocks, threads, 0, stream>>>(
            tmp_Wx_.data(), W_sum_mul_x_zp_.data(),
            rescale_param_.n_W_mul_x_div_Wx_.data(),
            rescale_param_.zp_Wx_,
            hidden_size * 3, total_size);
    } else {
        // INT16: 使用融合的量化 GEMM（边算边减 zp，避免中间 int64 存储）
        // C[m,n] = rshift_round(sum_k(W[m,k] * (x[k,n] - zp_x)), shift[m]) + zp_Wx
        const int M = hidden_size * 3;
        const int N = steps * batch_size;
        const int K = input_size;

        dim3 blockDim(kernel::TILE_SIZE, kernel::TILE_SIZE);
        dim3 gridDim((N + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE,
                     (M + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE);

        kernel::quantizedGemmInt16Fused<WT, XT><<<gridDim, blockDim, 0, stream>>>(
            W, x, tmp_Wx_.data(),
            M, N, K,
            rescale_param_.zp_x_,
            rescale_param_.n_W_mul_x_div_Wx_.data(),
            rescale_param_.zp_Wx_);
    }
}

template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::ComputeRh(const RT *R, const HT *h) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream = data_->stream[0];
    const int total_size = hidden_size * 3 * batch_size;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    if constexpr (sizeof(HT) == 1) {
        // INT8: 直接调用 cuBLAS GEMM 输出 INT32（不会溢出）
        static const int32_t alpha32 = 1;
        static const int32_t beta32 = 0;

        // GEMM: R @ h -> tmp_Rh_ (直接输出 int32)
        blas<HT>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       hidden_size * 3, batch_size, hidden_size,
                       &alpha32, R, hidden_size * 3, h, hidden_size,
                       &beta32, tmp_Rh_.data(), hidden_size * 3);

        // Rescale: (Rh_i32 - R_sum_mul_h_zp) >> n + zp_Rh（原地操作）
        kernel::rescaleGemmI32<<<blocks, threads, 0, stream>>>(
            tmp_Rh_.data(), R_sum_mul_h_zp_.data(),
            rescale_param_.n_R_mul_h_div_Rh_.data(),
            rescale_param_.zp_Rh_,
            hidden_size * 3, total_size);
    } else {
        // INT16: 使用融合的量化 GEMM（边算边减 zp，避免中间 int64 存储）
        // C[m,n] = rshift_round(sum_k(R[m,k] * (h[k,n] - zp_h)), shift[m]) + zp_Rh
        const int M = hidden_size * 3;
        const int N = batch_size;
        const int K = hidden_size;

        dim3 blockDim(kernel::TILE_SIZE, kernel::TILE_SIZE);
        dim3 gridDim((N + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE,
                     (M + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE);

        kernel::quantizedGemmInt16Fused<RT, HT><<<gridDim, blockDim, 0, stream>>>(
            R, h, tmp_Rh_.data(),
            M, N, K,
            rescale_param_.zp_h_,
            rescale_param_.n_R_mul_h_div_Rh_.data(),
            rescale_param_.zp_Rh_);
    }
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
    const int32_t *cur_Wx_,       // [N,H*3] 当前时间步的 W @ x 结果
    const float zoneout_prob,
    const HT *zoneout_mask  // Zoneout mask [N,H]
) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    cublasSetStream(blas_handle, stream1);

    // 计算 R @ h GEMM（结果存入 tmp_Rh_）
    ComputeRh(R, h);

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
// 注意：使用内部缓冲区 cur_Wx_ 和 tmp_Rh_.data()
#define LAUNCH_KERNEL(QuantZ, QuantR, Training, ApplyZoneout)                                   \
    kernel::PointwiseOperationsQuant<HT, QuantZ, QuantR, HT, Training, ApplyZoneout>            \
        <<<gridDim, blockDim, 0, stream1>>>(                                                    \
            batch_size, hidden_size, cur_Wx_, tmp_Rh_.data(), bx, br, h,                        \
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
    const WT *W,        // [C,H*3], 输入到隐藏状态的权重矩阵（Wx）
    const RT *R,        // [H,H*3], 隐状态到隐藏状态的权重矩阵（Rh）
    const int32_t *bx,  // [H*3], 输入偏置（bias for W）
    const int32_t *br,  // [H*3], 隐状态偏置（bias for R）
    const XT *x,        // [N*T,C], 输入序列
    HT *h,              // [(T+1)*N,H], 输出隐藏状态
    int32_t *v,         // [T*N,H*4], 中间激活值（训练模式需要）
    const float zoneout_prob,  // Zoneout 概率
    const HT *zoneout_mask     // Zoneout mask [T*N,H]
) {
    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    // 预分配缓冲区（只在第一次调用或 steps 增大时分配）
    EnsureBuffersAllocated(steps);

    // 预计算权重和（权重不变时只计算一次）
    PrecomputeWeightSums(W, R);

    cudaStream_t save_stream;
    cublasGetStream(data_->blas_handle, &save_stream);

    cublasSetStream(data_->blas_handle, stream2);

    // 计算 W @ x GEMM（所有时间步一次性计算，结果存入 tmp_Wx_）
    ComputeWx(W, x, steps);

    // 同步 Wx 计算
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, bx, br,
                        h + i * NH,           // 输入 h
                        h + (i + 1) * NH,     // 输出 h
                        v + i * NH * 4,       // 中间激活
                        tmp_Wx_.data() + i * NH3,    // 当前时间步的 Wx
                        zoneout_prob,
                        zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    cublasSetStream(data_->blas_handle, save_stream);
}

// 显式实例化：四个类型参数相同的情况
template struct ForwardPassQuant<int8_t, int8_t, int8_t, int8_t>;
template struct ForwardPassQuant<int16_t, int16_t, int16_t, int16_t>;

}  // namespace gru
