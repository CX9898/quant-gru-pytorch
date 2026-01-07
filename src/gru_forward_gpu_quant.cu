// ============================================================================
// gru_forward_gpu_quant.cu - 量化 GRU 前向传播 CUDA 实现
// ============================================================================
//
// 文件结构:
//   1. GEMM Kernels        - 量化矩阵乘法 (INT32 存储)
//   2. Rescale Kernels     - GEMM 结果缩放
//   3. GRU Gate Functions  - 门计算函数 (computeZ/R/G/H)
//   4. Pointwise Kernel    - GRU 逐点运算主 kernel
//   5. ForwardPassQuant    - 前向传播封装类
//
// 量化方案:
//   - 所有量化值使用 int32_t 统一存储
//   - 实际值通过 clamp_by_bitwidth 限制到配置的位宽范围
//   - 通过 bitwidth_config_ 枚举动态选择对应位宽的处理
//
// ============================================================================

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "blas.h"
#include "dev_vector.h"
#include "gru_quant.h"
#include "quantize_ops_helper.h"

namespace kernel {

// 调试开关
// #define DEBUG_QUANT           // 启用量化调试输出
// #define DEBUG_QUANT_DETAIL    // 启用详细量化调试（含理论值对比）

// 调试：浮点 sigmoid 和 tanh（用于对比）
__device__ __forceinline__ float sigmoid_fp(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float tanh_fp(float x) { return tanhf(x); }

// ============================================================================
// 1. GEMM Kernels - 量化矩阵乘法 (int32_t 存储)
// ============================================================================

constexpr int TILE_SIZE = 16;

// 统一融合 GEMM: C = rshift(A * (B - zp_B), shift) + zp_out
// A, B, C 都使用 int32_t 存储，实际值通过位宽配置限制
__global__ void quantizedGemmFused(const int32_t *__restrict__ A,  // [M, K] 权重，列主序
                                   const int32_t *__restrict__ B,  // [K, N] 输入，列主序
                                   int32_t *__restrict__ C,        // [M, N] 输出，列主序
                                   int M, int N, int K,
                                   int32_t zp_B,                              // 输入的 zero-point
                                   const int8_t *__restrict__ shift_per_row,  // [M] per-row shift
                                   int32_t zp_out,                            // 输出的 zero-point
                                   QuantBitWidth output_bw                    // 输出位宽配置
) {
    // 使用 int64_t 累加器避免溢出
    __shared__ int32_t As[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict
    __shared__ int32_t Bs[TILE_SIZE][TILE_SIZE + 1];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // m in [0, M)
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // n in [0, N)

    int64_t acc = 0;

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 加载 A tile（列主序：A[k*M + m]）
        const int aK = t * TILE_SIZE + threadIdx.x;
        if (row < M && aK < K) {
            As[threadIdx.y][threadIdx.x] = A[aK * M + row];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // 加载 B tile 并减去 zp_B（列主序：B[n*K + k]）
        const int bK = t * TILE_SIZE + threadIdx.y;
        if (col < N && bK < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + bK] - zp_B;
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += static_cast<int64_t>(As[threadIdx.y][k]) * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回结果：rshift_round + zp_out + clamp
    if (row < M && col < N) {
        const int8_t n = shift_per_row[row];
        int64_t result;

        // rshift_round
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

        // 根据位宽配置 clamp 并输出（列主序：C[n*M + m]）
        C[col * M + row] = clamp_by_bitwidth(static_cast<int32_t>(result), output_bw);
    }
}

// 将 int32 GEMM 结果原地 rescale（根据位宽配置自动 clamp）
__global__ void rescaleGemmI32(
    int32_t *__restrict__ data,                // [hidden*3, batch*steps] GEMM 输出（原地修改）
    const int64_t *__restrict__ compensation,  // [hidden*3] W_sum_mul_x_zp
    const int8_t *__restrict__ shift,          // [hidden*3] per-channel shift
    int32_t zp,                                // zero point
    int hidden3,                               // hidden_size * 3
    int total_size,                            // hidden*3 * batch*steps
    QuantBitWidth output_bw                    // 输出位宽配置
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

    // 根据位宽配置 clamp
    data[idx] = clamp_by_bitwidth(static_cast<int32_t>(result), output_bw);
}

// ============================================================================
// 3. GRU Gate Functions - 门计算函数
// ============================================================================
// 所有中间值使用 int32_t 存储，通过 bitwidth_config_ 选择对应位宽的 LUT 和 clamp

// z = sigmoid(Wx + Rh + bx + br) - 更新门
__device__ __forceinline__ int32_t computeZ(const int channel_idx, const int32_t Wx_val,
                                            const int32_t Rh_val, const int32_t bx_val,
                                            const int32_t br_val,
                                            const QuantGRUReScale &rescale_params,
                                            const int debug_idx = -1) {
    const int32_t Wx_shifted =
        rshift_round(Wx_val - rescale_params.zp_Wx_, rescale_params.exp2_inv_Wx_div_z_pre_);
    const int32_t Rh_shifted =
        rshift_round(Rh_val - rescale_params.zp_Rh_, rescale_params.exp2_inv_Rh_div_z_pre_);
    const int32_t bx_shifted = rshift_round(bx_val, rescale_params.n_bx_div_z_[channel_idx]);
    const int32_t br_shifted = rshift_round(br_val, rescale_params.n_br_div_z_[channel_idx]);

    const int32_t z_pre_i32 =
        Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale_params.zp_z_pre_;

    // 使用参数中的 LUT（避免全局 LUT 覆盖问题）
    const auto &bw_cfg = rescale_params.bitwidth_config_;
    const int32_t z = piecewise_linear(z_pre_i32, rescale_params.sigmoid_z_lut_,
                                        bw_cfg.z_pre_, bw_cfg.z_out_);

#ifdef DEBUG_QUANT
    if (debug_idx == 0) {
        float z_pre_fp = (float)(z_pre_i32 - rescale_params.zp_z_pre_) /
                         (float)(1 << rescale_params.test.exp2_inv_z_pre_);
        float z_fp = (float)(z - rescale_params.zp_z_out_) /
                     (float)(1 << rescale_params.test.exp2_inv_z_out_);
        printf("[QUANT_I32] computeZ: z_pre_q=%d, z_pre_fp=%.6f, z_q=%d, z_fp=%.6f\n", z_pre_i32,
               z_pre_fp, z, z_fp);
    }
#endif

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0 && debug_idx < 3) {
        // 反量化 z_pre
        float z_pre_fp = (float)(z_pre_i32 - rescale_params.zp_z_pre_) /
                         (float)(1 << rescale_params.test.exp2_inv_z_pre_);
        // 反量化 sigmoid 输出
        float z_quant_fp = (float)(z - rescale_params.zp_z_out_) /
                           (float)(1 << rescale_params.test.exp2_inv_z_out_);
        // 理论 sigmoid 值
        float z_theory = sigmoid_fp(z_pre_fp);
        // 误差
        float error = z_quant_fp - z_theory;
        printf("[Z] idx=%d z_pre_q=%d z_pre_fp=%.4f | z_q=%d z_fp=%.4f | theory=%.4f | err=%.6f\n",
               debug_idx, z_pre_i32, z_pre_fp, z, z_quant_fp, z_theory, error);
    }
#endif

    return z;
}

// r = sigmoid(Wx + Rh + bx + br) - 重置门
__device__ __forceinline__ int32_t computeR(const int channel_idx, const int32_t Wx_val,
                                            const int32_t Rh_val, const int32_t bx_val,
                                            const int32_t br_val,
                                            const QuantGRUReScale &rescale_params,
                                            const int debug_idx = -1) {
    const int32_t Wx_shifted =
        rshift_round(Wx_val - rescale_params.zp_Wx_, rescale_params.exp2_inv_Wx_div_r_pre_);
    const int32_t Rh_shifted =
        rshift_round(Rh_val - rescale_params.zp_Rh_, rescale_params.exp2_inv_Rh_div_r_pre_);
    const int32_t bx_shifted = rshift_round(bx_val, rescale_params.n_bx_div_r_[channel_idx]);
    const int32_t br_shifted = rshift_round(br_val, rescale_params.n_br_div_r_[channel_idx]);

    const int32_t r_pre_i32 =
        Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale_params.zp_r_pre_;

    // 使用参数中的 LUT（避免全局 LUT 覆盖问题）
    const auto &bw_cfg = rescale_params.bitwidth_config_;
    const int32_t r = piecewise_linear(r_pre_i32, rescale_params.sigmoid_r_lut_,
                                        bw_cfg.r_pre_, bw_cfg.r_out_);

#ifdef DEBUG_QUANT
    if (debug_idx == 0) {
        float r_pre_fp = (float)(r_pre_i32 - rescale_params.zp_r_pre_) /
                         (float)(1 << rescale_params.test.exp2_inv_r_pre_);
        float r_fp = (float)(r - rescale_params.zp_r_out_) /
                     (float)(1 << rescale_params.test.exp2_inv_r_out_);
        printf("[QUANT_I32] computeR: r_pre_q=%d, r_pre_fp=%.6f, r_q=%d, r_fp=%.6f\n", r_pre_i32,
               r_pre_fp, r, r_fp);
    }
#endif

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0 && debug_idx < 3) {
        float r_pre_fp = (float)(r_pre_i32 - rescale_params.zp_r_pre_) /
                         (float)(1 << rescale_params.test.exp2_inv_r_pre_);
        float r_quant_fp = (float)(r - rescale_params.zp_r_out_) /
                           (float)(1 << rescale_params.test.exp2_inv_r_out_);
        float r_theory = sigmoid_fp(r_pre_fp);
        float error = r_quant_fp - r_theory;
        printf("[R] idx=%d r_pre_q=%d r_pre_fp=%.4f | r_q=%d r_fp=%.4f | theory=%.4f | err=%.6f\n",
               debug_idx, r_pre_i32, r_pre_fp, r, r_quant_fp, r_theory, error);
    }
#endif

    return r;
}

// g = tanh(Wx + r * (Rh + br) + bx) - 候选门
__device__ __forceinline__ int32_t computeG(const int channel_idx, const int32_t Wx_val,
                                            const int32_t Rh_val, const int32_t bx_val,
                                            const int32_t br_val, const int32_t r,
                                            const QuantGRUReScale &rescale_params,
                                            int32_t &Rh_add_br_g, const int debug_idx = -1) {
    Rh_add_br_g = rshift_round(Rh_val - rescale_params.zp_Rh_, rescale_params.n_Rh_div_Rh_add_br_) +
                  rshift_round(br_val, rescale_params.n_br_div_Rh_add_br_[channel_idx]) +
                  rescale_params.zp_Rh_add_br_;
    // 添加 clamp 到配置的位宽范围，防止混合精度时中间结果溢出
    Rh_add_br_g = clamp_by_bitwidth(Rh_add_br_g, rescale_params.bitwidth_config_.Rh_add_br_);

    const int64_t r_diff = static_cast<int64_t>(r) - rescale_params.zp_r_out_;
    const int64_t Rh_add_br_diff = static_cast<int64_t>(Rh_add_br_g) - rescale_params.zp_Rh_add_br_;
    const int64_t rRh_mul_i64 = r_diff * Rh_add_br_diff;

    int32_t rRh =
        static_cast<int32_t>(rshift_round(rRh_mul_i64, rescale_params.n_r_mul_Rh_add_br_div_rRh_)) +
        rescale_params.zp_rRh_;
    // 添加 clamp 到配置的位宽范围
    rRh = clamp_by_bitwidth(rRh, rescale_params.bitwidth_config_.rRh_);

    const int32_t Wx_shifted =
        rshift_round(Wx_val - rescale_params.zp_Wx_, rescale_params.n_Wx_div_g_pre_);
    const int32_t rRh_shifted =
        rshift_round(rRh - rescale_params.zp_rRh_, rescale_params.n_rRh_div_g_pre_);
    const int32_t bx_shifted =
        rshift_round(bx_val, rescale_params.exp2_inv_bx_div_g_pre_[channel_idx]);

    const int32_t g_pre_i32 = Wx_shifted + rRh_shifted + bx_shifted + rescale_params.zp_g_pre_;

    // 使用参数中的 LUT（避免全局 LUT 覆盖问题）
    const auto &bw_cfg = rescale_params.bitwidth_config_;
    const int32_t g = piecewise_linear(g_pre_i32, rescale_params.tanh_g_lut_,
                                        bw_cfg.g_pre_, bw_cfg.g_out_);

#ifdef DEBUG_QUANT
    if (debug_idx == 0) {
        float Rh_add_br_fp = (float)(Rh_add_br_g - rescale_params.zp_Rh_add_br_) /
                             (float)(1 << rescale_params.test.exp2_inv_Rh_add_br_);
        float rRh_fp =
            (float)(rRh - rescale_params.zp_rRh_) / (float)(1 << rescale_params.test.exp2_inv_rRh_);
        float g_pre_fp = (float)(g_pre_i32 - rescale_params.zp_g_pre_) /
                         (float)(1 << rescale_params.test.exp2_inv_g_pre_);
        float g_fp = (float)(g - rescale_params.zp_g_out_) /
                     (float)(1 << rescale_params.test.exp2_inv_g_out_);
        printf("[QUANT_I32] computeG: Rh_add_br_fp=%.6f, rRh_fp=%.6f, g_pre_fp=%.6f, g_fp=%.6f\n",
               Rh_add_br_fp, rRh_fp, g_pre_fp, g_fp);
    }
#endif

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0 && debug_idx < 3) {
        float g_pre_fp = (float)(g_pre_i32 - rescale_params.zp_g_pre_) /
                         (float)(1 << rescale_params.test.exp2_inv_g_pre_);
        float g_quant_fp = (float)(g - rescale_params.zp_g_out_) /
                           (float)(1 << rescale_params.test.exp2_inv_g_out_);
        float g_theory = tanh_fp(g_pre_fp);
        float error = g_quant_fp - g_theory;
        printf("[G] idx=%d g_pre_q=%d g_pre_fp=%.4f | g_q=%d g_fp=%.4f | theory=%.4f | err=%.6f\n",
               debug_idx, g_pre_i32, g_pre_fp, g, g_quant_fp, g_theory, error);

        // 打印 tanh LUT 参数
        if (rescale_params.bitwidth_config_.g_out_.bits_ > 8) {
            printf(
                "[G LUT] idx=%d shift_x=%d zp_x=%d shift_y=%d zp_y=%d | exp2_inv_g_pre=%d "
                "exp2_inv_g_out=%d\n",
                debug_idx, d_tanh_lut_int16.shift_bits_x, d_tanh_lut_int16.zp_x,
                d_tanh_lut_int16.shift_bits_y, d_tanh_lut_int16.zp_y,
                rescale_params.test.exp2_inv_g_pre_, rescale_params.test.exp2_inv_g_out_);

            // 手动计算期望的 tanh 输出
            const int16_t g_pre_i16 = clamp_to_type<int16_t>(g_pre_i32);
            int seg_id =
                find_segment(static_cast<int32_t>(g_pre_i16), d_tanh_lut_int16.segments);
            printf("[G LUT] idx=%d seg_id=%d g_pre_i16=%d threshold[seg]=%d\n", debug_idx, seg_id,
                   g_pre_i16, d_tanh_lut_int16.segments[seg_id].threshold);
        }
    }
#endif

    return g;
}

// h = z * h_old + (1 - z) * g - 最终隐藏状态
__device__ __forceinline__ int32_t computeH(const int32_t z, const int32_t g, const int32_t h_old,
                                           const QuantGRUReScale &rescale_params,
                                           const int debug_idx = -1) {
    const int64_t z_diff = static_cast<int64_t>(z) - rescale_params.zp_z_out_;
    const int64_t h_diff = static_cast<int64_t>(h_old) - rescale_params.zp_h_;
    const int64_t old_contrib_mul_i64 = z_diff * h_diff;

    int32_t old_contrib =
        static_cast<int32_t>(
            rshift_round(old_contrib_mul_i64, rescale_params.n_z_mul_h_div_old_contrib_)) +
        rescale_params.zp_old_contrib_;
    // 添加 clamp 到配置的位宽范围，防止混合精度时中间结果溢出
    old_contrib = clamp_by_bitwidth(old_contrib, rescale_params.bitwidth_config_.old_contrib_);

    // 计算 (1-z) 在量化空间的差值表示
    // 【公式推导】
    //   设 one_minus_update = q(1-z) = one_in_z_scale_ - z + zp_z_out_
    //   其中 one_in_z_scale_ = round(1.0 / scale_z_out) + zp_z_out_ 是常数 1 在 z_out
    //   量化空间的表示 则 one_minus_diff = one_minus_update - zp_z_out_
    //                     = (one_in_z_scale_ - z + zp_z_out_) - zp_z_out_
    //                     = one_in_z_scale_ - z
    // 【优化】省去中间变量 one_minus_update，直接计算 one_minus_diff
    const int64_t one_minus_diff = static_cast<int64_t>(rescale_params.one_in_z_scale_) - z;
    const int64_t g_diff = static_cast<int64_t>(g) - rescale_params.zp_g_out_;
    const int64_t new_contrib_mul_i64 = one_minus_diff * g_diff;

    int32_t new_contrib =
        static_cast<int32_t>(
            rshift_round(new_contrib_mul_i64, rescale_params.n_z_out_mul_g_div_new_contrib_)) +
        rescale_params.zp_new_contrib_;
    // 添加 clamp 到配置的位宽范围，防止混合精度时中间结果溢出
    new_contrib = clamp_by_bitwidth(new_contrib, rescale_params.bitwidth_config_.new_contrib_);

    const int32_t h_i32 = rshift_round(old_contrib - rescale_params.zp_old_contrib_,
                                       rescale_params.n_old_contrib_div_h_) +
                          rshift_round(new_contrib - rescale_params.zp_new_contrib_,
                                       rescale_params.n_new_contrib_div_h_) +
                          rescale_params.zp_h_;

    // 根据 h 的位宽配置进行 clamp
    const int32_t h = clamp_by_bitwidth(h_i32, rescale_params.bitwidth_config_.h_);

#ifdef DEBUG_QUANT
    if (debug_idx == 0) {
        float z_fp = (float)(z - rescale_params.zp_z_out_) /
                     (float)(1 << rescale_params.test.exp2_inv_z_out_);
        float g_fp = (float)(g - rescale_params.zp_g_out_) /
                     (float)(1 << rescale_params.test.exp2_inv_g_out_);
        float h_old_fp =
            (float)(h_old - rescale_params.zp_h_) / (float)(1 << rescale_params.test.exp2_inv_h_);
        float h_fp =
            (float)(h - rescale_params.zp_h_) / (float)(1 << rescale_params.test.exp2_inv_h_);
        printf("[QUANT_I32] computeH: z_fp=%.6f, g_fp=%.6f, h_old_fp=%.6f, h_new_fp=%.6f\n", z_fp,
               g_fp, h_old_fp, h_fp);
    }
#endif

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0 && debug_idx < 3) {
        // 反量化各变量
        float z_fp = (float)(z - rescale_params.zp_z_out_) /
                     (float)(1 << rescale_params.test.exp2_inv_z_out_);
        float g_fp = (float)(g - rescale_params.zp_g_out_) /
                     (float)(1 << rescale_params.test.exp2_inv_g_out_);
        float h_old_fp =
            (float)(h_old - rescale_params.zp_h_) / (float)(1 << rescale_params.test.exp2_inv_h_);
        float h_quant_fp =
            (float)(h - rescale_params.zp_h_) / (float)(1 << rescale_params.test.exp2_inv_h_);

        // 理论计算: h_new = z * h_old + (1-z) * g
        float h_theory = z_fp * h_old_fp + (1.0f - z_fp) * g_fp;
        float error = h_quant_fp - h_theory;

        // 中间结果分析
        float old_contrib_fp = (float)(old_contrib - rescale_params.zp_old_contrib_) /
                               (float)(1 << rescale_params.test.exp2_inv_old_contrib_);
        float new_contrib_fp = (float)(new_contrib - rescale_params.zp_new_contrib_) /
                               (float)(1 << rescale_params.test.exp2_inv_new_contrib_);
        float old_contrib_theory = z_fp * h_old_fp;
        float new_contrib_theory = (1.0f - z_fp) * g_fp;

        printf(
            "[H] idx=%d z=%.4f g=%.4f h_old=%.4f | old_contrib: q=%.4f th=%.4f | new_contrib: "
            "q=%.4f th=%.4f | h: q=%.4f th=%.4f err=%.6f\n",
            debug_idx, z_fp, g_fp, h_old_fp, old_contrib_fp, old_contrib_theory, new_contrib_fp,
            new_contrib_theory, h_quant_fp, h_theory, error);
    }
#endif

    return h;
}

// ============================================================================
// 4. Pointwise Kernel - GRU 逐点运算
// ============================================================================
// 每个线程处理一个 (batch, hidden) 位置
// 所有量化值使用 int32_t 存储

template <bool Training, bool ApplyZoneout>
__global__ void PointwiseOperationsQuant(
    const int batch_dim, const int hidden_dim, const int32_t *Wx, const int32_t *Rh,
    const int32_t *bx, const int32_t *br, const int32_t *h, int32_t *h_out, int32_t *v,
    const float zoneout_prob, const int32_t *zoneout_mask, const QuantGRUReScale rescale_params) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim) return;

    const int weight_idx = col * (hidden_dim * 3) + row;
    const int output_idx = col * hidden_dim + row;
    const int z_idx = weight_idx + 0 * hidden_dim;
    const int r_idx = weight_idx + 1 * hidden_dim;
    const int g_idx = weight_idx + 2 * hidden_dim;
    const int b_z_idx = row + 0 * hidden_dim;
    const int b_r_idx = row + 1 * hidden_dim;
    const int b_g_idx = row + 2 * hidden_dim;

#ifdef DEBUG_QUANT_DETAIL
    // ============ 调试：同时进行浮点计算用于对比 ============
    const int debug_idx = (col == 0 && row < 3) ? row : -1;

    if (debug_idx >= 0) {
        // 反量化 Wx, Rh (GEMM 结果是 int32, 需要用 Wx 和 Rh 的 scale)
        const float scale_Wx = 1.0f / (float)(1 << rescale_params.test.exp2_inv_Wx_);
        const float scale_Rh = 1.0f / (float)(1 << rescale_params.test.exp2_inv_Rh_);
        const float scale_h = 1.0f / (float)(1 << rescale_params.test.exp2_inv_h_);

        // 反量化 GEMM 结果
        float Wx_z_fp = (float)(Wx[z_idx] - rescale_params.zp_Wx_) * scale_Wx;
        float Wx_r_fp = (float)(Wx[r_idx] - rescale_params.zp_Wx_) * scale_Wx;
        float Wx_g_fp = (float)(Wx[g_idx] - rescale_params.zp_Wx_) * scale_Wx;

        float Rh_z_fp = (float)(Rh[z_idx] - rescale_params.zp_Rh_) * scale_Rh;
        float Rh_r_fp = (float)(Rh[r_idx] - rescale_params.zp_Rh_) * scale_Rh;
        float Rh_g_fp = (float)(Rh[g_idx] - rescale_params.zp_Rh_) * scale_Rh;

        // 反量化 bias (bias 是 int32, 使用各自 channel 的 scale，从 device vector 获取)
        float bx_z_fp = (float)bx[b_z_idx] / (float)(1 << rescale_params.exp2_inv_bx_dev_[b_z_idx]);
        float bx_r_fp = (float)bx[b_r_idx] / (float)(1 << rescale_params.exp2_inv_bx_dev_[b_r_idx]);
        float bx_g_fp = (float)bx[b_g_idx] / (float)(1 << rescale_params.exp2_inv_bx_dev_[b_g_idx]);

        float br_z_fp = (float)br[b_z_idx] / (float)(1 << rescale_params.exp2_inv_br_dev_[b_z_idx]);
        float br_r_fp = (float)br[b_r_idx] / (float)(1 << rescale_params.exp2_inv_br_dev_[b_r_idx]);
        float br_g_fp = (float)br[b_g_idx] / (float)(1 << rescale_params.exp2_inv_br_dev_[b_g_idx]);

        // 反量化 h_old
        float h_old_fp = (float)(h[output_idx] - rescale_params.zp_h_) * scale_h;

        // ========== 浮点 GRU 计算 ==========
        float z_pre_fp = Wx_z_fp + Rh_z_fp + bx_z_fp + br_z_fp;
        float z_fp = sigmoid_fp(z_pre_fp);

        float r_pre_fp = Wx_r_fp + Rh_r_fp + bx_r_fp + br_r_fp;
        float r_fp = sigmoid_fp(r_pre_fp);

        float Rh_add_br_g_fp = Rh_g_fp + br_g_fp;
        float g_pre_fp = Wx_g_fp + r_fp * Rh_add_br_g_fp + bx_g_fp;
        float g_fp = tanh_fp(g_pre_fp);

        float h_new_fp = z_fp * h_old_fp + (1.0f - z_fp) * g_fp;

        printf("\n===== [DEBUG idx=%d batch=0] =====\n", debug_idx);
        printf("Wx: z_q=%d r_q=%d g_q=%d | z_fp=%.4f r_fp=%.4f g_fp=%.4f\n", Wx[z_idx], Wx[r_idx],
               Wx[g_idx], Wx_z_fp, Wx_r_fp, Wx_g_fp);
        printf("Rh: z_q=%d r_q=%d g_q=%d | z_fp=%.4f r_fp=%.4f g_fp=%.4f\n", Rh[z_idx], Rh[r_idx],
               Rh[g_idx], Rh_z_fp, Rh_r_fp, Rh_g_fp);
        printf("bx: z_q=%d r_q=%d g_q=%d | z_fp=%.4f r_fp=%.4f g_fp=%.4f\n", bx[b_z_idx],
               bx[b_r_idx], bx[b_g_idx], bx_z_fp, bx_r_fp, bx_g_fp);
        printf("br: z_q=%d r_q=%d g_q=%d | z_fp=%.4f r_fp=%.4f g_fp=%.4f\n", br[b_z_idx],
               br[b_r_idx], br[b_g_idx], br_z_fp, br_r_fp, br_g_fp);
        printf("h_old: q=%d fp=%.4f\n", h[output_idx], h_old_fp);
        printf("[FLOAT] z_pre=%.4f z=%.4f | r_pre=%.4f r=%.4f | g_pre=%.4f g=%.4f | h_new=%.4f\n",
               z_pre_fp, z_fp, r_pre_fp, r_fp, g_pre_fp, g_fp, h_new_fp);
    }
#else
    const int debug_idx = -1;
#endif

    // GRU 门计算
    const int32_t z = computeZ(b_z_idx, Wx[z_idx], Rh[z_idx], bx[b_z_idx], br[b_z_idx],
                               rescale_params, debug_idx);

    const int32_t r = computeR(b_r_idx, Wx[r_idx], Rh[r_idx], bx[b_r_idx], br[b_r_idx],
                               rescale_params, debug_idx);

    int32_t Rh_add_br_g;
    const int32_t g = computeG(b_g_idx, Wx[g_idx], Rh[g_idx], bx[b_g_idx], br[b_g_idx], r,
                               rescale_params, Rh_add_br_g, debug_idx);

    // Training: 保存中间值
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh_add_br_g;
    }

    // 计算新的隐藏状态
    auto cur_h = computeH(z, g, h[output_idx], rescale_params, debug_idx);

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0) {
        // 反量化最终结果与浮点对比
        const float scale_z = 1.0f / (float)(1 << rescale_params.test.exp2_inv_z_out_);
        const float scale_g = 1.0f / (float)(1 << rescale_params.test.exp2_inv_g_out_);
        const float scale_h = 1.0f / (float)(1 << rescale_params.test.exp2_inv_h_);

        float z_quant_fp = (float)(z - rescale_params.zp_z_out_) * scale_z;
        float g_quant_fp = (float)(g - rescale_params.zp_g_out_) * scale_g;
        float h_quant_fp = (float)(cur_h - rescale_params.zp_h_) * scale_h;

        printf("[QUANT] z_q=%d z_fp=%.4f | g_q=%d g_fp=%.4f | h_q=%d h_fp=%.4f\n", z, z_quant_fp, g,
               g_quant_fp, cur_h, h_quant_fp);
        printf("=====================================\n");
    }
#endif

    h_out[output_idx] = cur_h;
}

// ============================================================================
// 辅助 Kernel: int32 → int8/int16 转换（用于 cuBLAS INT8 GEMM 优化）
// ============================================================================

// int32 → int8 转换 kernel（值已经在 [-128, 127] 范围内）
__global__ void convertI32ToI8(const int32_t *__restrict__ src, int8_t *__restrict__ dst,
                               size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] = static_cast<int8_t>(src[idx]);
}

// int32 → int16 转换 kernel（值已经在 [-32768, 32767] 范围内）
__global__ void convertI32ToI16(const int32_t *__restrict__ src, int16_t *__restrict__ dst,
                                size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    dst[idx] = static_cast<int16_t>(src[idx]);
}

}  // namespace kernel

// ============================================================================
// 5. ForwardPassQuant - 前向传播封装类
// ============================================================================

namespace gru {

struct ForwardPassQuant::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[2];
    cudaEvent_t event;
    cudaStream_t sync_stream;
};

ForwardPassQuant::ForwardPassQuant(const bool training, const int batch_size,
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

ForwardPassQuant::~ForwardPassQuant() {
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

// cuBLAS INT8 GEMM N 维度对齐常量
constexpr int CUBLAS_INT8_N_ALIGNMENT = 32;
constexpr int CUBLAS_INT8_N_THRESHOLD = 16;

// 计算填充后的 N 值
inline int computePaddedN(int N) {
    if (N > CUBLAS_INT8_N_THRESHOLD && N % CUBLAS_INT8_N_ALIGNMENT != 0) {
        return ((N + CUBLAS_INT8_N_ALIGNMENT - 1) / CUBLAS_INT8_N_ALIGNMENT) *
               CUBLAS_INT8_N_ALIGNMENT;
    }
    return N;
}

void ForwardPassQuant::EnsureBuffersAllocated(int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;

    // 如果已分配且足够大，直接返回
    if (steps <= max_steps_) {
        return;
    }

    // GEMM 结果缓冲区（int32）
    tmp_Wx_.resize(hidden3 * steps * batch_size);
    tmp_Rh_.resize(hidden3 * batch_size);

    // 权重和常量
    if (W_sum_mul_x_zp_.size() == 0) {
        W_sum_mul_x_zp_.resize(hidden3);
        R_sum_mul_h_zp_.resize(hidden3);
    }

    // INT8 GEMM 优化缓冲区（当位宽 <= 8 时使用）
    const auto &bw_cfg = rescale_param_.bitwidth_config_;
    if (bw_cfg.W_.fitsInt8() && bw_cfg.x_.fitsInt8()) {
        // 权重 int8 缓存（只分配一次）
        if (tmp_W_i8_.size() == 0) {
            tmp_W_i8_.resize(hidden3 * input_size);
            tmp_R_i8_.resize(hidden3 * hidden_size);
        }
        
        // 输入 int8 缓存
        const int N_Wx = steps * batch_size;
        const int N_Wx_padded = computePaddedN(N_Wx);
        tmp_x_i8_.resize(input_size * N_Wx_padded);
        if (N_Wx_padded != N_Wx) {
            tmp_x_i8_.zero();  // 初始化填充部分为零
        }
        
        // ComputeRh: N = batch_size（固定）
        if (N_padded_Rh_ == 0) {
            N_padded_Rh_ = computePaddedN(batch_size);
            tmp_h_i8_.resize(hidden_size * batch_size);
            if (N_padded_Rh_ != batch_size) {
                h_padded_i8_.resize(hidden_size * N_padded_Rh_);
                h_padded_i8_.zero();
            }
        }
    }

    max_steps_ = steps;
    weight_sums_computed_ = false;  // 需要重新计算
}

void ForwardPassQuant::PrecomputeWeightSums(const int32_t *W, const int32_t *R) {
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
                          rescale_param_.n_W_mul_x_div_Wx_.data(), hidden_size * 3, input_size,
                          stream);

    // 计算 R_sum_mul_h_zp
    computeWeightSumMulzp(R, R_sum_mul_h_zp_.data(), rescale_param_.zp_h_,
                          rescale_param_.n_R_mul_h_div_Rh_.data(), hidden_size * 3, hidden_size,
                          stream);

    cudaStreamSynchronize(stream);
    weight_sums_computed_ = true;
}

void ForwardPassQuant::ComputeWx(const int32_t *W, const int32_t *x, int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream = data_->stream[1];
    const int total_size = hidden_size * 3 * steps * batch_size;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    const int M = hidden_size * 3;
    const int N = steps * batch_size;
    const int K = input_size;

    const auto &bw_cfg = rescale_param_.bitwidth_config_;

    // 检查是否可以使用 cuBLAS INT8 GEMM 优化
    if (bw_cfg.W_.fitsInt8() && bw_cfg.x_.fitsInt8()) {
        // INT8 GEMM 优化路径：int32 → int8 转换后调用 cuBLAS
        static const int32_t alpha32 = 1;
        static const int32_t beta32 = 0;

        const int N_padded = computePaddedN(N);
        const size_t W_size = static_cast<size_t>(M) * K;
        const size_t x_size = static_cast<size_t>(K) * N;

        // 转换权重 int32 → int8（只在首次或权重变化时）
        kernel::convertI32ToI8<<<(W_size + 255) / 256, 256, 0, stream>>>(
            W, tmp_W_i8_.data(), W_size);

        // 转换输入 int32 → int8
        kernel::convertI32ToI8<<<(x_size + 255) / 256, 256, 0, stream>>>(
            x, tmp_x_i8_.data(), x_size);

        // 调用 cuBLAS INT8 GEMM
        if (N_padded != N) {
            // 需要填充：将输入复制到填充缓冲区
            // tmp_x_i8_ 已经足够大（在 EnsureBuffersAllocated 中分配）
            blas<int8_t>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N_padded, K, &alpha32,
                               tmp_W_i8_.data(), M, tmp_x_i8_.data(), K, &beta32,
                               tmp_Wx_.data(), M);
        } else {
            blas<int8_t>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha32,
                               tmp_W_i8_.data(), M, tmp_x_i8_.data(), K, &beta32,
                               tmp_Wx_.data(), M);
        }

        // Rescale: 只处理实际的 N 列
        kernel::rescaleGemmI32<<<blocks, threads, 0, stream>>>(
            tmp_Wx_.data(), W_sum_mul_x_zp_.data(), rescale_param_.n_W_mul_x_div_Wx_.data(),
            rescale_param_.zp_Wx_, hidden_size * 3, total_size,
            rescale_param_.bitwidth_config_.Wx_);
    } else {
        // 非 INT8 情况：使用融合 GEMM（int32_t 输入输出）
        dim3 blockDim(kernel::TILE_SIZE, kernel::TILE_SIZE);
        dim3 gridDim((N + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE,
                     (M + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE);

        kernel::quantizedGemmFused<<<gridDim, blockDim, 0, stream>>>(
            W, x, tmp_Wx_.data(), M, N, K, rescale_param_.zp_x_,
            rescale_param_.n_W_mul_x_div_Wx_.data(), rescale_param_.zp_Wx_,
            rescale_param_.bitwidth_config_.Wx_);
    }
}

void ForwardPassQuant::ComputeRh(const int32_t *R, const int32_t *h) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream = data_->stream[0];
    const int total_size = hidden_size * 3 * batch_size;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    const int M = hidden_size * 3;
    const int N = batch_size;
    const int K = hidden_size;

    const auto &bw_cfg = rescale_param_.bitwidth_config_;

    // 检查是否可以使用 cuBLAS INT8 GEMM 优化
    if (bw_cfg.R_.fitsInt8() && bw_cfg.h_.fitsInt8()) {
        // INT8 GEMM 优化路径
        static const int32_t alpha32 = 1;
        static const int32_t beta32 = 0;

        const size_t R_size = static_cast<size_t>(M) * K;
        const size_t h_size = static_cast<size_t>(K) * N;

        // 转换递归权重 int32 → int8（只在首次或权重变化时）
        kernel::convertI32ToI8<<<(R_size + 255) / 256, 256, 0, stream>>>(
            R, tmp_R_i8_.data(), R_size);

        // 转换隐藏状态 int32 → int8
        kernel::convertI32ToI8<<<(h_size + 255) / 256, 256, 0, stream>>>(
            h, tmp_h_i8_.data(), h_size);

        // 调用 cuBLAS INT8 GEMM
        if (N_padded_Rh_ != N) {
            // 需要填充
            d2d(h_padded_i8_.data(), tmp_h_i8_.data(), h_size);
            blas<int8_t>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N_padded_Rh_, K, &alpha32,
                               tmp_R_i8_.data(), M, h_padded_i8_.data(), K, &beta32,
                               tmp_Rh_.data(), M);
        } else {
            blas<int8_t>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha32,
                               tmp_R_i8_.data(), M, tmp_h_i8_.data(), K, &beta32,
                               tmp_Rh_.data(), M);
        }

        // Rescale
        kernel::rescaleGemmI32<<<blocks, threads, 0, stream>>>(
            tmp_Rh_.data(), R_sum_mul_h_zp_.data(), rescale_param_.n_R_mul_h_div_Rh_.data(),
            rescale_param_.zp_Rh_, hidden_size * 3, total_size,
            rescale_param_.bitwidth_config_.Rh_);
    } else {
        // 非 INT8 情况：使用融合 GEMM
        dim3 blockDim(kernel::TILE_SIZE, kernel::TILE_SIZE);
        dim3 gridDim((N + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE,
                     (M + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE);

        kernel::quantizedGemmFused<<<gridDim, blockDim, 0, stream>>>(
            R, h, tmp_Rh_.data(), M, N, K, rescale_param_.zp_h_,
            rescale_param_.n_R_mul_h_div_Rh_.data(), rescale_param_.zp_Rh_,
            rescale_param_.bitwidth_config_.Rh_);
    }
}

void ForwardPassQuant::IterateInternal(
    const int32_t *R,         // [H,H*3]
    const int32_t *bx,        // [H*3]
    const int32_t *br,        // [H*3]
    const int32_t *h,         // [N,H]
    int32_t *h_out,           // [N,H]
    int32_t *v,               // [N,H*4]
    const int32_t *cur_Wx_,   // [N,H*3] 当前时间步的 W @ x 结果
    const float zoneout_prob,
    const int32_t *zoneout_mask  // Zoneout mask [N,H]
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

    // 启动量化 GRU kernel（使用统一 int32_t 存储）
    if (training) {
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperationsQuant<true, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_,
                                                    tmp_Rh_.data(), bx, br, h, h_out, v,
                                                    zoneout_prob, zoneout_mask, rescale_param_);
        } else {
            kernel::PointwiseOperationsQuant<true, false>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_,
                                                    tmp_Rh_.data(), bx, br, h, h_out, v, 0.0f,
                                                    nullptr, rescale_param_);
        }
    } else {
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperationsQuant<false, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_,
                                                    tmp_Rh_.data(), bx, br, h, h_out, nullptr,
                                                    zoneout_prob, zoneout_mask, rescale_param_);
        } else {
            kernel::PointwiseOperationsQuant<false, false>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_,
                                                    tmp_Rh_.data(), bx, br, h, h_out, nullptr, 0.0f,
                                                    nullptr, rescale_param_);
        }
    }
}

void ForwardPassQuant::setRescaleParam(const GRUQuantitativeParameters &parms) {
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

    // 保存位宽配置（用于运行时选择正确的 kernel 实例）
    rescale_param_.bitwidth_config_ = parms.bitwidth_config_;

    // 复制 LUT 表（从 GRUQuantitativeParameters 复制到 QuantGRUReScale）
    // 这样每层 GRU 使用自己的 LUT，避免全局 __constant__ LUT 覆盖问题
    rescale_param_.sigmoid_z_lut_ = parms.sigmoid_z_lut_;
    rescale_param_.sigmoid_r_lut_ = parms.sigmoid_r_lut_;
    rescale_param_.tanh_g_lut_ = parms.tanh_g_lut_;

#ifdef DEBUG
    // 调试用：保存完整的量化参数
    rescale_param_.test = parms;
    // 将 bias 的 scale 拷贝到 device 可访问的 vector
    rescale_param_.exp2_inv_bx_dev_ = dev::vector<int8_t>(parms.exp2_inv_bx_);
    rescale_param_.exp2_inv_br_dev_ = dev::vector<int8_t>(parms.exp2_inv_br_);
#endif
}

void ForwardPassQuant::Run(
    const int steps,              // 时间步数, 序列长度T
    const int32_t *W,             // [C,H*3], 输入到隐藏状态的权重矩阵（int32_t 存储）
    const int32_t *R,             // [H,H*3], 隐状态到隐藏状态的权重矩阵（int32_t 存储）
    const int32_t *bx,            // [H*3], 输入偏置
    const int32_t *br,            // [H*3], 隐状态偏置
    const int32_t *x,             // [N*T,C], 输入序列（int32_t 存储）
    int32_t *h,                   // [(T+1)*N,H], 输出隐藏状态（int32_t 存储）
    int32_t *v,                   // [T*N,H*4], 中间激活值（训练模式需要）
    const float zoneout_prob,     // Zoneout 概率
    const int32_t *zoneout_mask   // Zoneout mask [T*N,H]（int32_t 存储）
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
                        h + i * NH,                // 输入 h
                        h + (i + 1) * NH,          // 输出 h
                        v + i * NH * 4,            // 中间激活
                        tmp_Wx_.data() + i * NH3,  // 当前时间步的 Wx
                        zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    cublasSetStream(data_->blas_handle, save_stream);
}

}  // namespace gru

