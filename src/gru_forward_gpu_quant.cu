// ============================================================================
// gru_forward_gpu_quant.cu - 量化 GRU 前向传播 CUDA 实现
// ============================================================================
//
// 文件结构:
//   1. GEMM Kernels        - 量化矩阵乘法 (INT8/INT16)
//   2. Rescale Kernels     - GEMM 结果缩放
//   3. GRU Gate Functions  - 门计算函数 (computeZ/R/G/H_i32)
//   4. Pointwise Kernel    - GRU 逐点运算主 kernel
//   5. ForwardPassQuant    - 前向传播封装类
//
// 量化方案:
//   - 所有中间值使用 int32_t 统一存储
//   - 通过 bitwidth_config_ 枚举动态选择 8/16 位 LUT
//   - 无模板类型转换开销
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
#include "devVector.h"
#include "gru_quant.h"
#include "quantize_ops.cuh"
#include "quantize_ops_helper.hpp"

namespace kernel {

// 调试开关
// #define DEBUG_QUANT           // 启用量化调试输出

// ============================================================================
// 1. GEMM Kernels - 量化矩阵乘法
// ============================================================================

// INT16 融合 GEMM: C = rshift(A * (B - zp_B), shift) + zp_out
constexpr int TILE_SIZE = 16;

template <typename AT, typename BT>
__global__ void quantizedGemmInt16Fused(
    const AT *__restrict__ A,  // [M, K] 权重（W 或 R），行主序
    const BT *__restrict__ B,  // [K, N] 输入（x 或 h），列主序（cuBLAS 风格）
    int32_t *__restrict__ C,   // [M, N] 输出，列主序
    int M, int N, int K,
    int32_t zp_B,                              // 输入的 zero-point
    const int8_t *__restrict__ shift_per_row,  // [M] per-row shift
    int32_t zp_out                             // 输出的 zero-point
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

// INT8: 将 int32 GEMM 结果原地 rescale
__global__ void rescaleGemmI32(
    int32_t *__restrict__ data,                // [hidden*3, batch*steps] GEMM 输出（原地修改）
    const int64_t *__restrict__ compensation,  // [hidden*3] W_sum_mul_x_zp
    const int8_t *__restrict__ shift,          // [hidden*3] per-channel shift
    int32_t zp,                                // zero point
    int hidden3,                               // hidden_size * 3
    int total_size                             // hidden*3 * batch*steps
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

// ============================================================================
// 3. GRU Gate Functions - 门计算函数
// ============================================================================
// 所有中间值使用 int32_t 存储，通过 bitwidth_config_ 枚举选择 8/16 位 LUT

// z = sigmoid(Wx + Rh + bx + br) - 更新门
__device__ __forceinline__ int32_t computeZ(const int channel_idx, const int32_t Wx_val,
                                            const int32_t Rh_val, const int32_t bx_val,
                                            const int32_t br_val,
                                            const QuantGRUReScale &rescale_params,
                                            const int debug_idx = -1) {
    const int32_t Wx = Wx_val;
    const int32_t Rh = Rh_val;

    const int32_t Wx_shifted =
        rshift_round(Wx - rescale_params.zp_Wx_, rescale_params.exp2_inv_Wx_div_z_pre_);
    const int32_t Rh_shifted =
        rshift_round(Rh - rescale_params.zp_Rh_, rescale_params.exp2_inv_Rh_div_z_pre_);
    const int32_t bx_shifted = rshift_round(bx_val, rescale_params.n_bx_div_z_[channel_idx]);
    const int32_t br_shifted = rshift_round(br_val, rescale_params.n_br_div_z_[channel_idx]);

    const int32_t z_pre_i32 =
        Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale_params.zp_z_pre_;

    int32_t z;
    if (rescale_params.bitwidth_config_.z_out_ == QuantBitWidth::UINT16) {
        const int16_t z_pre_i16 = dev::clamp<int16_t>(z_pre_i32);
        z = dev::sigmoid_piecewise_linear_int16(z_pre_i16, d_sigmoid_z_lut_int16);
    } else {
        const int8_t z_pre_i8 = dev::clamp<int8_t>(z_pre_i32);
        z = dev::sigmoid_piecewise_linear_int8(z_pre_i8, d_sigmoid_z_lut_int8);
    }

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

    return z;
}

// r = sigmoid(Wx + Rh + bx + br) - 重置门
__device__ __forceinline__ int32_t computeR(const int channel_idx, const int32_t Wx_val,
                                            const int32_t Rh_val, const int32_t bx_val,
                                            const int32_t br_val,
                                            const QuantGRUReScale &rescale_params,
                                            const int debug_idx = -1) {
    const int32_t Wx = Wx_val;
    const int32_t Rh = Rh_val;

    const int32_t Wx_shifted =
        rshift_round(Wx - rescale_params.zp_Wx_, rescale_params.exp2_inv_Wx_div_r_pre_);
    const int32_t Rh_shifted =
        rshift_round(Rh - rescale_params.zp_Rh_, rescale_params.exp2_inv_Rh_div_r_pre_);
    const int32_t bx_shifted = rshift_round(bx_val, rescale_params.n_bx_div_r_[channel_idx]);
    const int32_t br_shifted = rshift_round(br_val, rescale_params.n_br_div_r_[channel_idx]);

    const int32_t r_pre_i32 =
        Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale_params.zp_r_pre_;

    int32_t r;
    if (rescale_params.bitwidth_config_.r_out_ == QuantBitWidth::UINT16) {
        const int16_t r_pre_i16 = dev::clamp<int16_t>(r_pre_i32);
        r = dev::sigmoid_piecewise_linear_int16(r_pre_i16, d_sigmoid_r_lut_int16);
    } else {
        const int8_t r_pre_i8 = dev::clamp<int8_t>(r_pre_i32);
        r = dev::sigmoid_piecewise_linear_int8(r_pre_i8, d_sigmoid_r_lut_int8);
    }

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

    return r;
}

// g = tanh(Wx + r * (Rh + br) + bx) - 候选门
__device__ __forceinline__ int32_t computeG(const int channel_idx, const int32_t Wx_val,
                                            const int32_t Rh_val, const int32_t bx_val,
                                            const int32_t br_val, const int32_t r,
                                            const QuantGRUReScale &rescale_params,
                                            int32_t &Rh_add_br_g, const int debug_idx = -1) {
    const int32_t Wx = Wx_val;
    const int32_t Rh = Rh_val;

    Rh_add_br_g = rshift_round(Rh - rescale_params.zp_Rh_, rescale_params.n_Rh_div_Rh_add_br_) +
                  rshift_round(br_val, rescale_params.n_br_div_Rh_add_br_[channel_idx]) +
                  rescale_params.zp_Rh_add_br_;

    const int64_t r_diff = static_cast<int64_t>(r) - rescale_params.zp_r_out_;
    const int64_t Rh_add_br_diff = static_cast<int64_t>(Rh_add_br_g) - rescale_params.zp_Rh_add_br_;
    const int64_t rRh_mul_i64 = r_diff * Rh_add_br_diff;

    const int32_t rRh =
        static_cast<int32_t>(rshift_round(rRh_mul_i64, rescale_params.n_r_mul_Rh_add_br_div_rRh_)) +
        rescale_params.zp_rRh_;

    const int32_t Wx_shifted =
        rshift_round(Wx - rescale_params.zp_Wx_, rescale_params.n_Wx_div_g_pre_);
    const int32_t rRh_shifted =
        rshift_round(rRh - rescale_params.zp_rRh_, rescale_params.n_rRh_div_g_pre_);
    const int32_t bx_shifted =
        rshift_round(bx_val, rescale_params.exp2_inv_bx_div_g_pre_[channel_idx]);

    const int32_t g_pre_i32 = Wx_shifted + rRh_shifted + bx_shifted + rescale_params.zp_g_pre_;

    int32_t g;
    if (rescale_params.bitwidth_config_.g_out_ == QuantBitWidth::INT16) {
        const int16_t g_pre_i16 = dev::clamp<int16_t>(g_pre_i32);
        g = dev::tanh_piecewise_linear_int16(g_pre_i16, d_tanh_lut_int16);
    } else {
        const int8_t g_pre_i8 = dev::clamp<int8_t>(g_pre_i32);
        g = dev::tanh_piecewise_linear_int8(g_pre_i8, d_tanh_lut_int8);
    }

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

    return g;
}

// h = z * h_old + (1 - z) * g - 最终隐藏状态
template <typename QuantT>
__device__ __forceinline__ QuantT computeH(const int32_t z, const int32_t g, const QuantT h_old,
                                           const QuantGRUReScale &rescale_params,
                                           const int debug_idx = -1) {
    const int64_t z_diff = static_cast<int64_t>(z) - rescale_params.zp_z_out_;
    const int64_t h_diff = static_cast<int64_t>(h_old) - rescale_params.zp_h_;
    const int64_t old_contrib_mul_i64 = z_diff * h_diff;

    const int32_t old_contrib =
        static_cast<int32_t>(
            rshift_round(old_contrib_mul_i64, rescale_params.n_z_mul_h_div_old_contrib_)) +
        rescale_params.zp_old_contrib_;

    // 1-z 在 z_out 的量化空间计算
    const int32_t one_minus_update = rescale_params.one_in_z_scale_ - z + rescale_params.zp_z_out_;

    const int64_t one_minus_diff =
        static_cast<int64_t>(one_minus_update) - rescale_params.zp_z_out_;
    const int64_t g_diff = static_cast<int64_t>(g) - rescale_params.zp_g_out_;
    const int64_t new_contrib_mul_i64 = one_minus_diff * g_diff;

    const int32_t new_contrib =
        static_cast<int32_t>(
            rshift_round(new_contrib_mul_i64, rescale_params.n_z_out_mul_g_div_new_contrib_)) +
        rescale_params.zp_new_contrib_;

    const int32_t h_i32 = rshift_round(old_contrib - rescale_params.zp_old_contrib_,
                                       rescale_params.n_old_contrib_div_h_) +
                          rshift_round(new_contrib - rescale_params.zp_new_contrib_,
                                       rescale_params.n_new_contrib_div_h_) +
                          rescale_params.zp_h_;

    const QuantT h = dev::clamp<QuantT>(h_i32);

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

    return h;
}

// ============================================================================
// 4. Pointwise Kernel - GRU 逐点运算
// ============================================================================
// 每个线程处理一个 (batch, hidden) 位置
// 模板参数: QuantT (隐藏状态类型), Training, ApplyZoneout

template <typename QuantT, bool Training, bool ApplyZoneout>
__global__ void PointwiseOperationsQuantDynamic(
    const int batch_dim, const int hidden_dim, const int32_t *Wx, const int32_t *Rh,
    const int32_t *bx, const int32_t *br, const QuantT *h, QuantT *h_out, int32_t *v,
    const QuantT zoneout_prob, const QuantT *zoneout_mask, const QuantGRUReScale rescale_params) {
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

    // GRU 门计算
    const int32_t z =
        computeZ(b_z_idx, Wx[z_idx], Rh[z_idx], bx[b_z_idx], br[b_z_idx], rescale_params);

    const int32_t r =
        computeR(b_r_idx, Wx[r_idx], Rh[r_idx], bx[b_r_idx], br[b_r_idx], rescale_params);

    int32_t Rh_add_br_g;
    const int32_t g = computeG(b_g_idx, Wx[g_idx], Rh[g_idx], bx[b_g_idx], br[b_g_idx], r,
                               rescale_params, Rh_add_br_g);

    // Training: 保存中间值
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh_add_br_g;
    }

    // 计算新的隐藏状态
    auto cur_h = computeH<QuantT>(z, g, h[output_idx], rescale_params);

    h_out[output_idx] = cur_h;
}

}  // namespace kernel

// ============================================================================
// 5. ForwardPassQuant - 前向传播封装类
// ============================================================================

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
                          rescale_param_.n_W_mul_x_div_Wx_.data(), hidden_size * 3, input_size,
                          stream);

    // 计算 R_sum_mul_h_zp
    computeWeightSumMulzp(R, R_sum_mul_h_zp_.data(), rescale_param_.zp_h_,
                          rescale_param_.n_R_mul_h_div_Rh_.data(), hidden_size * 3, hidden_size,
                          stream);

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
        blas<WT>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, steps * batch_size,
                       input_size, &alpha32, W, hidden_size * 3, x, input_size, &beta32,
                       tmp_Wx_.data(), hidden_size * 3);

        // Rescale: (Wx_i32 - W_sum_mul_x_zp) >> n + zp_Wx（原地操作）
        kernel::rescaleGemmI32<<<blocks, threads, 0, stream>>>(
            tmp_Wx_.data(), W_sum_mul_x_zp_.data(), rescale_param_.n_W_mul_x_div_Wx_.data(),
            rescale_param_.zp_Wx_, hidden_size * 3, total_size);
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
            W, x, tmp_Wx_.data(), M, N, K, rescale_param_.zp_x_,
            rescale_param_.n_W_mul_x_div_Wx_.data(), rescale_param_.zp_Wx_);
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
        blas<HT>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 3, batch_size,
                       hidden_size, &alpha32, R, hidden_size * 3, h, hidden_size, &beta32,
                       tmp_Rh_.data(), hidden_size * 3);

        // Rescale: (Rh_i32 - R_sum_mul_h_zp) >> n + zp_Rh（原地操作）
        kernel::rescaleGemmI32<<<blocks, threads, 0, stream>>>(
            tmp_Rh_.data(), R_sum_mul_h_zp_.data(), rescale_param_.n_R_mul_h_div_Rh_.data(),
            rescale_param_.zp_Rh_, hidden_size * 3, total_size);
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
            R, h, tmp_Rh_.data(), M, N, K, rescale_param_.zp_h_,
            rescale_param_.n_R_mul_h_div_Rh_.data(), rescale_param_.zp_Rh_);
    }
}

template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::IterateInternal(
    // C = input_size(输入维度), H = hidden_size(隐藏层维度),
    // T = time_steps(时间步), N = batch_size(批量大小)
    const RT *R,             // [H,H*3]
    const int32_t *bx,       // [H*3]
    const int32_t *br,       // [H*3]
    const HT *h,             // [N,H]
    HT *h_out,               // [N,H]
    int32_t *v,              // [N,H*4]
    const int32_t *cur_Wx_,  // [N,H*3] 当前时间步的 W @ x 结果
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

    // 启动量化 GRU kernel（使用统一 int32_t 存储，通过 bitwidth_config_ 动态选择 LUT）
    // 模板参数只需要 HT（隐藏状态类型）、Training 和 ApplyZoneout
    if (training) {
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperationsQuantDynamic<HT, true, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_,
                                                    tmp_Rh_.data(), bx, br, h, h_out, v,
                                                    zoneout_prob, zoneout_mask, rescale_param_);
        } else {
            kernel::PointwiseOperationsQuantDynamic<HT, true, false>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_,
                                                    tmp_Rh_.data(), bx, br, h, h_out, v, 0.0f,
                                                    nullptr, rescale_param_);
        }
    } else {
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperationsQuantDynamic<HT, false, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_,
                                                    tmp_Rh_.data(), bx, br, h, h_out, nullptr,
                                                    zoneout_prob, zoneout_mask, rescale_param_);
        } else {
            kernel::PointwiseOperationsQuantDynamic<HT, false, false>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_,
                                                    tmp_Rh_.data(), bx, br, h, h_out, nullptr, 0.0f,
                                                    nullptr, rescale_param_);
        }
    }
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

    // 保存位宽配置（用于运行时选择正确的 kernel 实例）
    rescale_param_.bitwidth_config_ = parms.bitwidth_config_;

    // 调试用：保存完整的量化参数
    rescale_param_.test = parms;
}

// C = input_size(输入维度), H = hidden_size(隐藏层维度),
// T = time_steps(时间步), N = batch_size(批量大小)
template <typename XT, typename HT, typename WT, typename RT>
void ForwardPassQuant<XT, HT, WT, RT>::Run(
    const int steps,           // 时间步数, 序列长度T
    const WT *W,               // [C,H*3], 输入到隐藏状态的权重矩阵（Wx）
    const RT *R,               // [H,H*3], 隐状态到隐藏状态的权重矩阵（Rh）
    const int32_t *bx,         // [H*3], 输入偏置（bias for W）
    const int32_t *br,         // [H*3], 隐状态偏置（bias for R）
    const XT *x,               // [N*T,C], 输入序列
    HT *h,                     // [(T+1)*N,H], 输出隐藏状态
    int32_t *v,                // [T*N,H*4], 中间激活值（训练模式需要）
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
                        h + i * NH,                // 输入 h
                        h + (i + 1) * NH,          // 输出 h
                        v + i * NH * 4,            // 中间激活
                        tmp_Wx_.data() + i * NH3,  // 当前时间步的 Wx
                        zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    cublasSetStream(data_->blas_handle, save_stream);
}

// 显式实例化：四个类型参数相同的情况
template struct ForwardPassQuant<int8_t, int8_t, int8_t, int8_t>;
template struct ForwardPassQuant<int16_t, int16_t, int16_t, int16_t>;

}  // namespace gru
