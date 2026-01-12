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
        const int64_t result = rshift_round(acc, shift_per_row[row]) + zp_out;

        // 根据位宽配置 clamp 并输出（列主序：C[n*M + m]）
        C[col * M + row] = clamp_by_bitwidth(static_cast<int32_t>(result), output_bw);
    }
}

// 融合 GEMM + bias: C = rshift(A * (B - zp_B), shift_gemm) + rshift(bias, shift_bias) + zp_out
// A: [M, K] 权重，列主序
// B: [K, N] 输入，列主序
// C: [M, N] 输出，列主序
// bias: [M] 偏置（per-channel）
// shift_gemm_per_row: [M] GEMM 的 per-row shift
// shift_bias_per_row: [M] bias 的 per-row shift
__global__ void quantizedGemmBiasFused(
    const int32_t *__restrict__ A,               // [M, K] 权重，列主序
    const int32_t *__restrict__ B,               // [K, N] 输入，列主序
    int32_t *__restrict__ C,                     // [M, N] 输出，列主序
    const int32_t *__restrict__ bias,            // [M] 偏置
    int M, int N, int K,
    int32_t zp_B,                                // 输入的 zero-point
    const int8_t *__restrict__ shift_gemm_per_row,   // [M] GEMM per-row shift
    const int8_t *__restrict__ shift_bias_per_row,   // [M] bias per-row shift
    int32_t zp_out,                              // 输出的 zero-point
    QuantBitWidth output_bw                      // 输出位宽配置
) {
    __shared__ int32_t As[TILE_SIZE][TILE_SIZE + 1];
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

    // 写回结果：rshift(GEMM, shift_gemm) + rshift(bias, shift_bias) + zp_out
    if (row < M && col < N) {
        const int8_t n_gemm = shift_gemm_per_row[row];
        const int8_t n_bias = shift_bias_per_row[row];
        const int32_t bias_val = bias[row];

        // 使用 rshift_round 进行 rescale
        const int64_t gemm_result = rshift_round(acc, n_gemm);
        const int64_t bias_result = rshift_round(static_cast<int64_t>(bias_val), n_bias);

        // 合并结果
        const int64_t result = gemm_result + bias_result + zp_out;

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
// 3. GRU Gate Functions - 使用 quantize_ops_helper.h 中的模板函数
// ============================================================================
// computeZ, computeR, computeG, computeH 已统一定义在 quantize_ops_helper.h 中
// 使用模板函数支持 GateQuantParams（门计算参数）
// 调试代码通过 DEBUG_QUANT 和 DEBUG_QUANT_DETAIL 宏控制

// ============================================================================
// 4. Pointwise Kernel - GRU 逐点运算 (GEMM+bias 融合版本)
// ============================================================================
// 每个线程处理一个 (batch, hidden) 位置
// 所有量化值使用 int32_t 存储
// Wx_bx = W*x + bx, Rh_br = R*h + br (GEMM 已融合 bias)

template <bool Training, bool ApplyZoneout>
__global__ void PointwiseOperationsQuant(
    const int batch_dim, const int hidden_dim, 
    const int32_t *Wx_bx,   // GEMM+bias 融合输出: W*x + bx [batch, hidden*3]
    const int32_t *Rh_br,   // GEMM+bias 融合输出: R*h + br [batch, hidden*3]
    const int32_t *h, int32_t *h_out, int32_t *v,
    const float zoneout_prob, const int32_t *zoneout_mask, const GateQuantParams rescale_params) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim) return;

    const int weight_idx = col * (hidden_dim * 3) + row;
    const int output_idx = col * hidden_dim + row;
    const int z_idx = weight_idx + 0 * hidden_dim;
    const int r_idx = weight_idx + 1 * hidden_dim;
    const int g_idx = weight_idx + 2 * hidden_dim;

#ifdef DEBUG_QUANT_DETAIL
    // ============ 调试：同时进行浮点计算用于对比 ============
    const int debug_idx = (col == 0 && row < 3) ? row : -1;

    if (debug_idx >= 0) {
        // 反量化 Wx_bx, Rh_br (GEMM+bias 融合结果)
        const float scale_Wx = 1.0f / (float)(1 << rescale_params.test.exp2_inv_Wx_);
        const float scale_Rh = 1.0f / (float)(1 << rescale_params.test.exp2_inv_Rh_);
        const float scale_h = 1.0f / (float)(1 << rescale_params.test.exp2_inv_h_);

        // 反量化 GEMM+bias 融合结果
        float Wx_bx_z_fp = (float)(Wx_bx[z_idx] - rescale_params.zp_Wx_) * scale_Wx;
        float Wx_bx_r_fp = (float)(Wx_bx[r_idx] - rescale_params.zp_Wx_) * scale_Wx;
        float Wx_bx_g_fp = (float)(Wx_bx[g_idx] - rescale_params.zp_Wx_) * scale_Wx;

        float Rh_br_z_fp = (float)(Rh_br[z_idx] - rescale_params.zp_Rh_) * scale_Rh;
        float Rh_br_r_fp = (float)(Rh_br[r_idx] - rescale_params.zp_Rh_) * scale_Rh;
        float Rh_br_g_fp = (float)(Rh_br[g_idx] - rescale_params.zp_Rh_) * scale_Rh;

        // 反量化 h_old
        float h_old_fp = (float)(h[output_idx] - rescale_params.zp_h_) * scale_h;

        // ========== 浮点 GRU 计算 (使用融合后的值) ==========
        float z_pre_fp = Wx_bx_z_fp + Rh_br_z_fp;
        float z_fp = sigmoid_fp(z_pre_fp);

        float r_pre_fp = Wx_bx_r_fp + Rh_br_r_fp;
        float r_fp = sigmoid_fp(r_pre_fp);

        float g_pre_fp = Wx_bx_g_fp + r_fp * Rh_br_g_fp;
        float g_fp = tanh_fp(g_pre_fp);

        float h_new_fp = z_fp * h_old_fp + (1.0f - z_fp) * g_fp;

        printf("\n===== [DEBUG idx=%d batch=0] =====\n", debug_idx);
        printf("Wx_bx: z_q=%d r_q=%d g_q=%d | z_fp=%.4f r_fp=%.4f g_fp=%.4f\n", 
               Wx_bx[z_idx], Wx_bx[r_idx], Wx_bx[g_idx], Wx_bx_z_fp, Wx_bx_r_fp, Wx_bx_g_fp);
        printf("Rh_br: z_q=%d r_q=%d g_q=%d | z_fp=%.4f r_fp=%.4f g_fp=%.4f\n", 
               Rh_br[z_idx], Rh_br[r_idx], Rh_br[g_idx], Rh_br_z_fp, Rh_br_r_fp, Rh_br_g_fp);
        printf("h_old: q=%d fp=%.4f\n", h[output_idx], h_old_fp);
        printf("[FLOAT] z_pre=%.4f z=%.4f | r_pre=%.4f r=%.4f | g_pre=%.4f g=%.4f | h_new=%.4f\n",
               z_pre_fp, z_fp, r_pre_fp, r_fp, g_pre_fp, g_fp, h_new_fp);
    }
#else
    const int debug_idx = -1;
#endif

    // GRU 门计算 (使用 GEMM+bias 融合版本)
    const int32_t z = computeZ(Wx_bx[z_idx], Rh_br[z_idx], rescale_params, debug_idx);

    const int32_t r = computeR(Wx_bx[r_idx], Rh_br[r_idx], rescale_params, debug_idx);

    int32_t Rh_br_g;
    const int32_t g = computeG(Wx_bx[g_idx], Rh_br[g_idx], r, rescale_params, Rh_br_g, debug_idx);

    // Training: 保存中间值
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh_br_g;
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
    tmp_Wx_bx_.resize(hidden3 * steps * batch_size);
    tmp_Rh_br_.resize(hidden3 * batch_size);

    // 权重和常量
    if (W_sum_mul_x_zp_.size() == 0) {
        W_sum_mul_x_zp_.resize(hidden3);
        R_sum_mul_h_zp_.resize(hidden3);
    }

    // INT8 GEMM 优化缓冲区（当位宽 <= 8 时使用）
    const auto &bw_cfg = gate_params_.bitwidth_config_;
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
    computeWeightSumMulzp(W, W_sum_mul_x_zp_.data(), linear_params_.zp_x_,
                          linear_params_.n_W_mul_x_div_Wx_.data(), hidden_size * 3, input_size,
                          stream);

    // 计算 R_sum_mul_h_zp
    computeWeightSumMulzp(R, R_sum_mul_h_zp_.data(), linear_params_.zp_h_,
                          linear_params_.n_R_mul_h_div_Rh_.data(), hidden_size * 3, hidden_size,
                          stream);

    cudaStreamSynchronize(stream);
    weight_sums_computed_ = true;
}

void ForwardPassQuant::ComputeWxBx(const int32_t *W, const int32_t *x, const int32_t *bx, int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[1];

    const int M = hidden_size * 3;
    const int N = steps * batch_size;
    const int K = input_size;

    // 使用 GEMM+bias 融合 kernel: W*x + bx
    dim3 blockDim(kernel::TILE_SIZE, kernel::TILE_SIZE);
    dim3 gridDim((N + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE,
                 (M + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE);

    kernel::quantizedGemmBiasFused<<<gridDim, blockDim, 0, stream>>>(
        W, x, tmp_Wx_bx_.data(), bx, M, N, K,
        linear_params_.zp_x_,
        linear_params_.n_W_mul_x_div_Wx_.data(),
        linear_params_.n_bx_div_Wx_.data(),
        gate_params_.zp_Wx_,
        gate_params_.bitwidth_config_.Wx_);
}

void ForwardPassQuant::ComputeRhBr(const int32_t *R, const int32_t *h, const int32_t *br) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[0];

    const int M = hidden_size * 3;
    const int N = batch_size;
    const int K = hidden_size;

    // 使用 GEMM+bias 融合 kernel: R*h + br
    dim3 blockDim(kernel::TILE_SIZE, kernel::TILE_SIZE);
    dim3 gridDim((N + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE,
                 (M + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE);

    kernel::quantizedGemmBiasFused<<<gridDim, blockDim, 0, stream>>>(
        R, h, tmp_Rh_br_.data(), br, M, N, K,
        linear_params_.zp_h_,
        linear_params_.n_R_mul_h_div_Rh_.data(),
        linear_params_.n_br_div_Rh_.data(),
        gate_params_.zp_Rh_,
        gate_params_.bitwidth_config_.Rh_);
}

void ForwardPassQuant::IterateInternal(
    const int32_t *R,         // [H,H*3]
    const int32_t *br,        // [H*3] (用于 GEMM+bias 融合)
    const int32_t *h,         // [N,H]
    int32_t *h_out,           // [N,H]
    int32_t *v,               // [N,H*4]
    const int32_t *cur_Wx_bx, // [N,H*3] 当前时间步的 W*x + bx 结果 (GEMM+bias 融合)
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

    // 计算 R*h + br GEMM+bias 融合（结果存入 tmp_Rh_br_）
    ComputeRhBr(R, h, br);

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    // 启动量化 GRU kernel（使用 GEMM+bias 融合版本，传递 gate_params_）
    if (training) {
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperationsQuant<true, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_bx,
                                                    tmp_Rh_br_.data(), h, h_out, v,
                                                    zoneout_prob, zoneout_mask, gate_params_);
        } else {
            kernel::PointwiseOperationsQuant<true, false>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_bx,
                                                    tmp_Rh_br_.data(), h, h_out, v, 0.0f,
                                                    nullptr, gate_params_);
        }
    } else {
        if (zoneout_prob && zoneout_mask) {
            kernel::PointwiseOperationsQuant<false, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_bx,
                                                    tmp_Rh_br_.data(), h, h_out, nullptr,
                                                    zoneout_prob, zoneout_mask, gate_params_);
        } else {
            kernel::PointwiseOperationsQuant<false, false>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_Wx_bx,
                                                    tmp_Rh_br_.data(), h, h_out, nullptr, 0.0f,
                                                    nullptr, gate_params_);
        }
    }
}

void ForwardPassQuant::setRescaleParam(const GRUQuantitativeParameters &parms) {
    const int channel = parms.hidden_ * 3;

    // ==================== Linear 层参数（per-channel）====================
    linear_params_.zp_x_ = parms.zp_x_;
    linear_params_.zp_h_ = parms.zp_h_;

    // 计算并存储 per-channel 重缩放参数
    std::vector<int8_t> n_W_mul_x_div_Wx(channel);
    std::vector<int8_t> n_R_mul_h_div_Rh(channel);
    std::vector<int8_t> n_bx_div_Wx(channel);
    std::vector<int8_t> n_br_div_Rh(channel);

    for (int idx = 0; idx < channel; ++idx) {
        n_W_mul_x_div_Wx[idx] = (parms.exp2_inv_W_[idx] + parms.exp2_inv_x_) - parms.exp2_inv_Wx_;
        n_R_mul_h_div_Rh[idx] = (parms.exp2_inv_R_[idx] + parms.exp2_inv_h_) - parms.exp2_inv_Rh_;
        n_bx_div_Wx[idx] = parms.exp2_inv_bx_[idx] - parms.exp2_inv_Wx_;
        n_br_div_Rh[idx] = parms.exp2_inv_br_[idx] - parms.exp2_inv_Rh_;
    }

    linear_params_.n_W_mul_x_div_Wx_ = dev::vector<int8_t>(n_W_mul_x_div_Wx);
    linear_params_.n_bx_div_Wx_ = dev::vector<int8_t>(n_bx_div_Wx);
    linear_params_.n_R_mul_h_div_Rh_ = dev::vector<int8_t>(n_R_mul_h_div_Rh);
    linear_params_.n_br_div_Rh_ = dev::vector<int8_t>(n_br_div_Rh);

#ifdef DEBUG
    linear_params_.exp2_inv_bx_ = dev::vector<int8_t>(parms.exp2_inv_bx_);
    linear_params_.exp2_inv_br_ = dev::vector<int8_t>(parms.exp2_inv_br_);
#endif

    // ==================== 门计算参数（标量）====================
    gate_params_.zp_Wx_ = parms.zp_Wx_;
    gate_params_.zp_Rh_ = parms.zp_Rh_;
    gate_params_.zp_h_ = parms.zp_h_;

    // z门
    gate_params_.zp_z_pre_ = parms.zp_z_pre_;
    gate_params_.zp_z_out_ = parms.zp_z_out_;
    gate_params_.exp2_inv_Wx_div_z_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_z_pre_;
    gate_params_.exp2_inv_Rh_div_z_pre_ = parms.exp2_inv_Rh_ - parms.exp2_inv_z_pre_;

    // r门
    gate_params_.zp_r_pre_ = parms.zp_r_pre_;
    gate_params_.zp_r_out_ = parms.zp_r_out_;
    gate_params_.exp2_inv_Wx_div_r_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_r_pre_;
    gate_params_.exp2_inv_Rh_div_r_pre_ = parms.exp2_inv_Rh_ - parms.exp2_inv_r_pre_;

    // g门
    gate_params_.zp_g_pre_ = parms.zp_g_pre_;
    gate_params_.zp_g_out_ = parms.zp_g_out_;
    gate_params_.n_r_mul_Rh_div_rRh_ =
        (parms.exp2_inv_r_out_ + parms.exp2_inv_Rh_) - parms.exp2_inv_rRh_;
    gate_params_.zp_rRh_ = parms.zp_rRh_;
    gate_params_.n_Wx_div_g_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_g_pre_;
    gate_params_.n_rRh_div_g_pre_ = parms.exp2_inv_rRh_ - parms.exp2_inv_g_pre_;

    // h_new
    gate_params_.one_in_z_scale_ = rshift_round(1, -parms.exp2_inv_z_out_) + parms.zp_z_out_;
    gate_params_.zp_new_contrib_ = parms.zp_new_contrib_;
    gate_params_.n_z_out_mul_g_div_new_contrib_ =
        (parms.exp2_inv_z_out_ + parms.exp2_inv_g_out_) - parms.exp2_inv_new_contrib_;
    gate_params_.zp_old_contrib_ = parms.zp_old_contrib_;
    gate_params_.n_z_mul_h_div_old_contrib_ =
        (parms.exp2_inv_z_out_ + parms.exp2_inv_h_) - parms.exp2_inv_old_contrib_;
    gate_params_.n_new_contrib_div_h_ = parms.exp2_inv_new_contrib_ - parms.exp2_inv_h_;
    gate_params_.n_old_contrib_div_h_ = parms.exp2_inv_old_contrib_ - parms.exp2_inv_h_;

    // 位宽配置和 LUT
    gate_params_.bitwidth_config_ = parms.bitwidth_config_;
    gate_params_.sigmoid_z_lut_ = parms.sigmoid_z_lut_;
    gate_params_.sigmoid_r_lut_ = parms.sigmoid_r_lut_;
    gate_params_.tanh_g_lut_ = parms.tanh_g_lut_;

#ifdef DEBUG
    gate_params_.test = parms;
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

    // 计算 W*x + bx GEMM+bias 融合（所有时间步一次性计算，结果存入 tmp_Wx_bx_）
    ComputeWxBx(W, x, bx, steps);

    // 同步 Wx_bx 计算
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, br,
                        h + i * NH,                    // 输入 h
                        h + (i + 1) * NH,              // 输出 h
                        v + i * NH * 4,                // 中间激活
                        tmp_Wx_bx_.data() + i * NH3,   // 当前时间步的 W*x + bx
                        zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    cublasSetStream(data_->blas_handle, save_stream);
}

}  // namespace gru

