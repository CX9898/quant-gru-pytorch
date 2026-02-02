// ============================================================================
// gru_forward_gpu_quant.cu - 量化 GRU 前向传播 CUDA 实现
// ============================================================================
//
// 文件结构:
//   1. GEMM Kernels        - 量化矩阵乘法 (INT32 存储)
//   2. Rescale Kernels     - GEMM 结果缩放
//   3. GRU Gate Functions  - 门计算函数 (computeUpdateGate/ResetGate/NewGate/HiddenState)
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
#include "parallel_algorithm.h"

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
// 融合 GEMM + bias: C = rshift(A * (B - zp_B), shift_gemm) + rshift(bias, shift_bias) + zp_out
// 
// @tparam Training 是否训练模式（决定是否使用 mask）
// @param C_mask 训练模式时保存 clamp mask，推理模式时可为 nullptr
// 辅助函数：根据粒度配置获取 rescale 后的 shift 值（device 函数）
__device__ __forceinline__ int8_t get_shift_gemm(
    int8_t granularity,
    int row,
    int hidden_size,
    int8_t shift_tensor,
    const int8_t shift_gate[3],
    const int8_t *shift_channel
) {
    if (granularity == 0) {  // PER_TENSOR
        return shift_tensor;
    } else if (granularity == 1) {  // PER_GATE
        int gate_idx = row / hidden_size;
        return shift_gate[gate_idx];
    } else {  // PER_CHANNEL
        return shift_channel[row];
    }
}

template <bool Training>
__global__ void quantizedGemmBiasFused(
    const int32_t *__restrict__ A,               // [M, K] 权重，列主序
    const int32_t *__restrict__ B,               // [K, N] 输入，列主序
    int32_t *__restrict__ C,                     // [M, N] 输出，列主序
    const int32_t *__restrict__ bias,            // [M] 偏置
    uint8_t *__restrict__ C_mask,                // [M, N] clamp mask，训练模式时有效
    int M, int N, int K,
    int32_t zp_B,                                // 输入的 zero-point
    // 粒度配置（gemm 和 bias 的粒度，通过参数内存传递，访问快）
    int8_t gemm_granularity,  // GEMM 的粒度（W_granularity 或 R_granularity）
    int8_t bias_granularity,  // bias 的粒度（bw_granularity 或 br_granularity）
    int hidden_size,
    // Per-tensor rescale 参数（已计算 rescale，通过参数内存传递）
    int8_t shift_gemm_tensor,  // GEMM rescale per-tensor shift
    int8_t shift_bias_tensor,  // bias rescale per-tensor shift
    // Per-gate rescale 参数（已计算 rescale，通过参数内存传递）
    const int8_t shift_gemm_gate[3],  // GEMM rescale per-gate shift [z, r, g]
    const int8_t shift_bias_gate[3],  // bias rescale per-gate shift [z, r, g]
    // Per-channel 参数（数组指针，PER_CHANNEL 粒度时使用）
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
        // 根据粒度配置获取 rescale 后的 shift 值（通过参数内存访问，比全局内存快）
        const int8_t n_gemm = get_shift_gemm(gemm_granularity, row, hidden_size,
                                            shift_gemm_tensor, shift_gemm_gate, shift_gemm_per_row);
        const int8_t n_bias = get_shift_gemm(bias_granularity, row, hidden_size,
                                            shift_bias_tensor, shift_bias_gate, shift_bias_per_row);
        const int32_t bias_val = bias[row];

        // 使用 rshift_round 进行 rescale
        const int64_t bias_result = rshift_round(static_cast<int64_t>(bias_val), n_bias);
        const int64_t gemm_result = rshift_round(acc + static_cast<int64_t>(bias_result), n_gemm);

        // 合并结果
        const int64_t result = gemm_result + zp_out;

        // 根据位宽配置 clamp 并输出（列主序：C[n*M + m]）
        uint8_t was_clamped;
        C[col * M + row] = clamp_by_bitwidth<Training>(static_cast<int32_t>(result), output_bw, 
                                                        Training ? &was_clamped : nullptr);
        if constexpr (Training) {
            C_mask[col * M + row] = was_clamped;
        }
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
// computeUpdateGate, computeResetGate, computeNewGate, computeHiddenState 已统一定义在 quantize_ops_helper.h 中
// 使用模板函数支持 GateQuantParams（门计算参数）
// 调试代码通过 DEBUG_QUANT 和 DEBUG_QUANT_DETAIL 宏控制

// ============================================================================
// 4. Pointwise Kernel - GRU 逐点运算 (Linear 融合版本)
// ============================================================================
// 每个线程处理一个 (batch, hidden) 位置
// 所有量化值使用 int32_t 存储
// weight_ih_linear = W*x + bw, weight_hh_linear = R*h + br (Linear 变换)

template <bool Training, bool ApplyZoneout>
__global__ void PointwiseOperationsQuant(
    const int batch_dim, const int hidden_dim, 
    const int32_t *weight_ih_linear,  // 输入 Linear 变换: W*x + bw [batch, hidden*3]
    const int32_t *weight_hh_linear,  // 隐状态 Linear 变换: R*h + br [batch, hidden*3]
    const int32_t *h, int32_t *h_out, int32_t *v,
    const float zoneout_prob, const int32_t *zoneout_mask, const GateQuantParams gate_params,
    uint8_t *gate_input_mask = nullptr,         // [batch, hidden*3] 门输入 clamp mask
    uint8_t *gate_output_mask = nullptr,        // [batch, hidden*3] 门输出 clamp mask
    uint8_t *h_mask = nullptr                   // [batch, hidden] 隐状态输出 mask
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim) return;

    const int weight_idx = col * (hidden_dim * 3) + row;
    const int output_idx = col * hidden_dim + row;
    const int update_idx = weight_idx + 0 * hidden_dim;
    const int reset_idx = weight_idx + 1 * hidden_dim;
    const int new_idx = weight_idx + 2 * hidden_dim;

#ifdef DEBUG_QUANT_DETAIL
    // ============ 调试：同时进行浮点计算用于对比 ============
    const int debug_idx = (col == 0 && row < 3) ? row : -1;

    if (debug_idx >= 0) {
        // 反量化 Linear 变换结果（使用通用函数）
        float ih_u_fp = dequantize<int32_t>(weight_ih_linear[update_idx], gate_params.test.shift_weight_ih_linear_, gate_params.zp_weight_ih_linear_);
        float ih_r_fp = dequantize<int32_t>(weight_ih_linear[reset_idx], gate_params.test.shift_weight_ih_linear_, gate_params.zp_weight_ih_linear_);
        float ih_n_fp = dequantize<int32_t>(weight_ih_linear[new_idx], gate_params.test.shift_weight_ih_linear_, gate_params.zp_weight_ih_linear_);

        float hh_u_fp = dequantize<int32_t>(weight_hh_linear[update_idx], gate_params.test.shift_weight_hh_linear_, gate_params.zp_weight_hh_linear_);
        float hh_r_fp = dequantize<int32_t>(weight_hh_linear[reset_idx], gate_params.test.shift_weight_hh_linear_, gate_params.zp_weight_hh_linear_);
        float hh_n_fp = dequantize<int32_t>(weight_hh_linear[new_idx], gate_params.test.shift_weight_hh_linear_, gate_params.zp_weight_hh_linear_);

        // 反量化 h_old
        float h_old_fp = dequantize<int32_t>(h[output_idx], gate_params.test.shift_h_, gate_params.zp_h_);

        // ========== 浮点 GRU 计算 ==========
        float u_pre_fp = ih_u_fp + hh_u_fp;
        float u_fp = sigmoid_fp(u_pre_fp);

        float r_pre_fp = ih_r_fp + hh_r_fp;
        float r_fp = sigmoid_fp(r_pre_fp);

        float n_pre_fp = ih_n_fp + r_fp * hh_n_fp;
        float n_fp = tanh_fp(n_pre_fp);

        float h_new_fp = u_fp * h_old_fp + (1.0f - u_fp) * n_fp;

        printf("\n===== [DEBUG idx=%d batch=0] =====\n", debug_idx);
        printf("weight_ih_linear: u_q=%d r_q=%d n_q=%d | u_fp=%.4f r_fp=%.4f n_fp=%.4f\n", 
               weight_ih_linear[update_idx], weight_ih_linear[reset_idx], weight_ih_linear[new_idx], ih_u_fp, ih_r_fp, ih_n_fp);
        printf("weight_hh_linear: u_q=%d r_q=%d n_q=%d | u_fp=%.4f r_fp=%.4f n_fp=%.4f\n", 
               weight_hh_linear[update_idx], weight_hh_linear[reset_idx], weight_hh_linear[new_idx], hh_u_fp, hh_r_fp, hh_n_fp);
        printf("h_old: q=%d fp=%.4f\n", h[output_idx], h_old_fp);
        printf("[FLOAT] u_pre=%.4f u=%.4f | r_pre=%.4f r=%.4f | n_pre=%.4f n=%.4f | h_new=%.4f\n",
               u_pre_fp, u_fp, r_pre_fp, r_fp, n_pre_fp, n_fp, h_new_fp);
    }
#else
    const int debug_idx = -1;
#endif

    // 统一声明 mask 变量（函数内部会根据 Training 模板参数决定是否使用）
    uint8_t update_input_mask, update_output_mask;
    uint8_t reset_input_mask, reset_output_mask;
    uint8_t new_input_mask, new_output_mask;
    uint8_t h_out_mask;
    
    // 统一声明门计算结果变量
    int32_t update_gate, reset_gate, new_gate, cur_h;
    
    // 直接调用函数，函数内部会根据 Training 模板参数决定实现
    update_gate = computeUpdateGate<Training>(
        weight_ih_linear[update_idx], weight_hh_linear[update_idx], gate_params,
        &update_input_mask, &update_output_mask, debug_idx);

    reset_gate = computeResetGate<Training>(
        weight_ih_linear[reset_idx], weight_hh_linear[reset_idx], gate_params,
        &reset_input_mask, &reset_output_mask, debug_idx);

    new_gate = computeNewGate<Training>(
        weight_ih_linear[new_idx], weight_hh_linear[new_idx], reset_gate, gate_params,
        &new_input_mask, &new_output_mask, debug_idx);

    // 保存门输入 mask（只在训练模式时执行）
    if constexpr (Training) {
        if (gate_input_mask != nullptr) {
            gate_input_mask[update_idx] = update_input_mask;
            gate_input_mask[reset_idx] = reset_input_mask;
            gate_input_mask[new_idx] = new_input_mask;
        }
        
        // 保存门输出 mask
        if (gate_output_mask != nullptr) {
            gate_output_mask[update_idx] = update_output_mask;
            gate_output_mask[reset_idx] = reset_output_mask;
            gate_output_mask[new_idx] = new_output_mask;
        }

        // Training: 保存中间值
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = update_gate;
        v[base_v_idx + 1 * hidden_dim] = reset_gate;
        v[base_v_idx + 2 * hidden_dim] = new_gate;
        v[base_v_idx + 3 * hidden_dim] = weight_hh_linear[new_idx];  // 直接使用 weight_hh_linear[new_idx]
    }

    // 计算新的隐藏状态
    cur_h = computeHiddenState<Training>(update_gate, new_gate, h[output_idx], gate_params, &h_out_mask, debug_idx);

    // 保存隐状态 mask（只在训练模式时执行）
    if constexpr (Training) {
        if (h_mask != nullptr) {
            h_mask[output_idx] = h_out_mask;
        }
    }

    // Zoneout（如果启用）
    if constexpr (ApplyZoneout) {
        const int32_t mask = zoneout_mask[output_idx];
        cur_h = mask * h[output_idx] + (gate_params.quant_one_in_update_gate_scale_ - mask) * cur_h;
    }

#ifdef DEBUG_QUANT_DETAIL
    if (debug_idx >= 0) {
        // 反量化最终结果与浮点对比（使用通用函数）
        float u_quant_fp = dequantize<int32_t>(update_gate, gate_params.test.shift_update_gate_output_, gate_params.zp_update_gate_output_);
        float n_quant_fp = dequantize<int32_t>(new_gate, gate_params.test.shift_new_gate_output_, gate_params.zp_new_gate_output_);
        float h_quant_fp = dequantize<int32_t>(cur_h, gate_params.test.shift_h_, gate_params.zp_h_);

        printf("[QUANT] u_q=%d u_fp=%.4f | n_q=%d n_fp=%.4f | h_q=%d h_fp=%.4f\n", update_gate, u_quant_fp, new_gate,
               n_quant_fp, cur_h, h_quant_fp);
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
    tmp_weight_ih_linear_.resize(hidden3 * steps * batch_size);
    tmp_weight_hh_linear_.resize(hidden3 * batch_size);

    // 权重和常量
    if (W_sum_mul_x_zp_.size() == 0) {
        W_sum_mul_x_zp_.resize(hidden3);
        R_sum_mul_h_zp_.resize(hidden3);
    }

    // INT8 GEMM 优化缓冲区（当位宽 <= 8 时使用）
    const auto &bw_cfg = gate_params_.bitwidth_config_;
    if (bw_cfg.W_.bits_ <= 8 && bw_cfg.x_.bits_ <= 8) {
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
    const int hidden3 = hidden_size * 3;
    const cudaStream_t stream = data_->stream[1];

    // 计算 W_sum_mul_x_zp（如果 zp_x != 0，否则直接清零）
    if (linear_params_.zp_x_ != 0) {
        computeWeightSumMulzp(W, W_sum_mul_x_zp_.data(), linear_params_.zp_x_,
                              linear_params_.shift_gemm_x_to_weight_ih_linear_.data(), hidden3, input_size,
                              stream);
    } else {
        dev::fill_n(W_sum_mul_x_zp_.data(), hidden3, static_cast<int64_t>(0));
    }

    // 计算 R_sum_mul_h_zp（如果 zp_h != 0，否则直接清零）
    if (linear_params_.zp_h_ != 0) {
        computeWeightSumMulzp(R, R_sum_mul_h_zp_.data(), linear_params_.zp_h_,
                              linear_params_.shift_gemm_h_to_weight_hh_linear_.data(), hidden3, hidden_size,
                              stream);
    } else {
        dev::fill_n(R_sum_mul_h_zp_.data(), hidden3, static_cast<int64_t>(0));
    }

    cudaStreamSynchronize(stream);
    weight_sums_computed_ = true;
}

void ForwardPassQuant::ComputeLinearX(const int32_t *W, const int32_t *x, const int32_t *bw, int steps,
                                       uint8_t *weight_ih_linear_mask) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[1];
    const bool training = data_->training;

    const int M = hidden_size * 3;
    const int N = steps * batch_size;
    const int K = input_size;

    // 使用 GEMM+bias 融合 kernel: W*x + bw
    dim3 blockDim(kernel::TILE_SIZE, kernel::TILE_SIZE);
    dim3 gridDim((N + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE,
                 (M + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE);

    // 使用模板版本的 quantizedGemmBiasFused，内部根据 Training 决定是否使用 mask
    // ComputeLinearX 使用 W_granularity 和 bw_granularity
    if (training) {
        kernel::quantizedGemmBiasFused<true><<<gridDim, blockDim, 0, stream>>>(
            W, x, tmp_weight_ih_linear_.data(), bw, weight_ih_linear_mask, M, N, K,
            linear_params_.zp_x_,
            // 粒度配置（W 和 bw）
            linear_params_.W_granularity_, linear_params_.bw_granularity_,
            linear_params_.hidden_size_,
            // Per-tensor 参数
            linear_params_.shift_gemm_x_tensor_, linear_params_.shift_bw_tensor_,
            // Per-gate 参数
            linear_params_.shift_gemm_x_gate_, linear_params_.shift_bw_gate_,
            // Per-channel 参数（数组指针）
            linear_params_.shift_gemm_x_to_weight_ih_linear_.data(),
            linear_params_.shift_bw_to_weight_ih_linear_.data(),
            gate_params_.zp_weight_ih_linear_,
            gate_params_.bitwidth_config_.weight_ih_linear_);
    } else {
        kernel::quantizedGemmBiasFused<false><<<gridDim, blockDim, 0, stream>>>(
            W, x, tmp_weight_ih_linear_.data(), bw, nullptr, M, N, K,
            linear_params_.zp_x_,
            // 粒度配置（W 和 bw）
            linear_params_.W_granularity_, linear_params_.bw_granularity_,
            linear_params_.hidden_size_,
            // Per-tensor 参数
            linear_params_.shift_gemm_x_tensor_, linear_params_.shift_bw_tensor_,
            // Per-gate 参数
            linear_params_.shift_gemm_x_gate_, linear_params_.shift_bw_gate_,
            // Per-channel 参数（数组指针）
            linear_params_.shift_gemm_x_to_weight_ih_linear_.data(),
            linear_params_.shift_bw_to_weight_ih_linear_.data(),
            gate_params_.zp_weight_ih_linear_,
            gate_params_.bitwidth_config_.weight_ih_linear_);
    }
}

void ForwardPassQuant::ComputeLinearH(const int32_t *R, const int32_t *h, const int32_t *br,
                                       uint8_t *weight_hh_linear_mask) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[0];
    const bool training = data_->training;

    const int M = hidden_size * 3;
    const int N = batch_size;
    const int K = hidden_size;

    // 使用 GEMM+bias 融合 kernel: R*h + br
    dim3 blockDim(kernel::TILE_SIZE, kernel::TILE_SIZE);
    dim3 gridDim((N + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE,
                 (M + kernel::TILE_SIZE - 1) / kernel::TILE_SIZE);

    // 使用模板版本的 quantizedGemmBiasFused，内部根据 Training 决定是否使用 mask
    // ComputeLinearH 使用 R_granularity 和 br_granularity
    if (training) {
        kernel::quantizedGemmBiasFused<true><<<gridDim, blockDim, 0, stream>>>(
            R, h, tmp_weight_hh_linear_.data(), br, weight_hh_linear_mask, M, N, K,
            linear_params_.zp_h_,
            // 粒度配置（R 和 br）
            linear_params_.R_granularity_, linear_params_.br_granularity_,
            linear_params_.hidden_size_,
            // Per-tensor 参数
            linear_params_.shift_gemm_h_tensor_, linear_params_.shift_br_tensor_,
            // Per-gate 参数
            linear_params_.shift_gemm_h_gate_, linear_params_.shift_br_gate_,
            // Per-channel 参数（数组指针）
            linear_params_.shift_gemm_h_to_weight_hh_linear_.data(),
            linear_params_.shift_br_to_weight_hh_linear_.data(),
            gate_params_.zp_weight_hh_linear_,
            gate_params_.bitwidth_config_.weight_hh_linear_);
    } else {
        kernel::quantizedGemmBiasFused<false><<<gridDim, blockDim, 0, stream>>>(
            R, h, tmp_weight_hh_linear_.data(), br, nullptr, M, N, K,
            linear_params_.zp_h_,
            // 粒度配置（R 和 br）
            linear_params_.R_granularity_, linear_params_.br_granularity_,
            linear_params_.hidden_size_,
            // Per-tensor 参数
            linear_params_.shift_gemm_h_tensor_, linear_params_.shift_br_tensor_,
            // Per-gate 参数
            linear_params_.shift_gemm_h_gate_, linear_params_.shift_br_gate_,
            // Per-channel 参数（数组指针）
            linear_params_.shift_gemm_h_to_weight_hh_linear_.data(),
            linear_params_.shift_br_to_weight_hh_linear_.data(),
            gate_params_.zp_weight_hh_linear_,
            gate_params_.bitwidth_config_.weight_hh_linear_);
    }
}

void ForwardPassQuant::IterateInternal(
    const int32_t *R,           // [H,H*3]
    const int32_t *br,          // [H*3] (用于 Linear 变换)
    const int32_t *h,           // [N,H]
    int32_t *h_out,             // [N,H]
    int32_t *v,                 // [N,H*4]
    const int32_t *cur_weight_ih_linear, // [N,H*3] 当前时间步的 W*x + bw 结果
    const float zoneout_prob,
    const int32_t *zoneout_mask,  // Zoneout mask [N,H]
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask
) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    cublasSetStream(blas_handle, stream1);

    // 计算隐状态 Linear 变换: R*h + br（结果存入 tmp_weight_hh_linear_，可选 mask）
    ComputeLinearH(R, h, br, weight_hh_linear_mask);

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    const bool apply_zoneout = (zoneout_prob > 0.0f && zoneout_mask != nullptr);

    // 启动量化 GRU kernel（统一接口，所有模式都传递 mask 参数）
    if (training) {
        if (apply_zoneout) {
            kernel::PointwiseOperationsQuant<true, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_weight_ih_linear,
                                                    tmp_weight_hh_linear_.data(), h, h_out, v,
                                                    zoneout_prob, zoneout_mask, gate_params_,
                                                    gate_input_mask, gate_output_mask, h_mask);
        } else {
            kernel::PointwiseOperationsQuant<true, false>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_weight_ih_linear,
                                                    tmp_weight_hh_linear_.data(), h, h_out, v, 0.0f,
                                                    nullptr, gate_params_,
                                                    gate_input_mask, gate_output_mask, h_mask);
        }
    } else {
        if (apply_zoneout) {
            kernel::PointwiseOperationsQuant<false, true>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_weight_ih_linear,
                                                    tmp_weight_hh_linear_.data(), h, h_out, nullptr,
                                                    zoneout_prob, zoneout_mask, gate_params_,
                                                    nullptr, nullptr, nullptr);
        } else {
            kernel::PointwiseOperationsQuant<false, false>
                <<<gridDim, blockDim, 0, stream1>>>(batch_size, hidden_size, cur_weight_ih_linear,
                                                    tmp_weight_hh_linear_.data(), h, h_out, nullptr, 0.0f,
                                                    nullptr, gate_params_,
                                                    nullptr, nullptr, nullptr);
        }
    }
}

void ForwardPassQuant::setRescaleParam(const GRUQuantParams &parms) {
    const int channel = parms.hidden_ * 3;
    const int hidden_size = parms.hidden_;
    const auto& cfg = parms.bitwidth_config_;

    // ==================== Linear 层参数 =====================
    linear_params_.zp_x_ = parms.zp_x_;
    linear_params_.zp_h_ = parms.zp_h_;

    // 存储粒度配置和 hidden_size（用于 kernel 中判断）
    linear_params_.W_granularity_ = static_cast<int8_t>(cfg.W_granularity_);
    linear_params_.R_granularity_ = static_cast<int8_t>(cfg.R_granularity_);
    linear_params_.bw_granularity_ = static_cast<int8_t>(cfg.bw_granularity_);
    linear_params_.br_granularity_ = static_cast<int8_t>(cfg.br_granularity_);
    linear_params_.hidden_size_ = hidden_size;

    // 计算并存储 per-tensor rescale 参数（通过参数内存传递，访问快）
    if (cfg.W_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        int8_t shift_W = parms.shift_W_tensor_;
        linear_params_.shift_gemm_x_tensor_ = (shift_W + parms.shift_x_) - parms.shift_weight_ih_linear_;
        linear_params_.shift_bw_tensor_ = parms.shift_bw_tensor_ - (shift_W + parms.shift_x_);
    }
    if (cfg.R_granularity_ == OperatorQuantConfig::PER_TENSOR) {
        int8_t shift_R = parms.shift_R_tensor_;
        linear_params_.shift_gemm_h_tensor_ = (shift_R + parms.shift_h_) - parms.shift_weight_hh_linear_;
        linear_params_.shift_br_tensor_ = parms.shift_br_tensor_ - (shift_R + parms.shift_h_);
    }

    // 计算并存储 per-gate rescale 参数（通过参数内存传递，访问快）
    if (cfg.W_granularity_ == OperatorQuantConfig::PER_GATE) {
        for (int gate = 0; gate < 3; ++gate) {
            int8_t shift_W = parms.shift_W_gate_[gate];
            int8_t shift_bw = parms.shift_bw_gate_[gate];
            linear_params_.shift_gemm_x_gate_[gate] = (shift_W + parms.shift_x_) - parms.shift_weight_ih_linear_;
            linear_params_.shift_bw_gate_[gate] = shift_bw - (shift_W + parms.shift_x_);
        }
    }
    if (cfg.R_granularity_ == OperatorQuantConfig::PER_GATE) {
        for (int gate = 0; gate < 3; ++gate) {
            int8_t shift_R = parms.shift_R_gate_[gate];
            int8_t shift_br = parms.shift_br_gate_[gate];
            linear_params_.shift_gemm_h_gate_[gate] = (shift_R + parms.shift_h_) - parms.shift_weight_hh_linear_;
            linear_params_.shift_br_gate_[gate] = shift_br - (shift_R + parms.shift_h_);
        }
    }

    // 计算 per-channel 移位参数（仅 PER_CHANNEL 粒度时使用）
    // 对于 PER_TENSOR 和 PER_GATE，这些数组不会被使用，但仍需要分配以避免 kernel 中空指针
    std::vector<int8_t> shift_gemm_x(channel);
    std::vector<int8_t> shift_gemm_h(channel);
    std::vector<int8_t> shift_bw(channel);
    std::vector<int8_t> shift_br(channel);

    // 辅助函数：根据粒度配置获取 W 的 shift 值
    auto get_W_shift = [&](int idx) -> int8_t {
        if (cfg.W_granularity_ == OperatorQuantConfig::PER_TENSOR) {
            return parms.shift_W_tensor_;
        } else if (cfg.W_granularity_ == OperatorQuantConfig::PER_GATE) {
            int gate_idx = idx / hidden_size;
            return parms.shift_W_gate_[gate_idx];
        } else {  // PER_CHANNEL
            return parms.shift_W_[idx];
        }
    };

    // 辅助函数：根据粒度配置获取 R 的 shift 值
    auto get_R_shift = [&](int idx) -> int8_t {
        if (cfg.R_granularity_ == OperatorQuantConfig::PER_TENSOR) {
            return parms.shift_R_tensor_;
        } else if (cfg.R_granularity_ == OperatorQuantConfig::PER_GATE) {
            int gate_idx = idx / hidden_size;
            return parms.shift_R_gate_[gate_idx];
        } else {  // PER_CHANNEL
            return parms.shift_R_[idx];
        }
    };

    // 辅助函数：根据粒度配置获取 bw 的 shift 值
    auto get_bw_shift = [&](int idx) -> int8_t {
        if (cfg.bw_granularity_ == OperatorQuantConfig::PER_TENSOR) {
            return parms.shift_bw_tensor_;
        } else if (cfg.bw_granularity_ == OperatorQuantConfig::PER_GATE) {
            int gate_idx = idx / hidden_size;
            return parms.shift_bw_gate_[gate_idx];
        } else {  // PER_CHANNEL
            return parms.shift_bw_[idx];
        }
    };

    // 辅助函数：根据粒度配置获取 br 的 shift 值
    auto get_br_shift = [&](int idx) -> int8_t {
        if (cfg.br_granularity_ == OperatorQuantConfig::PER_TENSOR) {
            return parms.shift_br_tensor_;
        } else if (cfg.br_granularity_ == OperatorQuantConfig::PER_GATE) {
            int gate_idx = idx / hidden_size;
            return parms.shift_br_gate_[gate_idx];
        } else {  // PER_CHANNEL
            return parms.shift_br_[idx];
        }
    };

    for (int idx = 0; idx < channel; ++idx) {
        int8_t shift_W = get_W_shift(idx);
        int8_t shift_R = get_R_shift(idx);
        int8_t shift_bw_val = get_bw_shift(idx);
        int8_t shift_br_val = get_br_shift(idx);
        
        shift_gemm_x[idx] = (shift_W + parms.shift_x_) - parms.shift_weight_ih_linear_;
        shift_gemm_h[idx] = (shift_R + parms.shift_h_) - parms.shift_weight_hh_linear_;
        // bias 先移位到 GEMM scale，再和 GEMM 结果一起移位到 Linear scale
        // shift_bw_to_gemm = shift_bw - (shift_W + shift_x)
        // 如果 shift_bw = shift_W + shift_x，则 shift_bw_to_gemm = 0（不需要移位）
        shift_bw[idx] = shift_bw_val - (shift_W + parms.shift_x_);
        shift_br[idx] = shift_br_val - (shift_R + parms.shift_h_);
    }

    // 存储 per-channel 数组（PER_CHANNEL 粒度时使用，其他粒度时作为占位符）
    linear_params_.shift_gemm_x_to_weight_ih_linear_ = dev::vector<int8_t>(shift_gemm_x);
    linear_params_.shift_bw_to_weight_ih_linear_ = dev::vector<int8_t>(shift_bw);
    linear_params_.shift_gemm_h_to_weight_hh_linear_ = dev::vector<int8_t>(shift_gemm_h);
    linear_params_.shift_br_to_weight_hh_linear_ = dev::vector<int8_t>(shift_br);

#ifdef DEBUG
    linear_params_.shift_bw_ = dev::vector<int8_t>(parms.shift_bw_);
    linear_params_.shift_br_ = dev::vector<int8_t>(parms.shift_br_);
#endif

    // ==================== 门计算参数（标量）====================
    gate_params_.zp_weight_ih_linear_ = parms.zp_weight_ih_linear_;
    gate_params_.zp_weight_hh_linear_ = parms.zp_weight_hh_linear_;
    gate_params_.zp_h_ = parms.zp_h_;

    // update gate
    gate_params_.zp_update_gate_input_ = parms.zp_update_gate_input_;
    gate_params_.zp_update_gate_output_ = parms.zp_update_gate_output_;
    gate_params_.shift_weight_ih_linear_to_update_gate_input_ = parms.shift_weight_ih_linear_ - parms.shift_update_gate_input_;
    gate_params_.shift_weight_hh_linear_to_update_gate_input_ = parms.shift_weight_hh_linear_ - parms.shift_update_gate_input_;

    // reset gate
    gate_params_.zp_reset_gate_input_ = parms.zp_reset_gate_input_;
    gate_params_.zp_reset_gate_output_ = parms.zp_reset_gate_output_;
    gate_params_.shift_weight_ih_linear_to_reset_gate_input_ = parms.shift_weight_ih_linear_ - parms.shift_reset_gate_input_;
    gate_params_.shift_weight_hh_linear_to_reset_gate_input_ = parms.shift_weight_hh_linear_ - parms.shift_reset_gate_input_;

    // new gate（乘法scale融合：r*weight_hh_linear 直接对齐到 new_gate_input）
    gate_params_.zp_new_gate_input_ = parms.zp_new_gate_input_;
    gate_params_.zp_new_gate_output_ = parms.zp_new_gate_output_;
    gate_params_.shift_weight_ih_linear_to_new_gate_input_ = parms.shift_weight_ih_linear_ - parms.shift_new_gate_input_;
    gate_params_.shift_reset_mul_hh_to_new_gate_input_ =
        (parms.shift_reset_gate_output_ + parms.shift_weight_hh_linear_) - parms.shift_new_gate_input_;

    // h_new（统一scale空间优化：先将new_gate对齐到h，然后在统一scale下计算和相加）
    gate_params_.quant_one_in_update_gate_scale_ = rshift_round(1, -parms.shift_update_gate_output_) + parms.zp_update_gate_output_;
    // new_gate_output 对齐到 h 的移位
    gate_params_.shift_new_gate_output_to_h_ = parms.shift_new_gate_output_ - parms.shift_h_;
    // 统一scale到h的移位（= shift_update_gate_output，因为 scale_h / scale_h = 1）
    gate_params_.shift_update_old_to_h_ = parms.shift_update_gate_output_;

    // 位宽配置和 LUT
    gate_params_.bitwidth_config_ = parms.bitwidth_config_;
    gate_params_.sigmoid_update_gate_lut_ = parms.sigmoid_update_gate_lut_;
    gate_params_.sigmoid_reset_gate_lut_ = parms.sigmoid_reset_gate_lut_;
    gate_params_.tanh_new_gate_lut_ = parms.tanh_new_gate_lut_;

#ifdef DEBUG
    gate_params_.test = parms;
#endif
}

void ForwardPassQuant::Run(
    const int steps,              // 时间步数, 序列长度T
    const int32_t *W,             // [C,H*3], 输入到隐藏状态的权重矩阵（int32_t 存储）
    const int32_t *R,             // [H,H*3], 隐状态到隐藏状态的权重矩阵（int32_t 存储）
    const int32_t *bw,            // [H*3], 输入偏置
    const int32_t *br,            // [H*3], 隐状态偏置
    const int32_t *x,             // [N*T,C], 输入序列（int32_t 存储）
    int32_t *h,                   // [(T+1)*N,H], 输出隐藏状态（int32_t 存储）
    int32_t *v,                   // [T*N,H*4], 中间激活值（训练模式需要）
    const float zoneout_prob,     // Zoneout 概率
    const int32_t *zoneout_mask,  // Zoneout mask [T*N,H]（int32_t 存储）
    uint8_t *weight_ih_linear_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask
) {
    // 量化模式：禁用 TensorCore 以提高精度（与浮点模式保持一致）
    // TensorCore 使用 TF32 精度，可能导致精度问题
    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);  // 注释掉以禁用 TensorCore
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

    // 计算输入 Linear 变换（所有时间步一次性计算，结果存入 tmp_weight_ih_linear_，可选 mask）
    ComputeLinearX(W, x, bw, steps, weight_ih_linear_mask);

    // 同步 Linear 计算
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, br,
                        h + i * NH,                      // 输入 h
                        h + (i + 1) * NH,                // 输出 h
                        v + i * NH * 4,                  // 中间激活
                        tmp_weight_ih_linear_.data() + i * NH3,  // 当前时间步的 W*x + bw
                        zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr,
                        weight_hh_linear_mask ? weight_hh_linear_mask + i * NH3 : nullptr,
                        gate_input_mask ? gate_input_mask + i * NH3 : nullptr,
                        gate_output_mask ? gate_output_mask + i * NH3 : nullptr,
                        h_mask ? h_mask + i * NH : nullptr);
    }

    cublasSetStream(data_->blas_handle, save_stream);
}

}  // namespace gru
