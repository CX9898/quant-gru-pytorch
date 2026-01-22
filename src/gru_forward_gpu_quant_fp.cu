// ============================================================================
// gru_forward_gpu_quant_fp.cu - 浮点存储版量化 GRU 前向传播 CUDA 实现
// ============================================================================
//
// 文件结构:
//   1. 辅助函数          - div_round, clamp_f 等
//   2. 激活函数          - real_sigmoid_f, real_tanh_f
//   3. 门计算函数        - computeUpdateGateFP, computeResetGateFP 等
//   4. Bias+Rescale Kernel - GEMM 后处理
//   5. Pointwise Kernel  - GRU 逐点运算
//   6. ForwardPassQuantFP - 前向传播封装类
//
// 与 gru_forward_gpu_quant.cu 的区别:
//   - 所有量化值使用 float 存储（值仍是定点整数）
//   - 使用 cuBLAS SGEMM + 单独的 bias/rescale kernel
//   - 只使用 real_sigmoid/real_tanh，不用 LUT
//   - shift 预处理为除数，避免运行时位移
//
// ============================================================================

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <vector>

#include "blas.h"
#include "dev_vector.h"
#include "gru_quant.h"

namespace kernel {

// ============================================================================
// 1. 辅助函数
// ============================================================================

/**
 * @brief 带四舍五入的除法（替代 rshift_round）
 * 
 * result = round(x / divisor)
 * divisor 已预计算为 2^shift
 */
__device__ __forceinline__ 
float div_round(float x, float divisor) {
    return roundf(x / divisor);
}

/**
 * @brief 带四舍五入的除法（double 精度版本，用于乘法累加）
 */
__device__ __forceinline__ 
double div_round_d(double x, double divisor) {
    return round(x / divisor);
}

/**
 * @brief 浮点版 clamp（值是定点整数，用 float 存储）
 */
__device__ __forceinline__ 
float clamp_f(float val, QuantBitWidth bw) {
    float lo = static_cast<float>(bw.qmin());
    float hi = static_cast<float>(bw.qmax());
    return fmaxf(lo, fminf(val, hi));
}

// ============================================================================
// 2. 激活函数
// ============================================================================

/**
 * @brief 真实 Sigmoid 函数（反量化 → sigmoid → 量化）
 * 
 * @param q_x 量化输入
 * @param scale_x 输入反量化 scale = 2^(-shift_x)
 * @param zp_x 输入零点
 * @param scale_y 输出量化 scale = 2^(-shift_y)
 * @param zp_y 输出零点
 * @param out_bw 输出位宽配置
 */
__device__ __forceinline__ 
float real_sigmoid_f(float q_x, float scale_x, float zp_x,
                     float scale_y, float zp_y, QuantBitWidth out_bw) {
    // 反量化: x_fp = (q_x - zp_x) * scale_x
    float x_fp = (q_x - zp_x) * scale_x;
    // sigmoid: y_fp = 1 / (1 + exp(-x_fp))
    float y_fp = 1.0f / (1.0f + expf(-x_fp));
    // 量化: q_y = round(y_fp / scale_y + zp_y)
    float q_y = roundf(y_fp / scale_y + zp_y);
    return clamp_f(q_y, out_bw);
}

/**
 * @brief 真实 Tanh 函数（反量化 → tanh → 量化）
 */
__device__ __forceinline__ 
float real_tanh_f(float q_x, float scale_x, float zp_x,
                  float scale_y, float zp_y, QuantBitWidth out_bw) {
    float x_fp = (q_x - zp_x) * scale_x;
    float y_fp = tanhf(x_fp);
    float q_y = roundf(y_fp / scale_y + zp_y);
    return clamp_f(q_y, out_bw);
}

// ============================================================================
// 3. 门计算函数
// ============================================================================

/**
 * @brief 计算更新门 update_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 */
__device__ __forceinline__ 
float computeUpdateGateFP(float weight_ih_linear, float weight_hh_linear, 
                          const GateQuantParamsFP &p) {
    // 重缩放到 update_gate_input 空间（除法替代位移）
    float ih = div_round(weight_ih_linear - p.zp_weight_ih_linear_, 
                         p.div_weight_ih_linear_to_update_gate_input_);
    float hh = div_round(weight_hh_linear - p.zp_weight_hh_linear_, 
                         p.div_weight_hh_linear_to_update_gate_input_);
    float input = ih + hh + p.zp_update_gate_input_;
    
    // 使用 real_sigmoid
    return real_sigmoid_f(input, 
                          p.scale_update_gate_input_, p.zp_update_gate_input_,
                          p.scale_update_gate_output_, p.zp_update_gate_output_,
                          p.bitwidth_config_.update_gate_output_);
}

/**
 * @brief 计算重置门 reset_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 */
__device__ __forceinline__ 
float computeResetGateFP(float weight_ih_linear, float weight_hh_linear,
                         const GateQuantParamsFP &p) {
    float ih = div_round(weight_ih_linear - p.zp_weight_ih_linear_,
                         p.div_weight_ih_linear_to_reset_gate_input_);
    float hh = div_round(weight_hh_linear - p.zp_weight_hh_linear_,
                         p.div_weight_hh_linear_to_reset_gate_input_);
    float input = ih + hh + p.zp_reset_gate_input_;
    
    return real_sigmoid_f(input,
                          p.scale_reset_gate_input_, p.zp_reset_gate_input_,
                          p.scale_reset_gate_output_, p.zp_reset_gate_output_,
                          p.bitwidth_config_.reset_gate_output_);
}

/**
 * @brief 计算候选门 new_gate = tanh(weight_ih_linear + reset_gate * weight_hh_linear)
 * 
 * @param weight_hh_linear_g [out] 中间结果，用于存储到 v（训练时反向传播需要）
 */
__device__ __forceinline__ 
float computeNewGateFP(float weight_ih_linear, float weight_hh_linear, float reset_gate,
                       const GateQuantParamsFP &p, float &weight_hh_linear_g) {
    weight_hh_linear_g = weight_hh_linear;
    
    // 计算 reset_gate * weight_hh_linear，直接对齐到 new_gate_input
    double r_diff = static_cast<double>(reset_gate) - static_cast<double>(p.zp_reset_gate_output_);
    double hh_diff = static_cast<double>(weight_hh_linear_g) - static_cast<double>(p.zp_weight_hh_linear_);
    double reset_hidden_mul = r_diff * hh_diff;
    float rh = static_cast<float>(div_round_d(reset_hidden_mul, 
                                              static_cast<double>(p.div_reset_mul_hh_to_new_gate_input_)));
    
    // weight_ih_linear 重缩放到 new_gate_input 空间
    float ih = div_round(weight_ih_linear - p.zp_weight_ih_linear_,
                         p.div_weight_ih_linear_to_new_gate_input_);
    float input = ih + rh + p.zp_new_gate_input_;
    
    return real_tanh_f(input,
                       p.scale_new_gate_input_, p.zp_new_gate_input_,
                       p.scale_new_gate_output_, p.zp_new_gate_output_,
                       p.bitwidth_config_.new_gate_output_);
}

/**
 * @brief 计算隐藏状态 h_new = update_gate * h_old + (1 - update_gate) * new_gate
 */
__device__ __forceinline__ 
float computeHiddenStateFP(float update_gate, float new_gate, float h_old,
                           const GateQuantParamsFP &p) {
    // 计算 update_gate * h_old，直接对齐到 h
    double u_diff = static_cast<double>(update_gate) - static_cast<double>(p.zp_update_gate_output_);
    double h_diff = static_cast<double>(h_old) - static_cast<double>(p.zp_h_);
    double old_contribution_mul = u_diff * h_diff;
    float old_term = static_cast<float>(div_round_d(old_contribution_mul, 
                                                    static_cast<double>(p.div_update_old_to_h_)));
    
    // 计算 (1 - update_gate) * new_gate，直接对齐到 h
    double one_minus_u = static_cast<double>(p.quant_one_in_update_gate_scale_) - 
                         static_cast<double>(update_gate);
    double n_diff = static_cast<double>(new_gate) - static_cast<double>(p.zp_new_gate_output_);
    double new_contribution_mul = one_minus_u * n_diff;
    float new_term = static_cast<float>(div_round_d(new_contribution_mul, 
                                                    static_cast<double>(p.div_update_new_to_h_)));
    
    float h_new = old_term + new_term + p.zp_h_;
    return clamp_f(h_new, p.bitwidth_config_.h_);
}

// ============================================================================
// 4. 自定义 Float GEMM（与 INT32 版本相同逻辑，用于验证）
// ============================================================================

// 启用自定义 GEMM 的编译开关（设为 1 使用自定义 GEMM，0 使用 cuBLAS）
#define USE_CUSTOM_FLOAT_GEMM 1

constexpr int TILE_SIZE_FP = 16;

/**
 * @brief 自定义 Float GEMM + Bias + Rescale（与 INT32 版本完全对齐）
 * 
 * 计算: C = clamp(round((A * (B - zp_B) + bias) / div_gemm) + zp_out)
 * 
 * 关键特点：
 * - 在加载 B tile 时就减去零点（与 INT32 版本一致）
 * - 使用 double 累加确保精度
 * - 与 INT32 版本的 rshift_round 行为一致
 * 
 * @param A [M, K] 权重，列主序
 * @param B [K, N] 输入，列主序
 * @param C [M, N] 输出，列主序
 * @param bias [M] 偏置
 * @param M 输出行数 (hidden*3)
 * @param N 输出列数 (batch)
 * @param K 内部维度 (hidden)
 * @param zp_B 输入零点
 * @param div_gemm [M] per-row GEMM 除数
 * @param div_bias [M] per-row bias 除数
 * @param zp_out 输出零点
 * @param output_bw 输出位宽配置
 */
__global__ void customFloatGemmBiasFused(
    const float *__restrict__ A,               // [M, K] 权重，列主序
    const float *__restrict__ B,               // [K, N] 输入，列主序
    float *__restrict__ C,                     // [M, N] 输出，列主序
    const float *__restrict__ bias,            // [M] 偏置
    int M, int N, int K,
    float zp_B,                                // 输入的 zero-point
    const float *__restrict__ div_gemm,        // [M] per-row GEMM 除数
    const float *__restrict__ div_bias,        // [M] per-row bias 除数
    float zp_out,                              // 输出的 zero-point
    QuantBitWidth output_bw                    // 输出位宽配置
) {
    __shared__ float As[TILE_SIZE_FP][TILE_SIZE_FP + 1];
    __shared__ float Bs[TILE_SIZE_FP][TILE_SIZE_FP + 1];

    const int row = blockIdx.y * TILE_SIZE_FP + threadIdx.y;  // m in [0, M)
    const int col = blockIdx.x * TILE_SIZE_FP + threadIdx.x;  // n in [0, N)

    // 使用 double 累加以匹配 INT32 版本的 int64 精度
    double acc = 0.0;

    const int numTiles = (K + TILE_SIZE_FP - 1) / TILE_SIZE_FP;

    for (int t = 0; t < numTiles; t++) {
        // 加载 A tile（列主序：A[k*M + m]）
        const int aK = t * TILE_SIZE_FP + threadIdx.x;
        if (row < M && aK < K) {
            As[threadIdx.y][threadIdx.x] = A[aK * M + row];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B tile 并减去 zp_B（列主序：B[n*K + k]）
        // 关键：与 INT32 版本一样，在加载时就减零点！
        const int bK = t * TILE_SIZE_FP + threadIdx.y;
        if (col < N && bK < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + bK] - zp_B;
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE_FP; k++) {
            acc += static_cast<double>(As[threadIdx.y][k]) * 
                   static_cast<double>(Bs[k][threadIdx.x]);
        }

        __syncthreads();
    }

    // 写回结果：与 INT32 版本相同的 rescale 逻辑
    if (row < M && col < N) {
        const double div_g = static_cast<double>(div_gemm[row]);
        const double div_b = static_cast<double>(div_bias[row]);
        const double bias_val = static_cast<double>(bias[row]);

        // bias 先 rescale（与 INT32 的 rshift_round 对应）
        double bias_result = round(bias_val / div_b);
        
        // GEMM + bias 一起 rescale
        double gemm_result = round((acc + bias_result) / div_g);

        // 合并结果
        double result = gemm_result + static_cast<double>(zp_out);

        // clamp 并输出（列主序：C[n*M + m]）
        C[col * M + row] = clamp_f(static_cast<float>(result), output_bw);
    }
}

// ============================================================================
// 5. Bias + Rescale Kernel（cuBLAS GEMM 后处理，当不使用自定义 GEMM 时使用）
// ============================================================================

/**
 * @brief GEMM 结果加 bias 并 rescale
 * 
 * out[i] = clamp(round((gemm[i] - W_sum_mul_zp[row]) / div_gemm[row]) 
 *              + round(bias[row] / div_bias[row]) + zp_out)
 * 
 * @param gemm_result GEMM 原始输出 [M, N]（列主序）
 * @param output rescale 后输出 [M, N]
 * @param bias [M] 偏置
 * @param W_sum_mul_zp [M] 预计算的 sum(W)*zp_x（用于零点补偿）
 * @param div_gemm [M] per-row GEMM 除数
 * @param div_bias [M] per-row bias 除数
 * @param zp_out 输出零点
 * @param M 输出通道数（hidden*3）
 * @param N batch * steps
 * @param output_bw 输出位宽配置
 */
__global__ void biasRescaleKernel(
    const float *__restrict__ gemm_result,
    float *__restrict__ output,
    const float *__restrict__ bias,
    const double *__restrict__ W_sum_mul_zp,
    const float *__restrict__ div_gemm,
    const float *__restrict__ div_bias,
    float zp_out,
    int M, int N,
    QuantBitWidth output_bw
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    
    int row = idx % M;  // 列主序
    
    // GEMM 结果减去零点补偿
    double val = static_cast<double>(gemm_result[idx]) - W_sum_mul_zp[row];
    
    // bias 先 rescale
    double bias_term = round(static_cast<double>(bias[row]) / static_cast<double>(div_bias[row]));
    
    // GEMM + bias 一起 rescale
    double result = round((val + bias_term) / static_cast<double>(div_gemm[row])) + 
                    static_cast<double>(zp_out);
    
    output[idx] = clamp_f(static_cast<float>(result), output_bw);
}

/**
 * @brief 计算权重列和乘以零点（浮点版）
 * 
 * W_sum_mul_zp[j] = sum_i(W[i,j]) * zp
 */
__global__ void computeWeightSumKernel(
    const float *__restrict__ W,
    double *__restrict__ W_sum_mul_zp,
    float zp,
    int M, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    double sum = 0.0;
    // W 是列主序 [M, K]，遍历第 row 行的所有 K 个元素
    for (int k = 0; k < K; ++k) {
        sum += static_cast<double>(W[k * M + row]);
    }
    W_sum_mul_zp[row] = sum * static_cast<double>(zp);
}

// ============================================================================
// 5. Pointwise Kernel - GRU 逐点运算
// ============================================================================

/**
 * @brief GRU 逐点运算 Kernel（浮点版）
 * 
 * 每个线程处理一个 (batch, hidden) 位置
 */
template <bool Training, bool ApplyZoneout>
__global__ void PointwiseOperationsFP(
    int batch_dim, int hidden_dim,
    const float *__restrict__ weight_ih_linear,
    const float *__restrict__ weight_hh_linear,
    const float *__restrict__ h,
    float *__restrict__ h_out,
    float *__restrict__ v,
    float zoneout_prob,
    const float *__restrict__ zoneout_mask,
    GateQuantParamsFP gate_params
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim) return;

    const int weight_idx = col * (hidden_dim * 3) + row;
    const int output_idx = col * hidden_dim + row;
    const int update_idx = weight_idx + 0 * hidden_dim;
    const int reset_idx = weight_idx + 1 * hidden_dim;
    const int new_idx = weight_idx + 2 * hidden_dim;

    // 计算更新门
    float update_gate = computeUpdateGateFP(
        weight_ih_linear[update_idx], 
        weight_hh_linear[update_idx], 
        gate_params);

    // 计算重置门
    float reset_gate = computeResetGateFP(
        weight_ih_linear[reset_idx], 
        weight_hh_linear[reset_idx], 
        gate_params);

    // 计算候选门
    float weight_hh_linear_g;
    float new_gate = computeNewGateFP(
        weight_ih_linear[new_idx], 
        weight_hh_linear[new_idx], 
        reset_gate, gate_params, weight_hh_linear_g);

    // Training: 保存中间值
    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = update_gate;
        v[base_v_idx + 1 * hidden_dim] = reset_gate;
        v[base_v_idx + 2 * hidden_dim] = new_gate;
        v[base_v_idx + 3 * hidden_dim] = weight_hh_linear_g;
    }

    // 计算新的隐藏状态
    float cur_h = computeHiddenStateFP(update_gate, new_gate, h[output_idx], gate_params);

    // Zoneout（如果启用）
    if (ApplyZoneout) {
        float mask = zoneout_mask[output_idx];
        cur_h = mask * h[output_idx] + (1.0f - mask) * cur_h;
    }

    h_out[output_idx] = cur_h;
}

}  // namespace kernel

// ============================================================================
// 6. ForwardPassQuantFP - 前向传播封装类
// ============================================================================

namespace gru {

struct ForwardPassQuantFP::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[2];
    cudaEvent_t event;
    cudaStream_t sync_stream;
};

ForwardPassQuantFP::ForwardPassQuantFP(bool training, int batch_size,
                                       int input_size, int hidden_size,
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

ForwardPassQuantFP::~ForwardPassQuantFP() {
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

void ForwardPassQuantFP::setRescaleParam(const GRUQuantParams &src) {
    const int channel = src.hidden_ * 3;

    // ========== 转换 GateQuantParamsFP ==========
    auto &g = gate_params_;
    
    // 零点转换
    g.zp_weight_ih_linear_ = static_cast<float>(src.zp_weight_ih_linear_);
    g.zp_weight_hh_linear_ = static_cast<float>(src.zp_weight_hh_linear_);
    g.zp_h_ = static_cast<float>(src.zp_h_);

    // Update gate
    g.zp_update_gate_input_ = static_cast<float>(src.zp_update_gate_input_);
    g.zp_update_gate_output_ = static_cast<float>(src.zp_update_gate_output_);
    int8_t shift_ih_u = src.shift_weight_ih_linear_ - src.shift_update_gate_input_;
    int8_t shift_hh_u = src.shift_weight_hh_linear_ - src.shift_update_gate_input_;
    g.div_weight_ih_linear_to_update_gate_input_ = ldexpf(1.0f, shift_ih_u);
    g.div_weight_hh_linear_to_update_gate_input_ = ldexpf(1.0f, shift_hh_u);

    // Reset gate
    g.zp_reset_gate_input_ = static_cast<float>(src.zp_reset_gate_input_);
    g.zp_reset_gate_output_ = static_cast<float>(src.zp_reset_gate_output_);
    int8_t shift_ih_r = src.shift_weight_ih_linear_ - src.shift_reset_gate_input_;
    int8_t shift_hh_r = src.shift_weight_hh_linear_ - src.shift_reset_gate_input_;
    g.div_weight_ih_linear_to_reset_gate_input_ = ldexpf(1.0f, shift_ih_r);
    g.div_weight_hh_linear_to_reset_gate_input_ = ldexpf(1.0f, shift_hh_r);

    // New gate
    g.zp_new_gate_input_ = static_cast<float>(src.zp_new_gate_input_);
    g.zp_new_gate_output_ = static_cast<float>(src.zp_new_gate_output_);
    int8_t shift_ih_n = src.shift_weight_ih_linear_ - src.shift_new_gate_input_;
    int8_t shift_rh_n = (src.shift_reset_gate_output_ + src.shift_weight_hh_linear_) - src.shift_new_gate_input_;
    g.div_weight_ih_linear_to_new_gate_input_ = ldexpf(1.0f, shift_ih_n);
    g.div_reset_mul_hh_to_new_gate_input_ = ldexpf(1.0f, shift_rh_n);

    // Hidden state
    // quant_one = rshift_round(1, -shift) + zp = (1 << shift) + zp = 2^shift + zp
    // 注意：rshift_round(1, -n) 当 n>0 时等于 1 << n
    g.quant_one_in_update_gate_scale_ = ldexpf(1.0f, src.shift_update_gate_output_) + 
                                        static_cast<float>(src.zp_update_gate_output_);
    int8_t shift_un = (src.shift_update_gate_output_ + src.shift_new_gate_output_) - src.shift_h_;
    int8_t shift_uh = src.shift_update_gate_output_;  // shift_update_gate_output_ + shift_h_ - shift_h_
    g.div_update_new_to_h_ = ldexpf(1.0f, shift_un);
    g.div_update_old_to_h_ = ldexpf(1.0f, shift_uh);

    // 激活函数 scale = 2^(-shift)
    g.scale_update_gate_input_ = ldexpf(1.0f, -src.shift_update_gate_input_);
    g.scale_update_gate_output_ = ldexpf(1.0f, -src.shift_update_gate_output_);
    g.scale_reset_gate_input_ = ldexpf(1.0f, -src.shift_reset_gate_input_);
    g.scale_reset_gate_output_ = ldexpf(1.0f, -src.shift_reset_gate_output_);
    g.scale_new_gate_input_ = ldexpf(1.0f, -src.shift_new_gate_input_);
    g.scale_new_gate_output_ = ldexpf(1.0f, -src.shift_new_gate_output_);

    g.bitwidth_config_ = src.bitwidth_config_;

    // ========== 转换 LinearQuantParamsGPUFP ==========
    auto &l = linear_params_;
    l.zp_x_ = static_cast<float>(src.zp_x_);
    l.zp_h_ = static_cast<float>(src.zp_h_);
    l.zp_weight_ih_linear_ = static_cast<float>(src.zp_weight_ih_linear_);
    l.zp_weight_hh_linear_ = static_cast<float>(src.zp_weight_hh_linear_);

    std::vector<float> div_gemm_x(channel), div_bw(channel);
    std::vector<float> div_gemm_h(channel), div_br(channel);
    for (int i = 0; i < channel; ++i) {
        // GEMM: scale_W * scale_x -> scale_weight_ih_linear
        // shift = (shift_W + shift_x) - shift_weight_ih_linear
        int8_t shift_gx = (src.shift_W_[i] + src.shift_x_) - src.shift_weight_ih_linear_;
        // bias: 先 shift 到 GEMM scale，再和 GEMM 一起 shift
        int8_t shift_bw = src.shift_bw_[i] - (src.shift_W_[i] + src.shift_x_);
        div_gemm_x[i] = ldexpf(1.0f, shift_gx);
        div_bw[i] = ldexpf(1.0f, shift_bw);

        int8_t shift_gh = (src.shift_R_[i] + src.shift_h_) - src.shift_weight_hh_linear_;
        int8_t shift_br = src.shift_br_[i] - (src.shift_R_[i] + src.shift_h_);
        div_gemm_h[i] = ldexpf(1.0f, shift_gh);
        div_br[i] = ldexpf(1.0f, shift_br);
    }
    l.div_gemm_x_to_weight_ih_linear_ = dev::vector<float>(div_gemm_x);
    l.div_bw_to_weight_ih_linear_ = dev::vector<float>(div_bw);
    l.div_gemm_h_to_weight_hh_linear_ = dev::vector<float>(div_gemm_h);
    l.div_br_to_weight_hh_linear_ = dev::vector<float>(div_br);
    
    l.output_bw_ih_ = src.bitwidth_config_.weight_ih_linear_;
    l.output_bw_hh_ = src.bitwidth_config_.weight_hh_linear_;

    // 重置权重和计算标志
    weight_sums_computed_ = false;
}

void ForwardPassQuantFP::EnsureBuffersAllocated(int steps) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;

    if (steps <= max_steps_) {
        return;
    }

    // GEMM 原始结果（较大的那个）
    size_t gemm_size = std::max(hidden3 * steps * batch_size, 
                                 hidden3 * batch_size);
    tmp_gemm_result_.resize(gemm_size);

    // Linear 变换结果
    tmp_weight_ih_linear_.resize(hidden3 * steps * batch_size);
    tmp_weight_hh_linear_.resize(hidden3 * batch_size);

    // 权重和常量
    if (W_sum_mul_x_zp_.size() == 0) {
        W_sum_mul_x_zp_.resize(hidden3);
        R_sum_mul_h_zp_.resize(hidden3);
    }

    max_steps_ = steps;
    weight_sums_computed_ = false;
}

void ForwardPassQuantFP::PrecomputeWeightSums(const float *W, const float *R) {
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

    // 计算 W_sum * zp_x
    int threads = 256;
    int blocks = (hidden3 + threads - 1) / threads;
    kernel::computeWeightSumKernel<<<blocks, threads, 0, stream>>>(
        W, W_sum_mul_x_zp_.data(), linear_params_.zp_x_, hidden3, input_size);

    // 计算 R_sum * zp_h
    kernel::computeWeightSumKernel<<<blocks, threads, 0, stream>>>(
        R, R_sum_mul_h_zp_.data(), linear_params_.zp_h_, hidden3, hidden_size);

    cudaStreamSynchronize(stream);
    weight_sums_computed_ = true;
}

void ForwardPassQuantFP::ComputeLinearX(const float *W, const float *x, 
                                        const float *bw, int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[1];

    const int M = hidden_size * 3;
    const int N = steps * batch_size;
    const int K = input_size;

#if USE_CUSTOM_FLOAT_GEMM
    // 使用自定义 GEMM（与 INT32 版本相同逻辑）
    dim3 blockDim(kernel::TILE_SIZE_FP, kernel::TILE_SIZE_FP);
    dim3 gridDim((N + kernel::TILE_SIZE_FP - 1) / kernel::TILE_SIZE_FP,
                 (M + kernel::TILE_SIZE_FP - 1) / kernel::TILE_SIZE_FP);
    
    kernel::customFloatGemmBiasFused<<<gridDim, blockDim, 0, stream>>>(
        W, x, tmp_weight_ih_linear_.data(), bw,
        M, N, K,
        linear_params_.zp_x_,
        linear_params_.div_gemm_x_to_weight_ih_linear_.data(),
        linear_params_.div_bw_to_weight_ih_linear_.data(),
        linear_params_.zp_weight_ih_linear_,
        linear_params_.output_bw_ih_);
#else
    // 使用 cuBLAS SGEMM + 后处理
    cublasSetStream(data_->blas_handle, stream);

    // 1. cuBLAS SGEMM: tmp_gemm = W * x
    // W: [M, K] 列主序，x: [K, N] 列主序，输出 [M, N]
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(data_->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &alpha, W, M, x, K, &beta, tmp_gemm_result_.data(), M);

    // 2. Bias + Rescale kernel
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel::biasRescaleKernel<<<blocks, threads, 0, stream>>>(
        tmp_gemm_result_.data(), 
        tmp_weight_ih_linear_.data(),
        bw, 
        W_sum_mul_x_zp_.data(),
        linear_params_.div_gemm_x_to_weight_ih_linear_.data(),
        linear_params_.div_bw_to_weight_ih_linear_.data(),
        linear_params_.zp_weight_ih_linear_,
        M, N, 
        linear_params_.output_bw_ih_);
#endif
}

void ForwardPassQuantFP::ComputeLinearH(const float *R, const float *h, 
                                        const float *br) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[0];

    const int M = hidden_size * 3;
    const int N = batch_size;
    const int K = hidden_size;

#if USE_CUSTOM_FLOAT_GEMM
    // 使用自定义 GEMM（与 INT32 版本相同逻辑）
    dim3 blockDim(kernel::TILE_SIZE_FP, kernel::TILE_SIZE_FP);
    dim3 gridDim((N + kernel::TILE_SIZE_FP - 1) / kernel::TILE_SIZE_FP,
                 (M + kernel::TILE_SIZE_FP - 1) / kernel::TILE_SIZE_FP);
    
    kernel::customFloatGemmBiasFused<<<gridDim, blockDim, 0, stream>>>(
        R, h, tmp_weight_hh_linear_.data(), br,
        M, N, K,
        linear_params_.zp_h_,
        linear_params_.div_gemm_h_to_weight_hh_linear_.data(),
        linear_params_.div_br_to_weight_hh_linear_.data(),
        linear_params_.zp_weight_hh_linear_,
        linear_params_.output_bw_hh_);
#else
    // 使用 cuBLAS SGEMM + 后处理
    cublasSetStream(data_->blas_handle, stream);

    // 1. cuBLAS SGEMM: tmp_gemm = R * h
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(data_->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &alpha, R, M, h, K, &beta, tmp_gemm_result_.data(), M);

    // 2. Bias + Rescale kernel
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel::biasRescaleKernel<<<blocks, threads, 0, stream>>>(
        tmp_gemm_result_.data(), 
        tmp_weight_hh_linear_.data(),
        br, 
        R_sum_mul_h_zp_.data(),
        linear_params_.div_gemm_h_to_weight_hh_linear_.data(),
        linear_params_.div_br_to_weight_hh_linear_.data(),
        linear_params_.zp_weight_hh_linear_,
        M, N, 
        linear_params_.output_bw_hh_);
#endif
}

void ForwardPassQuantFP::IterateInternal(
    const float *R,
    const float *br,
    const float *h,
    float *h_out,
    float *v,
    const float *cur_weight_ih_linear,
    float zoneout_prob,
    const float *zoneout_mask
) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    cublasSetStream(data_->blas_handle, stream1);

    // 计算隐状态 Linear 变换: R*h + br
    ComputeLinearH(R, h, br);

    // Pointwise kernel 配置
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    // 启动 GRU pointwise kernel
    if (training) {
        if (zoneout_prob > 0.0f && zoneout_mask) {
            kernel::PointwiseOperationsFP<true, true>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size, 
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_);
        } else {
            kernel::PointwiseOperationsFP<true, false>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size, 
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    h, h_out, v, 0.0f, nullptr, gate_params_);
        }
    } else {
        if (zoneout_prob > 0.0f && zoneout_mask) {
            kernel::PointwiseOperationsFP<false, true>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size, 
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    h, h_out, nullptr, zoneout_prob, zoneout_mask, gate_params_);
        } else {
            kernel::PointwiseOperationsFP<false, false>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size, 
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    h, h_out, nullptr, 0.0f, nullptr, gate_params_);
        }
    }
}

void ForwardPassQuantFP::Run(
    int steps,
    const float *W,
    const float *R,
    const float *bw,
    const float *br,
    const float *x,
    float *h,
    float *v,
    float zoneout_prob,
    const float *zoneout_mask
) {
    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    // 预分配缓冲区
    EnsureBuffersAllocated(steps);

    // 预计算权重和
    PrecomputeWeightSums(W, R);

    cudaStream_t save_stream;
    cublasGetStream(data_->blas_handle, &save_stream);

    cublasSetStream(data_->blas_handle, stream2);

    // 计算输入 Linear 变换（所有时间步一次性计算）
    ComputeLinearX(W, x, bw, steps);

    // 同步 Linear 计算
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    // 时间步循环
    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, br,
                        h + i * NH,                           // 输入 h
                        h + (i + 1) * NH,                     // 输出 h
                        v ? v + i * NH * 4 : nullptr,         // 中间激活
                        tmp_weight_ih_linear_.data() + i * NH3,  // 当前时间步的 W*x + bw
                        zoneout_prob, 
                        zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }

    cublasSetStream(data_->blas_handle, save_stream);
}

}  // namespace gru
