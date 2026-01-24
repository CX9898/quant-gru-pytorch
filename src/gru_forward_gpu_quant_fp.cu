// ============================================================================
// gru_forward_gpu_quant_fp.cu - 浮点存储版量化 GRU 前向传播 CUDA 实现
// ============================================================================
//
// 文件结构:
//   1. 辅助函数          - div_round 等（本地使用）
//   2. 门计算函数        - computeUpdateGateFP, computeResetGateFP 等
//   3. Bias+Rescale Kernel - cuBLAS GEMM 后处理
//   4. Pointwise Kernel  - GRU 逐点运算
//   5. ForwardPassQuantFP - 前向传播封装类
//
// 通用函数在 quantize_ops_helper.h:
//   - clamp_f, quantize_f, dequantize_f
//   - real_sigmoid_f, real_tanh_f
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
#include "parallel_algorithm.h"

namespace kernel {

// ============================================================================
// 1. 辅助函数（本地使用，不通用）
// ============================================================================

/**
 * @brief 带四舍五入的除法（替代 rshift_round）
 * 
 * result = round(x / divisor)
 * divisor 已预计算为 2^shift
 */
__device__ __forceinline__ 
float div_round(float x, float divisor) {
    return round_f(x / divisor);
}

// 注：clamp_f, real_sigmoid_f, real_tanh_f 已移至 quantize_ops_helper.h

// ============================================================================
// 2. 门计算函数
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
 * @brief 计算更新门（带输入和输出 mask 分离输出，用于 QAT）
 * 
 * @param input_was_clamped [out] 门输入是否被截断
 * @param output_was_clamped [out] 门输出是否被截断
 */
__device__ __forceinline__ 
float computeUpdateGateFP_with_mask(float weight_ih_linear, float weight_hh_linear, 
                                    const GateQuantParamsFP &p, 
                                    uint8_t& input_was_clamped,
                                    uint8_t& output_was_clamped) {
    float ih = div_round(weight_ih_linear - p.zp_weight_ih_linear_, 
                         p.div_weight_ih_linear_to_update_gate_input_);
    float hh = div_round(weight_hh_linear - p.zp_weight_hh_linear_, 
                         p.div_weight_hh_linear_to_update_gate_input_);
    float input = ih + hh + p.zp_update_gate_input_;
    
    // 对门输入进行位宽截断并记录 mask
    float clamped_input = clamp_f_with_mask(input, p.bitwidth_config_.update_gate_input_, input_was_clamped);
    
    // 使用截断后的输入计算激活函数
    return real_sigmoid_f_with_mask(clamped_input, 
                          p.scale_update_gate_input_, p.zp_update_gate_input_,
                          p.scale_update_gate_output_, p.zp_update_gate_output_,
                          p.bitwidth_config_.update_gate_output_, output_was_clamped);
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
 * @brief 计算重置门（带输入和输出 mask 分离输出，用于 QAT）
 */
__device__ __forceinline__ 
float computeResetGateFP_with_mask(float weight_ih_linear, float weight_hh_linear,
                                   const GateQuantParamsFP &p, 
                                   uint8_t& input_was_clamped,
                                   uint8_t& output_was_clamped) {
    float ih = div_round(weight_ih_linear - p.zp_weight_ih_linear_,
                         p.div_weight_ih_linear_to_reset_gate_input_);
    float hh = div_round(weight_hh_linear - p.zp_weight_hh_linear_,
                         p.div_weight_hh_linear_to_reset_gate_input_);
    float input = ih + hh + p.zp_reset_gate_input_;
    
    // 对门输入进行位宽截断并记录 mask
    float clamped_input = clamp_f_with_mask(input, p.bitwidth_config_.reset_gate_input_, input_was_clamped);
    
    return real_sigmoid_f_with_mask(clamped_input,
                          p.scale_reset_gate_input_, p.zp_reset_gate_input_,
                          p.scale_reset_gate_output_, p.zp_reset_gate_output_,
                          p.bitwidth_config_.reset_gate_output_, output_was_clamped);
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
    // 使用 float 精度（单个乘积最大 4,161,409，在 FP32 精确范围内）
    float r_diff = reset_gate - p.zp_reset_gate_output_;
    float hh_diff = weight_hh_linear_g - p.zp_weight_hh_linear_;
    float reset_hidden_mul = r_diff * hh_diff;
    float rh = div_round(reset_hidden_mul, p.div_reset_mul_hh_to_new_gate_input_);
    
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
 * @brief 计算候选门（带输入和输出 mask 分离输出，用于 QAT）
 */
__device__ __forceinline__ 
float computeNewGateFP_with_mask(float weight_ih_linear, float weight_hh_linear, float reset_gate,
                                 const GateQuantParamsFP &p, float &weight_hh_linear_g,
                                 uint8_t& input_was_clamped,
                                 uint8_t& output_was_clamped) {
    weight_hh_linear_g = weight_hh_linear;
    
    // 使用 float 精度（单个乘积最大 4,161,409，在 FP32 精确范围内）
    float r_diff = reset_gate - p.zp_reset_gate_output_;
    float hh_diff = weight_hh_linear_g - p.zp_weight_hh_linear_;
    float reset_hidden_mul = r_diff * hh_diff;
    float rh = div_round(reset_hidden_mul, p.div_reset_mul_hh_to_new_gate_input_);
    
    float ih = div_round(weight_ih_linear - p.zp_weight_ih_linear_,
                         p.div_weight_ih_linear_to_new_gate_input_);
    float input = ih + rh + p.zp_new_gate_input_;
    
    // 对门输入进行位宽截断并记录 mask
    float clamped_input = clamp_f_with_mask(input, p.bitwidth_config_.new_gate_input_, input_was_clamped);
    
    return real_tanh_f_with_mask(clamped_input,
                       p.scale_new_gate_input_, p.zp_new_gate_input_,
                       p.scale_new_gate_output_, p.zp_new_gate_output_,
                       p.bitwidth_config_.new_gate_output_, output_was_clamped);
}

/**
 * @brief 计算隐藏状态 h_new = update_gate * h_old + (1 - update_gate) * new_gate
 */
__device__ __forceinline__ 
float computeHiddenStateFP(float update_gate, float new_gate, float h_old,
                           const GateQuantParamsFP &p) {
    // 计算 update_gate * h_old，直接对齐到 h
    // 使用 float 精度（单个乘积最大 4,161,409，在 FP32 精确范围内）
    float u_diff = update_gate - p.zp_update_gate_output_;
    float h_diff = h_old - p.zp_h_;
    float old_contribution_mul = u_diff * h_diff;
    float old_term = div_round(old_contribution_mul, p.div_update_old_to_h_);
    
    // 计算 (1 - update_gate) * new_gate，直接对齐到 h
    // quant_one = 2^shift + zp，是常数 1 在 update_gate_output 量化空间的完整表示
    // one_minus_u = quant_one - update_gate = (2^shift + zp) - update_gate
    float one_minus_u = p.quant_one_in_update_gate_scale_ - update_gate;
    float n_diff = new_gate - p.zp_new_gate_output_;
    float new_contribution_mul = one_minus_u * n_diff;
    float new_term = div_round(new_contribution_mul, p.div_update_new_to_h_);
    
    float h_new = old_term + new_term + p.zp_h_;
    return clamp_f(h_new, p.bitwidth_config_.h_);
}

/**
 * @brief 计算隐藏状态（带 mask 输出，用于 QAT）
 */
__device__ __forceinline__ 
float computeHiddenStateFP_with_mask(float update_gate, float new_gate, float h_old,
                                     const GateQuantParamsFP &p, uint8_t& was_clamped) {
    // 计算 update_gate * h_old，直接对齐到 h
    // 使用 float 精度（单个乘积最大 4,161,409，在 FP32 精确范围内）
    float u_diff = update_gate - p.zp_update_gate_output_;
    float h_diff = h_old - p.zp_h_;
    float old_contribution_mul = u_diff * h_diff;
    float old_term = div_round(old_contribution_mul, p.div_update_old_to_h_);
    
    // 计算 (1 - update_gate) * new_gate，直接对齐到 h
    // quant_one = 2^shift + zp，是常数 1 在 update_gate_output 量化空间的完整表示
    // one_minus_u = quant_one - update_gate = (2^shift + zp) - update_gate
    float one_minus_u = p.quant_one_in_update_gate_scale_ - update_gate;
    float n_diff = new_gate - p.zp_new_gate_output_;
    float new_contribution_mul = one_minus_u * n_diff;
    float new_term = div_round(new_contribution_mul, p.div_update_new_to_h_);
    
    float h_new = old_term + new_term + p.zp_h_;
    return clamp_f_with_mask(h_new, p.bitwidth_config_.h_, was_clamped);
}

// ============================================================================
// 4. Bias + Rescale Kernel（cuBLAS GEMM 后处理）
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
    const float *__restrict__ W_sum_mul_zp,
    const float *__restrict__ div_gemm,
    const float *__restrict__ div_bias,
    float zp_out,
    int M, int N,
    QuantBitWidth output_bw
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    
    int row = idx % M;  // 列主序
    
    // GEMM 结果减去零点补偿（W_sum_mul_zp 是 float，权重总是 8bit）
    float val = gemm_result[idx] - W_sum_mul_zp[row];
    
    // bias 先 rescale（bias 是 8bit/16bit 整数，可以用 float 精确表示）
    float bias_term = round_f(bias[row] / div_bias[row]);
    
    // GEMM + bias 一起 rescale（最终结果在量化范围内）
    float result = round_f((val + bias_term) / div_gemm[row]) + zp_out;
    
    output[idx] = clamp_f(result, output_bw);
}

/**
 * @brief GEMM 结果加 bias 并 rescale（原地处理，带可选 mask 输出版本）
 * @tparam Training 训练模式时保存 clamp mask
 */
template <bool Training>
__global__ void biasRescaleKernelWithMask(
    float *__restrict__ data,  // 输入：GEMM 结果，输出：rescale 后的结果（原地处理）
    const float *__restrict__ bias,
    const float *__restrict__ W_sum_mul_zp,
    const float *__restrict__ div_gemm,
    const float *__restrict__ div_bias,
    float zp_out,
    int M, int N,
    QuantBitWidth output_bw,
    uint8_t *__restrict__ mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    
    int row = idx % M;
    
    // 读取 GEMM 结果，减去零点补偿（W_sum_mul_zp 是 float，权重总是 8bit）
    float val = data[idx] - W_sum_mul_zp[row];
    // bias 先 rescale（bias 是 8bit/16bit 整数，可以用 float 精确表示）
    float bias_term = round_f(bias[row] / div_bias[row]);
    // GEMM + bias 一起 rescale（最终结果在量化范围内）
    float result = round_f((val + bias_term) / div_gemm[row]) + zp_out;
    
    // 原地写回结果
    if constexpr (Training) {
        uint8_t was_clamped;
        data[idx] = clamp_f_with_mask(static_cast<float>(result), output_bw, was_clamped);
        mask[idx] = was_clamped;
    } else {
        data[idx] = clamp_f(static_cast<float>(result), output_bw);
    }
}

/**
 * @brief 计算权重列和乘以零点（浮点版）
 * 
 * W_sum_mul_zp[j] = sum_i(W[i,j]) * zp
 */
__global__ void computeWeightSumKernel(
    const float *__restrict__ W,
    float *__restrict__ W_sum_mul_zp,
    float zp,
    int M, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    // 使用 float 累加（权重总是 8bit，K < 1024 且 zp < 100 时 float 足够精确）
    float sum = 0.0f;
    // W 是列主序 [M, K]，遍历第 row 行的所有 K 个元素
    for (int k = 0; k < K; ++k) {
        sum += W[k * M + row];
    }
    W_sum_mul_zp[row] = sum * zp;
}

// ============================================================================
// 5. Pointwise Kernel - GRU 逐点运算
// ============================================================================

/**
 * @brief GRU 逐点运算 Kernel（浮点版）
 * 
 * 每个线程处理一个 (batch, hidden) 位置
 * 
 * @tparam Training 是否训练模式（保存中间值到 v，保存 clamp mask 用于 QAT）
 * @tparam ApplyZoneout 是否应用 Zoneout
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
    GateQuantParamsFP gate_params,
    // Mask 输出（Training=true 时使用）
    uint8_t *__restrict__ gate_input_mask,   // [batch, hidden*3] 门输入 clamp mask（新增）
    uint8_t *__restrict__ gate_output_mask,  // [batch, hidden*3] 门输出 clamp mask（原 gate_mask）
    uint8_t *__restrict__ h_mask             // [batch, hidden] 隐状态输出 mask
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim) return;

    const int weight_idx = col * (hidden_dim * 3) + row;
    const int output_idx = col * hidden_dim + row;
    const int update_idx = weight_idx + 0 * hidden_dim;
    const int reset_idx = weight_idx + 1 * hidden_dim;
    const int new_idx = weight_idx + 2 * hidden_dim;

    float update_gate, reset_gate, new_gate, cur_h;
    float weight_hh_linear_g;

    if constexpr (Training) {
        // 训练模式：带输入和输出 mask 分离的版本
        uint8_t update_input_clamped, update_output_clamped;
        uint8_t reset_input_clamped, reset_output_clamped;
        uint8_t new_input_clamped, new_output_clamped;
        uint8_t h_clamped;
        
        update_gate = computeUpdateGateFP_with_mask(
            weight_ih_linear[update_idx], 
            weight_hh_linear[update_idx], 
            gate_params, update_input_clamped, update_output_clamped);

        reset_gate = computeResetGateFP_with_mask(
            weight_ih_linear[reset_idx], 
            weight_hh_linear[reset_idx], 
            gate_params, reset_input_clamped, reset_output_clamped);

        new_gate = computeNewGateFP_with_mask(
            weight_ih_linear[new_idx], 
            weight_hh_linear[new_idx], 
            reset_gate, gate_params, weight_hh_linear_g, 
            new_input_clamped, new_output_clamped);

        cur_h = computeHiddenStateFP_with_mask(
            update_gate, new_gate, h[output_idx], gate_params, h_clamped);
        
        // 保存门输入 mask（新增）
        gate_input_mask[update_idx] = update_input_clamped;
        gate_input_mask[reset_idx] = reset_input_clamped;
        gate_input_mask[new_idx] = new_input_clamped;
        
        // 保存门输出 mask（原 gate_mask）
        gate_output_mask[update_idx] = update_output_clamped;
        gate_output_mask[reset_idx] = reset_output_clamped;
        gate_output_mask[new_idx] = new_output_clamped;
        
        h_mask[output_idx] = h_clamped;
    } else {
        // 不保存 mask 的版本
        update_gate = computeUpdateGateFP(
            weight_ih_linear[update_idx], 
            weight_hh_linear[update_idx], 
            gate_params);

        reset_gate = computeResetGateFP(
            weight_ih_linear[reset_idx], 
            weight_hh_linear[reset_idx], 
            gate_params);

        new_gate = computeNewGateFP(
            weight_ih_linear[new_idx], 
            weight_hh_linear[new_idx], 
            reset_gate, gate_params, weight_hh_linear_g);

        cur_h = computeHiddenStateFP(update_gate, new_gate, h[output_idx], gate_params);
    }

    // Training: 保存中间值
    if constexpr (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = update_gate;
        v[base_v_idx + 1 * hidden_dim] = reset_gate;
        v[base_v_idx + 2 * hidden_dim] = new_gate;
        v[base_v_idx + 3 * hidden_dim] = weight_hh_linear_g;
    }

    // Zoneout（如果启用）
    if constexpr (ApplyZoneout) {
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
    g.scale_update_gate_input_ = exp2_scale(src.shift_update_gate_input_);
    g.scale_update_gate_output_ = exp2_scale(src.shift_update_gate_output_);
    g.scale_reset_gate_input_ = exp2_scale(src.shift_reset_gate_input_);
    g.scale_reset_gate_output_ = exp2_scale(src.shift_reset_gate_output_);
    g.scale_new_gate_input_ = exp2_scale(src.shift_new_gate_input_);
    g.scale_new_gate_output_ = exp2_scale(src.shift_new_gate_output_);

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

    // Linear 变换结果（cuBLAS GEMM 直接写入，然后原地 rescale）
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

    int threads = 256;
    int blocks = (hidden3 + threads - 1) / threads;

    // 计算 W_sum * zp_x（如果 zp_x != 0，否则直接清零）
    if (linear_params_.zp_x_ != 0.0f) {
        kernel::computeWeightSumKernel<<<blocks, threads, 0, stream>>>(
            W, W_sum_mul_x_zp_.data(), linear_params_.zp_x_, hidden3, input_size);
    } else {
        dev::fill_n(W_sum_mul_x_zp_.data(), hidden3, 0.0f);
    }

    // 计算 R_sum * zp_h（如果 zp_h != 0，否则直接清零）
    if (linear_params_.zp_h_ != 0.0f) {
        kernel::computeWeightSumKernel<<<blocks, threads, 0, stream>>>(
            R, R_sum_mul_h_zp_.data(), linear_params_.zp_h_, hidden3, hidden_size);
    } else {
        dev::fill_n(R_sum_mul_h_zp_.data(), hidden3, 0.0f);
    }

    cudaStreamSynchronize(stream);
    weight_sums_computed_ = true;
}

void ForwardPassQuantFP::ComputeLinearX(const float *W, const float *x, 
                                        const float *bw, int steps,
                                        uint8_t *weight_ih_linear_mask) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[1];

    const int M = hidden_size * 3;
    const int N = steps * batch_size;
    const int K = input_size;
    
    const bool training = data_->training;

    // 使用 cuBLAS SGEMM + 后处理
    cublasSetStream(data_->blas_handle, stream);

    // 1. cuBLAS SGEMM: 直接写入 tmp_weight_ih_linear_（后续原地处理）
    // W: [M, K] 列主序，x: [K, N] 列主序，输出 [M, N]
    static const float alpha = 1.0f;
    static const float beta = 0.0f;
    blas<float>::gemm(data_->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      M, N, K, &alpha, W, M, x, K, &beta, tmp_weight_ih_linear_.data(), M);

    // 2. Bias + Rescale kernel（原地处理）
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (training) {
        kernel::biasRescaleKernelWithMask<true><<<blocks, threads, 0, stream>>>(
            tmp_weight_ih_linear_.data(),  // 输入：GEMM 结果，输出：rescale 后的结果
            bw, 
            W_sum_mul_x_zp_.data(),
            linear_params_.div_gemm_x_to_weight_ih_linear_.data(),
            linear_params_.div_bw_to_weight_ih_linear_.data(),
            linear_params_.zp_weight_ih_linear_,
            M, N, 
            linear_params_.output_bw_ih_,
            weight_ih_linear_mask);
    } else {
        kernel::biasRescaleKernelWithMask<false><<<blocks, threads, 0, stream>>>(
            tmp_weight_ih_linear_.data(),  // 输入：GEMM 结果，输出：rescale 后的结果
            bw, 
            W_sum_mul_x_zp_.data(),
            linear_params_.div_gemm_x_to_weight_ih_linear_.data(),
            linear_params_.div_bw_to_weight_ih_linear_.data(),
            linear_params_.zp_weight_ih_linear_,
            M, N, 
            linear_params_.output_bw_ih_,
            nullptr);
    }
}

void ForwardPassQuantFP::ComputeLinearH(const float *R, const float *h, 
                                        const float *br,
                                        uint8_t *weight_hh_linear_mask) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[0];
    const bool training = data_->training;

    const int M = hidden_size * 3;
    const int N = batch_size;
    const int K = hidden_size;

    // 使用 cuBLAS SGEMM + 后处理
    cublasSetStream(data_->blas_handle, stream);

    // 1. cuBLAS SGEMM: 直接写入 tmp_weight_hh_linear_（后续原地处理）
    static const float alpha = 1.0f;
    static const float beta = 0.0f;
    blas<float>::gemm(data_->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      M, N, K, &alpha, R, M, h, K, &beta, tmp_weight_hh_linear_.data(), M);

    // 2. Bias + Rescale kernel（原地处理）
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (training) {
        kernel::biasRescaleKernelWithMask<true><<<blocks, threads, 0, stream>>>(
            tmp_weight_hh_linear_.data(),  // 输入：GEMM 结果，输出：rescale 后的结果
            br, 
            R_sum_mul_h_zp_.data(),
            linear_params_.div_gemm_h_to_weight_hh_linear_.data(),
            linear_params_.div_br_to_weight_hh_linear_.data(),
            linear_params_.zp_weight_hh_linear_,
            M, N, 
            linear_params_.output_bw_hh_,
            weight_hh_linear_mask);
    } else {
        kernel::biasRescaleKernelWithMask<false><<<blocks, threads, 0, stream>>>(
            tmp_weight_hh_linear_.data(),  // 输入：GEMM 结果，输出：rescale 后的结果
            br, 
            R_sum_mul_h_zp_.data(),
            linear_params_.div_gemm_h_to_weight_hh_linear_.data(),
            linear_params_.div_br_to_weight_hh_linear_.data(),
            linear_params_.zp_weight_hh_linear_,
            M, N, 
            linear_params_.output_bw_hh_,
            nullptr);
    }
}

void ForwardPassQuantFP::IterateInternal(
    const float *R,
    const float *br,
    const float *h,
    float *h_out,
    float *v,
    const float *cur_weight_ih_linear,
    float zoneout_prob,
    const float *zoneout_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask
) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    cublasSetStream(data_->blas_handle, stream1);

    // 计算隐状态 Linear 变换: R*h + br（与 stream2 上的 ComputeLinearX 并行执行）
    ComputeLinearH(R, h, br, weight_hh_linear_mask);

    // Pointwise kernel 配置
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    // 等待 ComputeLinearX 完成（pointwise kernel 需要同时使用 weight_ih_linear 和 weight_hh_linear）
    cudaStreamWaitEvent(stream1, event, 0);

    const bool apply_zoneout = (zoneout_prob > 0.0f && zoneout_mask != nullptr);

    // 启动 GRU pointwise kernel（4 种组合：Training * Zoneout）
    // Training 模式自动保存 mask（用于 QAT 反向传播）
    if (training) {
        if (apply_zoneout) {
            kernel::PointwiseOperationsFP<true, true>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size,
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_,
                    gate_input_mask, gate_output_mask, h_mask);
        } else {
            kernel::PointwiseOperationsFP<true, false>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size,
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_,
                    gate_input_mask, gate_output_mask, h_mask);
        }
    } else {
        if (apply_zoneout) {
            kernel::PointwiseOperationsFP<false, true>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size,
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_,
                    nullptr, nullptr, nullptr);
        } else {
            kernel::PointwiseOperationsFP<false, false>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size,
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_,
                    nullptr, nullptr, nullptr);
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
    const float *zoneout_mask,
    uint8_t *weight_ih_linear_mask,
    uint8_t *weight_hh_linear_mask,
    uint8_t *gate_input_mask,
    uint8_t *gate_output_mask,
    uint8_t *h_mask
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
    // 当 training=true 时，ComputeLinearX 内部会填充 weight_ih_linear_mask
    ComputeLinearX(W, x, bw, steps, weight_ih_linear_mask);

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
                        zoneout_mask ? zoneout_mask + i * NH : nullptr,
                        weight_hh_linear_mask ? weight_hh_linear_mask + i * NH3 : nullptr,
                        gate_input_mask ? gate_input_mask + i * NH3 : nullptr,
                        gate_output_mask ? gate_output_mask + i * NH3 : nullptr,
                        h_mask ? h_mask + i * NH : nullptr);
    }

    cublasSetStream(data_->blas_handle, save_stream);
}

}  // namespace gru
