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

/**
 * @brief 内联函数：乘法 + 四舍五入（优化版本，使用倒数）
 * 
 * 使用预计算的倒数，乘法比除法快 3-5 倍
 * 
 * @param x 被除数
 * @param inv_divisor 倒数（1.0f / divisor）
 */
__device__ __forceinline__ 
float mul_round(float x, float inv_divisor) {
    return round_f(x * inv_divisor);
}

// 注：clamp_f, real_sigmoid_f, real_tanh_f 已移至 quantize_ops_helper.h

// ============================================================================
// 2. 门计算函数
// ============================================================================

/**
 * @brief 内联函数：对 GEMM 结果进行 Bias + Rescale（融合到 PointwiseOperationsFP）
 * 
 * 计算: result = clamp(round((gemm_result - W_sum_mul_zp) / div_gemm 
 *                          + round(bias / div_bias) + zp_out))
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 */
template <bool Training>
__device__ __forceinline__
float biasRescaleInline(
    float gemm_result,
    float bias,
    float W_sum_mul_zp,
    float inv_div_gemm,  // 倒数（1.0f / div_gemm）
    float inv_div_bias,  // 倒数（1.0f / div_bias）
    float zp_out,
    QuantBitWidth output_bw,
    uint8_t* was_clamped = nullptr
) {
    // GEMM 结果减去零点补偿
    const float val = gemm_result - W_sum_mul_zp;
    // bias 先 rescale（使用倒数，乘法替代除法）
    const float bias_term = round_f(bias * inv_div_bias);
    // GEMM + bias 一起 rescale（使用倒数，乘法替代除法）
    const float result = round_f((val + bias_term) * inv_div_gemm) + zp_out;
    
    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        return clamp_f_with_mask(result, output_bw, *was_clamped);
    } else {
        return clamp_f(result, output_bw);
    }
}

/**
 * @brief 计算更新门 update_gate = sigmoid(weight_ih_linear + weight_hh_linear)
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param input_was_clamped 训练模式时保存输入 clamp mask，推理模式时可为 nullptr
 * @param output_was_clamped 训练模式时保存输出 clamp mask，推理模式时可为 nullptr
 */
template <bool Training>
__device__ __forceinline__ 
float computeUpdateGateFP(float weight_ih_linear, float weight_hh_linear, 
                          const GateQuantParamsFP &p,
                          uint8_t* input_was_clamped = nullptr,
                          uint8_t* output_was_clamped = nullptr) {
    // 重缩放到 update_gate_input 空间（使用倒数，乘法替代除法）
    const float ih = mul_round(weight_ih_linear - p.zp_weight_ih_linear_, 
                               p.inv_div_weight_ih_linear_to_update_gate_input_);
    const float hh = mul_round(weight_hh_linear - p.zp_weight_hh_linear_, 
                               p.inv_div_weight_hh_linear_to_update_gate_input_);
    const float input = ih + hh + p.zp_update_gate_input_;
    
    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        const float clamped_input = clamp_f_with_mask(input, p.bitwidth_config_.update_gate_input_, *input_was_clamped);
        return real_sigmoid_f_with_mask(clamped_input, 
                          p.scale_update_gate_input_, p.zp_update_gate_input_,
                          p.scale_update_gate_output_, p.zp_update_gate_output_,
                          p.bitwidth_config_.update_gate_output_, *output_was_clamped);
    } else {
        return real_sigmoid_f(input, 
                          p.scale_update_gate_input_, p.zp_update_gate_input_,
                          p.scale_update_gate_output_, p.zp_update_gate_output_,
                          p.bitwidth_config_.update_gate_output_);
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
__device__ __forceinline__ 
float computeResetGateFP(float weight_ih_linear, float weight_hh_linear,
                         const GateQuantParamsFP &p,
                         uint8_t* input_was_clamped = nullptr,
                         uint8_t* output_was_clamped = nullptr) {
    float ih = mul_round(weight_ih_linear - p.zp_weight_ih_linear_,
                         p.inv_div_weight_ih_linear_to_reset_gate_input_);
    float hh = mul_round(weight_hh_linear - p.zp_weight_hh_linear_,
                         p.inv_div_weight_hh_linear_to_reset_gate_input_);
    float input = ih + hh + p.zp_reset_gate_input_;
    
    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        const float clamped_input = clamp_f_with_mask(input, p.bitwidth_config_.reset_gate_input_, *input_was_clamped);
        return real_sigmoid_f_with_mask(clamped_input,
                          p.scale_reset_gate_input_, p.zp_reset_gate_input_,
                          p.scale_reset_gate_output_, p.zp_reset_gate_output_,
                          p.bitwidth_config_.reset_gate_output_, *output_was_clamped);
    } else {
        return real_sigmoid_f(input,
                          p.scale_reset_gate_input_, p.zp_reset_gate_input_,
                          p.scale_reset_gate_output_, p.zp_reset_gate_output_,
                          p.bitwidth_config_.reset_gate_output_);
    }
}

/**
 * @brief 计算候选门 new_gate = tanh(weight_ih_linear + reset_gate * weight_hh_linear)
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param input_was_clamped 训练模式时保存输入 clamp mask，推理模式时可为 nullptr
 * @param output_was_clamped 训练模式时保存输出 clamp mask，推理模式时可为 nullptr
 * @note weight_hh_linear 参数（即 R*h + br）在反向传播时直接保存到 v，无需额外输出参数
 */
template <bool Training>
__device__ __forceinline__ 
float computeNewGateFP(float weight_ih_linear, float weight_hh_linear, float reset_gate,
                       const GateQuantParamsFP &p,
                       uint8_t* input_was_clamped = nullptr,
                       uint8_t* output_was_clamped = nullptr) {
    // 计算 reset_gate * weight_hh_linear，直接对齐到 new_gate_input
    // 使用 float 精度（单个乘积最大 4,161,409，在 FP32 精确范围内）
    const float r_diff = reset_gate - p.zp_reset_gate_output_;
    const float hh_diff = weight_hh_linear - p.zp_weight_hh_linear_;
    const float reset_hidden_mul = r_diff * hh_diff;
    const float rh = mul_round(reset_hidden_mul, p.inv_div_reset_mul_hh_to_new_gate_input_);
    
    // weight_ih_linear 重缩放到 new_gate_input 空间（使用倒数，乘法替代除法）
    const float ih = mul_round(weight_ih_linear - p.zp_weight_ih_linear_,
                               p.inv_div_weight_ih_linear_to_new_gate_input_);
    const float input = ih + rh + p.zp_new_gate_input_;
    
    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        const float clamped_input = clamp_f_with_mask(input, p.bitwidth_config_.new_gate_input_, *input_was_clamped);
        return real_tanh_f_with_mask(clamped_input,
                       p.scale_new_gate_input_, p.zp_new_gate_input_,
                       p.scale_new_gate_output_, p.zp_new_gate_output_,
                       p.bitwidth_config_.new_gate_output_, *output_was_clamped);
    } else {
        return real_tanh_f(input,
                       p.scale_new_gate_input_, p.zp_new_gate_input_,
                       p.scale_new_gate_output_, p.zp_new_gate_output_,
                       p.bitwidth_config_.new_gate_output_);
    }
}

/**
 * @brief 计算隐藏状态 h_new = update_gate * h_old + (1 - update_gate) * new_gate
 * 
 * @tparam Training 是否训练模式（决定是否使用 mask）
 * @param was_clamped 训练模式时保存 clamp mask，推理模式时可为 nullptr
 */
template <bool Training>
__device__ __forceinline__ 
float computeHiddenStateFP(float update_gate, float new_gate, float h_old,
                           const GateQuantParamsFP &p,
                           uint8_t* was_clamped = nullptr) {
    // 计算 update_gate * h_old，直接对齐到 h
    // 使用 float 精度（单个乘积最大 4,161,409，在 FP32 精确范围内）
    const float u_diff = update_gate - p.zp_update_gate_output_;
    const float h_diff = h_old - p.zp_h_;
    const float old_contribution_mul = u_diff * h_diff;
    const float old_term = mul_round(old_contribution_mul, p.inv_div_update_old_to_h_);
    
    // 计算 (1 - update_gate) * new_gate，直接对齐到 h
    // quant_one = 2^shift + zp，是常数 1 在 update_gate_output 量化空间的完整表示
    // one_minus_u = quant_one - update_gate = (2^shift + zp) - update_gate
    const float one_minus_u = p.quant_one_in_update_gate_scale_ - update_gate;
    const float n_diff = new_gate - p.zp_new_gate_output_;
    const float new_contribution_mul = one_minus_u * n_diff;
    const float new_term = mul_round(new_contribution_mul, p.inv_div_update_new_to_h_);
    
    const float h_new = old_term + new_term + p.zp_h_;
    
    // 使用 if constexpr 避免运行时分支
    if constexpr (Training) {
        return clamp_f_with_mask(h_new, p.bitwidth_config_.h_, *was_clamped);
    } else {
        return clamp_f(h_new, p.bitwidth_config_.h_);
    }
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
    
    const int row = idx % M;  // 列主序
    
    // GEMM 结果减去零点补偿（W_sum_mul_zp 是 float，权重总是 8bit）
    const float val = gemm_result[idx] - W_sum_mul_zp[row];
    
    // bias 先 rescale（bias 是 8bit/16bit 整数，可以用 float 精确表示）
    const float bias_term = round_f(bias[row] / div_bias[row]);
    
    // GEMM + bias 一起 rescale（最终结果在量化范围内）
    const float result = round_f((val + bias_term) / div_gemm[row]) + zp_out;
    
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
    
    const int row = idx % M;
    
    // 读取 GEMM 结果，减去零点补偿（W_sum_mul_zp 是 float，权重总是 8bit）
    const float val = data[idx] - W_sum_mul_zp[row];
    // bias 先 rescale（bias 是 8bit/16bit 整数，可以用 float 精确表示）
    const float bias_term = round_f(bias[row] / div_bias[row]);
    // GEMM + bias 一起 rescale（最终结果在量化范围内）
    const float result = round_f((val + bias_term) / div_gemm[row]) + zp_out;
    
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
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    // 使用 float 累加（权重总是 8bit，K < 1024 且 zp < 100 时 float 足够精确）
    float sum = 0.0f;  // sum 在循环中累加，不能使用 const
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
    // 未 rescale 的 GEMM 结果（融合 BiasRescale 到本 kernel）
    const float *__restrict__ gemm_weight_ih_linear,  // [batch*3*hidden] 未 rescale 的 W*x
    const float *__restrict__ gemm_weight_hh_linear,  // [batch*3*hidden] 未 rescale 的 R*h
    // BiasRescale 参数（打包到结构体中，减少参数数量）
    LinearRescaleParamsFP linear_rescale_params,
    // 偏置（直接从 IterateInternal 参数传入）
    const float *__restrict__ bw,  // [3*hidden] 输入偏置
    const float *__restrict__ br,  // [3*hidden] 循环偏置
    // 其他参数
    const float *__restrict__ h,
    float *__restrict__ h_out,
    float *__restrict__ v,
    float zoneout_prob,
    const float *__restrict__ zoneout_mask,
    GateQuantParamsFP gate_params,
    // Mask 输出（Training=true 时使用）
    uint8_t *__restrict__ weight_ih_linear_mask,     // [batch*3*hidden] weight_ih_linear rescale mask
    uint8_t *__restrict__ weight_hh_linear_mask,     // [batch*3*hidden] weight_hh_linear rescale mask
    uint8_t *__restrict__ gate_input_mask,           // [batch*3*hidden] 门输入 clamp mask
    uint8_t *__restrict__ gate_output_mask,          // [batch*3*hidden] 门输出 clamp mask
    uint8_t *__restrict__ h_mask                     // [batch*hidden] 隐状态输出 mask
) {
    // ========== Shared Memory 缓存 ==========
    // 缓存 bw 和 br（每个 row 只需要加载一次，16 个 col 线程共享）
    // 使用 [4] 而不是 [3] 避免 bank conflict（padding）
    __shared__ float shared_bw[32][4];   // [blockDim.x, 3 gates + 1 padding] = 512 字节
    __shared__ float shared_br[32][4];   // [blockDim.x, 3 gates + 1 padding] = 512 字节
    
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim) return;

    // ========== 协作加载 bw 和 br 到 shared memory ==========
    // 只需要 threadIdx.y == 0 的线程加载（16 个线程中的 1 个）
    if (threadIdx.y == 0) {
        // 预计算索引（避免重复计算）
        const int row_update = row;
        const int row_reset = row + hidden_dim;
        const int row_new = row + 2 * hidden_dim;
        
        shared_bw[threadIdx.x][0] = bw[row_update];
        shared_bw[threadIdx.x][1] = bw[row_reset];
        shared_bw[threadIdx.x][2] = bw[row_new];
        // shared_bw[threadIdx.x][3] 未使用（padding，避免 bank conflict）
        shared_br[threadIdx.x][0] = br[row_update];
        shared_br[threadIdx.x][1] = br[row_reset];
        shared_br[threadIdx.x][2] = br[row_new];
        // shared_br[threadIdx.x][3] 未使用（padding，避免 bank conflict）
    }
    __syncthreads();  // 确保所有数据加载完成

    const int weight_idx = col * (hidden_dim * 3) + row;
    const int output_idx = col * hidden_dim + row;
    const int update_idx = weight_idx + 0 * hidden_dim;
    const int reset_idx = weight_idx + 1 * hidden_dim;
    const int new_idx = weight_idx + 2 * hidden_dim;

    // ========== 预计算索引 ==========
    const int row_update = row;
    const int row_reset = row + hidden_dim;
    const int row_new = row + 2 * hidden_dim;

    // ========== 从 shared memory 读取 bw 和 br ==========
    const float bw_update = shared_bw[threadIdx.x][0];
    const float bw_reset = shared_bw[threadIdx.x][1];
    const float bw_new = shared_bw[threadIdx.x][2];
    const float br_update = shared_br[threadIdx.x][0];
    const float br_reset = shared_br[threadIdx.x][1];
    const float br_new = shared_br[threadIdx.x][2];

    // 统一声明 mask 变量（函数内部会根据 Training 模板参数决定是否使用）
    uint8_t ih_update_mask, ih_reset_mask, ih_new_mask;
    uint8_t hh_update_mask, hh_reset_mask, hh_new_mask;

    // 在 kernel 内部进行 BiasRescale（减少全局内存读取）
    // 计算后不变的值使用 const
    const float weight_ih_linear_update = biasRescaleInline<Training>(
        gemm_weight_ih_linear[update_idx], 
        bw_update,  // 从 shared memory 读取（原：bw[row]）
        linear_rescale_params.W_sum_mul_x_zp[row_update],
        linear_rescale_params.inv_div_gemm_x_to_weight_ih_linear_[row_update], 
        linear_rescale_params.inv_div_bw_to_weight_ih_linear_[row_update],
        linear_rescale_params.zp_weight_ih_linear_, 
        linear_rescale_params.output_bw_ih_, 
        &ih_update_mask);
    const float weight_ih_linear_reset = biasRescaleInline<Training>(
        gemm_weight_ih_linear[reset_idx], 
        bw_reset,  // 从 shared memory 读取（原：bw[row + hidden_dim]）
        linear_rescale_params.W_sum_mul_x_zp[row_reset],
        linear_rescale_params.inv_div_gemm_x_to_weight_ih_linear_[row_reset], 
        linear_rescale_params.inv_div_bw_to_weight_ih_linear_[row_reset],
        linear_rescale_params.zp_weight_ih_linear_, 
        linear_rescale_params.output_bw_ih_, 
        &ih_reset_mask);
    const float weight_ih_linear_new = biasRescaleInline<Training>(
        gemm_weight_ih_linear[new_idx], 
        bw_new,  // 从 shared memory 读取（原：bw[row + 2 * hidden_dim]）
        linear_rescale_params.W_sum_mul_x_zp[row_new],
        linear_rescale_params.inv_div_gemm_x_to_weight_ih_linear_[row_new], 
        linear_rescale_params.inv_div_bw_to_weight_ih_linear_[row_new],
        linear_rescale_params.zp_weight_ih_linear_, 
        linear_rescale_params.output_bw_ih_, 
        &ih_new_mask);
    
    // Rescale weight_hh_linear (update, reset, new) - 使用倒数，乘法替代除法
    // 使用从 shared memory 读取的 br_update, br_reset, br_new
    const float weight_hh_linear_update = biasRescaleInline<Training>(
        gemm_weight_hh_linear[update_idx], 
        br_update,  // 从 shared memory 读取（原：br[row]）
        linear_rescale_params.R_sum_mul_h_zp[row_update],
        linear_rescale_params.inv_div_gemm_h_to_weight_hh_linear_[row_update], 
        linear_rescale_params.inv_div_br_to_weight_hh_linear_[row_update],
        linear_rescale_params.zp_weight_hh_linear_, 
        linear_rescale_params.output_bw_hh_, 
        &hh_update_mask);
    const float weight_hh_linear_reset = biasRescaleInline<Training>(
        gemm_weight_hh_linear[reset_idx], 
        br_reset,  // 从 shared memory 读取（原：br[row + hidden_dim]）
        linear_rescale_params.R_sum_mul_h_zp[row_reset],
        linear_rescale_params.inv_div_gemm_h_to_weight_hh_linear_[row_reset], 
        linear_rescale_params.inv_div_br_to_weight_hh_linear_[row_reset],
        linear_rescale_params.zp_weight_hh_linear_, 
        linear_rescale_params.output_bw_hh_, 
        &hh_reset_mask);
    const float weight_hh_linear_new = biasRescaleInline<Training>(
        gemm_weight_hh_linear[new_idx], 
        br_new,  // 从 shared memory 读取（原：br[row + 2 * hidden_dim]）
        linear_rescale_params.R_sum_mul_h_zp[row_new],
        linear_rescale_params.inv_div_gemm_h_to_weight_hh_linear_[row_new], 
        linear_rescale_params.inv_div_br_to_weight_hh_linear_[row_new],
        linear_rescale_params.zp_weight_hh_linear_, 
        linear_rescale_params.output_bw_hh_, 
        &hh_new_mask);
    
    // 保存 rescale mask（只在训练模式时执行）
    // 注意：Training=true 时，weight_ih_linear_mask 和 weight_hh_linear_mask 必须非空（外部负责）
    if constexpr (Training) {
        weight_ih_linear_mask[update_idx] = ih_update_mask;
        weight_ih_linear_mask[reset_idx] = ih_reset_mask;
        weight_ih_linear_mask[new_idx] = ih_new_mask;

        weight_hh_linear_mask[update_idx] = hh_update_mask;
        weight_hh_linear_mask[reset_idx] = hh_reset_mask;
        weight_hh_linear_mask[new_idx] = hh_new_mask;
    }

    // 统一声明 mask 变量（函数内部会根据 Training 模板参数决定是否使用）
    uint8_t update_input_clamped, update_output_clamped;
    uint8_t reset_input_clamped, reset_output_clamped;
    uint8_t new_input_clamped, new_output_clamped;
    uint8_t h_clamped;
    
    // 计算门结果（按依赖顺序，计算后不变的值使用 const）
    const float update_gate = computeUpdateGateFP<Training>(
        weight_ih_linear_update, 
        weight_hh_linear_update, 
        gate_params,
        &update_input_clamped, &update_output_clamped);

    const float reset_gate = computeResetGateFP<Training>(
        weight_ih_linear_reset, 
        weight_hh_linear_reset, 
        gate_params,
        &reset_input_clamped, &reset_output_clamped);

    const float new_gate = computeNewGateFP<Training>(
        weight_ih_linear_new, 
        weight_hh_linear_new, 
        reset_gate, gate_params,
        &new_input_clamped, &new_output_clamped);

    // cur_h 可能被 Zoneout 修改，不能使用 const
    float cur_h = computeHiddenStateFP<Training>(
        update_gate, new_gate, h[output_idx], gate_params,
        &h_clamped);
    
    // Training: 保存 mask 和中间值
    if constexpr (Training) {
        // 保存门输入 mask
        gate_input_mask[update_idx] = update_input_clamped;
        gate_input_mask[reset_idx] = reset_input_clamped;
        gate_input_mask[new_idx] = new_input_clamped;
        
        // 保存门输出 mask
        gate_output_mask[update_idx] = update_output_clamped;
        gate_output_mask[reset_idx] = reset_output_clamped;
        gate_output_mask[new_idx] = new_output_clamped;
        
        // 保存隐状态 mask
        h_mask[output_idx] = h_clamped;
        
        // 保存中间值
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = update_gate;
        v[base_v_idx + 1 * hidden_dim] = reset_gate;
        v[base_v_idx + 2 * hidden_dim] = new_gate;
        v[base_v_idx + 3 * hidden_dim] = weight_hh_linear_new;  // 直接使用 weight_hh_linear_new
    }

    // Zoneout（如果启用）
    if constexpr (ApplyZoneout) {
        const float mask = zoneout_mask[output_idx];
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
    // 直接计算倒数（优化：乘法替代除法）
    float div_ih_u = exp2_scale(-shift_ih_u);
    float div_hh_u = exp2_scale(-shift_hh_u);
    g.inv_div_weight_ih_linear_to_update_gate_input_ = 1.0f / div_ih_u;
    g.inv_div_weight_hh_linear_to_update_gate_input_ = 1.0f / div_hh_u;

    // Reset gate
    g.zp_reset_gate_input_ = static_cast<float>(src.zp_reset_gate_input_);
    g.zp_reset_gate_output_ = static_cast<float>(src.zp_reset_gate_output_);
    int8_t shift_ih_r = src.shift_weight_ih_linear_ - src.shift_reset_gate_input_;
    int8_t shift_hh_r = src.shift_weight_hh_linear_ - src.shift_reset_gate_input_;
    float div_ih_r = exp2_scale(-shift_ih_r);
    float div_hh_r = exp2_scale(-shift_hh_r);
    g.inv_div_weight_ih_linear_to_reset_gate_input_ = 1.0f / div_ih_r;
    g.inv_div_weight_hh_linear_to_reset_gate_input_ = 1.0f / div_hh_r;

    // New gate
    g.zp_new_gate_input_ = static_cast<float>(src.zp_new_gate_input_);
    g.zp_new_gate_output_ = static_cast<float>(src.zp_new_gate_output_);
    int8_t shift_ih_n = src.shift_weight_ih_linear_ - src.shift_new_gate_input_;
    int8_t shift_rh_n = (src.shift_reset_gate_output_ + src.shift_weight_hh_linear_) - src.shift_new_gate_input_;
    float div_ih_n = exp2_scale(-shift_ih_n);
    float div_rh_n = exp2_scale(-shift_rh_n);
    g.inv_div_weight_ih_linear_to_new_gate_input_ = 1.0f / div_ih_n;
    g.inv_div_reset_mul_hh_to_new_gate_input_ = 1.0f / div_rh_n;

    // Hidden state
    // quant_one = rshift_round(1, -shift) + zp = (1 << shift) + zp = 2^shift + zp
    // 注意：rshift_round(1, -n) 当 n>0 时等于 1 << n
    g.quant_one_in_update_gate_scale_ = exp2_scale(-src.shift_update_gate_output_) + 
                                        static_cast<float>(src.zp_update_gate_output_);
    int8_t shift_un = (src.shift_update_gate_output_ + src.shift_new_gate_output_) - src.shift_h_;
    int8_t shift_uh = src.shift_update_gate_output_;  // shift_update_gate_output_ + shift_h_ - shift_h_
    float div_un = exp2_scale(-shift_un);
    float div_uh = exp2_scale(-shift_uh);
    g.inv_div_update_new_to_h_ = 1.0f / div_un;
    g.inv_div_update_old_to_h_ = 1.0f / div_uh;

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

    // 直接计算倒数数组（优化：乘法替代除法）
    std::vector<float> inv_div_gemm_x(channel), inv_div_bw(channel);
    std::vector<float> inv_div_gemm_h(channel), inv_div_br(channel);
    for (int i = 0; i < channel; ++i) {
        // GEMM: scale_W * scale_x -> scale_weight_ih_linear
        // shift = (shift_W + shift_x) - shift_weight_ih_linear
        int8_t shift_gx = (src.shift_W_[i] + src.shift_x_) - src.shift_weight_ih_linear_;
        // bias: 先 shift 到 GEMM scale，再和 GEMM 一起 shift
        int8_t shift_bw = src.shift_bw_[i] - (src.shift_W_[i] + src.shift_x_);
        float div_gemm_x = exp2_scale(-shift_gx);
        float div_bw = exp2_scale(-shift_bw);
        // 直接计算倒数
        inv_div_gemm_x[i] = 1.0f / div_gemm_x;
        inv_div_bw[i] = 1.0f / div_bw;

        int8_t shift_gh = (src.shift_R_[i] + src.shift_h_) - src.shift_weight_hh_linear_;
        int8_t shift_br = src.shift_br_[i] - (src.shift_R_[i] + src.shift_h_);
        float div_gemm_h = exp2_scale(-shift_gh);
        float div_br = exp2_scale(-shift_br);
        // 直接计算倒数
        inv_div_gemm_h[i] = 1.0f / div_gemm_h;
        inv_div_br[i] = 1.0f / div_br;
    }
    // 只存储倒数数组
    l.inv_div_gemm_x_to_weight_ih_linear_ = dev::vector<float>(inv_div_gemm_x);
    l.inv_div_bw_to_weight_ih_linear_ = dev::vector<float>(inv_div_bw);
    l.inv_div_gemm_h_to_weight_hh_linear_ = dev::vector<float>(inv_div_gemm_h);
    l.inv_div_br_to_weight_hh_linear_ = dev::vector<float>(inv_div_br);
    
    l.output_bw_ih_ = src.bitwidth_config_.weight_ih_linear_;
    l.output_bw_hh_ = src.bitwidth_config_.weight_hh_linear_;

    // 填充 LinearRescaleParamsFP 的静态部分（指针和标量值）
    // 注意：
    //   - W_sum_mul_x_zp 和 R_sum_mul_h_zp 指针在 EnsureBuffersAllocated 中更新（确保缓冲区已分配）
    //   - bw 和 br 指针在 IterateInternal 中更新（从参数传入）
    linear_rescale_params_.inv_div_gemm_x_to_weight_ih_linear_ = l.inv_div_gemm_x_to_weight_ih_linear_.data();
    linear_rescale_params_.inv_div_bw_to_weight_ih_linear_ = l.inv_div_bw_to_weight_ih_linear_.data();
    linear_rescale_params_.zp_weight_ih_linear_ = l.zp_weight_ih_linear_;
    linear_rescale_params_.output_bw_ih_ = l.output_bw_ih_;
    linear_rescale_params_.inv_div_gemm_h_to_weight_hh_linear_ = l.inv_div_gemm_h_to_weight_hh_linear_.data();
    linear_rescale_params_.inv_div_br_to_weight_hh_linear_ = l.inv_div_br_to_weight_hh_linear_.data();
    linear_rescale_params_.zp_weight_hh_linear_ = l.zp_weight_hh_linear_;
    linear_rescale_params_.output_bw_hh_ = l.output_bw_hh_;

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

    // 更新 LinearRescaleParamsFP 中的指针（确保缓冲区已分配）
    linear_rescale_params_.W_sum_mul_x_zp = W_sum_mul_x_zp_.data();
    linear_rescale_params_.R_sum_mul_h_zp = R_sum_mul_h_zp_.data();

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
                                        const float *bw, int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[1];

    const int M = hidden_size * 3;
    const int N = steps * batch_size;
    const int K = input_size;

    // 使用 cuBLAS SGEMM（BiasRescale 已融合到 PointwiseOperationsFP）
    cublasSetStream(data_->blas_handle, stream);

    // cuBLAS SGEMM: 直接写入 tmp_weight_ih_linear_（未 rescale 的原始 GEMM 结果）
    // W: [M, K] 列主序，x: [K, N] 列主序，输出 [M, N]
    static const float alpha = 1.0f;
    static const float beta = 0.0f;
    blas<float>::gemm(data_->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      M, N, K, &alpha, W, M, x, K, &beta, tmp_weight_ih_linear_.data(), M);
    
    // 注意：BiasRescale 已融合到 PointwiseOperationsFP，减少全局内存读取
}

void ForwardPassQuantFP::ComputeLinearH(const float *R, const float *h, 
                                        const float *br,
                                        uint8_t *weight_hh_linear_mask) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cudaStream_t stream = data_->stream[0];

    const int M = hidden_size * 3;
    const int N = batch_size;
    const int K = hidden_size;

    // 使用 cuBLAS SGEMM（BiasRescale 已融合到 PointwiseOperationsFP）
    cublasSetStream(data_->blas_handle, stream);

    // cuBLAS SGEMM: 直接写入 tmp_weight_hh_linear_（未 rescale 的原始 GEMM 结果）
    static const float alpha = 1.0f;
    static const float beta = 0.0f;
    blas<float>::gemm(data_->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      M, N, K, &alpha, R, M, h, K, &beta, tmp_weight_hh_linear_.data(), M);
    
    // 注意：BiasRescale 已融合到 PointwiseOperationsFP，减少全局内存读取
}

void ForwardPassQuantFP::IterateInternal(
    const float *R,
    const float *bw,
    const float *br,
    const float *h,
    float *h_out,
    float *v,
    const float *cur_weight_ih_linear,
    float zoneout_prob,
    const float *zoneout_mask,
    uint8_t *weight_ih_linear_mask,
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
    // BiasRescale 已融合到 PointwiseOperationsFP，减少全局内存读取
    // Training 模式自动保存 mask（用于 QAT 反向传播）
    // bw 和 br 直接从 IterateInternal 参数传入，不需要存到 linear_rescale_params_ 中
    if (training) {
        if (apply_zoneout) {
            kernel::PointwiseOperationsFP<true, true>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size,
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    linear_rescale_params_,
                    bw, br,
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_,
                    weight_ih_linear_mask, weight_hh_linear_mask,
                    gate_input_mask, gate_output_mask, h_mask);
        } else {
            kernel::PointwiseOperationsFP<true, false>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size,
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    linear_rescale_params_,
                    bw, br,
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_,
                    weight_ih_linear_mask, weight_hh_linear_mask,
                    gate_input_mask, gate_output_mask, h_mask);
        }
    } else {
        if (apply_zoneout) {
            kernel::PointwiseOperationsFP<false, true>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size,
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    linear_rescale_params_,
                    bw, br,
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_,
                    nullptr, nullptr,
                    nullptr, nullptr, nullptr);
        } else {
            kernel::PointwiseOperationsFP<false, false>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size,
                    cur_weight_ih_linear, tmp_weight_hh_linear_.data(),
                    linear_rescale_params_,
                    bw, br,
                    h, h_out, v, zoneout_prob, zoneout_mask, gate_params_,
                    nullptr, nullptr,
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
    // 量化模式：禁用 TensorCore 以提高精度（与浮点模式保持一致）
    // TensorCore 使用 TF32 精度，可能导致精度问题
    // const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);  // 注释掉以禁用 TensorCore
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
    // 注意：weight_ih_linear_mask 现在在 PointwiseOperationsFP 中填充
    ComputeLinearX(W, x, bw, steps);

    // 同步 Linear 计算
    cudaEventRecord(event, stream2);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    // 时间步循环
    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, bw, br,
                        h + i * NH,                           // 输入 h
                        h + (i + 1) * NH,                     // 输出 h
                        v ? v + i * NH * 4 : nullptr,         // 中间激活
                        tmp_weight_ih_linear_.data() + i * NH3,  // 当前时间步的 W*x（未 rescale）
                        zoneout_prob, 
                        zoneout_mask ? zoneout_mask + i * NH : nullptr,
                        weight_ih_linear_mask ? weight_ih_linear_mask + i * NH3 : nullptr,
                        weight_hh_linear_mask ? weight_hh_linear_mask + i * NH3 : nullptr,
                        gate_input_mask ? gate_input_mask + i * NH3 : nullptr,
                        gate_output_mask ? gate_output_mask + i * NH3 : nullptr,
                        h_mask ? h_mask + i * NH : nullptr);
    }

    cublasSetStream(data_->blas_handle, save_stream);
}

}  // namespace gru
