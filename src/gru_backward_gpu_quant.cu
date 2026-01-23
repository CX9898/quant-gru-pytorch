// Copyright 2020 LMNT, Inc. All Rights Reserved.
// Copyright 2024 Quant-GRU Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
//
// 量化 GRU 反向传播实现（支持 QAT mask）
//
// 基于 gru_backward_gpu.cu，增加 QAT mask 支持：
//   - x_mask [T*N, C] → dx
//   - h0_mask [N, H] → dh（最终传回初始状态的梯度）
//   - W_mask [C, H*3] → dW
//   - R_mask [H, H*3] → dR
//   - bw_mask [H*3] → dbw
//   - br_mask [H*3] → dbr
//   - weight_ih_linear_mask [T*N, H*3] → dp
//   - weight_hh_linear_mask [T*N, H*3] → dq
//   - gate_mask [T*N, H*3] → 门梯度
//   - h_mask [T*N, H] → 隐状态梯度
//
// STE (Straight-Through Estimator) 实现：
//   - mask=1（被 clamp）: 梯度置零
//   - mask=0（未被 clamp）: 梯度正常传播
//
// ==============================================================================

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "blas.h"
#include "device_assert.h"
#include "gru_quant.h"
#include "inline_ops.h"

namespace kernel {

/**
 * @brief 通用 mask 应用 kernel（STE）
 * 对 data 中 mask=1 的位置置零
 */
template <typename T>
__global__ void ApplyMaskKernel(T *data, const uint8_t *mask, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (mask[idx] != 0) {
        data[idx] = static_cast<T>(0.0);
    }
}

/**
 * @brief 量化 GRU 反向传播 pointwise 操作（带 mask 支持和 rescale 补偿）
 *
 * @tparam T 数据类型
 * @tparam ApplyZoneout 是否应用 Zoneout
 * @tparam ApplyRescale 是否应用 rescale 补偿
 *
 * 输入：
 *   - h: [N, H] 上一时间步隐状态
 *   - v: [N, H*4] 中间值（z, r, g, q_g）
 *   - dh_new: [N, H] 当前时间步隐状态梯度
 *   - zoneout_mask: [N, H] Zoneout mask（ApplyZoneout=true 时使用）
 *   - weight_hh_linear_mask: [N, H*3] R*h+br 输出的 clamp mask
 *   - gate_mask: [N, H*3] 门输出 clamp mask
 *   - h_mask: [N, H] 隐状态输出 clamp mask
 *   - bw_mask: [H*3] 偏置 bw 量化 clamp mask
 *   - br_mask: [H*3] 偏置 br 量化 clamp mask
 *   - rescale_params: 反向传播 rescale 参数（ApplyRescale=true 时使用）
 *
 * 输出：
 *   - dbw_out: [H*3] 输入偏置梯度（atomicAdd 累加）
 *   - dbr_out: [H*3] 循环偏置梯度（atomicAdd 累加）
 *   - dh_inout: [N, H] 传递到上一时间步的隐状态梯度
 *   - dp_out: [N, H*3] dp 中间梯度（传回 weight_ih_linear）
 *   - dq_out: [N, H*3] dq 中间梯度（传回 weight_hh_linear）
 *
 * Rescale 补偿说明：
 *   前向传播中：gate_input = div_round(linear_output, divisor)
 *   反向传播中：d_linear_output = d_gate_input * divisor
 *   dp 和 dq 是相对于 gate_input 的梯度，需要乘以 divisor 才能传回 linear_output
 */
template <typename T, bool ApplyZoneout, bool ApplyRescale>
__global__ void PointwiseOperationsQuant(
    const int batch_dim, const int hidden_dim,
    const T *h, const T *v, const T *dh_new,
    T *dbw_out, T *dbr_out, T *dh_inout, T *dp_out, T *dq_out,
    const T *zoneout_mask,
    const uint8_t *weight_hh_linear_mask,
    const uint8_t *gate_mask,
    const uint8_t *h_mask,
    const uint8_t *bw_mask,
    const uint8_t *br_mask,
    // Rescale 参数（ApplyRescale=true 时使用）
    const T mul_update_ih, const T mul_update_hh,
    const T mul_reset_ih, const T mul_reset_hh,
    const T mul_new_ih, const T mul_new_rh,
    const T mul_h_new, const T mul_h_old) {
    
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim) return;

    const int base_idx = col * hidden_dim + row;

    T dh_total = dh_new[base_idx] + dh_inout[base_idx];

    // 应用 h_mask（STE：隐状态被 clamp 时梯度置零）
    if (h_mask != nullptr && h_mask[base_idx] != 0) {
        dh_total = static_cast<T>(0.0);
    }

    const int stride4_base_idx = col * (hidden_dim * 4) + row;
    const int z_idx = stride4_base_idx + 0 * hidden_dim;
    const int r_idx = stride4_base_idx + 1 * hidden_dim;
    const int g_idx = stride4_base_idx + 2 * hidden_dim;
    const int q_g_idx = stride4_base_idx + 3 * hidden_dim;

    const T z = v[z_idx];
    const T r = v[r_idx];
    const T g = v[g_idx];
    const T q_g = v[q_g_idx];

    if (ApplyZoneout) {
        const T mask = zoneout_mask[base_idx];
        dh_inout[base_idx] = (static_cast<T>(1.0) - mask) * dh_total;
        dh_total = mask * dh_total;
        dh_inout[base_idx] += z * dh_total;
    } else {
        dh_inout[base_idx] = z * dh_total;
    }

    // 标准 GRU 反向传播计算
    const T dg = (static_cast<T>(1.0) - z) * dh_total;
    const T dz = (h[base_idx] - g) * dh_total;
    const T dp_g = d_tanh(g) * dg;
    const T dq_g = dp_g * r;
    const T dr = dp_g * q_g;
    const T dp_r = d_sigmoid(r) * dr;
    const T dq_r = dp_r;
    const T dp_z = d_sigmoid(z) * dz;
    const T dq_z = dp_z;

    const int idx = col * (hidden_dim * 3) + row;

    // 初始化输出梯度（这些是相对于 gate_input 的梯度）
    T final_dp_z = dp_z;
    T final_dp_r = dp_r;
    T final_dp_g = dp_g;
    T final_dq_z = dq_z;
    T final_dq_r = dq_r;
    T final_dq_g = dq_g;

    // 应用 gate_mask（STE：门输出被 clamp 时梯度置零）
    if (gate_mask != nullptr) {
        const int z_mask_idx = idx + 0 * hidden_dim;
        const int r_mask_idx = idx + 1 * hidden_dim;
        const int g_mask_idx = idx + 2 * hidden_dim;

        if (gate_mask[z_mask_idx] != 0) {
            final_dp_z = static_cast<T>(0.0);
            final_dq_z = static_cast<T>(0.0);
        }
        if (gate_mask[r_mask_idx] != 0) {
            final_dp_r = static_cast<T>(0.0);
            final_dq_r = static_cast<T>(0.0);
        }
        if (gate_mask[g_mask_idx] != 0) {
            final_dp_g = static_cast<T>(0.0);
            final_dq_g = static_cast<T>(0.0);
        }
    }

    // ========== Rescale 补偿 ==========
    // 前向：gate_input = div_round(linear_output, divisor)
    // 反向：d_linear_output = d_gate_input * divisor
    // dp 传回 weight_ih_linear，dq 传回 weight_hh_linear
    if constexpr (ApplyRescale) {
        // Update gate: dp_z 和 dq_z 分别乘以对应的 rescale 因子
        final_dp_z *= mul_update_ih;
        final_dq_z *= mul_update_hh;
        
        // Reset gate: dp_r 和 dq_r
        final_dp_r *= mul_reset_ih;
        final_dq_r *= mul_reset_hh;
        
        // New gate: dp_g 和 dq_g
        // 注意：new gate 的 hh 路径经过 reset_gate * hh_linear，rescale 更复杂
        final_dp_g *= mul_new_ih;
        final_dq_g *= mul_new_rh;  // r * weight_hh_linear_g 的 rescale
    }

    // 应用 weight_hh_linear_mask（STE：R*h+br 输出被 clamp 时 dq 梯度置零）
    if (weight_hh_linear_mask != nullptr) {
        const int z_mask_idx = idx + 0 * hidden_dim;
        const int r_mask_idx = idx + 1 * hidden_dim;
        const int g_mask_idx = idx + 2 * hidden_dim;

        if (weight_hh_linear_mask[z_mask_idx] != 0) final_dq_z = static_cast<T>(0.0);
        if (weight_hh_linear_mask[r_mask_idx] != 0) final_dq_r = static_cast<T>(0.0);
        if (weight_hh_linear_mask[g_mask_idx] != 0) final_dq_g = static_cast<T>(0.0);
    }

    // 写出 dp 和 dq（现在是相对于 linear_output 的梯度）
    dp_out[idx + 0 * hidden_dim] = final_dp_z;
    dp_out[idx + 1 * hidden_dim] = final_dp_r;
    dp_out[idx + 2 * hidden_dim] = final_dp_g;

    dq_out[idx + 0 * hidden_dim] = final_dq_z;
    dq_out[idx + 1 * hidden_dim] = final_dq_r;
    dq_out[idx + 2 * hidden_dim] = final_dq_g;

    // 累加偏置梯度（应用 bw_mask 和 br_mask）
    T dbw_z = final_dp_z;
    T dbw_r = final_dp_r;
    T dbw_g = final_dp_g;
    T dbr_z = final_dq_z;
    T dbr_r = final_dq_r;
    T dbr_g = final_dq_g;

    if (bw_mask != nullptr) {
        if (bw_mask[row + 0 * hidden_dim] != 0) dbw_z = static_cast<T>(0.0);
        if (bw_mask[row + 1 * hidden_dim] != 0) dbw_r = static_cast<T>(0.0);
        if (bw_mask[row + 2 * hidden_dim] != 0) dbw_g = static_cast<T>(0.0);
    }
    if (br_mask != nullptr) {
        if (br_mask[row + 0 * hidden_dim] != 0) dbr_z = static_cast<T>(0.0);
        if (br_mask[row + 1 * hidden_dim] != 0) dbr_r = static_cast<T>(0.0);
        if (br_mask[row + 2 * hidden_dim] != 0) dbr_g = static_cast<T>(0.0);
    }

    atomicAdd(&dbw_out[row + 0 * hidden_dim], dbw_z);
    atomicAdd(&dbw_out[row + 1 * hidden_dim], dbw_r);
    atomicAdd(&dbw_out[row + 2 * hidden_dim], dbw_g);

    atomicAdd(&dbr_out[row + 0 * hidden_dim], dbr_z);
    atomicAdd(&dbr_out[row + 1 * hidden_dim], dbr_r);
    atomicAdd(&dbr_out[row + 2 * hidden_dim], dbr_g);
}

}  // namespace kernel

namespace gru {

template <typename T>
struct BackwardPassQuant<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[2];
    cudaEvent_t event;
    cudaStream_t sync_stream;
};

template <typename T>
BackwardPassQuant<T>::BackwardPassQuant(
    int batch_size, int input_size, int hidden_size,
    const cublasHandle_t &blas_handle, const cudaStream_t &stream)
    : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->sync_stream = stream;
    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template <typename T>
BackwardPassQuant<T>::~BackwardPassQuant() {
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

template <typename T>
void BackwardPassQuant<T>::setRescaleParam(const GRUQuantParams &params) {
    // 从前向传播参数计算反向传播 rescale 因子
    //
    // 前向：gate_input = div_round(linear_output, divisor) ≈ linear_output / divisor
    //       其中 divisor = 2^(shift_linear - shift_gate_input)
    //
    // 反向（链式法则）：
    //   ∂gate_input/∂linear_output = 1/divisor
    //   d_linear_output = d_gate_input × (1/divisor) = d_gate_input / divisor
    //
    // 所以 mul = 1/divisor = 2^(-(shift_linear - shift_gate_input)) = 2^(shift_gate_input - shift_linear)
    //
    // 注意：取负号！前向除以 divisor，反向也除以 divisor（不是乘以）
    
    // Update gate rescale 因子
    // shift_diff = shift_update_gate_input - shift_weight_ih_linear (取负)
    int shift_diff_update_ih = params.shift_update_gate_input_ - params.shift_weight_ih_linear_;
    int shift_diff_update_hh = params.shift_update_gate_input_ - params.shift_weight_hh_linear_;
    rescale_params_.mul_update_gate_input_to_weight_ih_linear_ = exp2f(static_cast<float>(shift_diff_update_ih));
    rescale_params_.mul_update_gate_input_to_weight_hh_linear_ = exp2f(static_cast<float>(shift_diff_update_hh));
    
    // Reset gate rescale 因子
    int shift_diff_reset_ih = params.shift_reset_gate_input_ - params.shift_weight_ih_linear_;
    int shift_diff_reset_hh = params.shift_reset_gate_input_ - params.shift_weight_hh_linear_;
    rescale_params_.mul_reset_gate_input_to_weight_ih_linear_ = exp2f(static_cast<float>(shift_diff_reset_ih));
    rescale_params_.mul_reset_gate_input_to_weight_hh_linear_ = exp2f(static_cast<float>(shift_diff_reset_hh));
    
    // New gate rescale 因子
    int shift_diff_new_ih = params.shift_new_gate_input_ - params.shift_weight_ih_linear_;
    rescale_params_.mul_new_gate_input_to_weight_ih_linear_ = exp2f(static_cast<float>(shift_diff_new_ih));
    
    // r * weight_hh_linear_g 的 rescale：
    // 前向：reset_mul_hh = r * hh_linear，然后 rescale 到 new_gate_input
    // shift_reset_mul_hh = shift_reset_gate_output + shift_weight_hh_linear
    // shift_diff = shift_new_gate_input - shift_reset_mul_hh (取负)
    int shift_reset_mul_hh = params.shift_reset_gate_output_ + params.shift_weight_hh_linear_;
    int shift_diff_new_rh = params.shift_new_gate_input_ - shift_reset_mul_hh;
    rescale_params_.mul_new_gate_input_to_reset_mul_hh_ = exp2f(static_cast<float>(shift_diff_new_rh));
    
    // Hidden state 更新 rescale 因子
    // (1-u) * n 和 u * h 到 h 的 rescale
    // 前向：h_new = div_round((1-u)*n, divisor_new) + div_round(u*h_old, divisor_old)
    int shift_update_new = params.shift_update_gate_output_ + params.shift_new_gate_output_;
    int shift_update_old = params.shift_update_gate_output_ + params.shift_h_;
    int shift_diff_h_new = params.shift_h_ - shift_update_new;
    int shift_diff_h_old = params.shift_h_ - shift_update_old;
    rescale_params_.mul_h_to_update_new_ = exp2f(static_cast<float>(shift_diff_h_new));
    rescale_params_.mul_h_to_update_old_ = exp2f(static_cast<float>(shift_diff_h_old));
    
    rescale_params_set_ = true;
}

template <typename T>
void BackwardPassQuant<T>::IterateInternal(
    const T *R_t, const T *h, const T *v, const T *dh_new,
    T *dbw, T *dbr, T *dh, T *dp, T *dq,
    const T *zoneout_mask,
    const uint8_t *weight_hh_linear_mask,
    const uint8_t *gate_mask,
    const uint8_t *h_mask,
    const uint8_t *bw_mask,
    const uint8_t *br_mask) {
    
    const T alpha = static_cast<T>(1.0);
    const T beta_sum = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    // Compute launch configuration for pointwise operations kernel
    const dim3 blockDim(32, 16);
    const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y);

    const bool apply_zoneout = (zoneout_mask != nullptr);
    const bool apply_rescale = rescale_params_set_;
    
    // Rescale 参数
    const T mul_update_ih = static_cast<T>(rescale_params_.mul_update_gate_input_to_weight_ih_linear_);
    const T mul_update_hh = static_cast<T>(rescale_params_.mul_update_gate_input_to_weight_hh_linear_);
    const T mul_reset_ih = static_cast<T>(rescale_params_.mul_reset_gate_input_to_weight_ih_linear_);
    const T mul_reset_hh = static_cast<T>(rescale_params_.mul_reset_gate_input_to_weight_hh_linear_);
    const T mul_new_ih = static_cast<T>(rescale_params_.mul_new_gate_input_to_weight_ih_linear_);
    const T mul_new_rh = static_cast<T>(rescale_params_.mul_new_gate_input_to_reset_mul_hh_);
    const T mul_h_new = static_cast<T>(rescale_params_.mul_h_to_update_new_);
    const T mul_h_old = static_cast<T>(rescale_params_.mul_h_to_update_old_);

    // 根据 zoneout 和 rescale 选择正确的 kernel 实例
    if (apply_rescale) {
        if (apply_zoneout) {
            kernel::PointwiseOperationsQuant<T, true, true>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size, h, v, dh_new,
                    dbw, dbr, dh, dp, dq,
                    zoneout_mask, weight_hh_linear_mask, gate_mask, h_mask,
                    bw_mask, br_mask,
                    mul_update_ih, mul_update_hh, mul_reset_ih, mul_reset_hh,
                    mul_new_ih, mul_new_rh, mul_h_new, mul_h_old);
        } else {
            kernel::PointwiseOperationsQuant<T, false, true>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size, h, v, dh_new,
                    dbw, dbr, dh, dp, dq,
                    nullptr, weight_hh_linear_mask, gate_mask, h_mask,
                    bw_mask, br_mask,
                    mul_update_ih, mul_update_hh, mul_reset_ih, mul_reset_hh,
                    mul_new_ih, mul_new_rh, mul_h_new, mul_h_old);
        }
    } else {
        // 不应用 rescale（兼容非量化训练）
        if (apply_zoneout) {
            kernel::PointwiseOperationsQuant<T, true, false>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size, h, v, dh_new,
                    dbw, dbr, dh, dp, dq,
                    zoneout_mask, weight_hh_linear_mask, gate_mask, h_mask,
                    bw_mask, br_mask,
                    static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0),
                    static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0));
        } else {
            kernel::PointwiseOperationsQuant<T, false, false>
                <<<gridDim, blockDim, 0, stream1>>>(
                    batch_size, hidden_size, h, v, dh_new,
                    dbw, dbr, dh, dp, dq,
                    nullptr, weight_hh_linear_mask, gate_mask, h_mask,
                    bw_mask, br_mask,
                    static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0),
                    static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0), static_cast<T>(1.0));
        }
    }
    cudaEventRecord(event, stream1);

    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size * 3,
                  &alpha, R_t, hidden_size, dq, hidden_size * 3,
                  &beta_sum, dh, hidden_size);
}

/// @brief 辅助函数：对 dp 应用 weight_ih_linear_mask（在所有时间步完成后调用）
template <typename T>
static void ApplyMaskToBuffer(T *buffer, const uint8_t *mask, size_t size, cudaStream_t stream) {
    if (mask == nullptr) return;
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    kernel::ApplyMaskKernel<T><<<grid_size, block_size, 0, stream>>>(buffer, mask, size);
}

template <typename T>
void BackwardPassQuant<T>::Run(
    int steps, const T *W_t, const T *R_t, const T *bw, const T *br,
    const T *x_t, const T *h, const T *v, const T *dh_new,
    T *dx, T *dW, T *dR, T *dbw, T *dbr, T *dh, T *dp, T *dq,
    const T *zoneout_mask,
    // QAT masks
    const uint8_t *x_mask,
    const uint8_t *h0_mask,
    const uint8_t *W_mask,
    const uint8_t *R_mask,
    const uint8_t *bw_mask,
    const uint8_t *br_mask,
    const uint8_t *weight_ih_linear_mask,
    const uint8_t *weight_hh_linear_mask,
    const uint8_t *gate_mask,
    const uint8_t *h_mask) {
    
    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const T alpha = static_cast<T>(1.0);
    const T beta_sum = static_cast<T>(1.0);
    const T beta_assign = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;
    
    // 反向遍历所有时间步
    for (int i = steps - 1; i >= 0; --i) {
        IterateInternal(
            R_t,
            h + i * NH,
            v + i * NH * 4,
            dh_new + (i + 1) * NH,
            dbw, dbr, dh,
            dp + i * NH3,
            dq + i * NH3,
            zoneout_mask ? zoneout_mask + i * NH : nullptr,
            weight_hh_linear_mask ? weight_hh_linear_mask + i * NH3 : nullptr,
            gate_mask ? gate_mask + i * NH3 : nullptr,
            h_mask ? h_mask + i * NH : nullptr,
            bw_mask,  // bw_mask 是全局的，不按时间步偏移
            br_mask); // br_mask 是全局的，不按时间步偏移
    }

    // 对 dp 应用 weight_ih_linear_mask（W*x+bw 输出被 clamp 时，dp 梯度置零）
    if (weight_ih_linear_mask != nullptr) {
        ApplyMaskToBuffer(dp, weight_ih_linear_mask, static_cast<size_t>(steps) * NH3, stream1);
    }

    // Wait for pointwise operations to complete
    cudaStreamWaitEvent(stream2, event, 0);

    // dx = W_t * dp（输入梯度）
    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  input_size, batch_size * steps, hidden_size * 3,
                  &alpha, W_t, input_size, dp, hidden_size * 3,
                  &beta_assign, dx, input_size);
    
    // 对 dx 应用 x_mask（输入量化被 clamp 时，dx 梯度置零）
    if (x_mask != nullptr) {
        cudaStreamSynchronize(stream2);  // 等待 GEMM 完成
        const size_t dx_size = static_cast<size_t>(steps) * batch_size * input_size;
        ApplyMaskToBuffer(dx, x_mask, dx_size, stream2);
    }

    // dR = dq * h^T（循环权重梯度）
    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                  hidden_size * 3, hidden_size, batch_size * steps,
                  &alpha, dq, hidden_size * 3, h, hidden_size,
                  &beta_sum, dR, hidden_size * 3);
    
    // 对 dR 应用 R_mask（循环权重量化被 clamp 时，dR 梯度置零）
    if (R_mask != nullptr) {
        cudaStreamSynchronize(stream2);  // 等待 GEMM 完成
        const size_t dR_size = static_cast<size_t>(hidden_size) * hidden_size * 3;
        ApplyMaskToBuffer(dR, R_mask, dR_size, stream2);
    }

    // dW = dp * x_t^T（输入权重梯度）
    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  hidden_size * 3, input_size, batch_size * steps,
                  &alpha, dp, hidden_size * 3, x_t, batch_size * steps,
                  &beta_sum, dW, hidden_size * 3);
    
    // 对 dW 应用 W_mask（输入权重量化被 clamp 时，dW 梯度置零）
    if (W_mask != nullptr) {
        cudaStreamSynchronize(stream1);  // 等待 GEMM 完成
        const size_t dW_size = static_cast<size_t>(input_size) * hidden_size * 3;
        ApplyMaskToBuffer(dW, W_mask, dW_size, stream1);
    }

    // 对 dh（初始隐状态梯度）应用 h0_mask
    if (h0_mask != nullptr) {
        const size_t dh_size = static_cast<size_t>(batch_size) * hidden_size;
        ApplyMaskToBuffer(dh, h0_mask, dh_size, stream1);
    }

    cublasSetStream(blas_handle, save_stream);
}

// 显式模板实例化
template struct BackwardPassQuant<float>;
template struct BackwardPassQuant<double>;

}  // namespace gru
