// ============================================================================
// gru_forward_cpu_quant.cc - 纯 C++ 定点 GRU 前向传播实现
// ============================================================================
//
// 设计原则:
//   - 与 GPU 版本保持数值一致性
//   - 所有量化值使用 int32_t 统一存储，通过 bitwidth_config_ 控制实际位宽
//   - 复用 quantize_lut_types.h 中的 LUT 结构和 generate_*_lut 函数
//   - 复用 quantize_ops_helper.h 中的通用函数
//   - 支持 OpenMP 并行加速
//
// ============================================================================

#include "gru_quant_cpu.h"

#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

// MSVC OpenMP 只支持 2.0 版本，不完全支持 collapse 子句
// 使用条件编译来区分 MSVC 和 GCC/Clang
#if defined(_MSC_VER)
#define OMP_PARALLEL_FOR_2D _Pragma("omp parallel for")
#else
#define OMP_PARALLEL_FOR_2D _Pragma("omp parallel for collapse(2)")
#endif

namespace cpu {

// ============================================================================
// 1. LUT 查找函数
// ============================================================================
// 注意：find_segment, piecewise_linear_raw, piecewise_linear, clamp_by_bitwidth
// 已统一定义在 quantize_ops_helper.h 中，使用 __host__ __device__ 标记，
// 可在 CPU 和 GPU 上共用。

// ============================================================================
// 2. GRU Gate Functions - 使用 quantize_ops_helper.h 中的模板函数
// ============================================================================

// computeUpdateGate, computeResetGate, computeNewGate, computeHiddenState 现在使用 quantize_ops_helper.h 中的模板函数
// 通过模板参数 RescaleT 支持 GateQuantParams（门计算参数）

// ============================================================================
// 3. ForwardPassQuantCPU 实现
// ============================================================================

struct ForwardPassQuantCPU::PrivateData {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
};

ForwardPassQuantCPU::ForwardPassQuantCPU(bool training, int batch_size,
                                         int input_size, int hidden_size)
    : data_(std::make_unique<PrivateData>()) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
}

ForwardPassQuantCPU::~ForwardPassQuantCPU() = default;

void ForwardPassQuantCPU::EnsureBuffersAllocated(int steps) {
    if (steps <= max_steps_) return;

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;

    tmp_weight_ih_linear_.resize(static_cast<size_t>(hidden3) * steps * batch_size);
    tmp_weight_hh_linear_.resize(static_cast<size_t>(hidden3) * batch_size);

    if (W_sum_mul_x_zp_.empty()) {
        W_sum_mul_x_zp_.resize(hidden3);
        R_sum_mul_h_zp_.resize(hidden3);
    }

    max_steps_ = steps;
    weight_sums_computed_ = false;
}

void ForwardPassQuantCPU::PrecomputeWeightSums(const int32_t *W, const int32_t *R) {
    if (cached_W_ != W || cached_R_ != R) {
        weight_sums_computed_ = false;
        cached_W_ = W;
        cached_R_ = R;
    }

    if (weight_sums_computed_) return;

    const int hidden_size = data_->hidden_size;
    const int input_size = data_->input_size;
    const int hidden3 = hidden_size * 3;

#pragma omp parallel for
    for (int m = 0; m < hidden3; m++) {
        int64_t sum = 0;
        for (int k = 0; k < input_size; k++) {
            sum += static_cast<int64_t>(W[k * hidden3 + m]);
        }
        // 与 GPU 保持一致：不在此处移位，只计算 W_sum * zp_x
        W_sum_mul_x_zp_[m] = sum * linear_params_.zp_x_;
    }

#pragma omp parallel for
    for (int m = 0; m < hidden3; m++) {
        int64_t sum = 0;
        for (int k = 0; k < hidden_size; k++) {
            sum += static_cast<int64_t>(R[k * hidden3 + m]);
        }
        // 与 GPU 保持一致：不在此处移位，只计算 R_sum * zp_h
        R_sum_mul_h_zp_[m] = sum * linear_params_.zp_h_;
    }

    weight_sums_computed_ = true;
}

void ForwardPassQuantCPU::ComputeLinearX(const int32_t *W, const int32_t *x, const int32_t *bw, int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;
    const int N = steps * batch_size;

OMP_PARALLEL_FOR_2D
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < hidden3; m++) {
            // GEMM: W*x
            int64_t acc = 0;
            for (int k = 0; k < input_size; k++) {
                acc += static_cast<int64_t>(W[k * hidden3 + m]) *
                       static_cast<int64_t>(x[n * input_size + k]);
            }
            int64_t gemm_val = acc - W_sum_mul_x_zp_[m];
            int32_t gemm_result = static_cast<int32_t>(
                rshift_round(gemm_val, linear_params_.shift_gemm_x_to_weight_ih_linear_[m])) +
                gate_params_.zp_weight_ih_linear_;
            
            // 添加 bias: bw 移位到 weight_ih_linear 空间
            int32_t bias_rescaled = rshift_round(bw[m], linear_params_.shift_bw_to_weight_ih_linear_[m]);
            int32_t result = gemm_result + bias_rescaled;
            tmp_weight_ih_linear_[n * hidden3 + m] = clamp_by_bitwidth(result, gate_params_.bitwidth_config_.weight_ih_linear_);
        }
    }
}

void ForwardPassQuantCPU::ComputeLinearH(const int32_t *R, const int32_t *h, const int32_t *br) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;

OMP_PARALLEL_FOR_2D
    for (int n = 0; n < batch_size; n++) {
        for (int m = 0; m < hidden3; m++) {
            // GEMM: R*h
            int64_t acc = 0;
            for (int k = 0; k < hidden_size; k++) {
                acc += static_cast<int64_t>(R[k * hidden3 + m]) *
                       static_cast<int64_t>(h[n * hidden_size + k]);
            }
            int64_t gemm_val = acc - R_sum_mul_h_zp_[m];
            int32_t gemm_result = static_cast<int32_t>(
                rshift_round(gemm_val, linear_params_.shift_gemm_h_to_weight_hh_linear_[m])) +
                gate_params_.zp_weight_hh_linear_;
            
            // 添加 bias: br 移位到 weight_hh_linear 空间
            int32_t bias_rescaled = rshift_round(br[m], linear_params_.shift_br_to_weight_hh_linear_[m]);
            int32_t result = gemm_result + bias_rescaled;
            tmp_weight_hh_linear_[n * hidden3 + m] = clamp_by_bitwidth(result, gate_params_.bitwidth_config_.weight_hh_linear_);
        }
    }
}

void ForwardPassQuantCPU::IterateInternal(const int32_t *R, const int32_t *br,
                                          const int32_t *h, int32_t *h_out, int32_t *v,
                                          const int32_t *cur_weight_ih_linear,
                                          float zoneout_prob,
                                          const int32_t *zoneout_mask) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    // 计算 R*h + br GEMM+bias 融合
    ComputeLinearH(R, h, br);

OMP_PARALLEL_FOR_2D
    for (int col = 0; col < batch_size; col++) {
        for (int row = 0; row < hidden_size; row++) {
            const int weight_idx = col * (hidden_size * 3) + row;
            const int output_idx = col * hidden_size + row;
            const int update_idx = weight_idx + 0 * hidden_size;
            const int reset_idx = weight_idx + 1 * hidden_size;
            const int new_idx = weight_idx + 2 * hidden_size;

            // GEMM+bias 融合版本：直接使用 Wx_bw 和 Rh_br
            const int32_t update_gate = computeUpdateGate(cur_weight_ih_linear[update_idx], tmp_weight_hh_linear_[update_idx], gate_params_);
            const int32_t reset_gate = computeResetGate(cur_weight_ih_linear[reset_idx], tmp_weight_hh_linear_[reset_idx], gate_params_);

            int32_t weight_hh_linear_g;
            const int32_t new_gate = computeNewGate(cur_weight_ih_linear[new_idx], tmp_weight_hh_linear_[new_idx], reset_gate, gate_params_, weight_hh_linear_g);

            if (training && v != nullptr) {
                const int base_v_idx = col * (hidden_size * 4) + row;
                v[base_v_idx + 0 * hidden_size] = update_gate;
                v[base_v_idx + 1 * hidden_size] = reset_gate;
                v[base_v_idx + 2 * hidden_size] = new_gate;
                v[base_v_idx + 3 * hidden_size] = weight_hh_linear_g;
            }

            int32_t cur_h = computeHiddenState(update_gate, new_gate, h[output_idx], gate_params_);

            if (zoneout_prob > 0.0f && zoneout_mask != nullptr) {
                if (zoneout_mask[output_idx] != 0) cur_h = h[output_idx];
            }

            h_out[output_idx] = cur_h;
        }
    }
}

// setRescaleParam: 直接复用 quantize_lut_types.h 中声明的 generate_*_lut 函数
void ForwardPassQuantCPU::setRescaleParam(const GRUQuantParams &parms) {
    const int channel = parms.hidden_ * 3;

    // ==================== Linear 层参数（per-channel）====================
    linear_params_.zp_x_ = parms.zp_x_;
    linear_params_.zp_h_ = parms.zp_h_;

    // 计算 per-channel 移位参数
    linear_params_.shift_gemm_x_to_weight_ih_linear_.resize(channel);
    linear_params_.shift_bw_to_weight_ih_linear_.resize(channel);
    linear_params_.shift_gemm_h_to_weight_hh_linear_.resize(channel);
    linear_params_.shift_br_to_weight_hh_linear_.resize(channel);

    for (int idx = 0; idx < channel; ++idx) {
        linear_params_.shift_gemm_x_to_weight_ih_linear_[idx] = (parms.shift_W_[idx] + parms.shift_x_) - parms.shift_weight_ih_linear_;
        linear_params_.shift_gemm_h_to_weight_hh_linear_[idx] = (parms.shift_R_[idx] + parms.shift_h_) - parms.shift_weight_hh_linear_;
        linear_params_.shift_bw_to_weight_ih_linear_[idx] = parms.shift_bw_[idx] - parms.shift_weight_ih_linear_;
        linear_params_.shift_br_to_weight_hh_linear_[idx] = parms.shift_br_[idx] - parms.shift_weight_hh_linear_;
    }

#ifdef DEBUG
    linear_params_.shift_bw_ = parms.shift_bw_;
    linear_params_.shift_br_ = parms.shift_br_;
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

    // new gate
    gate_params_.zp_new_gate_input_ = parms.zp_new_gate_input_;
    gate_params_.zp_new_gate_output_ = parms.zp_new_gate_output_;
    gate_params_.shift_reset_gate_mul_hh_to_mul_reset_hidden_ =
        (parms.shift_reset_gate_output_ + parms.shift_weight_hh_linear_) - parms.shift_mul_reset_hidden_;
    gate_params_.zp_mul_reset_hidden_ = parms.zp_mul_reset_hidden_;
    gate_params_.shift_weight_ih_linear_to_new_gate_input_ = parms.shift_weight_ih_linear_ - parms.shift_new_gate_input_;
    gate_params_.shift_mul_reset_hidden_to_new_gate_input_ = parms.shift_mul_reset_hidden_ - parms.shift_new_gate_input_;

    // h_new
    gate_params_.quant_one_in_update_gate_scale_ = rshift_round(1, -parms.shift_update_gate_output_) + parms.zp_update_gate_output_;
    gate_params_.zp_mul_new_contribution_ = parms.zp_mul_new_contribution_;
    gate_params_.shift_update_new_to_mul_new_contribution_ =
        (parms.shift_update_gate_output_ + parms.shift_new_gate_output_) - parms.shift_mul_new_contribution_;
    gate_params_.zp_mul_old_contribution_ = parms.zp_mul_old_contribution_;
    gate_params_.shift_update_h_to_mul_old_contribution_ =
        (parms.shift_update_gate_output_ + parms.shift_h_) - parms.shift_mul_old_contribution_;
    gate_params_.shift_mul_new_contribution_to_h_ = parms.shift_mul_new_contribution_ - parms.shift_h_;
    gate_params_.shift_mul_old_contribution_to_h_ = parms.shift_mul_old_contribution_ - parms.shift_h_;

    // 位宽配置和 LUT
    gate_params_.bitwidth_config_ = parms.bitwidth_config_;
    gate_params_.sigmoid_update_gate_lut_ = generate_sigmoid_lut(
        parms.shift_update_gate_input_, parms.zp_update_gate_input_, parms.shift_update_gate_output_, parms.zp_update_gate_output_,
        parms.bitwidth_config_.update_gate_input_, parms.bitwidth_config_.update_gate_output_);
    gate_params_.sigmoid_reset_gate_lut_ = generate_sigmoid_lut(
        parms.shift_reset_gate_input_, parms.zp_reset_gate_input_, parms.shift_reset_gate_output_, parms.zp_reset_gate_output_,
        parms.bitwidth_config_.reset_gate_input_, parms.bitwidth_config_.reset_gate_output_);
    gate_params_.tanh_new_gate_lut_ = generate_tanh_lut(
        parms.shift_new_gate_input_, parms.zp_new_gate_input_, parms.shift_new_gate_output_, parms.zp_new_gate_output_,
        parms.bitwidth_config_.new_gate_input_, parms.bitwidth_config_.new_gate_output_);

#ifdef DEBUG
    gate_params_.test = parms;
#endif
}

void ForwardPassQuantCPU::Run(int steps, const int32_t *W, const int32_t *R,
                               const int32_t *bw, const int32_t *br, const int32_t *x,
                               int32_t *h, int32_t *v, float zoneout_prob,
                               const int32_t *zoneout_mask) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    EnsureBuffersAllocated(steps);
    PrecomputeWeightSums(W, R);
    
    // 计算 W*x + bw GEMM+bias 融合
    ComputeLinearX(W, x, bw, steps);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, br, h + i * NH, h + (i + 1) * NH,
                        v ? v + i * NH * 4 : nullptr,
                        tmp_weight_ih_linear_.data() + i * NH3,
                        zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }
}

}  // namespace cpu
