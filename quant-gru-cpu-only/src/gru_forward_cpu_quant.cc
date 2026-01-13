#include "gru_quant_cpu.h"

namespace cpu {

struct ForwardPassQuantCPU::PrivateData {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
};

ForwardPassQuantCPU::ForwardPassQuantCPU(bool training, int batch_size, int input_size, int hidden_size)
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

    for (int m = 0; m < hidden3; m++) {
        int64_t sum = 0;
        for (int k = 0; k < input_size; k++)
            sum += static_cast<int64_t>(W[k * hidden3 + m]);
        W_sum_mul_x_zp_[m] = sum * linear_params_.zp_x_;
    }

    for (int m = 0; m < hidden3; m++) {
        int64_t sum = 0;
        for (int k = 0; k < hidden_size; k++)
            sum += static_cast<int64_t>(R[k * hidden3 + m]);
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

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < hidden3; m++) {
            int64_t acc = 0;
            for (int k = 0; k < input_size; k++)
                acc += static_cast<int64_t>(W[k * hidden3 + m]) * static_cast<int64_t>(x[n * input_size + k]);
            
            int32_t result = static_cast<int32_t>(rshift_round(acc - W_sum_mul_x_zp_[m], 
                             linear_params_.shift_gemm_x_to_weight_ih_linear_[m])) + gate_params_.zp_weight_ih_linear_;
            result += rshift_round(bw[m], linear_params_.shift_bw_to_weight_ih_linear_[m]);
            tmp_weight_ih_linear_[n * hidden3 + m] = clamp_by_bitwidth(result, gate_params_.bitwidth_config_.weight_ih_linear_);
        }
    }
}

void ForwardPassQuantCPU::ComputeLinearH(const int32_t *R, const int32_t *h, const int32_t *br) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;

    for (int n = 0; n < batch_size; n++) {
        for (int m = 0; m < hidden3; m++) {
            int64_t acc = 0;
            for (int k = 0; k < hidden_size; k++)
                acc += static_cast<int64_t>(R[k * hidden3 + m]) * static_cast<int64_t>(h[n * hidden_size + k]);
            
            int32_t result = static_cast<int32_t>(rshift_round(acc - R_sum_mul_h_zp_[m],
                             linear_params_.shift_gemm_h_to_weight_hh_linear_[m])) + gate_params_.zp_weight_hh_linear_;
            result += rshift_round(br[m], linear_params_.shift_br_to_weight_hh_linear_[m]);
            tmp_weight_hh_linear_[n * hidden3 + m] = clamp_by_bitwidth(result, gate_params_.bitwidth_config_.weight_hh_linear_);
        }
    }
}

void ForwardPassQuantCPU::IterateInternal(const int32_t *R, const int32_t *br, const int32_t *h,
                                          int32_t *h_out, int32_t *v, const int32_t *cur_ih,
                                          float zoneout_prob, const int32_t *zoneout_mask) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    ComputeLinearH(R, h, br);

    for (int col = 0; col < batch_size; col++) {
        for (int row = 0; row < hidden_size; row++) {
            const int idx = col * (hidden_size * 3) + row;
            const int out_idx = col * hidden_size + row;

            int32_t z = computeUpdateGate(cur_ih[idx], tmp_weight_hh_linear_[idx], gate_params_);
            int32_t r = computeResetGate(cur_ih[idx + hidden_size], tmp_weight_hh_linear_[idx + hidden_size], gate_params_);
            
            int32_t hh_g;
            int32_t n = computeNewGate(cur_ih[idx + 2*hidden_size], tmp_weight_hh_linear_[idx + 2*hidden_size], r, gate_params_, hh_g);

            if (training && v) {
                const int v_idx = col * (hidden_size * 4) + row;
                v[v_idx] = z;
                v[v_idx + hidden_size] = r;
                v[v_idx + 2*hidden_size] = n;
                v[v_idx + 3*hidden_size] = hh_g;
            }

            int32_t h_new = computeHiddenState(z, n, h[out_idx], gate_params_);
            if (zoneout_prob > 0.0f && zoneout_mask && zoneout_mask[out_idx])
                h_new = h[out_idx];
            h_out[out_idx] = h_new;
        }
    }
}

void ForwardPassQuantCPU::setRescaleParam(const GRUQuantParams &p) {
    const int channel = p.hidden_ * 3;

    linear_params_.zp_x_ = p.zp_x_;
    linear_params_.zp_h_ = p.zp_h_;
    linear_params_.shift_gemm_x_to_weight_ih_linear_.resize(channel);
    linear_params_.shift_bw_to_weight_ih_linear_.resize(channel);
    linear_params_.shift_gemm_h_to_weight_hh_linear_.resize(channel);
    linear_params_.shift_br_to_weight_hh_linear_.resize(channel);

    for (int i = 0; i < channel; ++i) {
        linear_params_.shift_gemm_x_to_weight_ih_linear_[i] = (p.shift_W_[i] + p.shift_x_) - p.shift_weight_ih_linear_;
        linear_params_.shift_gemm_h_to_weight_hh_linear_[i] = (p.shift_R_[i] + p.shift_h_) - p.shift_weight_hh_linear_;
        linear_params_.shift_bw_to_weight_ih_linear_[i] = p.shift_bw_[i] - p.shift_weight_ih_linear_;
        linear_params_.shift_br_to_weight_hh_linear_[i] = p.shift_br_[i] - p.shift_weight_hh_linear_;
    }

    gate_params_.zp_weight_ih_linear_ = p.zp_weight_ih_linear_;
    gate_params_.zp_weight_hh_linear_ = p.zp_weight_hh_linear_;
    gate_params_.zp_h_ = p.zp_h_;

    gate_params_.zp_update_gate_input_ = p.zp_update_gate_input_;
    gate_params_.zp_update_gate_output_ = p.zp_update_gate_output_;
    gate_params_.shift_weight_ih_linear_to_update_gate_input_ = p.shift_weight_ih_linear_ - p.shift_update_gate_input_;
    gate_params_.shift_weight_hh_linear_to_update_gate_input_ = p.shift_weight_hh_linear_ - p.shift_update_gate_input_;

    gate_params_.zp_reset_gate_input_ = p.zp_reset_gate_input_;
    gate_params_.zp_reset_gate_output_ = p.zp_reset_gate_output_;
    gate_params_.shift_weight_ih_linear_to_reset_gate_input_ = p.shift_weight_ih_linear_ - p.shift_reset_gate_input_;
    gate_params_.shift_weight_hh_linear_to_reset_gate_input_ = p.shift_weight_hh_linear_ - p.shift_reset_gate_input_;

    gate_params_.zp_new_gate_input_ = p.zp_new_gate_input_;
    gate_params_.zp_new_gate_output_ = p.zp_new_gate_output_;
    gate_params_.shift_reset_gate_mul_hh_to_mul_reset_hidden_ = (p.shift_reset_gate_output_ + p.shift_weight_hh_linear_) - p.shift_mul_reset_hidden_;
    gate_params_.zp_mul_reset_hidden_ = p.zp_mul_reset_hidden_;
    gate_params_.shift_weight_ih_linear_to_new_gate_input_ = p.shift_weight_ih_linear_ - p.shift_new_gate_input_;
    gate_params_.shift_mul_reset_hidden_to_new_gate_input_ = p.shift_mul_reset_hidden_ - p.shift_new_gate_input_;

    gate_params_.quant_one_in_update_gate_scale_ = rshift_round(1, -p.shift_update_gate_output_) + p.zp_update_gate_output_;
    gate_params_.zp_mul_new_contribution_ = p.zp_mul_new_contribution_;
    gate_params_.shift_update_new_to_mul_new_contribution_ = (p.shift_update_gate_output_ + p.shift_new_gate_output_) - p.shift_mul_new_contribution_;
    gate_params_.zp_mul_old_contribution_ = p.zp_mul_old_contribution_;
    gate_params_.shift_update_h_to_mul_old_contribution_ = (p.shift_update_gate_output_ + p.shift_h_) - p.shift_mul_old_contribution_;
    gate_params_.shift_mul_new_contribution_to_h_ = p.shift_mul_new_contribution_ - p.shift_h_;
    gate_params_.shift_mul_old_contribution_to_h_ = p.shift_mul_old_contribution_ - p.shift_h_;

    gate_params_.bitwidth_config_ = p.bitwidth_config_;
    gate_params_.sigmoid_update_gate_lut_ = generate_sigmoid_lut(
        p.shift_update_gate_input_, p.zp_update_gate_input_, p.shift_update_gate_output_, p.zp_update_gate_output_,
        p.bitwidth_config_.update_gate_input_, p.bitwidth_config_.update_gate_output_);
    gate_params_.sigmoid_reset_gate_lut_ = generate_sigmoid_lut(
        p.shift_reset_gate_input_, p.zp_reset_gate_input_, p.shift_reset_gate_output_, p.zp_reset_gate_output_,
        p.bitwidth_config_.reset_gate_input_, p.bitwidth_config_.reset_gate_output_);
    gate_params_.tanh_new_gate_lut_ = generate_tanh_lut(
        p.shift_new_gate_input_, p.zp_new_gate_input_, p.shift_new_gate_output_, p.zp_new_gate_output_,
        p.bitwidth_config_.new_gate_input_, p.bitwidth_config_.new_gate_output_);
}

void ForwardPassQuantCPU::Run(int steps, const int32_t *W, const int32_t *R,
                               const int32_t *bw, const int32_t *br, const int32_t *x,
                               int32_t *h, int32_t *v, float zoneout_prob, const int32_t *zoneout_mask) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    EnsureBuffersAllocated(steps);
    PrecomputeWeightSums(W, R);
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
