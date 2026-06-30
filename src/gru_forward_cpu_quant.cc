// ============================================================================
// gru_forward_cpu_quant.cc - 纯 C++ 定点 GRU 前向传播实现
// ============================================================================
//
// 设计原则:
//   - 与 GPU 版本保持数值一致性
//   - 所有量化值使用 int32_t 统一存储，通过 bitwidth_config_ 控制实际位宽
//   - 复用 quantize_lut_types.h 中的 LUT 结构和 generate_*_lut 函数
//   - 复用 quantize_ops_helper.h 中的通用函数
//
// ============================================================================

#include "gru_quant_cpu.h"
#include "quantize_ops_helper.h"  // makeRescale* 及 GRU 门模板/通用算子

#include <cstdio>
#include <stdexcept>

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

    const int32_t zp_x = use_pot2_ ? linear_params_pot2_.zp_x_ : linear_params_mshift_.zp_x_;
    const int32_t zp_h = use_pot2_ ? linear_params_pot2_.zp_h_ : linear_params_mshift_.zp_h_;

    for (int m = 0; m < hidden3; m++) {
        int64_t sum = 0;
        for (int k = 0; k < input_size; k++) {
            sum += static_cast<int64_t>(W[k * hidden3 + m]);
        }
        // 与 GPU 保持一致：不在此处移位，只计算 W_sum * zp_x
        W_sum_mul_x_zp_[m] = sum * zp_x;
    }

    for (int m = 0; m < hidden3; m++) {
        int64_t sum = 0;
        for (int k = 0; k < hidden_size; k++) {
            sum += static_cast<int64_t>(R[k * hidden3 + m]);
        }
        // 与 GPU 保持一致：不在此处移位，只计算 R_sum * zp_h
        R_sum_mul_h_zp_[m] = sum * zp_h;
    }

    weight_sums_computed_ = true;
}

template <class R>
void ForwardPassQuantCPU::ComputeLinearX(const LinearQuantParamsCPUT<R> &lp, const GateQuantParamsT<R> &gp,
                                         const int32_t *W, const int32_t *x, const int32_t *bw, int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;
    const int N = steps * batch_size;

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < hidden3; m++) {
            // GEMM: W*x
            int64_t acc = 0;
            for (int k = 0; k < input_size; k++) {
                acc += static_cast<int64_t>(W[k * hidden3 + m]) *
                       static_cast<int64_t>(x[n * input_size + k]);
            }
            int64_t gemm_val = acc - W_sum_mul_x_zp_[m];

            // bias 先 rescale 到 GEMM 累加器 scale，再和 GEMM 结果一起 rescale 到 Linear scale
            const int64_t bias_shifted =
                applyRescale(static_cast<int64_t>(bw[m]), lp.rescale_bw_to_weight_ih_linear_[m]);
            const int64_t gemm_plus_bias_rescaled =
                applyRescale(gemm_val + bias_shifted, lp.rescale_gemm_x_to_weight_ih_linear_[m]);
            int32_t result = static_cast<int32_t>(gemm_plus_bias_rescaled) + gp.zp_weight_ih_linear_;
            tmp_weight_ih_linear_[n * hidden3 + m] = clamp_by_bitwidth(result, gp.bitwidth_config_.weight_ih_linear_);
        }
    }
}

template <class R>
void ForwardPassQuantCPU::ComputeLinearH(const LinearQuantParamsCPUT<R> &lp, const GateQuantParamsT<R> &gp,
                                         const int32_t *R_w, const int32_t *h, const int32_t *br) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;

    for (int n = 0; n < batch_size; n++) {
        for (int m = 0; m < hidden3; m++) {
            // GEMM: R*h
            int64_t acc = 0;
            for (int k = 0; k < hidden_size; k++) {
                acc += static_cast<int64_t>(R_w[k * hidden3 + m]) *
                       static_cast<int64_t>(h[n * hidden_size + k]);
            }
            int64_t gemm_val = acc - R_sum_mul_h_zp_[m];

            const int64_t bias_shifted =
                applyRescale(static_cast<int64_t>(br[m]), lp.rescale_br_to_weight_hh_linear_[m]);
            const int64_t gemm_plus_bias_rescaled =
                applyRescale(gemm_val + bias_shifted, lp.rescale_gemm_h_to_weight_hh_linear_[m]);
            int32_t result = static_cast<int32_t>(gemm_plus_bias_rescaled) + gp.zp_weight_hh_linear_;
            tmp_weight_hh_linear_[n * hidden3 + m] = clamp_by_bitwidth(result, gp.bitwidth_config_.weight_hh_linear_);
        }
    }
}

template <class R>
void ForwardPassQuantCPU::IterateInternal(const LinearQuantParamsCPUT<R> &lp, const GateQuantParamsT<R> &gp,
                                          const int32_t *R_w, const int32_t *br,
                                          const int32_t *h, int32_t *h_out, int32_t *v,
                                          const int32_t *cur_weight_ih_linear,
                                          float zoneout_prob,
                                          const int32_t *zoneout_mask) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    // 计算 R*h + br GEMM+bias 融合
    ComputeLinearH<R>(lp, gp, R_w, h, br);

    for (int col = 0; col < batch_size; col++) {
        for (int row = 0; row < hidden_size; row++) {
            const int weight_idx = col * (hidden_size * 3) + row;
            const int output_idx = col * hidden_size + row;
            const int update_idx = weight_idx + 0 * hidden_size;
            const int reset_idx = weight_idx + 1 * hidden_size;
            const int new_idx = weight_idx + 2 * hidden_size;

            // GEMM+bias 融合版本：直接使用 Wx_bw 和 Rh_br
            // CPU 版本不使用 mask（推理模式），传递 false 模板参数
            const int32_t update_gate = computeUpdateGate<false>(cur_weight_ih_linear[update_idx], tmp_weight_hh_linear_[update_idx], gp);
            const int32_t reset_gate = computeResetGate<false>(cur_weight_ih_linear[reset_idx], tmp_weight_hh_linear_[reset_idx], gp);

            const int32_t new_gate = computeNewGate<false>(cur_weight_ih_linear[new_idx], tmp_weight_hh_linear_[new_idx], reset_gate, gp);

            if (training && v != nullptr) {
                const int base_v_idx = col * (hidden_size * 4) + row;
                v[base_v_idx + 0 * hidden_size] = update_gate;
                v[base_v_idx + 1 * hidden_size] = reset_gate;
                v[base_v_idx + 2 * hidden_size] = new_gate;
                v[base_v_idx + 3 * hidden_size] = tmp_weight_hh_linear_[new_idx];  // 直接使用 tmp_weight_hh_linear_[new_idx]
            }

            int32_t cur_h = computeHiddenState<false>(update_gate, new_gate, h[output_idx], gp);

            if (zoneout_prob > 0.0f && zoneout_mask != nullptr) {
                if (zoneout_mask[output_idx] != 0) cur_h = h[output_idx];
            }

            h_out[output_idx] = cur_h;
        }
    }
}

namespace {

// 由单一权威种子派生 CPU Linear 参数（仅 per-channel，每条一份 R 表示）。
template <class R>
void fillLinearParamsCPU(LinearQuantParamsCPUT<R> &lp, const GRUQuantParams &parms) {
    const int channel = parms.hidden_ * 3;
    lp.zp_x_ = parms.x_.zero_point;
    lp.zp_h_ = parms.h_.zero_point;

    lp.rescale_gemm_x_to_weight_ih_linear_.resize(channel);
    lp.rescale_bw_to_weight_ih_linear_.resize(channel);
    lp.rescale_gemm_h_to_weight_hh_linear_.resize(channel);
    lp.rescale_br_to_weight_hh_linear_.resize(channel);
    for (int c = 0; c < channel; ++c) {
        // GEMM 结果(累加器 scale = scale_W*scale_x) -> weight_ih_linear
        lp.rescale_gemm_x_to_weight_ih_linear_[c] =
            makeRescaleProduct<R>(parms.W_.channel(c), parms.x_, parms.weight_ih_linear_);
        // bias bw(自身 scale) -> 累加器 scale = scale_W*scale_x
        lp.rescale_bw_to_weight_ih_linear_[c] =
            makeRescaleToProduct<R>(parms.bw_.channel(c), parms.W_.channel(c), parms.x_);
        lp.rescale_gemm_h_to_weight_hh_linear_[c] =
            makeRescaleProduct<R>(parms.R_.channel(c), parms.h_, parms.weight_hh_linear_);
        lp.rescale_br_to_weight_hh_linear_[c] =
            makeRescaleToProduct<R>(parms.br_.channel(c), parms.R_.channel(c), parms.h_);
    }
}

// 由单一权威种子派生门计算参数（每条一份 R 表示）。
template <class R>
void fillGateParamsCPU(GateQuantParamsT<R> &gp, const GRUQuantParams &parms) {
    gp.zp_weight_ih_linear_ = parms.weight_ih_linear_.zero_point;
    gp.zp_weight_hh_linear_ = parms.weight_hh_linear_.zero_point;
    gp.zp_h_ = parms.h_.zero_point;

    gp.zp_update_gate_input_ = parms.update_gate_input_.zero_point;
    gp.zp_update_gate_output_ = parms.update_gate_output_.zero_point;
    gp.rescale_weight_ih_linear_to_update_gate_input_ = makeRescale<R>(parms.weight_ih_linear_, parms.update_gate_input_);
    gp.rescale_weight_hh_linear_to_update_gate_input_ = makeRescale<R>(parms.weight_hh_linear_, parms.update_gate_input_);

    gp.zp_reset_gate_input_ = parms.reset_gate_input_.zero_point;
    gp.zp_reset_gate_output_ = parms.reset_gate_output_.zero_point;
    gp.rescale_weight_ih_linear_to_reset_gate_input_ = makeRescale<R>(parms.weight_ih_linear_, parms.reset_gate_input_);
    gp.rescale_weight_hh_linear_to_reset_gate_input_ = makeRescale<R>(parms.weight_hh_linear_, parms.reset_gate_input_);

    gp.zp_new_gate_input_ = parms.new_gate_input_.zero_point;
    gp.zp_new_gate_output_ = parms.new_gate_output_.zero_point;
    gp.rescale_weight_ih_linear_to_new_gate_input_ = makeRescale<R>(parms.weight_ih_linear_, parms.new_gate_input_);
    // r*weight_hh_linear（scale = scale_reset_out*scale_weight_hh_linear）-> new_gate_input
    gp.rescale_reset_mul_hh_to_new_gate_input_ =
        makeRescaleProduct<R>(parms.reset_gate_output_, parms.weight_hh_linear_, parms.new_gate_input_);

    // 常数 1 量化到 update_gate_output 空间
    gp.quant_one_in_update_gate_scale_ =
        round_to_int(1.0f / parms.update_gate_output_.scale) + parms.update_gate_output_.zero_point;
    gp.rescale_new_gate_output_to_h_ = makeRescale<R>(parms.new_gate_output_, parms.h_);
    // u*h（scale = scale_update_out*scale_h）-> h
    gp.rescale_update_old_to_h_ = makeRescaleProduct<R>(parms.update_gate_output_, parms.h_, parms.h_);

    gp.bitwidth_config_ = parms.bitwidth_config_;
    // 直接复用校准时预计算的 LUT（与 GPU 路径一致）。
    gp.sigmoid_update_gate_lut_ = parms.sigmoid_update_gate_lut_;
    gp.sigmoid_reset_gate_lut_ = parms.sigmoid_reset_gate_lut_;
    gp.tanh_new_gate_lut_ = parms.tanh_new_gate_lut_;

#ifdef DEBUG
    gp.test = parms;
#endif
}

}  // namespace

// setRescaleParam: 只填充生效的那套（编译期两套实例都存在，运行时按模式选择）
void ForwardPassQuantCPU::setRescaleParam(const GRUQuantParams &parms) {
    use_pot2_ = parms.bitwidth_config_.usePOT2_;
    bitwidth_config_ = parms.bitwidth_config_;
    if (use_pot2_) {
        fillLinearParamsCPU<Pot2Rescale>(linear_params_pot2_, parms);
        fillGateParamsCPU<Pot2Rescale>(gate_params_pot2_, parms);
    } else {
        fillLinearParamsCPU<FixedPointScale>(linear_params_mshift_, parms);
        fillGateParamsCPU<FixedPointScale>(gate_params_mshift_, parms);
    }
}

template <class R>
void ForwardPassQuantCPU::RunImpl(const LinearQuantParamsCPUT<R> &lp, const GateQuantParamsT<R> &gp,
                                  int steps, const int32_t *W, const int32_t *R_w, const int32_t *bw,
                                  const int32_t *br, const int32_t *x, int32_t *h, int32_t *v,
                                  float zoneout_prob, const int32_t *zoneout_mask) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    EnsureBuffersAllocated(steps);
    PrecomputeWeightSums(W, R_w);

    // 计算 W*x + bw GEMM+bias 融合
    ComputeLinearX<R>(lp, gp, W, x, bw, steps);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    for (int i = 0; i < steps; ++i) {
        IterateInternal<R>(lp, gp, R_w, br, h + i * NH, h + (i + 1) * NH,
                           v ? v + i * NH * 4 : nullptr,
                           tmp_weight_ih_linear_.data() + i * NH3,
                           zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }
}

void ForwardPassQuantCPU::Run(int steps, const int32_t *W, const int32_t *R,
                               const int32_t *bw, const int32_t *br, const int32_t *x,
                               int32_t *h, int32_t *v, float zoneout_prob,
                               const int32_t *zoneout_mask) {
    if (use_pot2_) {
        RunImpl<Pot2Rescale>(linear_params_pot2_, gate_params_pot2_, steps, W, R, bw, br, x, h, v,
                             zoneout_prob, zoneout_mask);
    } else {
        RunImpl<FixedPointScale>(linear_params_mshift_, gate_params_mshift_, steps, W, R, bw, br, x, h, v,
                                 zoneout_prob, zoneout_mask);
    }
}

}  // namespace cpu
