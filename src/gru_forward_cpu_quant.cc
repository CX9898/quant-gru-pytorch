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

// computeZ, computeR, computeG, computeH 现在使用 quantize_ops_helper.h 中的模板函数
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

    tmp_Wx_bx_.resize(static_cast<size_t>(hidden3) * steps * batch_size);
    tmp_Rh_br_.resize(static_cast<size_t>(hidden3) * batch_size);

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

void ForwardPassQuantCPU::ComputeWxBx(const int32_t *W, const int32_t *x, const int32_t *bx, int steps) {
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
                rshift_round(gemm_val, linear_params_.n_W_mul_x_div_Wx_[m])) +
                gate_params_.zp_Wx_;
            
            // 添加 bias: bx 重缩放到 Wx 空间
            int32_t bias_rescaled = rshift_round(bx[m], linear_params_.n_bx_div_Wx_[m]);
            int32_t result = gemm_result + bias_rescaled;
            tmp_Wx_bx_[n * hidden3 + m] = clamp_by_bitwidth(result, gate_params_.bitwidth_config_.Wx_);
        }
    }
}

void ForwardPassQuantCPU::ComputeRhBr(const int32_t *R, const int32_t *h, const int32_t *br) {
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
                rshift_round(gemm_val, linear_params_.n_R_mul_h_div_Rh_[m])) +
                gate_params_.zp_Rh_;
            
            // 添加 bias: br 重缩放到 Rh 空间
            int32_t bias_rescaled = rshift_round(br[m], linear_params_.n_br_div_Rh_[m]);
            int32_t result = gemm_result + bias_rescaled;
            tmp_Rh_br_[n * hidden3 + m] = clamp_by_bitwidth(result, gate_params_.bitwidth_config_.Rh_);
        }
    }
}

void ForwardPassQuantCPU::IterateInternal(const int32_t *R, const int32_t *br,
                                          const int32_t *h, int32_t *h_out, int32_t *v,
                                          const int32_t *cur_Wx_bx,
                                          float zoneout_prob,
                                          const int32_t *zoneout_mask) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    // 计算 R*h + br GEMM+bias 融合
    ComputeRhBr(R, h, br);

OMP_PARALLEL_FOR_2D
    for (int col = 0; col < batch_size; col++) {
        for (int row = 0; row < hidden_size; row++) {
            const int weight_idx = col * (hidden_size * 3) + row;
            const int output_idx = col * hidden_size + row;
            const int z_idx = weight_idx + 0 * hidden_size;
            const int r_idx = weight_idx + 1 * hidden_size;
            const int g_idx = weight_idx + 2 * hidden_size;

            // GEMM+bias 融合版本：直接使用 Wx_bx 和 Rh_br
            const int32_t z = computeZ(cur_Wx_bx[z_idx], tmp_Rh_br_[z_idx], gate_params_);
            const int32_t r = computeR(cur_Wx_bx[r_idx], tmp_Rh_br_[r_idx], gate_params_);

            int32_t Rh_br_g;
            const int32_t g = computeG(cur_Wx_bx[g_idx], tmp_Rh_br_[g_idx], r, gate_params_, Rh_br_g);

            if (training && v != nullptr) {
                const int base_v_idx = col * (hidden_size * 4) + row;
                v[base_v_idx + 0 * hidden_size] = z;
                v[base_v_idx + 1 * hidden_size] = r;
                v[base_v_idx + 2 * hidden_size] = g;
                v[base_v_idx + 3 * hidden_size] = Rh_br_g;
            }

            int32_t cur_h = computeH(z, g, h[output_idx], gate_params_);

            if (zoneout_prob > 0.0f && zoneout_mask != nullptr) {
                if (zoneout_mask[output_idx] != 0) cur_h = h[output_idx];
            }

            h_out[output_idx] = cur_h;
        }
    }
}

// setRescaleParam: 直接复用 quantize_lut_types.h 中声明的 generate_*_lut 函数
void ForwardPassQuantCPU::setRescaleParam(const GRUQuantitativeParameters &parms) {
    const int channel = parms.hidden_ * 3;

    // ==================== Linear 层参数（per-channel）====================
    linear_params_.zp_x_ = parms.zp_x_;
    linear_params_.zp_h_ = parms.zp_h_;

    // 计算并存储 per-channel 重缩放参数
    linear_params_.n_W_mul_x_div_Wx_.resize(channel);
    linear_params_.n_bx_div_Wx_.resize(channel);
    linear_params_.n_R_mul_h_div_Rh_.resize(channel);
    linear_params_.n_br_div_Rh_.resize(channel);

    for (int idx = 0; idx < channel; ++idx) {
        linear_params_.n_W_mul_x_div_Wx_[idx] = (parms.exp2_inv_W_[idx] + parms.exp2_inv_x_) - parms.exp2_inv_Wx_;
        linear_params_.n_R_mul_h_div_Rh_[idx] = (parms.exp2_inv_R_[idx] + parms.exp2_inv_h_) - parms.exp2_inv_Rh_;
        linear_params_.n_bx_div_Wx_[idx] = parms.exp2_inv_bx_[idx] - parms.exp2_inv_Wx_;
        linear_params_.n_br_div_Rh_[idx] = parms.exp2_inv_br_[idx] - parms.exp2_inv_Rh_;
    }

#ifdef DEBUG
    linear_params_.exp2_inv_bx_ = parms.exp2_inv_bx_;
    linear_params_.exp2_inv_br_ = parms.exp2_inv_br_;
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
    gate_params_.sigmoid_z_lut_ = generate_sigmoid_lut(
        parms.exp2_inv_z_pre_, parms.zp_z_pre_, parms.exp2_inv_z_out_, parms.zp_z_out_,
        parms.bitwidth_config_.z_pre_, parms.bitwidth_config_.z_out_);
    gate_params_.sigmoid_r_lut_ = generate_sigmoid_lut(
        parms.exp2_inv_r_pre_, parms.zp_r_pre_, parms.exp2_inv_r_out_, parms.zp_r_out_,
        parms.bitwidth_config_.r_pre_, parms.bitwidth_config_.r_out_);
    gate_params_.tanh_g_lut_ = generate_tanh_lut(
        parms.exp2_inv_g_pre_, parms.zp_g_pre_, parms.exp2_inv_g_out_, parms.zp_g_out_,
        parms.bitwidth_config_.g_pre_, parms.bitwidth_config_.g_out_);

#ifdef DEBUG
    gate_params_.test = parms;
#endif
}

void ForwardPassQuantCPU::Run(int steps, const int32_t *W, const int32_t *R,
                               const int32_t *bx, const int32_t *br, const int32_t *x,
                               int32_t *h, int32_t *v, float zoneout_prob,
                               const int32_t *zoneout_mask) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    EnsureBuffersAllocated(steps);
    PrecomputeWeightSums(W, R);
    
    // 计算 W*x + bx GEMM+bias 融合
    ComputeWxBx(W, x, bx, steps);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, br, h + i * NH, h + (i + 1) * NH,
                        v ? v + i * NH * 4 : nullptr,
                        tmp_Wx_bx_.data() + i * NH3,
                        zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }
}

}  // namespace cpu
