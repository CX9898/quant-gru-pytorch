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
// 2. GRU Gate Functions - 与 GPU 版本 gru_forward_gpu_quant.cu 计算一致
// ============================================================================

namespace {

int32_t computeZ(int channel_idx, int32_t Wx_val, int32_t Rh_val, int32_t bx_val, int32_t br_val,
                 const QuantGRUReScaleCPU &rescale) {
    const int32_t Wx_shifted = rshift_round(Wx_val - rescale.zp_Wx_, rescale.exp2_inv_Wx_div_z_pre_);
    const int32_t Rh_shifted = rshift_round(Rh_val - rescale.zp_Rh_, rescale.exp2_inv_Rh_div_z_pre_);
    const int32_t bx_shifted = rshift_round(bx_val, rescale.n_bx_div_z_[channel_idx]);
    const int32_t br_shifted = rshift_round(br_val, rescale.n_br_div_z_[channel_idx]);

    const int32_t z_pre_i32 = Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale.zp_z_pre_;

    const auto &bw_cfg = rescale.bitwidth_config_;
    return piecewise_linear(z_pre_i32, rescale.sigmoid_z_lut_, bw_cfg.z_pre_, bw_cfg.z_out_);
}

int32_t computeR(int channel_idx, int32_t Wx_val, int32_t Rh_val, int32_t bx_val, int32_t br_val,
                 const QuantGRUReScaleCPU &rescale) {
    const int32_t Wx_shifted = rshift_round(Wx_val - rescale.zp_Wx_, rescale.exp2_inv_Wx_div_r_pre_);
    const int32_t Rh_shifted = rshift_round(Rh_val - rescale.zp_Rh_, rescale.exp2_inv_Rh_div_r_pre_);
    const int32_t bx_shifted = rshift_round(bx_val, rescale.n_bx_div_r_[channel_idx]);
    const int32_t br_shifted = rshift_round(br_val, rescale.n_br_div_r_[channel_idx]);

    const int32_t r_pre_i32 = Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale.zp_r_pre_;

    const auto &bw_cfg = rescale.bitwidth_config_;
    return piecewise_linear(r_pre_i32, rescale.sigmoid_r_lut_, bw_cfg.r_pre_, bw_cfg.r_out_);
}

int32_t computeG(int channel_idx, int32_t Wx_val, int32_t Rh_val, int32_t bx_val, int32_t br_val,
                 int32_t r, const QuantGRUReScaleCPU &rescale, int32_t &Rh_add_br_g) {
    Rh_add_br_g = rshift_round(Rh_val - rescale.zp_Rh_, rescale.n_Rh_div_Rh_add_br_) +
                  rshift_round(br_val, rescale.n_br_div_Rh_add_br_[channel_idx]) +
                  rescale.zp_Rh_add_br_;
    Rh_add_br_g = clamp_by_bitwidth(Rh_add_br_g, rescale.bitwidth_config_.Rh_add_br_);

    const int64_t r_diff = static_cast<int64_t>(r) - rescale.zp_r_out_;
    const int64_t Rh_add_br_diff = static_cast<int64_t>(Rh_add_br_g) - rescale.zp_Rh_add_br_;
    const int64_t rRh_mul_i64 = r_diff * Rh_add_br_diff;

    int32_t rRh = static_cast<int32_t>(rshift_round(rRh_mul_i64, rescale.n_r_mul_Rh_add_br_div_rRh_)) +
                  rescale.zp_rRh_;
    rRh = clamp_by_bitwidth(rRh, rescale.bitwidth_config_.rRh_);

    const int32_t Wx_shifted = rshift_round(Wx_val - rescale.zp_Wx_, rescale.n_Wx_div_g_pre_);
    const int32_t rRh_shifted = rshift_round(rRh - rescale.zp_rRh_, rescale.n_rRh_div_g_pre_);
    const int32_t bx_shifted = rshift_round(bx_val, rescale.exp2_inv_bx_div_g_pre_[channel_idx]);

    const int32_t g_pre_i32 = Wx_shifted + rRh_shifted + bx_shifted + rescale.zp_g_pre_;

    const auto &bw_cfg = rescale.bitwidth_config_;
    return piecewise_linear(g_pre_i32, rescale.tanh_g_lut_, bw_cfg.g_pre_, bw_cfg.g_out_);
}

// computeH: 统一使用 int32_t 存储，通过位宽配置控制实际范围
int32_t computeH(int32_t z, int32_t g, int32_t h_old, const QuantGRUReScaleCPU &rescale) {
    const int64_t z_diff = static_cast<int64_t>(z) - rescale.zp_z_out_;
    const int64_t h_diff = static_cast<int64_t>(h_old) - rescale.zp_h_;
    const int64_t old_contrib_mul_i64 = z_diff * h_diff;

    int32_t old_contrib = static_cast<int32_t>(
        rshift_round(old_contrib_mul_i64, rescale.n_z_mul_h_div_old_contrib_)) +
        rescale.zp_old_contrib_;
    old_contrib = clamp_by_bitwidth(old_contrib, rescale.bitwidth_config_.old_contrib_);

    const int64_t one_minus_diff = static_cast<int64_t>(rescale.one_in_z_scale_) - z;
    const int64_t g_diff = static_cast<int64_t>(g) - rescale.zp_g_out_;
    const int64_t new_contrib_mul_i64 = one_minus_diff * g_diff;

    int32_t new_contrib = static_cast<int32_t>(
        rshift_round(new_contrib_mul_i64, rescale.n_z_out_mul_g_div_new_contrib_)) +
        rescale.zp_new_contrib_;
    new_contrib = clamp_by_bitwidth(new_contrib, rescale.bitwidth_config_.new_contrib_);

    const int32_t h_i32 =
        rshift_round(old_contrib - rescale.zp_old_contrib_, rescale.n_old_contrib_div_h_) +
        rshift_round(new_contrib - rescale.zp_new_contrib_, rescale.n_new_contrib_div_h_) +
        rescale.zp_h_;

    // 根据 h 的位宽配置进行 clamp
    return clamp_by_bitwidth(h_i32, rescale.bitwidth_config_.h_);
}

}  // namespace

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

    tmp_Wx_.resize(static_cast<size_t>(hidden3) * steps * batch_size);
    tmp_Rh_.resize(static_cast<size_t>(hidden3) * batch_size);

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
        W_sum_mul_x_zp_[m] = sum * rescale_param_.zp_x_;
    }

#pragma omp parallel for
    for (int m = 0; m < hidden3; m++) {
        int64_t sum = 0;
        for (int k = 0; k < hidden_size; k++) {
            sum += static_cast<int64_t>(R[k * hidden3 + m]);
        }
        // 与 GPU 保持一致：不在此处移位，只计算 R_sum * zp_h
        R_sum_mul_h_zp_[m] = sum * rescale_param_.zp_h_;
    }

    weight_sums_computed_ = true;
}

void ForwardPassQuantCPU::ComputeWx(const int32_t *W, const int32_t *x, int steps) {
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;
    const int N = steps * batch_size;

OMP_PARALLEL_FOR_2D
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < hidden3; m++) {
            int64_t acc = 0;
            for (int k = 0; k < input_size; k++) {
                acc += static_cast<int64_t>(W[k * hidden3 + m]) *
                       static_cast<int64_t>(x[n * input_size + k]);
            }
            int64_t val = acc - W_sum_mul_x_zp_[m];
            int32_t result = static_cast<int32_t>(
                rshift_round(val, rescale_param_.n_W_mul_x_div_Wx_[m])) +
                rescale_param_.zp_Wx_;
            tmp_Wx_[n * hidden3 + m] = clamp_by_bitwidth(result, rescale_param_.bitwidth_config_.Wx_);
        }
    }
}

void ForwardPassQuantCPU::ComputeRh(const int32_t *R, const int32_t *h) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const int hidden3 = hidden_size * 3;

OMP_PARALLEL_FOR_2D
    for (int n = 0; n < batch_size; n++) {
        for (int m = 0; m < hidden3; m++) {
            int64_t acc = 0;
            for (int k = 0; k < hidden_size; k++) {
                acc += static_cast<int64_t>(R[k * hidden3 + m]) *
                       static_cast<int64_t>(h[n * hidden_size + k]);
            }
            int64_t val = acc - R_sum_mul_h_zp_[m];
            int32_t result = static_cast<int32_t>(
                rshift_round(val, rescale_param_.n_R_mul_h_div_Rh_[m])) +
                rescale_param_.zp_Rh_;
            tmp_Rh_[n * hidden3 + m] = clamp_by_bitwidth(result, rescale_param_.bitwidth_config_.Rh_);
        }
    }
}

void ForwardPassQuantCPU::IterateInternal(const int32_t *R, const int32_t *bx,
                                          const int32_t *br, const int32_t *h,
                                          int32_t *h_out, int32_t *v,
                                          const int32_t *cur_Wx,
                                          float zoneout_prob,
                                          const int32_t *zoneout_mask) {
    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    ComputeRh(R, h);

OMP_PARALLEL_FOR_2D
    for (int col = 0; col < batch_size; col++) {
        for (int row = 0; row < hidden_size; row++) {
            const int weight_idx = col * (hidden_size * 3) + row;
            const int output_idx = col * hidden_size + row;
            const int z_idx = weight_idx + 0 * hidden_size;
            const int r_idx = weight_idx + 1 * hidden_size;
            const int g_idx = weight_idx + 2 * hidden_size;
            const int b_z_idx = row + 0 * hidden_size;
            const int b_r_idx = row + 1 * hidden_size;
            const int b_g_idx = row + 2 * hidden_size;

            const int32_t z = computeZ(b_z_idx, cur_Wx[z_idx], tmp_Rh_[z_idx],
                                       bx[b_z_idx], br[b_z_idx], rescale_param_);
            const int32_t r = computeR(b_r_idx, cur_Wx[r_idx], tmp_Rh_[r_idx],
                                       bx[b_r_idx], br[b_r_idx], rescale_param_);

            int32_t Rh_add_br_g;
            const int32_t g = computeG(b_g_idx, cur_Wx[g_idx], tmp_Rh_[g_idx],
                                       bx[b_g_idx], br[b_g_idx], r, rescale_param_, Rh_add_br_g);

            if (training && v != nullptr) {
                const int base_v_idx = col * (hidden_size * 4) + row;
                v[base_v_idx + 0 * hidden_size] = z;
                v[base_v_idx + 1 * hidden_size] = r;
                v[base_v_idx + 2 * hidden_size] = g;
                v[base_v_idx + 3 * hidden_size] = Rh_add_br_g;
            }

            int32_t cur_h = computeH(z, g, h[output_idx], rescale_param_);

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

    std::vector<int8_t> n_W_mul_x_div_Wx(channel);
    std::vector<int8_t> n_R_mul_h_div_Rh(channel);
    std::vector<int8_t> n_bx_to_z(channel), n_br_to_z(channel);
    std::vector<int8_t> n_bx_to_r(channel), n_br_to_r(channel);
    std::vector<int8_t> n_br_to_Rh_add_br(channel), n_bx_to_g(channel);

    for (int idx = 0; idx < channel; ++idx) {
        n_W_mul_x_div_Wx[idx] = (parms.exp2_inv_W_[idx] + parms.exp2_inv_x_) - parms.exp2_inv_Wx_;
        n_R_mul_h_div_Rh[idx] = (parms.exp2_inv_R_[idx] + parms.exp2_inv_h_) - parms.exp2_inv_Rh_;
        n_bx_to_z[idx] = parms.exp2_inv_bx_[idx] - parms.exp2_inv_z_pre_;
        n_br_to_z[idx] = parms.exp2_inv_br_[idx] - parms.exp2_inv_z_pre_;
        n_bx_to_r[idx] = parms.exp2_inv_bx_[idx] - parms.exp2_inv_r_pre_;
        n_br_to_r[idx] = parms.exp2_inv_br_[idx] - parms.exp2_inv_r_pre_;
        n_br_to_Rh_add_br[idx] = parms.exp2_inv_br_[idx] - parms.exp2_inv_Rh_add_br_;
        n_bx_to_g[idx] = parms.exp2_inv_bx_[idx] - parms.exp2_inv_g_pre_;
    }

    rescale_param_.zp_x_ = parms.zp_x_;
    rescale_param_.zp_h_ = parms.zp_h_;
    rescale_param_.n_W_mul_x_div_Wx_ = std::move(n_W_mul_x_div_Wx);
    rescale_param_.zp_Wx_ = parms.zp_Wx_;
    rescale_param_.n_R_mul_h_div_Rh_ = std::move(n_R_mul_h_div_Rh);
    rescale_param_.zp_Rh_ = parms.zp_Rh_;

    rescale_param_.zp_z_pre_ = parms.zp_z_pre_;
    rescale_param_.zp_z_out_ = parms.zp_z_out_;
    rescale_param_.exp2_inv_Wx_div_z_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_z_pre_;
    rescale_param_.exp2_inv_Rh_div_z_pre_ = parms.exp2_inv_Rh_ - parms.exp2_inv_z_pre_;
    rescale_param_.n_bx_div_z_ = std::move(n_bx_to_z);
    rescale_param_.n_br_div_z_ = std::move(n_br_to_z);

    rescale_param_.zp_r_pre_ = parms.zp_r_pre_;
    rescale_param_.zp_r_out_ = parms.zp_r_out_;
    rescale_param_.exp2_inv_Wx_div_r_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_r_pre_;
    rescale_param_.exp2_inv_Rh_div_r_pre_ = parms.exp2_inv_Rh_ - parms.exp2_inv_r_pre_;
    rescale_param_.n_bx_div_r_ = std::move(n_bx_to_r);
    rescale_param_.n_br_div_r_ = std::move(n_br_to_r);

    rescale_param_.zp_g_pre_ = parms.zp_g_pre_;
    rescale_param_.zp_g_out_ = parms.zp_g_out_;
    rescale_param_.n_Rh_div_Rh_add_br_ = parms.exp2_inv_Rh_ - parms.exp2_inv_Rh_add_br_;
    rescale_param_.n_br_div_Rh_add_br_ = std::move(n_br_to_Rh_add_br);
    rescale_param_.zp_Rh_add_br_ = parms.zp_Rh_add_br_;
    rescale_param_.n_r_mul_Rh_add_br_div_rRh_ =
        (parms.exp2_inv_r_out_ + parms.exp2_inv_Rh_add_br_) - parms.exp2_inv_rRh_;
    rescale_param_.zp_rRh_ = parms.zp_rRh_;
    rescale_param_.n_Wx_div_g_pre_ = parms.exp2_inv_Wx_ - parms.exp2_inv_g_pre_;
    rescale_param_.n_rRh_div_g_pre_ = parms.exp2_inv_rRh_ - parms.exp2_inv_g_pre_;
    rescale_param_.exp2_inv_bx_div_g_pre_ = std::move(n_bx_to_g);

    rescale_param_.one_in_z_scale_ = rshift_round(1, -parms.exp2_inv_z_out_) + parms.zp_z_out_;
    rescale_param_.zp_new_contrib_ = parms.zp_new_contrib_;
    rescale_param_.n_z_out_mul_g_div_new_contrib_ =
        (parms.exp2_inv_z_out_ + parms.exp2_inv_g_out_) - parms.exp2_inv_new_contrib_;
    rescale_param_.zp_old_contrib_ = parms.zp_old_contrib_;
    rescale_param_.n_z_mul_h_div_old_contrib_ =
        (parms.exp2_inv_z_out_ + parms.exp2_inv_h_) - parms.exp2_inv_old_contrib_;
    rescale_param_.n_new_contrib_div_h_ = parms.exp2_inv_new_contrib_ - parms.exp2_inv_h_;
    rescale_param_.n_old_contrib_div_h_ = parms.exp2_inv_old_contrib_ - parms.exp2_inv_h_;

    rescale_param_.bitwidth_config_ = parms.bitwidth_config_;

    // 直接复用 quantize_lut_types.h 中声明的 LUT 生成函数（实现在 quantize_ops.cu）
    // CPU 版本：LUT 存储在普通变量中，不拷贝到 CUDA 常量内存
    rescale_param_.sigmoid_z_lut_ = generate_sigmoid_lut(
        parms.exp2_inv_z_pre_, parms.zp_z_pre_, parms.exp2_inv_z_out_, parms.zp_z_out_,
        parms.bitwidth_config_.z_pre_, parms.bitwidth_config_.z_out_);

    rescale_param_.sigmoid_r_lut_ = generate_sigmoid_lut(
        parms.exp2_inv_r_pre_, parms.zp_r_pre_, parms.exp2_inv_r_out_, parms.zp_r_out_,
        parms.bitwidth_config_.r_pre_, parms.bitwidth_config_.r_out_);

    rescale_param_.tanh_g_lut_ = generate_tanh_lut(
        parms.exp2_inv_g_pre_, parms.zp_g_pre_, parms.exp2_inv_g_out_, parms.zp_g_out_,
        parms.bitwidth_config_.g_pre_, parms.bitwidth_config_.g_out_);
}

void ForwardPassQuantCPU::Run(int steps, const int32_t *W, const int32_t *R,
                               const int32_t *bx, const int32_t *br, const int32_t *x,
                               int32_t *h, int32_t *v, float zoneout_prob,
                               const int32_t *zoneout_mask) {
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

    EnsureBuffersAllocated(steps);
    PrecomputeWeightSums(W, R);
    ComputeWx(W, x, steps);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;

    for (int i = 0; i < steps; ++i) {
        IterateInternal(R, bx, br, h + i * NH, h + (i + 1) * NH,
                        v ? v + i * NH * 4 : nullptr,
                        tmp_Wx_.data() + i * NH3,
                        zoneout_prob, zoneout_mask ? zoneout_mask + i * NH : nullptr);
    }
}

}  // namespace cpu
