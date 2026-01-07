#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"
#include "quantize_ops_helper.h"

// ============================================================================
// gru_quant_cpu.h - 纯 C++ 定点 GRU 前向传播接口
// ============================================================================

namespace cpu {

// ==================== CPU 量化 rescale 参数 ====================

struct QuantGRUReScaleCPU {
    int32_t zp_x_;
    int32_t zp_h_;

    std::vector<int8_t> n_W_mul_x_div_Wx_;
    int32_t zp_Wx_;
    std::vector<int8_t> n_R_mul_h_div_Rh_;
    int32_t zp_Rh_;

    // z门
    int32_t zp_z_pre_;
    int32_t zp_z_out_;
    int8_t exp2_inv_Wx_div_z_pre_;
    int8_t exp2_inv_Rh_div_z_pre_;
    std::vector<int8_t> n_bx_div_z_;
    std::vector<int8_t> n_br_div_z_;

    // r门
    int32_t zp_r_pre_;
    int32_t zp_r_out_;
    int8_t exp2_inv_Wx_div_r_pre_;
    int8_t exp2_inv_Rh_div_r_pre_;
    std::vector<int8_t> n_bx_div_r_;
    std::vector<int8_t> n_br_div_r_;

    // g门
    int32_t zp_g_pre_;
    int32_t zp_g_out_;
    int8_t n_Rh_div_Rh_add_br_;
    std::vector<int8_t> n_br_div_Rh_add_br_;
    int32_t zp_Rh_add_br_;
    int8_t n_r_mul_Rh_add_br_div_rRh_;
    int32_t zp_rRh_;
    int8_t n_Wx_div_g_pre_;
    int8_t n_rRh_div_g_pre_;
    std::vector<int8_t> exp2_inv_bx_div_g_pre_;

    // h_new
    int32_t one_in_z_scale_;
    int32_t zp_new_contrib_;
    int8_t n_z_out_mul_g_div_new_contrib_;
    int32_t zp_old_contrib_;
    int8_t n_z_mul_h_div_old_contrib_;
    int8_t n_new_contrib_div_h_;
    int8_t n_old_contrib_div_h_;

    // 位宽配置
    OperatorQuantConfig bitwidth_config_;

    // LUT 表
    SigmoidLUT sigmoid_z_lut_;
    SigmoidLUT sigmoid_r_lut_;
    SigmoidLUT tanh_g_lut_;
};

// ==================== 前向传播类声明 ====================

template <typename XT, typename HT, typename WT, typename RT>
class ForwardPassQuantCPU {
   public:
    ForwardPassQuantCPU(bool training, int batch_size, int input_size, int hidden_size);
    ~ForwardPassQuantCPU();

    void setRescaleParam(const GRUQuantitativeParameters &params);

    void Run(int steps, const WT *W, const RT *R, const int32_t *bx, const int32_t *br,
             const XT *x, HT *h, int32_t *v, float zoneout_prob, const HT *zoneout_mask);

   private:
    struct PrivateData;
    std::unique_ptr<PrivateData> data_;

    QuantGRUReScaleCPU rescale_param_;

    int max_steps_ = 0;
    std::vector<int32_t> tmp_Wx_;
    std::vector<int32_t> tmp_Rh_;
    std::vector<int64_t> W_sum_mul_x_zp_;
    std::vector<int64_t> R_sum_mul_h_zp_;
    bool weight_sums_computed_ = false;
    const WT *cached_W_ = nullptr;
    const RT *cached_R_ = nullptr;

    void EnsureBuffersAllocated(int steps);
    void PrecomputeWeightSums(const WT *W, const RT *R);
    void ComputeWx(const WT *W, const XT *x, int steps);
    void ComputeRh(const RT *R, const HT *h);
    void IterateInternal(const RT *R, const int32_t *bx, const int32_t *br, const HT *h, HT *h_out,
                         int32_t *v, const int32_t *cur_Wx, float zoneout_prob,
                         const HT *zoneout_mask);
};

}  // namespace cpu

