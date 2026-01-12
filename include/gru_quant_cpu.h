#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"   // 复用 LUT 结构体和生成函数（CPU/GPU 共用）
#include "quantize_ops_helper.h"  // 复用 rshift_round, quantize_* 等函数

// ============================================================================
// gru_quant_cpu.h - 纯 C++ 定点 GRU 前向传播接口
// ============================================================================
//
// 设计说明:
//   - 完全不依赖 CUDA，可在任意 CPU 平台运行
//   - 与 GPU 版本保持数值一致性（用于验证和部署）
//   - 所有量化值使用 int32_t 统一存储，通过 bitwidth_config_ 控制实际位宽
//   - 复用 quantize_lut_types.h 中的 LUT 结构和生成函数
//   - 复用 quantize_ops_helper.h 中的通用函数
//
// ============================================================================

namespace cpu {

// ==================== CPU 专用辅助函数 ====================
// 以下函数已统一定义在 quantize_ops_helper.h 中，使用 __host__ __device__ 标记：
//   - rshift_round (int32_t / int64_t)
//   - clamp_by_bitwidth
//   - clamp_to_type<T>
//   - find_segment
//   - piecewise_linear_raw
//   - piecewise_linear


// ==================== CPU 量化 rescale 参数 ====================

/// @brief CPU 端 GRU 量化参数（与 GPU 版本 QuantGRUReScale 对应）
/// 注意：使用 std::vector 代替 dev::vector，使用 SigmoidLUT（与 GPU 共用）
struct QuantGRUReScaleCPU {
    int32_t zp_x_;
    int32_t zp_h_;

    // Linear 重缩放参数 (GEMM+bias)
    std::vector<int8_t> n_W_mul_x_div_Wx_;  // W*x 的 per-channel 重缩放移位（到 Wx+bx）
    std::vector<int8_t> n_bx_div_Wx_;       // bx 的 per-channel 重缩放移位（到 Wx+bx）
    int32_t zp_Wx_;
    std::vector<int8_t> n_R_mul_h_div_Rh_;  // R*h 的 per-channel 重缩放移位（到 Rh+br）
    std::vector<int8_t> n_br_div_Rh_;       // br 的 per-channel 重缩放移位（到 Rh+br）
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

    // g门 (New Gate)
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

    // LUT 表（每个门独立，使用与 GPU 共用的 SigmoidLUT 结构）
    SigmoidLUT sigmoid_z_lut_;
    SigmoidLUT sigmoid_r_lut_;
    SigmoidLUT tanh_g_lut_;

#ifdef DEBUG
    // -------------------- 调试参数 --------------------
    GRUQuantitativeParameters test;  ///< 保存完整量化参数用于调试
#endif
};

// ==================== 前向传播类声明 ====================

/// @brief CPU 版本量化 GRU 前向传播类（非模板，统一 int32_t 存储）
class ForwardPassQuantCPU {
   public:
    ForwardPassQuantCPU(bool training, int batch_size, int input_size, int hidden_size);
    ~ForwardPassQuantCPU();

    /// @brief 设置量化 rescale 参数（使用 quantize_ops_helper.h 中定义的通用结构）
    void setRescaleParam(const GRUQuantitativeParameters &params);

    /// @brief 执行前向传播（所有量化值使用 int32_t 存储）
    void Run(int steps, const int32_t *W, const int32_t *R, const int32_t *bx, const int32_t *br,
             const int32_t *x, int32_t *h, int32_t *v, float zoneout_prob,
             const int32_t *zoneout_mask);

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
    const int32_t *cached_W_ = nullptr;
    const int32_t *cached_R_ = nullptr;

    void EnsureBuffersAllocated(int steps);
    void PrecomputeWeightSums(const int32_t *W, const int32_t *R);
    void ComputeWx(const int32_t *W, const int32_t *x, int steps);
    void ComputeRh(const int32_t *R, const int32_t *h);
    void IterateInternal(const int32_t *R, const int32_t *bx, const int32_t *br, const int32_t *h,
                         int32_t *h_out, int32_t *v, const int32_t *cur_Wx, float zoneout_prob,
                         const int32_t *zoneout_mask);
};

}  // namespace cpu
