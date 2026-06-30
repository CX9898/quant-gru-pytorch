#pragma once

// ============================================================================
// quantize_param_types.h - GRU 量化参数结构体定义
// ============================================================================
//
// 设计（单一权威种子 + 类型驱动派发）：
//   1. QuantParam        - 单一权威量化种子（原始 scale + 零点），所有表示按需派生
//   2. ChannelQuantParam - 权重/偏置的多粒度容器，仅保留一份 per-channel 权威数组
//   3. GRUQuantParams    - Host 端完整量化参数（校准阶段使用），由上面两者组成
//   4. Pot2Rescale / FixedPointScale / FloatRescale - 强类型 rescale 表示（设备执行用）
//   5. GateQuantParamsT<R> / LinearQuantParamsGPUT<R> / LinearQuantParamsCPUT<R>
//        - INT 后端按表示分裂的设备结构体（每个只带一份执行表示）
//   6. GateQuantParamsFP / LinearRescaleParamsFP - FP 后端独立结构体
//
// 关键约定：
//   - QuantParam.scale 是唯一权威：
//       * 仿射(M+shift)模式 = 校准原始连续 scale（绝不经 encode/decode 往返）
//       * POT2 模式         = 2^-shift（精确，shift 可由 scale 无损还原）
//   - 设备执行表示（Pot2Rescale/FixedPointScale/FloatRescale）只在 setRescaleParam
//     里从权威种子派生一次，kernel 通过重载的 applyRescale 使用。
//
// ============================================================================

#include <array>
#include <vector>

#include "dev_vector.h"
#include "quantize_bitwidth_config.h"
#include "quantize_lut_types.h"

// ============================================================================
// 强类型 rescale 表示（供 applyRescale 重载派发）
// ============================================================================

// 仿射 M+shift：rescale(x) = round((x * multiplier) >> shift)；POT2 退化为 {1, shift}
struct FixedPointScale {
    uint16_t multiplier = 1;
    int8_t shift = 0;
};

// POT2：rescale(x) = round(x >> shift)（纯移位，shift 可为负表示左移）
struct Pot2Rescale {
    int8_t shift = 0;
};

// FP 存储版：rescale(x) = x * inv_div
struct FloatRescale {
    float inv_div = 1.0f;
};

// ============================================================================
// 单一权威量化种子
// ============================================================================

/**
 * @brief 单一权威量化种子。
 *
 * 只存储原始 scale 与零点；shift / FixedPointScale / inv_div 等执行表示一律按需派生
 * （见 quantize_ops_helper.h 的 pot2Shift / toFixedScale / makeRescale），
 * 从结构上消除"同一数值存多份、用错版本"的隐藏 BUG。本类型层保持纯数据、不含算子逻辑。
 */
struct QuantParam {
    float scale = 0.0f;       ///< 权威 scale：仿射=连续校准值；POT2=2^-shift
    int32_t zero_point = 0;   ///< 零点
};

/**
 * @brief 权重/偏置多粒度容器。
 *
 * 仅保留一份 per-channel 权威数组（按粒度广播填满 3*hidden）。
 * tensor()/gate(g)/channel(c) 均是对该数组取索引的只读视图，无独立副本。
 */
struct ChannelQuantParam {
    OperatorQuantConfig::QuantizationGranularity granularity = OperatorQuantConfig::PER_CHANNEL;
    int hidden = 0;                       ///< channel = hidden * 3
    std::vector<QuantParam> channels;     ///< 唯一权威存储，size = hidden * 3

    void resize(int channel_size) { channels.assign(channel_size, QuantParam{}); }
    size_t size() const { return channels.size(); }

    const QuantParam& channel(int c) const { return channels[c]; }
    QuantParam& channel(int c) { return channels[c]; }
    const QuantParam& gate(int g) const { return channels[g * hidden]; }
    const QuantParam& tensor() const { return channels[0]; }
};

// ============================================================================
// GRU 完整量化参数结构体（Host 端，单一权威种子）
// ============================================================================

/**
 * @brief GRU 量化参数结构体（Host 端）
 *
 * 每个量只存一个 QuantParam（权重/偏置用 ChannelQuantParam）。推理阶段通过
 * setRescaleParam 派生为对应表示的精简设备结构体。
 */
struct GRUQuantParams {
    OperatorQuantConfig bitwidth_config_;  ///< 各算子的量化位宽配置

    int hidden_ = 0;  ///< 隐藏层大小，channel = hidden * 3

    // -------------------- 基础输入 --------------------
    QuantParam x_;
    QuantParam h_;

    // -------------------- 权重 / 偏置（多粒度，唯一 per-channel 权威） --------------------
    ChannelQuantParam W_;
    ChannelQuantParam R_;
    ChannelQuantParam bw_;
    ChannelQuantParam br_;

    // -------------------- Linear 输出 (GEMM+bias) --------------------
    QuantParam weight_ih_linear_;  ///< W*x + bw
    QuantParam weight_hh_linear_;  ///< R*h + br

    // -------------------- 门激活前（pre-activation） --------------------
    QuantParam update_gate_input_;
    QuantParam reset_gate_input_;
    QuantParam new_gate_input_;

    // -------------------- 门激活后（post-activation） --------------------
    QuantParam update_gate_output_;
    QuantParam reset_gate_output_;
    QuantParam new_gate_output_;

    // -------------------- 中间计算 --------------------
    QuantParam mul_reset_hidden_;       ///< r * weight_hh_linear

    // -------------------- 隐状态更新 --------------------
    QuantParam mul_new_contribution_;   ///< (1-u) * n
    QuantParam mul_old_contribution_;   ///< u * h

    // -------------------- LUT 表（每层独立，finalize_calibration 时生成） --------------------
    SigmoidLUT sigmoid_update_gate_lut_;
    SigmoidLUT sigmoid_reset_gate_lut_;
    SigmoidLUT tanh_new_gate_lut_;
};

// ============================================================================
// GRU 门计算量化参数（纯标量，CPU/GPU 共用，按 rescale 表示 R 模板化）
// ============================================================================

/**
 * @brief 门计算量化参数（按 rescale 表示 R 模板化，每个实例只带一份执行表示）。
 *
 * R ∈ { Pot2Rescale, FixedPointScale }。kernel 通过重载的 applyRescale(x, R) 使用，
 * 无运行时 usePOT2 分支、无多版本拷贝。
 */
template <class R>
struct GateQuantParamsT {
    // -------------------- Linear 输出零点 --------------------
    int32_t zp_weight_ih_linear_ = 0;
    int32_t zp_weight_hh_linear_ = 0;
    int32_t zp_h_ = 0;

    // -------------------- Update Gate --------------------
    int32_t zp_update_gate_input_ = 0;
    int32_t zp_update_gate_output_ = 0;
    R rescale_weight_ih_linear_to_update_gate_input_{};
    R rescale_weight_hh_linear_to_update_gate_input_{};

    // -------------------- Reset Gate --------------------
    int32_t zp_reset_gate_input_ = 0;
    int32_t zp_reset_gate_output_ = 0;
    R rescale_weight_ih_linear_to_reset_gate_input_{};
    R rescale_weight_hh_linear_to_reset_gate_input_{};

    // -------------------- New Gate（候选隐状态） --------------------
    int32_t zp_new_gate_input_ = 0;
    int32_t zp_new_gate_output_ = 0;
    R rescale_weight_ih_linear_to_new_gate_input_{};
    R rescale_reset_mul_hh_to_new_gate_input_{};

    // -------------------- 隐状态更新（统一 scale 空间优化） --------------------
    int32_t quant_one_in_update_gate_scale_ = 0;  ///< 常数 1 量化到 update_gate_output 空间 = round(1/scale)+zp
    R rescale_new_gate_output_to_h_{};
    R rescale_update_old_to_h_{};

    // -------------------- 运行时配置 --------------------
    OperatorQuantConfig bitwidth_config_;

    // -------------------- LUT 表 --------------------
    SigmoidLUT sigmoid_update_gate_lut_;
    SigmoidLUT sigmoid_reset_gate_lut_;
    SigmoidLUT tanh_new_gate_lut_;

#ifdef DEBUG
    GRUQuantParams test;  ///< 保存完整量化参数用于调试
#endif
};

using GateQuantParamsPot2 = GateQuantParamsT<Pot2Rescale>;
using GateQuantParamsMShift = GateQuantParamsT<FixedPointScale>;

// ============================================================================
// Linear 层量化参数（INT 后端，仅 per-channel，按 rescale 表示 R 模板化）
// ============================================================================

/**
 * @brief Linear 层量化参数（GPU 版本）。
 *
 * 设备层只保留 per-channel 数组（已删除 per-tensor/per-gate 快路径与 granularity 分支）。
 * 每条 rescale 只存一份 R 表示。
 */
template <class R>
struct LinearQuantParamsGPUT {
    int32_t zp_x_ = 0;
    int32_t zp_h_ = 0;

    dev::vector<R> rescale_gemm_x_to_weight_ih_linear_;  ///< W*x per-channel rescale
    dev::vector<R> rescale_bw_to_weight_ih_linear_;      ///< bw per-channel rescale
    dev::vector<R> rescale_gemm_h_to_weight_hh_linear_;  ///< R*h per-channel rescale
    dev::vector<R> rescale_br_to_weight_hh_linear_;      ///< br per-channel rescale
};

/**
 * @brief Linear 层量化参数（CPU 版本，std::vector）。
 */
template <class R>
struct LinearQuantParamsCPUT {
    int32_t zp_x_ = 0;
    int32_t zp_h_ = 0;

    std::vector<R> rescale_gemm_x_to_weight_ih_linear_;
    std::vector<R> rescale_bw_to_weight_ih_linear_;
    std::vector<R> rescale_gemm_h_to_weight_hh_linear_;
    std::vector<R> rescale_br_to_weight_hh_linear_;
};

// ============================================================================
// 浮点存储版量化参数（FP 后端独立，含激活 scale、float 零点、无 LUT）
// ============================================================================

/**
 * @brief 门计算量化参数（浮点存储版本）。
 *
 * rescale 以 inv_div = src_scale/dst_scale 形式存储；零点使用 float；激活用 real_sigmoid/real_tanh。
 */
struct GateQuantParamsFP {
    // -------------------- Linear 输出零点 --------------------
    float zp_weight_ih_linear_ = 0.0f;
    float zp_weight_hh_linear_ = 0.0f;
    float zp_h_ = 0.0f;

    // -------------------- Update Gate --------------------
    float zp_update_gate_input_ = 0.0f;
    float zp_update_gate_output_ = 0.0f;
    float inv_div_weight_ih_linear_to_update_gate_input_ = 1.0f;
    float inv_div_weight_hh_linear_to_update_gate_input_ = 1.0f;

    // -------------------- Reset Gate --------------------
    float zp_reset_gate_input_ = 0.0f;
    float zp_reset_gate_output_ = 0.0f;
    float inv_div_weight_ih_linear_to_reset_gate_input_ = 1.0f;
    float inv_div_weight_hh_linear_to_reset_gate_input_ = 1.0f;

    // -------------------- New Gate --------------------
    float zp_new_gate_input_ = 0.0f;
    float zp_new_gate_output_ = 0.0f;
    float inv_div_weight_ih_linear_to_new_gate_input_ = 1.0f;
    float inv_div_reset_mul_hh_to_new_gate_input_ = 1.0f;

    // -------------------- 隐状态更新 --------------------
    float quant_one_in_update_gate_scale_ = 0.0f;
    float inv_div_update_old_to_h_ = 1.0f;
    float inv_div_new_gate_output_to_h_ = 1.0f;

    // -------------------- 激活函数 scale（real_sigmoid/real_tanh） --------------------
    float scale_update_gate_input_ = 1.0f;
    float scale_update_gate_output_ = 1.0f;
    float scale_reset_gate_input_ = 1.0f;
    float scale_reset_gate_output_ = 1.0f;
    float scale_new_gate_input_ = 1.0f;
    float scale_new_gate_output_ = 1.0f;
};

/**
 * @brief Linear Rescale 参数（FP 版本，kernel 直接使用）。
 */
struct LinearRescaleParamsFP {
    float zp_x_ = 0.0f;
    float zp_h_ = 0.0f;

    const float *W_sum_mul_x_zp = nullptr;                    ///< [3*hidden] 运行时设置
    const float *inv_div_gemm_x_to_weight_ih_linear_ = nullptr;  ///< [3*hidden] 指向 dev::vector
    const float *inv_div_bw_to_gemm_x_ = nullptr;              ///< [3*hidden] 指向 dev::vector
    float zp_weight_ih_linear_ = 0.0f;

    const float *R_sum_mul_h_zp = nullptr;                    ///< [3*hidden] 运行时设置
    const float *inv_div_gemm_h_to_weight_hh_linear_ = nullptr;  ///< [3*hidden] 指向 dev::vector
    const float *inv_div_br_to_gemm_h_ = nullptr;              ///< [3*hidden] 指向 dev::vector
    float zp_weight_hh_linear_ = 0.0f;
};
