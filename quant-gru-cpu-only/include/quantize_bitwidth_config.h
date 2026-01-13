#pragma once

// ============================================================================
// quantize_bitwidth_config.h - 量化位宽配置
// ============================================================================

#include <cstdint>

/**
 * @brief 量化位宽配置
 */
struct QuantBitWidth {
    int8_t bits_ = 8;
    bool is_unsigned_ = false;  // false=INT(有符号), true=UINT(无符号)

    QuantBitWidth() = default;
    QuantBitWidth(int8_t b, bool is_unsigned = false) 
        : bits_(b), is_unsigned_(b >= 32 ? false : is_unsigned) {}

    int32_t qmin() const {
        if (bits_ >= 32) return static_cast<int32_t>(0x80000000);
        return is_unsigned_ ? 0 : -(1 << (bits_ - 1));
    }
    
    int32_t qmax() const {
        if (bits_ >= 32) return 0x7FFFFFFF;
        return is_unsigned_ ? (1 << bits_) - 1 : (1 << (bits_ - 1)) - 1;
    }
};

/**
 * @brief 各算子量化位宽配置
 */
struct OperatorQuantConfig {
    QuantBitWidth x_{8, false}, h_{8, false};
    QuantBitWidth W_{8, false}, R_{8, false}, bw_{8, false}, br_{8, false};
    QuantBitWidth weight_ih_linear_{8, false}, weight_hh_linear_{8, false};  // GEMM+bias 融合输出
    QuantBitWidth update_gate_input_{8, false}, update_gate_output_{8, true};   // update_gate_output: UINT
    QuantBitWidth reset_gate_input_{8, false}, reset_gate_output_{8, true};     // reset_gate_output: UINT
    QuantBitWidth new_gate_input_{8, false}, new_gate_output_{8, false};
    QuantBitWidth mul_reset_hidden_{8, false};  // r * weight_hh_linear (new gate中)
    QuantBitWidth mul_old_contribution_{8, false}, mul_new_contribution_{8, false};

    OperatorQuantConfig& setAllBitWidths(int8_t bits) {
        QuantBitWidth* signed_members[] = {
            &x_, &h_, &W_, &R_, &bw_, &br_, &weight_ih_linear_, &weight_hh_linear_,
            &update_gate_input_, &reset_gate_input_, &new_gate_input_, &new_gate_output_,
            &mul_reset_hidden_, &mul_old_contribution_, &mul_new_contribution_
        };
        for (auto* m : signed_members) {
            m->bits_ = bits;
            m->is_unsigned_ = false;  // 有符号
        }
        update_gate_output_ = {bits, true};   // UINT
        reset_gate_output_ = {bits, true};    // UINT
        return *this;
    }
};
