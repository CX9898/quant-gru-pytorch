#pragma once

#include <cstdint>
#include "cuda_compat.h"

struct QuantBitWidth {
    int8_t bits_ = 8;
    bool is_unsigned_ = false;  // false=INT(有符号), true=UINT(无符号)

    QuantBitWidth() = default;
    __host__ __device__ QuantBitWidth(int8_t b, bool is_unsigned = false) 
        : bits_(b), is_unsigned_(b >= 32 ? false : is_unsigned) {}

    __host__ __device__ int32_t qmin() const {
        if (bits_ >= 32) return static_cast<int32_t>(0x80000000);
        return is_unsigned_ ? 0 : -(1 << (bits_ - 1));
    }
    
    __host__ __device__ int32_t qmax() const {
        if (bits_ >= 32) return 0x7FFFFFFF;
        return is_unsigned_ ? (1 << bits_) - 1 : (1 << (bits_ - 1)) - 1;
    }
    
    __host__ __device__ int64_t range() const {
        return static_cast<int64_t>(qmax()) - static_cast<int64_t>(qmin());
    }

    bool operator==(const QuantBitWidth& other) const {
        return bits_ == other.bits_ && is_unsigned_ == other.is_unsigned_;
    }
    bool operator!=(const QuantBitWidth& other) const {
        return !(*this == other);
    }
};

struct OperatorQuantConfig {
    QuantBitWidth x_{16, false}, h_{16, false};
    QuantBitWidth W_{16, false}, R_{16, false}, bw_{16, false}, br_{16, false};
    QuantBitWidth weight_ih_linear_{16, false}, weight_hh_linear_{16, false};  // GEMM+bias 融合输出
    QuantBitWidth update_gate_input_{8, false}, update_gate_output_{8, true};   // update_gate_output: UINT
    QuantBitWidth reset_gate_input_{8, false}, reset_gate_output_{8, true};     // reset_gate_output: UINT
    QuantBitWidth new_gate_input_{8, false}, new_gate_output_{8, false};
    QuantBitWidth mul_reset_hidden_{16, false};  // r * weight_hh_linear (new gate中)
    QuantBitWidth mul_old_contribution_{16, false}, mul_new_contribution_{16, false};

    bool x_symmetric_ = false, h_symmetric_ = false;
    bool W_symmetric_ = true, R_symmetric_ = true, bw_symmetric_ = true, br_symmetric_ = true;
    bool weight_ih_linear_symmetric_ = false, weight_hh_linear_symmetric_ = false;
    bool update_gate_input_symmetric_ = false, update_gate_output_symmetric_ = false;
    bool reset_gate_input_symmetric_ = false, reset_gate_output_symmetric_ = false;
    bool new_gate_input_symmetric_ = false, new_gate_output_symmetric_ = true;
    bool mul_reset_hidden_symmetric_ = false;
    bool mul_old_contribution_symmetric_ = false, mul_new_contribution_symmetric_ = false;

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
