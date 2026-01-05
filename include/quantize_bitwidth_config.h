#pragma once

#include <cstdint>
#include "cuda_compat.h"

struct QuantBitWidth {
    int8_t bits_ = 8;
    bool is_signed_ = true;

    QuantBitWidth() = default;
    __host__ __device__ QuantBitWidth(int8_t b, bool s = true) 
        : bits_(b), is_signed_(b >= 32 ? true : s) {}

    __host__ __device__ int32_t qmin() const {
        if (bits_ >= 32) return static_cast<int32_t>(0x80000000);
        return is_signed_ ? -(1 << (bits_ - 1)) : 0;
    }
    
    __host__ __device__ int32_t qmax() const {
        if (bits_ >= 32) return 0x7FFFFFFF;
        return is_signed_ ? (1 << (bits_ - 1)) - 1 : (1 << bits_) - 1;
    }
    
    __host__ __device__ int64_t range() const {
        return static_cast<int64_t>(qmax()) - static_cast<int64_t>(qmin());
    }

    bool fitsInt8() const { return bits_ <= 8; }
    bool fitsInt16() const { return bits_ <= 16; }
    
    bool operator==(const QuantBitWidth& other) const {
        return bits_ == other.bits_ && is_signed_ == other.is_signed_;
    }
    bool operator!=(const QuantBitWidth& other) const {
        return !(*this == other);
    }
    
    static QuantBitWidth INT8() { return QuantBitWidth(8, true); }
    static QuantBitWidth INT16() { return QuantBitWidth(16, true); }
    static QuantBitWidth INT32() { return QuantBitWidth(32, true); }
    static QuantBitWidth UINT8() { return QuantBitWidth(8, false); }
    static QuantBitWidth UINT16() { return QuantBitWidth(16, false); }
};

struct OperatorQuantConfig {
    QuantBitWidth x_{8, true}, h_{8, true};
    QuantBitWidth W_{8, true}, R_{8, true}, bx_{16, true}, br_{16, true};
    QuantBitWidth Wx_{16, true}, Rh_{16, true};
    QuantBitWidth z_pre_{8, true}, z_out_{8, false};
    QuantBitWidth r_pre_{8, true}, r_out_{8, false};
    QuantBitWidth g_pre_{8, true}, g_out_{8, true};
    QuantBitWidth Rh_add_br_{16, true}, rRh_{16, true};
    QuantBitWidth old_contrib_{16, true}, new_contrib_{16, true};

    bool x_symmetric_ = false, h_symmetric_ = false;
    bool W_symmetric_ = true, R_symmetric_ = true, bx_symmetric_ = true, br_symmetric_ = true;
    bool Wx_symmetric_ = false, Rh_symmetric_ = false;
    bool z_pre_symmetric_ = false, z_out_symmetric_ = false;
    bool r_pre_symmetric_ = false, r_out_symmetric_ = false;
    bool g_pre_symmetric_ = false, g_out_symmetric_ = true;
    bool Rh_add_br_symmetric_ = false, rRh_symmetric_ = false;
    bool old_contrib_symmetric_ = false, new_contrib_symmetric_ = false;

    OperatorQuantConfig& setAllBitWidths(int8_t bits) {
        QuantBitWidth* signed_members[] = {
            &x_, &h_, &W_, &R_, &bx_, &br_, &Wx_, &Rh_,
            &z_pre_, &r_pre_, &g_pre_, &g_out_,
            &Rh_add_br_, &rRh_, &old_contrib_, &new_contrib_
        };
        for (auto* m : signed_members) {
            m->bits_ = bits;
            m->is_signed_ = true;
        }
        z_out_ = {bits, false};
        r_out_ = {bits, false};
        return *this;
    }
};
