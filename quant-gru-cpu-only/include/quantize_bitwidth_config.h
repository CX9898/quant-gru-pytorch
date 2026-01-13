#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

// ============================================================================
//                           量化位宽配置模块
// ============================================================================

// ==================== 量化位宽枚举 ====================

enum class QuantBitWidth : int8_t {
    // 有符号类型（负值）
    INT8 = -8,
    INT16 = -16,
    INT32 = -32,

    // 无符号类型（正值）
    UINT8 = 8,
    UINT16 = 16
};

// ==================== 运行时位宽分发器 ====================

template <typename T>
struct TypeTag {
    using type = T;
};

template <typename Func>
inline auto dispatchByBitWidth(QuantBitWidth bw, Func &&func) -> decltype(func(TypeTag<int8_t>{})) {
    switch (bw) {
        case QuantBitWidth::INT8:
            return func(TypeTag<int8_t>{});
        case QuantBitWidth::INT16:
            return func(TypeTag<int16_t>{});
        case QuantBitWidth::INT32:
            return func(TypeTag<int32_t>{});
        case QuantBitWidth::UINT8:
            return func(TypeTag<uint8_t>{});
        case QuantBitWidth::UINT16:
            return func(TypeTag<uint16_t>{});
        default:
            throw std::invalid_argument("Unknown QuantBitWidth: " +
                                        std::to_string(static_cast<int>(bw)));
    }
}

// ==================== 算子量化配置结构 ====================

struct OperatorQuantConfig {
    // 位宽配置
    QuantBitWidth x_ = QuantBitWidth::INT8;
    QuantBitWidth h_ = QuantBitWidth::INT8;
    QuantBitWidth W_ = QuantBitWidth::INT8;
    QuantBitWidth R_ = QuantBitWidth::INT8;
    QuantBitWidth bw_ = QuantBitWidth::INT8;
    QuantBitWidth br_ = QuantBitWidth::INT8;
    QuantBitWidth Wx_ = QuantBitWidth::INT8;
    QuantBitWidth Rh_ = QuantBitWidth::INT8;
    QuantBitWidth z_pre_ = QuantBitWidth::INT8;
    QuantBitWidth z_out_ = QuantBitWidth::UINT8;
    QuantBitWidth r_pre_ = QuantBitWidth::INT8;
    QuantBitWidth r_out_ = QuantBitWidth::UINT8;
    QuantBitWidth g_pre_ = QuantBitWidth::INT8;
    QuantBitWidth g_out_ = QuantBitWidth::INT8;
    QuantBitWidth Rh_add_br_ = QuantBitWidth::INT8;
    QuantBitWidth rRh_ = QuantBitWidth::INT8;
    QuantBitWidth old_contrib_ = QuantBitWidth::INT8;
    QuantBitWidth new_contrib_ = QuantBitWidth::INT8;

    // 对称量化配置
    bool x_symmetric_ = false;
    bool h_symmetric_ = false;
    bool W_symmetric_ = true;
    bool R_symmetric_ = true;
    bool bw_symmetric_ = true;
    bool br_symmetric_ = true;
    bool Wx_symmetric_ = false;
    bool Rh_symmetric_ = false;
    bool z_pre_symmetric_ = false;
    bool z_out_symmetric_ = false;
    bool r_pre_symmetric_ = false;
    bool r_out_symmetric_ = false;
    bool g_pre_symmetric_ = false;
    bool g_out_symmetric_ = false;
    bool Rh_add_br_symmetric_ = false;
    bool rRh_symmetric_ = false;
    bool old_contrib_symmetric_ = false;
    bool new_contrib_symmetric_ = false;

    OperatorQuantConfig& setAllBitWidths(int bits);
    static OperatorQuantConfig create(int bits);
};

// ==================== 辅助函数 ====================

inline QuantBitWidth bitsToSignedQuantBitWidth(int bits) {
    switch (bits) {
        case 8: return QuantBitWidth::INT8;
        case 16: return QuantBitWidth::INT16;
        case 32: return QuantBitWidth::INT32;
        default:
            throw std::invalid_argument("Unsupported bit width: " + std::to_string(bits));
    }
}

inline QuantBitWidth bitsToUnsignedQuantBitWidth(int bits) {
    switch (bits) {
        case 8: return QuantBitWidth::UINT8;
        case 16: return QuantBitWidth::UINT16;
        default:
            throw std::invalid_argument("Unsupported unsigned bit width: " + std::to_string(bits));
    }
}

inline OperatorQuantConfig& OperatorQuantConfig::setAllBitWidths(int bits) {
    QuantBitWidth signed_bw = bitsToSignedQuantBitWidth(bits);
    QuantBitWidth unsigned_bw = (bits == 32) ? QuantBitWidth::UINT16 : bitsToUnsignedQuantBitWidth(bits);

    x_ = signed_bw; h_ = signed_bw;
    W_ = signed_bw; R_ = signed_bw; bw_ = signed_bw; br_ = signed_bw;
    Wx_ = signed_bw; Rh_ = signed_bw;
    z_pre_ = signed_bw; z_out_ = unsigned_bw;
    r_pre_ = signed_bw; r_out_ = unsigned_bw;
    g_pre_ = signed_bw; g_out_ = signed_bw;
    Rh_add_br_ = signed_bw; rRh_ = signed_bw;
    old_contrib_ = signed_bw; new_contrib_ = signed_bw;

    return *this;
}

inline OperatorQuantConfig OperatorQuantConfig::create(int bits) {
    OperatorQuantConfig config;
    config.setAllBitWidths(bits);
    return config;
}

