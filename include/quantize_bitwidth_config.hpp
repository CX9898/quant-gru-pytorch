#pragma once

#include <cstdint>
#include <type_traits>

// ==================== 量化位宽配置 ====================

// 量化位宽枚举
// 注意：使用负值表示无符号类型，便于区分
enum class QuantBitWidth : int8_t {
    INT8 = 8,
    INT16 = 16,
    INT32 = 32,   // 用于中间累加
    UINT8 = -8,   // 无符号 8 位
    UINT16 = -16  // 无符号 16 位
};

// 根据位宽选择量化类型
template<QuantBitWidth BW>
struct QuantTypeSelector;

template<>
struct QuantTypeSelector<QuantBitWidth::INT8> {
    using type = int8_t;
};

template<>
struct QuantTypeSelector<QuantBitWidth::INT16> {
    using type = int16_t;
};

template<>
struct QuantTypeSelector<QuantBitWidth::INT32> {
    using type = int32_t;
};

template<>
struct QuantTypeSelector<QuantBitWidth::UINT8> {
    using type = uint8_t;
};

template<>
struct QuantTypeSelector<QuantBitWidth::UINT16> {
    using type = uint16_t;
};

// 类型别名：根据位宽获取对应的量化类型
template<QuantBitWidth BW>
using QuantType = typename QuantTypeSelector<BW>::type;

// 算子量化位宽配置结构
struct OperatorQuantConfig {
    QuantBitWidth x_bitwidth = QuantBitWidth::INT8;           // 输入 x
    QuantBitWidth h_bitwidth = QuantBitWidth::INT8;           // 隐藏状态 h
    QuantBitWidth W_bitwidth = QuantBitWidth::INT8;           // 权重 W
    QuantBitWidth R_bitwidth = QuantBitWidth::INT8;           // 权重 R
    QuantBitWidth bx_bitwidth = QuantBitWidth::INT32;         // 偏置 bx (通常用 int32)
    QuantBitWidth br_bitwidth = QuantBitWidth::INT32;         // 偏置 br (通常用 int32)
    QuantBitWidth Wx_bitwidth = QuantBitWidth::INT32;         // Wx 中间结果
    QuantBitWidth Rh_bitwidth = QuantBitWidth::INT32;         // Rh 中间结果
    QuantBitWidth z_pre_bitwidth = QuantBitWidth::INT8;       // z 门输入
    QuantBitWidth z_out_bitwidth = QuantBitWidth::INT8;       // z 门输出
    QuantBitWidth r_pre_bitwidth = QuantBitWidth::INT8;       // r 门输入
    QuantBitWidth r_out_bitwidth = QuantBitWidth::INT8;       // r 门输出
    QuantBitWidth g_pre_bitwidth = QuantBitWidth::INT8;       // g 门输入
    QuantBitWidth g_out_bitwidth = QuantBitWidth::INT8;       // g 门输出
    QuantBitWidth Rh_add_br_bitwidth = QuantBitWidth::INT32;  // Rh + br
    QuantBitWidth rRh_bitwidth = QuantBitWidth::INT32;        // r × Rh
    QuantBitWidth one_minus_update_bitwidth = QuantBitWidth::INT8;
    QuantBitWidth old_contrib_bitwidth = QuantBitWidth::INT32;
    QuantBitWidth new_contrib_bitwidth = QuantBitWidth::INT32;

    // 默认构造函数：使用 int8 量化
    OperatorQuantConfig() = default;

    // 验证配置合理性
    bool validate() const {
        // 检查位宽是否在合理范围内
        // 中间结果通常需要更高精度
        return true;
    }
};
