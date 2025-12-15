#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

// ==================== 量化位宽配置 ====================

// 量化位宽枚举
// 注意：使用负值表示无符号类型，便于区分
enum class QuantBitWidth : int8_t {
    INT8 = -8,
    INT16 = -16,
    INT32 = -32,  // 用于中间累加
    UINT8 = 8,    // 无符号 8 位
    UINT16 = 16   // 无符号 16 位
};

// ==================== 运行时位宽分发器 ====================
// 核心：根据运行时位宽值调用正确的模板实例

/**
 * @brief 通用位宽分发器
 * @tparam Func 可调用对象类型，签名为 template<typename QuantT> ReturnType(Args...)
 * @param bw 运行时位宽枚举
 * @param func 要调用的模板函数包装器
 *
 * 使用示例：
 * ```cpp
 * auto result = dispatchByBitWidth(QuantBitWidth::INT8, [&](auto type_tag) {
 *     using QuantT = typename decltype(type_tag)::type;
 *     return someFunction<QuantT>(args...);
 * });
 * ```
 */

// 类型标签，用于传递类型信息
template <typename T>
struct TypeTag {
    using type = T;
};

// 分发器实现
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

// ==================== 算子量化位宽配置结构 ====================
struct OperatorQuantConfig {
    QuantBitWidth x_ = QuantBitWidth::INT8;                 // 输入 x
    QuantBitWidth h_ = QuantBitWidth::INT8;                 // 隐藏状态 h
    QuantBitWidth W_ = QuantBitWidth::INT8;                 // 权重 W
    QuantBitWidth R_ = QuantBitWidth::INT8;                 // 权重 R
    QuantBitWidth bx_ = QuantBitWidth::INT8;                // 偏置 bx
    QuantBitWidth br_ = QuantBitWidth::INT8;                // 偏置 br
    QuantBitWidth Wx_ = QuantBitWidth::INT8;                // Wx 结果
    QuantBitWidth Rh_ = QuantBitWidth::INT8;                // Rh 结果
    QuantBitWidth z_pre_ = QuantBitWidth::INT8;             // z 门输入
    QuantBitWidth z_out_ = QuantBitWidth::UINT8;             // z 门输出
    QuantBitWidth r_pre_ = QuantBitWidth::INT8;             // r 门输入
    QuantBitWidth r_out_ = QuantBitWidth::UINT8;             // r 门输出
    QuantBitWidth g_pre_ = QuantBitWidth::INT8;             // g 门输入
    QuantBitWidth g_out_ = QuantBitWidth::INT8;             // g 门输出
    QuantBitWidth Rh_add_br_ = QuantBitWidth::INT8;         // Rh + br
    QuantBitWidth rRh_ = QuantBitWidth::INT8;               // r × Rh
    QuantBitWidth old_contrib_ = QuantBitWidth::INT8;       // z * h[output_idx]
    QuantBitWidth new_contrib_ = QuantBitWidth::INT8;       // (1.0 - z) * g
};
