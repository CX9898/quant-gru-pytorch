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
    // ==================== 位宽配置 ====================
    QuantBitWidth x_ = QuantBitWidth::INT8;                 // 输入 x
    QuantBitWidth h_ = QuantBitWidth::INT8;                 // 隐藏状态 h
    QuantBitWidth W_ = QuantBitWidth::INT8;                 // 权重 W
    QuantBitWidth R_ = QuantBitWidth::INT8;                 // 权重 R
    QuantBitWidth bx_ = QuantBitWidth::INT8;                // 偏置 bx
    QuantBitWidth br_ = QuantBitWidth::INT8;                // 偏置 br
    QuantBitWidth Wx_ = QuantBitWidth::INT8;                // Wx 结果
    QuantBitWidth Rh_ = QuantBitWidth::INT8;                // Rh 结果
    QuantBitWidth z_pre_ = QuantBitWidth::INT8;             // z 门输入
    QuantBitWidth z_out_ = QuantBitWidth::UINT8;            // z 门输出
    QuantBitWidth r_pre_ = QuantBitWidth::INT8;             // r 门输入
    QuantBitWidth r_out_ = QuantBitWidth::UINT8;            // r 门输出
    QuantBitWidth g_pre_ = QuantBitWidth::INT8;             // g 门输入
    QuantBitWidth g_out_ = QuantBitWidth::INT8;             // g 门输出
    QuantBitWidth Rh_add_br_ = QuantBitWidth::INT8;         // Rh + br
    QuantBitWidth rRh_ = QuantBitWidth::INT8;               // r × Rh
    QuantBitWidth one_minus_update_ = QuantBitWidth::INT8;  // 1 - z
    QuantBitWidth old_contrib_ = QuantBitWidth::INT8;       // z * h[output_idx]
    QuantBitWidth new_contrib_ = QuantBitWidth::INT8;       // (1.0 - z) * g

    // ==================== 对称量化配置 ====================
    // is_symmetric = true:  对称量化，zero_point = 0，适用于数据分布关于 0 对称的情况
    // is_symmetric = false: 非对称量化，zero_point ≠ 0，适用于数据分布不对称的情况
    bool x_symmetric_ = false;                 // 输入 x
    bool h_symmetric_ = false;                 // 隐藏状态 h
    bool W_symmetric_ = true;                 // 权重 W：权重通常对称
    bool R_symmetric_ = true;                 // 权重 R：权重通常对称
    bool bx_symmetric_ = true;                // 偏置 bx：偏置通常对称
    bool br_symmetric_ = true;                // 偏置 br：偏置通常对称
    bool Wx_symmetric_ = false;                // Wx 结果
    bool Rh_symmetric_ = false;                // Rh 结果
    bool z_pre_symmetric_ = false;             // z 门输入：sigmoid 前可正可负
    bool z_out_symmetric_ = false;            // z 门输出：sigmoid 后 [0,1]，非对称
    bool r_pre_symmetric_ = false;             // r 门输入：sigmoid 前可正可负
    bool r_out_symmetric_ = false;            // r 门输出：sigmoid 后 [0,1]，非对称
    bool g_pre_symmetric_ = false;             // g 门输入：tanh 前可正可负
    bool g_out_symmetric_ = false;             // g 门输出：tanh 后 [-1,1]，对称
    bool Rh_add_br_symmetric_ = false;         // Rh + br
    bool rRh_symmetric_ = false;               // r × Rh
    bool one_minus_update_symmetric_ = false;  // 1 - z：范围 [0,1]，可考虑非对称
    bool old_contrib_symmetric_ = false;       // z * h
    bool new_contrib_symmetric_ = false;       // (1.0 - z) * g
};
