#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

// ============================================================================
//                           量化位宽配置模块
// ============================================================================
//
// 设计原则：
// 1. 位宽配置：Python 端只配置位宽数量（8/16/32），C++ 端决定实际类型
// 2. 类型决策：大多数操作使用 INT 类型，sigmoid 输出（z_out/r_out）使用 UINT 类型
// 3. 对称量化：is_symmetric 只影响 zero_point 计算，与位宽类型完全解耦
//
// ============================================================================

// ==================== 量化位宽枚举 ====================

/**
 * @brief 量化位宽枚举
 *
 * 枚举值设计：
 * - 负值表示有符号类型（INT8, INT16, INT32）
 * - 正值表示无符号类型（UINT8, UINT16）
 *
 * 使用场景：
 * - INT 类型：大多数操作（输入、权重、中间计算等）
 * - UINT 类型：sigmoid 输出（z_out, r_out），因为输出范围是 [0, 1]
 */
enum class QuantBitWidth : int8_t {
    // 有符号类型（负值）
    INT8 = -8,    // 8 位有符号，范围 [-128, 127]
    INT16 = -16,  // 16 位有符号，范围 [-32768, 32767]
    INT32 = -32,  // 32 位有符号，用于中间累加

    // 无符号类型（正值）
    UINT8 = 8,   // 8 位无符号，范围 [0, 255]，用于 sigmoid 输出
    UINT16 = 16  // 16 位无符号，范围 [0, 65535]，用于 sigmoid 输出
};

// ==================== 运行时位宽分发器 ====================

/**
 * @brief 类型标签，用于在运行时传递类型信息到模板
 * @tparam T 要传递的类型
 */
template <typename T>
struct TypeTag {
    using type = T;
};

/**
 * @brief 通用位宽分发器
 *
 * 根据运行时位宽枚举值调用正确的模板实例，实现运行时多态到编译时多态的转换。
 *
 * @tparam Func 可调用对象类型
 * @param bw 运行时位宽枚举
 * @param func 要调用的模板函数包装器
 * @return 模板函数的返回值
 *
 * 使用示例：
 * @code
 * auto result = dispatchByBitWidth(QuantBitWidth::INT8, [&](auto type_tag) {
 *     using QuantT = typename decltype(type_tag)::type;
 *     return someFunction<QuantT>(args...);
 * });
 * @endcode
 */
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

/**
 * @brief GRU 算子量化配置
 *
 * 包含两类配置：
 * 1. 位宽配置：每个算子的量化位宽（由 C++ 决定实际类型）
 * 2. 对称量化配置：每个算子是否使用对称量化（只影响 zero_point）
 *
 * 特殊说明：
 * - z_out_ 和 r_out_ 默认使用 UINT8，因为 sigmoid 输出范围是 [0, 1]
 * - Python 端只配置位宽数量，C++ 的 to_cpp() 会自动为 z_out/r_out 选择 UINT 类型
 */
struct OperatorQuantConfig {
    // ==================== 位宽配置 ====================
    // 大多数操作使用 INT 类型，z_out/r_out 使用 UINT 类型（sigmoid 输出 [0,1]）

    QuantBitWidth x_ = QuantBitWidth::INT8;  // 输入序列 x
    QuantBitWidth h_ = QuantBitWidth::INT8;  // 隐藏状态 h

    // 权重
    QuantBitWidth W_ = QuantBitWidth::INT8;   // 输入权重 W (input -> gates)
    QuantBitWidth R_ = QuantBitWidth::INT8;   // 循环权重 R (hidden -> gates)
    QuantBitWidth bx_ = QuantBitWidth::INT8;  // 输入偏置 bx
    QuantBitWidth br_ = QuantBitWidth::INT8;  // 循环偏置 br

    // 矩阵乘法结果
    QuantBitWidth Wx_ = QuantBitWidth::INT8;  // W @ x 结果
    QuantBitWidth Rh_ = QuantBitWidth::INT8;  // R @ h 结果

    // 门控 - 更新门 (update gate)
    QuantBitWidth z_pre_ = QuantBitWidth::INT8;   // sigmoid 前
    QuantBitWidth z_out_ = QuantBitWidth::UINT8;  // sigmoid 后 [0,1]，使用 UINT

    // 门控 - 重置门 (reset gate)
    QuantBitWidth r_pre_ = QuantBitWidth::INT8;   // sigmoid 前
    QuantBitWidth r_out_ = QuantBitWidth::UINT8;  // sigmoid 后 [0,1]，使用 UINT

    // 门控 - 候选门 (candidate gate)
    QuantBitWidth g_pre_ = QuantBitWidth::INT8;  // tanh 前
    QuantBitWidth g_out_ = QuantBitWidth::INT8;  // tanh 后 [-1,1]

    // 中间运算
    QuantBitWidth Rh_add_br_ = QuantBitWidth::INT8;    // Rh + br
    QuantBitWidth rRh_ = QuantBitWidth::INT8;          // r × Rh
    QuantBitWidth old_contrib_ = QuantBitWidth::INT8;  // z × h[t-1]
    QuantBitWidth new_contrib_ = QuantBitWidth::INT8;  // (1-z) × g

    // ==================== 对称量化配置 ====================
    // is_symmetric 只影响 zero_point 的计算：
    //   - true:  对称量化，zp = 0（适用于对称分布，如权重、tanh 输出）
    //   - false: 非对称量化，zp ≠ 0（适用于非对称分布，如 sigmoid 输出 [0,1]）

    // 输入（通常非对称）
    bool x_symmetric_ = false;  // 输入 x
    bool h_symmetric_ = false;  // 隐藏状态 h

    // 权重（通常对称）
    bool W_symmetric_ = true;   // 权重 W
    bool R_symmetric_ = true;   // 权重 R
    bool bx_symmetric_ = true;  // 偏置 bx
    bool br_symmetric_ = true;  // 偏置 br

    // 矩阵乘法结果
    bool Wx_symmetric_ = false;  // W @ x
    bool Rh_symmetric_ = false;  // R @ h

    // 门控 - 更新门
    bool z_pre_symmetric_ = false;  // sigmoid 前（可正可负）
    bool z_out_symmetric_ = false;  // sigmoid 后 [0,1]（非对称）

    // 门控 - 重置门
    bool r_pre_symmetric_ = false;  // sigmoid 前（可正可负）
    bool r_out_symmetric_ = false;  // sigmoid 后 [0,1]（非对称）

    // 门控 - 候选门
    bool g_pre_symmetric_ = false;  // tanh 前（可正可负）
    bool g_out_symmetric_ = false;  // tanh 后 [-1,1]（理论上对称，但实际可能偏移）

    // 中间运算
    bool Rh_add_br_symmetric_ = false;    // Rh + br
    bool rRh_symmetric_ = false;          // r × Rh
    bool old_contrib_symmetric_ = false;  // z × h
    bool new_contrib_symmetric_ = false;  // (1-z) × g
};
