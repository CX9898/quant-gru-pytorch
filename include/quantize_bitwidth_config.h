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
 * - z_out_ 和 r_out_ 默认使用 UINT，因为 sigmoid 输出范围是 [0, 1]
 * - Python 端只配置位宽数量，C++ 的 to_cpp() 会自动为 z_out/r_out 选择 UINT 类型
 *
 * 使用方式：
 * - 默认构造使用 INT8/UINT8
 * - 调用 setAllBitWidths(bits) 设置所有位宽为指定值
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

    // ==================== 位宽设置接口 ====================

    /**
     * @brief 设置所有位宽为指定值
     *
     * 自动将有符号类型设置为 INTn，无符号类型（z_out, r_out）设置为 UINTn
     *
     * @param bits 位宽数值 (8 或 16)，32 位仅用于有符号类型
     * @return 返回自身引用，支持链式调用
     *
     * 使用示例：
     * @code
     * OperatorQuantConfig config;
     * config.setAllBitWidths(16);  // 全部设为 16 位
     * @endcode
     */
    OperatorQuantConfig& setAllBitWidths(int bits);

    /**
     * @brief 创建指定位宽的配置（静态工厂方法）
     * @param bits 位宽数值 (8, 16, 32)
     * @return 配置好的 OperatorQuantConfig 实例
     *
     * 使用示例：
     * @code
     * auto config = OperatorQuantConfig::create(16);
     * @endcode
     */
    static OperatorQuantConfig create(int bits);

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

// ==================== 辅助函数 ====================

/**
 * @brief 将位宽数值转换为有符号量化枚举
 * @param bits 位宽数值 (8, 16, 32)
 * @return 对应的有符号 QuantBitWidth 枚举
 */
 inline QuantBitWidth bitsToSignedQuantBitWidth(int bits) {
    switch (bits) {
        case 8:
            return QuantBitWidth::INT8;
        case 16:
            return QuantBitWidth::INT16;
        case 32:
            return QuantBitWidth::INT32;
        default:
            throw std::invalid_argument("Unsupported bit width: " + std::to_string(bits) +
                                        ". Supported values are 8, 16, 32.");
    }
}

/**
 * @brief 将位宽数值转换为无符号量化枚举
 * @param bits 位宽数值 (8, 16)
 * @return 对应的无符号 QuantBitWidth 枚举
 */
inline QuantBitWidth bitsToUnsignedQuantBitWidth(int bits) {
    switch (bits) {
        case 8:
            return QuantBitWidth::UINT8;
        case 16:
            return QuantBitWidth::UINT16;
        default:
            throw std::invalid_argument("Unsupported unsigned bit width: " + std::to_string(bits) +
                                        ". Supported values are 8, 16.");
    }
}

// ==================== OperatorQuantConfig 方法实现 ====================

inline OperatorQuantConfig& OperatorQuantConfig::setAllBitWidths(int bits) {
    QuantBitWidth signed_bw = bitsToSignedQuantBitWidth(bits);
    // 对于 32 位，无符号类型回退到 16 位（因为没有 UINT32）
    QuantBitWidth unsigned_bw = (bits == 32) ? QuantBitWidth::UINT16 : bitsToUnsignedQuantBitWidth(bits);

    // 输入
    x_ = signed_bw;
    h_ = signed_bw;

    // 权重
    W_ = signed_bw;
    R_ = signed_bw;
    bx_ = signed_bw;
    br_ = signed_bw;

    // 矩阵乘法结果
    Wx_ = signed_bw;
    Rh_ = signed_bw;

    // 门控 - 更新门
    z_pre_ = signed_bw;
    z_out_ = unsigned_bw;  // sigmoid 输出使用无符号

    // 门控 - 重置门
    r_pre_ = signed_bw;
    r_out_ = unsigned_bw;  // sigmoid 输出使用无符号

    // 门控 - 候选门
    g_pre_ = signed_bw;
    g_out_ = signed_bw;

    // 中间运算
    Rh_add_br_ = signed_bw;
    rRh_ = signed_bw;
    old_contrib_ = signed_bw;
    new_contrib_ = signed_bw;

    return *this;
}

inline OperatorQuantConfig OperatorQuantConfig::create(int bits) {
    OperatorQuantConfig config;
    config.setAllBitWidths(bits);
    return config;
}
