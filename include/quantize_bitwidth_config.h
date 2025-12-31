#pragma once

#include <cstdint>
#include "cuda_compat.h"

// ============================================================================
//                           量化位宽配置模块
// ============================================================================
//
// 设计原则：
// 1. 任意位宽：支持 1-31 位任意位宽量化
// 2. 统一存储：所有量化值使用 int32_t 存储，位宽仅控制 clamp 范围
// 3. 符号类型：C++ 端根据算子类型自动决定（sigmoid 输出无符号，其他有符号）
// 4. 对称量化：is_symmetric 只影响 zero_point 计算，与位宽完全解耦
//
// ============================================================================

/**
 * @brief 量化位宽配置
 *
 * 支持任意 1-31 位的量化位宽。
 * 量化范围通过 qmin()/qmax() 动态计算。
 */
struct QuantBitWidth {
    int8_t bits_ = 8;        ///< 位宽：1-32
    bool is_signed_ = true;  ///< 有符号/无符号（32位时强制有符号）

    QuantBitWidth() = default;
    __host__ __device__ QuantBitWidth(int8_t b, bool s = true) 
        : bits_(b), is_signed_(b >= 32 ? true : s) {}  // 32位强制有符号

    /// 量化最小值
    __host__ __device__ int32_t qmin() const {
        // 32位有符号: -2^31，使用 INT32_MIN 避免溢出
        if (bits_ >= 32) return static_cast<int32_t>(0x80000000);
        return is_signed_ ? -(1 << (bits_ - 1)) : 0;
    }
    
    /// 量化最大值
    __host__ __device__ int32_t qmax() const {
        // 32位有符号: 2^31-1
        if (bits_ >= 32) return 0x7FFFFFFF;
        return is_signed_ ? (1 << (bits_ - 1)) - 1 : (1 << bits_) - 1;
    }
    
    /// 量化范围大小
    __host__ __device__ int64_t range() const {
        return static_cast<int64_t>(qmax()) - static_cast<int64_t>(qmin());
    }

    /// 是否适合 INT8 GEMM
    bool fitsInt8() const { return bits_ <= 8; }
    
    /// 是否适合 INT16 GEMM
    bool fitsInt16() const { return bits_ <= 16; }
    
    /// 比较运算符
    bool operator==(const QuantBitWidth& other) const {
        return bits_ == other.bits_ && is_signed_ == other.is_signed_;
    }
    bool operator!=(const QuantBitWidth& other) const {
        return !(*this == other);
    }
};

// ==================== 算子量化配置结构 ====================

/**
 * @brief GRU 算子量化配置
 *
 * z_out/r_out 默认无符号（sigmoid 输出），其他默认有符号
 */
struct OperatorQuantConfig {
    // ==================== 位宽配置 ====================
    QuantBitWidth x_{8, true}, h_{8, true};                          // 输入/状态
    QuantBitWidth W_{8, true}, R_{8, true}, bx_{8, true}, br_{8, true};  // 权重
    QuantBitWidth Wx_{8, true}, Rh_{8, true};                        // GEMM 输出
    QuantBitWidth z_pre_{8, true}, z_out_{8, false};                 // 更新门（z_out 无符号）
    QuantBitWidth r_pre_{8, true}, r_out_{8, false};                 // 重置门（r_out 无符号）
    QuantBitWidth g_pre_{8, true}, g_out_{8, true};                  // 候选门
    QuantBitWidth Rh_add_br_{8, true}, rRh_{8, true};                // 中间运算
    QuantBitWidth old_contrib_{8, true}, new_contrib_{8, true};      // 输出计算

    // ==================== 对称量化配置 ====================
    bool x_symmetric_ = false, h_symmetric_ = false;
    bool W_symmetric_ = true, R_symmetric_ = true, bx_symmetric_ = true, br_symmetric_ = true;
    bool Wx_symmetric_ = false, Rh_symmetric_ = false;
    bool z_pre_symmetric_ = false, z_out_symmetric_ = false;
    bool r_pre_symmetric_ = false, r_out_symmetric_ = false;
    bool g_pre_symmetric_ = false, g_out_symmetric_ = true;
    bool Rh_add_br_symmetric_ = false, rRh_symmetric_ = false;
    bool old_contrib_symmetric_ = false, new_contrib_symmetric_ = false;

    // ==================== 位宽设置接口 ====================
    
    /// 设置所有位宽（保持 z_out/r_out 无符号）
    OperatorQuantConfig& setAllBitWidths(int8_t bits) {
        // 遍历所有位宽成员并设置
        QuantBitWidth* signed_members[] = {
            &x_, &h_, &W_, &R_, &bx_, &br_, &Wx_, &Rh_,
            &z_pre_, &r_pre_, &g_pre_, &g_out_,
            &Rh_add_br_, &rRh_, &old_contrib_, &new_contrib_
        };
        for (auto* m : signed_members) {
            m->bits_ = bits;
            m->is_signed_ = true;
        }
        // sigmoid 输出：无符号
        z_out_ = {bits, false};
        r_out_ = {bits, false};
        return *this;
    }

    /// 创建指定位宽的配置
    static OperatorQuantConfig create(int8_t bits) {
        OperatorQuantConfig cfg;
        cfg.setAllBitWidths(bits);
        return cfg;
    }
};
