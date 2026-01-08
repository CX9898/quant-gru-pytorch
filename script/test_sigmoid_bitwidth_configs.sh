#!/bin/bash
# 完整的位宽和对称配置枚举测试脚本
# 测试内容：
# 1. 所有64种激活位宽配置组合 (2^6)
# 2. 位宽配置与对称/非对称的组合测试

set -e

# 自动获取项目根目录（脚本位于 script/ 目录下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/include/quantize_bitwidth_config.h"
BUILD_DIR="$PROJECT_DIR/build"

# 保存原始配置内容（不创建备份文件）
ORIGINAL_CONFIG=$(cat "$CONFIG_FILE")

# 结果文件
RESULT_FILE="$PROJECT_DIR/test_sigmoid_results_full.txt"
CSV_FILE="$PROJECT_DIR/test_sigmoid_results_full.csv"

echo "===== 完整位宽和对称配置测试结果 =====" > "$RESULT_FILE"
echo "测试时间: $(date)" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# CSV 头
echo "config_name,z_pre,z_out,r_pre,r_out,g_pre,g_out,z_pre_sym,z_out_sym,r_pre_sym,r_out_sym,g_pre_sym,g_out_sym,mse,cosine_similarity" > "$CSV_FILE"

# 计数器
TEST_COUNT=0
TOTAL_TESTS=0
PASS_COUNT=0
FAIL_COUNT=0

# 阈值设置（必须同时满足才算通过）
COSINE_THRESHOLD=0.999    # 余弦相似度 >= 此值
MSE_THRESHOLD=1e-4        # MSE <= 此值

# 函数：修改配置文件中的位宽
# 新格式: QuantBitWidth z_pre_{bits, is_unsigned}
modify_bitwidth() {
    local z_pre=$1
    local z_out=$2
    local r_pre=$3
    local r_out=$4
    local g_pre=$5
    local g_out=$6
    
    # z_pre, r_pre, g_pre, g_out: 有符号 (true)
    # z_out, r_out: 无符号 (false)，sigmoid 输出
    sed -i "s/z_pre_{[0-9]*, [a-z]*}/z_pre_{${z_pre}, true}/g" "$CONFIG_FILE"
    sed -i "s/z_out_{[0-9]*, [a-z]*}/z_out_{${z_out}, false}/g" "$CONFIG_FILE"
    sed -i "s/r_pre_{[0-9]*, [a-z]*}/r_pre_{${r_pre}, true}/g" "$CONFIG_FILE"
    sed -i "s/r_out_{[0-9]*, [a-z]*}/r_out_{${r_out}, false}/g" "$CONFIG_FILE"
    sed -i "s/g_pre_{[0-9]*, [a-z]*}/g_pre_{${g_pre}, true}/g" "$CONFIG_FILE"
    sed -i "s/g_out_{[0-9]*, [a-z]*}/g_out_{${g_out}, true}/g" "$CONFIG_FILE"
}

# 函数：修改对称配置
modify_symmetric() {
    local z_pre_sym=$1
    local z_out_sym=$2
    local r_pre_sym=$3
    local r_out_sym=$4
    local g_pre_sym=$5
    local g_out_sym=$6
    
    sed -i "s/bool z_pre_symmetric_ = [a-z]*;/bool z_pre_symmetric_ = ${z_pre_sym};/" "$CONFIG_FILE"
    sed -i "s/bool z_out_symmetric_ = [a-z]*;/bool z_out_symmetric_ = ${z_out_sym};/" "$CONFIG_FILE"
    sed -i "s/bool r_pre_symmetric_ = [a-z]*;/bool r_pre_symmetric_ = ${r_pre_sym};/" "$CONFIG_FILE"
    sed -i "s/bool r_out_symmetric_ = [a-z]*;/bool r_out_symmetric_ = ${r_out_sym};/" "$CONFIG_FILE"
    sed -i "s/bool g_pre_symmetric_ = [a-z]*;/bool g_pre_symmetric_ = ${g_pre_sym};/" "$CONFIG_FILE"
    sed -i "s/bool g_out_symmetric_ = [a-z]*;/bool g_out_symmetric_ = ${g_out_sym};/" "$CONFIG_FILE"
}

# 函数：编译并运行测试
run_test() {
    local config_name=$1
    local z_pre=$2
    local z_out=$3
    local r_pre=$4
    local r_out=$5
    local g_pre=$6
    local g_out=$7
    local z_pre_sym=$8
    local z_out_sym=$9
    local r_pre_sym=${10}
    local r_out_sym=${11}
    local g_pre_sym=${12}
    local g_out_sym=${13}
    
    TEST_COUNT=$((TEST_COUNT + 1))
    
    echo "[$TEST_COUNT/$TOTAL_TESTS] 测试: $config_name"
    
    # 重新编译（静默模式）
    cd "$BUILD_DIR"
    if ! make -j$(nproc) gru_example > /dev/null 2>&1; then
        echo "  ❌ 编译失败"
        echo "配置: $config_name" >> "$RESULT_FILE"
        echo "  状态: 编译失败" >> "$RESULT_FILE"
        echo "" >> "$RESULT_FILE"
        echo "$config_name,$z_pre,$z_out,$r_pre,$r_out,$g_pre,$g_out,$z_pre_sym,$z_out_sym,$r_pre_sym,$r_out_sym,$g_pre_sym,$g_out_sym,COMPILE_ERROR,COMPILE_ERROR" >> "$CSV_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    
    # 运行测试并提取结果
    local output
    local exit_code=0
    output=$(./gru_example 2>&1) || exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        local error_msg=$(echo "$output" | grep -i "unsupported\|error\|exception" | head -1 | tr ',' ';')
        if [ -z "$error_msg" ]; then
            error_msg="Runtime error (exit code: $exit_code)"
        fi
        echo "  ❌ 运行失败: $error_msg"
        echo "配置: $config_name" >> "$RESULT_FILE"
        echo "  状态: 运行失败 - $error_msg" >> "$RESULT_FILE"
        echo "" >> "$RESULT_FILE"
        echo "$config_name,$z_pre,$z_out,$r_pre,$r_out,$g_pre,$g_out,$z_pre_sym,$z_out_sym,$r_pre_sym,$r_out_sym,$g_pre_sym,$g_out_sym,RUNTIME_ERROR,RUNTIME_ERROR" >> "$CSV_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    
    local mse=$(echo "$output" | grep "Overall H: MSE" | head -1 | sed 's/.*MSE = \([0-9.e+-]*\),.*/\1/')
    local cos=$(echo "$output" | grep "Overall H: MSE" | head -1 | sed 's/.*Cosine Similarity = \([0-9.]*\)/\1/')
    
    # 处理空值
    if [ -z "$mse" ]; then mse="N/A"; fi
    if [ -z "$cos" ]; then cos="N/A"; fi
    
    # 判断是否同时满足 MSE 和余弦相似度阈值
    local passed=false
    local cos_ok=false
    local mse_ok=false
    local fail_reason=""
    
    if [ "$cos" != "N/A" ] && [ "$mse" != "N/A" ]; then
        # 检查余弦相似度
        if awk -v val="$cos" -v threshold="$COSINE_THRESHOLD" 'BEGIN {exit !(val >= threshold)}'; then
            cos_ok=true
        fi
        # 检查 MSE
        if awk -v val="$mse" -v threshold="$MSE_THRESHOLD" 'BEGIN {exit !(val <= threshold)}'; then
            mse_ok=true
        fi
        
        if $cos_ok && $mse_ok; then
            passed=true
            echo "  ✓ MSE: $mse (<= $MSE_THRESHOLD), Cosine: $cos (>= $COSINE_THRESHOLD)"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            # 构建失败原因
            if ! $cos_ok; then
                fail_reason="Cosine < $COSINE_THRESHOLD"
            fi
            if ! $mse_ok; then
                [ -n "$fail_reason" ] && fail_reason="$fail_reason, "
                fail_reason="${fail_reason}MSE > $MSE_THRESHOLD"
            fi
            echo "  ✗ MSE: $mse, Cosine: $cos ($fail_reason)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "  ✗ MSE: $mse, Cosine: $cos (无法提取结果)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    # 记录到结果文件
    echo "配置: $config_name" >> "$RESULT_FILE"
    echo "  位宽: z_pre=$z_pre, z_out=$z_out, r_pre=$r_pre, r_out=$r_out, g_pre=$g_pre, g_out=$g_out" >> "$RESULT_FILE"
    echo "  对称: z_pre=$z_pre_sym, z_out=$z_out_sym, r_pre=$r_pre_sym, r_out=$r_out_sym, g_pre=$g_pre_sym, g_out=$g_out_sym" >> "$RESULT_FILE"
    echo "  MSE: $mse, Cosine Similarity: $cos" >> "$RESULT_FILE"
    echo "" >> "$RESULT_FILE"
    
    # 记录到 CSV
    echo "$config_name,$z_pre,$z_out,$r_pre,$r_out,$g_pre,$g_out,$z_pre_sym,$z_out_sym,$r_pre_sym,$r_out_sym,$g_pre_sym,$g_out_sym,$mse,$cos" >> "$CSV_FILE"
}

# ==================== 第一部分：8-16位完整基准测试 ====================
echo ""
echo "==================== 第一部分：8-16位完整基准测试 ===================="
echo ""
echo "==================== 第一部分：8-16位完整基准测试 ====================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 位宽选项
BITWIDTHS_8_TO_16=(8 9 10 11 12 13 14 15 16)
BITWIDTHS_STANDARD=(8 16)
BITWIDTHS_EXTENDED=(4 10 12 24)

# 计算总测试数
# 8-16位基准测试(9) + 标准64种位宽配置 + 扩展位宽测试(约20种) + 64种对称配置 + 16种典型组合
TOTAL_TESTS=$((9 + 64 + 20 + 64 + 16))

echo "预计总测试数: $TOTAL_TESTS"
echo ""

# 设置默认对称配置（全部非对称）
modify_symmetric false false false false false false

# 测试 8-16 位之间所有位宽的统一配置
for bits in "${BITWIDTHS_8_TO_16[@]}"; do
    config_name="BW_FULL${bits}"
    modify_bitwidth $bits $bits $bits $bits $bits $bits
    run_test "$config_name" $bits $bits $bits $bits $bits $bits false false false false false false
done

# ==================== 第1.1部分：标准64种位宽配置 ====================
echo ""
echo "==================== 第1.1部分：标准64种激活位宽配置 (8/16位) ===================="
echo ""
echo "==================== 第1.1部分：标准64种激活位宽配置 ====================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 枚举所有64种标准位宽配置 (8/16位)
for z_pre in "${BITWIDTHS_STANDARD[@]}"; do
    for z_out in "${BITWIDTHS_STANDARD[@]}"; do
        for r_pre in "${BITWIDTHS_STANDARD[@]}"; do
            for r_out in "${BITWIDTHS_STANDARD[@]}"; do
                for g_pre in "${BITWIDTHS_STANDARD[@]}"; do
                    for g_out in "${BITWIDTHS_STANDARD[@]}"; do
                        config_name="BW_z${z_pre}${z_out}_r${r_pre}${r_out}_g${g_pre}${g_out}"
                        modify_bitwidth $z_pre $z_out $r_pre $r_out $g_pre $g_out
                        run_test "$config_name" $z_pre $z_out $r_pre $r_out $g_pre $g_out false false false false false false
                    done
                done
            done
        done
    done
done

# ==================== 第1.5部分：扩展位宽配置测试 ====================
echo ""
echo "==================== 第1.5部分：扩展位宽配置 (4/10/12/24位) ===================="
echo ""
echo "==================== 第1.5部分：扩展位宽配置 ====================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 低位宽测试 (4位)
for gate_bits in 4 8 16; do
    config_name="BW_LOW4_gate${gate_bits}"
    modify_bitwidth 4 4 4 4 $gate_bits $gate_bits
    run_test "$config_name" 4 4 4 4 $gate_bits $gate_bits false false false false false false
done

# 非标准位宽测试 (10位)
for gate_bits in 8 10 16; do
    config_name="BW_MID10_gate${gate_bits}"
    modify_bitwidth 10 10 10 10 $gate_bits $gate_bits
    run_test "$config_name" 10 10 10 10 $gate_bits $gate_bits false false false false false false
done

# 12位位宽测试
for gate_bits in 8 12 16; do
    config_name="BW_MID12_gate${gate_bits}"
    modify_bitwidth 12 12 12 12 $gate_bits $gate_bits
    run_test "$config_name" 12 12 12 12 $gate_bits $gate_bits false false false false false false
done

# 高位宽测试 (24位)
for gate_bits in 16 24; do
    config_name="BW_HIGH24_gate${gate_bits}"
    modify_bitwidth 24 24 24 24 $gate_bits $gate_bits
    run_test "$config_name" 24 24 24 24 $gate_bits $gate_bits false false false false false false
done

# 混合扩展位宽测试
# z/r 用 4 位，g 用 16 位
modify_bitwidth 4 4 4 4 16 16
run_test "BW_ZR4_G16" 4 4 4 4 16 16 false false false false false false

# z/r 用 10 位，g 用 8 位
modify_bitwidth 10 10 10 10 8 8
run_test "BW_ZR10_G8" 10 10 10 10 8 8 false false false false false false

# z/r 用 12 位，g 用 12 位
modify_bitwidth 12 12 12 12 12 12
run_test "BW_FULL12" 12 12 12 12 12 12 false false false false false false

# 预激活高精度，输出低精度
modify_bitwidth 16 8 16 8 16 8
run_test "BW_PRE16_OUT8" 16 8 16 8 16 8 false false false false false false

# 预激活低精度，输出高精度
modify_bitwidth 8 16 8 16 8 16
run_test "BW_PRE8_OUT16" 8 16 8 16 8 16 false false false false false false

# 4-8-16 渐进位宽
modify_bitwidth 4 8 4 8 8 16
run_test "BW_PROGRESSIVE_4_8_16" 4 8 4 8 8 16 false false false false false false

# 全 4 位极限测试
modify_bitwidth 4 4 4 4 4 4
run_test "BW_FULL4" 4 4 4 4 4 4 false false false false false false

# 全 24 位高精度测试
modify_bitwidth 24 24 24 24 24 24
run_test "BW_FULL24" 24 24 24 24 24 24 false false false false false false

# ==================== 第二部分：所有64种对称配置（使用全16位位宽）====================
echo ""
echo "==================== 第二部分：所有64种对称配置（全16位位宽）===================="
echo ""
echo "==================== 第二部分：所有64种对称配置（全16位位宽）====================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 设置全16位位宽
modify_bitwidth 16 16 16 16 16 16

# 对称选项
SYMMETRICS=(false true)

# 枚举所有64种对称配置
for z_pre_sym in "${SYMMETRICS[@]}"; do
    for z_out_sym in "${SYMMETRICS[@]}"; do
        for r_pre_sym in "${SYMMETRICS[@]}"; do
            for r_out_sym in "${SYMMETRICS[@]}"; do
                for g_pre_sym in "${SYMMETRICS[@]}"; do
                    for g_out_sym in "${SYMMETRICS[@]}"; do
                        # 将 true/false 转为 T/F 用于命名
                        z_pre_s=$([ "$z_pre_sym" = "true" ] && echo "T" || echo "F")
                        z_out_s=$([ "$z_out_sym" = "true" ] && echo "T" || echo "F")
                        r_pre_s=$([ "$r_pre_sym" = "true" ] && echo "T" || echo "F")
                        r_out_s=$([ "$r_out_sym" = "true" ] && echo "T" || echo "F")
                        g_pre_s=$([ "$g_pre_sym" = "true" ] && echo "T" || echo "F")
                        g_out_s=$([ "$g_out_sym" = "true" ] && echo "T" || echo "F")
                        
                        config_name="SYM_z${z_pre_s}${z_out_s}_r${r_pre_s}${r_out_s}_g${g_pre_s}${g_out_s}"
                        modify_symmetric $z_pre_sym $z_out_sym $r_pre_sym $r_out_sym $g_pre_sym $g_out_sym
                        run_test "$config_name" 16 16 16 16 16 16 $z_pre_sym $z_out_sym $r_pre_sym $r_out_sym $g_pre_sym $g_out_sym
                    done
                done
            done
        done
    done
done

# ==================== 第三部分：典型位宽+对称组合测试 ====================
echo ""
echo "==================== 第三部分：典型位宽+对称组合测试 ===================="
echo ""
echo "==================== 第三部分：典型位宽+对称组合测试 ====================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 典型组合测试
# 格式: z_pre z_out r_pre r_out g_pre g_out z_pre_sym z_out_sym r_pre_sym r_out_sym g_pre_sym g_out_sym name
TYPICAL_CONFIGS=(
    # 全8位 + 不同对称配置
    "8 8 8 8 8 8 false false false false false false FULL8_ASYM"
    "8 8 8 8 8 8 true true true true true true FULL8_SYM"
    "8 8 8 8 8 8 true false true false true false FULL8_PRE_SYM"
    "8 8 8 8 8 8 false true false true false true FULL8_OUT_SYM"
    
    # 全16位 + 不同对称配置
    "16 16 16 16 16 16 false false false false false false FULL16_ASYM"
    "16 16 16 16 16 16 true true true true true true FULL16_SYM"
    "16 16 16 16 16 16 true false true false true false FULL16_PRE_SYM"
    "16 16 16 16 16 16 false true false true false true FULL16_OUT_SYM"
    
    # 最佳位宽配置 zr8-g16 + 不同对称配置
    "8 8 8 8 16 16 false false false false false false ZR8G16_ASYM"
    "8 8 8 8 16 16 true true true true true true ZR8G16_SYM"
    "8 8 8 8 16 16 false false false false true true ZR8G16_G_SYM"
    "8 8 8 8 16 16 true true true true false false ZR8G16_ZR_SYM"
    
    # 混合配置
    "16 8 16 8 16 8 false false false false false false PRE16OUT8_ASYM"
    "8 16 8 16 8 16 false false false false false false PRE8OUT16_ASYM"
    "16 16 16 16 8 8 false false false false false false ZR16G8_ASYM"
    "16 16 16 16 8 16 false false false false false false ORIG_DEFAULT"
)

for config in "${TYPICAL_CONFIGS[@]}"; do
    read -r z_pre z_out r_pre r_out g_pre g_out z_pre_sym z_out_sym r_pre_sym r_out_sym g_pre_sym g_out_sym name <<< "$config"
    config_name="COMBO_${name}"
    modify_bitwidth $z_pre $z_out $r_pre $r_out $g_pre $g_out
    modify_symmetric $z_pre_sym $z_out_sym $r_pre_sym $r_out_sym $g_pre_sym $g_out_sym
    run_test "$config_name" $z_pre $z_out $r_pre $r_out $g_pre $g_out $z_pre_sym $z_out_sym $r_pre_sym $r_out_sym $g_pre_sym $g_out_sym
done

# 恢复原始配置
echo "$ORIGINAL_CONFIG" > "$CONFIG_FILE"
echo ""
echo "原始配置已恢复"

# ==================== 生成排序后的结果摘要 ====================
echo ""
echo "==================== 结果摘要（按余弦相似度排序）===================="
echo ""
echo "==================== 结果摘要（按余弦相似度排序）====================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 跳过 CSV 头，按余弦相似度（第15列）降序排序，取前20名
echo "Top 20 最佳配置:" | tee -a "$RESULT_FILE"
echo "排名 | 配置名称 | MSE | 余弦相似度" | tee -a "$RESULT_FILE"
echo "------|----------|-----|------------" | tee -a "$RESULT_FILE"
tail -n +2 "$CSV_FILE" | sort -t',' -k15 -rn | head -20 | nl -w2 | while read rank line; do
    name=$(echo "$line" | cut -d',' -f1)
    mse=$(echo "$line" | cut -d',' -f14)
    cos=$(echo "$line" | cut -d',' -f15)
    printf "%s | %s | %s | %s\n" "$rank" "$name" "$mse" "$cos" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "Bottom 5 最差配置:" | tee -a "$RESULT_FILE"
echo "排名 | 配置名称 | MSE | 余弦相似度" | tee -a "$RESULT_FILE"
echo "------|----------|-----|------------" | tee -a "$RESULT_FILE"
tail -n +2 "$CSV_FILE" | sort -t',' -k15 -n | head -5 | nl -w2 | while read rank line; do
    name=$(echo "$line" | cut -d',' -f1)
    mse=$(echo "$line" | cut -d',' -f14)
    cos=$(echo "$line" | cut -d',' -f15)
    printf "%s | %s | %s | %s\n" "$rank" "$name" "$mse" "$cos" | tee -a "$RESULT_FILE"
done

# 显示失败测试列表
if [ $FAIL_COUNT -gt 0 ]; then
    echo "" | tee -a "$RESULT_FILE"
    echo "==================== 失败测试列表 ====================" | tee -a "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
    printf "%-4s | %-40s | %-15s | %-12s\n" "序号" "配置名称" "MSE" "余弦相似度" | tee -a "$RESULT_FILE"
    printf "%-4s-+-%-40s-+-%-15s-+-%-12s\n" "----" "----------------------------------------" "---------------" "------------" | tee -a "$RESULT_FILE"
    tail -n +2 "$CSV_FILE" | grep -E "ERROR|N/A" | nl -w2 | while IFS= read -r line; do
        rank=$(echo "$line" | awk '{print $1}')
        data=$(echo "$line" | cut -f2-)
        name=$(echo "$data" | cut -d',' -f1)
        mse=$(echo "$data" | cut -d',' -f14)
        cos=$(echo "$data" | cut -d',' -f15)
        printf "%-4s | %-40s | %-15s | %-12s\n" "$rank" "$name" "$mse" "$cos" | tee -a "$RESULT_FILE"
    done
    # 还要显示精度不达标的配置
    tail -n +2 "$CSV_FILE" | grep -v "ERROR" | while IFS=',' read -r name z_pre z_out r_pre r_out g_pre g_out z_pre_sym z_out_sym r_pre_sym r_out_sym g_pre_sym g_out_sym mse cos; do
        if [ "$cos" != "N/A" ] && [ "$mse" != "N/A" ]; then
            # 检查是否不满足阈值
            if ! awk -v val="$cos" -v threshold="$COSINE_THRESHOLD" 'BEGIN {exit !(val >= threshold)}' || ! awk -v val="$mse" -v threshold="$MSE_THRESHOLD" 'BEGIN {exit !(val <= threshold)}'; then
                printf "     | %-40s | %-15s | %-12s\n" "$name" "$mse" "$cos" | tee -a "$RESULT_FILE"
            fi
        fi
    done
    echo "" | tee -a "$RESULT_FILE"
fi

echo ""
echo "===== 测试完成 ====="
echo "总测试数: $TEST_COUNT"
echo "通过 (Cosine >= $COSINE_THRESHOLD 且 MSE <= $MSE_THRESHOLD): $PASS_COUNT"
echo "失败: $FAIL_COUNT"
echo "详细结果: $RESULT_FILE"
echo "CSV 数据: $CSV_FILE"

# 如果有失败的测试，返回非零退出码
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
