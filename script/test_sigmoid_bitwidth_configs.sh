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

# 保存原始配置
cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

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
modify_bitwidth() {
    local z_pre=$1
    local z_out=$2
    local r_pre=$3
    local r_out=$4
    local g_pre=$5
    local g_out=$6
    
    # 根据位宽生成正确的类型
    local z_pre_type="INT${z_pre}"
    local z_out_type="UINT${z_out}"
    local r_pre_type="INT${r_pre}"
    local r_out_type="UINT${r_out}"
    local g_pre_type="INT${g_pre}"
    local g_out_type="INT${g_out}"
    
    # 使用 sed 替换配置
    sed -i "s/QuantBitWidth z_pre_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth z_pre_ = QuantBitWidth::${z_pre_type};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth z_out_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth z_out_ = QuantBitWidth::${z_out_type};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth r_pre_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth r_pre_ = QuantBitWidth::${r_pre_type};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth r_out_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth r_out_ = QuantBitWidth::${r_out_type};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth g_pre_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth g_pre_ = QuantBitWidth::${g_pre_type};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth g_out_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth g_out_ = QuantBitWidth::${g_out_type};/" "$CONFIG_FILE"
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
    make -j$(nproc) gru_example > /dev/null 2>&1
    
    # 运行测试并提取结果
    local output=$(./gru_example 2>&1)
    local mse=$(echo "$output" | grep "Overall H: MSE" | sed 's/.*MSE = \([0-9.e+-]*\),.*/\1/')
    local cos=$(echo "$output" | grep "Overall H: MSE" | sed 's/.*Cosine Similarity = \([0-9.]*\)/\1/')
    
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
        if awk "BEGIN {exit !($cos >= $COSINE_THRESHOLD)}"; then
            cos_ok=true
        fi
        # 检查 MSE
        if awk "BEGIN {exit !($mse <= $MSE_THRESHOLD)}"; then
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

# ==================== 第一部分：所有64种位宽配置（对称使用默认值 false）====================
echo ""
echo "==================== 第一部分：所有64种激活位宽配置 ===================="
echo ""
echo "==================== 第一部分：所有64种激活位宽配置 ====================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 计算总测试数
# 64种位宽配置 + 64种对称配置（使用最佳位宽）+ 16种典型组合
TOTAL_TESTS=$((64 + 64 + 16))

echo "预计总测试数: $TOTAL_TESTS"
echo ""

# 位宽选项
BITWIDTHS=(8 16)

# 设置默认对称配置（全部非对称）
modify_symmetric false false false false false false

# 枚举所有64种位宽配置
for z_pre in "${BITWIDTHS[@]}"; do
    for z_out in "${BITWIDTHS[@]}"; do
        for r_pre in "${BITWIDTHS[@]}"; do
            for r_out in "${BITWIDTHS[@]}"; do
                for g_pre in "${BITWIDTHS[@]}"; do
                    for g_out in "${BITWIDTHS[@]}"; do
                        config_name="BW_z${z_pre}${z_out}_r${r_pre}${r_out}_g${g_pre}${g_out}"
                        modify_bitwidth $z_pre $z_out $r_pre $r_out $g_pre $g_out
                        run_test "$config_name" $z_pre $z_out $r_pre $r_out $g_pre $g_out false false false false false false
                    done
                done
            done
        done
    done
done

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
cp "$CONFIG_FILE.backup" "$CONFIG_FILE"
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
