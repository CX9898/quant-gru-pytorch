#!/bin/bash
# 中间运算算子位宽及对称配置测试脚本
# 测试内容：
# 1. GEMM 结果位宽: Wx_, Rh_
# 2. 偏置位宽: bx_, br_
# 3. 中间运算位宽: Rh_add_br_, rRh_, old_contrib_, new_contrib_
# 4. 各算子的对称配置
# 5. 不同位宽和对称组合对精度的影响
#
# 配置变量说明：
#   GEMM结果类:    Wx_, Rh_               (W@x 和 R@h 的结果)
#   偏置类:        bx_, br_               (输入偏置和循环偏置)
#   中间运算类:    Rh_add_br_, rRh_       (Rh+br 和 r×Rh)
#                  old_contrib_, new_contrib_ (z×h[t-1] 和 (1-z)×g)

set -e

# 自动获取项目根目录（脚本位于 script/ 目录下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/include/quantize_bitwidth_config.h"
BUILD_DIR="$PROJECT_DIR/build"

# 保存原始配置
cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

# 结果文件
RESULT_FILE="$PROJECT_DIR/test_intermediate_ops_results.txt"
CSV_FILE="$PROJECT_DIR/test_intermediate_ops_results.csv"

echo "===== 中间运算算子位宽及对称配置测试结果 =====" > "$RESULT_FILE"
echo "测试时间: $(date)" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# CSV 头
echo "config_name,Wx,Rh,bx,br,Rh_add_br,rRh,old_contrib,new_contrib,Wx_sym,Rh_sym,Rh_add_br_sym,rRh_sym,old_contrib_sym,new_contrib_sym,mse,cosine_similarity" > "$CSV_FILE"

# 计数器
TEST_COUNT=0
TOTAL_TESTS=0
PASS_COUNT=0
FAIL_COUNT=0

# 函数：修改 GEMM 结果位宽 (Wx_, Rh_)
modify_gemm_bitwidth() {
    local Wx_bits=$1
    local Rh_bits=$2
    
    sed -i "s/QuantBitWidth Wx_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth Wx_ = QuantBitWidth::INT${Wx_bits};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth Rh_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth Rh_ = QuantBitWidth::INT${Rh_bits};/" "$CONFIG_FILE"
}

# 函数：修改偏置位宽 (bx_, br_)
modify_bias_bitwidth() {
    local bx_bits=$1
    local br_bits=$2
    
    sed -i "s/QuantBitWidth bx_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth bx_ = QuantBitWidth::INT${bx_bits};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth br_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth br_ = QuantBitWidth::INT${br_bits};/" "$CONFIG_FILE"
}

# 函数：修改中间运算位宽
modify_intermediate_bitwidth() {
    local Rh_add_br_bits=$1
    local rRh_bits=$2
    local old_contrib_bits=$3
    local new_contrib_bits=$4
    
    sed -i "s/QuantBitWidth Rh_add_br_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth Rh_add_br_ = QuantBitWidth::INT${Rh_add_br_bits};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth rRh_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth rRh_ = QuantBitWidth::INT${rRh_bits};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth old_contrib_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth old_contrib_ = QuantBitWidth::INT${old_contrib_bits};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth new_contrib_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth new_contrib_ = QuantBitWidth::INT${new_contrib_bits};/" "$CONFIG_FILE"
}

# 函数：修改 GEMM 结果对称配置
modify_gemm_symmetric() {
    local Wx_sym=$1
    local Rh_sym=$2
    
    sed -i "s/bool Wx_symmetric_ = [a-z]*;/bool Wx_symmetric_ = ${Wx_sym};/" "$CONFIG_FILE"
    sed -i "s/bool Rh_symmetric_ = [a-z]*;/bool Rh_symmetric_ = ${Rh_sym};/" "$CONFIG_FILE"
}

# 函数：修改中间运算对称配置
modify_intermediate_symmetric() {
    local Rh_add_br_sym=$1
    local rRh_sym=$2
    local old_contrib_sym=$3
    local new_contrib_sym=$4
    
    sed -i "s/bool Rh_add_br_symmetric_ = [a-z]*;/bool Rh_add_br_symmetric_ = ${Rh_add_br_sym};/" "$CONFIG_FILE"
    sed -i "s/bool rRh_symmetric_ = [a-z]*;/bool rRh_symmetric_ = ${rRh_sym};/" "$CONFIG_FILE"
    sed -i "s/bool old_contrib_symmetric_ = [a-z]*;/bool old_contrib_symmetric_ = ${old_contrib_sym};/" "$CONFIG_FILE"
    sed -i "s/bool new_contrib_symmetric_ = [a-z]*;/bool new_contrib_symmetric_ = ${new_contrib_sym};/" "$CONFIG_FILE"
}

# 函数：设置所有位宽配置
set_all_bitwidth() {
    local Wx=$1
    local Rh=$2
    local bx=$3
    local br=$4
    local Rh_add_br=$5
    local rRh=$6
    local old_contrib=$7
    local new_contrib=$8
    
    modify_gemm_bitwidth $Wx $Rh
    modify_bias_bitwidth $bx $br
    modify_intermediate_bitwidth $Rh_add_br $rRh $old_contrib $new_contrib
}

# 函数：设置所有对称配置
set_all_symmetric() {
    local Wx_sym=$1
    local Rh_sym=$2
    local Rh_add_br_sym=$3
    local rRh_sym=$4
    local old_contrib_sym=$5
    local new_contrib_sym=$6
    
    modify_gemm_symmetric $Wx_sym $Rh_sym
    modify_intermediate_symmetric $Rh_add_br_sym $rRh_sym $old_contrib_sym $new_contrib_sym
}

# 函数：编译并运行测试
run_test() {
    local config_name=$1
    local Wx=$2
    local Rh=$3
    local bx=$4
    local br=$5
    local Rh_add_br=$6
    local rRh=$7
    local old_contrib=$8
    local new_contrib=$9
    local Wx_sym=${10}
    local Rh_sym=${11}
    local Rh_add_br_sym=${12}
    local rRh_sym=${13}
    local old_contrib_sym=${14}
    local new_contrib_sym=${15}
    
    TEST_COUNT=$((TEST_COUNT + 1))
    
    echo "[$TEST_COUNT/$TOTAL_TESTS] 测试: $config_name"
    
    # 设置配置
    set_all_bitwidth $Wx $Rh $bx $br $Rh_add_br $rRh $old_contrib $new_contrib
    set_all_symmetric $Wx_sym $Rh_sym $Rh_add_br_sym $rRh_sym $old_contrib_sym $new_contrib_sym
    
    # 重新编译（静默模式）
    cd "$BUILD_DIR"
    if ! make -j$(nproc) gru_example > /dev/null 2>&1; then
        echo "  ❌ 编译失败"
        echo "配置: $config_name" >> "$RESULT_FILE"
        echo "  状态: 编译失败" >> "$RESULT_FILE"
        echo "" >> "$RESULT_FILE"
        echo "$config_name,$Wx,$Rh,$bx,$br,$Rh_add_br,$rRh,$old_contrib,$new_contrib,$Wx_sym,$Rh_sym,$Rh_add_br_sym,$rRh_sym,$old_contrib_sym,$new_contrib_sym,COMPILE_ERROR,COMPILE_ERROR" >> "$CSV_FILE"
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
        echo "$config_name,$Wx,$Rh,$bx,$br,$Rh_add_br,$rRh,$old_contrib,$new_contrib,$Wx_sym,$Rh_sym,$Rh_add_br_sym,$rRh_sym,$old_contrib_sym,$new_contrib_sym,RUNTIME_ERROR,RUNTIME_ERROR" >> "$CSV_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    
    local mse=$(echo "$output" | grep "Overall H: MSE" | sed 's/.*MSE = \([0-9.e+-]*\),.*/\1/')
    local cos=$(echo "$output" | grep "Overall H: MSE" | sed 's/.*Cosine Similarity = \([0-9.]*\)/\1/')
    
    # 处理空值
    if [ -z "$mse" ]; then mse="N/A"; fi
    if [ -z "$cos" ]; then cos="N/A"; fi
    
    echo "  ✓ MSE: $mse, Cosine: $cos"
    PASS_COUNT=$((PASS_COUNT + 1))
    
    # 记录到结果文件
    echo "配置: $config_name" >> "$RESULT_FILE"
    echo "  位宽: Wx=$Wx, Rh=$Rh, bx=$bx, br=$br" >> "$RESULT_FILE"
    echo "        Rh_add_br=$Rh_add_br, rRh=$rRh, old_contrib=$old_contrib, new_contrib=$new_contrib" >> "$RESULT_FILE"
    echo "  对称: Wx=$Wx_sym, Rh=$Rh_sym, Rh_add_br=$Rh_add_br_sym, rRh=$rRh_sym" >> "$RESULT_FILE"
    echo "        old_contrib=$old_contrib_sym, new_contrib=$new_contrib_sym" >> "$RESULT_FILE"
    echo "  MSE: $mse, Cosine Similarity: $cos" >> "$RESULT_FILE"
    echo "" >> "$RESULT_FILE"
    
    # 记录到 CSV
    echo "$config_name,$Wx,$Rh,$bx,$br,$Rh_add_br,$rRh,$old_contrib,$new_contrib,$Wx_sym,$Rh_sym,$Rh_add_br_sym,$rRh_sym,$old_contrib_sym,$new_contrib_sym,$mse,$cos" >> "$CSV_FILE"
}

# 简化版 run_test（位宽测试，对称全部使用默认 false）
run_bitwidth_test() {
    local config_name=$1
    local Wx=$2
    local Rh=$3
    local bx=$4
    local br=$5
    local Rh_add_br=$6
    local rRh=$7
    local old_contrib=$8
    local new_contrib=$9
    
    run_test "$config_name" $Wx $Rh $bx $br $Rh_add_br $rRh $old_contrib $new_contrib false false false false false false
}

# 简化版 run_test（对称测试，位宽全部使用 16 位）
run_symmetric_test() {
    local config_name=$1
    local Wx_sym=$2
    local Rh_sym=$3
    local Rh_add_br_sym=$4
    local rRh_sym=$5
    local old_contrib_sym=$6
    local new_contrib_sym=$7
    
    run_test "$config_name" 16 16 16 16 16 16 16 16 $Wx_sym $Rh_sym $Rh_add_br_sym $rRh_sym $old_contrib_sym $new_contrib_sym
}

# ==================== 开始测试 ====================
echo ""
echo "==================== 中间运算算子位宽及对称配置测试 ===================="
echo ""

# 确保构建目录存在
if [ ! -d "$BUILD_DIR" ]; then
    echo "创建构建目录: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake ..
fi

# 位宽选项
BITWIDTHS=(8 16)
SYMMETRICS=(false true)

# 计算总测试数
# 第一部分：基准测试 (2)
# 第二部分：GEMM 结果位宽 (4)
# 第三部分：偏置位宽 (4)
# 第四部分：中间运算位宽单独测试 (4)
# 第五部分：关键路径测试 (8)
# 第六部分：对称配置测试 (64)
# 第七部分：典型组合测试 (16)
# 第八部分：全位宽组合测试 (256) - 可选，耗时较长
TOTAL_TESTS=$((2 + 4 + 4 + 4 + 8 + 64 + 16))

echo "预计测试数: $TOTAL_TESTS"
echo ""

# ==================== 第一部分：基准测试 ====================
echo ""
echo "==================== 第一部分：基准测试 ===================="
echo ""
echo "==================== 第一部分：基准测试 ====================" >> "$RESULT_FILE"

# 设置默认对称配置
set_all_symmetric false false false false false false

run_bitwidth_test "BASELINE_INT8" 8 8 8 8 8 8 8 8
run_bitwidth_test "BASELINE_INT16" 16 16 16 16 16 16 16 16

# ==================== 第二部分：GEMM 结果位宽测试 ====================
echo ""
echo "==================== 第二部分：GEMM 结果位宽测试 (Wx_, Rh_) ===================="
echo ""
echo "==================== 第二部分：GEMM 结果位宽测试 ====================" >> "$RESULT_FILE"

# 其他固定为 8 位，测试 GEMM 结果的不同组合
for Wx in "${BITWIDTHS[@]}"; do
    for Rh in "${BITWIDTHS[@]}"; do
        config_name="GEMM_Wx${Wx}_Rh${Rh}"
        run_bitwidth_test "$config_name" $Wx $Rh 8 8 8 8 8 8
    done
done

# ==================== 第三部分：偏置位宽测试 ====================
echo ""
echo "==================== 第三部分：偏置位宽测试 (bx_, br_) ===================="
echo ""
echo "==================== 第三部分：偏置位宽测试 ====================" >> "$RESULT_FILE"

# 其他固定为 8 位，测试偏置的不同组合
for bx in "${BITWIDTHS[@]}"; do
    for br in "${BITWIDTHS[@]}"; do
        config_name="BIAS_bx${bx}_br${br}"
        run_bitwidth_test "$config_name" 8 8 $bx $br 8 8 8 8
    done
done

# ==================== 第四部分：中间运算位宽单独测试 ====================
echo ""
echo "==================== 第四部分：中间运算位宽单独测试 ===================="
echo ""
echo "==================== 第四部分：中间运算位宽单独测试 ====================" >> "$RESULT_FILE"

# 每个中间算子单独升级到 16 位
run_bitwidth_test "INTER_Rh_add_br_16" 8 8 8 8 16 8 8 8
run_bitwidth_test "INTER_rRh_16" 8 8 8 8 8 16 8 8
run_bitwidth_test "INTER_old_contrib_16" 8 8 8 8 8 8 16 8
run_bitwidth_test "INTER_new_contrib_16" 8 8 8 8 8 8 8 16

# ==================== 第五部分：关键路径位宽测试 ====================
echo ""
echo "==================== 第五部分：关键路径位宽测试 ===================="
echo ""
echo "==================== 第五部分：关键路径位宽测试 ====================" >> "$RESULT_FILE"

# GEMM 16 位，中间运算 8 位
run_bitwidth_test "PATH_GEMM16_INTER8" 16 16 8 8 8 8 8 8

# GEMM 8 位，中间运算 16 位
run_bitwidth_test "PATH_GEMM8_INTER16" 8 8 8 8 16 16 16 16

# 偏置 16 位，其他 8 位
run_bitwidth_test "PATH_BIAS16" 8 8 16 16 8 8 8 8

# 候选门路径 (Rh_add_br -> rRh) 16 位
run_bitwidth_test "PATH_CANDIDATE_16" 8 16 8 16 16 16 8 8

# 输出路径 (old_contrib, new_contrib) 16 位
run_bitwidth_test "PATH_OUTPUT_16" 8 8 8 8 8 8 16 16

# Rh 相关路径全 16 位 (Rh, br, Rh_add_br, rRh)
run_bitwidth_test "PATH_RH_CHAIN_16" 8 16 8 16 16 16 8 8

# 混合：GEMM 16 位 + 输出路径 16 位
run_bitwidth_test "PATH_GEMM16_OUTPUT16" 16 16 8 8 8 8 16 16

# 全链路高精度
run_bitwidth_test "PATH_FULL_HIGH_PREC" 16 16 16 16 16 16 16 16

# ==================== 第六部分：对称配置测试 ====================
echo ""
echo "==================== 第六部分：对称配置测试（全 16 位位宽）===================="
echo ""
echo "==================== 第六部分：对称配置测试 ====================" >> "$RESULT_FILE"

# 枚举所有 64 种对称配置（6 个布尔变量）
for Wx_sym in "${SYMMETRICS[@]}"; do
    for Rh_sym in "${SYMMETRICS[@]}"; do
        for Rh_add_br_sym in "${SYMMETRICS[@]}"; do
            for rRh_sym in "${SYMMETRICS[@]}"; do
                for old_contrib_sym in "${SYMMETRICS[@]}"; do
                    for new_contrib_sym in "${SYMMETRICS[@]}"; do
                        # 将 true/false 转为 T/F 用于命名
                        Wx_s=$([ "$Wx_sym" = "true" ] && echo "T" || echo "F")
                        Rh_s=$([ "$Rh_sym" = "true" ] && echo "T" || echo "F")
                        Rh_add_br_s=$([ "$Rh_add_br_sym" = "true" ] && echo "T" || echo "F")
                        rRh_s=$([ "$rRh_sym" = "true" ] && echo "T" || echo "F")
                        old_s=$([ "$old_contrib_sym" = "true" ] && echo "T" || echo "F")
                        new_s=$([ "$new_contrib_sym" = "true" ] && echo "T" || echo "F")
                        
                        config_name="SYM_${Wx_s}${Rh_s}_${Rh_add_br_s}${rRh_s}_${old_s}${new_s}"
                        run_symmetric_test "$config_name" $Wx_sym $Rh_sym $Rh_add_br_sym $rRh_sym $old_contrib_sym $new_contrib_sym
                    done
                done
            done
        done
    done
done

# ==================== 第七部分：典型组合测试 ====================
echo ""
echo "==================== 第七部分：典型组合测试 ===================="
echo ""
echo "==================== 第七部分：典型组合测试 ====================" >> "$RESULT_FILE"

# 典型组合测试
# 格式: Wx Rh bx br Rh_add_br rRh old new Wx_sym Rh_sym Rh_add_br_sym rRh_sym old_sym new_sym name
TYPICAL_CONFIGS=(
    # 全 8 位 + 不同对称配置
    "8 8 8 8 8 8 8 8 false false false false false false FULL8_ASYM"
    "8 8 8 8 8 8 8 8 true true true true true true FULL8_SYM"
    "8 8 8 8 8 8 8 8 true true false false false false FULL8_GEMM_SYM"
    "8 8 8 8 8 8 8 8 false false false false true true FULL8_OUTPUT_SYM"
    
    # 全 16 位 + 不同对称配置
    "16 16 16 16 16 16 16 16 false false false false false false FULL16_ASYM"
    "16 16 16 16 16 16 16 16 true true true true true true FULL16_SYM"
    "16 16 16 16 16 16 16 16 true true false false false false FULL16_GEMM_SYM"
    "16 16 16 16 16 16 16 16 false false false false true true FULL16_OUTPUT_SYM"
    
    # GEMM 高精度配置
    "16 16 8 8 8 8 8 8 false false false false false false GEMM16_ONLY"
    "16 16 8 8 8 8 8 8 true true false false false false GEMM16_SYM"
    "16 16 16 16 8 8 8 8 false false false false false false GEMM16_BIAS16"
    "16 16 16 16 16 16 8 8 false false false false false false GEMM16_BIAS16_CAND16"
    
    # 混合精度配置（部分高精度）
    "8 16 8 16 16 16 8 8 false false false false false false RH_PATH_16"
    "16 8 16 8 8 8 16 16 false false false false false false WX_OUTPUT_16"
    "8 8 16 16 16 16 16 16 false false false false false false BIAS_INTER_16"
    "16 16 8 8 16 16 16 16 false false false false false false GEMM16_INTER16"
)

for config in "${TYPICAL_CONFIGS[@]}"; do
    read -r Wx Rh bx br Rh_add_br rRh old new Wx_sym Rh_sym Rh_add_br_sym rRh_sym old_sym new_sym name <<< "$config"
    config_name="COMBO_${name}"
    run_test "$config_name" $Wx $Rh $bx $br $Rh_add_br $rRh $old $new $Wx_sym $Rh_sym $Rh_add_br_sym $rRh_sym $old_sym $new_sym
done

# 恢复原始配置
cp "$CONFIG_FILE.backup" "$CONFIG_FILE"
rm -f "$CONFIG_FILE.backup"
echo ""
echo "原始配置已恢复"

# ==================== 生成排序后的结果摘要 ====================
echo ""
echo "==================== 结果摘要 ===================="
echo ""
echo "==================== 结果摘要 ====================" >> "$RESULT_FILE"

echo "" | tee -a "$RESULT_FILE"
echo "测试统计:" | tee -a "$RESULT_FILE"
echo "  总测试数: $TEST_COUNT" | tee -a "$RESULT_FILE"
echo "  通过: $PASS_COUNT" | tee -a "$RESULT_FILE"
echo "  失败: $FAIL_COUNT" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

# 按余弦相似度降序排序
echo "Top 20 最佳配置（按余弦相似度排序）:" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"
printf "%-4s | %-30s | %-15s | %-12s\n" "排名" "配置名称" "MSE" "余弦相似度" | tee -a "$RESULT_FILE"
printf "%-4s-+-%-30s-+-%-15s-+-%-12s\n" "----" "------------------------------" "---------------" "------------" | tee -a "$RESULT_FILE"
tail -n +2 "$CSV_FILE" | grep -v "ERROR" | grep -v "N/A" | sort -t',' -k17 -rn | head -20 | nl -w2 | while IFS= read -r line; do
    rank=$(echo "$line" | awk '{print $1}')
    data=$(echo "$line" | cut -f2-)
    name=$(echo "$data" | cut -d',' -f1)
    mse=$(echo "$data" | cut -d',' -f16)
    cos=$(echo "$data" | cut -d',' -f17)
    printf "%-4s | %-30s | %-15s | %-12s\n" "$rank" "$name" "$mse" "$cos" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "Bottom 10 最差配置:" | tee -a "$RESULT_FILE"
printf "%-4s | %-30s | %-15s | %-12s\n" "排名" "配置名称" "MSE" "余弦相似度" | tee -a "$RESULT_FILE"
printf "%-4s-+-%-30s-+-%-15s-+-%-12s\n" "----" "------------------------------" "---------------" "------------" | tee -a "$RESULT_FILE"
tail -n +2 "$CSV_FILE" | grep -v "ERROR" | grep -v "N/A" | sort -t',' -k17 -n | head -10 | nl -w2 | while IFS= read -r line; do
    rank=$(echo "$line" | awk '{print $1}')
    data=$(echo "$line" | cut -f2-)
    name=$(echo "$data" | cut -d',' -f1)
    mse=$(echo "$data" | cut -d',' -f16)
    cos=$(echo "$data" | cut -d',' -f17)
    printf "%-4s | %-30s | %-15s | %-12s\n" "$rank" "$name" "$mse" "$cos" | tee -a "$RESULT_FILE"
done

# 按位宽分析
echo "" | tee -a "$RESULT_FILE"
echo "按位宽分类统计（仅位宽测试，排除对称测试）:" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

# 统计全 8 位配置的平均值
echo "全 8 位配置:" | tee -a "$RESULT_FILE"
tail -n +2 "$CSV_FILE" | grep "^BASELINE_INT8\|^GEMM_Wx8_Rh8\|^BIAS_bx8_br8\|^COMBO_FULL8" | grep -v "ERROR" | while IFS=',' read -r rest; do
    echo "  $rest" | cut -d',' -f1,16,17
done | head -5 | tee -a "$RESULT_FILE"

echo "" | tee -a "$RESULT_FILE"
echo "全 16 位配置:" | tee -a "$RESULT_FILE"
tail -n +2 "$CSV_FILE" | grep "^BASELINE_INT16\|^COMBO_FULL16" | grep -v "ERROR" | while IFS=',' read -r rest; do
    echo "  $rest" | cut -d',' -f1,16,17
done | head -5 | tee -a "$RESULT_FILE"

echo ""
echo "===== 测试完成 ====="
echo "总测试数: $TEST_COUNT"
echo "通过: $PASS_COUNT, 失败: $FAIL_COUNT"
echo "详细结果: $RESULT_FILE"
echo "CSV 数据: $CSV_FILE"
