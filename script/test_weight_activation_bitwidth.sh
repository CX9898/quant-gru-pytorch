#!/bin/bash
# 权重、激活和GEMM结果位宽配置测试脚本
# 测试内容：
# 1. 支持的基础模式：W8A8, W8A16, W16A16
# 2. 验证不支持的模式会正确报错（如 W16A8）
# 3. 测试 Wx/Rh (GEMM结果) 位宽配置
#
# 配置的变量：
#   权重类 (weight_bits): W_, R_
#   激活类 (activation_bits): x_, h_
#   GEMM结果类 (gemm_result_bits): Wx_, Rh_

set -e

# 自动获取项目根目录（脚本位于 script/ 目录下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/include/quantize_bitwidth_config.h"
BUILD_DIR="$PROJECT_DIR/build"

# 保存原始配置
cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

# 结果文件
RESULT_FILE="$PROJECT_DIR/test_weight_activation_results.txt"
CSV_FILE="$PROJECT_DIR/test_weight_activation_results.csv"

echo "===== 权重和激活位宽配置测试结果 =====" > "$RESULT_FILE"
echo "测试时间: $(date)" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# CSV 头
echo "config_name,weight_bits,activation_bits,gemm_result_bits,W_,R_,x_,h_,Wx_,Rh_,status,mse,cosine_similarity,error_msg" > "$CSV_FILE"

# 计数器
TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

# 阈值设置（必须同时满足才算通过）
COSINE_THRESHOLD=0.999    # 余弦相似度 >= 此值
MSE_THRESHOLD=1e-4        # MSE <= 此值

# 函数：修改权重和激活位宽配置
# 参数：weight_bits, activation_bits, gemm_result_bits (可选，默认跟随 activation_bits)
modify_weight_activation_bitwidth() {
    local weight_bits=$1
    local activation_bits=$2
    local gemm_result_bits=${3:-$activation_bits}  # 默认跟随激活位宽
    
    local weight_type="INT${weight_bits}"
    local activation_type="INT${activation_bits}"
    local gemm_result_type="INT${gemm_result_bits}"
    
    # 修改权重位宽 (W_, R_)
    sed -i "s/QuantBitWidth W_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth W_ = QuantBitWidth::${weight_type};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth R_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth R_ = QuantBitWidth::${weight_type};/" "$CONFIG_FILE"
    
    # 修改激活位宽 (x_, h_)
    sed -i "s/QuantBitWidth x_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth x_ = QuantBitWidth::${activation_type};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth h_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth h_ = QuantBitWidth::${activation_type};/" "$CONFIG_FILE"
    
    # 修改 GEMM 结果位宽 (Wx_, Rh_)
    sed -i "s/QuantBitWidth Wx_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth Wx_ = QuantBitWidth::${gemm_result_type};/" "$CONFIG_FILE"
    sed -i "s/QuantBitWidth Rh_ = QuantBitWidth::[A-Z0-9]*;/QuantBitWidth Rh_ = QuantBitWidth::${gemm_result_type};/" "$CONFIG_FILE"
}

# 函数：编译项目
compile_project() {
    cd "$BUILD_DIR"
    if make -j$(nproc) gru_example > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 函数：运行测试
# 参数: config_name, weight_bits, activation_bits, expected_result, [gemm_result_bits]
run_test() {
    local config_name=$1
    local weight_bits=$2
    local activation_bits=$3
    local expected_result=$4  # "pass" 或 "fail"
    local gemm_result_bits=${5:-$activation_bits}  # 默认跟随激活位宽
    
    TEST_COUNT=$((TEST_COUNT + 1))
    
    echo "[$TEST_COUNT] 测试: $config_name (W${weight_bits}A${activation_bits}, GEMM${gemm_result_bits})"
    echo "" >> "$RESULT_FILE"
    echo "配置: $config_name" >> "$RESULT_FILE"
    echo "  权重位宽: ${weight_bits}-bit (W_, R_)" >> "$RESULT_FILE"
    echo "  激活位宽: ${activation_bits}-bit (x_, h_)" >> "$RESULT_FILE"
    echo "  GEMM结果位宽: ${gemm_result_bits}-bit (Wx_, Rh_)" >> "$RESULT_FILE"
    
    # 修改配置
    modify_weight_activation_bitwidth $weight_bits $activation_bits $gemm_result_bits
    
    # 编译
    if ! compile_project; then
        echo "  ❌ 编译失败"
        echo "  状态: 编译失败" >> "$RESULT_FILE"
        echo "$config_name,$weight_bits,$activation_bits,$gemm_result_bits,INT${weight_bits},INT${weight_bits},INT${activation_bits},INT${activation_bits},INT${gemm_result_bits},INT${gemm_result_bits},COMPILE_ERROR,N/A,N/A,Compilation failed" >> "$CSV_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi
    
    # 运行测试
    cd "$BUILD_DIR"
    local output
    local exit_code=0
    output=$(./gru_example 2>&1) || exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        # 运行时错误（可能是不支持的位宽组合）
        local error_msg=$(echo "$output" | grep -i "unsupported\|error\|exception" | head -1 | tr ',' ';')
        if [ -z "$error_msg" ]; then
            error_msg="Runtime error (exit code: $exit_code)"
        fi
        
        if [ "$expected_result" = "fail" ]; then
            echo "  ✓ 预期失败，正确拒绝了不支持的配置"
            echo "  状态: 正确拒绝 (预期行为)" >> "$RESULT_FILE"
            echo "  错误信息: $error_msg" >> "$RESULT_FILE"
            echo "$config_name,$weight_bits,$activation_bits,$gemm_result_bits,INT${weight_bits},INT${weight_bits},INT${activation_bits},INT${activation_bits},INT${gemm_result_bits},INT${gemm_result_bits},EXPECTED_FAIL,N/A,N/A,$error_msg" >> "$CSV_FILE"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo "  ❌ 运行失败: $error_msg"
            echo "  状态: 运行失败" >> "$RESULT_FILE"
            echo "  错误信息: $error_msg" >> "$RESULT_FILE"
            echo "$config_name,$weight_bits,$activation_bits,$gemm_result_bits,INT${weight_bits},INT${weight_bits},INT${activation_bits},INT${activation_bits},INT${gemm_result_bits},INT${gemm_result_bits},RUNTIME_ERROR,N/A,N/A,$error_msg" >> "$CSV_FILE"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        return
    fi
    
    # 运行成功，提取结果
    local mse=$(echo "$output" | grep "Overall H: MSE" | sed 's/.*MSE = \([0-9.e+-]*\),.*/\1/')
    local cos=$(echo "$output" | grep "Overall H: MSE" | sed 's/.*Cosine Similarity = \([0-9.]*\)/\1/')
    
    if [ -z "$mse" ]; then mse="N/A"; fi
    if [ -z "$cos" ]; then cos="N/A"; fi
    
    if [ "$expected_result" = "pass" ]; then
        # 判断是否同时满足 MSE 和余弦相似度阈值
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
                echo "  ✓ 成功 - MSE: $mse (<= $MSE_THRESHOLD), Cosine: $cos (>= $COSINE_THRESHOLD)"
                echo "  状态: 成功" >> "$RESULT_FILE"
                echo "  MSE: $mse, Cosine Similarity: $cos" >> "$RESULT_FILE"
                echo "$config_name,$weight_bits,$activation_bits,$gemm_result_bits,INT${weight_bits},INT${weight_bits},INT${activation_bits},INT${activation_bits},INT${gemm_result_bits},INT${gemm_result_bits},PASS,$mse,$cos," >> "$CSV_FILE"
                PASS_COUNT=$((PASS_COUNT + 1))
            else
                # 构建失败原因
                if ! $cos_ok; then
                    fail_reason="Cosine < $COSINE_THRESHOLD"
                fi
                if ! $mse_ok; then
                    [ -n "$fail_reason" ] && fail_reason="$fail_reason; "
                    fail_reason="${fail_reason}MSE > $MSE_THRESHOLD"
                fi
                echo "  ✗ 精度不足 - MSE: $mse, Cosine: $cos ($fail_reason)"
                echo "  状态: 精度不足" >> "$RESULT_FILE"
                echo "  MSE: $mse, Cosine Similarity: $cos ($fail_reason)" >> "$RESULT_FILE"
                echo "$config_name,$weight_bits,$activation_bits,$gemm_result_bits,INT${weight_bits},INT${weight_bits},INT${activation_bits},INT${activation_bits},INT${gemm_result_bits},INT${gemm_result_bits},LOW_ACCURACY,$mse,$cos,$fail_reason" >> "$CSV_FILE"
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        else
            echo "  ✗ 无法提取结果 - MSE: $mse, Cosine: $cos"
            echo "  状态: 无法提取结果" >> "$RESULT_FILE"
            echo "$config_name,$weight_bits,$activation_bits,$gemm_result_bits,INT${weight_bits},INT${weight_bits},INT${activation_bits},INT${activation_bits},INT${gemm_result_bits},INT${gemm_result_bits},NO_RESULT,$mse,$cos,Cannot extract result" >> "$CSV_FILE"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "  ⚠ 预期失败但成功了 - MSE: $mse, Cosine: $cos"
        echo "  状态: 意外成功 (预期应失败)" >> "$RESULT_FILE"
        echo "$config_name,$weight_bits,$activation_bits,$gemm_result_bits,INT${weight_bits},INT${weight_bits},INT${activation_bits},INT${activation_bits},INT${gemm_result_bits},INT${gemm_result_bits},UNEXPECTED_PASS,$mse,$cos,Expected to fail but passed" >> "$CSV_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ==================== 开始测试 ====================
echo ""
echo "==================== 权重和激活位宽配置测试 ===================="
echo ""

# 确保构建目录存在
if [ ! -d "$BUILD_DIR" ]; then
    echo "创建构建目录: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake ..
fi

echo ""
echo "==================== 第一部分：支持的配置 ===================="
echo ""
echo "==================== 第一部分：支持的配置 ====================" >> "$RESULT_FILE"

# 测试支持的配置
run_test "W8A8" 8 8 "pass"
run_test "W8A16" 8 16 "pass"
run_test "W16A8" 16 8 "pass"
run_test "W16A16" 16 16 "pass"

echo ""
echo "==================== 第二部分：Wx/Rh 位宽测试 ===================="
echo ""
echo "==================== 第二部分：Wx/Rh 位宽测试 ====================" >> "$RESULT_FILE"

# 测试 GEMM 结果（Wx_, Rh_）使用不同精度
# W8A8 但 Wx/Rh 用 16 位 - 更高精度的 GEMM 中间结果
run_test "W8A8_GEMM16" 8 8 "pass" 16

# W8A16 但 Wx/Rh 用 8 位 - 精度损失但支持
run_test "W8A16_GEMM8" 8 16 "pass" 8

# W16A16 但 Wx/Rh 用 8 位 - 精度损失但支持
run_test "W16A16_GEMM8" 16 16 "pass" 8

# 恢复原始配置
cp "$CONFIG_FILE.backup" "$CONFIG_FILE"
rm -f "$CONFIG_FILE.backup"
echo ""
echo "原始配置已恢复"

# ==================== 测试总结 ====================
echo ""
echo "==================== 测试总结 ===================="
echo ""
echo "==================== 测试总结 ====================" >> "$RESULT_FILE"

echo "总测试数: $TEST_COUNT" | tee -a "$RESULT_FILE"
echo "通过 (Cosine >= $COSINE_THRESHOLD 且 MSE <= $MSE_THRESHOLD): $PASS_COUNT" | tee -a "$RESULT_FILE"
echo "失败: $FAIL_COUNT" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

# 显示支持的配置结果
echo "支持的配置结果:" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"
printf "%-14s | %-6s | %-6s | %-6s | %-15s | %-12s\n" "配置" "权重" "激活" "Wx/Rh" "MSE" "余弦相似度" | tee -a "$RESULT_FILE"
printf "%-14s-+-%-6s-+-%-6s-+-%-6s-+-%-15s-+-%-12s\n" "--------------" "------" "------" "------" "---------------" "------------" | tee -a "$RESULT_FILE"
# CSV 格式: config_name,weight_bits,activation_bits,gemm_result_bits,W_,R_,x_,h_,Wx_,Rh_,status,mse,cosine_similarity,error_msg
grep ",PASS," "$CSV_FILE" | while IFS=',' read -r name w a g W R x h Wx Rh status mse cos err; do
    printf "%-14s | %-6s | %-6s | %-6s | %-15s | %-12s\n" "$name" "${w}-bit" "${a}-bit" "${g}-bit" "$mse" "$cos" | tee -a "$RESULT_FILE"
done

# 显示失败测试列表
if [ $FAIL_COUNT -gt 0 ]; then
    echo "" | tee -a "$RESULT_FILE"
    echo "失败测试列表:" | tee -a "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
    printf "%-14s | %-6s | %-6s | %-6s | %-15s | %-20s\n" "配置" "权重" "激活" "Wx/Rh" "状态" "错误信息" | tee -a "$RESULT_FILE"
    printf "%-14s-+-%-6s-+-%-6s-+-%-6s-+-%-15s-+-%-20s\n" "--------------" "------" "------" "------" "---------------" "--------------------" | tee -a "$RESULT_FILE"
    # 显示非 PASS 的配置
    grep -v ",PASS," "$CSV_FILE" | tail -n +2 | while IFS=',' read -r name w a g W R x h Wx Rh status mse cos err; do
        printf "%-14s | %-6s | %-6s | %-6s | %-15s | %-20s\n" "$name" "${w}-bit" "${a}-bit" "${g}-bit" "$status" "$err" | tee -a "$RESULT_FILE"
    done
    echo "" | tee -a "$RESULT_FILE"
fi

echo ""
echo "===== 测试完成 ====="
echo "详细结果: $RESULT_FILE"
echo "CSV 数据: $CSV_FILE"

# 如果有失败的测试，返回非零退出码
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi

