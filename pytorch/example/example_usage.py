"""
QuantGRU 量化库使用示例

本示例展示如何使用 QuantGRU 进行：
- 基本推理（浮点/量化）
- 量化感知训练（QAT）
- 校准方法选择（MinMax / Histogram）
- 双向 GRU
- ONNX 导出（QDQ / 定点 / 浮点模式）
"""

import torch
import torch.nn as nn

# 添加库路径（根据实际安装位置修改）
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_gru import QuantGRU


def example_basic_usage():
    """
    示例 1: 基本使用（非量化）
    
    与 nn.GRU 用法完全一致
    """
    print("\n" + "=" * 60)
    print("示例 1: 基本使用（非量化）")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # 创建模型
    gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True  # 输入格式 [batch, seq, feature]
    ).cuda()
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, input_size).cuda()
    
    # 前向传播
    output, h_n = gru(x)
    
    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {output.shape}")
    print(f"隐藏状态:   {h_n.shape}")
    print("✅ 基本使用完成！")


def example_quantization_with_json():
    """
    示例 2: 使用 JSON 配置进行量化
    
    推荐方式：通过 JSON 文件配置量化参数
    """
    print("\n" + "=" * 60)
    print("示例 2: 使用 JSON 配置进行量化")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # 1. 创建模型并加载配置
    gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    # 加载 JSON 配置（自动设置 use_quantization）
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config/gru_quant_bitwidth_config.json"
    )
    gru.load_bitwidth_config(config_path)
    print(f"✅ 加载配置: {config_path}")
    print(f"   量化开关: use_quantization = {gru.use_quantization}")
    
    # 2. 校准（使用代表性数据）
    print("\n📊 开始校准...")
    calibration_data = torch.randn(batch_size, seq_len, input_size).cuda()
    gru.calibrate(calibration_data)
    print("✅ 校准完成！")
    
    # 3. 推理
    print("\n🚀 开始推理...")
    x = torch.randn(batch_size, seq_len, input_size).cuda()
    output, h_n = gru(x)
    
    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {output.shape}")
    print(f"隐藏状态:   {h_n.shape}")
    print("✅ 量化推理完成！")


def example_quantization_manual(bitwidth=8):
    """
    示例 3: 手动配置量化参数
    
    不使用 JSON 文件，直接在代码中设置
    
    Args:
        bitwidth: 量化位宽（8 或 16）
    """
    print("\n" + "=" * 60)
    print(f"示例 3: 手动配置量化参数 ({bitwidth}bit)")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # 1. 创建模型
    gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    # 2. 设置位宽
    gru.set_all_bitwidth(bitwidth)
    print(f"✅ 设置位宽: {bitwidth}bit 对称量化")
    
    # 3. 校准
    print("\n📊 开始校准...")
    calibration_data = torch.randn(batch_size, seq_len, input_size).cuda()
    gru.calibrate(calibration_data)
    print("✅ 校准完成！")
    
    # 4. 开启量化并推理
    gru.use_quantization = True
    print(f"   量化开关: use_quantization = {gru.use_quantization}")
    
    print("\n🚀 开始推理...")
    x = torch.randn(batch_size, seq_len, input_size).cuda()
    output, h_n = gru(x)
    
    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {output.shape}")
    print(f"隐藏状态:   {h_n.shape}")
    print(f"✅ {bitwidth}bit 量化推理完成！")


def example_compare_precision(bitwidth=8):
    """
    示例 4: 比较量化前后的精度差异
    
    Args:
        bitwidth: 量化位宽（8 或 16）
    """
    print("\n" + "=" * 60)
    print(f"示例 4: 比较量化前后的精度差异 ({bitwidth}bit)")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # 创建非量化模型（基准）
    gru_float = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True,
        use_quantization=False
    ).cuda()
    
    # 创建量化模型（复制权重）
    gru_quant = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    # 复制权重
    gru_quant.weight_ih_l0.data.copy_(gru_float.weight_ih_l0.data)
    gru_quant.weight_hh_l0.data.copy_(gru_float.weight_hh_l0.data)
    gru_quant.bias_ih_l0.data.copy_(gru_float.bias_ih_l0.data)
    gru_quant.bias_hh_l0.data.copy_(gru_float.bias_hh_l0.data)
    
    # 校准并开启量化
    x = torch.randn(batch_size, seq_len, input_size).cuda()
    gru_quant.set_all_bitwidth(bitwidth)
    gru_quant.calibrate(x)
    gru_quant.use_quantization = True
    
    # 比较输出
    gru_float.eval()
    gru_quant.eval()
    
    with torch.no_grad():
        output_float, _ = gru_float(x)
        output_quant, _ = gru_quant(x)
    
    # 计算误差
    mse = torch.mean((output_float - output_quant) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        output_float.flatten().unsqueeze(0),
        output_quant.flatten().unsqueeze(0)
    ).item()
    
    print(f"📊 {bitwidth}bit 精度比较结果:")
    print(f"   MSE (均方误差):     {mse:.6f}")
    print(f"   余弦相似度:         {cos_sim:.6f}")
    print(f"✅ {bitwidth}bit 精度比较完成！")


def example_training(bitwidth=8):
    """
    示例 5: 量化感知训练（QAT）
    
    任务：学习输入序列的简单变换（输入乘以固定系数）
    注意：前向传播使用量化，反向传播使用浮点
    
    Args:
        bitwidth: 量化位宽（8 或 16）
    """
    print("\n" + "=" * 60)
    print(f"示例 5: 量化感知训练 ({bitwidth}bit)")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 64  # 与 input_size 相同，便于构造目标
    batch_size = 8
    seq_len = 20
    num_epochs = 5
    
    # 创建模型
    gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    # 固定随机种子，确保每次运行结果一致
    torch.manual_seed(42)
    
    # 生成固定的训练数据（学习输入的 0.5 倍变换）
    x_train = torch.randn(batch_size, seq_len, input_size).cuda() * 0.5
    target_train = x_train * 0.5  # 简单的线性变换作为目标
    
    # 校准
    gru.set_all_bitwidth(bitwidth)
    gru.calibrate(x_train)
    gru.use_quantization = True
    
    # 创建优化器
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    
    # 训练循环
    gru.train()
    print(f"\n🏋️ 开始 {bitwidth}bit 量化训练...")
    
    for epoch in range(num_epochs):
        # 前向传播
        optimizer.zero_grad()
        output, _ = gru(x_train)
        
        # 计算损失
        loss = torch.mean((output - target_train) ** 2)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    print(f"✅ {bitwidth}bit 训练完成！（Loss 应持续下降）")


def example_calibration_method():
    """
    示例 6: 校准方法选择
    
    QuantGRU 支持两种校准方法:
    - 'minmax': 快速，适合对速度要求高的场景
    - 'histogram': AIMET 风格，精度更高，适合对精度要求高的场景
    """
    print("\n" + "=" * 60)
    print("示例 6: 校准方法选择")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # 创建基准模型（FP32）
    gru_base = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True,
        use_quantization=False
    ).cuda()
    
    # 生成测试数据
    torch.manual_seed(42)
    test_input = torch.randn(batch_size, seq_len, input_size).cuda()
    
    # FP32 基准输出
    gru_base.eval()
    with torch.no_grad():
        fp32_output, _ = gru_base(test_input)
    
    print("\n📊 对比两种校准方法:")
    print("-" * 50)
    
    results = {}
    
    for method in ['minmax', 'histogram']:
        # 创建量化模型（复制权重）
        gru_quant = QuantGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        ).cuda()
        
        # 复制权重
        gru_quant.weight_ih_l0.data.copy_(gru_base.weight_ih_l0.data)
        gru_quant.weight_hh_l0.data.copy_(gru_base.weight_hh_l0.data)
        gru_quant.bias_ih_l0.data.copy_(gru_base.bias_ih_l0.data)
        gru_quant.bias_hh_l0.data.copy_(gru_base.bias_hh_l0.data)
        
        # 设置校准方法
        gru_quant.calibration_method = method
        
        # 设置位宽并校准
        gru_quant.set_all_bitwidth(16)
        
        # 多批次校准（histogram 方法在多批次下效果更好）
        for _ in range(3):
            calib_data = torch.randn(batch_size, seq_len, input_size).cuda()
            gru_quant.calibrate(calib_data)
        
        # 开启量化并推理
        gru_quant.use_quantization = True
        gru_quant.eval()
        
        with torch.no_grad():
            quant_output, _ = gru_quant(test_input)
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            fp32_output.flatten().unsqueeze(0),
            quant_output.flatten().unsqueeze(0)
        ).item()
        
        results[method] = cos_sim
        method_desc = "MinMax (快速)" if method == 'minmax' else "Histogram (高精度)"
        print(f"   {method_desc:<20} 余弦相似度: {cos_sim:.6f}")
    
    print("-" * 50)
    print("\n💡 选择建议:")
    print("   • minmax:    校准速度快，适合快速迭代和调试")
    print("   • histogram: 精度更高，适合最终部署（推荐）")
    print(f"\n   默认使用 'histogram' 方法")
    print("✅ 校准方法对比完成！")

def example_bidirectional():
    """
    示例 7: 双向 GRU
    """
    print("\n" + "=" * 60)
    print("示例 7: 双向 GRU")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # 创建双向模型
    gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True,
        bidirectional=True  # 双向
    ).cuda()
    
    # 校准并开启量化
    x = torch.randn(batch_size, seq_len, input_size).cuda()
    gru.set_all_bitwidth(8)
    gru.calibrate(x)
    gru.use_quantization = True
    
    # 推理
    output, h_n = gru(x)
    
    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {output.shape}  (hidden_size * 2 = {hidden_size * 2})")
    print(f"隐藏状态:   {h_n.shape}  (num_directions = 2)")
    print("✅ 双向 GRU 完成！")


def example_onnx_export():
    """
    示例 8: ONNX 导出
    
    QuantGRU 支持导出为 ONNX 格式，便于部署到各类推理引擎。
    
    导出模式说明:
    - export_mode=False (默认): 使用 CUDA C++ 实现（高性能推理）
    - export_mode=True: 使用纯 PyTorch 实现（可被 ONNX 追踪）
    
    导出格式 (export_format):
    - 'float': 浮点格式（默认，与 Haste GRU 行为一致）
    - 'qdq': QDQ 格式，量化模型推荐（需要先校准）
    - 'fixedpoint': 纯定点，与 CUDA 量化完全一致（精度验证）
    """
    print("\n" + "=" * 60)
    print("示例 8: ONNX 导出")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 1
    seq_len = 20
    
    # 1. 创建并配置模型
    gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    print("\n📦 步骤 1: 配置量化参数")
    gru.set_all_bitwidth(16)  # 16bit 量化
    print("   ✅ 设置 16bit 量化")
    
    # 2. 校准
    print("\n📊 步骤 2: 校准模型")
    calibration_data = torch.randn(batch_size, seq_len, input_size).cuda()
    gru.calibrate(calibration_data)
    gru.finalize_calibration()
    gru.use_quantization = True
    print("   ✅ 校准完成")
    
    # 3. 切换到导出模式
    print("\n🔄 步骤 3: 切换到导出模式")
    gru.export_mode = True
    gru.eval()
    print(f"   export_mode = {gru.export_mode}")
    print(f"   导出格式: {gru.export_format}")
    
    # 4. 导出 ONNX
    print("\n📤 步骤 4: 导出 ONNX 模型")
    dummy_input = torch.randn(batch_size, seq_len, input_size).cuda()
    onnx_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "quant_gru_example.onnx"
    )
    
    torch.onnx.export(
        gru,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output', 'hidden'],
        dynamic_axes={
            'input': {0: 'batch', 1: 'seq_len'},
            'output': {0: 'batch', 1: 'seq_len'}
        },
        opset_version=14,
        dynamo=False,  # 使用传统 TorchScript 导出，避免 torch.export 兼容性问题
        verbose=False
    )
    print(f"   ✅ 导出成功: {onnx_path}")
    
    # 5. 验证导出的模型
    print("\n🔍 步骤 5: 验证 ONNX 模型")
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("   ✅ ONNX 模型验证通过")
        
        # 打印模型信息
        print(f"\n   模型信息:")
        print(f"   - IR 版本: {model.ir_version}")
        print(f"   - Opset 版本: {model.opset_import[0].version}")
        print(f"   - 输入数量: {len(model.graph.input)}")
        print(f"   - 输出数量: {len(model.graph.output)}")
    except ImportError:
        print("   ⚠️ 未安装 onnx 库，跳过验证")
    except Exception as e:
        print(f"   ⚠️ 验证失败: {e}")
    
    # 6. 恢复 CUDA 模式
    gru.export_mode = False
    print(f"\n🔄 恢复 CUDA 模式: export_mode = {gru.export_mode}")
    
    print("\n✅ ONNX 导出示例完成！")
    
    # 清理临时文件
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"   已清理临时文件: {onnx_path}")


def example_onnx_export_modes():
    """
    示例 9: ONNX 导出格式对比
    
    演示三种 ONNX 导出格式的区别和使用场景
    """
    print("\n" + "=" * 60)
    print("示例 9: ONNX 导出格式对比")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 4
    seq_len = 20
    
    # 创建基准模型
    gru_base = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    # 校准
    calibration_data = torch.randn(batch_size, seq_len, input_size).cuda()
    gru_base.set_all_bitwidth(16)
    gru_base.calibrate(calibration_data)
    gru_base.finalize_calibration()
    gru_base.use_quantization = True
    
    # 获取 CUDA 参考输出
    gru_base.eval()
    test_input = torch.randn(batch_size, seq_len, input_size).cuda()
    with torch.no_grad():
        cuda_output, _ = gru_base(test_input)
    
    print("\n📊 对比三种 ONNX 导出格式:")
    print("-" * 50)
    
    modes = [
        ('qdq', 'QDQ 格式（量化推荐）'),
        ('fixedpoint', '纯定点格式'),
        ('float', '浮点格式（默认）')
    ]
    
    gru_base.export_mode = True
    
    for mode, desc in modes:
        gru_base.export_format = mode
        
        with torch.no_grad():
            export_output, _ = gru_base(test_input)
        
        # 计算与 CUDA 输出的相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            cuda_output.flatten().unsqueeze(0),
            export_output.flatten().unsqueeze(0)
        ).item()
        
        mse = torch.mean((cuda_output - export_output) ** 2).item()
        
        print(f"\n   模式: {mode}")
        print(f"   描述: {desc}")
        print(f"   余弦相似度: {cos_sim:.6f}")
        print(f"   MSE: {mse:.8f}")
    
    gru_base.export_mode = False
    
    print("\n" + "-" * 50)
    print("\n💡 模式选择建议:")
    print("   • 'qdq':        生产部署，推理引擎自动优化")
    print("   • 'fixedpoint': 精度验证，与 CUDA 完全一致")
    print("   • 'float':      调试和基准测试")
    
    print("\n✅ 导出模式对比完成！")


def main():
    """运行所有示例"""
    print("=" * 60)
    print("  QuantGRU 量化库使用示例")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ 错误: 需要 CUDA 支持")
        return
    
    try:
        # 运行所有示例
        example_basic_usage()
        example_quantization_with_json()
        
        # 示例 3: 手动配置量化参数（8bit 和 16bit）
        example_quantization_manual(bitwidth=8)
        example_quantization_manual(bitwidth=16)
        
        # 示例 4: 比较量化前后的精度差异（8bit 和 16bit）
        example_compare_precision(bitwidth=8)
        example_compare_precision(bitwidth=16)
        
        # 示例 5: 量化感知训练（8bit 和 16bit）
        example_training(bitwidth=8)
        example_training(bitwidth=16)
        
        # 示例 6: 校准方法选择
        example_calibration_method()
        
        # 示例 7: 双向 GRU
        example_bidirectional()
        
        # 示例 8: ONNX 导出
        example_onnx_export()
        
        # 示例 9: ONNX 导出子模式对比
        example_onnx_export_modes()
        
        print("\n" + "=" * 60)
        print("  所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

