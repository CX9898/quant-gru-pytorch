"""
QuantGRU 量化库使用示例

本示例展示如何使用 QuantGRU 进行：
- 基本推理（浮点/量化）
- 量化感知训练（QAT）
- 校准方法选择（MinMax / SQNR / Percentile）
- 双向 GRU
- ONNX 导出（float 浮点 / qdq 伪量化）
- 量化参数导出/导入
- 量化配置调整与查看
"""

import torch
import torch.nn as nn

# 添加库路径（根据实际安装位置修改）
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_gru import QuantGRU, print_quant_config, print_quant_params


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
    
    注意：在 JSON 配置文件中，可以为权重(W, R)和偏置(bw, br)设置量化粒度：
    - "quantization_granularity": "PER_TENSOR" - 整个tensor一个scale
    - "quantization_granularity": "PER_GATE" - 每个门一个scale（3个门）
    - "quantization_granularity": "PER_CHANNEL" - 每个输出通道一个scale（默认）
    
    详见示例 13 了解如何通过代码设置量化粒度
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
    
    # 新的校准方式：设置 calibrating=True 后进行 forward
    gru.calibrating = True
    _ = gru(calibration_data)
    gru.calibrating = False
    
    print("✅ 校准完成！")
    
    # 3. 推理
    print("\n🚀 开始推理...")
    gru.use_quantization = True
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
    
    gru.calibrating = True
    _ = gru(calibration_data)
    gru.calibrating = False
    
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
    quant_gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    # 复制权重
    quant_gru.weight_ih_l0.data.copy_(gru_float.weight_ih_l0.data)
    quant_gru.weight_hh_l0.data.copy_(gru_float.weight_hh_l0.data)
    quant_gru.bias_ih_l0.data.copy_(gru_float.bias_ih_l0.data)
    quant_gru.bias_hh_l0.data.copy_(gru_float.bias_hh_l0.data)
    
    # 校准并开启量化
    x = torch.randn(batch_size, seq_len, input_size).cuda()
    quant_gru.set_all_bitwidth(bitwidth)
    
    quant_gru.calibrating = True
    _ = quant_gru(x)
    quant_gru.calibrating = False
    
    quant_gru.use_quantization = True
    
    # 比较输出
    gru_float.eval()
    quant_gru.eval()
    
    with torch.no_grad():
        output_float, _ = gru_float(x)
        output_quant, _ = quant_gru(x)
    
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
    
    gru.calibrating = True
    _ = gru(x_train)
    gru.calibrating = False
    
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
    
    QuantGRU 支持三种校准方法:
    - 'minmax': 快速，使用 min/max 范围
    - 'sqnr': SQNR 优化搜索最优 scale（基于直方图，高精度）
    - 'percentile': 百分位裁剪（基于直方图）
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
    
    print("\n📊 对比三种校准方法:")
    print("-" * 50)
    
    results = {}
    
    for method in ['minmax', 'sqnr', 'percentile']:
        # 创建量化模型（复制权重）
        quant_gru = QuantGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        ).cuda()
        
        # 复制权重
        quant_gru.weight_ih_l0.data.copy_(gru_base.weight_ih_l0.data)
        quant_gru.weight_hh_l0.data.copy_(gru_base.weight_hh_l0.data)
        quant_gru.bias_ih_l0.data.copy_(gru_base.bias_ih_l0.data)
        quant_gru.bias_hh_l0.data.copy_(gru_base.bias_hh_l0.data)
        
        # 设置校准方法
        quant_gru.calibration_method = method
        
        # 如果是 percentile 方法，可以设置百分位值
        if method == 'percentile':
            quant_gru.percentile_value = 99.99
        
        # 设置位宽
        quant_gru.set_all_bitwidth(16)
        
        # 多批次校准（sqnr/percentile 方法在多批次下效果更好）
        quant_gru.calibrating = True
        for _ in range(3):
            calib_data = torch.randn(batch_size, seq_len, input_size).cuda()
            _ = quant_gru(calib_data)
        quant_gru.calibrating = False
        
        # 开启量化并推理
        quant_gru.use_quantization = True
        quant_gru.eval()
        
        with torch.no_grad():
            quant_output, _ = quant_gru(test_input)
        
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            fp32_output.flatten().unsqueeze(0),
            quant_output.flatten().unsqueeze(0)
        ).item()
        
        results[method] = cos_sim
        method_desc = {
            'minmax': 'MinMax (快速)',
            'sqnr': 'SQNR (高精度)',
            'percentile': 'Percentile (抗异常值)'
        }[method]
        print(f"   {method_desc:<25} 余弦相似度: {cos_sim:.6f}")
    
    print("-" * 50)
    print("\n💡 选择建议:")
    print("   • minmax:     校准速度快，适合快速迭代和调试")
    print("   • sqnr:       精度更高，搜索最优 scale（推荐）")
    print("   • percentile: 对异常值鲁棒，适合含噪声数据")
    print(f"\n   默认使用 'minmax' 方法")
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
    
    gru.calibrating = True
    _ = gru(x)
    gru.calibrating = False
    
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
    
    导出模式 (export_mode):
    - False (默认): 使用 CUDA C++ 实现（高性能推理）
    - True: 使用纯 PyTorch 实现（可被 ONNX 追踪）
    
    导出格式 (export_format):
    - 'float': 浮点格式（默认）
    - 'qdq': QDQ 伪量化格式（量化模型推荐，需先校准）
    
    注意事项:
    - 导出前必须设置 export_mode = True
    - QDQ 格式需要先校准
    - 导出后应恢复 export_mode = False 以使用 CUDA 推理
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
    
    gru.calibrating = True
    _ = gru(calibration_data)
    gru.calibrating = False
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
    
    演示两种 ONNX 导出格式的区别和使用场景:
    - 'float': 浮点格式（默认）
    - 'qdq': QDQ 伪量化格式（量化模型推荐，需先校准）
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
    
    gru_base.calibrating = True
    _ = gru_base(calibration_data)
    gru_base.calibrating = False
    gru_base.finalize_calibration()
    
    gru_base.use_quantization = True
    
    # 获取 CUDA 参考输出
    gru_base.eval()
    test_input = torch.randn(batch_size, seq_len, input_size).cuda()
    with torch.no_grad():
        cuda_output, _ = gru_base(test_input)
    
    print("\n📊 对比两种 ONNX 导出格式:")
    print("-" * 50)
    
    modes = [
        ('float', '浮点格式（默认）'),
        ('qdq', 'QDQ 伪量化格式（量化推荐，需先校准）')
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
    print("\n💡 导出格式选择建议:")
    print("   • 'float': 非量化模型、通用部署（默认）")
    print("   • 'qdq':   量化模型部署，推理引擎自动优化（需先校准）")
    
    print("\n✅ 导出格式对比完成！")


def example_quant_params_export_import():
    """
    示例 10: 量化参数导出/导入
    
    演示如何：
    1. 校准后导出量化参数到 JSON 文件
    2. 在部署环境从 JSON 加载量化参数（无需重新校准）
    """
    print("\n" + "=" * 60)
    print("示例 10: 量化参数导出/导入")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # ========== 训练/校准环境 ==========
    print("\n📦 [训练环境] 校准并导出量化参数")
    print("-" * 50)
    
    gru_train = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    # 设置位宽并校准
    gru_train.set_all_bitwidth(8)
    gru_train.calibration_method = 'sqnr'  # 使用 SQNR 高精度校准
    
    calibration_data = torch.randn(batch_size, seq_len, input_size).cuda()
    gru_train.calibrating = True
    _ = gru_train(calibration_data)
    gru_train.calibrating = False
    gru_train.finalize_calibration()
    
    print("   ✅ 校准完成")
    
    # 导出量化参数
    quant_params_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "example_quant_params.json"
    )
    gru_train.export_quant_params(quant_params_path, verbose=True)
    
    # 同时保存模型权重
    weights_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "example_weights.pth"
    )
    torch.save(gru_train.state_dict(), weights_path)
    print(f"   ✅ 权重已保存到: {weights_path}")
    
    # ========== 部署环境 ==========
    print("\n📥 [部署环境] 加载量化参数")
    print("-" * 50)
    
    # 从 JSON 读取模型配置
    import json
    with open(quant_params_path) as f:
        config = json.load(f)["model_info"]
    
    # 创建模型
    gru_deploy = QuantGRU(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        batch_first=config["batch_first"],
        bidirectional=config["bidirectional"]
    ).cuda()
    
    # 加载权重
    gru_deploy.load_state_dict(torch.load(weights_path))
    print(f"   ✅ 权重已加载")
    
    # 加载量化参数
    gru_deploy.load_quant_params(quant_params_path, verbose=True)
    
    # 开启量化推理
    gru_deploy.use_quantization = True
    
    # ========== 验证一致性 ==========
    print("\n🔍 验证导出/导入一致性")
    print("-" * 50)
    
    gru_train.use_quantization = True
    gru_train.eval()
    gru_deploy.eval()
    
    test_input = torch.randn(batch_size, seq_len, input_size).cuda()
    with torch.no_grad():
        output_train, _ = gru_train(test_input)
        output_deploy, _ = gru_deploy(test_input)
    
    mse = torch.mean((output_train - output_deploy) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        output_train.flatten().unsqueeze(0),
        output_deploy.flatten().unsqueeze(0)
    ).item()
    
    print(f"   训练模型 vs 部署模型:")
    print(f"   MSE: {mse:.10f}")
    print(f"   余弦相似度: {cos_sim:.6f}")
    
    if mse < 1e-10:
        print("   ✅ 导出/导入一致性验证通过！")
    else:
        print("   ⚠️ 存在微小差异（可能是数值精度问题）")
    
    # 清理临时文件
    for path in [quant_params_path, weights_path]:
        if os.path.exists(path):
            os.remove(path)
    print(f"\n   已清理临时文件")
    
    print("\n✅ 量化参数导出/导入示例完成！")


def example_adjust_quant_config():
    """
    示例 11: 调整量化配置
    
    演示如何：
    1. 查看当前量化配置
    2. 调整单个算子的位宽/scale
    3. 观察调整前后的效果
    """
    print("\n" + "=" * 60)
    print("示例 11: 调整量化配置")
    print("=" * 60)
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # 创建并校准模型
    gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    gru.set_all_bitwidth(8)
    
    calibration_data = torch.randn(batch_size, seq_len, input_size).cuda()
    gru.calibrating = True
    _ = gru(calibration_data)
    gru.calibrating = False
    gru.finalize_calibration()
    
    print("\n📋 查看量化配置")
    print("-" * 50)
    
    # 查看单个算子配置
    config = gru.get_quant_config("update_gate_output")
    print(f"   update_gate_output 配置: {config}")
    
    # 查看所有配置（使用调试工具）
    print("\n📊 所有量化配置:")
    print_quant_config(gru, ["x", "h", "update_gate_output", "reset_gate_output", "new_gate_output"])
    
    # ========== 调整配置 ==========
    print("\n🔧 调整 update_gate_output 位宽: 8bit -> 16bit")
    print("-" * 50)
    
    # 调整前获取基准输出
    gru.use_quantization = True
    gru.eval()
    test_input = torch.randn(batch_size, seq_len, input_size).cuda()
    
    with torch.no_grad():
        output_before, _ = gru(test_input)
    
    # 调整位宽（会自动调整 scale）
    gru.adjust_quant_config("update_gate_output", bitwidth=16, verbose=True)
    
    # 调整后输出
    with torch.no_grad():
        output_after, _ = gru(test_input)
    
    # 比较差异
    diff = torch.mean((output_before - output_after) ** 2).item()
    print(f"\n   调整前后输出差异 (MSE): {diff:.8f}")
    
    # 查看调整后的配置
    new_config = gru.get_quant_config("update_gate_output")
    print(f"   调整后 update_gate_output 配置: {new_config}")
    
    print("\n✅ 量化配置调整示例完成！")


def example_debug_tools():
    """
    示例 12: 调试工具使用
    
    演示调试工具的使用方法：
    - print_quant_params(): 打印量化参数
    - print_quant_config(): 打印量化配置
    """
    print("\n" + "=" * 60)
    print("示例 12: 调试工具使用")
    print("=" * 60)
    
    from quant_gru import print_quant_params, print_quant_config, print_quant_ranges
    
    # 模型参数
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 20
    
    # 创建并校准模型
    gru = QuantGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True
    ).cuda()
    
    gru.set_all_bitwidth(8)
    gru.calibration_method = 'minmax'  # minmax 方法会记录范围
    
    calibration_data = torch.randn(batch_size, seq_len, input_size).cuda()
    gru.calibrating = True
    _ = gru(calibration_data)
    gru.calibrating = False
    
    # 1. 打印量化范围（校准收集的数值范围）
    print("\n📊 1. 量化范围 (print_quant_ranges)")
    print("-" * 50)
    print_quant_ranges(gru)
    
    # 2. 完成校准
    gru.finalize_calibration()
    
    # 3. 打印量化参数
    print("\n📊 2. 量化参数 (print_quant_params)")
    print("-" * 50)
    print_quant_params(gru)
    
    # 4. 打印量化配置（更详细的视图）
    print("\n📊 3. 量化配置详情 (print_quant_config)")
    print("-" * 50)
    print_quant_config(gru)
    
    print("\n✅ 调试工具示例完成！")


def example_weight_bias_granularity():
    """
    示例 13: 权重和偏置的量化粒度设置
    
    演示如何为权重(W, R)和偏置(bw, br)设置不同的量化粒度：
    - PER_TENSOR: 整个tensor使用一个scale（最简单，精度可能较低）
    - PER_GATE: 每个门使用一个scale（3个门：update, reset, new）
    - PER_CHANNEL: 每个输出通道使用一个scale（默认，精度最高）
    
    注意：量化粒度仅对 W, R, bw, br 四个算子有效
    """
    print("\n" + "=" * 60)
    print("示例 13: 权重和偏置的量化粒度设置")
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
    
    print("\n📊 对比三种量化粒度:")
    print("-" * 60)
    
    granularity_configs = [
        ('PER_TENSOR', '整个tensor一个scale（最简单）'),
        ('PER_GATE', '每个门一个scale（3个门）'),
        ('PER_CHANNEL', '每个输出通道一个scale（默认）')
    ]
    
    results = {}
    
    for granularity_name, description in granularity_configs:
        print(f"\n🔧 配置: {granularity_name}")
        print(f"   描述: {description}")
        
        # 创建量化模型（复制权重）
        quant_gru = QuantGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        ).cuda()
        
        # 复制权重
        quant_gru.weight_ih_l0.data.copy_(gru_base.weight_ih_l0.data)
        quant_gru.weight_hh_l0.data.copy_(gru_base.weight_hh_l0.data)
        quant_gru.bias_ih_l0.data.copy_(gru_base.bias_ih_l0.data)
        quant_gru.bias_hh_l0.data.copy_(gru_base.bias_hh_l0.data)
        
        # 设置位宽
        quant_gru.set_all_bitwidth(8)
        
        # 设置量化粒度（仅对 W, R, bw, br 有效）
        granularity_map = {
            'PER_TENSOR': 0,
            'PER_GATE': 1,
            'PER_CHANNEL': 2
        }
        granularity_value = granularity_map[granularity_name]
        
        # 设置权重和偏置的量化粒度
        quant_gru._bitwidth_config.W_granularity_ = granularity_value
        quant_gru._bitwidth_config.R_granularity_ = granularity_value
        quant_gru._bitwidth_config.bw_granularity_ = granularity_value
        quant_gru._bitwidth_config.br_granularity_ = granularity_value
        
        print(f"   ✅ W_granularity = {granularity_value} ({granularity_name})")
        print(f"   ✅ R_granularity = {granularity_value} ({granularity_name})")
        print(f"   ✅ bw_granularity = {granularity_value} ({granularity_name})")
        print(f"   ✅ br_granularity = {granularity_value} ({granularity_name})")
        
        # 校准
        quant_gru.calibration_method = 'minmax'
        quant_gru.calibrating = True
        _ = quant_gru(test_input)
        quant_gru.calibrating = False
        quant_gru.finalize_calibration()
        
        # 开启量化并推理
        quant_gru.use_quantization = True
        quant_gru.eval()
        
        with torch.no_grad():
            quant_output, _ = quant_gru(test_input)
        
        # 计算精度指标
        mse = torch.mean((fp32_output - quant_output) ** 2).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            fp32_output.flatten().unsqueeze(0),
            quant_output.flatten().unsqueeze(0)
        ).item()
        
        results[granularity_name] = {
            'mse': mse,
            'cos_sim': cos_sim
        }
        
        print(f"   📈 MSE: {mse:.8f}")
        print(f"   📈 余弦相似度: {cos_sim:.6f}")
    
    print("\n" + "-" * 60)
    print("\n📊 量化粒度对比总结:")
    print("-" * 60)
    for granularity_name, description in granularity_configs:
        result = results[granularity_name]
        print(f"   {granularity_name:<15} MSE: {result['mse']:.8f}, 余弦相似度: {result['cos_sim']:.6f}")
    
    print("\n✅ 量化粒度设置示例完成！")


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
        
        # 示例 9: ONNX 导出格式对比
        example_onnx_export_modes()
        
        # 示例 10: 量化参数导出/导入
        example_quant_params_export_import()
        
        # 示例 11: 调整量化配置
        example_adjust_quant_config()
        
        # 示例 12: 调试工具使用
        example_debug_tools()
        
        # 示例 13: 权重和偏置的量化粒度设置
        example_weight_bias_granularity()
        
        print("\n" + "=" * 60)
        print("  所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
