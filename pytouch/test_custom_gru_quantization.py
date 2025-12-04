"""
测试 nn.GRU 和 CustomGRU 量化前向传播的比较

比较 PyTorch 原生 nn.GRU 和 CustomGRU 量化版本的前向传播结果
"""

import torch
import torch.nn as nn
from custom_gru import CustomGRU
import numpy as np

# ============================================================================
# 全局测试配置参数
# 修改这些参数可以统一控制所有测试的设置
# ============================================================================
# GRU 模型参数
INPUT_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1
BATCH_FIRST = False
BIAS = True

# 输入数据参数
BATCH_SIZE = 64
SEQ_LEN = 50

# 输入数据生成参数
INPUT_LOW = -0.7
INPUT_HIGH = 0.8
INPUT_DEVICE = 'cuda'
INPUT_DTYPE = torch.float32

# ============================================================================


def create_uniform_input(shape, device=None, dtype=None, low=None, high=None):
    """
    创建均匀分布的输入张量
    
    Args:
        shape: 张量形状
        device: 设备（默认使用全局配置 INPUT_DEVICE）
        dtype: 数据类型（默认使用全局配置 INPUT_DTYPE）
        low: 下界（默认使用全局配置 INPUT_LOW）
        high: 上界（默认使用全局配置 INPUT_HIGH）
    
    Returns:
        均匀分布在 [low, high] 的张量
    """
    if device is None:
        device = INPUT_DEVICE
    if dtype is None:
        dtype = INPUT_DTYPE
    if low is None:
        low = INPUT_LOW
    if high is None:
        high = INPUT_HIGH
    return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)


def create_test_input(batch_first=None, batch_size=None, seq_len=None, input_size=None):
    """
    根据全局配置创建测试输入数据
    
    Args:
        batch_first: 是否 batch_first（默认使用全局配置 BATCH_FIRST）
        batch_size: batch 大小（默认使用全局配置 BATCH_SIZE）
        seq_len: 序列长度（默认使用全局配置 SEQ_LEN）
        input_size: 输入大小（默认使用全局配置 INPUT_SIZE）
    
    Returns:
        测试输入张量
    """
    if batch_first is None:
        batch_first = BATCH_FIRST
    if batch_size is None:
        batch_size = BATCH_SIZE
    if seq_len is None:
        seq_len = SEQ_LEN
    if input_size is None:
        input_size = INPUT_SIZE
    
    if batch_first:
        return create_uniform_input((batch_size, seq_len, input_size))
    else:
        return create_uniform_input((seq_len, batch_size, input_size))


def compare_gru_outputs(
    pytorch_gru: nn.GRU,
    custom_gru: CustomGRU,
    input_tensor: torch.Tensor,
    hx: torch.Tensor = None,
    verbose: bool = True
):
    """
    比较 PyTorch GRU 和 CustomGRU 的输出

    Args:
        pytorch_gru: PyTorch 原生 GRU 模型
        custom_gru: CustomGRU 模型（量化或非量化）
        input_tensor: 输入张量
        hx: 初始隐藏状态（可选）
        verbose: 是否打印详细信息

    Returns:
        dict: 包含比较结果的字典
    """
    # 设置为评估模式
    pytorch_gru.eval()
    custom_gru.eval()

    with torch.no_grad():
        # PyTorch GRU 前向传播
        output_pt, h_n_pt = pytorch_gru(input_tensor, hx)

        # CustomGRU 前向传播
        output_custom, h_n_custom = custom_gru(input_tensor, hx)

    # 确保输出在同一设备上（用于比较）
    output_pt_cpu = output_pt.cpu()
    output_custom_cpu = output_custom.cpu()
    h_n_pt_cpu = h_n_pt.cpu()
    h_n_custom_cpu = h_n_custom.cpu()

    # 计算整体 MSE
    mse_output = torch.mean((output_pt_cpu - output_custom_cpu) ** 2).item()
    mse_h_n = torch.mean((h_n_pt_cpu - h_n_custom_cpu) ** 2).item()

    # 计算相对误差
    abs_output_pt = torch.abs(output_pt_cpu)
    rel_error_output = torch.mean(
        torch.abs(output_pt_cpu - output_custom_cpu) / (abs_output_pt + 1e-8)
    ).item()

    abs_h_n_pt = torch.abs(h_n_pt_cpu)
    rel_error_h_n = torch.mean(
        torch.abs(h_n_pt_cpu - h_n_custom_cpu) / (abs_h_n_pt + 1e-8)
    ).item()

    # 计算余弦相似度
    def cosine_similarity(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot_product = torch.sum(a_flat * b_flat).item()
        norm_a = torch.norm(a_flat).item()
        norm_b = torch.norm(b_flat).item()
        return dot_product / (norm_a * norm_b + 1e-8)

    cos_sim_output = cosine_similarity(output_pt_cpu, output_custom_cpu)
    cos_sim_h_n = cosine_similarity(h_n_pt_cpu, h_n_custom_cpu)

    # 计算每个时间步的误差（如果输入是序列）
    if len(output_pt_cpu.shape) == 3:
        seq_len = output_pt_cpu.shape[0] if not custom_gru.batch_first else output_pt_cpu.shape[1]
        mse_per_timestep = []
        cos_sim_per_timestep = []

        for t in range(seq_len):
            if custom_gru.batch_first:
                output_pt_t = output_pt_cpu[:, t, :].flatten()
                output_custom_t = output_custom_cpu[:, t, :].flatten()
            else:
                output_pt_t = output_pt_cpu[t, :, :].flatten()
                output_custom_t = output_custom_cpu[t, :, :].flatten()

            mse_t = torch.mean((output_pt_t - output_custom_t) ** 2).item()
            mse_per_timestep.append(mse_t)

            cos_sim_t = cosine_similarity(
                output_pt_t.unsqueeze(0),
                output_custom_t.unsqueeze(0)
            )
            cos_sim_per_timestep.append(cos_sim_t)
    else:
        mse_per_timestep = None
        cos_sim_per_timestep = None

    results = {
        'mse_output': mse_output,
        'mse_h_n': mse_h_n,
        'rel_error_output': rel_error_output,
        'rel_error_h_n': rel_error_h_n,
        'cos_sim_output': cos_sim_output,
        'cos_sim_h_n': cos_sim_h_n,
        'mse_per_timestep': mse_per_timestep,
        'cos_sim_per_timestep': cos_sim_per_timestep,
        'output_pt': output_pt_cpu,
        'output_custom': output_custom_cpu,
        'h_n_pt': h_n_pt_cpu,
        'h_n_custom': h_n_custom_cpu
    }

    if verbose:
        print_results(results, custom_gru)

    return results


def print_results(results: dict, custom_gru: CustomGRU):
    """打印比较结果"""
    print("=" * 80)
    print("nn.GRU vs CustomGRU 输出比较结果")
    print("=" * 80)
    print(f"量化类型: {custom_gru.quant_type if custom_gru.use_quantization else '非量化'}")
    print(f"使用量化: {custom_gru.use_quantization}")
    print()

    print("整体统计:")
    print(f"  输出 MSE:           {results['mse_output']:.10f}")
    print(f"  输出相对误差:       {results['rel_error_output']*100:.6f}%")
    print(f"  输出余弦相似度:     {results['cos_sim_output']:.10f}")
    print(f"  最终状态 MSE:       {results['mse_h_n']:.10f}")
    print(f"  最终状态相对误差:   {results['rel_error_h_n']*100:.6f}%")
    print(f"  最终状态余弦相似度: {results['cos_sim_h_n']:.10f}")
    print()

    if results['mse_per_timestep'] is not None:
        mse_list = results['mse_per_timestep']
        cos_sim_list = results['cos_sim_per_timestep']
        seq_len = len(mse_list)

        print("每个时间步的统计:")
        print(f"{'时间步':<8} {'MSE':<20} {'Cosine Similarity':<20}")
        print("-" * 80)

        # 打印前10个、后10个和中间的几个时间步
        print_indices = list(range(min(10, seq_len)))
        if seq_len > 20:
            print_indices.extend(list(range(seq_len // 2 - 2, seq_len // 2 + 3)))
        print_indices.extend(list(range(max(0, seq_len - 10), seq_len)))

        for t in sorted(set(print_indices)):
            print(f"{t:<8} {mse_list[t]:<20.10f} {cos_sim_list[t]:<20.10f}")

        print("-" * 80)
        print(f"{'平均':<8} {np.mean(mse_list):<20.10f} {np.mean(cos_sim_list):<20.10f}")
        print(f"{'最小':<8} {np.min(mse_list):<20.10f} {np.min(cos_sim_list):<20.10f}")
        print(f"{'最大':<8} {np.max(mse_list):<20.10f} {np.max(cos_sim_list):<20.10f}")
        print()

    print("=" * 80)


def test_non_quantized():
    """测试非量化版本"""
    print("\n" + "=" * 80)
    print("测试 1: 非量化 CustomGRU vs nn.GRU")
    print("=" * 80)

    # 使用全局配置参数
    # 创建 PyTorch GRU
    pytorch_gru = nn.GRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS
    ).cuda()

    # 创建 CustomGRU（非量化）
    custom_gru = CustomGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS,
        use_quantization=False
    ).cuda()

    # 复制权重
    custom_gru.weight_ih_l0.data.copy_(pytorch_gru.weight_ih_l0.data)
    custom_gru.weight_hh_l0.data.copy_(pytorch_gru.weight_hh_l0.data)
    custom_gru.bias_ih_l0.data.copy_(pytorch_gru.bias_ih_l0.data)
    custom_gru.bias_hh_l0.data.copy_(pytorch_gru.bias_hh_l0.data)

    # 创建输入（使用全局配置）
    x = create_test_input()

    # 比较输出
    results = compare_gru_outputs(pytorch_gru, custom_gru, x, verbose=True)

    # 验证结果（非量化应该非常接近）
    assert results['mse_output'] < 1e-5, f"非量化版本 MSE 过大: {results['mse_output']}"
    assert results['cos_sim_output'] > 0.9999, f"非量化版本余弦相似度过低: {results['cos_sim_output']}"
    print("✅ 非量化测试通过！")


def test_quantized_int8():
    """测试 int8 量化版本：nn.GRU vs CustomGRU 量化版本"""
    print("\n" + "=" * 80)
    print("测试 2: int8 量化 CustomGRU vs nn.GRU")
    print("=" * 80)

    # 使用全局配置参数
    # 创建 PyTorch GRU
    pytorch_gru = nn.GRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS
    ).cuda()

    # 创建校准数据（使用全局配置）
    calibration_data = create_test_input()
    
    # 保存原始校准数据的副本，用于后续测试（避免被 _initialize_quantization 修改）
    calibration_data_backup = calibration_data.clone()

    # 先创建 CustomGRU（不初始化量化，避免使用随机权重初始化量化）
    custom_gru = CustomGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS,
        use_quantization=False  # 先不启用量化
    ).cuda()

    # 复制权重（确保两个模型使用相同的权重）
    custom_gru.weight_ih_l0.data.copy_(pytorch_gru.weight_ih_l0.data)
    custom_gru.weight_hh_l0.data.copy_(pytorch_gru.weight_hh_l0.data)
    custom_gru.bias_ih_l0.data.copy_(pytorch_gru.bias_ih_l0.data)
    custom_gru.bias_hh_l0.data.copy_(pytorch_gru.bias_hh_l0.data)

    # 启用量化并初始化（使用正确的权重）
    custom_gru.use_quantization = True
    custom_gru.quant_type = 'int8'
    custom_gru._initialize_quantization(calibration_data)

    # 创建测试输入（使用备份的校准数据，确保一致性）
    # 注意：gru.cc 中使用相同的输入进行校准和测试，这样可以更准确地评估量化误差
    x = calibration_data.clone()  # 使用相同的输入数据

    # 比较输出
    results = compare_gru_outputs(pytorch_gru, custom_gru, x, verbose=True)
    # results2 = compare_gru_outputs(pytorch_gru, custom_gru, x, verbose=True)

    # 量化版本会有误差，但应该在合理范围内
    print(f"✅ int8 量化测试完成！MSE: {results['mse_output']:.6f}, "
          f"余弦相似度: {results['cos_sim_output']:.6f}")


def test_quantized_int16():
    """测试 int16 量化版本"""
    print("\n" + "=" * 80)
    print("测试 3: int16 量化 CustomGRU vs nn.GRU")
    print("=" * 80)

    # 使用全局配置参数
    # 创建 PyTorch GRU
    pytorch_gru = nn.GRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS
    ).cuda()

    # 创建校准数据（使用全局配置）
    calibration_data = create_test_input()

    # 先创建 CustomGRU（不初始化量化，避免使用随机权重初始化量化）
    custom_gru = CustomGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS,
        use_quantization=False  # 先不启用量化
    ).cuda()

    # 复制权重（确保两个模型使用相同的权重）
    custom_gru.weight_ih_l0.data.copy_(pytorch_gru.weight_ih_l0.data)
    custom_gru.weight_hh_l0.data.copy_(pytorch_gru.weight_hh_l0.data)
    custom_gru.bias_ih_l0.data.copy_(pytorch_gru.bias_ih_l0.data)
    custom_gru.bias_hh_l0.data.copy_(pytorch_gru.bias_hh_l0.data)

    # 启用量化并初始化（使用正确的权重）
    custom_gru.use_quantization = True
    custom_gru.quant_type = 'int16'
    custom_gru._initialize_quantization(calibration_data)

    # 创建测试输入（使用全局配置）
    x = create_test_input()

    # 比较输出
    results = compare_gru_outputs(pytorch_gru, custom_gru, x, verbose=True)

    # int16 量化应该比 int8 更精确
    print(f"✅ int16 量化测试完成！MSE: {results['mse_output']:.6f}, "
          f"余弦相似度: {results['cos_sim_output']:.6f}")


def test_batch_first():
    """测试 batch_first=True 的情况"""
    print("\n" + "=" * 80)
    print("测试 4: batch_first=True 的量化比较")
    print("=" * 80)

    # 覆盖全局参数（此测试需要不同的配置）
    input_size = 64
    hidden_size = 128
    batch_size = 8
    seq_len = 30
    batch_first = True

    # 创建 PyTorch GRU
    pytorch_gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
        batch_first=batch_first,
        bias=BIAS
    ).cuda()

    # 创建校准数据（使用覆盖的参数）
    calibration_data = create_test_input(
        batch_first=batch_first,
        batch_size=batch_size,
        seq_len=seq_len,
        input_size=input_size
    )

    # 先创建 CustomGRU（不初始化量化，避免使用随机权重初始化量化）
    custom_gru = CustomGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
        batch_first=batch_first,
        bias=BIAS,
        use_quantization=False  # 先不启用量化
    ).cuda()

    # 复制权重（确保两个模型使用相同的权重）
    custom_gru.weight_ih_l0.data.copy_(pytorch_gru.weight_ih_l0.data)
    custom_gru.weight_hh_l0.data.copy_(pytorch_gru.weight_hh_l0.data)
    custom_gru.bias_ih_l0.data.copy_(pytorch_gru.bias_ih_l0.data)
    custom_gru.bias_hh_l0.data.copy_(pytorch_gru.bias_hh_l0.data)

    # 启用量化并初始化（使用正确的权重）
    custom_gru.use_quantization = True
    custom_gru.quant_type = 'int8'
    custom_gru._initialize_quantization(calibration_data)

    # 创建测试输入（使用覆盖的参数）
    x = create_test_input(
        batch_first=batch_first,
        batch_size=batch_size,
        seq_len=seq_len,
        input_size=input_size
    )

    # 比较输出
    results = compare_gru_outputs(pytorch_gru, custom_gru, x, verbose=True)

    print(f"✅ batch_first=True 测试完成！")


def test_quantized_vs_non_quantized_int8():
    """测试 int8 量化版本：CustomGRU 非量化 vs CustomGRU 量化
    
    这个测试比较 CustomGRU 的非量化和量化版本，两者都使用相同的 Haste 格式权重
    这样可以更准确地评估量化带来的误差（类似 example/gru.cc 中的比较）
    """
    print("\n" + "=" * 80)
    print("测试 5: CustomGRU 非量化 vs CustomGRU int8 量化")
    print("=" * 80)

    # 使用全局配置参数
    # 创建 PyTorch GRU（用于获取初始权重）
    pytorch_gru = nn.GRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS
    ).cuda()

    # 创建校准数据（使用全局配置）
    calibration_data = create_test_input()

    # 创建 CustomGRU 非量化版本（作为基准）
    custom_gru_non_quant = CustomGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS,
        use_quantization=False
    ).cuda()

    # 创建 CustomGRU 量化版本
    custom_gru_quant = CustomGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS,
        use_quantization=False  # 先不启用量化
    ).cuda()

    # 复制权重（确保两个模型使用相同的权重）
    custom_gru_non_quant.weight_ih_l0.data.copy_(pytorch_gru.weight_ih_l0.data)
    custom_gru_non_quant.weight_hh_l0.data.copy_(pytorch_gru.weight_hh_l0.data)
    custom_gru_non_quant.bias_ih_l0.data.copy_(pytorch_gru.bias_ih_l0.data)
    custom_gru_non_quant.bias_hh_l0.data.copy_(pytorch_gru.bias_hh_l0.data)

    custom_gru_quant.weight_ih_l0.data.copy_(pytorch_gru.weight_ih_l0.data)
    custom_gru_quant.weight_hh_l0.data.copy_(pytorch_gru.weight_hh_l0.data)
    custom_gru_quant.bias_ih_l0.data.copy_(pytorch_gru.bias_ih_l0.data)
    custom_gru_quant.bias_hh_l0.data.copy_(pytorch_gru.bias_hh_l0.data)

    # 启用量化并初始化（使用正确的权重）
    custom_gru_quant.use_quantization = True
    custom_gru_quant.quant_type = 'int8'
    custom_gru_quant._initialize_quantization(calibration_data)

    # 创建测试输入（使用与校准数据相同的输入，确保一致性）
    # 注意：使用与校准数据相同的输入，确保一致性（类似 gru.cc 的做法）
    x = calibration_data.clone()  # 使用相同的输入数据

    # 调试：打印权重格式信息
    print("\n" + "=" * 80)
    print("调试信息：权重格式检查")
    print("=" * 80)
    weight_ih_pt = pytorch_gru.weight_ih_l0  # [3*hidden, input]
    weight_ih_t = weight_ih_pt.t()  # [input, 3*hidden]
    W_converted = custom_gru_quant._reorder_weights_pytorch_to_haste(weight_ih_t)
    print(f"PyTorch weight_ih_l0 shape: {weight_ih_pt.shape}")
    print(f"转置后 shape: {weight_ih_t.shape}")
    print(f"重排序后 shape: {W_converted.shape}")
    print(f"量化参数是否初始化: {custom_gru_quant.quant_params is not None}")
    if custom_gru_quant.quant_params is not None:
        print(f"量化参数 exp2_inv_h_: {custom_gru_quant.quant_params.exp2_inv_h_}")
        print(f"量化参数 zp_h_: {custom_gru_quant.quant_params.zp_h_}")
    print("=" * 80 + "\n")

    # 比较输出（非量化 vs 量化，两者都使用相同的 Haste 格式）
    results = compare_gru_outputs(custom_gru_non_quant, custom_gru_quant, x, verbose=True)

    # 量化版本会有误差，但应该在合理范围内（参考 example/gru.cc 的结果）
    # example 中 MSE 约 0.000007，余弦相似度约 0.999
    print(f"✅ CustomGRU 非量化 vs 量化测试完成！MSE: {results['mse_output']:.6f}, "
          f"余弦相似度: {results['cos_sim_output']:.6f}")
    print(f"   参考值（example/gru.cc）: MSE ~ 0.000007, 余弦相似度 ~ 0.999")
    
    # 验证结果是否在合理范围内
    if results['mse_output'] < 0.001 and results['cos_sim_output'] > 0.99:
        print("   ✅ 量化误差在合理范围内")
    else:
        print("   ⚠️  量化误差较大，可能需要检查量化实现")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("nn.GRU vs CustomGRU 量化前向传播比较测试")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持")
        return

    try:
        # 测试非量化版本
        test_non_quantized()

        # 测试 int8 量化
        test_quantized_int8()

        # # 测试 int16 量化
        # test_quantized_int16()

        # # 测试 batch_first=True
        # test_batch_first()

        # 测试 CustomGRU 非量化 vs 量化（更准确的量化误差评估）
        # test_quantized_vs_non_quantized_int8()

        print("\n" + "=" * 80)
        print("所有测试完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

