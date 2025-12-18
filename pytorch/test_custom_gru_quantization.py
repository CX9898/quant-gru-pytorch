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
    print(f"使用量化: {custom_gru.use_quantization}")
    print()

    print("整体统计:")
    print(f"  输出 MSE:           {results['mse_output']:.10f}")
    print(f"  输出相对误差:       {results['rel_error_output'] * 100:.6f}%")
    print(f"  输出余弦相似度:     {results['cos_sim_output']:.10f}")
    print(f"  最终状态 MSE:       {results['mse_h_n']:.10f}")
    print(f"  最终状态相对误差:   {results['rel_error_h_n'] * 100:.6f}%")
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
    """测试 8bit 量化版本：nn.GRU vs CustomGRU 量化版本"""
    print("\n" + "=" * 80)
    print("测试 2: 8bit 量化 CustomGRU vs nn.GRU")
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

    # 设置 8bit 量化
    custom_gru.set_all_bitwidth(8, verbose=True)

    # 校准并完成量化初始化
    custom_gru.calibrate(calibration_data)
    custom_gru.finalize_calibration()
    custom_gru.use_quantization = True

    # 创建测试输入（使用与校准数据相同的输入，确保一致性）
    # 注意：gru.cc 中使用相同的输入进行校准和测试，这样可以更准确地评估量化误差
    x = calibration_data.clone()  # 使用相同的输入数据

    # 比较输出
    results = compare_gru_outputs(pytorch_gru, custom_gru, x, verbose=True)

    # 量化版本会有误差，但应该在合理范围内
    print(f"✅ 8bit 量化测试完成！MSE: {results['mse_output']:.6f}, "
          f"余弦相似度: {results['cos_sim_output']:.6f}")


def test_quantized_int16():
    """测试 16bit 量化版本"""
    print("\n" + "=" * 80)
    print("测试 3: 16bit 量化 CustomGRU vs nn.GRU")
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

    # 设置 16bit 量化
    custom_gru.set_all_bitwidth(16, verbose=True)

    # 校准并完成量化初始化
    custom_gru.calibrate(calibration_data)
    custom_gru.finalize_calibration()
    custom_gru.use_quantization = True

    # 创建测试输入（使用全局配置）
    x = create_test_input()

    # 比较输出
    results = compare_gru_outputs(pytorch_gru, custom_gru, x, verbose=True)

    # 16bit 量化应该比 8bit 更精确
    print(f"✅ 16bit 量化测试完成！MSE: {results['mse_output']:.6f}, "
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

    # 校准并完成量化初始化
    custom_gru.calibrate(calibration_data)
    custom_gru.finalize_calibration()
    custom_gru.use_quantization = True

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


def compare_gru_training(
        pytorch_gru: nn.GRU,
        custom_gru: CustomGRU,
        input_tensor: torch.Tensor,
        hx: torch.Tensor = None,
        verbose: bool = True
):
    """
    比较 PyTorch GRU 和 CustomGRU 的训练（前向 + 反向传播）

    Args:
        pytorch_gru: PyTorch 原生 GRU 模型
        custom_gru: CustomGRU 模型（量化或非量化）
        input_tensor: 输入张量
        hx: 初始隐藏状态（可选）
        verbose: 是否打印详细信息

    Returns:
        dict: 包含比较结果的字典
    """
    # 设置为训练模式
    pytorch_gru.train()
    custom_gru.train()

    # 清除之前的梯度
    pytorch_gru.zero_grad()
    custom_gru.zero_grad()

    # 需要输入梯度
    input_pt = input_tensor.clone().detach().requires_grad_(True)
    input_custom = input_tensor.clone().detach().requires_grad_(True)

    # 前向传播
    output_pt, h_n_pt = pytorch_gru(input_pt, hx)
    output_custom, h_n_custom = custom_gru(input_custom, hx)

    # 创建目标和损失函数
    target = torch.randn_like(output_pt)
    
    # 计算损失（MSE Loss）
    loss_pt = torch.mean((output_pt - target) ** 2)
    loss_custom = torch.mean((output_custom - target) ** 2)

    # 反向传播
    loss_pt.backward()
    loss_custom.backward()

    # 收集梯度
    grad_input_pt = input_pt.grad.cpu() if input_pt.grad is not None else None
    grad_input_custom = input_custom.grad.cpu() if input_custom.grad is not None else None

    grad_weight_ih_pt = pytorch_gru.weight_ih_l0.grad.cpu() if pytorch_gru.weight_ih_l0.grad is not None else None
    grad_weight_ih_custom = custom_gru.weight_ih_l0.grad.cpu() if custom_gru.weight_ih_l0.grad is not None else None

    grad_weight_hh_pt = pytorch_gru.weight_hh_l0.grad.cpu() if pytorch_gru.weight_hh_l0.grad is not None else None
    grad_weight_hh_custom = custom_gru.weight_hh_l0.grad.cpu() if custom_gru.weight_hh_l0.grad is not None else None

    grad_bias_ih_pt = pytorch_gru.bias_ih_l0.grad.cpu() if pytorch_gru.bias_ih_l0.grad is not None else None
    grad_bias_ih_custom = custom_gru.bias_ih_l0.grad.cpu() if custom_gru.bias_ih_l0.grad is not None else None

    grad_bias_hh_pt = pytorch_gru.bias_hh_l0.grad.cpu() if pytorch_gru.bias_hh_l0.grad is not None else None
    grad_bias_hh_custom = custom_gru.bias_hh_l0.grad.cpu() if custom_gru.bias_hh_l0.grad is not None else None

    # 计算各种比较指标
    def compute_metrics(a, b, name):
        """计算两个张量之间的比较指标"""
        if a is None or b is None:
            return {'name': name, 'available': False}
        
        mse = torch.mean((a - b) ** 2).item()
        max_diff = torch.max(torch.abs(a - b)).item()
        
        # 相对误差
        abs_a = torch.abs(a)
        rel_error = torch.mean(torch.abs(a - b) / (abs_a + 1e-8)).item()
        
        # 余弦相似度
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot_product = torch.sum(a_flat * b_flat).item()
        norm_a = torch.norm(a_flat).item()
        norm_b = torch.norm(b_flat).item()
        cos_sim = dot_product / (norm_a * norm_b + 1e-8)
        
        return {
            'name': name,
            'available': True,
            'mse': mse,
            'max_diff': max_diff,
            'rel_error': rel_error,
            'cos_sim': cos_sim
        }

    # 前向传播比较
    output_pt_cpu = output_pt.detach().cpu()
    output_custom_cpu = output_custom.detach().cpu()
    
    results = {
        'loss_pt': loss_pt.item(),
        'loss_custom': loss_custom.item(),
        'forward': compute_metrics(output_pt_cpu, output_custom_cpu, 'output'),
        'grad_input': compute_metrics(grad_input_pt, grad_input_custom, 'grad_input'),
        'grad_weight_ih': compute_metrics(grad_weight_ih_pt, grad_weight_ih_custom, 'grad_weight_ih'),
        'grad_weight_hh': compute_metrics(grad_weight_hh_pt, grad_weight_hh_custom, 'grad_weight_hh'),
        'grad_bias_ih': compute_metrics(grad_bias_ih_pt, grad_bias_ih_custom, 'grad_bias_ih'),
        'grad_bias_hh': compute_metrics(grad_bias_hh_pt, grad_bias_hh_custom, 'grad_bias_hh'),
    }

    if verbose:
        print_training_results(results, custom_gru)

    return results


def print_training_results(results: dict, custom_gru: CustomGRU):
    """打印训练比较结果"""
    print("=" * 80)
    print("nn.GRU vs CustomGRU 训练比较结果")
    print("=" * 80)
    print(f"使用量化: {custom_gru.use_quantization}")
    print()

    print(f"损失值比较:")
    print(f"  PyTorch GRU Loss:  {results['loss_pt']:.10f}")
    print(f"  CustomGRU Loss:    {results['loss_custom']:.10f}")
    print(f"  Loss 差异:         {abs(results['loss_pt'] - results['loss_custom']):.10f}")
    print()

    print("梯度比较:")
    print(f"{'名称':<20} {'MSE':<20} {'最大差异':<20} {'相对误差%':<15} {'余弦相似度':<15}")
    print("-" * 90)

    for key in ['forward', 'grad_input', 'grad_weight_ih', 'grad_weight_hh', 'grad_bias_ih', 'grad_bias_hh']:
        m = results[key]
        if m['available']:
            print(f"{m['name']:<20} {m['mse']:<20.10f} {m['max_diff']:<20.10f} "
                  f"{m['rel_error']*100:<15.6f} {m['cos_sim']:<15.10f}")
        else:
            print(f"{m['name']:<20} {'N/A':<20} {'N/A':<20} {'N/A':<15} {'N/A':<15}")

    print("=" * 80)


def test_training_non_quantized():
    """测试非量化版本的训练比较"""
    print("\n" + "=" * 80)
    print("测试: 非量化 CustomGRU vs nn.GRU 训练比较")
    print("=" * 80)

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

    # 创建输入
    x = create_test_input()

    # 比较训练
    results = compare_gru_training(pytorch_gru, custom_gru, x, verbose=True)

    # 验证结果（非量化应该非常接近）
    assert results['forward']['mse'] < 1e-5, f"非量化版本前向 MSE 过大: {results['forward']['mse']}"
    assert results['grad_input']['mse'] < 1e-5, f"非量化版本输入梯度 MSE 过大: {results['grad_input']['mse']}"
    assert results['grad_weight_ih']['mse'] < 1e-4, f"非量化版本权重梯度 MSE 过大: {results['grad_weight_ih']['mse']}"
    print("✅ 非量化训练测试通过！")


def test_training_quantized_int8():
    """测试 8bit 量化版本的训练比较
    
    注意：量化版本的前向传播使用量化，但反向传播仍使用浮点计算。
    因此梯度差异主要来源于前向传播的量化误差。
    """
    print("\n" + "=" * 80)
    print("测试: 8bit 量化 CustomGRU vs nn.GRU 训练比较")
    print("=" * 80)

    # 创建 PyTorch GRU
    pytorch_gru = nn.GRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS
    ).cuda()

    # 创建校准数据
    calibration_data = create_test_input()

    # 创建 CustomGRU
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

    # 设置 8bit 量化并校准
    custom_gru.set_all_bitwidth(8)
    custom_gru.calibrate(calibration_data)
    custom_gru.finalize_calibration()
    custom_gru.use_quantization = True

    # 创建测试输入
    x = calibration_data.clone()

    # 比较训练
    results = compare_gru_training(pytorch_gru, custom_gru, x, verbose=True)

    print(f"✅ 8bit 量化训练测试完成！")
    print(f"   前向 MSE: {results['forward']['mse']:.6f}, 余弦相似度: {results['forward']['cos_sim']:.6f}")
    print(f"   输入梯度 MSE: {results['grad_input']['mse']:.6f}, 余弦相似度: {results['grad_input']['cos_sim']:.6f}")


def test_training_multiple_steps():
    """测试多步训练，比较参数更新后的差异"""
    print("\n" + "=" * 80)
    print("测试: 多步训练比较（非量化）")
    print("=" * 80)

    num_steps = 5
    learning_rate = 0.01

    # 创建 PyTorch GRU
    pytorch_gru = nn.GRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS
    ).cuda()

    # 创建 CustomGRU
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

    # 创建优化器
    optimizer_pt = torch.optim.SGD(pytorch_gru.parameters(), lr=learning_rate)
    optimizer_custom = torch.optim.SGD(custom_gru.parameters(), lr=learning_rate)

    # 设置为训练模式
    pytorch_gru.train()
    custom_gru.train()

    print(f"{'步骤':<8} {'PT Loss':<15} {'Custom Loss':<15} {'Loss 差异':<15} {'权重 MSE':<15}")
    print("-" * 70)

    # 固定随机种子以获得相同的目标
    torch.manual_seed(42)

    for step in range(num_steps):
        # 创建输入和目标
        x = create_test_input()
        target = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, device='cuda') if not BATCH_FIRST else \
                 torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device='cuda')

        # PyTorch GRU 前向 + 反向
        optimizer_pt.zero_grad()
        output_pt, _ = pytorch_gru(x)
        loss_pt = torch.mean((output_pt - target) ** 2)
        loss_pt.backward()
        optimizer_pt.step()

        # CustomGRU 前向 + 反向
        optimizer_custom.zero_grad()
        output_custom, _ = custom_gru(x)
        loss_custom = torch.mean((output_custom - target) ** 2)
        loss_custom.backward()
        optimizer_custom.step()

        # 计算权重差异
        weight_mse = torch.mean((pytorch_gru.weight_ih_l0 - custom_gru.weight_ih_l0) ** 2).item()

        print(f"{step:<8} {loss_pt.item():<15.6f} {loss_custom.item():<15.6f} "
              f"{abs(loss_pt.item() - loss_custom.item()):<15.10f} {weight_mse:<15.10f}")

    # 最终权重比较
    print("\n最终权重比较:")
    weight_ih_mse = torch.mean((pytorch_gru.weight_ih_l0 - custom_gru.weight_ih_l0) ** 2).item()
    weight_hh_mse = torch.mean((pytorch_gru.weight_hh_l0 - custom_gru.weight_hh_l0) ** 2).item()
    print(f"  weight_ih MSE: {weight_ih_mse:.10f}")
    print(f"  weight_hh MSE: {weight_hh_mse:.10f}")

    assert weight_ih_mse < 1e-4, f"多步训练后权重差异过大: {weight_ih_mse}"
    print("✅ 多步训练测试通过！")


def test_training_long_run(num_steps=100, use_quantization=False, bitwidth=8, print_interval=10):
    """
    长时间训练测试，观察误差累积情况
    
    Args:
        num_steps: 训练步数
        use_quantization: 是否使用量化
        bitwidth: 量化位宽（8 或 16）
        print_interval: 打印间隔
    """
    print("\n" + "=" * 80)
    quant_str = f"{bitwidth}bit 量化" if use_quantization else "非量化"
    print(f"测试: 长时间训练比较（{quant_str}，{num_steps} 步）")
    print("=" * 80)

    learning_rate = 0.01

    # 创建 PyTorch GRU
    pytorch_gru = nn.GRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS
    ).cuda()

    # 创建校准数据（如果需要量化）
    calibration_data = create_test_input() if use_quantization else None

    # 创建 CustomGRU
    custom_gru = CustomGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=BATCH_FIRST,
        bias=BIAS,
        use_quantization=False  # 先不启用
    ).cuda()

    # 复制权重
    custom_gru.weight_ih_l0.data.copy_(pytorch_gru.weight_ih_l0.data)
    custom_gru.weight_hh_l0.data.copy_(pytorch_gru.weight_hh_l0.data)
    custom_gru.bias_ih_l0.data.copy_(pytorch_gru.bias_ih_l0.data)
    custom_gru.bias_hh_l0.data.copy_(pytorch_gru.bias_hh_l0.data)

    # 启用量化（如果需要）
    if use_quantization:
        custom_gru.set_all_bitwidth(bitwidth)
        custom_gru.calibrate(calibration_data)
        custom_gru.finalize_calibration()
        custom_gru.use_quantization = True

    # 创建优化器
    optimizer_pt = torch.optim.SGD(pytorch_gru.parameters(), lr=learning_rate)
    optimizer_custom = torch.optim.SGD(custom_gru.parameters(), lr=learning_rate)

    # 设置为训练模式
    pytorch_gru.train()
    custom_gru.train()

    print(f"{'步骤':<8} {'PT Loss':<12} {'Custom Loss':<12} {'Loss 差异':<15} "
          f"{'权重ih MSE':<15} {'输出余弦':<12}")
    print("-" * 85)

    # 固定随机种子
    torch.manual_seed(42)

    # 记录历史数据
    loss_diff_history = []
    weight_mse_history = []
    output_cos_history = []

    for step in range(num_steps):
        # 创建输入和目标
        x = create_test_input()
        target = torch.randn(SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, device='cuda') if not BATCH_FIRST else \
                 torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device='cuda')

        # PyTorch GRU 前向 + 反向
        optimizer_pt.zero_grad()
        output_pt, _ = pytorch_gru(x)
        loss_pt = torch.mean((output_pt - target) ** 2)
        loss_pt.backward()
        optimizer_pt.step()

        # CustomGRU 前向 + 反向
        optimizer_custom.zero_grad()
        output_custom, _ = custom_gru(x)
        loss_custom = torch.mean((output_custom - target) ** 2)
        loss_custom.backward()
        optimizer_custom.step()

        # 计算指标
        loss_diff = abs(loss_pt.item() - loss_custom.item())
        weight_mse = torch.mean((pytorch_gru.weight_ih_l0 - custom_gru.weight_ih_l0) ** 2).item()
        
        # 计算输出余弦相似度
        output_pt_flat = output_pt.detach().flatten()
        output_custom_flat = output_custom.detach().flatten()
        cos_sim = (torch.dot(output_pt_flat, output_custom_flat) / 
                   (torch.norm(output_pt_flat) * torch.norm(output_custom_flat) + 1e-8)).item()

        loss_diff_history.append(loss_diff)
        weight_mse_history.append(weight_mse)
        output_cos_history.append(cos_sim)

        # 按间隔打印
        if step % print_interval == 0 or step == num_steps - 1:
            print(f"{step:<8} {loss_pt.item():<12.6f} {loss_custom.item():<12.6f} "
                  f"{loss_diff:<15.10f} {weight_mse:<15.10f} {cos_sim:<12.8f}")

    # 最终统计
    print("\n" + "-" * 85)
    print("训练统计:")
    print(f"  总步数: {num_steps}")
    print(f"  Loss 差异 - 平均: {np.mean(loss_diff_history):.10f}, 最大: {np.max(loss_diff_history):.10f}")
    print(f"  权重 MSE - 平均: {np.mean(weight_mse_history):.10f}, 最大: {np.max(weight_mse_history):.10f}")
    print(f"  输出余弦 - 平均: {np.mean(output_cos_history):.8f}, 最小: {np.min(output_cos_history):.8f}")

    # 最终权重比较
    print("\n最终权重比较:")
    weight_ih_mse = torch.mean((pytorch_gru.weight_ih_l0 - custom_gru.weight_ih_l0) ** 2).item()
    weight_hh_mse = torch.mean((pytorch_gru.weight_hh_l0 - custom_gru.weight_hh_l0) ** 2).item()
    bias_ih_mse = torch.mean((pytorch_gru.bias_ih_l0 - custom_gru.bias_ih_l0) ** 2).item()
    bias_hh_mse = torch.mean((pytorch_gru.bias_hh_l0 - custom_gru.bias_hh_l0) ** 2).item()
    print(f"  weight_ih MSE: {weight_ih_mse:.10f}")
    print(f"  weight_hh MSE: {weight_hh_mse:.10f}")
    print(f"  bias_ih MSE:   {bias_ih_mse:.10f}")
    print(f"  bias_hh MSE:   {bias_hh_mse:.10f}")

    # 计算权重余弦相似度
    def weight_cos_sim(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        return (torch.dot(a_flat, b_flat) / (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)).item()

    print("\n最终权重余弦相似度:")
    print(f"  weight_ih: {weight_cos_sim(pytorch_gru.weight_ih_l0, custom_gru.weight_ih_l0):.10f}")
    print(f"  weight_hh: {weight_cos_sim(pytorch_gru.weight_hh_l0, custom_gru.weight_hh_l0):.10f}")

    print(f"\n✅ 长时间训练测试完成！（{quant_str}）")


def test_quantized_vs_non_quantized(bitwidth=8):
    """测试量化版本：CustomGRU 非量化 vs CustomGRU 量化
    
    这个测试比较 CustomGRU 的非量化和量化版本，两者都使用相同的 Haste 格式权重
    这样可以更准确地评估量化带来的误差（类似 example/gru.cc 中的比较）
    
    Args:
        bitwidth: 量化位宽（8 或 16）
    """
    print("\n" + "=" * 80)
    print(f"测试 5: CustomGRU 非量化 vs CustomGRU {bitwidth}bit 量化")
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

    # 设置位宽并校准
    custom_gru_quant.set_all_bitwidth(bitwidth)
    custom_gru_quant.calibrate(calibration_data)
    custom_gru_quant.finalize_calibration()
    custom_gru_quant.use_quantization = True

    # 创建测试输入（使用与校准数据相同的输入，确保一致性）
    # 注意：使用与校准数据相同的输入，确保一致性（类似 gru.cc 的做法）
    x = calibration_data.clone()  # 使用相同的输入数据

    # 调试：打印量化参数信息
    print("\n" + "=" * 80)
    print("调试信息：量化参数检查")
    print("=" * 80)
    print(f"量化位宽: {bitwidth}bit")
    print(f"量化参数是否初始化: {custom_gru_quant.quant_params is not None}")
    if custom_gru_quant.quant_params is not None:
        print(f"量化参数 exp2_inv_h_: {custom_gru_quant.quant_params.exp2_inv_h_}")
        print(f"量化参数 zp_h_: {custom_gru_quant.quant_params.zp_h_}")
    print("=" * 80 + "\n")

    # 比较输出（非量化 vs 量化，两者都使用相同的 Haste 格式）
    results = compare_gru_outputs(custom_gru_non_quant, custom_gru_quant, x, verbose=True)

    # 量化版本会有误差，但应该在合理范围内（参考 example/gru.cc 的结果）
    print(f"✅ CustomGRU 非量化 vs {bitwidth}bit 量化测试完成！MSE: {results['mse_output']:.6f}, "
          f"余弦相似度: {results['cos_sim_output']:.6f}")

    # 验证结果是否在合理范围内
    if results['mse_output'] < 0.001 and results['cos_sim_output'] > 0.99:
        print("   ✅ 量化误差在合理范围内")
    else:
        print("   ⚠️  量化误差较大，可能需要检查量化实现")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("nn.GRU vs CustomGRU 量化前向传播与训练比较测试")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 支持")
        return

    try:
        # ==================== 前向传播测试 ====================
        print("\n" + "#" * 80)
        print("# 前向传播测试")
        print("#" * 80)

        # 测试非量化版本
        test_non_quantized()

        # 测试 8bit 量化
        test_quantized_int8()

        # 测试 16bit 量化
        test_quantized_int16()

        # # 测试 batch_first=True
        # test_batch_first()

        # 测试 CustomGRU 非量化 vs 量化（更准确的量化误差评估）
        # test_quantized_vs_non_quantized_int8()

        # ==================== 训练测试 ====================
        print("\n" + "#" * 80)
        print("# 训练测试（前向 + 反向传播）")
        print("#" * 80)

        # 测试非量化版本训练
        test_training_non_quantized()

        # 测试 8bit 量化版本训练
        test_training_quantized_int8()

        # 测试多步训练
        test_training_multiple_steps()

        # ==================== 长时间训练测试 ====================
        print("\n" + "#" * 80)
        print("# 长时间训练测试（观察误差累积）")
        print("#" * 80)

        # 非量化长时间训练（100步）
        test_training_long_run(num_steps=100, use_quantization=False, print_interval=20)

        # 8bit 量化长时间训练（100步）
        test_training_long_run(num_steps=100, use_quantization=True, bitwidth=8, print_interval=20)

        # 16bit 量化长时间训练（100步）
        test_training_long_run(num_steps=100, use_quantization=True, bitwidth=16, print_interval=20)

        print("\n" + "=" * 80)
        print("所有测试完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
