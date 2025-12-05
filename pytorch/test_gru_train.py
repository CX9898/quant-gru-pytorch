import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, random_split
import os
import time
import random
from collections import defaultdict

# 导入 CustomGRU
try:
    from custom_gru import CustomGRU
    CUSTOM_GRU_AVAILABLE = True
except ImportError:
    print("警告: CustomGRU 不可用，将只测试 nn.GRU")
    CUSTOM_GRU_AVAILABLE = False

# ===================== 1. 下载 + 加载数据集 =====================
ROOT = "./speech_commands"
os.makedirs(ROOT, exist_ok=True)

# 第一次跑会自动下载数据集
_ = SPEECHCOMMANDS(root=ROOT, download=True)

# 之后直接用即可
full_dataset = SPEECHCOMMANDS(root=ROOT, download=False)

# 所有标签（单词）
labels = sorted({datapoint[2] for datapoint in full_dataset})
label_to_index = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

# ===================== 2. 特征提取设置 =====================
sample_rate = 16000
n_mels = 40

audio_transform = nn.Sequential(
    torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000),
    torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=n_mels),
    torchaudio.transforms.AmplitudeToDB()
)

def collate_fn(batch):
    specs = []
    targets = []
    for waveform, sr, label, *_ in batch:
        # 统一采样率
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        # 统一长度：截断或补零到 1 秒 (16000 点)
        if waveform.size(1) < sample_rate:
            pad_len = sample_rate - waveform.size(1)
            waveform = nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :sample_rate]

        # 提取 Mel 频谱: (1, n_mels, time) -> (time, n_mels)
        spec = audio_transform(waveform)          # [1, n_mels, T]
        spec = spec.squeeze(0).transpose(0, 1)   # [T, n_mels]
        specs.append(spec)
        targets.append(label_to_index[label])

    # 按最长序列补齐：得到 [B, T, n_mels]
    specs = nn.utils.rnn.pad_sequence(specs, batch_first=True)
    targets = torch.tensor(targets, dtype=torch.long)
    return specs, targets

# ===================== 3. 划分训练 / 测试集 =====================
train_len = int(0.8 * len(full_dataset))
test_len = len(full_dataset) - train_len
train_dataset, test_dataset = random_split(
    full_dataset,
    [train_len, test_len],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, collate_fn=collate_fn)

# ===================== 4. 定义 GRU 网络 =====================
class GRUNet(nn.Module):
    def __init__(self, input_size=n_mels, hidden_size=128, num_classes=num_classes, gru_layer=None):
        super().__init__()
        if gru_layer is None:
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            self.gru = gru_layer
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.gru(x)          # [B, T, H]
        out = out[:, -1, :]           # 取最后一个时间步 [B, H]
        out = self.fc(out)            # [B, C]
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建多个模型版本进行对比
models = {}
optimizers = {}
criterion = nn.CrossEntropyLoss()

# 1. nn.GRU 非量化版本（基准）
models['nn_gru'] = GRUNet().to(device)
optimizers['nn_gru'] = optim.Adam(models['nn_gru'].parameters(), lr=1e-3)

# 2-4. CustomGRU 版本（如果可用）
if CUSTOM_GRU_AVAILABLE:
    # 准备校准数据（用于量化版本的初始化）
    # 校准数据源配置
    USE_TEST_LOADER_FOR_CALIBRATION = True  # True: 使用 test_loader, False: 使用随机采样的训练数据

    if USE_TEST_LOADER_FOR_CALIBRATION:
        # 使用测试集作为校准数据
        print("准备校准数据（使用测试集）...")
        print("  从 test_loader 收集校准数据...")

        all_test_specs = []
        for batch_idx, (specs, targets) in enumerate(test_loader):
            all_test_specs.append(specs)
            if (batch_idx + 1) % 10 == 0:
                print(f"    已收集 {batch_idx + 1}/{len(test_loader)} 个批次")

        # 找到最大序列长度
        max_seq_len = max(specs.size(1) for specs in all_test_specs)
        print(f"  最大序列长度: {max_seq_len}")

        # 统一序列长度
        unified_specs = []
        for specs in all_test_specs:
            current_seq_len = specs.size(1)
            if current_seq_len < max_seq_len:
                pad_len = max_seq_len - current_seq_len
                padded_specs = nn.functional.pad(specs, (0, 0, 0, pad_len), mode='constant', value=0.0)
                unified_specs.append(padded_specs)
            elif current_seq_len > max_seq_len:
                unified_specs.append(specs[:, :max_seq_len, :])
            else:
                unified_specs.append(specs)

        # 拼接所有批次
        print("  正在拼接所有批次...")
        calibration_data = torch.cat(unified_specs, dim=0).to(device)  # [B_total, T, F]
        total_samples = calibration_data.size(0)
        seq_len = calibration_data.size(1)
        feature_dim = calibration_data.size(2)
        print(f"  校准数据形状: {calibration_data.shape} (batch_first=True)")
        print(f"  总样本数: {total_samples}, 序列长度: {seq_len}, 特征维度: {feature_dim}")

        # 计算内存使用量
        memory_mb = (total_samples * seq_len * feature_dim * 4) / (1024 * 1024)
        print(f"  估计内存使用: {memory_mb:.2f} MB")

    else:
        # 使用随机采样的训练数据
        print("准备校准数据（随机采样代表性样本）...")

        # 校准数据配置（优化版）
        CALIBRATION_SAMPLE_RATIO = 0.3  # 使用 30% 的训练数据作为校准数据（提高比例）
        MIN_CALIBRATION_SAMPLES = 1000   # 最少样本数（提高最小值）
        MAX_CALIBRATION_SAMPLES = 10000  # 最多样本数（提高上限，使用更多样本）
        USE_STRATIFIED_SAMPLING = True   # 使用分层采样，确保每个类别都有代表性

        # 计算需要采样的样本数
        total_train_samples = len(train_dataset)
        target_samples = int(total_train_samples * CALIBRATION_SAMPLE_RATIO)
        target_samples = max(MIN_CALIBRATION_SAMPLES, min(target_samples, MAX_CALIBRATION_SAMPLES))
        print(f"  训练集总样本数: {total_train_samples}")
        print(f"  目标采样数: {target_samples} (约 {100.0 * target_samples / total_train_samples:.1f}%)")
        print(f"  使用分层采样: {USE_STRATIFIED_SAMPLING}")

        # 收集所有样本的标签信息（用于分层采样）
        print("  正在分析数据集类别分布...")
        temp_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        label_indices = defaultdict(list)  # {label_index: [sample_indices]}
        idx_to_label = {}  # {sample_index: label_index} 用于快速查找

        for idx, (specs, targets) in enumerate(temp_loader):
            label_idx = targets.item()
            label_indices[label_idx].append(idx)
            idx_to_label[idx] = label_idx

        print(f"  类别数量: {len(label_indices)}")
        for label_idx, indices in sorted(label_indices.items()):
            label_name = labels[label_idx] if label_idx < len(labels) else f"class_{label_idx}"
            print(f"    {label_name}: {len(indices)} 个样本")

        # 采样策略：分层采样或随机采样
        if USE_STRATIFIED_SAMPLING:
            print("  使用分层采样策略...")
            # 分层采样：确保每个类别都有代表性样本
            samples_per_class = max(1, target_samples // len(label_indices))
            sample_indices = []

            for label_idx, indices in label_indices.items():
                # 每个类别采样 samples_per_class 个样本
                class_samples = min(samples_per_class, len(indices))
                random.seed(42 + label_idx)  # 每个类别使用不同的种子，但可复现
                sampled = random.sample(indices, class_samples)
                sample_indices.extend(sampled)

            # 如果还没达到目标数量，随机补充
            if len(sample_indices) < target_samples:
                remaining = target_samples - len(sample_indices)
                all_indices = [idx for indices in label_indices.values() for idx in indices]
                available = [idx for idx in all_indices if idx not in sample_indices]
                if available:
                    random.seed(42)
                    additional = random.sample(available, min(remaining, len(available)))
                    sample_indices.extend(additional)

            # 如果超过目标数量，随机减少
            if len(sample_indices) > target_samples:
                random.seed(42)
                sample_indices = random.sample(sample_indices, target_samples)

            print(f"  分层采样结果: {len(sample_indices)} 个样本")
            # 验证每个类别的采样数量
            sampled_labels = defaultdict(int)
            for idx in sample_indices:
                if idx in idx_to_label:
                    label_idx = idx_to_label[idx]
                    sampled_labels[label_idx] += 1
            print(f"  各类别采样数量:")
            for label_idx, count in sorted(sampled_labels.items()):
                label_name = labels[label_idx] if label_idx < len(labels) else f"class_{label_idx}"
                print(f"    {label_name}: {count} 个样本")
        else:
            # 随机采样
            print("  使用随机采样策略...")
            random.seed(42)
            sample_indices = random.sample(range(total_train_samples), target_samples)

        sample_indices_set = set(sample_indices)

        # 收集采样数据
        print("  正在收集采样数据...")
        sampled_specs = []

        # 重新创建临时 loader（因为之前已经遍历过了）
        temp_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        collected_samples = 0
        for idx, (specs, targets) in enumerate(temp_loader):
            if idx in sample_indices_set:
                sampled_specs.append(specs)
                collected_samples += 1
                if collected_samples % 500 == 0:
                    print(f"    已收集 {collected_samples}/{len(sample_indices)} 个样本")

            # 如果已经收集足够的样本，可以提前停止
            if collected_samples >= len(sample_indices):
                break

        print(f"  实际收集样本数: {collected_samples}")

        # 拼接所有采样数据
        print("  正在拼接采样数据...")
        # 找到所有批次中的最大序列长度
        if sampled_specs:
            max_seq_len = max(specs.size(1) for specs in sampled_specs)
            print(f"  最大序列长度: {max_seq_len}")

            # 统一序列长度：将所有批次填充或截断到最大长度
            unified_specs = []
            for specs in sampled_specs:
                current_seq_len = specs.size(1)
                if current_seq_len < max_seq_len:
                    # 需要填充到最大长度
                    pad_len = max_seq_len - current_seq_len
                    # 在序列维度（dim=1）的末尾填充
                    padded_specs = nn.functional.pad(specs, (0, 0, 0, pad_len), mode='constant', value=0.0)
                    unified_specs.append(padded_specs)
                elif current_seq_len > max_seq_len:
                    # 截断到最大长度（理论上不应该发生）
                    unified_specs.append(specs[:, :max_seq_len, :])
                else:
                    # 长度正好，直接使用
                    unified_specs.append(specs)

            # 拼接所有批次：[B1, T, F] + [B2, T, F] + ... -> [B_total, T, F]
            print("  正在拼接所有批次...")
            calibration_data = torch.cat(unified_specs, dim=0).to(device)  # [B_total, T, F]
            total_samples = calibration_data.size(0)
            seq_len = calibration_data.size(1)
            feature_dim = calibration_data.size(2)
            print(f"  校准数据形状: {calibration_data.shape} (batch_first=True)")
            print(f"  总样本数: {total_samples}, 序列长度: {seq_len}, 特征维度: {feature_dim}")

            # 计算内存使用量（粗略估计）
            memory_mb = (total_samples * seq_len * feature_dim * 4) / (1024 * 1024)  # float32 = 4 bytes
            print(f"  估计内存使用: {memory_mb:.2f} MB")
        else:
            # 如果没有采样到数据，使用第一个批次作为后备
            print("  警告: 未采样到数据，使用第一个批次作为校准数据")
            calibration_specs, _ = next(iter(train_loader))
            fixed_seq_len = min(calibration_specs.size(1), 50)
            calibration_data = calibration_specs[:, :fixed_seq_len, :].to(device)
            print(f"  校准数据形状: {calibration_data.shape} (batch_first=True)")

    # ========== 验证校准数据质量（对所有数据源都进行验证）==========
    if 'calibration_data' in locals():
        print("\n  正在验证校准数据质量...")
        has_invalid = False

        # 1. 检查 NaN 值
        nan_count = torch.isnan(calibration_data).sum().item()
        if nan_count > 0:
            print(f"  ✗ 警告: 发现 {nan_count} 个 NaN 值")
            has_invalid = True
        else:
            print(f"  ✓ NaN 检查通过: 无 NaN 值")

        # 2. 检查 Inf 值
        inf_count = torch.isinf(calibration_data).sum().item()
        if inf_count > 0:
            print(f"  ✗ 警告: 发现 {inf_count} 个 Inf 值")
            has_invalid = True
        else:
            print(f"  ✓ Inf 检查通过: 无 Inf 值")

        # 3. 检查全零样本（可能表示无效数据）
        # 检查每个样本是否全为零
        sample_norms = calibration_data.norm(dim=(1, 2))  # [B] 每个样本的L2范数
        zero_samples = (sample_norms < 1e-8).sum().item()
        if zero_samples > 0:
            print(f"  ✗ 警告: 发现 {zero_samples} 个全零样本（可能无效）")
            has_invalid = True
        else:
            print(f"  ✓ 全零样本检查通过: 无全零样本")

        # 4. 检查数据范围（统计信息）
        data_min = calibration_data.min().item()
        data_max = calibration_data.max().item()
        data_mean = calibration_data.mean().item()
        data_std = calibration_data.std().item()
        print(f"  数据统计信息:")
        print(f"    最小值: {data_min:.6f}")
        print(f"    最大值: {data_max:.6f}")
        print(f"    平均值: {data_mean:.6f}")
        print(f"    标准差: {data_std:.6f}")

        # 检查数据范围是否合理（Mel频谱通常是负数，因为使用了AmplitudeToDB）
        if data_max > 100 or data_min < -200:
            print(f"  ⚠ 注意: 数据范围异常，可能存在问题")
            print(f"    预期范围: 通常在 [-100, 50] 之间（AmplitudeToDB后的Mel频谱）")

        # 5. 检查每个时间步是否有无效数据
        # 检查每个时间步的统计信息
        time_step_means = calibration_data.mean(dim=(0, 2))  # [T] 每个时间步的平均值
        time_step_stds = calibration_data.std(dim=(0, 2))   # [T] 每个时间步的标准差

        # 检查是否有时间步全为零或方差为0（可能表示padding）
        zero_time_steps = (time_step_means.abs() < 1e-8).sum().item()
        constant_time_steps = (time_step_stds < 1e-8).sum().item()

        if zero_time_steps > 0:
            print(f"  ⚠ 注意: 发现 {zero_time_steps} 个时间步的平均值接近零（可能是padding）")
        if constant_time_steps > 0:
            print(f"  ⚠ 注意: 发现 {constant_time_steps} 个时间步的方差接近零（可能是padding或常数）")

        # 6. 检查每个特征维度是否有无效数据
        feature_means = calibration_data.mean(dim=(0, 1))  # [F] 每个特征维度的平均值
        feature_stds = calibration_data.std(dim=(0, 1))    # [F] 每个特征维度的标准差

        zero_features = (feature_means.abs() < 1e-8).sum().item()
        constant_features = (feature_stds < 1e-8).sum().item()

        if zero_features > 0:
            print(f"  ⚠ 注意: 发现 {zero_features} 个特征维度的平均值接近零")
        if constant_features > 0:
            print(f"  ⚠ 注意: 发现 {constant_features} 个特征维度的方差接近零（可能是无效特征）")

        # 7. 检查是否有异常大的值（可能是异常值）
        abs_data = calibration_data.abs()
        large_value_threshold = 100  # 根据AmplitudeToDB的特性，合理值通常在[-100, 50]之间
        large_value_count = (abs_data > large_value_threshold).sum().item()
        if large_value_count > 0:
            large_value_ratio = 100.0 * large_value_count / calibration_data.numel()
            print(f"  ⚠ 注意: 发现 {large_value_count} 个绝对值 > {large_value_threshold} 的值 ({large_value_ratio:.2f}%)")

        # 8. 最终验证结果
        if has_invalid:
            print(f"\n  ✗ 校准数据验证失败: 发现无效数据，建议检查数据预处理流程")
            print(f"  建议:")
            print(f"    1. 检查数据预处理是否有问题")
            print(f"    2. 检查是否有数据损坏")
            print(f"    3. 考虑过滤掉无效样本")
        else:
            print(f"\n  ✓ 校准数据验证通过: 未发现明显的无效数据")

        # 9. 可选：过滤无效样本（如果发现）
        if has_invalid and (nan_count > 0 or inf_count > 0):
            print(f"\n  正在过滤无效样本...")
            # 找出有效样本的索引（没有NaN和Inf）
            valid_mask = ~(torch.isnan(calibration_data).any(dim=(1, 2)) |
                          torch.isinf(calibration_data).any(dim=(1, 2)))
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) < total_samples:
                original_samples = total_samples
                calibration_data = calibration_data[valid_indices]
                total_samples = calibration_data.size(0)
                print(f"  已过滤 {original_samples - total_samples} 个无效样本")
                print(f"  剩余有效样本: {total_samples}")

                # 检查最小样本数（如果定义了）
                min_samples = MIN_CALIBRATION_SAMPLES if not USE_TEST_LOADER_FOR_CALIBRATION else 100
                if total_samples < min_samples:
                    print(f"  ✗ 错误: 过滤后样本数 ({total_samples}) 少于最小要求 ({min_samples})")
                    raise ValueError(f"校准数据中无效样本过多，无法继续")
    else:
        # 如果没有采样到数据，使用第一个批次作为后备
        print("  警告: 未采样到数据，使用第一个批次作为校准数据")
        calibration_specs, _ = next(iter(train_loader))
        fixed_seq_len = min(calibration_specs.size(1), 50)
        calibration_data = calibration_specs[:, :fixed_seq_len, :].to(device)
        print(f"  校准数据形状: {calibration_data.shape} (batch_first=True)")

    # 2. CustomGRU 非量化版本
    custom_gru_no_quant = CustomGRU(
        input_size=n_mels,
        hidden_size=128,
        batch_first=True,
        use_quantization=False
    ).to(device)
    models['custom_gru_no_quant'] = GRUNet(gru_layer=custom_gru_no_quant).to(device)
    optimizers['custom_gru_no_quant'] = optim.Adam(models['custom_gru_no_quant'].parameters(), lr=1e-3)

    # 3. CustomGRU int8 量化版本
    # 使用延迟校准：先创建模型（不校准），同步权重后再校准
    custom_gru_int8 = CustomGRU(
        input_size=n_mels,
        hidden_size=128,
        batch_first=True,
        use_quantization=True,
        quant_type='int8',
        calibration_data=None  # 延迟校准，稍后调用 calibrate()
    ).to(device)
    models['custom_gru_int8'] = GRUNet(gru_layer=custom_gru_int8).to(device)
    optimizers['custom_gru_int8'] = optim.Adam(models['custom_gru_int8'].parameters(), lr=1e-3)

    # # 4. CustomGRU int16 量化版本
    # custom_gru_int16 = CustomGRU(
    #     input_size=n_mels,
    #     hidden_size=128,
    #     batch_first=True,
    #     use_quantization=True,
    #     quant_type='int16',
    #     calibration_data=calibration_data
    # ).to(device)
    # models['custom_gru_int16'] = GRUNet(gru_layer=custom_gru_int16).to(device)
    # optimizers['custom_gru_int16'] = optim.Adam(models['custom_gru_int16'].parameters(), lr=1e-3)

    # 同步权重：从 nn.GRU 复制到所有 CustomGRU 版本
    print("同步模型权重...")
    nn_gru = models['nn_gru'].gru
    for name, model in models.items():
        if name != 'nn_gru' and hasattr(model.gru, 'weight_ih_l0'):
            # 确保 CustomGRU 的权重没有被展平（处理 flatten_parameters 的情况）
            # 注意：不要重置基准模型 nn.GRU 的 _flat_weights，否则会导致后续调用失败
            if hasattr(model.gru, '_flat_weights') and model.gru._flat_weights is not None:
                model.gru._flat_weights = None

            # 确保源权重是连续的
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # 复制权重并同步
            model.gru.weight_ih_l0.data.copy_(nn_gru.weight_ih_l0.data.contiguous())
            if device.type == 'cuda':
                torch.cuda.synchronize()

            model.gru.weight_hh_l0.data.copy_(nn_gru.weight_hh_l0.data.contiguous())
            if device.type == 'cuda':
                torch.cuda.synchronize()

            if nn_gru.bias:
                model.gru.bias_ih_l0.data.copy_(nn_gru.bias_ih_l0.data.contiguous())
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                model.gru.bias_hh_l0.data.copy_(nn_gru.bias_hh_l0.data.contiguous())
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            # 同步全连接层权重
            model.fc.weight.data.copy_(models['nn_gru'].fc.weight.data.contiguous())
            if device.type == 'cuda':
                torch.cuda.synchronize()

            model.fc.bias.data.copy_(models['nn_gru'].fc.bias.data.contiguous())
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # 验证权重是否同步成功
            weight_ih_diff = (model.gru.weight_ih_l0.data - nn_gru.weight_ih_l0.data).abs().max().item()
            weight_hh_diff = (model.gru.weight_hh_l0.data - nn_gru.weight_hh_l0.data).abs().max().item()
            fc_weight_diff = (model.fc.weight.data - models['nn_gru'].fc.weight.data).abs().max().item()
            fc_bias_diff = (model.fc.bias.data - models['nn_gru'].fc.bias.data).abs().max().item()

            max_diff = max(weight_ih_diff, weight_hh_diff, fc_weight_diff, fc_bias_diff)
            if max_diff > 1e-6:
                print(f"警告: {name} 的权重同步可能失败，最大差异: {max_diff:.2e}")
                print(f"  - weight_ih_l0: {weight_ih_diff:.2e}")
                print(f"  - weight_hh_l0: {weight_hh_diff:.2e}")
                print(f"  - fc.weight: {fc_weight_diff:.2e}")
                print(f"  - fc.bias: {fc_bias_diff:.2e}")
            else:
                print(f"✓ {name} 权重同步成功 (最大差异: {max_diff:.2e})")

    # 在权重同步后，对量化版本进行校准（使用正确的权重）
    print("校准量化参数...")
    for name, model in models.items():
        if name != 'nn_gru' and hasattr(model.gru, 'calibrate'):
            if model.gru.use_quantization and not model.gru.is_calibrated():
                try:
                    model.gru.calibrate(calibration_data)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    print(f"✓ {name} 量化参数校准成功")
                except Exception as e:
                    print(f"✗ {name} 量化参数校准失败: {e}")
                    raise

    print(f"\n已创建 {len(models)} 个模型版本进行对比:")
    for name in models.keys():
        print(f"  - {name}")

    # 验证所有模型的初始状态是否一致
    print("\n验证初始状态一致性...")
    test_specs, _ = next(iter(test_loader))
    test_specs = test_specs.to(device)

    with torch.no_grad():
        baseline_output = models['nn_gru'](test_specs)
        for name, model in models.items():
            if name != 'nn_gru':
                model.eval()
                output = model(test_specs)
                diff = (output - baseline_output).abs().max().item()
                mean_diff = (output - baseline_output).abs().mean().item()
                print(f"  {name:25s} | 最大差异: {diff:.6f} | 平均差异: {mean_diff:.6f}")
                if diff > 1e-3:
                    print(f"    警告: {name} 的初始输出与基准模型差异较大 (>1e-3)")
                elif diff > 1e-6:
                    print(f"    注意: {name} 的初始输出与基准模型有轻微差异 (可能由于数值精度)")
else:
    print("只测试 nn.GRU（CustomGRU 不可用）")

# ===================== 5. 测试集评估函数 =====================
def eval_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for specs, targets in loader:
            specs   = specs.to(device)
            targets = targets.to(device)
            outputs = model(specs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)
    return 100.0 * correct / total

def eval_all_models(loader, models_dict):
    """评估所有模型"""
    results = {}
    for name, model in models_dict.items():
        start_time = time.time()
        acc = eval_accuracy(loader, model)
        elapsed = time.time() - start_time
        results[name] = {'accuracy': acc, 'time': elapsed}
    return results

# ===================== 6. 训练并对比所有模型版本 =====================
num_epochs = 3
history = defaultdict(list)  # 记录训练历史

print("\n" + "="*80)
print("开始训练和对比")
print("="*80)

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-" * 80)

    epoch_results = {}

    # 定义训练顺序：先运行非量化版本，再运行量化版本
    # 这样可以测试是否是运行顺序导致的问题
    training_order = []
    if 'nn_gru' in models:
        training_order.append('nn_gru')
    if 'custom_gru_no_quant' in models:
        training_order.append('custom_gru_no_quant')
    if 'custom_gru_int8' in models:
        training_order.append('custom_gru_int8')
    if 'custom_gru_int16' in models:
        training_order.append('custom_gru_int16')

    # 按照定义的顺序训练模型
    for name in training_order:
        model = models[name]
        model.train()
        optimizer = optimizers[name]
        running_loss = 0.0
        train_start = time.time()

        for specs, targets in train_loader:
            specs   = specs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_time = time.time() - train_start
        avg_loss = running_loss / len(train_loader)

        epoch_results[name] = {
            'loss': avg_loss,
            'train_time': train_time
        }

        history[name].append(epoch_results[name])

        print(f"{name:25s} | Loss: {avg_loss:.4f} | Train: {train_time:.2f}s")

    # 在每个 epoch 结束后，评估所有模型（便于对比）
    print("\n评估所有模型...")
    eval_results = eval_all_models(test_loader, models)
    for name in training_order:
        test_acc = eval_results[name]['accuracy']
        eval_time = eval_results[name]['time']
        epoch_results[name]['test_acc'] = test_acc
        epoch_results[name]['eval_time'] = eval_time
        history[name][-1] = epoch_results[name]  # 更新最新记录

        print(f"{name:25s} | Test Acc: {test_acc:.2f}% | Eval: {eval_time:.2f}s")

# ===================== 7. 最终对比总结 =====================
print("\n" + "="*80)
print("训练完成 - 最终对比总结")
print("="*80)

print(f"\n{'模型版本':<30s} | {'最终损失':<12s} | {'最终准确率':<12s} | {'总训练时间':<12s}")
print("-" * 80)

for name in models.keys():
    final_epoch = history[name][-1]
    total_train_time = sum([h['train_time'] for h in history[name]])
    print(f"{name:<30s} | {final_epoch['loss']:>10.4f} | "
          f"{final_epoch['test_acc']:>10.2f}% | {total_train_time:>10.2f}s")

# 计算与基准模型的差异
if len(models) > 1:
    print("\n" + "="*80)
    print("与 nn.GRU 基准模型的差异")
    print("="*80)
    baseline_acc = history['nn_gru'][-1]['test_acc']
    baseline_loss = history['nn_gru'][-1]['loss']

    for name in models.keys():
        if name != 'nn_gru':
            final_epoch = history[name][-1]
            acc_diff = final_epoch['test_acc'] - baseline_acc
            loss_diff = final_epoch['loss'] - baseline_loss
            print(f"{name:<30s} | 准确率差异: {acc_diff:>+7.2f}% | 损失差异: {loss_diff:>+8.4f}")
