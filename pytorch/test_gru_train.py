import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path
import os
import time
import random
import math
from collections import defaultdict
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking

# 导入 CustomGRU
try:
    from custom_gru import CustomGRU
    CUSTOM_GRU_AVAILABLE = True
except ImportError:
    print("警告: CustomGRU 不可用，将只测试 nn.GRU")
    CUSTOM_GRU_AVAILABLE = False

# ===================== 0. SpeechCommands 数据集类 =====================
def list_from_txt(p):
    with open(p, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

class SpeechCommands(torch.utils.data.Dataset):
    """
    Builds train/val/test from speech_commands_v0.02 directory.
    Produces (sequence[T, 1], label_idx).
    """
    def __init__(self, root, split="train", sample_rate=16000,
                 n_mels=40, win_ms=25.0, hop_ms=10.0,
                 time_mask_p=0.2, freq_mask_p=0.2, add_noise_p=0.6,
                 time_shift_max=0.1, target_dur=1.0):
        self.root = Path(root)
        assert self.root.exists(), f"Data root not found: {root}"
        self.split = split
        self.sr = sample_rate
        self.target_len = int(target_dur * sample_rate)
        self.time_shift_max = time_shift_max
        self.add_noise_p = add_noise_p
        self.time_mask_p = time_mask_p
        self.freq_mask_p = freq_mask_p
        self.n_mels = n_mels

        # Build label list from subfolders (exclude background noise)
        self.labels = sorted([d.name for d in self.root.iterdir()
                              if d.is_dir() and not d.name.startswith('_')])
        self.label_to_idx = {c: i for i, c in enumerate(self.labels)}

        # Split using official lists
        val_list = set(list_from_txt(self.root / "validation_list.txt"))
        test_list = set(list_from_txt(self.root / "testing_list.txt"))

        all_items = []
        for label in self.labels:
            for wav in (self.root / label).glob("*.wav"):
                rel = f"{label}/{wav.name}"
                if rel in test_list:
                    sp = "test"
                elif rel in val_list:
                    sp = "val"
                else:
                    sp = "train"
                all_items.append((wav, label, sp))

        self.items = [(p, l) for (p, l, sp) in all_items if sp == split]
        if len(self.items) == 0:
            raise RuntimeError(f"No items for split={split} at {root}")

        # Load background noise wavs (for augmentation only)
        self.bg_noises = []
        noise_dir = self.root / "_background_noise_"
        if noise_dir.exists():
            for w in noise_dir.glob("*.wav"):
                wav, sr = torchaudio.load(w)
                if sr != self.sr:
                    wav = torchaudio.functional.resample(wav, sr, self.sr)
                self.bg_noises.append(wav.squeeze(0))  # mono

        # Feature pipeline
        win_length = int(self.sr * (win_ms / 1000.0))
        hop_length = int(self.sr * (hop_ms / 1000.0))
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2
        self.db = AmplitudeToDB()
        self.tmask = TimeMasking(time_mask_param=10)
        self.fmask = FrequencyMasking(freq_mask_param=8)

        # 特征提取：将波形转换为 Mel 频谱
        self.audio_transform = nn.Sequential(
            torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000),
            torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=n_mels),
            torchaudio.transforms.AmplitudeToDB()
        )

    def __len__(self):
        return len(self.items)

    def _pad_or_crop(self, wav, train=True):
        L = wav.shape[-1]
        if L < self.target_len:
            pad = self.target_len - L
            wav = torch.nn.functional.pad(wav, (0, pad))
        elif L > self.target_len:
            if self.split == "train" and train:
                start = random.randint(0, L - self.target_len)
            else:
                start = (L - self.target_len) // 2
            wav = wav[:, start:start + self.target_len]
        return wav

    def _time_shift(self, wav):
        if self.time_shift_max <= 0:
            return wav
        max_shift = int(self.target_len * self.time_shift_max)
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(wav, shifts=shift, dims=-1)

    def _mix_bg_noise(self, wav):
        if not self.bg_noises or random.random() > self.add_noise_p:
            return wav
        noise = random.choice(self.bg_noises)
        if noise.numel() < self.target_len:
            rep = (self.target_len // noise.numel()) + 1
            noise = noise.repeat(rep)
        start = random.randint(0, noise.numel() - self.target_len)
        noise_seg = noise[start:start + self.target_len].unsqueeze(0)

        # Random SNR from ~[-3, 15] dB
        snr_db = random.uniform(-3.0, 15.0)
        sig_pow = wav.pow(2).mean()
        noi_pow = noise_seg.pow(2).mean() + 1e-9
        k = math.sqrt(sig_pow / (noi_pow * (10 ** (snr_db / 10.0))))
        mixed = torch.clamp(wav + k * noise_seg, -1.0, 1.0)
        return mixed

    def _specaug(self, spec):
        if self.split == "train":
            if random.random() < self.time_mask_p:
                spec = self.tmask(spec)
            if random.random() < self.freq_mask_p:
                spec = self.fmask(spec)
        return spec

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav, sr = torchaudio.load(path)
        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        else:
            # ensure [1, T]
            pass
        # resample if needed
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        # training-time waveform augmentation
        train_mode = (self.split == "train")
        wav = self._pad_or_crop(wav, train=train_mode)
        if train_mode:
            wav = self._time_shift(wav)
            wav = self._mix_bg_noise(wav)

        # 提取 Mel 频谱: (1, n_mels, time) -> (time, n_mels)
        spec = self.audio_transform(wav)          # [1, n_mels, T]
        spec = spec.squeeze(0).transpose(0, 1)   # [T, n_mels]

        y = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return spec, y

# ===================== 1. 加载数据集 =====================
root_path = "../../../datasets/speech_commands/SpeechCommands/speech_commands_v0.02"  # 数据集路径
sample_rate = 16000
n_mels = 40

# 创建数据集实例（train, test）
train_dataset = SpeechCommands(root=root_path, split="train", n_mels=n_mels,
                                time_mask_p=0.0, freq_mask_p=0.0, add_noise_p=0.0)
test_dataset = SpeechCommands(root=root_path, split="test", n_mels=n_mels,
                                time_mask_p=0.0, freq_mask_p=0.0, add_noise_p=0.0)

# 获取标签信息（从训练集获取，因为所有split应该有相同的标签）
labels = train_dataset.labels
label_to_index = train_dataset.label_to_idx
num_classes = len(labels)

def collate_fn(batch):
    """
    处理 SpeechCommands 返回的数据格式: (spec, y)
    spec: [T, n_mels] 形状的 Mel 频谱特征
    y: 标签索引

    注意：由于所有序列都经过预处理为固定长度（1秒），
    经过 Mel 频谱转换后时间步数也相同，可以直接 stack
    """
    specs, targets = zip(*batch)
    specs = torch.stack(specs, dim=0)  # [B, T, n_mels]
    targets = torch.tensor(targets, dtype=torch.long)
    return specs, targets

# ===================== 3. 创建 DataLoader =====================
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True,
                          collate_fn=collate_fn, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True,
                          collate_fn=collate_fn)
calib_loader = DataLoader(test_dataset, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True,
                          collate_fn=collate_fn, drop_last=True)

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
    print("准备校准数据...")
    all_specs = []
    for batch_idx, (specs, targets) in enumerate(calib_loader):
        all_specs.append(specs)
        if (batch_idx + 1) % 10 == 0:
            print(f"  已收集 {batch_idx + 1}/{len(calib_loader)} 个批次")

    # 拼接所有批次
    calibration_data = torch.cat(all_specs, dim=0).to(device)  # [B_total, T, F]
    print(f"校准数据形状: {calibration_data.shape}")

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
