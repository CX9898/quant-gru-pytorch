import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, random_split
import os
import time
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
    print("准备校准数据...")
    calibration_specs, _ = next(iter(train_loader))
    calibration_data = calibration_specs.to(device)

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
    custom_gru_int8 = CustomGRU(
        input_size=n_mels,
        hidden_size=128,
        batch_first=True,
        use_quantization=True,
        quant_type='int8',
        calibration_data=calibration_data
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
            model.gru.weight_ih_l0.data.copy_(nn_gru.weight_ih_l0.data)
            model.gru.weight_hh_l0.data.copy_(nn_gru.weight_hh_l0.data)
            if nn_gru.bias:
                model.gru.bias_ih_l0.data.copy_(nn_gru.bias_ih_l0.data)
                model.gru.bias_hh_l0.data.copy_(nn_gru.bias_hh_l0.data)
            # 同步全连接层权重
            model.fc.weight.data.copy_(models['nn_gru'].fc.weight.data)
            model.fc.bias.data.copy_(models['nn_gru'].fc.bias.data)

    print(f"已创建 {len(models)} 个模型版本进行对比:")
    for name in models.keys():
        print(f"  - {name}")
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

        # 评估
        eval_results = eval_all_models(test_loader, {name: model})
        test_acc = eval_results[name]['accuracy']
        eval_time = eval_results[name]['time']

        epoch_results[name] = {
            'loss': avg_loss,
            'test_acc': test_acc,
            'train_time': train_time,
            'eval_time': eval_time
        }

        history[name].append(epoch_results[name])

        print(f"{name:25s} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Train: {train_time:.2f}s | Eval: {eval_time:.2f}s")

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
