import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, random_split
import os

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
    def __init__(self, input_size=n_mels, hidden_size=128, num_classes=num_classes):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.gru(x)          # [B, T, H]
        out = out[:, -1, :]           # 取最后一个时间步 [B, H]
        out = self.fc(out)            # [B, C]
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

# ===================== 6. 训练 3 个 epoch，并输出测试集准确率 =====================
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for specs, targets in train_loader:
        specs   = specs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    test_acc = eval_accuracy(test_loader, model)
    print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")
