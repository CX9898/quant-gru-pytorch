"""
AIMET 量化演示 - 简洁版
使用 utils.py 工具函数，代码结构清晰
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from aimet_torch.v2 import quantsim
from aimet_torch import model_preparer
import aimet_torch.v2 as aimet


from pathlib import Path
from tqdm import tqdm
import random
import math
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking


# 导入工具函数
from aimet_torch.utils_rx import (
    setup_percentile_calibration,
    verify_percentile_calibration,
    apply_power_of_2_workflow,
    freeze_quantizer_parameters,
    set_train_mode_freeze_bn
)

# 导入模型定义
from common.fft2band import BandConverter
from common.torch_stft import STFT
from aimet_torch.optimized_quantizable_gru import OptimizedQuantizableGRU



def list_from_txt(p):
    with open(p, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

class SpeechCommands(Dataset):
    """
    Builds train/val/test from speech_commands_v0.02 directory.
    Produces (sequence[T, F], label_idx).
    """
    def __init__(self, root, split="train", sample_rate=16000,
                 n_mels=1, win_ms=25.0, hop_ms=10.0,
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

        wav_tc = wav.T # from [1,T] to [T, 1]

        y = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return wav_tc, y

# 2) Define the MRNN model
EPS = 1e-8

# ============= 新增：将 power_compress 和 hypot_fun 改成 nn.Module =============
class PowerCompress(nn.Module):
    """Power compression module: (abs(x) ** 0.5) * sign(x)"""
    def forward(self, x):
        o = (torch.abs(x) ** 0.5) * torch.sign(x)
        return o

class HypotFun(nn.Module):
    """Hypot function module: sqrt(x^2 + y^2 + EPS)"""
    def forward(self, x, y):
        o = torch.sqrt(x ** 2 + y ** 2 + EPS)
        return o
# ============================================================================

# ??????????
class CLN(nn.Module):
    def __init__(self, factor=32):
        super(CLN, self).__init__()
        self.factor = factor

    def forward(self, x):
        # ??????ONNX???
        std = torch.mean(x**2, dim=(1, 3), keepdim=True)  # [B, 1, T, 1]
        std = torch.sqrt(std + EPS)  # [1, 1, 26, 1]
        x = x / std
        return x

class cfLN2D(nn.Module):
    def __init__(self, channel_size):
        super(cfLN2D, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [B C T F]
        Returns:
            o: [B C T F]
        """
        mean = torch.mean(y, dim=1, keepdim=True) # [B, 1, T, F]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [B, 1, T, F]
        o = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return o

class RNN2D(nn.Module):
    def __init__(self, H, layers=1, idx=0):
        """
        Args:
            H: Number of channels
            causal: causal or non-causal
        """
        super(RNN2D, self).__init__()
        # ✅ 使用优化版 GRU，共享算术 Module（减少 62.5% Module 数量）
        self.seq_t = OptimizedQuantizableGRU(input_size=H, hidden_size=H, batch_first=True, num_layers=layers)
        self.seq_t = nn.GRU(input_size=H, hidden_size=H, batch_first=True, num_layers=layers)
        self.conv_t = nn.Conv2d(H, H, (1, 1))
        self.idx = idx
        #self.rnn2d_bn = nn.GroupNorm(num_groups=1, num_channels=H, affine=False)
        self.rnn2d_bn = nn.BatchNorm2d(H, affine=True, momentum=0.01)
        self.cln = cfLN2D(H)

    def forward(self, x):
        x = self.rnn2d_bn(x)
        #x = torch.clamp(x, -4, 4)
        x = self.cln(x)
        #x = torch.clamp(x, 0, 4)
        # o = rearrange(x, 'b c t f -> (b f) t c')  # ????????????, ??????
        b, c, t, f = x.shape
        o = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        o = self.seq_t(o)[0]
        #o_t = rearrange(o, '(b f) t c -> b c t f', f=x.shape[-1])
        o_t = o.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()
        o_t = self.conv_t(o_t)
        return o_t


class FrequencyDownSampling(nn.Module):
    def __init__(self, in_size, out_size, stride=(1, 4), kernel_size=(1, 4)):
        super(FrequencyDownSampling, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size,
                                stride=stride, padding=(kernel_size[0] // 2, 0))

    def forward(self, ipt):
        """
        Args:
            ipts: [B, C, T, F1]
        Returns:
           opts: [B, C', T, F2]
        """
        ipt = torch.clamp(ipt, -0, 2)
        out = self.conv2d(ipt)
        out = torch.clamp(out, -2, 2)
        return out

class MRNN(nn.Module):
    def __init__(self, output_dim=256, # CNENDE
                 mic_num=1, fs=16000, NFFT=512, frame_size=160, fbank_num=240,
                 TimeMask=False, FreqMask=False, PixelMask=False):
        super(MRNN, self).__init__()
        self.TimeMask, self.FreqMask, self.PixelMask = TimeMask, FreqMask, PixelMask

        self.mic_num = mic_num
        self.fs, self.NFFT, self.frame_size = fs, NFFT, frame_size
        channels = [120, 240, 320]
        self.trans = STFT(filter_length=NFFT, hop_length=frame_size)
        freq_bins = NFFT // 2

        self.conv_in = nn.Conv2d(1, channels[0], (3, 3), stride=(1, 2), padding=(1, 1), groups=1)
        self.enc_seqs, self.freq_downs, self.neck_seqs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.freq_downs.append(FrequencyDownSampling(in_size=channels[0], out_size=channels[0], kernel_size=(2, 4), stride=(2, 4)))
        self.enc_seqs.append(RNN2D(channels[0]))
        self.freq_downs.append(FrequencyDownSampling(in_size=channels[0], out_size=channels[1], kernel_size=(2, 5), stride=(2, 5)))
        self.enc_seqs.append(RNN2D(channels[1]))
        self.freq_downs.append(FrequencyDownSampling(in_size=channels[1], out_size=channels[2], kernel_size=(1, 6), stride=(1, 6)))
        self.neck_seqs.append(RNN2D(channels[-1], layers=1))
        self.neck_seqs.append(RNN2D(channels[-1], layers=1))
        self.fc0 = nn.Linear(channels[-1], output_dim)

        #self.sigmoid0 = nn.Sigmoid()
        self.pre_bn = nn.BatchNorm2d(freq_bins, affine=True, momentum=0.01)
        #self.pre_bn = nn.GroupNorm(num_groups=1, num_channels=freq_bins, affine=False)
        self.fft2band = BandConverter(band_num=fbank_num, freq_bins=freq_bins)

        # ============= 注册 power_compress 和 hypot_fun 为子模块 =============
        self.power_compress_1 = PowerCompress()  # 第1次调用（Line 336）
        self.power_compress_2 = PowerCompress()  # 第2次调用（Line 351）
        self.hypot_fun = HypotFun()              # hypot_fun（Line 347）
        # ====================================================================

    def forward(self, ipt):
        """
        Args:
            ipt: [B T C] C = mic_num
            wav_len: B
        returns:
            opt: [B T2 classes]: probability of each class
        """
        #assert ipt.shape[-1] == self.mic_num, f"input shape should be [B T {self.mic_num}], but got {ipt.shape}"
        iptc = self.trans(ipt)  # (B C T F 2)
        iptc = self.power_compress_1(iptc[:, :, :, 1:, :])  # ← 使用 power_compress_1
        #iptc = rearrange(iptc, 'b c t f r -> b f t (c r)')
        iptc = iptc.permute(0, 3, 2, 1, 4).contiguous()  # (B, F, T, C, R)
        iptc = iptc.flatten(-2)                          # (B, F, T, C*R)
        iptc = self.pre_bn(iptc)
        #iptc = torch.clamp(iptc, -4, 4)
        #iptc = rearrange(iptc, 'b f t (c r) -> b c t f r', r=2)
        b, f, t, cr = iptc.shape
        r = 2
        c = cr // r
        iptc = iptc.view(b, f, t, c, r).permute(0, 3, 2, 1, 4).contiguous()
        mag = self.hypot_fun(iptc[..., 0], iptc[..., 1])  # ← 使用 self.hypot_fun
        #mag = torch.clamp(mag, 0, 4)
        opt = self.fft2band(mag)  # ??dp02 BM\BS
        #opt = torch.clamp(opt, 0, 4)
        opt = self.power_compress_2(opt)  # ← 使用 power_compress_2
        #opt = torch.clamp(opt, 0, 2)
        opt = self.conv_in(opt)
        opt = self.freq_downs[0](opt)
        for i in range(len(self.enc_seqs)):
            opt = self.enc_seqs[i](opt)
            opt = self.freq_downs[i + 1](opt)
        for i in range(len(self.neck_seqs)):
            opt = self.neck_seqs[i](opt)

        #opt = torch.clamp(opt, 0, 4)
        opt = opt.squeeze(-1).transpose(-1, -2)
        opt = self.fc0(opt)
        #opt = torch.clamp(opt, -8, 8)
        # opt = opt.mean(dim=1)
        opt = torch.mean(opt, dim=1)
        return opt


def evaluate(model, loader, device):
    """评估模型精度"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


# ========================================================================
# 1. 初始化
# ========================================================================
print("\n" + "="*70)
print("步骤 1: 初始化")
print("="*70)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*70)
print("AIMET 量化演示 - 简洁版")
print("="*70)
print(f"\n使用设备: {device}")

# ========================================================================
# 2. 加载数据和模型（省略详细代码）
# ========================================================================
print("\n" + "="*70)
print("步骤 2: 加载数据和模型")
print("="*70)
root_path = "/data1/zcx/datasets/speech_commands_v0.02"  # Provide path to SpeechCommands dataset
train_dataset = SpeechCommands(root_path, split='train',time_mask_p=0.0, freq_mask_p=0.0, add_noise_p=0.0)
test_dataset = SpeechCommands(root_path, split='test',time_mask_p=0.0, freq_mask_p=0.0, add_noise_p=0.0)
val_dataset = SpeechCommands(root_path, split='val',time_mask_p=0.0, freq_mask_p=0.0, add_noise_p=0.0)

def collate(batch):
    # All sequences are equal length after our preprocessing, so just stack.
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True, collate_fn=collate, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=4, pin_memory=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=4, pin_memory=True, collate_fn=collate)
calib_loader = DataLoader(test_dataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True, collate_fn=collate, drop_last=True)


# 3) Instantiate and train the model
model = MRNN(output_dim=35)  # Assuming 12 classes for speech commands
model.to(device)

# ========================================================================
# 3. 浮点模型训练和评估
# ========================================================================
print("\n" + "="*70)
print("步骤 3: 浮点模型训练和评估")
print("="*70)
# 3.1) 损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epoch_fp = 1
fp_model_path = "/home/sdong/Program/AIMET/output/float_model_trained.pth"
# 3.2) 训练模型
'''for epoch in range(1,epoch_fp+1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_fp}")
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(pbar, 1):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")
    # Save the trained floating-point model
    torch.save(model.state_dict(), fp_model_path)
    print(f"Floating-point model saved to: {fp_model_path}")'''
# 3.3) 评估浮点模型

#model.load_state_dict(torch.load(fp_model_path, map_location=device),strict=False)
model.eval()
fp_accuracy = evaluate(model, test_loader, device)
print(f"Floating point accuracy: {fp_accuracy}")
# ========================================================================
# 4. 准备模型
# ========================================================================
print("\n" + "="*70)
print("步骤 4: 准备模型")
print("="*70)
prepared_model = model_preparer.prepare_model(
    model,
    module_classes_to_exclude=[OptimizedQuantizableGRU],
    stateless_modules_to_preserve=[PowerCompress, HypotFun]
)

# ========================================================================
# 5. 创建 QuantizationSimModel
# ========================================================================
print("\n" + "="*70)
print("步骤 5: 创建 QuantizationSimModel")
print("="*70)
sample_input, _ = next(iter(train_loader))

# 使用自定义的量化配置文件
#config_file = "/home/sdong/Program/aimet/config/mrnn_quantsim_config_custom_mixed_precision.json"
#print(f"✅ 使用量化配置文件: {config_file}")

sim = quantsim.QuantizationSimModel(
    prepared_model,
    dummy_input=sample_input.to(device),
    quant_scheme='tf_enhanced',
    default_output_bw=8,
    default_param_bw=8,
)
print(sim.model)
# ========================================================================
# 6. 使用 Percentile 校准替代 MinMax（使用工具函数）
# ========================================================================
print("\n" + "="*70)
print("步骤 6: Percentile 观察量化器信息")
print("="*70)
converted_count = setup_percentile_calibration(
    sim.model,
    percentile=99.0,
    num_bins=2048,
    verbose=True
)
stats = verify_percentile_calibration(sim.model, verbose=True)

# ========================================================================
# 7. 校准量化参数
# ========================================================================
print("\n" + "="*70)
print("步骤 5: 校准量化参数")
print("="*70)

calib_batches = 100  # 使用 10 个批次校准
batch_count = 0
with aimet.nn.compute_encodings(sim.model):
    for idx, (x, _) in enumerate(calib_loader):
        if idx >= calib_batches:
            break
        x = x.to(device)
        sim.model(x)
        batch_count = idx + 1
print(f"✅ 校准完成，使用了 {batch_count} 个批次")

# ========================================================================
# 8. PTQ 评估
# ========================================================================
print("\n" + "="*70)
print("步骤 6: PTQ 评估")
print("="*70)
ptq_accuracy = evaluate(sim.model, test_loader, device)
print(f"PTQ 量化精度: {ptq_accuracy*100:.2f}%")

# ========================================================================
# 9. Power-of-2 量化（使用工具函数）
# ========================================================================
print("\n" + "="*70)
print("步骤 7: Power-of-2 量化")
print("="*70)

# ✅ 使用工具函数完成所有 Power-of-2 量化步骤
# 包括：打印量化器信息、收集修改前信息、应用 Po2、验证、对比
po2_results = apply_power_of_2_workflow(sim.model, verbose=True)

# 评估 Power-of-2 量化后的精度
po2_accuracy = evaluate(sim.model, test_loader, device)
print(f"Power-of-2 量化精度: {po2_accuracy*100:.2f}%")

# # ========================================================================
# # 10. QAT 微调
# # ========================================================================
# print("\n" + "="*70)
# print("步骤 8: QAT 微调")
# print("="*70)
#
# # 10.1) 冻结量化器参数（使用工具函数）
# freeze_result = freeze_quantizer_parameters(sim.model, verbose=True)
#
# # 10.2) 训练
# qat_epochs = 1
# optimizer = torch.optim.Adam([p for p in sim.model.parameters() if p.requires_grad], lr=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=qat_epochs)
# loss_fn = nn.CrossEntropyLoss()
#
#
# for epoch in range(1, qat_epochs + 1):
#     print(f"\n--- Epoch {epoch}/{qat_epochs} ---")
#     # 设置训练模式，但 BatchNorm 保持评估模式（使用工具函数）
#     set_train_mode_freeze_bn(sim.model)
#
#     running_loss = 0.0
#     valid_batches = 0
#     pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{qat_epochs}")
#
#     for batch_idx, (x, y) in enumerate(pbar, 1):
#         inputs, labels = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         outputs = sim.model(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         valid_batches += 1
#         pbar.set_postfix(
#             loss=f"{running_loss / valid_batches:.4f}",
#             lr=f"{scheduler.get_last_lr()[0]:.2e}"
#         )
#
# # ========================================================================
# # 11. 最终评估
# # ========================================================================
# print("\n" + "="*70)
# print("步骤 9: 最终评估")
# print("="*70)
# qat_accuracy = evaluate(sim.model, test_loader, device)
#
# # ========================================================================
# # 12. 打印总结
# # ========================================================================
# print("\n" + "="*70)
# print("量化精度总结")
# print("="*70)
# print(f"浮点模型精度:           {fp_accuracy*100:.2f}%")
# print(f"PTQ 量化精度:           {ptq_accuracy*100:.2f}%")
# print(f"Power-of-2 量化精度:    {po2_accuracy*100:.2f}%")
# print(f"QAT 微调后精度:         {qat_accuracy*100:.2f}%")
# print("="*70)
#
# # ========================================================================
# # 13. 保存模型
# # ========================================================================
# #sim.export('/home/sdong/Program/aimet/quantized_model', 'quantized_mrnn', dummy_input=sample_input)
# #print("\n✅ 模型已保存到 ./quantized_model")
#
# print("\n" + "="*70)
# print("量化流程完成！")
# print("="*70)
#
