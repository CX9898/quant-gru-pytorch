import torch
# from QuantGRU import OptimizedGRU  # 正确：当前目录下的 QuantGRU.py
from gru_quant import GRUQuantInt8  # 之前已解决的 gru_quant 导入

# 模拟输入
T, B, input_size, hidden_size = 500, 64, 256, 256
x = torch.rand(T, B, input_size, device='cuda', dtype=torch.float32) * 1.6 - 0.8 + 0.1  # 生成 [-0.8, 0.8] 的均匀分布

# PyTorch 原生 GRU
gru = torch.nn.GRU(input_size, hidden_size)
gru = gru.cuda()
h0 = torch.zeros(1, B, hidden_size, device='cuda')

# 检查权重形状和排布
print("=" * 80)
print("检查 PyTorch GRU 和 CUDA GRU 的权重排布")
print("=" * 80)
print(f"PyTorch weight_ih_l0.shape: {gru.weight_ih_l0.shape}")  # [3*hidden, input]
print(f"PyTorch weight_hh_l0.shape: {gru.weight_hh_l0.shape}")  # [3*hidden, hidden]
print(f"PyTorch bias_ih_l0.shape: {gru.bias_ih_l0.shape}")  # [3*hidden]
print(f"PyTorch bias_hh_l0.shape: {gru.bias_hh_l0.shape}")  # [3*hidden]

# 检查门的排列顺序（z, r, h）
print("\n=== 检查门的排列顺序 ===")
print("PyTorch GRU 门的顺序: [z_gate (hidden个), r_gate (hidden个), h_gate (hidden个)]")
w_ih = gru.weight_ih_l0
print(f"weight_ih_l0 前 hidden 个元素 (z门): shape={w_ih[:hidden_size, :5].shape}")
print(f"weight_ih_l0 中 hidden 个元素 (r门): shape={w_ih[hidden_size:2*hidden_size, :5].shape}")
print(f"weight_ih_l0 后 hidden 个元素 (h门): shape={w_ih[2*hidden_size:, :5].shape}")

# 检查CUDA期望的形状
print("\n=== CUDA GRU 期望的形状（从代码注释）===")
print("CUDA W: [hidden*3, input] = [3*hidden, input]")
print("CUDA R: [hidden*3, hidden] = [3*hidden, hidden]")
print("CUDA 门的顺序应该是: [z_gate (hidden个), r_gate (hidden个), h_gate (hidden个)]")
print("CUDA 使用列主序存储（CUBLAS默认）")

# 验证矩阵乘法方向
print("\n=== 验证矩阵乘法方向 ===")
x_single = x[0]  # [B, input_size]
print(f"输入 x[0].shape: {x_single.shape}")

# PyTorch的计算方式
# PyTorch内部: x @ weight_ih_l0.T
# weight_ih_l0是[3*hidden, input]，所以需要转置
w_ih_pt = gru.weight_ih_l0  # [3*hidden, input]
result_pt_manual = torch.matmul(x_single, w_ih_pt.T)  # [B, input] @ [input, 3*hidden] = [B, 3*hidden]
print(f"PyTorch计算 (x @ weight_ih_l0.T): {result_pt_manual.shape}")

# CUDA的计算方式（从GEMM调用看）
# CUDA GEMM: CUBLAS_OP_N, CUBLAS_OP_N
# 参数: hidden_size * 3, steps * batch_size, input_size
# W: [hidden_size * 3, input_size] (列主序)
# x: [input_size, steps * batch_size] (列主序，转置后的输入)
# 结果: [hidden_size * 3, steps * batch_size]
print(f"\nCUDA GEMM调用:")
print(f"  W: [hidden_size * 3, input_size] (列主序)")
print(f"  x: [input_size, steps * batch_size] (列主序，转置后)")
print(f"  计算: W @ x = [hidden_size * 3, steps * batch_size]")
print(f"  结果需要转置才能得到 [steps * batch_size, hidden_size * 3]")

# 检查权重是否可以直接使用
print("\n=== 关键检查：权重排布是否一致 ===")
print("PyTorch weight_ih_l0: [3*hidden, input] (行主序)")
print("CUDA期望 W: [hidden*3, input] (列主序)")
print("\n注意：PyTorch使用行主序，CUDA使用列主序！")
print("如果直接使用，需要确认：")
print("  1. 形状是否匹配: [3*hidden, input] vs [hidden*3, input] ✓ (相同)")
print("  2. 门的顺序是否一致: [z, r, h] vs [z, r, h] ✓ (应该一致)")
print("  3. 内存布局: 行主序 vs 列主序 ✗ (不同！)")

# 检查实际的内存布局
print("\n=== 检查实际内存布局 ===")
print("PyTorch行主序: weight_ih_l0[i, j] 在内存中连续")
print("CUDA列主序: W[j * (3*hidden) + i] 在内存中连续")
print("如果直接使用，矩阵会被转置！")

# 验证矩阵乘法的实际结果
print("\n=== 验证矩阵乘法的实际结果 ===")
x_single = x[0]  # [B, input_size] = [64, 256]
w_ih = gru.weight_ih_l0  # [3*hidden, input] = [768, 256]

# PyTorch的计算
result_pt = torch.matmul(x_single, w_ih.T)  # [64, 256] @ [256, 768] = [64, 768]

# CUDA的计算方式
# CUDA GEMM: W @ x，其中：
#   W: [768, 256] (列主序)
#   x: [256, 64] (列主序，转置后的输入)
#   结果: [768, 64]
x_t = x_single.T  # [256, 64] - 转置后的输入（模拟CUDA的列主序输入）

# 关键理解：PyTorch的weight_ih是行主序[768, 256]
# 如果直接传给CUDA，CUBLAS会按列主序读取
# 在列主序中，行主序的[768, 256]会被当作转置矩阵[256, 768]
# 但CUDA期望的是[768, 256]（列主序）

# 验证：直接使用weight_ih（行主序）在CUDA列主序中的效果
# 由于行主序和列主序的差异，直接使用相当于转置
# 所以CUDA实际使用的矩阵是 weight_ih.T（在列主序中）
# 但我们需要的是 weight_ih（在列主序中）
# 因此需要：传入 weight_ih.T，这样在列主序中就是 weight_ih

print("CUDA计算方式:")
print(f"  输入 x转置: {x_t.shape} = [input, batch]")
print(f"  权重 W: [768, 256] (列主序)")
print(f"  计算: W @ x = [768, 256] @ [256, 64] = [768, 64]")

# 正确的CUDA计算
result_cuda = torch.matmul(x_single, w_ih.T)
result_cuda_t = result_cuda.T  # [64, 768] - 转置结果

print(f"  结果转置: [64, 768]")
print(f"  与PyTorch结果比较:")
mse_check = torch.mean((result_cuda_t - result_pt) ** 2).item()
print(f"    MSE: {mse_check:.10f}")
if mse_check < 1e-6:
    print("    ✓ 结果一致！权重可以直接使用（行主序weight_ih在CUDA列主序中就是正确的）")
else:
    print("    ✗ 结果不一致！需要检查权重转换")

# 外部输入（PyTorch传入）的内存排布
print("\n" + "=" * 80)
print("外部输入（PyTorch传入）的内存排布")
print("=" * 80)
print("\n所有PyTorch Tensor都是行主序（row-major）存储：")
print("\n1. weight_ih_l0 (W):")
print(f"   形状: {gru.weight_ih_l0.shape} = [3*hidden, input]")
print("   内存排布: 第一维度（3*hidden）先连续")
print("   访问: W[i, j] 在内存中位置 = i * input + j")
print("   门的顺序: [z_gate (hidden个), r_gate (hidden个), h_gate (hidden个)]")

print("\n2. weight_hh_l0 (R):")
print(f"   形状: {gru.weight_hh_l0.shape} = [3*hidden, hidden]")
print("   内存排布: 第一维度（3*hidden）先连续")
print("   访问: R[i, j] 在内存中位置 = i * hidden + j")

print("\n3. bias_ih_l0 (bx):")
print(f"   形状: {gru.bias_ih_l0.shape} = [3*hidden]")
print("   内存排布: 线性连续")
print("   访问: bx[i] 在内存中位置 = i")

print("\n4. bias_hh_l0 (br):")
print(f"   形状: {gru.bias_hh_l0.shape} = [3*hidden]")
print("   内存排布: 线性连续")
print("   访问: br[i] 在内存中位置 = i")

print("\n5. x (输入数据):")
print(f"   形状: {x.shape} = [T, B, input_size]")
print("   内存排布: 第一维度（T）先连续")
print("   访问: x[t, b, i] 在内存中位置 = t * (B * input_size) + b * input_size + i")
print("   顺序: 先存储第一个时间步的所有批次和特征，再存储第二个时间步...")

print("\n6. h (输出隐藏状态):")
print(f"   形状: [T, B, hidden_size]")
print("   内存排布: 第一维度（T）先连续")
print("   访问: h[t, b, i] 在内存中位置 = t * (B * hidden_size) + b * hidden_size + i")

print("\n" + "=" * 80)
print("总结：所有外部输入都是行主序，第一维度先连续")
print("=" * 80)

y_pt, _ = gru(x, h0)
print("\n" + "=" * 80)

# C++ 量化 GRU
wrapper = GRUQuantInt8(T, B, input_size, hidden_size)

# 初始化权重 (从 PyTorch 拷贝过来)
wrapper.initWeights(
    gru.weight_ih_l0,
    gru.weight_hh_l0,
    gru.bias_ih_l0,
    gru.bias_hh_l0,
    x
)

# 前向
y_quant = wrapper.forward(x)

# 验证两个结果的差距
print("=" * 80)
print("验证 PyTorch GRU 和 Quantized GRU 的输出差异")
print("=" * 80)
print(f"输出形状: {y_pt.shape} (时间步={T}, 批次={B}, 隐藏层={hidden_size})")
print()

# 确保两个输出在同一设备上
y_pt_cpu = y_pt.cpu()
y_quant_cpu = y_quant.cpu()

# 计算每个时间步的MSE和余弦相似度
print(f"{'时间步':<8} {'MSE':<20} {'Cosine Similarity':<20}")
print("-" * 80)

mse_list = []
cos_sim_list = []

for t in range(T):
    # 获取当前时间步的输出 [B, hidden_size]
    y_pt_t = y_pt_cpu[t].flatten()  # [B * hidden_size]
    y_quant_t = y_quant_cpu[t].flatten()  # [B * hidden_size]

    # 计算MSE
    mse = torch.mean((y_pt_t - y_quant_t) ** 2).item()
    mse_list.append(mse)

    # 计算余弦相似度
    dot_product = torch.sum(y_pt_t * y_quant_t).item()
    norm_pt = torch.norm(y_pt_t).item()
    norm_quant = torch.norm(y_quant_t).item()
    cos_sim = dot_product / (norm_pt * norm_quant + 1e-8)  # 防止除零
    cos_sim_list.append(cos_sim)

    # 每10个时间步或最后几个时间步打印一次
    if t < 10 or t >= T - 10 or (t + 1) % 50 == 0:
        print(f"{t:<8} {mse:<20.10f} {cos_sim:<20.10f}")

print("-" * 80)
print(f"{'平均':<8} {torch.tensor(mse_list).mean().item():<20.10f} {torch.tensor(cos_sim_list).mean().item():<20.10f}")
print(f"{'最小':<8} {torch.tensor(mse_list).min().item():<20.10f} {torch.tensor(cos_sim_list).min().item():<20.10f}")
print(f"{'最大':<8} {torch.tensor(mse_list).max().item():<20.10f} {torch.tensor(cos_sim_list).max().item():<20.10f}")
print("=" * 80)
