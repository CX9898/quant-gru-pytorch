"""
自定义 GRU 类，继承自 PyTorch 的 nn.GRU
支持量化和非量化两种前向传播模式
"""

import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

try:
    import gru_interface_binding as gru_ops
except ImportError:
    raise ImportError(
        "gru_interface_binding module not found. "
        "Please compile the C++ extension first using setup.py"
    )


# ==================== 位宽配置工具函数 ====================


def _get_bitwidth_value(op_cfg: dict) -> int:
    """
    从操作配置中获取位宽值
    
    Python 端只关注位宽数量（8, 16, 32），不关心实际类型（INT/UINT）。
    实际类型由 C++ 端在 to_cpp() 时根据位宽数值决定。
    
    返回值:
        正整数表示位宽: 8, 16, 32
    """
    return op_cfg.get('bitwidth', 8)


def _get_symmetric_value(op_cfg: dict) -> bool:
    """
    从操作配置中获取是否使用对称量化
    
    Args:
        op_cfg: 操作配置字典
        
    Returns:
        True 表示对称量化，False 表示非对称量化
    """
    return op_cfg.get('is_symmetric', True)


def load_bitwidth_config(config_file: str) -> gru_ops.OperatorQuantConfig:
    """
    从 JSON 配置文件加载量化位宽配置（包括对称量化配置）
    
    Args:
        config_file: JSON 配置文件路径
        
    Returns:
        OperatorQuantConfig 对象
        
    JSON 格式示例:
    {
        "GRU_config": {
            "operator_config": {
                "input.x": { "bitwidth": 8, "is_symmetric": true },
                "gate.z_out": { "bitwidth": 8, "is_symmetric": false },
                ...
            }
        }
    }
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    config = gru_ops.OperatorQuantConfig()
    gru_config = data.get('GRU_config', {})
    op_config = gru_config.get('operator_config', {})

    # 字段映射: JSON key -> (位宽属性名, 对称量化属性名)
    field_map = {
        "input.x": ("x_", "x_symmetric_"),
        "input.h": ("h_", "h_symmetric_"),
        "weight.W": ("W_", "W_symmetric_"),
        "weight.R": ("R_", "R_symmetric_"),
        "weight.bx": ("bx_", "bx_symmetric_"),
        "weight.br": ("br_", "br_symmetric_"),
        "matmul.Wx": ("Wx_", "Wx_symmetric_"),
        "matmul.Rh": ("Rh_", "Rh_symmetric_"),
        "gate.z_pre": ("z_pre_", "z_pre_symmetric_"),
        "gate.z_out": ("z_out_", "z_out_symmetric_"),
        "gate.r_pre": ("r_pre_", "r_pre_symmetric_"),
        "gate.r_out": ("r_out_", "r_out_symmetric_"),
        "gate.g_pre": ("g_pre_", "g_pre_symmetric_"),
        "gate.g_out": ("g_out_", "g_out_symmetric_"),
        "op.Rh_add_br": ("Rh_add_br_", "Rh_add_br_symmetric_"),
        "op.rRh": ("rRh_", "rRh_symmetric_"),
        "op.old_contrib": ("old_contrib_", "old_contrib_symmetric_"),
        "op.new_contrib": ("new_contrib_", "new_contrib_symmetric_"),
    }

    for json_key, (bw_attr, sym_attr) in field_map.items():
        if json_key in op_config:
            op_cfg = op_config[json_key]
            # 设置位宽
            bw_val = _get_bitwidth_value(op_cfg)
            setattr(config, bw_attr, bw_val)
            # 设置对称量化配置
            sym_val = _get_symmetric_value(op_cfg)
            setattr(config, sym_attr, sym_val)

    return config


def _format_bitwidth(val: int) -> str:
    """格式化位宽值为可读字符串"""
    # Python 端只显示位宽数量
    return f"{abs(val)}bit"


def _format_symmetric(is_symmetric: bool) -> str:
    """格式化对称量化值为可读字符串"""
    return "对称" if is_symmetric else "非对称"


def apply_bitwidth_config(config: gru_ops.OperatorQuantConfig,
                          config_file: str,
                          verbose: bool = False) -> int:
    """
    从 JSON 配置文件应用量化位宽配置（包括对称量化配置）
    
    Args:
        config: 要更新的 OperatorQuantConfig 对象
        config_file: JSON 配置文件路径
        verbose: 是否打印详细信息
        
    Returns:
        成功配置的字段数量
    """
    loaded = load_bitwidth_config(config_file)

    # 复制位宽配置字段
    bitwidth_attrs = ['x_', 'h_', 'W_', 'R_', 'bx_', 'br_', 'Wx_', 'Rh_',
                      'z_pre_', 'z_out_', 'r_pre_', 'r_out_', 'g_pre_', 'g_out_',
                      'Rh_add_br_', 'rRh_', 'old_contrib_', 'new_contrib_']
    for attr in bitwidth_attrs:
        setattr(config, attr, getattr(loaded, attr))

    # 复制对称量化配置字段
    symmetric_attrs = ['x_symmetric_', 'h_symmetric_', 'W_symmetric_', 'R_symmetric_',
                       'bx_symmetric_', 'br_symmetric_', 'Wx_symmetric_', 'Rh_symmetric_',
                       'z_pre_symmetric_', 'z_out_symmetric_', 'r_pre_symmetric_', 'r_out_symmetric_',
                       'g_pre_symmetric_', 'g_out_symmetric_', 'Rh_add_br_symmetric_', 'rRh_symmetric_',
                       'old_contrib_symmetric_', 'new_contrib_symmetric_']
    for attr in symmetric_attrs:
        setattr(config, attr, getattr(loaded, attr))

    if verbose:
        print("\n" + "=" * 70)
        print("🔧 应用 GRU 量化配置（位宽 + 对称量化）")
        print("=" * 70)
        print(f"📄 配置文件: {config_file}")
        print("-" * 70)
        print(f"  [输入]  x: {_format_bitwidth(config.x_):6s} ({_format_symmetric(config.x_symmetric_)})")
        print(f"          h: {_format_bitwidth(config.h_):6s} ({_format_symmetric(config.h_symmetric_)})")
        print(f"  [权重]  W: {_format_bitwidth(config.W_):6s} ({_format_symmetric(config.W_symmetric_)})")
        print(f"          R: {_format_bitwidth(config.R_):6s} ({_format_symmetric(config.R_symmetric_)})")
        print(f"          bx: {_format_bitwidth(config.bx_):6s} ({_format_symmetric(config.bx_symmetric_)})")
        print(f"          br: {_format_bitwidth(config.br_):6s} ({_format_symmetric(config.br_symmetric_)})")
        print(f"  [矩阵]  Wx: {_format_bitwidth(config.Wx_):6s} ({_format_symmetric(config.Wx_symmetric_)})")
        print(f"          Rh: {_format_bitwidth(config.Rh_):6s} ({_format_symmetric(config.Rh_symmetric_)})")
        print(f"  [门控]  z_pre: {_format_bitwidth(config.z_pre_):6s} ({_format_symmetric(config.z_pre_symmetric_)})")
        print(f"          z_out: {_format_bitwidth(config.z_out_):6s} ({_format_symmetric(config.z_out_symmetric_)})")
        print(f"          r_pre: {_format_bitwidth(config.r_pre_):6s} ({_format_symmetric(config.r_pre_symmetric_)})")
        print(f"          r_out: {_format_bitwidth(config.r_out_):6s} ({_format_symmetric(config.r_out_symmetric_)})")
        print(f"          g_pre: {_format_bitwidth(config.g_pre_):6s} ({_format_symmetric(config.g_pre_symmetric_)})")
        print(f"          g_out: {_format_bitwidth(config.g_out_):6s} ({_format_symmetric(config.g_out_symmetric_)})")
        print(
            f"  [运算]  Rh+br: {_format_bitwidth(config.Rh_add_br_):6s} ({_format_symmetric(config.Rh_add_br_symmetric_)})")
        print(f"          rRh: {_format_bitwidth(config.rRh_):6s} ({_format_symmetric(config.rRh_symmetric_)})")
        print(
            f"  [输出]  old: {_format_bitwidth(config.old_contrib_):6s} ({_format_symmetric(config.old_contrib_symmetric_)})")
        print(
            f"          new: {_format_bitwidth(config.new_contrib_):6s} ({_format_symmetric(config.new_contrib_symmetric_)})")
        print("=" * 70 + "\n")

    return 38  # 19 位宽字段 + 19 对称量化字段


# ==================== 工具函数：权重格式转换 ====================

def reorder_weights_pytorch_to_haste(w: torch.Tensor) -> torch.Tensor:
    """
    将 PyTorch GRU 权重格式 (r, z, n) 转换为 Haste GRU 权重格式 (z, r, n)

    Args:
        w: 权重张量，第一维是 3*hidden_size，顺序为 r, z, n
           - 权重矩阵：形状为 [3*hidden, input] 或 [3*hidden, hidden]
           - 偏置向量：形状为 [3*hidden]

    Returns:
        重排序后的权重张量，顺序为 z, r, n，形状保持不变
    """
    w = w.contiguous()
    hidden_size_3 = w.shape[0] // 3
    device = w.device

    # PyTorch: [r0...rH, z0...zH, n0...nH] -> Haste: [z0...zH, r0...rH, n0...nH]
    indices = torch.cat([
        torch.arange(hidden_size_3, 2 * hidden_size_3, device=device),  # z
        torch.arange(0, hidden_size_3, device=device),  # r
        torch.arange(2 * hidden_size_3, 3 * hidden_size_3, device=device)  # n
    ])

    return w.index_select(0, indices).contiguous()


def reorder_weights_haste_to_pytorch(w: torch.Tensor) -> torch.Tensor:
    """
    将 Haste GRU 权重格式 (z, r, n) 转换回 PyTorch GRU 权重格式 (r, z, n)

    Args:
        w: 权重张量，第一维是 3*hidden_size，顺序为 z, r, n

    Returns:
        重排序后的权重张量，顺序为 r, z, n，形状保持不变
    """
    w = w.contiguous()
    hidden_size_3 = w.shape[0] // 3
    device = w.device

    # Haste: [z0...zH, r0...rH, n0...nH] -> PyTorch: [r0...rH, z0...zH, n0...nH]
    indices = torch.cat([
        torch.arange(hidden_size_3, 2 * hidden_size_3, device=device),  # r (在 Haste 中是第二部分)
        torch.arange(0, hidden_size_3, device=device),  # z (在 Haste 中是第一部分)
        torch.arange(2 * hidden_size_3, 3 * hidden_size_3, device=device)  # n (在 Haste 中是第三部分)
    ])

    return w.index_select(0, indices).contiguous()


def ensure_cuda_float32(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    确保张量在指定设备上且为 float32 类型（保持梯度追踪）

    Args:
        tensor: 输入张量
        device: 目标设备

    Returns:
        转换后的张量
    """
    if not tensor.is_cuda:
        tensor = tensor.to(device)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


# ==================== GRUFunction：自定义 autograd Function ====================

class GRUFunction(torch.autograd.Function):
    """
    GRU 的自定义 autograd Function，支持反向传播

    职责：
    - 处理 PyTorch 和 Haste 格式之间的转换
    - 调用 C++ 接口进行前向和反向传播
    - 管理中间结果的保存和恢复
    """

    @staticmethod
    def forward(ctx, input, weight_ih, weight_hh, bias_ih, bias_hh, h0, is_training,
                use_quantization=False, quant_params=None):
        """
        前向传播

        Args:
            ctx: 上下文对象
            input: 输入序列 [time_steps, batch_size, input_size]
            weight_ih: 输入权重 [3*hidden_size, input_size] (PyTorch 格式: r, z, n)
            weight_hh: 循环权重 [3*hidden_size, hidden_size] (PyTorch 格式: r, z, n)
            bias_ih: 输入偏置 [3*hidden_size] (PyTorch 格式: r, z, n) 或 None
            bias_hh: 循环偏置 [3*hidden_size] (PyTorch 格式: r, z, n) 或 None
            h0: 初始隐藏状态 [batch_size, hidden_size] 或 None
            is_training: 是否处于训练模式
            use_quantization: 是否使用量化
            quant_params: 量化参数（包含位宽配置）

        Returns:
            output: 输出序列 [time_steps, batch_size, hidden_size]
            h_n: 最终隐藏状态 [1, batch_size, hidden_size]
        """
        time_steps, batch_size, input_size = input.shape
        hidden_size = weight_hh.shape[1]

        # 保存上下文信息
        ctx.time_steps = time_steps
        ctx.batch_size = batch_size
        ctx.input_size = input_size
        ctx.hidden_size = hidden_size
        ctx.bias_ih_is_none = (bias_ih is None)
        ctx.bias_hh_is_none = (bias_hh is None)
        ctx.h0_is_none = (h0 is None)

        # 确保输入在 CUDA 上
        device = input.device if input.is_cuda else torch.device('cuda')
        input = ensure_cuda_float32(input, device)

        # 转换权重格式：PyTorch (r, z, n) -> Haste (z, r, n)
        # 权重矩阵需要转置：[3*hidden, input] -> [input, 3*hidden]
        weight_ih = ensure_cuda_float32(weight_ih, device)
        weight_hh = ensure_cuda_float32(weight_hh, device)
        W = reorder_weights_pytorch_to_haste(weight_ih).t().contiguous()
        R = reorder_weights_pytorch_to_haste(weight_hh).t().contiguous()

        # 处理偏置
        if bias_ih is not None and bias_hh is not None:
            bias_ih = ensure_cuda_float32(bias_ih, device)
            bias_hh = ensure_cuda_float32(bias_hh, device)
            bx = reorder_weights_pytorch_to_haste(bias_ih).contiguous()
            br = reorder_weights_pytorch_to_haste(bias_hh).contiguous()
        else:
            bx = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)
            br = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)

        # 准备 h0
        if h0 is not None:
            h0_tensor = ensure_cuda_float32(h0, device)
        else:
            h0_tensor = torch.empty(0, device=device, dtype=torch.float32)

        # 准备量化参数
        if use_quantization:
            if quant_params is None:
                raise RuntimeError("quant_params is required when use_quantization=True")
        else:
            quant_params = gru_ops.GRUQuantitativeParameters()

        # 调用 C++ 接口
        output_full, v = gru_ops.forward_interface(
            is_training=is_training,
            is_quant=use_quantization,
            time_steps=time_steps,
            batch_size=batch_size,
            input_size=input_size,
            hidden_size=hidden_size,
            W=W,
            R=R,
            bx=bx,
            br=br,
            x=input,
            h0=h0_tensor,
            quant_params=quant_params
        )

        # # 浮点前向
        # output_full_no_quant, v_no_quant = gru_ops.haste_gru_forward(
        #     is_training=is_training,
        #     time_steps=time_steps,
        #     batch_size=batch_size,
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     W=W,
        #     R=R,
        #     bx=bx,
        #     br=br,
        #     x=input,
        #     h0=h0_tensor,
        # )

        # 分离输出：output_full[0] 是初始状态，output_full[1:] 是时间步输出
        output = output_full[1:]  # [time_steps, batch_size, hidden_size]
        h_n = output_full[-1:]  # [1, batch_size, hidden_size]

        # 保存中间结果用于反向传播
        ctx.save_for_backward(W, R, bx, br, input, output_full, v)

        return output, h_n

    @staticmethod
    def backward(ctx, grad_output, grad_h_n):
        """
        反向传播

        Args:
            ctx: 上下文对象
            grad_output: 输出序列的梯度 [time_steps, batch_size, hidden_size]
            grad_h_n: 最终隐藏状态的梯度 [1, batch_size, hidden_size]

        Returns:
            各输入参数的梯度
        """
        W, R, bx, br, input, h, v = ctx.saved_tensors
        time_steps = ctx.time_steps
        batch_size = ctx.batch_size
        input_size = ctx.input_size
        hidden_size = ctx.hidden_size

        # 确保所有数据在 CUDA 上
        device = grad_output.device
        if not W.is_cuda:
            W = W.to(device)
        if not R.is_cuda:
            R = R.to(device)
        if not bx.is_cuda:
            bx = bx.to(device)
        if not br.is_cuda:
            br = br.to(device)
        if not input.is_cuda:
            input = input.to(device)
        if not h.is_cuda:
            h = h.to(device)
        if v is not None and not v.is_cuda:
            v = v.to(device)
        if not grad_output.is_cuda:
            grad_output = grad_output.to(device)
        if grad_h_n is not None and not grad_h_n.is_cuda:
            grad_h_n = grad_h_n.to(device)

        # 构建隐藏状态梯度
        # C++ 接口需要 [time_steps + 1, batch_size, hidden_size] 格式
        # dh_new[0] 是初始状态梯度（保持为 0），dh_new[1:] 是时间步梯度
        dh_new = torch.zeros(
            (time_steps + 1, batch_size, hidden_size),
            device=device,
            dtype=grad_output.dtype
        )
        dh_new[1:] = grad_output

        # 处理最终隐藏状态的梯度（output[-1] 和 h_n[0] 指向同一个状态）
        if grad_h_n is not None and grad_h_n.numel() > 0:
            dh_new[-1] = dh_new[-1] + grad_h_n[0]

        # 调用 C++ 反向传播接口
        # Python 绑定层会内部处理转置，使其与 haste 的实现一致：
        # - x: [T,B,I] -> x_t: [I,T,B]
        # - W: [C,H*3] -> W_t: [H*3,C]
        # - R: [H,H*3] -> R_t: [H*3,H]
        dx, dW, dR, dbx, dbr, dh = gru_ops.haste_gru_backward(
            time_steps=time_steps,
            batch_size=batch_size,
            input_size=input_size,
            hidden_size=hidden_size,
            W=W,  # [C, H*3] - Python 绑定层会转置为 [H*3, C]
            R=R,  # [H, H*3] - Python 绑定层会转置为 [H*3, H]
            bx=bx,
            br=br,
            x=input,  # [T, B, I] - Python 绑定层会转置为 [I, T, B]
            dh_new=dh_new,
            h=h,
            v=v
        )

        # 转换梯度格式：Haste (z, r, n) -> PyTorch (r, z, n)
        # 梯度矩阵需要转置：[input, 3*hidden] -> [3*hidden, input]
        dW_pytorch = reorder_weights_haste_to_pytorch(dW.t()).contiguous()
        dR_pytorch = reorder_weights_haste_to_pytorch(dR.t()).contiguous()
        dbx_pytorch = reorder_weights_haste_to_pytorch(dbx).contiguous()
        dbr_pytorch = reorder_weights_haste_to_pytorch(dbr).contiguous()

        # 处理偏置梯度
        if ctx.bias_ih_is_none:
            dbx_pytorch = None
        if ctx.bias_hh_is_none:
            dbr_pytorch = None

        # 处理 h0 梯度
        grad_h0 = None if ctx.h0_is_none else dh

        # 返回梯度（对应 forward 的 9 个参数）
        return dx, dW_pytorch, dR_pytorch, dbx_pytorch, dbr_pytorch, grad_h0, None, None, None


# ==================== CustomGRU：自定义 GRU 类 ====================

class CustomGRU(nn.Module):
    """
    自定义 GRU 实现，支持量化前向传播和双向 GRU
    
    设计原则：
        - 延迟初始化：CUDA handle 在首次 forward/calibrate 时初始化，而非构造时
        - 配置与创建分离：位宽配置通过 load_bitwidth_config() 单独加载
        - 校准与创建分离：校准通过 calibrate() + finalize_calibration() 单独执行
        - 可序列化：使用 Python 字典存储配置，支持 pickle/deepcopy
        - 双向支持：内部使用两个单向 GRU 模拟双向 GRU，对外接口与 nn.GRU 一致

    量化使用流程：
        1. 创建模型：gru = CustomGRU(..., use_quantization=True)
        2. (可选) 加载位宽配置：gru.load_bitwidth_config("config.json")
        3. 累积校准数据：gru.calibrate(data1), gru.calibrate(data2), ...
        4. 完成校准：gru.finalize_calibration()
        5. 正常推理：output, h_n = gru(input)
        
    增量校准（支持中途重新校准）：
        - 可随时调用 calibrate() 累积更多数据
        - 在下次 forward() 前调用 finalize_calibration() 更新量化参数
        - 如需完全重置范围：gru.reset_calibration()

    内部状态：
        - _cublas_initialized: CUDA handle 是否已初始化
        - _bitwidth_config_dict: 位宽配置（Python 字典，可序列化）
        - quant_ranges / quant_ranges_reverse: 校准范围（C++ 对象，calibrate() 时创建）
        - quant_params / quant_params_reverse: 量化参数（C++ 对象，finalize_calibration() 时创建）

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: GRU 层数（目前仅支持 1）
        bias: 是否使用偏置
        batch_first: 如果为 True，输入形状为 [batch, seq, feature]
        dropout: 层间 dropout 概率（目前不支持）
        bidirectional: 是否双向 GRU（True 时输出维度为 2*hidden_size）

    Attributes:
        use_quantization: 是否启用量化（默认 False，可随时修改）

    Examples:
        >>> # 基本使用（非量化，单向）
        >>> gru = CustomGRU(64, 128, batch_first=True)
        >>> output, h_n = gru(input_data)
        
        >>> # 双向 GRU（与 nn.GRU 接口一致）
        >>> gru = CustomGRU(64, 128, batch_first=True, bidirectional=True)
        >>> output, h_n = gru(input_data)  # output: [B, T, 2*H], h_n: [2, B, H]
        
        >>> # 量化使用（先校准，再开启量化）
        >>> gru = CustomGRU(64, 128)
        >>> gru.load_bitwidth_config("config.json")  # 可选
        >>> for batch in calibration_loader:
        ...     gru.calibrate(batch)  # 校准时无需开启量化
        >>> gru.finalize_calibration()
        >>> gru.use_quantization = True  # 推理时开启量化
        >>> output, h_n = gru(input_data)
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            bidirectional: bool = False,
            use_quantization: bool = False,
    ):
        """
        初始化 CustomGRU
        
        设计原则：
            - __init__ 只做最基本的属性初始化
            - 复杂操作（CUDA 初始化、校准等）延迟到需要时执行
            - 位宽配置通过 load_bitwidth_config() 单独加载
            - 双向 GRU 内部使用两套权重，分别处理正向和反向
            - 量化开关 use_quantization 可随时修改，仅影响 forward
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            num_layers: GRU 层数（目前仅支持 1）
            bias: 是否使用偏置
            batch_first: 输入格式是否为 [batch, seq, feature]
            dropout: dropout 概率（目前不支持）
            bidirectional: 是否双向 GRU
        
        量化使用流程：
            1. 创建模型: gru = CustomGRU(...)
            2. (可选) 加载位宽配置: gru.load_bitwidth_config("config.json")
            3. 累积校准: gru.calibrate(data1), gru.calibrate(data2), ...
            4. 完成校准: gru.finalize_calibration()
            5. 开启量化: gru.use_quantization = True
            6. 正常推理: output, h_n = gru(input)
        """
        super(CustomGRU, self).__init__()

        # 检查限制
        if num_layers != 1:
            raise NotImplementedError("Currently only supports num_layers=1")
        if dropout > 0:
            raise NotImplementedError("Currently does not support dropout")

        # ===== 基本配置 =====
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_quantization = use_quantization  # 默认关闭量化，校准完成后可设置为 True
        self.num_directions = 2 if bidirectional else 1

        # ===== 权重参数（与 nn.GRU 命名一致） =====
        # 前向方向权重
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter('bias_ih_l0', None)
            self.register_parameter('bias_hh_l0', None)

        # 反向方向权重（仅双向时使用）
        if bidirectional:
            self.weight_ih_l0_reverse = nn.Parameter(torch.empty(3 * hidden_size, input_size))
            self.weight_hh_l0_reverse = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            if bias:
                self.bias_ih_l0_reverse = nn.Parameter(torch.empty(3 * hidden_size))
                self.bias_hh_l0_reverse = nn.Parameter(torch.empty(3 * hidden_size))
            else:
                self.register_parameter('bias_ih_l0_reverse', None)
                self.register_parameter('bias_hh_l0_reverse', None)

        # 初始化权重
        self._init_weights()

        # ===== 量化状态（初始化为 None，延迟创建） =====
        self.quant_ranges = None  # 前向 C++ 对象，calibrate() 时创建
        self.quant_params = None  # 前向 C++ 对象，finalize_calibration() 时创建
        if bidirectional:
            self.quant_ranges_reverse = None  # 反向 C++ 对象
            self.quant_params_reverse = None  # 反向 C++ 对象

        # ===== 位宽配置（延迟初始化，使用 Python 字典以支持序列化） =====
        self._bitwidth_config_dict = None  # 延迟初始化，首次访问时创建默认配置

        # ===== CUDA 初始化标志（延迟初始化） =====
        self._cublas_initialized = False

    def _init_weights(self):
        """初始化权重，使用与 nn.GRU 相同的初始化策略"""
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    # -------------------- CUDA 延迟初始化 --------------------

    def _ensure_cublas_initialized(self):
        """
        确保 cublas handle 已初始化（延迟初始化模式）
        只在第一次需要时初始化，避免在 __init__ 中过早初始化
        """
        if not self._cublas_initialized:
            gru_ops.init_gru_cublas()
            self._cublas_initialized = True

    # -------------------- 位宽配置内部方法 --------------------

    def _load_bitwidth_config_to_dict(self, config_file: str):
        """从 JSON 文件加载配置到内部字典"""
        # 初始化字典（只存储用户指定的配置）
        if self._bitwidth_config_dict is None:
            self._bitwidth_config_dict = {}

        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 读取 GRU_config 节点下的配置
        gru_config = data.get('GRU_config', {})

        # 读取全局配置
        default_config = gru_config.get('default_config', {})
        if 'disable_quantization' in default_config:
            # disable_quantization=true 表示禁用量化，所以 use_quantization 取反
            self.use_quantization = not default_config['disable_quantization']

        op_config = gru_config.get('operator_config', {})

        # 字段映射: JSON key -> (位宽属性名, 对称量化属性名)
        field_map = {
            "input.x": ("x_", "x_symmetric_"),
            "input.h": ("h_", "h_symmetric_"),
            "weight.W": ("W_", "W_symmetric_"),
            "weight.R": ("R_", "R_symmetric_"),
            "weight.bx": ("bx_", "bx_symmetric_"),
            "weight.br": ("br_", "br_symmetric_"),
            "matmul.Wx": ("Wx_", "Wx_symmetric_"),
            "matmul.Rh": ("Rh_", "Rh_symmetric_"),
            "gate.z_pre": ("z_pre_", "z_pre_symmetric_"),
            "gate.z_out": ("z_out_", "z_out_symmetric_"),
            "gate.r_pre": ("r_pre_", "r_pre_symmetric_"),
            "gate.r_out": ("r_out_", "r_out_symmetric_"),
            "gate.g_pre": ("g_pre_", "g_pre_symmetric_"),
            "gate.g_out": ("g_out_", "g_out_symmetric_"),
            "op.Rh_add_br": ("Rh_add_br_", "Rh_add_br_symmetric_"),
            "op.rRh": ("rRh_", "rRh_symmetric_"),
            "op.old_contrib": ("old_contrib_", "old_contrib_symmetric_"),
            "op.new_contrib": ("new_contrib_", "new_contrib_symmetric_"),
        }

        for json_key, (bw_attr, sym_attr) in field_map.items():
            if json_key in op_config:
                op_cfg = op_config[json_key]
                self._bitwidth_config_dict[bw_attr] = op_cfg.get('bitwidth', 8)
                self._bitwidth_config_dict[sym_attr] = op_cfg.get('is_symmetric', True)

    def _get_cpp_bitwidth_config(self) -> gru_ops.OperatorQuantConfig:
        """
        获取 C++ OperatorQuantConfig 对象
        
        如果用户未加载自定义配置，返回默认的 C++ 对象（C++ 端使用默认值）
        如果用户已加载配置，从 Python 字典创建 C++ 对象
        """
        config = gru_ops.OperatorQuantConfig()

        # 只有用户加载了自定义配置时，才覆盖 C++ 默认值
        if self._bitwidth_config_dict is not None:
            for attr, value in self._bitwidth_config_dict.items():
                setattr(config, attr, value)

        return config

    # -------------------- 位宽配置公开接口 --------------------

    def load_bitwidth_config(self, config_file: str, verbose: bool = False):
        """
        从 JSON 配置文件加载量化位宽配置
        
        Args:
            config_file: JSON 配置文件路径
            verbose: 是否打印详细信息
            
        使用示例:
            gru.load_bitwidth_config("config/gru_quant_bitwidth_config.json", verbose=True)
        """
        self._load_bitwidth_config_to_dict(config_file)
        if verbose:
            cpp_config = self._get_cpp_bitwidth_config()
            apply_bitwidth_config(cpp_config, config_file, verbose=True)
            print(f"  [全局]  use_quantization: {self.use_quantization}")

    # -------------------- 校准状态查询 --------------------

    def is_calibrated(self) -> bool:
        """
        检查量化是否已完成校准

        Returns:
            True 如果已调用 finalize_calibration()，否则 False
            对于双向 GRU，需要正向和反向都已校准
        """
        if self.bidirectional:
            return self.quant_params is not None and self.quant_params_reverse is not None
        return self.quant_params is not None

    # -------------------- 公共校准接口 --------------------

    def calibrate(self, calibration_data: torch.Tensor):
        """
        累积校准数据，更新量化范围

        可随时调用，每次调用会将新数据的范围与已有范围合并（取并集）。
        完成数据收集后，需调用 finalize_calibration() 计算量化参数。

        Args:
            calibration_data: 校准数据，形状为 [seq_len, batch, input_size]
                             （如果 batch_first=True，则为 [batch, seq_len, input_size]）

        Note:
            - 校准时无需开启 use_quantization，校准与量化开关解耦
            - 支持增量校准：即使已调用过 finalize_calibration()，仍可继续调用
              calibrate() 累积更多数据，然后再次调用 finalize_calibration()
            - 校准完成后，通过设置 use_quantization = True 开启量化推理
        """
        self._accumulate_calibration_ranges(calibration_data)

    def finalize_calibration(self):
        """
        完成校准，计算量化参数并初始化 LUT 表

        根据累积的量化范围和位宽配置计算各算子的 scale 和 zero_point。
        可多次调用，每次会根据当前累积的范围重新计算量化参数。

        Raises:
            RuntimeError: 未调用过 calibrate()

        Note:
            支持增量校准流程：
                calibrate(data1) -> finalize_calibration() -> forward() ->
                calibrate(data2) -> finalize_calibration() -> forward() -> ...
            
            如果需要自定义位宽配置，请在调用此方法前先调用 load_bitwidth_config()。
            如需完全重置范围，请调用 reset_calibration()。
            对于双向 GRU，会为正向和反向分别计算量化参数。
        """
        if self.quant_ranges is None:
            raise RuntimeError(
                "No calibration data accumulated. "
                "Call calibrate(data) at least once before finalize_calibration()."
            )

        # ===== 前向方向：计算量化参数 =====
        if self._bitwidth_config_dict is not None:
            self.quant_params = gru_ops.calculate_gru_quantitative_parameters(
                quant_ranges=self.quant_ranges,
                bitwidth_config=self._get_cpp_bitwidth_config()
            )
        else:
            self.quant_params = gru_ops.calculate_gru_quantitative_parameters(
                quant_ranges=self.quant_ranges
            )

        # 初始化查找表（前向）
        gru_ops.initialize_quantization_lut(quant_params=self.quant_params)

        # ===== 反向方向：计算量化参数（仅双向时） =====
        if self.bidirectional:
            if self.quant_ranges_reverse is None:
                raise RuntimeError(
                    "No reverse calibration data accumulated. "
                    "This should not happen for bidirectional GRU."
                )

            if self._bitwidth_config_dict is not None:
                self.quant_params_reverse = gru_ops.calculate_gru_quantitative_parameters(
                    quant_ranges=self.quant_ranges_reverse,
                    bitwidth_config=self._get_cpp_bitwidth_config()
                )
            else:
                self.quant_params_reverse = gru_ops.calculate_gru_quantitative_parameters(
                    quant_ranges=self.quant_ranges_reverse
                )

            # 初始化查找表（反向）
            gru_ops.initialize_quantization_lut(quant_params=self.quant_params_reverse)

    def reset_calibration(self):
        """
        重置校准状态

        清除累积的量化范围和量化参数，允许重新开始校准流程。
        对于双向 GRU，会同时重置正向和反向的状态。
        """
        self.quant_ranges = None
        self.quant_params = None
        if self.bidirectional:
            self.quant_ranges_reverse = None
            self.quant_params_reverse = None

    # -------------------- 调试与打印 --------------------

    def print_quant_params(self):
        """
        打印量化参数

        Raises:
            RuntimeError: 未调用过 finalize_calibration()
        """
        if not self.is_calibrated():
            raise RuntimeError(
                "Quantization parameters not available. "
                "Call finalize_calibration() first."
            )

        params = self.quant_params
        print("=" * 60)
        print("GRUQuantitativeParameters (量化参数)")
        print("=" * 60)
        print(f"  hidden_ = {params.hidden_}")
        print(f"  [x]  exp2_inv={params.exp2_inv_x_:3d}, zp={params.zp_x_}")
        print(f"  [h]  exp2_inv={params.exp2_inv_h_:3d}, zp={params.zp_h_}")
        print(f"  [Wx] exp2_inv={params.exp2_inv_Wx_:3d}, zp={params.zp_Wx_}")
        print(f"  [Rh] exp2_inv={params.exp2_inv_Rh_:3d}, zp={params.zp_Rh_}")
        print("-" * 60)
        print(f"  [z_pre] exp2_inv={params.exp2_inv_z_pre_:3d}, zp={params.zp_z_pre_}")
        print(f"  [r_pre] exp2_inv={params.exp2_inv_r_pre_:3d}, zp={params.zp_r_pre_}")
        print(f"  [g_pre] exp2_inv={params.exp2_inv_g_pre_:3d}, zp={params.zp_g_pre_}")
        print(f"  [z_out] exp2_inv={params.exp2_inv_z_out_:3d}, zp={params.zp_z_out_}")
        print(f"  [r_out] exp2_inv={params.exp2_inv_r_out_:3d}, zp={params.zp_r_out_}")
        print(f"  [g_out] exp2_inv={params.exp2_inv_g_out_:3d}, zp={params.zp_g_out_}")
        print("-" * 60)
        print(f"  [Rh_add_br_g]        exp2_inv={params.exp2_inv_Rh_add_br_:3d}, zp={params.zp_Rh_add_br_}")
        print(f"  [rRh]              exp2_inv={params.exp2_inv_rRh_:3d}, zp={params.zp_rRh_}")
        print(f"  [new_contrib]      exp2_inv={params.exp2_inv_new_contrib_:3d}, zp={params.zp_new_contrib_}")
        print(f"  [old_contrib]      exp2_inv={params.exp2_inv_old_contrib_:3d}, zp={params.zp_old_contrib_}")
        print("-" * 60)
        if params.exp2_inv_W_:
            print(f"  [W] exp2_inv (first 5): {list(params.exp2_inv_W_[:5])} ...")
        if params.exp2_inv_R_:
            print(f"  [R] exp2_inv (first 5): {list(params.exp2_inv_R_[:5])} ...")
        if params.exp2_inv_bx_:
            print(f"  [bx] exp2_inv (first 5): {list(params.exp2_inv_bx_[:5])} ...")
        if params.exp2_inv_br_:
            print(f"  [br] exp2_inv (first 5): {list(params.exp2_inv_br_[:5])} ...")
        print("=" * 60)

    def print_quant_ranges(self):
        """
        打印量化范围

        Raises:
            RuntimeError: 未调用过 calibrate()
        """
        if self.quant_ranges is None:
            raise RuntimeError(
                "No calibration data accumulated. "
                "Call calibrate(data) first."
            )

        r = self.quant_ranges
        print("=" * 60)
        print("GRUQuantizationRanges (量化范围)")
        print("=" * 60)
        print(f"  hidden_ = {r.hidden_}")
        print(f"  [x]  min={r.min_x_:12.6f}, max={r.max_x_:12.6f}")
        print(f"  [h]  min={r.min_h_:12.6f}, max={r.max_h_:12.6f}")
        print(f"  [Wx] min={r.min_Wx_:12.6f}, max={r.max_Wx_:12.6f}")
        print(f"  [Rh] min={r.min_Rh_:12.6f}, max={r.max_Rh_:12.6f}")
        print("-" * 60)
        print(f"  [z_pre] min={r.min_z_pre_:12.6f}, max={r.max_z_pre_:12.6f}")
        print(f"  [r_pre] min={r.min_r_pre_:12.6f}, max={r.max_r_pre_:12.6f}")
        print(f"  [g_pre] min={r.min_g_pre_:12.6f}, max={r.max_g_pre_:12.6f}")
        print(f"  [z_out] min={r.min_z_out_:12.6f}, max={r.max_z_out_:12.6f}")
        print(f"  [r_out] min={r.min_r_out_:12.6f}, max={r.max_r_out_:12.6f}")
        print(f"  [g_out] min={r.min_g_out_:12.6f}, max={r.max_g_out_:12.6f}")
        print("-" * 60)
        print(f"  [Rh_add_br_g]        min={r.min_Rh_add_br_g_:12.6f}, max={r.max_Rh_add_br_g_:12.6f}")
        print(f"  [rRh]              min={r.min_rRh_:12.6f}, max={r.max_rRh_:12.6f}")
        print(f"  [new_contrib]      min={r.min_new_contrib_:12.6f}, max={r.max_new_contrib_:12.6f}")
        print(f"  [old_contrib]      min={r.min_old_contrib_:12.6f}, max={r.max_old_contrib_:12.6f}")
        print("=" * 60)

    # -------------------- 内部方法 --------------------

    def _convert_weights_to_haste_format(self, device: torch.device, reverse: bool = False):
        """
        将 PyTorch 格式的权重转换为 Haste 格式（用于量化校准）

        Args:
            device: 目标设备
            reverse: 是否获取反向方向的权重（仅双向 GRU 时有效）

        Returns:
            W, R, bx, br: Haste 格式的权重和偏置
        """
        if reverse and self.bidirectional:
            weight_ih = ensure_cuda_float32(self.weight_ih_l0_reverse, device)
            weight_hh = ensure_cuda_float32(self.weight_hh_l0_reverse, device)
        else:
            weight_ih = ensure_cuda_float32(self.weight_ih_l0, device)
            weight_hh = ensure_cuda_float32(self.weight_hh_l0, device)

        W = reorder_weights_pytorch_to_haste(weight_ih).t().contiguous()
        R = reorder_weights_pytorch_to_haste(weight_hh).t().contiguous()

        if self.bias:
            if reverse and self.bidirectional:
                bias_ih = ensure_cuda_float32(self.bias_ih_l0_reverse, device)
                bias_hh = ensure_cuda_float32(self.bias_hh_l0_reverse, device)
            else:
                bias_ih = ensure_cuda_float32(self.bias_ih_l0, device)
                bias_hh = ensure_cuda_float32(self.bias_hh_l0, device)
            bx = reorder_weights_pytorch_to_haste(bias_ih).contiguous()
            br = reorder_weights_pytorch_to_haste(bias_hh).contiguous()
        else:
            hidden_size = self.hidden_size
            bx = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)
            br = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)

        return W, R, bx, br

    def _accumulate_calibration_ranges(self, calibration_data: torch.Tensor):
        """累积校准范围（内部方法）"""
        # 延迟初始化 cublas
        self._ensure_cublas_initialized()

        # 确保校准数据在 CUDA 上
        device = calibration_data.device if calibration_data.is_cuda else torch.device('cuda')
        if not calibration_data.is_cuda:
            calibration_data = calibration_data.to(device)

        # 确保模型参数在 GPU 上
        if not next(self.parameters()).is_cuda:
            for param in self.parameters():
                param.data = param.data.to(device)
            for buffer in self.buffers():
                buffer.data = buffer.data.to(device)

        # 处理 batch_first
        if self.batch_first:
            calibration_data = calibration_data.transpose(0, 1).contiguous()

        time_steps, batch_size, input_size = calibration_data.shape
        hidden_size = self.hidden_size

        # ===== 前向方向校准 =====
        W, R, bx, br = self._convert_weights_to_haste_format(device, reverse=False)

        # 初始化 quant_ranges（如果尚未初始化）
        if self.quant_ranges is None:
            self.quant_ranges = gru_ops.GRUQuantizationRanges(hidden_size)

        # 累积更新量化范围（前向）
        gru_ops.calibrate_gru_ranges(
            time_steps=time_steps,
            batch_size=batch_size,
            input_size=input_size,
            hidden_size=hidden_size,
            W=W,
            R=R,
            bx=bx,
            br=br,
            x=calibration_data,
            quant_ranges=self.quant_ranges
        )

        # ===== 反向方向校准（仅双向时） =====
        if self.bidirectional:
            W_rev, R_rev, bx_rev, br_rev = self._convert_weights_to_haste_format(device, reverse=True)

            # 初始化反向 quant_ranges
            if self.quant_ranges_reverse is None:
                self.quant_ranges_reverse = gru_ops.GRUQuantizationRanges(hidden_size)

            # 反向输入：时间维度翻转
            calibration_data_reversed = calibration_data.flip(0).contiguous()

            # 累积更新量化范围（反向）
            gru_ops.calibrate_gru_ranges(
                time_steps=time_steps,
                batch_size=batch_size,
                input_size=input_size,
                hidden_size=hidden_size,
                W=W_rev,
                R=R_rev,
                bx=bx_rev,
                br=br_rev,
                x=calibration_data_reversed,
                quant_ranges=self.quant_ranges_reverse
            )

        # 确保权重连续性
        self.weight_ih_l0.data = self.weight_ih_l0.data.contiguous()
        self.weight_hh_l0.data = self.weight_hh_l0.data.contiguous()
        if self.bias:
            self.bias_ih_l0.data = self.bias_ih_l0.data.contiguous()
            self.bias_hh_l0.data = self.bias_hh_l0.data.contiguous()

        if self.bidirectional:
            self.weight_ih_l0_reverse.data = self.weight_ih_l0_reverse.data.contiguous()
            self.weight_hh_l0_reverse.data = self.weight_hh_l0_reverse.data.contiguous()
            if self.bias:
                self.bias_ih_l0_reverse.data = self.bias_ih_l0_reverse.data.contiguous()
                self.bias_hh_l0_reverse.data = self.bias_hh_l0_reverse.data.contiguous()

    def _initialize_quantization(self, calibration_data: torch.Tensor):
        """一次性完成校准（内部方法，向后兼容）"""
        self._accumulate_calibration_ranges(calibration_data)
        self.finalize_calibration()

    # -------------------- 重写方法 --------------------

    def forward(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            input: 输入张量，形状为 [seq_len, batch, input_size] 或 [batch, seq_len, input_size]
            hx: 初始隐藏状态
                - 单向: [num_layers, batch, hidden_size]
                - 双向: [num_layers * 2, batch, hidden_size]

        Returns:
            output: 输出张量
                - 单向: [seq_len, batch, hidden_size] 或 [batch, seq_len, hidden_size]
                - 双向: [seq_len, batch, 2*hidden_size] 或 [batch, seq_len, 2*hidden_size]
            h_n: 最终隐藏状态
                - 单向: [num_layers, batch, hidden_size]
                - 双向: [num_layers * 2, batch, hidden_size]

        Raises:
            RuntimeError: 如果启用了量化但未校准
        """
        # 初始化 cublas
        self._ensure_cublas_initialized()

        # 检查量化是否已校准完成
        if self.use_quantization and not self.is_calibrated():
            if self.quant_ranges is not None:
                # 已累积范围但未完成校准，自动调用 finalize
                self.finalize_calibration()
            else:
                # 未进行任何校准
                raise RuntimeError(
                    "Quantization is enabled but not calibrated. "
                    "Please call calibrate(data) then finalize_calibration() before forward pass."
                )

        # 处理 batch_first
        if self.batch_first:
            input = input.transpose(0, 1).contiguous()  # [B, T, I] -> [T, B, I]

        seq_len, batch_size, input_size = input.shape
        hidden_size = self.hidden_size

        # 确保输入在 CUDA 上且为 float32
        device = input.device if input.is_cuda else torch.device('cuda')
        input = ensure_cuda_float32(input, device)

        # 处理初始隐藏状态
        h0_forward = None
        h0_reverse = None
        if hx is not None:
            expected_layers = self.num_layers * self.num_directions
            expected_shape = (expected_layers, batch_size, hidden_size)
            if hx.shape != expected_shape:
                raise ValueError(
                    f"Expected hx shape {expected_shape} (num_layers*num_directions={expected_layers}, "
                    f"batch_size={batch_size}, hidden_size={hidden_size}), got {hx.shape}"
                )
            h0_forward = ensure_cuda_float32(hx[0], device)
            if self.bidirectional:
                h0_reverse = ensure_cuda_float32(hx[1], device)

        # ===== 前向方向 =====
        weight_ih = self.weight_ih_l0
        weight_hh = self.weight_hh_l0
        bias_ih = self.bias_ih_l0 if self.bias else None
        bias_hh = self.bias_hh_l0 if self.bias else None

        output_forward, h_n_forward = GRUFunction.apply(
            input,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            h0_forward,
            self.training,
            self.use_quantization,
            self.quant_params
        )

        if self.bidirectional:
            # ===== 反向方向 =====
            weight_ih_rev = self.weight_ih_l0_reverse
            weight_hh_rev = self.weight_hh_l0_reverse
            bias_ih_rev = self.bias_ih_l0_reverse if self.bias else None
            bias_hh_rev = self.bias_hh_l0_reverse if self.bias else None

            # 反转输入的时间维度
            input_reversed = input.flip(0)

            output_reverse, h_n_reverse = GRUFunction.apply(
                input_reversed,
                weight_ih_rev,
                weight_hh_rev,
                bias_ih_rev,
                bias_hh_rev,
                h0_reverse,
                self.training,
                self.use_quantization,
                self.quant_params_reverse
            )

            # 反转反向输出以对齐时间步
            output_reverse = output_reverse.flip(0)

            # 拼接前向和反向输出：[T, B, H] + [T, B, H] -> [T, B, 2H]
            output = torch.cat([output_forward, output_reverse], dim=-1)

            # 拼接隐藏状态：[1, B, H] + [1, B, H] -> [2, B, H]
            h_n = torch.cat([h_n_forward, h_n_reverse], dim=0)
        else:
            output = output_forward
            h_n = h_n_forward

        # 处理 batch_first
        if self.batch_first:
            output = output.transpose(0, 1).contiguous()  # [T, B, H] -> [B, T, H]

        return output, h_n
