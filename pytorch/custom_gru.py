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
        "operator_config": {
            "input.x": { "output_bitwidth": 8, "is_symmetric": true },
            "gate.z_out": { "output_bitwidth": 8, "is_symmetric": false },
            ...
        }
    }
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    config = gru_ops.OperatorQuantConfig()
    op_config = data.get('operator_config', {})
    
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
        "op.one_minus_update": ("one_minus_update_", "one_minus_update_symmetric_"),
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
                      'Rh_add_br_', 'rRh_', 'one_minus_update_', 'old_contrib_', 'new_contrib_']
    for attr in bitwidth_attrs:
        setattr(config, attr, getattr(loaded, attr))
    
    # 复制对称量化配置字段
    symmetric_attrs = ['x_symmetric_', 'h_symmetric_', 'W_symmetric_', 'R_symmetric_',
                       'bx_symmetric_', 'br_symmetric_', 'Wx_symmetric_', 'Rh_symmetric_',
                       'z_pre_symmetric_', 'z_out_symmetric_', 'r_pre_symmetric_', 'r_out_symmetric_',
                       'g_pre_symmetric_', 'g_out_symmetric_', 'Rh_add_br_symmetric_', 'rRh_symmetric_',
                       'one_minus_update_symmetric_', 'old_contrib_symmetric_', 'new_contrib_symmetric_']
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
        print(f"  [运算]  Rh+br: {_format_bitwidth(config.Rh_add_br_):6s} ({_format_symmetric(config.Rh_add_br_symmetric_)})")
        print(f"          rRh: {_format_bitwidth(config.rRh_):6s} ({_format_symmetric(config.rRh_symmetric_)})")
        print(f"          1-z: {_format_bitwidth(config.one_minus_update_):6s} ({_format_symmetric(config.one_minus_update_symmetric_)})")
        print(f"  [输出]  old: {_format_bitwidth(config.old_contrib_):6s} ({_format_symmetric(config.old_contrib_symmetric_)})")
        print(f"          new: {_format_bitwidth(config.new_contrib_):6s} ({_format_symmetric(config.new_contrib_symmetric_)})")
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
        torch.arange(hidden_size_3, 2*hidden_size_3, device=device),  # z
        torch.arange(0, hidden_size_3, device=device),                 # r
        torch.arange(2*hidden_size_3, 3*hidden_size_3, device=device) # n
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
        torch.arange(hidden_size_3, 2*hidden_size_3, device=device),  # r (在 Haste 中是第二部分)
        torch.arange(0, hidden_size_3, device=device),                 # z (在 Haste 中是第一部分)
        torch.arange(2*hidden_size_3, 3*hidden_size_3, device=device) # n (在 Haste 中是第三部分)
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
                use_quantization=False, quant_type='int8', quant_params=None):
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
            quant_type: 量化类型，'int8' 或 'int16'
            quant_params: 量化参数

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
        h_n = output_full[-1:]    # [1, batch_size, hidden_size]

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

        # 返回梯度（对应 forward 的 10 个参数）
        return dx, dW_pytorch, dR_pytorch, dbx_pytorch, dbr_pytorch, grad_h0, None, None, None, None


# ==================== CustomGRU：自定义 GRU 类 ====================

class CustomGRU(nn.GRU):
    """
    继承自 PyTorch nn.GRU 的自定义类，支持量化前向传播

    量化校准流程：
        1. 创建模型：gru = CustomGRU(..., use_quantization=True)
        2. 累积校准数据：gru.calibrate(data1), gru.calibrate(data2), ...（可多次调用）
        3. 完成校准：gru.finalize_calibration()（只能调用一次）
        4. 正常使用：output, h_n = gru(input)
        5. 如需重新校准：gru.reset_calibration() 后回到步骤 2

    校准状态：
        - quant_ranges: 累积的量化范围（min/max），calibrate() 时更新
        - quant_params: 最终的量化参数（scale/zp），finalize_calibration() 时计算
        - is_calibrated(): 返回 quant_params 是否已设置

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: GRU 层数（目前仅支持单层）
        bias: 是否使用偏置
        batch_first: 如果为 True，输入形状为 [batch, seq, feature]
        dropout: 层间 dropout 概率（目前不支持）
        bidirectional: 是否双向（目前不支持）
        use_quantization: 是否使用量化
        quant_type: 量化类型，'int8' 或 'int16'
        calibration_data: 用于校准的输入数据（可选，提供则立即完成校准）

    Examples:
        >>> # 多次校准（推荐）
        >>> gru = CustomGRU(64, 128, use_quantization=True)
        >>> for batch in calibration_loader:
        ...     gru.calibrate(batch)
        >>> gru.finalize_calibration()
        >>> output, h_n = gru(input_data)

        >>> # 一次性校准（向后兼容）
        >>> gru = CustomGRU(64, 128, use_quantization=True, calibration_data=data)
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
        quant_type: str = 'int8',
        calibration_data: Optional[torch.Tensor] = None,
        bitwidth_config_file: Optional[str] = None
    ):
        # 检查限制
        if num_layers != 1:
            raise NotImplementedError("Currently only supports num_layers=1")
        if bidirectional:
            raise NotImplementedError("Currently does not support bidirectional GRU")
        if dropout > 0:
            raise NotImplementedError("Currently does not support dropout")

        super(CustomGRU, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # 量化相关配置
        self.use_quantization = use_quantization
        self.quant_type = quant_type.lower()
        if self.quant_type not in ['int8', 'int16']:
            raise ValueError(f"quant_type must be 'int8' or 'int16', got {self.quant_type}")

        # 初始化 cublas handle
        gru_ops.init_gru_cublas()

        # 量化状态初始化
        self.quant_ranges = None  # 累积的量化范围（min/max）
        self.quant_params = None  # 计算得到的量化参数（scale/zp）
        
        # 位宽配置
        self.bitwidth_config = gru_ops.OperatorQuantConfig()
        if bitwidth_config_file is not None:
            self.load_bitwidth_config(bitwidth_config_file)

        # 如果提供了校准数据，立即完成校准（向后兼容）
        if self.use_quantization and calibration_data is not None:
            self._initialize_quantization(calibration_data)

    # -------------------- 位宽配置接口 --------------------
    
    def load_bitwidth_config(self, config_file: str, verbose: bool = False):
        """
        从 JSON 配置文件加载量化位宽配置
        
        Args:
            config_file: JSON 配置文件路径
            verbose: 是否打印详细信息
            
        使用示例:
            gru.load_bitwidth_config("config/gru_quant_bitwidth_config.json", verbose=True)
        """
        apply_bitwidth_config(self.bitwidth_config, config_file, verbose)
    
    def get_bitwidth_config(self) -> gru_ops.OperatorQuantConfig:
        """
        获取当前的位宽配置对象
        
        Returns:
            OperatorQuantConfig 对象
        """
        return self.bitwidth_config

    # -------------------- 校准状态查询 --------------------

    def is_calibrated(self) -> bool:
        """
        检查量化是否已完成校准

        Returns:
            True 如果已调用 finalize_calibration()，否则 False
        """
        return self.quant_params is not None

    # -------------------- 公共校准接口 --------------------

    def calibrate(self, calibration_data: torch.Tensor):
        """
        累积校准数据，更新量化范围

        可多次调用，每次调用会将新数据的范围与已有范围合并（取并集）。
        完成所有数据的校准后，需调用 finalize_calibration() 计算量化参数。

        Args:
            calibration_data: 校准数据，形状为 [seq_len, batch, input_size]
                             （如果 batch_first=True，则为 [batch, seq_len, input_size]）

        Raises:
            RuntimeError: 量化未启用，或已调用过 finalize_calibration()

        Note:
            一旦调用了 finalize_calibration()，再调用此方法会报错。
            如需重新校准，请先调用 reset_calibration()。
        """
        if not self.use_quantization:
            raise RuntimeError(
                "Cannot calibrate: quantization is not enabled. "
                "Set use_quantization=True when creating the model."
            )
        if self.is_calibrated():
            raise RuntimeError(
                "Cannot calibrate: finalize_calibration() has already been called. "
                "Call reset_calibration() first if you want to recalibrate."
            )
        self._accumulate_calibration_ranges(calibration_data)

    def finalize_calibration(self):
        """
        完成校准，计算量化参数并初始化 LUT 表

        根据累积的量化范围和位宽配置计算各算子的 scale 和 zero_point。
        此方法只能调用一次。

        Raises:
            RuntimeError: 未调用过 calibrate()，或已调用过此方法

        Note:
            调用此方法后，不能再调用 calibrate()。
            如需重新校准，请先调用 reset_calibration()。
            
            如果需要自定义位宽配置，请在调用此方法前先调用 load_bitwidth_config()。
        """
        if self.is_calibrated():
            raise RuntimeError(
                "finalize_calibration() has already been called. "
                "Call reset_calibration() first if you want to recalibrate."
            )
        if self.quant_ranges is None:
            raise RuntimeError(
                "No calibration data accumulated. "
                "Call calibrate(data) at least once before finalize_calibration()."
            )

        # 根据范围和位宽配置计算量化参数
        self.quant_params = gru_ops.calculate_gru_quantitative_parameters(
            quant_ranges=self.quant_ranges,
            bitwidth_config=self.bitwidth_config
        )
        torch.cuda.synchronize()

        # 初始化查找表
        gru_ops.initialize_quantization_lut(quant_params=self.quant_params)
        torch.cuda.synchronize()

    def reset_calibration(self):
        """
        重置校准状态

        清除累积的量化范围和量化参数，允许重新开始校准流程。
        """
        self.quant_ranges = None
        self.quant_params = None

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
        print(f"  [one_minus_update] exp2_inv={params.exp2_inv_one_minus_update_:3d}, zp={params.zp_one_minus_update_}")
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
        print(f"  [one_minus_update] min={r.min_one_minus_update_:12.6f}, max={r.max_one_minus_update_:12.6f}")
        print(f"  [new_contrib]      min={r.min_new_contrib_:12.6f}, max={r.max_new_contrib_:12.6f}")
        print(f"  [old_contrib]      min={r.min_old_contrib_:12.6f}, max={r.max_old_contrib_:12.6f}")
        print("=" * 60)

    # -------------------- 内部方法 --------------------

    def _convert_weights_to_haste_format(self, device: torch.device):
        """
        将 PyTorch 格式的权重转换为 Haste 格式（用于量化校准）

        Returns:
            W, R, bx, br: Haste 格式的权重和偏置
        """
        weight_ih = ensure_cuda_float32(self.weight_ih_l0, device)
        weight_hh = ensure_cuda_float32(self.weight_hh_l0, device)
        W = reorder_weights_pytorch_to_haste(weight_ih).t().contiguous()
        R = reorder_weights_pytorch_to_haste(weight_hh).t().contiguous()

        if self.bias:
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

        # 转换权重格式
        W, R, bx, br = self._convert_weights_to_haste_format(device)

        # 初始化 quant_ranges（如果尚未初始化）
        if self.quant_ranges is None:
            self.quant_ranges = gru_ops.GRUQuantizationRanges(hidden_size)

        # 累积更新量化范围
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
        torch.cuda.synchronize()

        # 确保权重连续性
        self.weight_ih_l0.data = self.weight_ih_l0.data.contiguous()
        self.weight_hh_l0.data = self.weight_hh_l0.data.contiguous()
        if self.bias:
            self.bias_ih_l0.data = self.bias_ih_l0.data.contiguous()
            self.bias_hh_l0.data = self.bias_hh_l0.data.contiguous()

        # 重置 flatten_parameters 的内部状态
        if hasattr(self, '_flat_weights'):
            self._flat_weights = None

    def _initialize_quantization(self, calibration_data: torch.Tensor):
        """一次性完成校准（内部方法，向后兼容）"""
        self._accumulate_calibration_ranges(calibration_data)
        self.finalize_calibration()

    # -------------------- 重写方法 --------------------

    def _apply(self, fn):
        """
        重写 _apply 方法，在量化校准后正确处理设备迁移

        量化已校准时手动应用函数，避免触发 flatten_parameters() 导致 CUDA 状态冲突。
        """
        if self.is_calibrated():
            if hasattr(self, '_flat_weights'):
                self._flat_weights = None
            for param in self.parameters():
                if param is not None:
                    param.data = fn(param.data)
                    if param._grad is not None:
                        param._grad.data = fn(param._grad.data)
            for buffer in self.buffers():
                if buffer is not None:
                    buffer.data = fn(buffer.data)
            return self
        return super(CustomGRU, self)._apply(fn)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            input: 输入张量，形状为 [seq_len, batch, input_size] 或 [batch, seq_len, input_size]
            hx: 初始隐藏状态，形状为 [num_layers, batch, hidden_size]

        Returns:
            output: 输出张量，形状与 input 相同但最后一维为 hidden_size
            h_n: 最终隐藏状态，形状为 [num_layers, batch, hidden_size]

        Raises:
            RuntimeError: 如果启用了量化但未校准
        """
        # 检查量化是否已校准完成
        if self.use_quantization and not self.is_calibrated():
            if self.quant_ranges is not None:
                # 已累积范围但未完成校准，自动调用 finalize
                self.finalize_calibration()
            else:
                # 未进行任何校准
                raise RuntimeError(
                    "Quantization is enabled but not calibrated. "
                    "Please call calibrate(data) before forward pass, "
                    "or provide calibration_data in __init__."
                )

        # 处理 batch_first
        if self.batch_first:
            input = input.transpose(0, 1).contiguous()  # [B, T, I] -> [T, B, I]，确保连续内存布局

        seq_len, batch_size, input_size = input.shape
        hidden_size = self.hidden_size

        # 处理初始隐藏状态
        h0 = None
        if hx is not None:
            expected_shape = (self.num_layers, batch_size, hidden_size)
            if hx.shape != expected_shape:
                raise ValueError(
                    f"Expected hx shape {expected_shape} (num_layers={self.num_layers}, "
                    f"batch_size={batch_size}, hidden_size={hidden_size}), got {hx.shape}"
                )
            device = input.device if input.is_cuda else torch.device('cuda')
            h0 = ensure_cuda_float32(hx[0], device)

        # 确保输入在 CUDA 上且为 float32
        device = input.device if input.is_cuda else torch.device('cuda')
        input = ensure_cuda_float32(input, device)

        # 获取权重和偏置
        weight_ih = self.weight_ih_l0
        weight_hh = self.weight_hh_l0
        bias_ih = self.bias_ih_l0 if self.bias else None
        bias_hh = self.bias_hh_l0 if self.bias else None

        # 调用 GRUFunction 进行前向传播
        output, h_n_from_func = GRUFunction.apply(
            input,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            h0,
            self.training,
            self.use_quantization,
            self.quant_type,
            self.quant_params
        )

        # 处理 batch_first
        if self.batch_first:
            output = output.transpose(0, 1).contiguous()  # [T, B, H] -> [B, T, H]，确保连续内存布局

        # 确保 h_n 形状正确
        assert h_n_from_func.shape[0] == 1, f"Expected h_n shape [1, batch, hidden_size], got {h_n_from_func.shape}"

        return output, h_n_from_func
