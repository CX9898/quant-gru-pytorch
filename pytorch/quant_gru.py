"""
QuantGRU - 支持量化的 GRU 实现

功能特性:
    - 兼容 nn.GRU 接口（支持 batch_first、bidirectional 等参数）
    - 支持 INT8/INT16/INT32 量化推理
    - 支持 MinMax 和 AIMET 风格直方图校准
    - 延迟初始化设计，支持 pickle/deepcopy 序列化
    - 支持 ONNX 导出（export_mode=True 时使用纯 PyTorch 实现）

典型用法:
    >>> gru = QuantGRU(64, 128, batch_first=True)
    >>> gru.calibrate(calibration_data)
    >>> gru.finalize_calibration()
    >>> gru.use_quantization = True
    >>> output, h_n = gru(input_data)

ONNX 导出用法:
    >>> from quant_gru import set_export_mode, verify_export_mode
    >>>
    >>> # 方式 1: 使用辅助函数
    >>> set_export_mode(model, True)   # 切换到 PyTorch 模式
    >>> torch.onnx.export(model, dummy_input, "model.onnx")
    >>> set_export_mode(model, False)  # 恢复 CUDA 模式
    >>>
    >>> # 方式 2: 直接设置属性
    >>> gru.export_mode = True
    >>> torch.onnx.export(model, dummy_input, "model.onnx")
    >>> gru.export_mode = False
    >>>
    >>> # 验证输出一致性
    >>> verify_export_mode(model, dummy_input)
"""

import json
import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import gru_interface_binding as gru_ops
except ImportError:
    raise ImportError(
        "gru_interface_binding 模块未找到，请先运行 setup.py 编译 C++ 扩展"
    )


# ============================================================
#                      位宽配置工具函数
# ============================================================


def _get_bitwidth_value(op_cfg: dict) -> int:
    """从配置中获取位宽值（8/16/32），默认 8"""
    return op_cfg.get('bitwidth', 8)


def _get_symmetric_value(op_cfg: dict) -> bool:
    """从配置中获取是否对称量化，默认 True"""
    return op_cfg.get('is_symmetric', True)


def load_bitwidth_config(config_file: str) -> gru_ops.OperatorQuantConfig:
    """
    从 JSON 文件加载量化配置
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        OperatorQuantConfig 对象
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
    """格式化位宽值: 8 -> '8bit'"""
    return f"{abs(val)}bit"


def _format_symmetric(is_symmetric: bool) -> str:
    """格式化对称量化: True -> '对称'"""
    return "对称" if is_symmetric else "非对称"


def apply_bitwidth_config(config: gru_ops.OperatorQuantConfig,
                          config_file: str,
                          verbose: bool = False) -> int:
    """
    从 JSON 文件应用配置到现有 OperatorQuantConfig 对象
    
    Args:
        config: 要更新的配置对象
        config_file: 配置文件路径
        verbose: 是否打印配置详情
        
    Returns:
        配置的字段数量
    """
    loaded = load_bitwidth_config(config_file)

    # 位宽配置字段（18 个）
    bitwidth_attrs = ['x_', 'h_', 'W_', 'R_', 'bx_', 'br_', 'Wx_', 'Rh_',
                      'z_pre_', 'z_out_', 'r_pre_', 'r_out_', 'g_pre_', 'g_out_',
                      'Rh_add_br_', 'rRh_', 'old_contrib_', 'new_contrib_']
    for attr in bitwidth_attrs:
        setattr(config, attr, getattr(loaded, attr))

    # 对称量化配置字段（18 个）
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

    return len(bitwidth_attrs) + len(symmetric_attrs)  # 36 个字段


# ============================================================
#                      权重格式转换
# ============================================================

def set_export_mode(model: nn.Module, mode: bool = True) -> int:
    """
    设置模型中所有 QuantGRU 的导出模式

    Args:
        model: 包含 QuantGRU 的模型
        mode: True 启用导出模式（纯 PyTorch），False 使用 CUDA

    Returns:
        设置的 QuantGRU 数量

    Example:
        >>> set_export_mode(model, True)
        >>> torch.onnx.export(model, dummy_input, "model.onnx")
        >>> set_export_mode(model, False)  # 恢复 CUDA 模式
    """
    count = 0
    for m in model.modules():
        if isinstance(m, QuantGRU):
            m.export_mode = mode
            count += 1
    return count


def verify_export_mode(model: nn.Module, dummy_input: torch.Tensor,
                       rtol: float = 1e-3, atol: float = 1e-5) -> bool:
    """
    验证 CUDA 和 Python 实现的输出一致性

    Args:
        model: 包含 QuantGRU 的模型
        dummy_input: 测试输入
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度

    Returns:
        True 如果输出一致，否则 False
    """
    model.eval()

    with torch.no_grad():
        # CUDA 模式
        set_export_mode(model, False)
        output_cuda = model(dummy_input)
        if isinstance(output_cuda, tuple):
            output_cuda = output_cuda[0]

        # Python 模式
        set_export_mode(model, True)
        output_python = model(dummy_input)
        if isinstance(output_python, tuple):
            output_python = output_python[0]

        # 恢复 CUDA 模式
        set_export_mode(model, False)

    # 比较
    if output_cuda.shape != output_python.shape:
        print(f"❌ 形状不匹配: CUDA {output_cuda.shape} vs Python {output_python.shape}")
        return False

    max_diff = (output_cuda - output_python).abs().max().item()
    mean_diff = (output_cuda - output_python).abs().mean().item()

    is_close = torch.allclose(output_cuda, output_python, rtol=rtol, atol=atol)

    if is_close:
        print(f"✅ 输出一致: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    else:
        print(f"❌ 输出不一致: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    return is_close


def reorder_weights_pytorch_to_haste(w: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 权重格式 (r,z,n) -> Haste 格式 (z,r,n)
    
    Args:
        w: 形状 [3*H, ...] 的权重张量
        
    Returns:
        重排序后的张量，形状不变
    """
    w = w.contiguous()
    h3 = w.shape[0] // 3
    device = w.device
    # [r, z, n] -> [z, r, n]
    indices = torch.cat([
        torch.arange(h3, 2 * h3, device=device),
        torch.arange(0, h3, device=device),
        torch.arange(2 * h3, 3 * h3, device=device)
    ])
    return w.index_select(0, indices).contiguous()


def reorder_weights_haste_to_pytorch(w: torch.Tensor) -> torch.Tensor:
    """
    Haste 权重格式 (z,r,n) -> PyTorch 格式 (r,z,n)
    
    Args:
        w: 形状 [3*H, ...] 的权重张量
        
    Returns:
        重排序后的张量，形状不变
    """
    w = w.contiguous()
    h3 = w.shape[0] // 3
    device = w.device
    # [z, r, n] -> [r, z, n]
    indices = torch.cat([
        torch.arange(h3, 2 * h3, device=device),
        torch.arange(0, h3, device=device),
        torch.arange(2 * h3, 3 * h3, device=device)
    ])
    return w.index_select(0, indices).contiguous()


def ensure_cuda_float32(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """确保张量在 CUDA 上且为 float32 类型"""
    if not tensor.is_cuda:
        tensor = tensor.to(device)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


# ============================================================
#                      GRUFunction (autograd)
# ============================================================

class GRUFunction(torch.autograd.Function):
    """
    GRU 自定义 autograd Function
    
    负责 PyTorch/Haste 格式转换、调用 C++ 接口、管理反向传播
    """

    @staticmethod
    def forward(ctx, input, weight_ih, weight_hh, bias_ih, bias_hh, h0, is_training,
                use_quantization=False, quant_params=None):
        """
        前向传播
        
        Args:
            input: [T, B, I] 输入序列
            weight_ih: [3*H, I] 输入权重 (PyTorch r,z,n 格式)
            weight_hh: [3*H, H] 循环权重
            bias_ih, bias_hh: [3*H] 偏置或 None
            h0: [B, H] 初始状态或 None
            is_training: 训练模式标志
            use_quantization: 量化开关
            quant_params: 量化参数
            
        Returns:
            output: [T, B, H] 输出序列
            h_n: [1, B, H] 最终状态
        """
        time_steps, batch_size, input_size = input.shape
        hidden_size = weight_hh.shape[1]

        # 保存维度信息和 None 标志
        ctx.time_steps, ctx.batch_size = time_steps, batch_size
        ctx.input_size, ctx.hidden_size = input_size, hidden_size
        ctx.bias_ih_is_none = (bias_ih is None)
        ctx.bias_hh_is_none = (bias_hh is None)
        ctx.h0_is_none = (h0 is None)

        device = input.device if input.is_cuda else torch.device('cuda')
        input = ensure_cuda_float32(input, device)

        # 权重格式转换: PyTorch (r,z,n) -> Haste (z,r,n)，并转置
        weight_ih = ensure_cuda_float32(weight_ih, device)
        weight_hh = ensure_cuda_float32(weight_hh, device)
        W = reorder_weights_pytorch_to_haste(weight_ih).t().contiguous()
        R = reorder_weights_pytorch_to_haste(weight_hh).t().contiguous()

        # 偏置处理
        if bias_ih is not None and bias_hh is not None:
            bias_ih = ensure_cuda_float32(bias_ih, device)
            bias_hh = ensure_cuda_float32(bias_hh, device)
            bx = reorder_weights_pytorch_to_haste(bias_ih).contiguous()
            br = reorder_weights_pytorch_to_haste(bias_hh).contiguous()
        else:
            bx = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)
            br = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)

        # 初始状态
        h0_tensor = ensure_cuda_float32(h0, device) if h0 is not None else torch.empty(0, device=device, dtype=torch.float32)

        # 量化参数
        if use_quantization:
            if quant_params is None:
                raise RuntimeError("use_quantization=True 时必须提供 quant_params")
        else:
            quant_params = gru_ops.GRUQuantitativeParameters()

        # 调用 C++ 前向接口
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

        # 分离输出: output_full[0] 是初始状态，[1:] 是时间步输出
        output = output_full[1:]
        h_n = output_full[-1:]

        # 保存反向传播所需的中间结果
        ctx.save_for_backward(W, R, bx, br, input, output_full, v)

        return output, h_n

    @staticmethod
    def backward(ctx, grad_output, grad_h_n):
        """
        反向传播
        
        Args:
            grad_output: [T, B, H] 输出梯度
            grad_h_n: [1, B, H] 最终状态梯度
            
        Returns:
            对应 forward 各参数的梯度
        """
        W, R, bx, br, input, h, v = ctx.saved_tensors
        time_steps, batch_size = ctx.time_steps, ctx.batch_size
        input_size, hidden_size = ctx.input_size, ctx.hidden_size

        # 确保所有张量在 CUDA 上
        device = grad_output.device
        tensors = [W, R, bx, br, input, h]
        W, R, bx, br, input, h = [t.to(device) if not t.is_cuda else t for t in tensors]
        if v is not None and not v.is_cuda:
            v = v.to(device)
        if not grad_output.is_cuda:
            grad_output = grad_output.to(device)
        if grad_h_n is not None and not grad_h_n.is_cuda:
            grad_h_n = grad_h_n.to(device)

        # 构建隐藏状态梯度
        # C++ 接口需要 [T+1, B, H] 格式
        # dh_new[0] 是初始状态梯度（保持为 0），dh_new[1:] 是时间步梯度
        dh_new = torch.zeros(
            (time_steps + 1, batch_size, hidden_size),
            device=device, dtype=grad_output.dtype
        )
        dh_new[1:] = grad_output

        # 累加最终状态梯度（output[-1] 和 h_n[0] 指向同一时间步）
        if grad_h_n is not None and grad_h_n.numel() > 0:
            dh_new[-1] = dh_new[-1] + grad_h_n[0]

        # 调用 C++ 反向接口（绑定层会处理格式转换）
        dx, dW, dR, dbx, dbr, dh = gru_ops.haste_gru_backward(
            time_steps=time_steps, batch_size=batch_size,
            input_size=input_size, hidden_size=hidden_size,
            W=W, R=R, bx=bx, br=br, x=input,
            dh_new=dh_new, h=h, v=v
        )

        # 梯度格式转换: Haste (z,r,n) -> PyTorch (r,z,n)
        dW_pytorch = reorder_weights_haste_to_pytorch(dW.t()).contiguous()
        dR_pytorch = reorder_weights_haste_to_pytorch(dR.t()).contiguous()
        dbx_pytorch = reorder_weights_haste_to_pytorch(dbx).contiguous() if not ctx.bias_ih_is_none else None
        dbr_pytorch = reorder_weights_haste_to_pytorch(dbr).contiguous() if not ctx.bias_hh_is_none else None
        grad_h0 = None if ctx.h0_is_none else dh

        # 返回梯度（对应 forward 的 9 个参数）
        return dx, dW_pytorch, dR_pytorch, dbx_pytorch, dbr_pytorch, grad_h0, None, None, None


# ============================================================
#                      QuantGRU 模块
# ============================================================

class QuantGRU(nn.Module):
    """
    支持量化的自定义 GRU 实现，兼容 nn.GRU 接口
    
    特性:
        - 延迟初始化: CUDA handle 在首次使用时初始化
        - 可序列化: 支持 pickle/deepcopy
        - 双向支持: bidirectional=True 时输出维度为 2*hidden_size
        - ONNX 导出: export_mode=True 时使用纯 PyTorch 实现

    量化流程:
        1. gru.load_bitwidth_config("config.json")  # 可选
        2. gru.calibrate(data1), gru.calibrate(data2), ...
        3. gru.finalize_calibration()
        4. gru.use_quantization = True
        5. output, h_n = gru(input)
    
    ONNX 导出流程:
        1. gru.export_mode = True
        2. torch.onnx.export(model, ...)
        3. gru.export_mode = False  # 可选，恢复 CUDA 模式

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: 层数（仅支持 1）
        bias: 是否使用偏置
        batch_first: True 时输入为 [B, T, I]
        dropout: 暂不支持
        bidirectional: 是否双向
    
    Attributes:
        use_quantization: 量化开关（默认 False）
        calibration_method: 校准方法 ('minmax' 或 'histogram')
        export_mode: ONNX 导出模式（默认 False，使用 CUDA；True 时使用纯 PyTorch）
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
        super(QuantGRU, self).__init__()

        if num_layers != 1:
            raise NotImplementedError("仅支持 num_layers=1")
        if dropout > 0:
            raise NotImplementedError("暂不支持 dropout")

        # 基本配置
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_quantization = use_quantization
        self.num_directions = 2 if bidirectional else 1

        # ONNX 导出模式：True 时使用纯 PyTorch 实现，可被 ONNX 追踪
        self.export_mode = False

        # 权重参数（命名与 nn.GRU 一致）
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter('bias_ih_l0', None)
            self.register_parameter('bias_hh_l0', None)

        # 反向权重（双向时）
        if bidirectional:
            self.weight_ih_l0_reverse = nn.Parameter(torch.empty(3 * hidden_size, input_size))
            self.weight_hh_l0_reverse = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            if bias:
                self.bias_ih_l0_reverse = nn.Parameter(torch.empty(3 * hidden_size))
                self.bias_hh_l0_reverse = nn.Parameter(torch.empty(3 * hidden_size))
            else:
                self.register_parameter('bias_ih_l0_reverse', None)
                self.register_parameter('bias_hh_l0_reverse', None)

        self.reset_parameters()

        # 量化状态（延迟创建）
        self.quant_ranges = None          # calibrate() 时创建
        self.quant_params = None          # finalize_calibration() 时创建
        if bidirectional:
            self.quant_ranges_reverse = None
            self.quant_params_reverse = None

        self._calibration_dirty = False   # 校准数据更新标志
        self._bitwidth_config_dict = None # 位宽配置（Python 字典，可序列化）
        self._cublas_initialized = False  # CUDA 延迟初始化标志

        # 校准方法: 'minmax'（快速）或 'histogram'（AIMET 风格，高精度）
        self.calibration_method = 'histogram'

        # 直方图收集器（histogram 方法使用）
        self.hist_collectors = None
        if bidirectional:
            self.hist_collectors_reverse = None

    def reset_parameters(self):
        """权重初始化（与 nn.GRU 相同的均匀分布）"""
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    # -------------------- 内部方法 --------------------

    def _ensure_cublas_initialized(self):
        """延迟初始化 cublas handle"""
        if not self._cublas_initialized:
            gru_ops.init_gru_cublas()
            self._cublas_initialized = True

    def _load_bitwidth_config_to_dict(self, config_file: str):
        """从 JSON 文件加载配置到内部字典"""
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
        """从 Python 字典创建 C++ OperatorQuantConfig 对象"""
        config = gru_ops.OperatorQuantConfig()
        if self._bitwidth_config_dict is not None:
            for attr, value in self._bitwidth_config_dict.items():
                setattr(config, attr, value)
        return config

    def _convert_weights_to_haste_format(self, device: torch.device, reverse: bool = False):
        """
        将权重转换为 Haste 格式 (z,r,n)
        
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
        """累积校准范围"""
        self._ensure_cublas_initialized()

        device = calibration_data.device if calibration_data.is_cuda else torch.device('cuda')
        if not calibration_data.is_cuda:
            calibration_data = calibration_data.to(device)

        # 确保模型在 GPU 上
        if not next(self.parameters()).is_cuda:
            for param in self.parameters():
                param.data = param.data.to(device)
            for buffer in self.buffers():
                buffer.data = buffer.data.to(device)

        if self.batch_first:
            calibration_data = calibration_data.transpose(0, 1).contiguous()

        time_steps, batch_size, input_size = calibration_data.shape
        hidden_size = self.hidden_size

        # 前向校准
        W, R, bx, br = self._convert_weights_to_haste_format(device, reverse=False)
        if self.calibration_method == 'histogram':
            if self.hist_collectors is None:
                self.hist_collectors = gru_ops.GRUHistogramCollectors(hidden_size, num_bins=2048)
            gru_ops.calibrate_gru_histograms(
                time_steps=time_steps, batch_size=batch_size, input_size=input_size, hidden_size=hidden_size,
                W=W, R=R, bx=bx, br=br, x=calibration_data, hist_collectors=self.hist_collectors)
        else:
            if self.quant_ranges is None:
                self.quant_ranges = gru_ops.GRUQuantizationRanges(hidden_size)
            gru_ops.calibrate_gru_ranges(
                time_steps=time_steps, batch_size=batch_size, input_size=input_size, hidden_size=hidden_size,
                W=W, R=R, bx=bx, br=br, x=calibration_data, quant_ranges=self.quant_ranges)

        # 反向校准（双向时）
        if self.bidirectional:
            W_rev, R_rev, bx_rev, br_rev = self._convert_weights_to_haste_format(device, reverse=True)
            calibration_data_reversed = calibration_data.flip(0).contiguous()

            if self.calibration_method == 'histogram':
                if self.hist_collectors_reverse is None:
                    self.hist_collectors_reverse = gru_ops.GRUHistogramCollectors(hidden_size, num_bins=2048)
                gru_ops.calibrate_gru_histograms(
                    time_steps=time_steps, batch_size=batch_size, input_size=input_size, hidden_size=hidden_size,
                    W=W_rev, R=R_rev, bx=bx_rev, br=br_rev, x=calibration_data_reversed,
                    hist_collectors=self.hist_collectors_reverse)
            else:
                if self.quant_ranges_reverse is None:
                    self.quant_ranges_reverse = gru_ops.GRUQuantizationRanges(hidden_size)
                gru_ops.calibrate_gru_ranges(
                    time_steps=time_steps, batch_size=batch_size, input_size=input_size, hidden_size=hidden_size,
                    W=W_rev, R=R_rev, bx=bx_rev, br=br_rev, x=calibration_data_reversed,
                    quant_ranges=self.quant_ranges_reverse)

        # 确保权重连续
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

    # -------------------- 公开接口 --------------------

    def load_bitwidth_config(self, config_file: str, verbose: bool = False):
        """从 JSON 文件加载位宽配置"""
        self._load_bitwidth_config_to_dict(config_file)
        if verbose:
            cpp_config = self._get_cpp_bitwidth_config()
            apply_bitwidth_config(cpp_config, config_file, verbose=True)
            print(f"  [全局]  use_quantization: {self.use_quantization}")

    def set_all_bitwidth(self, bitwidth: int = 8, is_symmetric: bool = True, verbose: bool = False):
        """
        设置所有算子统一的位宽和对称量化配置
        
        Args:
            bitwidth: 位宽 (8/16/32)
            is_symmetric: 是否对称量化
            verbose: 是否打印信息
        """
        if bitwidth not in (8, 16, 32):
            raise ValueError(f"bitwidth must be 8, 16 or 32, got {bitwidth}")
        
        # 初始化配置字典
        if self._bitwidth_config_dict is None:
            self._bitwidth_config_dict = {}
        
        # 位宽属性列表
        bitwidth_attrs = [
            'x_', 'h_', 'W_', 'R_', 'bx_', 'br_', 'Wx_', 'Rh_',
            'z_pre_', 'z_out_', 'r_pre_', 'r_out_', 'g_pre_', 'g_out_',
            'Rh_add_br_', 'rRh_', 'old_contrib_', 'new_contrib_'
        ]
        
        # 对称量化属性列表
        symmetric_attrs = [
            'x_symmetric_', 'h_symmetric_', 'W_symmetric_', 'R_symmetric_',
            'bx_symmetric_', 'br_symmetric_', 'Wx_symmetric_', 'Rh_symmetric_',
            'z_pre_symmetric_', 'z_out_symmetric_', 'r_pre_symmetric_', 'r_out_symmetric_',
            'g_pre_symmetric_', 'g_out_symmetric_', 'Rh_add_br_symmetric_', 'rRh_symmetric_',
            'old_contrib_symmetric_', 'new_contrib_symmetric_'
        ]
        
        # 设置所有位宽
        for attr in bitwidth_attrs:
            self._bitwidth_config_dict[attr] = bitwidth
        
        # 设置所有对称量化配置
        for attr in symmetric_attrs:
            self._bitwidth_config_dict[attr] = is_symmetric
        
        if verbose:
            sym_str = "对称" if is_symmetric else "非对称"
            print(f"\n[QuantGRU] 设置所有算子: {bitwidth}bit {sym_str}量化")

    def is_calibrated(self) -> bool:
        """检查是否已完成校准"""
        if self.bidirectional:
            return self.quant_params is not None and self.quant_params_reverse is not None
        return self.quant_params is not None

    def calibrate(self, calibration_data: torch.Tensor):
        """
        累积校准数据
        
        Args:
            calibration_data: [T, B, I] 或 [B, T, I] (batch_first) 的数据
        
        Note:
            支持增量校准，完成后需调用 finalize_calibration()
        """
        self._accumulate_calibration_ranges(calibration_data)
        self._calibration_dirty = True

    def finalize_calibration(self, verbose: bool = False):
        """
        完成校准，计算量化参数并初始化 LUT
        
        Args:
            verbose: 是否打印校准信息
            
        Raises:
            RuntimeError: 未调用过 calibrate()
        """
        use_histogram = (self.calibration_method == 'histogram')

        # 检查校准数据
        if use_histogram:
            if self.hist_collectors is None or not self.hist_collectors.is_valid():
                raise RuntimeError("未收集直方图数据，请先调用 calibrate()")
        else:
            if self.quant_ranges is None:
                raise RuntimeError("未收集校准数据，请先调用 calibrate()")

        cpp_config = self._get_cpp_bitwidth_config()

        if verbose:
            method_name = {'minmax': 'MINMAX', 'histogram': 'HISTOGRAM'}.get(
                self.calibration_method, self.calibration_method.upper())
            print(f"\n[QuantGRU] 校准方法: {method_name}")

        # 前向方向
        if use_histogram:
            self.quant_params = gru_ops.calculate_gru_quantitative_parameters_from_histograms(
                hist_collectors=self.hist_collectors, bitwidth_config=cpp_config, verbose=verbose)
        else:
            self.quant_params = gru_ops.calculate_gru_quantitative_parameters(
                quant_ranges=self.quant_ranges, bitwidth_config=cpp_config)
        gru_ops.initialize_quantization_lut(quant_params=self.quant_params)

        # 反向方向（双向时）
        if self.bidirectional:
            if use_histogram:
                if self.hist_collectors_reverse is None or not self.hist_collectors_reverse.is_valid():
                    raise RuntimeError("双向 GRU 反向直方图数据异常")
                self.quant_params_reverse = gru_ops.calculate_gru_quantitative_parameters_from_histograms(
                    hist_collectors=self.hist_collectors_reverse, bitwidth_config=cpp_config, verbose=verbose)
            else:
                if self.quant_ranges_reverse is None:
                    raise RuntimeError("双向 GRU 反向校准数据异常")
                self.quant_params_reverse = gru_ops.calculate_gru_quantitative_parameters(
                    quant_ranges=self.quant_ranges_reverse, bitwidth_config=cpp_config)
            gru_ops.initialize_quantization_lut(quant_params=self.quant_params_reverse)

        self._calibration_dirty = False

    def reset_calibration(self):
        """重置校准状态，清除所有累积的范围和参数"""
        self.quant_ranges = None
        self.quant_params = None
        self.hist_collectors = None
        self._calibration_dirty = False
        if self.bidirectional:
            self.quant_ranges_reverse = None
            self.quant_params_reverse = None
            self.hist_collectors_reverse = None

    def forward(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input: [T, B, I] 或 [B, T, I] (batch_first) 的输入
            hx: 初始隐藏状态，单向 [1, B, H]，双向 [2, B, H]
            
        Returns:
            output: [T, B, H] 或 [T, B, 2H] (双向)
            h_n: [1, B, H] 或 [2, B, H] (双向)

        Note:
            - export_mode=False (默认): 使用 CUDA C++ 实现（高性能）
            - export_mode=True: 使用纯 PyTorch 实现（可被 ONNX 追踪）
        """
        # ===== ONNX 导出模式：使用纯 PyTorch 实现 =====
        if self.export_mode:
            return self._forward_python(input, hx)

        # ===== 正常模式：使用 CUDA C++ 实现 =====
        self._ensure_cublas_initialized()

        # 量化模式下检查校准状态
        if self.use_quantization:
            if self._calibration_dirty:
                # 校准数据已更新，需要重新计算量化参数
                self.finalize_calibration()
            elif not self.is_calibrated():
                # 检查是否有未完成的校准数据（支持 minmax 和 histogram 两种方法）
                if self.quant_ranges is not None or self.hist_collectors is not None:
                    # 已累积数据但未完成校准，自动调用 finalize
                    self.finalize_calibration()
                else:
                    raise RuntimeError("量化已启用但未校准，请先调用 calibrate() 和 finalize_calibration()")

        if self.batch_first:
            input = input.transpose(0, 1).contiguous()

        seq_len, batch_size, input_size = input.shape
        hidden_size = self.hidden_size

        device = input.device if input.is_cuda else torch.device('cuda')
        input = ensure_cuda_float32(input, device)

        # 初始状态处理
        h0_forward, h0_reverse = None, None
        if hx is not None:
            expected_layers = self.num_layers * self.num_directions
            expected_shape = (expected_layers, batch_size, hidden_size)
            if hx.shape != expected_shape:
                raise ValueError(f"hx 形状应为 {expected_shape}，实际 {hx.shape}")
            h0_forward = ensure_cuda_float32(hx[0], device)
            if self.bidirectional:
                h0_reverse = ensure_cuda_float32(hx[1], device)

        # 前向方向
        output_forward, h_n_forward = GRUFunction.apply(
            input, self.weight_ih_l0, self.weight_hh_l0,
            self.bias_ih_l0 if self.bias else None,
            self.bias_hh_l0 if self.bias else None,
            h0_forward, self.training, self.use_quantization, self.quant_params)

        if self.bidirectional:
            # 反向方向
            output_reverse, h_n_reverse = GRUFunction.apply(
                input.flip(0), self.weight_ih_l0_reverse, self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse if self.bias else None,
                self.bias_hh_l0_reverse if self.bias else None,
                h0_reverse, self.training, self.use_quantization, self.quant_params_reverse)

            # 反转反向输出以对齐时间步
            output_reverse = output_reverse.flip(0)
            # 拼接输出: [T, B, H] + [T, B, H] -> [T, B, 2H]
            output = torch.cat([output_forward, output_reverse], dim=-1)
            # 拼接隐藏状态: [1, B, H] + [1, B, H] -> [2, B, H]
            h_n = torch.cat([h_n_forward, h_n_reverse], dim=0)
        else:
            output = output_forward
            h_n = h_n_forward

        if self.batch_first:
            output = output.transpose(0, 1).contiguous()

        return output, h_n


# ============================================================
#                      调试工具函数
# ============================================================

def print_quant_params(gru: QuantGRU):
    """
    打印 QuantGRU 的量化参数

    Args:
        gru: 已完成校准的 QuantGRU 实例
    """
    if not gru.is_calibrated():
        raise RuntimeError("请先调用 finalize_calibration()")

    params = gru.quant_params
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


def print_quant_ranges(gru: QuantGRU):
    """
    打印 QuantGRU 的量化范围

    Args:
        gru: 已调用 calibrate() 的 QuantGRU 实例
    """
    if gru.quant_ranges is None:
        raise RuntimeError("请先调用 calibrate()")

    r = gru.quant_ranges
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
