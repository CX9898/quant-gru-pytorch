"""
QuantGRU - 支持量化的 GRU 实现

功能特性:
    - 兼容 nn.GRU 接口(支持 batch_first、bidirectional 等参数)
    - 支持任意位宽 (1-32 bit) 混合精度量化推理
    - 支持 MinMax / SQNR / Percentile 校准方法
    - 支持 JSON 配置文件指定各算子的位宽和对称量化设置
    - 支持量化参数导出/导入（JSON 格式，便于部署和调试）
    - 支持 ONNX 导出(float / QDQ 两种格式)

关键属性:
    - use_quantization: 是否启用量化(默认 False)
    - calibrating: 是否在 forward 中收集校准数据(默认 False)
    - calibration_method: 校准方法 'minmax'|'sqnr'|'percentile'(默认 'sqnr')
    - export_mode: 是否使用 ONNX 导出模式(默认 False)
    - export_format: ONNX 导出格式 'float'|'qdq'(默认 'float')

典型用法:
    >>> from quant_gru import QuantGRU
    >>>
    >>> # 创建模型
    >>> gru = QuantGRU(64, 128, batch_first=True).cuda()
    >>>
    >>> # 加载位宽配置(可选，校准前使用)
    >>> gru.load_bitwidth_config("config.json", verbose=True)
    >>>
    >>> # 校准(在 forward 中收集校准数据)
    >>> gru.calibrating = True
    >>> output = gru(calibration_data)
    >>> gru.calibrating = False
    >>>
    >>> # 量化推理(自动调用 finalize_calibration)
    >>> gru.use_quantization = True
    >>> output = gru(x)

量化参数导出/导入:
    >>> # 导出（训练/校准环境）
    >>> torch.save(gru.state_dict(), "weights.pth")
    >>> gru.export_quant_params("quant_params.json")
    >>>
    >>> # 导入（部署环境）- 从 JSON 读取模型配置
    >>> import json
    >>> with open("quant_params.json") as f:
    ...     config = json.load(f)["model_info"]
    >>> gru2 = QuantGRU(
    ...     config["input_size"], config["hidden_size"],
    ...     batch_first=config["batch_first"],
    ...     bidirectional=config["bidirectional"]
    ... ).cuda()
    >>> gru2.load_state_dict(torch.load("weights.pth"))
    >>> gru2.load_quant_params("quant_params.json")
    >>> gru2.use_quantization = True

调整量化配置:
    >>> # 修改单个算子的位宽（自动调整 scale）
    >>> gru.adjust_quant_config("z_out", bitwidth=16, verbose=True)
    >>>
    >>> # 获取量化配置
    >>> config = gru.get_quant_config("z_out")
    >>> print(config)  # {'bitwidth': 16, 'exp2_inv': 14, ...}

调试工具（模块级函数）:
    >>> from quant_gru import print_quant_config, print_quant_params
    >>> print_quant_config(gru)  # 打印所有算子的量化配置
    >>> print_quant_params(gru)  # 打印量化参数详情

ONNX 导出:
    >>> gru.export_mode = True
    >>> gru.export_format = 'float'  # 或 'qdq' (量化模型, 可选)
    >>> torch.onnx.export(gru, x, "model.onnx", dynamo=False)  # PyTorch 2.x 需要 dynamo=False
    >>> gru.export_mode = False
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
#                   模块级常量与配置映射
# ============================================================
# 统一算子映射表（唯一数据源）
# 格式: "算子名" -> {
#   "bw_attr": 位宽属性名,
#   "sym_attr": 对称量化属性名,
#   "exp2_inv_attr": exp2_inv 属性名,
#   "zp_attr": zp 属性名 (None 表示无 zp，如 per-channel 权重),
#   "is_per_channel": 是否 per-channel
# }
def _make_op_info(base_name: str, is_per_channel: bool = False, default_unsigned: bool = False):
    """
    生成算子信息字典（减少重复代码）
    
    Args:
        base_name: 基础属性名（如 "x_", "z_out_"）
        is_per_channel: 是否 per-channel 量化
        default_unsigned: C++ 默认是否 unsigned（False=INT, True=UINT）
    
    属性命名规律:
        - bw_attr: "{base_name}" (位宽)
        - sym_attr: "{base_name}symmetric_" (对称量化)
        - unsigned_attr: "{base_name}unsigned_" (无符号量化，只标记例外)
        - exp2_inv_attr: "exp2_inv_{base_name}" (量化指数)
        - zp_attr: "zp_{base_name}" (零点，per-channel 为 None)
    """
    return {
        "bw_attr": base_name,
        "sym_attr": f"{base_name}symmetric_",
        "unsigned_attr": f"{base_name}unsigned_",
        "exp2_inv_attr": f"exp2_inv_{base_name}",
        "zp_attr": None if is_per_channel else f"zp_{base_name}",
        "is_per_channel": is_per_channel,
        "default_unsigned": default_unsigned,
    }


# 算子映射表：JSON 字段名 → C++ 属性名
# 每个算子的属性命名遵循统一规律
_OPERATOR_MAP = {
    # 输入
    "input.x": _make_op_info("x_"),
    # 隐藏状态输出（每时间步的输出，同时作为下一时间步的输入）
    "output.h": _make_op_info("h_"),
    # 权重（per-channel）
    "weight.W": _make_op_info("W_", is_per_channel=True),
    "weight.R": _make_op_info("R_", is_per_channel=True),
    "weight.bx": _make_op_info("bx_", is_per_channel=True),
    "weight.br": _make_op_info("br_", is_per_channel=True),
    # 矩阵乘法结果
    "matmul.Wx": _make_op_info("Wx_"),
    "matmul.Rh": _make_op_info("Rh_"),
    # 门控（激活前/后）
    "gate.z_pre": _make_op_info("z_pre_"),
    "gate.z_out": _make_op_info("z_out_", default_unsigned=True),  # Sigmoid [0,1] → UINT
    "gate.r_pre": _make_op_info("r_pre_"),
    "gate.r_out": _make_op_info("r_out_", default_unsigned=True),  # Sigmoid [0,1] → UINT
    "gate.g_pre": _make_op_info("g_pre_"),
    "gate.g_out": _make_op_info("g_out_"),  # Tanh [-1,1]
    # 中间操作
    "op.Rh_add_br": _make_op_info("Rh_add_br_"),
    "op.rRh": _make_op_info("rRh_"),
    "op.old_contrib": _make_op_info("old_contrib_"),
    "op.new_contrib": _make_op_info("new_contrib_"),
}

# 派生常量：从映射表提取的 C++ 属性名集合
_VALID_BITWIDTH_ATTRS = {info["bw_attr"] for info in _OPERATOR_MAP.values()}
_VALID_SYMMETRIC_ATTRS = {info["sym_attr"] for info in _OPERATOR_MAP.values()}
_VALID_UNSIGNED_ATTRS = {info["unsigned_attr"] for info in _OPERATOR_MAP.values()}

# 短名称到属性映射（从 _OPERATOR_MAP 提取，避免重复构建）
# 例如: "x" -> {"bw_attr": "x_", "sym_attr": "x_symmetric_", ...}
_OPERATOR_SHORT_NAME_MAP = {
    op_name.split('.')[-1]: {
        'bw_attr': info["bw_attr"],
        'sym_attr': info["sym_attr"],
        'unsigned_attr': info.get("unsigned_attr"),
        'exp2_inv_attr': info["exp2_inv_attr"],
        'zp_attr': info["zp_attr"],
        'is_per_channel': info["is_per_channel"],
        'json_key': op_name,
    }
    for op_name, info in _OPERATOR_MAP.items()
}

# 对称量化属性分类（用于 set_all_bitwidth）
# - 权重/偏置：始终使用对称量化（zero_point=0），计算效率更高
# - 激活值：可配置，非对称量化可能提高精度但增加计算开销
_WEIGHT_SYMMETRIC_ATTRS = {'W_symmetric_', 'R_symmetric_', 'bx_symmetric_', 'br_symmetric_'}
_ACTIVATION_SYMMETRIC_ATTRS = _VALID_SYMMETRIC_ATTRS - _WEIGHT_SYMMETRIC_ATTRS


def _validate_operator_map():
    """
    验证 Python 端映射表与 C++ 端属性定义一致性
    
    在模块加载时自动调用，确保 _OPERATOR_MAP 中的属性名
    与 C++ OperatorQuantConfig 的实际属性一致。
    不一致时立即抛出异常，避免运行时静默失败。
    """
    try:
        test_config = gru_ops.OperatorQuantConfig()
    except Exception as e:
        # gru_ops 可能未正确加载，跳过验证（后续使用时会报错）
        import warnings
        warnings.warn(f"无法验证 _OPERATOR_MAP: {e}")
        return

    missing_attrs = []
    for op_name, info in _OPERATOR_MAP.items():
        if not hasattr(test_config, info["bw_attr"]):
            missing_attrs.append(f"{op_name} -> {info['bw_attr']}")
        if not hasattr(test_config, info["sym_attr"]):
            missing_attrs.append(f"{op_name} -> {info['sym_attr']}")

    if missing_attrs:
        raise RuntimeError(
            f"_OPERATOR_MAP 与 C++ OperatorQuantConfig 不一致！\n"
            f"缺少属性: {missing_attrs}\n"
            f"请检查 gru_interface_binding.cc 中的 OperatorQuantConfigPy 定义"
        )


# 模块加载时执行一致性验证（import 时自动运行）
_validate_operator_map()


# ============================================================
#                      权重格式转换
# ============================================================
#
# PyTorch GRU 和 Haste GRU 使用不同的门控顺序：
#   - PyTorch: (reset, update, new) 即 (r, z, n)
#   - Haste:   (update, reset, new) 即 (z, r, n)
#
# 权重张量形状为 [3*H, ...]，需要重排序前 2/3 的部分。

def reorder_weights_pytorch_to_haste(w: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 权重格式 (r,z,n) -> Haste 格式 (z,r,n)
    
    注意：此操作与 reorder_weights_haste_to_pytorch 实现相同，
          因为交换 r 和 z 是自反操作（执行两次等于不变）。
    
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
    
    注意：此操作与 reorder_weights_pytorch_to_haste 实现相同，
          因为交换 r 和 z 是自反操作（执行两次等于不变）。
    
    Args:
        w: 形状 [3*H, ...] 的权重张量
        
    Returns:
        重排序后的张量，形状不变
    """
    return reorder_weights_pytorch_to_haste(w)


def ensure_cuda_float32(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """确保张量在 CUDA 上且为 float32 类型"""
    if not tensor.is_cuda:
        tensor = tensor.to(device)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


def convert_weights_to_haste_format(
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: Optional[torch.Tensor],
        bias_hh: Optional[torch.Tensor],
        hidden_size: int,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 PyTorch GRU 权重转换为 Haste 格式(独立工具函数)
    
    PyTorch 格式: (r, z, n)
    Haste 格式:   (z, r, n)
    
    Args:
        weight_ih: [3*H, I] 输入权重 (PyTorch 格式)
        weight_hh: [3*H, H] 循环权重 (PyTorch 格式)
        bias_ih: [3*H] 输入偏置 或 None
        bias_hh: [3*H] 循环偏置 或 None
        hidden_size: 隐藏层大小(用于创建零偏置)
        device: 目标设备
        
    Returns:
        (W, R, bx, br): Haste 格式的权重和偏置
            - W: [I, 3*H] 转置后的输入权重
            - R: [H, 3*H] 转置后的循环权重
            - bx: [3*H] 输入偏置
            - br: [3*H] 循环偏置
    """
    # 权重转换: 重排序 (r,z,n) -> (z,r,n) 并转置
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

    return W, R, bx, br


# ============================================================
#                      QDQ (Quantize-Dequantize) 伪量化
# ============================================================
#
# 伪量化用于 ONNX 导出，在浮点域模拟量化效果：
#   q = clamp(round(x / scale) + zp, qmin, qmax)
#   x' = (q - zp) * scale
#
# 推理引擎（如 TensorRT）会识别 QDQ 模式并替换为真实量化算子。
#
# 量化参数说明：
#   - exp2_inv: 量化指数，scale = 2^(-exp2_inv)
#   - zp: 零点（对称量化时为 0）
#   - bitwidth: 位宽 (1-32)

def get_quant_range(bitwidth: int, is_unsigned: bool = False) -> Tuple[int, int]:
    """
    计算任意位宽的量化范围
    
    Args:
        bitwidth: 位宽 (1-32)
        is_unsigned: 是否无符号（False=INT, True=UINT）
        
    Returns:
        (qmin, qmax): 量化范围
        
    Examples:
        >>> get_quant_range(8)          # INT8 (默认)
        (-128, 127)
        >>> get_quant_range(8, True)    # UINT8
        (0, 255)
        >>> get_quant_range(4)          # INT4
        (-8, 7)
    """
    if not (1 <= bitwidth <= 32):
        raise ValueError(f"bitwidth must be in range [1, 32], got {bitwidth}")
    
    if is_unsigned:
        qmin = 0
        qmax = (1 << bitwidth) - 1
    else:
        qmin = -(1 << (bitwidth - 1))
        qmax = (1 << (bitwidth - 1)) - 1
    
    return qmin, qmax

def fake_quantize(x: torch.Tensor, exp2_inv: int, zp: int = 0,
                  bitwidth: int = 8, symmetric: bool = True,
                  is_unsigned: bool = False) -> torch.Tensor:
    """
    伪量化(Fake Quantize): 量化后立即反量化，保持浮点格式
    
    用于 ONNX 导出，推理引擎会识别 QDQ 模式并优化
    
    [与 CUDA 一致] 量化参数 (exp2_inv, zp) 与 CUDA 端完全一致
    [ONNX 兼容] 使用浮点运算模拟量化效果
    
    Args:
        x: 输入张量
        exp2_inv: 量化指数 (scale = 2^(-exp2_inv))
        zp: 零点
        bitwidth: 位宽 (1-32)
        symmetric: 对称量化 (影响 zp 的使用方式)
        is_unsigned: 是否无符号（只标记 UINT 例外）
                     - False: INT 范围(默认)，如 8bit: -128~127
                     - True: UINT 范围，如 8bit: 0~255
    """
    # 计算 scale
    if exp2_inv >= 0:
        scale = 1.0 / (1 << exp2_inv)
    else:
        scale = float(1 << (-exp2_inv))

    # 确定量化范围
    qmin, qmax = get_quant_range(bitwidth, is_unsigned)

    # 量化: q = clamp(round(x / scale) + zp, qmin, qmax)
    # 注意: torch.round 使用银行家舍入，与 CUDA 的 round half up 略有差异
    # 但实际影响极小 (随机数据差异率 < 0.001%)
    q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)

    # 反量化: x' = (q - zp) * scale
    x_dequant = (q - zp) * scale

    return x_dequant


def fake_quantize_per_channel(x: torch.Tensor, exp2_invs: list, zp: int = 0,
                              bitwidth: int = 8, symmetric: bool = True,
                              is_unsigned: bool = False) -> torch.Tensor:
    """
    Per-channel 伪量化
    
    [与 CUDA 一致] per-channel 量化参数与 CUDA quantificationPerChannel 一致
    [ONNX 兼容] 使用浮点运算模拟量化效果
    
    Args:
        x: 输入张量
        exp2_invs: per-channel 量化指数列表
        zp: 零点
        bitwidth: 位宽 (1-32)
        symmetric: 对称量化
        is_unsigned: 是否无符号（只标记 UINT 例外）
    """
    # 确定量化范围
    qmin, qmax = get_quant_range(bitwidth, is_unsigned)

    device = x.device
    result = torch.zeros_like(x)
    channel_size = len(exp2_invs)

    for c in range(channel_size):
        exp2_inv = exp2_invs[c]
        if exp2_inv >= 0:
            scale = 1.0 / (1 << exp2_inv)
        else:
            scale = float(1 << (-exp2_inv))

        q = torch.clamp(torch.round(x[..., c] / scale) + zp, qmin, qmax)
        result[..., c] = (q - zp) * scale

    return result


# ============================================================
#                   GRUFunction (autograd.Function)
# ============================================================
#
# PyTorch 自定义算子，连接 Python 层与 C++ CUDA 实现。
# 负责：
#   1. 权重格式转换（PyTorch ↔ Haste）
#   2. 调用 C++ forward/backward 接口
#   3. 管理梯度计算和中间变量保存

class GRUFunction(torch.autograd.Function):
    """
    GRU 自定义 autograd Function
    
    职责：
        - forward: 权重格式转换 → 调用 gru_ops.forward → 返回输出
        - backward: 梯度格式转换 → 调用 gru_ops.haste_gru_backward → 返回梯度
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

        # 权重格式转换(使用统一工具函数)
        W, R, bx, br = convert_weights_to_haste_format(
            weight_ih, weight_hh, bias_ih, bias_hh, hidden_size, device
        )

        # 初始状态
        h0_tensor = ensure_cuda_float32(h0, device) if h0 is not None else torch.empty(0, device=device,
                                                                                       dtype=torch.float32)

        # 量化参数
        if use_quantization:
            if quant_params is None:
                raise RuntimeError("use_quantization=True 时必须提供 quant_params")
        else:
            quant_params = gru_ops.GRUQuantitativeParameters()

        # 调用 C++ 前向接口
        output_full, v = gru_ops.forward(
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
        # dh_new[0] 是初始状态梯度(保持为 0)，dh_new[1:] 是时间步梯度
        dh_new = torch.zeros(
            (time_steps + 1, batch_size, hidden_size),
            device=device, dtype=grad_output.dtype
        )
        dh_new[1:] = grad_output

        # 累加最终状态梯度(output[-1] 和 h_n[0] 指向同一时间步)
        if grad_h_n is not None and grad_h_n.numel() > 0:
            dh_new[-1] = dh_new[-1] + grad_h_n[0]

        # 调用 C++ 反向接口(绑定层会处理格式转换)
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

        # 返回梯度(对应 forward 的 9 个参数)
        return dx, dW_pytorch, dR_pytorch, dbx_pytorch, dbr_pytorch, grad_h0, None, None, None


# ============================================================
#                      QuantGRU 核心模块
# ============================================================
#
# QuantGRU 是本模块的核心类，提供：
#   - 兼容 nn.GRU 的接口
#   - 任意位宽 (1-32 bit) 混合精度量化推理
#   - 多种校准方法（MinMax/SQNR/Percentile）
#   - ONNX 导出支持（float/QDQ 格式）
#
# 内部状态管理：
#   - _bitwidth_config: C++ OperatorQuantConfig 对象（位宽配置）
#   - _quant_params_dirty: 脏标志（配置修改或校准数据变化时置 True）
#   - quant_params: 量化参数（finalize_calibration 后生成）

class QuantGRU(nn.Module):
    """
    支持量化的 GRU 实现，兼容 nn.GRU 接口
    
    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: 层数（仅支持 1）
        bias: 是否使用偏置（默认 True）
        batch_first: True 时输入为 [B, T, I]，False 时为 [T, B, I]（默认 False）
        dropout: 暂不支持，必须为 0
        bidirectional: 是否双向（默认 False）
        use_quantization: 是否启用量化（默认 False）
    
    Attributes:
        use_quantization (bool): 量化开关
        calibrating (bool): 校准模式开关，True 时 forward 会收集校准数据
        calibration_method (str): 校准方法 'minmax'|'sqnr'|'percentile'（默认 'sqnr'）
        percentile_value (float): 百分位值，仅 'percentile' 方法使用（默认 99.99）
        export_mode (bool): ONNX 导出模式，True 时使用纯 PyTorch 实现
        export_format (str): 导出格式 'float'|'qdq'（默认 'float'）
    
    Example:
        >>> gru = QuantGRU(64, 128, batch_first=True).cuda()
        >>> gru.calibrating = True
        >>> _ = gru(calibration_data)  # 收集校准数据
        >>> gru.calibrating = False
        >>> gru.use_quantization = True
        >>> output, h_n = gru(x)  # 量化推理
    
    Note:
        - 仅支持单层 GRU（num_layers=1）
        - 不支持 dropout
        - 量化推理需要先校准（设置 calibrating=True 并运行 forward）
        - 支持 pickle/deepcopy，但校准数据不会被保存（位宽配置会保留）
        - pickle/deepcopy 后如需量化推理，必须重新校准
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

        # ONNX 导出开关：True 时使用纯 PyTorch 实现，可被 ONNX 追踪
        self.export_mode = False
        # 导出格式(高级选项，仅在 export_mode=True 时有效)
        # 'float': 浮点(默认，与 Haste GRU 行为一致)
        # 'qdq': QDQ 伪量化(推荐用于量化模型)
        self._export_format = 'float'

        # 权重参数(命名与 nn.GRU 一致)
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter('bias_ih_l0', None)
            self.register_parameter('bias_hh_l0', None)

        # 反向权重(双向时)
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

        # 量化状态(延迟创建)
        self.quant_ranges = None  # calibrate() 时创建
        self.quant_params = None  # finalize_calibration() 时创建
        if bidirectional:
            self.quant_ranges_reverse = None
            self.quant_params_reverse = None

        # 统一脏标志：标记量化参数是否需要更新（校准数据变化或配置修改都会设置）
        self._quant_params_dirty = False

        # 位宽配置对象（直接初始化，避免延迟创建的线程安全问题）
        self._bitwidth_config = gru_ops.OperatorQuantConfig()  # 位宽配置(直接存储 C++ 对象)

        self._cublas_initialized = False  # CUDA 延迟初始化标志

        # 校准方法:
        #   - 'minmax': 使用 min/max 范围(快速，无直方图)
        #   - 'sqnr': SQNR 优化搜索最优 scale(基于直方图，高精度)
        #   - 'percentile': 百分位裁剪(基于直方图)
        self.calibration_method = 'minmax'

        # Percentile 配置(仅 calibration_method='percentile' 时使用)
        self.percentile_value = 99.99

        # 直方图收集器(sqnr/percentile 方法使用)
        self.hist_collectors = None
        if bidirectional:
            self.hist_collectors_reverse = None

        # 校准模式标志：当为 True 时，forward() 会同时收集校准数据
        self.calibrating = False

    def __getstate__(self):
        """
        序列化状态（用于 pickle/deepcopy）
        
        将 C++ 扩展对象转换为 Python 字典，使 QuantGRU 可被序列化。
        
        Note:
            - 位宽配置会被保留（包括 bitwidth、symmetric、unsigned）
            - 校准数据（quant_params 等）不会被保存，反序列化后需重新校准
        """
        state = self.__dict__.copy()
        
        # 将 _bitwidth_config (C++ 对象) 转换为 Python 字典
        if self._bitwidth_config is not None:
            bitwidth_dict = {}
            for op_name, info in _OPERATOR_MAP.items():
                bw_attr = info["bw_attr"]
                sym_attr = info["sym_attr"]
                unsigned_attr = info.get("unsigned_attr")
                bitwidth_dict[bw_attr] = getattr(self._bitwidth_config, bw_attr)
                bitwidth_dict[sym_attr] = getattr(self._bitwidth_config, sym_attr)
                if unsigned_attr:
                    bitwidth_dict[unsigned_attr] = getattr(self._bitwidth_config, unsigned_attr)
            state['_bitwidth_config'] = bitwidth_dict
        
        # C++ 对象无法序列化，设为 None（反序列化后需重新校准）
        state['quant_ranges'] = None
        state['quant_params'] = None
        state['hist_collectors'] = None
        if self.bidirectional:
            state['quant_ranges_reverse'] = None
            state['quant_params_reverse'] = None
            state['hist_collectors_reverse'] = None
        
        # 重置运行时状态（反序列化后需要重新初始化）
        state['_cublas_initialized'] = False
        state['_quant_params_dirty'] = False
        
        return state

    def __setstate__(self, state):
        """
        反序列化状态
        
        从 Python 字典重建 C++ 扩展对象。
        
        Note:
            反序列化后如需量化推理，必须重新校准。
        """
        # 恢复 _bitwidth_config
        bitwidth_dict = state.get('_bitwidth_config')
        if isinstance(bitwidth_dict, dict):
            # 从字典重建 C++ 对象
            config = gru_ops.OperatorQuantConfig()
            for attr, value in bitwidth_dict.items():
                setattr(config, attr, value)
            state['_bitwidth_config'] = config
        elif bitwidth_dict is None:
            # 创建默认配置
            state['_bitwidth_config'] = gru_ops.OperatorQuantConfig()
        
        self.__dict__.update(state)

    def reset_parameters(self):
        """权重初始化(与 nn.GRU 相同的均匀分布)"""
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    # -------------------- 内部方法 --------------------

    def _ensure_cublas_initialized(self):
        """延迟初始化 cublas handle"""
        if not self._cublas_initialized:
            gru_ops.init_gru_cublas()
            self._cublas_initialized = True

    def _use_histogram_collection(self) -> bool:
        """判断是否使用直方图收集(sqnr/percentile 都需要)"""
        return self.calibration_method in ('sqnr', 'percentile')

    def _parse_initial_state(
            self,
            hx: Optional[torch.Tensor],
            batch_size: int,
            device: torch.device = None,
            to_cuda: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        解析初始隐藏状态(统一接口)
        
        Args:
            hx: 初始隐藏状态，形状 [num_directions, B, H] 或 None
            batch_size: 批次大小
            device: 目标设备(to_cuda=True 时使用)
            to_cuda: 是否转换为 CUDA float32
            
        Returns:
            (h0_forward, h0_reverse): 前向和反向初始状态
        """
        h0_forward, h0_reverse = None, None
        if hx is not None:
            expected_layers = self.num_layers * self.num_directions
            expected_shape = (expected_layers, batch_size, self.hidden_size)
            if hx.shape != expected_shape:
                raise ValueError(f"hx 形状应为 {expected_shape}，实际 {hx.shape}")
            h0_forward = ensure_cuda_float32(hx[0], device) if to_cuda else hx[0]
            if self.bidirectional:
                h0_reverse = ensure_cuda_float32(hx[1], device) if to_cuda else hx[1]
        return h0_forward, h0_reverse

    def _combine_bidirectional_outputs(
            self,
            output_forward: torch.Tensor,
            h_n_forward: torch.Tensor,
            output_reverse: Optional[torch.Tensor] = None,
            h_n_reverse: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        合并双向 GRU 输出(统一接口)
        
        Args:
            output_forward: 前向输出 [T, B, H]
            h_n_forward: 前向最终状态 [1, B, H]
            output_reverse: 反向输出 [T, B, H](已翻转或未翻转均可)
            h_n_reverse: 反向最终状态 [1, B, H]
            
        Returns:
            (output, h_n): 合并后的输出和状态
        """
        if self.bidirectional and output_reverse is not None:
            # 拼接输出: [T, B, H] + [T, B, H] -> [T, B, 2H]
            output = torch.cat([output_forward, output_reverse], dim=-1)
            # 拼接隐藏状态: [1, B, H] + [1, B, H] -> [2, B, H]
            h_n = torch.cat([h_n_forward, h_n_reverse], dim=0)
        else:
            output = output_forward
            h_n = h_n_forward
        return output, h_n

    # -------------------- 公开接口 --------------------

    def load_bitwidth_config(self, config_file: str, verbose: bool = False):
        """
        从 JSON 文件加载位宽配置（2 层设计：JSON → C++ 对象）
        
        此方法用于在校准之前设置位宽配置。如果已完成校准或已加载量化参数，
        请使用 adjust_quant_config() 来修改配置。
        
        Args:
            config_file: JSON 配置文件路径
            verbose: 是否打印配置信息
            
        Raises:
            RuntimeError: 如果已加载量化参数（应使用 adjust_quant_config 替代）
        """
        import warnings
        
        # 问题 3 修复：如果已加载量化参数，禁止使用此方法
        if self.is_calibrated():
            raise RuntimeError(
                "已完成校准或已加载量化参数，不能使用 load_bitwidth_config()。\n"
                "如需修改位宽配置，请使用以下方法之一：\n"
                "  1. adjust_quant_config(gru, 'input.x', bitwidth=16, auto_scale=True)\n"
                "  2. 重新校准：gru.reset_calibration() 后再调用 load_bitwidth_config()"
            )

        # 解析 JSON 文件
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 读取 GRU_config 节点下的配置
        gru_config = data.get('GRU_config', {})

        # 读取全局配置
        default_config = gru_config.get('default_config', {})
        if 'disable_quantization' in default_config:
            self.use_quantization = not default_config['disable_quantization']

        # 直接将配置写入 C++ 对象
        op_config = gru_config.get('operator_config', {})

        # 检查 JSON 中缺失的字段并发出警告
        missing_fields = []
        for op_name, info in _OPERATOR_MAP.items():
            bw_attr, sym_attr = info["bw_attr"], info["sym_attr"]
            unsigned_attr = info.get("unsigned_attr")
            if op_name in op_config:
                op_cfg = op_config[op_name]
                bitwidth = op_cfg.get('bitwidth', 8)
                # 验证位宽范围 (1-32)
                if not (1 <= bitwidth <= 32):
                    raise ValueError(
                        f"Invalid bitwidth {bitwidth} for '{op_name}'. "
                        f"Must be in range [1, 32]."
                    )
                setattr(self._bitwidth_config, bw_attr, bitwidth)
                setattr(self._bitwidth_config, sym_attr, op_cfg.get('is_symmetric', True))
                # 设置 unsigned 属性（只标记 UINT 例外）
                if unsigned_attr:
                    default_unsigned = info.get("default_unsigned", False)
                    setattr(self._bitwidth_config, unsigned_attr, op_cfg.get('is_unsigned', default_unsigned))
            else:
                missing_fields.append(op_name)

        if missing_fields:
            warnings.warn(
                f"JSON 配置文件 '{config_file}' 缺少以下字段，将使用默认值 (8bit, 对称):\n"
                f"  {missing_fields}",
                UserWarning
            )

        # 标记量化参数需要更新（forward 时会自动调用 finalize_calibration）
        self._quant_params_dirty = True

        if verbose:
            print_bitwidth_config(self._bitwidth_config, config_file)
            print(f"  [全局]  use_quantization: {self.use_quantization}")

    def set_all_bitwidth(self, bitwidth: int = 8, is_symmetric: bool = True, verbose: bool = False):
        """
        设置所有算子统一的位宽和对称量化配置（2 层设计：直接操作 C++ 对象）
        
        Args:
            bitwidth: 位宽 (1-32)
            is_symmetric: 是否对称量化(仅对激活值生效，权重/偏置始终对称)
            verbose: 是否打印信息
        """
        if not (1 <= bitwidth <= 32):
            raise ValueError(f"bitwidth must be in range [1, 32], got {bitwidth}")

        # 设置所有位宽属性（使用模块级常量）
        for attr in _VALID_BITWIDTH_ATTRS:
            setattr(self._bitwidth_config, attr, bitwidth)

        # 权重/偏置始终使用对称量化（使用模块级常量）
        for attr in _WEIGHT_SYMMETRIC_ATTRS:
            setattr(self._bitwidth_config, attr, True)

        # 激活值对称量化配置由参数控制（使用模块级常量）
        for attr in _ACTIVATION_SYMMETRIC_ATTRS:
            setattr(self._bitwidth_config, attr, is_symmetric)

        # 标记量化参数需要更新（forward 时会自动调用 finalize_calibration）
        self._quant_params_dirty = True

        if verbose:
            sym_str = "对称" if is_symmetric else "非对称"
            print(f"\n[QuantGRU] 设置所有算子: {bitwidth}bit, 激活值{sym_str}量化, 权重/偏置对称量化")

    def is_calibrated(self) -> bool:
        """检查是否已完成校准"""
        if self.bidirectional:
            return self.quant_params is not None and self.quant_params_reverse is not None
        return self.quant_params is not None

    def finalize_calibration(self, verbose: bool = False):
        """
        完成校准，计算量化参数并初始化 LUT
        
        Args:
            verbose: 是否打印校准信息
            
        Raises:
            RuntimeError: 未收集校准数据
        """
        use_histogram = self._use_histogram_collection()
        use_percentile = (self.calibration_method == 'percentile')

        # 检查校准数据
        if use_histogram:
            if self.hist_collectors is None or not self.hist_collectors.is_valid():
                raise RuntimeError("未收集校准数据，请先设置 calibrating=True 并调用 forward()")
        else:
            if self.quant_ranges is None:
                raise RuntimeError("未收集校准数据，请先设置 calibrating=True 并调用 forward()")

        bitwidth_config = self._bitwidth_config

        if verbose:
            method_name = {
                'minmax': 'MINMAX',
                'sqnr': 'SQNR',
                'percentile': f'PERCENTILE ({self.percentile_value}%)'
            }.get(self.calibration_method, self.calibration_method.upper())
            print(f"\n[QuantGRU] 校准方法: {method_name}")

        # 前向方向
        if use_histogram:
            self.quant_params = gru_ops.calculate_gru_quantitative_parameters_from_histograms(
                hist_collectors=self.hist_collectors,
                bitwidth_config=bitwidth_config,
                verbose=verbose,
                use_percentile=use_percentile,
                percentile_value=self.percentile_value)
        else:
            self.quant_params = gru_ops.calculate_gru_quantitative_parameters(
                quant_ranges=self.quant_ranges, bitwidth_config=bitwidth_config)

        # 反向方向(双向时)
        if self.bidirectional:
            if use_histogram:
                if self.hist_collectors_reverse is None or not self.hist_collectors_reverse.is_valid():
                    raise RuntimeError("双向 GRU 反向直方图数据异常")
                self.quant_params_reverse = gru_ops.calculate_gru_quantitative_parameters_from_histograms(
                    hist_collectors=self.hist_collectors_reverse,
                    bitwidth_config=bitwidth_config,
                    verbose=verbose,
                    use_percentile=use_percentile,
                    percentile_value=self.percentile_value)
            else:
                if self.quant_ranges_reverse is None:
                    raise RuntimeError("双向 GRU 反向校准数据异常")
                self.quant_params_reverse = gru_ops.calculate_gru_quantitative_parameters(
                    quant_ranges=self.quant_ranges_reverse, bitwidth_config=bitwidth_config)

        # 量化参数已更新，清除脏标志
        self._quant_params_dirty = False

    def reset_calibration(self):
        """重置校准状态，清除所有累积的范围和参数"""
        self.quant_ranges = None
        self.quant_params = None
        self.hist_collectors = None
        # 重置校准后，脏标志清除（下次校准会重新应用配置）
        self._quant_params_dirty = False
        if self.bidirectional:
            self.quant_ranges_reverse = None
            self.quant_params_reverse = None
            self.hist_collectors_reverse = None

    # -------------------- ONNX 导出模式：纯 PyTorch 实现 --------------------

    def _get_config_attr(self, op_name: str, suffix: str, valid_set: set, default):
        """
        获取配置属性的通用方法
        
        Args:
            op_name: 操作名称（如 'x', 'h', 'Wx' 等）
            suffix: 属性后缀（'_', '_symmetric_', '_unsigned_'）
            valid_set: 有效属性集合
            default: 默认值
            
        Returns:
            属性值，无效操作名返回默认值并发出警告
        """
        attr_name = f'{op_name}{suffix}'
        
        if attr_name not in valid_set:
            import warnings
            warnings.warn(
                f"未知的配置属性名: '{attr_name}'，将返回默认值 {default}。"
                f"有效属性: {sorted(valid_set)}",
                UserWarning
            )
            return default
        
        return getattr(self._bitwidth_config, attr_name, default)

    def _get_bitwidth(self, op_name: str) -> int:
        """获取指定操作的位宽（如 'x', 'h', 'Wx' 等），无效返回 8"""
        return self._get_config_attr(op_name, '_', _VALID_BITWIDTH_ATTRS, 8)

    def _get_symmetric(self, op_name: str) -> bool:
        """获取指定操作是否对称量化，无效返回 True"""
        return self._get_config_attr(op_name, '_symmetric_', _VALID_SYMMETRIC_ATTRS, True)

    def _get_unsigned(self, op_name: str) -> bool:
        """获取指定操作是否无符号量化（False=INT, True=UINT），无效返回 False"""
        return self._get_config_attr(op_name, '_unsigned_', _VALID_UNSIGNED_ATTRS, False)

    @property
    def export_format(self) -> str:
        """
        获取导出格式(高级选项，仅在 export_mode=True 时有效)
        
        Returns:
            'float': 浮点格式(默认，与 Haste GRU 行为一致)
            'qdq': QDQ 伪量化格式(推荐用于量化模型 ONNX 导出)
        """
        return self._export_format

    @export_format.setter
    def export_format(self, mode: str):
        """
        设置导出格式(高级用法，大多数用户不需要修改)
        
        Args:
            mode: 'qdq' | 'float'
        """
        valid_modes = ('qdq', 'float')
        if mode not in valid_modes:
            raise ValueError(f"Invalid export_format: '{mode}'. Use one of {valid_modes}")
        self._export_format = mode

    def _forward_python_single_direction(
            self,
            input: torch.Tensor,
            h0: Optional[torch.Tensor],
            weight_ih: torch.Tensor,
            weight_hh: torch.Tensor,
            bias_ih: Optional[torch.Tensor],
            bias_hh: Optional[torch.Tensor],
            quant_params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        纯 PyTorch 实现的单向 GRU 前向传播(可被 ONNX 追踪)

        GRU 公式(Haste 格式，门顺序为 z, r, g)：
            z = sigmoid(W_z @ x + R_z @ h + bx_z + br_z)  # update gate
            r = sigmoid(W_r @ x + R_r @ h + bx_r + br_r)  # reset gate
            g = tanh(W_g @ x + r * (R_g @ h + br_g) + bx_g)  # candidate gate
            h' = z * h + (1 - z) * g

        量化模式下根据 ONNX 导出模式选择实现：
            - 'qdq': QDQ 格式，使用标准算子 + 伪量化
            - 'float': 标准浮点计算(Haste 格式)

        Args:
            input: [T, B, I] 输入序列
            h0: [B, H] 初始隐藏状态 或 None
            weight_ih: [3*H, I] 输入权重 (PyTorch r,z,n 格式，内部自动转换)
            weight_hh: [3*H, H] 循环权重 (PyTorch r,z,n 格式，内部自动转换)
            bias_ih: [3*H] 输入偏置 或 None (PyTorch 格式，内部自动转换)
            bias_hh: [3*H] 循环偏置 或 None (PyTorch 格式，内部自动转换)
            quant_params: 量化参数(来自 finalize_calibration)

        Returns:
            output: [T, B, H] 输出序列
            h_n: [1, B, H] 最终隐藏状态
        """
        # 根据 export_format 选择实现
        if self._export_format == 'float':
            # 浮点模式：直接使用浮点实现
            return self._forward_python_float_single_direction(
                input, h0, weight_ih, weight_hh, bias_ih, bias_hh
            )

        # qdq 需要量化参数
        if quant_params is None:
            raise RuntimeError(
                f"export_format='{self._export_format}' 需要量化参数，"
                f"请先设置 calibrating=True 并调用 forward()"
            )

        if self._export_format == 'qdq':
            return self._forward_onnx_qdq_single_direction(
                input, h0, weight_ih, weight_hh, bias_ih, bias_hh, quant_params
            )

        # 理论上不会执行到这里(setter 已限制值)，但为了健壮性抛出异常
        raise ValueError(f"未知的 export_format: '{self._export_format}'")

    def _forward_python_float_single_direction(
            self,
            input: torch.Tensor,
            h0: Optional[torch.Tensor],
            weight_ih: torch.Tensor,
            weight_hh: torch.Tensor,
            bias_ih: Optional[torch.Tensor],
            bias_hh: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        浮点实现的单向 GRU 前向传播(Haste 格式)
        
        与 HasteGRU CUDA 浮点推理行为一致
        门控顺序：Haste 格式 (z, r, g)
        
        公式(与 gru_forward_gpu.cu 一致)：
            z = sigmoid(Wx_z + Rh_z + bx_z + br_z)
            r = sigmoid(Wx_r + Rh_r + bx_r + br_r)
            g = tanh(Wx_g + r * (Rh_g + br_g) + bx_g)
            h_new = z * h_old + (1 - z) * g
        
        Args:
            input: [T, B, I] 输入序列
            h0: [B, H] 初始隐藏状态 或 None
            weight_ih: [3*H, I] 输入权重 (PyTorch r,z,n 格式，内部转换)
            weight_hh: [3*H, H] 循环权重 (PyTorch r,z,n 格式，内部转换)
            bias_ih: [3*H] 输入偏置 或 None (PyTorch 格式，内部转换)
            bias_hh: [3*H] 循环偏置 或 None (PyTorch 格式，内部转换)
            
        Returns:
            output: [T, B, H] 输出序列
            h_n: [1, B, H] 最终隐藏状态
        """
        T, B, I = input.shape
        H = self.hidden_size
        device = input.device
        dtype = input.dtype

        # 初始化隐藏状态
        if h0 is None:
            h = torch.zeros(B, H, device=device, dtype=dtype)
        else:
            h = h0

        # 权重格式转换：PyTorch (r,z,n) -> Haste (z,r,g)
        W = reorder_weights_pytorch_to_haste(weight_ih)  # [3*H, I]
        R = reorder_weights_pytorch_to_haste(weight_hh)  # [3*H, H]

        # 处理偏置并转换格式
        if bias_ih is None:
            bx = torch.zeros(3 * H, device=device, dtype=dtype)
        else:
            bx = reorder_weights_pytorch_to_haste(bias_ih)
        if bias_hh is None:
            br = torch.zeros(3 * H, device=device, dtype=dtype)
        else:
            br = reorder_weights_pytorch_to_haste(bias_hh)

        # ========== 循环外一次性计算 Wx GEMM(与 CUDA 一致)==========
        # input: [T, B, I] -> x_flat: [T*B, I]
        # W: [3*H, I] -> W.t(): [I, 3*H]
        # Wx_all: [T*B, 3*H] -> reshape: [T, B, 3*H]
        x_flat = input.reshape(T * B, I)
        Wx_all = torch.mm(x_flat, W.t())  # [T*B, 3*H]
        Wx_all = Wx_all.reshape(T, B, 3 * H)  # [T, B, 3*H]

        # 预分割偏置(循环外完成)
        bx_z, bx_r, bx_g = bx.chunk(3)
        br_z, br_r, br_g = br.chunk(3)

        outputs = []

        for t in range(T):
            # 获取当前时间步的 Wx(已在循环外计算好)
            Wx = Wx_all[t]  # [B, 3*H]

            # Rh = h @ R.T, shape [B, 3H](依赖上一步的 h，必须在循环内)
            Rh = torch.mm(h, R.t())

            # 分割门控(Haste 格式：z, r, g)
            Wx_z, Wx_r, Wx_g = Wx.chunk(3, dim=1)
            Rh_z, Rh_r, Rh_g = Rh.chunk(3, dim=1)

            # Update gate (z)
            z = torch.sigmoid(Wx_z + Rh_z + bx_z + br_z)

            # Reset gate (r)
            r = torch.sigmoid(Wx_r + Rh_r + bx_r + br_r)

            # Candidate gate (g): r 只乘以 (Rh_g + br_g)
            Rh_add_br_g = Rh_g + br_g
            g = torch.tanh(Wx_g + r * Rh_add_br_g + bx_g)

            # 新隐藏状态: h_new = z * h_old + (1 - z) * g
            h = z * h + (1 - z) * g

            outputs.append(h)

        # 堆叠输出: [T, B, H]
        output = torch.stack(outputs, dim=0)
        h_n = h.unsqueeze(0)  # [1, B, H]

        return output, h_n

    # -------------------- ONNX 导出版本(QDQ 格式)--------------------

    def _forward_onnx_qdq_single_direction(
            self,
            input: torch.Tensor,
            h0: Optional[torch.Tensor],
            weight_ih: torch.Tensor,
            weight_hh: torch.Tensor,
            bias_ih: Optional[torch.Tensor],
            bias_hh: Optional[torch.Tensor],
            quant_params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        用于 ONNX 导出的 QDQ 格式前向传播
        
        使用伪量化(Fake Quantize)在关键点插入 Q/DQ 操作，
        推理引擎会识别 QDQ 模式并自动优化为量化算子。
        
        设计原则：
        ==========
        [与 CUDA 一致]
          - 量化参数(scale/zp)完全一致
          - 计算图结构一致(门顺序、计算顺序)
          - 权重/偏置的 per-channel 量化参数一致
          
        [ONNX 兼容 - 与 CUDA 实现不同]
          - GEMM: 使用标准 torch.mm(推理引擎会用 MatMulInteger)
          - sigmoid/tanh: 使用标准 torch.sigmoid/tanh(推理引擎会优化)
          - rescale: 通过 QDQ 实现(不用显式 rshift_round)
        
        Args:
            input: [T, B, I] 输入序列
            h0: [B, H] 初始隐藏状态 或 None
            weight_ih: [3*H, I] 输入权重
            weight_hh: [3*H, H] 循环权重
            bias_ih: [3*H] 输入偏置 或 None
            bias_hh: [3*H] 循环偏置 或 None
            quant_params: 量化参数
            
        Returns:
            output: [T, B, H] 输出序列
            h_n: [1, B, H] 最终隐藏状态
        """
        T, B, I = input.shape
        H = self.hidden_size
        device = input.device
        dtype = input.dtype

        # ========== 量化参数提取 ==========
        # [与 CUDA 一致] 使用相同的量化参数
        exp2_x = quant_params.exp2_inv_x_
        zp_x = quant_params.zp_x_
        exp2_h = quant_params.exp2_inv_h_
        zp_h = quant_params.zp_h_
        exp2_Wx = quant_params.exp2_inv_Wx_
        zp_Wx = quant_params.zp_Wx_
        exp2_Rh = quant_params.exp2_inv_Rh_
        zp_Rh = quant_params.zp_Rh_

        # 激活函数量化参数
        exp2_z_pre = quant_params.exp2_inv_z_pre_
        zp_z_pre = quant_params.zp_z_pre_
        exp2_z_out = quant_params.exp2_inv_z_out_
        zp_z_out = quant_params.zp_z_out_

        exp2_r_pre = quant_params.exp2_inv_r_pre_
        zp_r_pre = quant_params.zp_r_pre_
        exp2_r_out = quant_params.exp2_inv_r_out_
        zp_r_out = quant_params.zp_r_out_

        exp2_g_pre = quant_params.exp2_inv_g_pre_
        zp_g_pre = quant_params.zp_g_pre_
        exp2_g_out = quant_params.exp2_inv_g_out_
        zp_g_out = quant_params.zp_g_out_

        # per-channel 量化参数
        exp2_W = list(quant_params.exp2_inv_W_)
        exp2_R = list(quant_params.exp2_inv_R_)
        exp2_bx = list(quant_params.exp2_inv_bx_)
        exp2_br = list(quant_params.exp2_inv_br_)

        # ========== 权重重排序 ==========
        # [与 CUDA 一致] PyTorch 格式 (r, z, n) -> Haste 格式 (z, r, n)
        W_reordered = reorder_weights_pytorch_to_haste(weight_ih)  # [3*H, I]
        R_reordered = reorder_weights_pytorch_to_haste(weight_hh)  # [3*H, H]

        if bias_ih is not None:
            bx_reordered = reorder_weights_pytorch_to_haste(bias_ih)  # [3*H]
        else:
            bx_reordered = torch.zeros(3 * H, device=device, dtype=dtype)

        if bias_hh is not None:
            br_reordered = reorder_weights_pytorch_to_haste(bias_hh)  # [3*H]
        else:
            br_reordered = torch.zeros(3 * H, device=device, dtype=dtype)

        # ========== 权重伪量化 ==========
        # [与 CUDA 一致] per-channel 量化
        # [ONNX 兼容] 使用 fake_quantize 保持浮点格式
        W_q = fake_quantize_per_channel(W_reordered.t(), exp2_W, zp=0,
                                        bitwidth=self._get_bitwidth('W'),
                                        symmetric=self._get_symmetric('W')).t()
        R_q = fake_quantize_per_channel(R_reordered.t(), exp2_R, zp=0,
                                        bitwidth=self._get_bitwidth('R'),
                                        symmetric=self._get_symmetric('R')).t()
        # 偏置使用配置的位宽(注意：偏置始终使用对称量化)
        bx_q = fake_quantize_per_channel(bx_reordered.unsqueeze(0), exp2_bx, zp=0,
                                         bitwidth=self._get_bitwidth('bx'),
                                         symmetric=self._get_symmetric('bx')).squeeze(0)
        br_q = fake_quantize_per_channel(br_reordered.unsqueeze(0), exp2_br, zp=0,
                                         bitwidth=self._get_bitwidth('br'),
                                         symmetric=self._get_symmetric('br')).squeeze(0)

        # 分割偏置(Haste 格式：z, r, g)
        bx_z, bx_r, bx_g = bx_q.chunk(3)  # 各 [H]
        br_z, br_r, br_g = br_q.chunk(3)  # 各 [H]

        # ========== 初始化隐藏状态 ==========
        if h0 is None:
            h = torch.zeros(B, H, device=device, dtype=dtype)
        else:
            h = h0

        # [与 CUDA 一致] 量化初始状态
        h = fake_quantize(h, exp2_h, zp_h, bitwidth=self._get_bitwidth('h'),
                          symmetric=self._get_symmetric('h'))

        # ========== 输入伪量化 ==========
        # [与 CUDA 一致] 所有时间步一起量化
        x_q = fake_quantize(input, exp2_x, zp_x, bitwidth=self._get_bitwidth('x'),
                            symmetric=self._get_symmetric('x'))

        # ========== Wx GEMM(循环外一次性计算)==========
        # [与 CUDA 一致] 计算顺序一致
        # [ONNX 兼容] 使用标准 matmul，推理引擎会替换为 MatMulInteger
        # x_q: [T, B, I], W_q: [3*H, I] -> Wx: [T, B, 3*H]
        Wx_all = torch.matmul(x_q, W_q.t())  # [T, B, 3*H]

        # [与 CUDA 一致] GEMM 输出量化
        Wx_all = fake_quantize(Wx_all, exp2_Wx, zp_Wx, bitwidth=self._get_bitwidth('Wx'),
                               symmetric=self._get_symmetric('Wx'))

        # 预分配输出张量(ONNX 友好，避免动态列表)
        outputs = torch.zeros(T, B, H, device=device, dtype=dtype)

        for t in range(T):
            Wx = Wx_all[t]  # [B, 3*H]

            # ========== Rh GEMM ==========
            # [与 CUDA 一致] 每个时间步计算 Rh
            # [ONNX 兼容] 使用标准 matmul
            Rh = torch.mm(h, R_q.t())  # [B, 3*H]

            # [与 CUDA 一致] GEMM 输出量化
            Rh = fake_quantize(Rh, exp2_Rh, zp_Rh, bitwidth=self._get_bitwidth('Rh'),
                               symmetric=self._get_symmetric('Rh'))

            # ========== 分割门控 ==========
            # [与 CUDA 一致] Haste 格式 (z, r, g)
            Wx_z, Wx_r, Wx_g = Wx.chunk(3, dim=1)  # 各 [B, H]
            Rh_z, Rh_r, Rh_g = Rh.chunk(3, dim=1)  # 各 [B, H]

            # ========== z 门(Update Gate)==========
            # [与 CUDA 一致] z = sigmoid(Wx_z + Rh_z + bx_z + br_z)
            z_pre = Wx_z + Rh_z + bx_z.unsqueeze(0) + br_z.unsqueeze(0)

            # [与 CUDA 一致] 激活前量化
            z_pre = fake_quantize(z_pre, exp2_z_pre, zp_z_pre,
                                  bitwidth=self._get_bitwidth('z_pre'),
                                  symmetric=self._get_symmetric('z_pre'))

            # [ONNX 兼容] 使用标准 sigmoid(推理引擎会用量化版本或 LUT)
            z = torch.sigmoid(z_pre)

            # [与 CUDA 一致] sigmoid 输出量化（从配置读取所有参数）
            z = fake_quantize(z, exp2_z_out, zp_z_out,
                              bitwidth=self._get_bitwidth('z_out'),
                              symmetric=self._get_symmetric('z_out'),
                              is_unsigned=self._get_unsigned('z_out'))

            # ========== r 门(Reset Gate)==========
            # [与 CUDA 一致] r = sigmoid(Wx_r + Rh_r + bx_r + br_r)
            r_pre = Wx_r + Rh_r + bx_r.unsqueeze(0) + br_r.unsqueeze(0)

            r_pre = fake_quantize(r_pre, exp2_r_pre, zp_r_pre,
                                  bitwidth=self._get_bitwidth('r_pre'),
                                  symmetric=self._get_symmetric('r_pre'))

            # [ONNX 兼容] 使用标准 sigmoid
            r = torch.sigmoid(r_pre)

            # [与 CUDA 一致] sigmoid 输出量化（从配置读取所有参数）
            r = fake_quantize(r, exp2_r_out, zp_r_out,
                              bitwidth=self._get_bitwidth('r_out'),
                              symmetric=self._get_symmetric('r_out'),
                              is_unsigned=self._get_unsigned('r_out'))

            # ========== g 门(New Gate / Candidate)==========
            # [与 CUDA 一致] g = tanh(Wx_g + r * (Rh_g + br_g) + bx_g)
            Rh_add_br = Rh_g + br_g.unsqueeze(0)

            # [与 CUDA 一致] 中间结果量化(从配置读取位宽)
            Rh_add_br = fake_quantize(Rh_add_br, quant_params.exp2_inv_Rh_add_br_,
                                      quant_params.zp_Rh_add_br_,
                                      bitwidth=self._get_bitwidth('Rh_add_br'),
                                      symmetric=self._get_symmetric('Rh_add_br'))

            rRh = r * Rh_add_br

            # [与 CUDA 一致] 乘积量化(从配置读取位宽)
            rRh = fake_quantize(rRh, quant_params.exp2_inv_rRh_,
                                quant_params.zp_rRh_,
                                bitwidth=self._get_bitwidth('rRh'),
                                symmetric=self._get_symmetric('rRh'))

            g_pre = Wx_g + rRh + bx_g.unsqueeze(0)

            g_pre = fake_quantize(g_pre, exp2_g_pre, zp_g_pre,
                                  bitwidth=self._get_bitwidth('g_pre'),
                                  symmetric=self._get_symmetric('g_pre'))

            # [ONNX 兼容] 使用标准 tanh
            g = torch.tanh(g_pre)

            # [与 CUDA 一致] 激活后量化，对称性从配置读取
            g = fake_quantize(g, exp2_g_out, zp_g_out,
                              bitwidth=self._get_bitwidth('g_out'),
                              symmetric=self._get_symmetric('g_out'))

            # ========== 新隐藏状态 ==========
            # [与 CUDA 一致] h_new = z * h + (1 - z) * g
            # CUDA computeH 分别计算并量化 old_contrib 和 new_contrib

            # old_contrib = z * h(从配置读取位宽)
            old_contrib = z * h
            old_contrib = fake_quantize(old_contrib, quant_params.exp2_inv_old_contrib_,
                                        quant_params.zp_old_contrib_,
                                        bitwidth=self._get_bitwidth('old_contrib'),
                                        symmetric=self._get_symmetric('old_contrib'))

            # new_contrib = (1 - z) * g(从配置读取位宽)
            new_contrib = (1 - z) * g
            new_contrib = fake_quantize(new_contrib, quant_params.exp2_inv_new_contrib_,
                                        quant_params.zp_new_contrib_,
                                        bitwidth=self._get_bitwidth('new_contrib'),
                                        symmetric=self._get_symmetric('new_contrib'))

            # h_new = old_contrib + new_contrib
            h_new = old_contrib + new_contrib

            # [与 CUDA 一致] 输出量化
            h_new = fake_quantize(h_new, exp2_h, zp_h,
                                  bitwidth=self._get_bitwidth('h'),
                                  symmetric=self._get_symmetric('h'))

            h = h_new

            # 使用索引赋值存储(ONNX 友好)
            outputs[t] = h

        # ========== 输出 ==========
        output = outputs  # [T, B, H]，已预分配
        h_n = h.unsqueeze(0)  # [1, B, H]

        return output, h_n

    def _forward_python(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        纯 PyTorch 实现的 GRU 前向传播(用于 ONNX 导出)

        支持单向和双向模式
        
        Note: batch_first 转换已在 forward() 中统一处理
        """
        # ===== QDQ 模式提前校验(快速失败)=====
        if self._export_format == 'qdq':
            if self.quant_params is None:
                raise RuntimeError(
                    "export_format='qdq' 需要量化参数，"
                    "请先设置 calibrating=True 进行校准"
                )
            if self.bidirectional and self.quant_params_reverse is None:
                raise RuntimeError(
                    "双向 GRU 的 export_format='qdq' 需要反向量化参数，"
                    "请先设置 calibrating=True 进行校准"
                )

        T, B, I = input.shape

        # 初始状态处理(统一接口)
        h0_forward, h0_reverse = self._parse_initial_state(hx, B, to_cuda=False)

        # 前向方向
        output_forward, h_n_forward = self._forward_python_single_direction(
            input, h0_forward,
            self.weight_ih_l0, self.weight_hh_l0,
            self.bias_ih_l0 if self.bias else None,
            self.bias_hh_l0 if self.bias else None,
            self.quant_params
        )

        # 反向方向(双向时)
        output_reverse, h_n_reverse = None, None
        if self.bidirectional:
            output_reverse, h_n_reverse = self._forward_python_single_direction(
                input.flip(0), h0_reverse,
                self.weight_ih_l0_reverse, self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse if self.bias else None,
                self.bias_hh_l0_reverse if self.bias else None,
                self.quant_params_reverse
            )
            # 反转反向输出以对齐时间步
            output_reverse = output_reverse.flip(0)

        # 合并双向输出(统一接口)
        return self._combine_bidirectional_outputs(
            output_forward, h_n_forward, output_reverse, h_n_reverse
        )

    # -------------------- 校准模式 forward --------------------

    def _ensure_calibration_collector(self, hidden_size: int, reverse: bool = False):
        """
        确保校准收集器已初始化（统一接口）
        
        根据 calibration_method 自动选择并初始化正确的收集器类型
        """
        if self._use_histogram_collection():
            # SQNR/Percentile: 使用直方图收集器
            if reverse:
                if self.hist_collectors_reverse is None:
                    self.hist_collectors_reverse = gru_ops.GRUHistogramCollectors(hidden_size, num_bins=2048)
            else:
                if self.hist_collectors is None:
                    self.hist_collectors = gru_ops.GRUHistogramCollectors(hidden_size, num_bins=2048)
        else:
            # MINMAX: 使用量化范围收集器
            if reverse:
                if self.quant_ranges_reverse is None:
                    self.quant_ranges_reverse = gru_ops.GRUQuantizationRanges(hidden_size)
            else:
                if self.quant_ranges is None:
                    self.quant_ranges = gru_ops.GRUQuantizationRanges(hidden_size)

    def _get_calibration_collectors(self, reverse: bool = False):
        """
        获取校准收集器（统一接口）
        
        Returns:
            (quant_ranges, hist_collectors): 根据校准方法返回对应的收集器
        """
        if self._use_histogram_collection():
            collectors = self.hist_collectors_reverse if reverse else self.hist_collectors
            return None, collectors
        else:
            collectors = self.quant_ranges_reverse if reverse else self.quant_ranges
            return collectors, None

    def _forward_with_calibration(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带校准数据收集的前向传播
        
        在 forward 过程中同时收集校准数据，校准数据通过指针参数原地累积
        
        Note: batch_first 转换已在 forward() 中统一处理
        """
        time_steps, batch_size, input_size = input.shape
        hidden_size = self.hidden_size

        device = input.device if input.is_cuda else torch.device('cuda')
        if not input.is_cuda:
            input = input.to(device)

        # 确保模型在 GPU 上
        if not next(self.parameters()).is_cuda:
            for param in self.parameters():
                param.data = param.data.to(device)
            for buffer in self.buffers():
                buffer.data = buffer.data.to(device)

        # 初始状态处理
        h0_forward, h0_reverse = self._parse_initial_state(hx, batch_size, device, to_cuda=True)

        # 初始化校准收集器
        self._ensure_calibration_collector(hidden_size, reverse=False)
        quant_ranges, hist_collectors = self._get_calibration_collectors(reverse=False)

        # 准备权重
        W, R, bx, br = convert_weights_to_haste_format(
            self.weight_ih_l0, self.weight_hh_l0,
            self.bias_ih_l0 if self.bias else None,
            self.bias_hh_l0 if self.bias else None,
            self.hidden_size, device
        )

        # 前向传播 + 校准数据收集（原地累积）
        h, v = gru_ops.forward_calibrate(
            is_training=True,
            time_steps=time_steps, batch_size=batch_size,
            input_size=input_size, hidden_size=hidden_size,
            W=W, R=R, bx=bx, br=br, x=input,
            h0=h0_forward if h0_forward is not None else torch.empty(0, device=device),
            calib_method=self.calibration_method,
            quant_ranges=quant_ranges,
            hist_collectors=hist_collectors
        )

        # 提取输出
        output_forward = h[1:].contiguous()
        h_n_forward = h[-1:].unsqueeze(0) if h.dim() == 2 else h[-1:].contiguous()

        if self.bidirectional:
            # 初始化反向校准收集器
            self._ensure_calibration_collector(hidden_size, reverse=True)
            quant_ranges_rev, hist_collectors_rev = self._get_calibration_collectors(reverse=True)

            W_rev, R_rev, bx_rev, br_rev = convert_weights_to_haste_format(
                self.weight_ih_l0_reverse, self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse if self.bias else None,
                self.bias_hh_l0_reverse if self.bias else None,
                self.hidden_size, device
            )
            input_reversed = input.flip(0).contiguous()

            h_rev, v_rev = gru_ops.forward_calibrate(
                is_training=True,
                time_steps=time_steps, batch_size=batch_size,
                input_size=input_size, hidden_size=hidden_size,
                W=W_rev, R=R_rev, bx=bx_rev, br=br_rev, x=input_reversed,
                h0=h0_reverse if h0_reverse is not None else torch.empty(0, device=device),
                calib_method=self.calibration_method,
                quant_ranges=quant_ranges_rev,
                hist_collectors=hist_collectors_rev
            )

            output_reverse = h_rev[1:].flip(0).contiguous()
            h_n_reverse = h_rev[-1:].contiguous()
        else:
            output_reverse, h_n_reverse = None, None

        # 标记量化参数需要更新
        self._quant_params_dirty = True

        return self._combine_bidirectional_outputs(
            output_forward, h_n_forward, output_reverse, h_n_reverse
        )

    # -------------------- 主 forward 方法 --------------------

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
            - export_mode=False (默认): 使用 CUDA C++ 实现(高性能)
            - export_mode=True: 使用纯 PyTorch 实现(可被 ONNX 追踪)
        """
        # ===== 统一处理 batch_first 输入转换(唯一入口)=====
        if self.batch_first:
            input = input.transpose(0, 1).contiguous()

        # ===== 根据模式选择执行路径 =====
        if self.export_mode:
            # ONNX 导出模式：使用纯 PyTorch 实现
            output, h_n = self._forward_python(input, hx)
        elif self.calibrating:
            # 校准模式：在 forward 过程中收集校准数据
            self._ensure_cublas_initialized()
            output, h_n = self._forward_with_calibration(input, hx)
        else:
            # 正常/量化推理模式：使用 CUDA C++ 实现
            self._ensure_cublas_initialized()
            output, h_n = self._forward_cuda(input, hx)

        # ===== 统一处理 batch_first 输出转换(唯一出口)=====
        if self.batch_first:
            output = output.transpose(0, 1).contiguous()

        return output, h_n

    def _forward_cuda(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CUDA C++ 实现的前向传播(正常/量化推理模式)
        
        Note: batch_first 转换已在 forward() 中统一处理
        """
        # 量化模式下检查校准状态
        if self.use_quantization:
            if self._quant_params_dirty:
                # 校准数据已更新或配置已修改，需要重新计算量化参数
                self.finalize_calibration()
            elif not self.is_calibrated():
                # 检查是否有未完成的校准数据(支持 minmax/histogram/percentile)
                if self.quant_ranges is not None or self.hist_collectors is not None:
                    # 已累积数据但未完成校准，自动调用 finalize
                    self.finalize_calibration()
                else:
                    raise RuntimeError(
                        "量化已启用但未校准。请先进行校准：\n"
                        "  1. gru.calibrating = True\n"
                        "  2. gru(calibration_data)\n"
                        "  3. gru.calibrating = False\n"
                        "注意：pickle/deepcopy 后校准数据会丢失，需要重新校准。"
                    )

        seq_len, batch_size, input_size = input.shape

        device = input.device if input.is_cuda else torch.device('cuda')
        input = ensure_cuda_float32(input, device)

        # 初始状态处理(统一接口)
        h0_forward, h0_reverse = self._parse_initial_state(hx, batch_size, device, to_cuda=True)

        # 前向方向
        # LUT 现在存储在 quant_params 中，通过 setRescaleParam 复制到 QuantGRUReScale
        output_forward, h_n_forward = GRUFunction.apply(
            input, self.weight_ih_l0, self.weight_hh_l0,
            self.bias_ih_l0 if self.bias else None,
            self.bias_hh_l0 if self.bias else None,
            h0_forward, self.training, self.use_quantization, self.quant_params)

        # 反向方向(双向时)
        output_reverse, h_n_reverse = None, None
        if self.bidirectional:
            # LUT 存储在 quant_params_reverse 中
            output_reverse, h_n_reverse = GRUFunction.apply(
                input.flip(0), self.weight_ih_l0_reverse, self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse if self.bias else None,
                self.bias_hh_l0_reverse if self.bias else None,
                h0_reverse, self.training, self.use_quantization, self.quant_params_reverse)
            # 反转反向输出以对齐时间步
            output_reverse = output_reverse.flip(0)

        # 合并双向输出(统一接口)
        return self._combine_bidirectional_outputs(
            output_forward, h_n_forward, output_reverse, h_n_reverse
        )

    # -------------------- 量化参数导出/导入/调整 --------------------

    def export_quant_params(
        self,
        export_path: str,
        include_weights: bool = False,
        verbose: bool = False
    ) -> None:
        """
        导出量化参数到 JSON 文件
        
        每个算子的所有信息（bitwidth、symmetric、scale、zp）放在一起
        
        Args:
            export_path: 导出文件路径（.json）
            include_weights: 是否包含量化后的权重（默认 False）
            verbose: 是否打印详情
            
        Example:
            >>> gru.export_quant_params("quant_params.json", verbose=True)
        """
        # 调用模块级实现
        _export_quant_params_impl(self, export_path, include_weights, verbose)

    def load_quant_params(self, import_path: str, verbose: bool = False) -> None:
        """
        从 JSON 文件加载量化参数
        
        加载后 is_calibrated() 返回 True，可直接进行量化推理。
        
        Args:
            import_path: JSON 文件路径
            verbose: 是否打印详情
            
        Example:
            >>> gru.load_quant_params("quant_params.json", verbose=True)
            >>> gru.use_quantization = True
            >>> output = gru(x)
        """
        # 调用模块级实现
        _load_quant_params_impl(self, import_path, verbose)

    def adjust_quant_config(
        self,
        operator: str,
        bitwidth: int = None,
        is_symmetric: bool = None,
        exp2_inv: int = None,
        zero_point: int = None,
        verbose: bool = False
    ) -> None:
        """
        手动调整指定算子的量化配置
        
        Args:
            operator: 算子名称 ("x", "h", "W", "z_out" 等)
            bitwidth: 新的位宽 (1-32)
            is_symmetric: 是否对称量化
            exp2_inv: 量化指数，None 表示自动计算
            zero_point: 零点
            verbose: 是否打印详情
            
        Example:
            >>> gru.adjust_quant_config("z_out", bitwidth=16, verbose=True)
        """
        # 调用模块级实现
        _adjust_quant_config_impl(self, operator, bitwidth, is_symmetric, exp2_inv, zero_point, verbose)

    def get_quant_config(self, operator: str = None) -> dict:
        """
        获取量化配置信息
        
        Args:
            operator: 算子名称，None 表示获取所有算子的配置
            
        Returns:
            配置字典
            
        Example:
            >>> config = gru.get_quant_config("z_out")
            >>> print(config)
        """
        return _get_quant_config_impl(self, operator)


# ============================================================
#                      调试与诊断工具（模块级函数）
# ============================================================
#
# 以下函数用于调试和诊断量化模型，便于快速查看模型状态。
# 
# 使用示例：
#   >>> from quant_gru import print_quant_params, print_quant_config
#   >>> print_quant_params(gru)   # 打印量化参数（需已校准）
#   >>> print_quant_ranges(gru)   # 打印校准收集的数值范围
#   >>> print_quant_config(gru)   # 打印量化配置详情
#   >>> print_bitwidth_config(gru._bitwidth_config)  # 打印位宽配置

def print_quant_params(gru: 'QuantGRU'):
    """
    打印 QuantGRU 的量化参数（scale/zero_point）
    
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


def print_quant_ranges(gru: 'QuantGRU'):
    """
    打印 QuantGRU 的量化范围
    
    Args:
        gru: 已完成校准的 QuantGRU 实例（calibrating=True 后调用过 forward）
    """
    if gru.quant_ranges is None:
        raise RuntimeError("请先设置 calibrating=True 并调用 forward()")

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


# ============================================================
#                   量化参数导出/导入（内部实现）
# ============================================================
#
# 以下为 QuantGRU 类方法的内部实现函数：
#   - gru.export_quant_params(path): 导出量化参数到 JSON 文件
#   - gru.load_quant_params(path):   从 JSON 文件加载量化参数
#   - gru.adjust_quant_config(op):   调整单个算子的量化配置
#   - gru.get_quant_config(op):      获取算子的量化配置
#
# JSON 格式说明（AIMET 兼容）：
#   - operators: 各算子的量化参数
#     - dtype: 数据类型 (INT8/UINT8/INT16 等)
#     - symmetric: 是否对称量化
#     - scale: 浮点 scale (= 2^(-n))
#     - zero_point: 零点
#     - n: power-of-2 指数 (exp2_inv)
#   - model_info: 模型元信息

def _exp2_inv_to_scale(exp2_inv: int) -> float:
    """
    将 power-of-2 指数转换为浮点 scale
    
    exp2_inv -> scale = 2^(-exp2_inv)
    例如: exp2_inv=7 -> scale=0.0078125 (1/128)
    """
    if exp2_inv >= 0:
        return 1.0 / (1 << exp2_inv)
    else:
        return float(1 << (-exp2_inv))


def _scale_to_exp2_inv(scale: float) -> int:
    """
    将浮点 scale 转换为最接近的 power-of-2 指数
    
    scale -> exp2_inv = -log2(scale)
    """
    import math
    if scale <= 0:
        return 0
    return int(round(-math.log2(scale)))


def _bitwidth_to_dtype(bitwidth: int, is_unsigned: bool = False) -> str:
    """将位宽转换为 AIMET 风格的 dtype 字符串"""
    prefix = "UINT" if is_unsigned else "INT"
    return f"{prefix}{bitwidth}"


def _build_operators_dict(bitwidth_config, quant_params) -> dict:
    """
    构建统一的 operators 字典（AIMET 兼容格式）
    
    Args:
        bitwidth_config: OperatorQuantConfig 对象
        quant_params: GRUQuantitativeParameters 对象
        
    Returns:
        operators 字典（per-channel 权重放在最后）
        
    输出字段顺序（AIMET 风格）:
        1. dtype: "INT8" 等
        2. symmetric: true/false
        3. scale: 浮点数或列表
        4. zero_point: 整数
        5. real_min: 量化表示的最小实际值
        6. real_max: 量化表示的最大实际值
        7. enc_type: "PER_TENSOR" 或 "PER_CHANNEL"
        8. n: exp2_inv 指数（scale = 2^(-n)）
    """
    operators = {}
    per_channel_ops = {}  # 存放 per-channel 算子，最后再添加
    
    for op_name, op_info in _OPERATOR_MAP.items():
        bitwidth = getattr(bitwidth_config, op_info["bw_attr"])
        is_symmetric = getattr(bitwidth_config, op_info["sym_attr"])
        is_per_channel = op_info["is_per_channel"]
        
        # 读取 is_unsigned（只标记 UINT 例外）
        unsigned_attr = op_info.get("unsigned_attr")
        if unsigned_attr and hasattr(bitwidth_config, unsigned_attr):
            is_unsigned = getattr(bitwidth_config, unsigned_attr)
        else:
            is_unsigned = False  # 默认有符号
        
        # 计算 qmin, qmax（复用 get_quant_range 函数）
        qmin, qmax = get_quant_range(bitwidth, is_unsigned)
        
        # 按 AIMET 字段顺序构建 op_data
        op_data = {}
        
        # 1. dtype
        op_data["dtype"] = _bitwidth_to_dtype(bitwidth, is_unsigned=is_unsigned)
        
        # 2. symmetric
        op_data["symmetric"] = is_symmetric
        
        # 3. scale
        scale_value = None
        n_value = None
        if hasattr(quant_params, op_info["exp2_inv_attr"]):
            exp2_inv_value = getattr(quant_params, op_info["exp2_inv_attr"])
            if is_per_channel:
                exp2_inv_list = list(exp2_inv_value)
                scale_value = [_exp2_inv_to_scale(e) for e in exp2_inv_list]
                n_value = exp2_inv_list
            else:
                n = int(exp2_inv_value)
                scale_value = _exp2_inv_to_scale(n)
                n_value = n
        if scale_value is not None:
            op_data["scale"] = scale_value
        
        # 4. zero_point
        zp_value = 0
        if op_info["zp_attr"] and hasattr(quant_params, op_info["zp_attr"]):
            zp_value = int(getattr(quant_params, op_info["zp_attr"]))
        op_data["zero_point"] = zp_value
        
        # 5. real_min, 6. real_max: scale * (q - zero_point)
        if scale_value is not None:
            if is_per_channel:
                # per-channel: 每个 channel 一个 real_min/real_max
                op_data["real_min"] = [s * (qmin - zp_value) for s in scale_value]
                op_data["real_max"] = [s * (qmax - zp_value) for s in scale_value]
            else:
                op_data["real_min"] = scale_value * (qmin - zp_value)
                op_data["real_max"] = scale_value * (qmax - zp_value)
        
        # 7. enc_type
        op_data["enc_type"] = "PER_CHANNEL" if is_per_channel else "PER_TENSOR"
        
        # 8. n (exp2_inv 指数)
        if n_value is not None:
            op_data["n"] = n_value
        
        # per-channel 放到最后
        if is_per_channel:
            per_channel_ops[op_name] = op_data
        else:
            operators[op_name] = op_data
    
    # 添加 per-channel 算子到最后
    operators.update(per_channel_ops)
    
    return operators


def _dtype_to_bitwidth(dtype: str) -> int:
    """从 AIMET 风格的 dtype 字符串解析位宽（如 "INT8" → 8）"""
    import re
    match = re.search(r'(\d+)', dtype)
    if match:
        return int(match.group(1))
    return 8  # 默认值


def _dtype_to_is_unsigned(dtype: str) -> bool:
    """从 AIMET 风格的 dtype 字符串解析是否无符号（如 "UINT8" → True, "INT8" → False）"""
    return dtype.upper().startswith("UINT")


def _parse_operators_dict(operators: dict, bitwidth_config, quant_params) -> None:
    """
    从 operators 字典解析并设置 bitwidth_config 和 quant_params
    
    AIMET 风格字段名：
        - dtype: "INT8" 等 → 位宽
        - symmetric: true/false → 对称量化
        - n: 量化指数（优先）
        - scale: 浮点 scale（当无 n 时使用）
        - zero_point: 零点
    
    Args:
        operators: operators 字典
        bitwidth_config: OperatorQuantConfig 对象（会被修改）
        quant_params: GRUQuantitativeParameters 对象（会被修改）
    """
    for op_name, op_data in operators.items():
        if op_name not in _OPERATOR_MAP:
            continue
            
        op_info = _OPERATOR_MAP[op_name]
        
        # 设置 bitwidth 和 is_unsigned（从 dtype 解析，如 "INT8" → 8, False；"UINT8" → 8, True）
        if "dtype" in op_data:
            dtype_str = op_data["dtype"]
            setattr(bitwidth_config, op_info["bw_attr"], _dtype_to_bitwidth(dtype_str))
            unsigned_attr = op_info.get("unsigned_attr")
            if unsigned_attr:
                setattr(bitwidth_config, unsigned_attr, _dtype_to_is_unsigned(dtype_str))
        
        # 设置 symmetric
        if "symmetric" in op_data:
            setattr(bitwidth_config, op_info["sym_attr"], op_data["symmetric"])
        
        # 设置 n / scale（优先使用 n，其次从 scale 计算）
        if "n" in op_data:
            value = op_data["n"]
            if op_info["is_per_channel"]:
                setattr(quant_params, op_info["exp2_inv_attr"], list(value))
            else:
                setattr(quant_params, op_info["exp2_inv_attr"], int(value))
        elif "scale" in op_data:
            value = op_data["scale"]
            if op_info["is_per_channel"]:
                setattr(quant_params, op_info["exp2_inv_attr"], 
                        [_scale_to_exp2_inv(v) for v in value])
            else:
                setattr(quant_params, op_info["exp2_inv_attr"], _scale_to_exp2_inv(value))
        
        # 设置 zero_point
        if op_info["zp_attr"] and "zero_point" in op_data:
            setattr(quant_params, op_info["zp_attr"], int(op_data["zero_point"]))


def _export_quant_params_impl(
    gru: 'QuantGRU',
    export_path: str,
    include_weights: bool = False,
    verbose: bool = False
) -> None:
    """导出量化参数到 JSON 文件（内部实现）"""
    if not gru.is_calibrated():
        raise RuntimeError(
            "请先完成校准。使用方法：\n"
            "  1. gru.calibrating = True\n"
            "  2. gru(calibration_data)\n"
            "  3. gru.calibrating = False\n"
            "  4. gru.finalize_calibration()"
        )
    
    # 构建导出数据（统一的 operators 结构）
    export_data = {
        "model_info": {
            "input_size": gru.input_size,
            "hidden_size": gru.hidden_size,
            "bias": gru.bias,
            "batch_first": gru.batch_first,
            "bidirectional": gru.bidirectional,
        },
        "operators": _build_operators_dict(gru._bitwidth_config, gru.quant_params),
    }
    
    # 双向 GRU 导出反向参数
    if gru.bidirectional and gru.quant_params_reverse is not None:
        export_data["operators_reverse"] = _build_operators_dict(
            gru._bitwidth_config, gru.quant_params_reverse
        )
    
    # 可选：导出量化后的权重
    if include_weights:
        export_data["quantized_weights"] = _export_quantized_weights(gru)
    
    # 写入文件
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n[QuantGRU] 量化参数已导出到: {export_path}")
        print(f"  - 模型配置: input_size={gru.input_size}, hidden_size={gru.hidden_size}")
        print(f"  - 双向: {gru.bidirectional}")
        if include_weights:
            print(f"  - 包含量化权重: 是")


def _export_quantized_weights(gru: QuantGRU) -> dict:
    """
    导出量化后的权重（内部函数）
    
    将权重转换为 Haste 格式并应用 per-channel 量化参数，
    返回量化后的整数权重。
    """
    device = next(gru.parameters()).device
    params = gru.quant_params
    
    # 转换为 Haste 格式
    W, R, bx, br = convert_weights_to_haste_format(
        gru.weight_ih_l0, gru.weight_hh_l0,
        gru.bias_ih_l0 if gru.bias else None,
        gru.bias_hh_l0 if gru.bias else None,
        gru.hidden_size, device
    )
    
    # 量化权重（使用 per-channel 参数）
    def quantize_per_channel(tensor, exp2_inv_list, bitwidth, symmetric):
        """对每个 channel 应用量化（权重使用有符号量化）"""
        qmin, qmax = get_quant_range(bitwidth)  # 权重使用有符号量化（默认）
        result = torch.zeros_like(tensor, dtype=torch.int32)
        
        for c in range(len(exp2_inv_list)):
            exp2_inv = exp2_inv_list[c]
            if exp2_inv >= 0:
                scale = 1.0 / (1 << exp2_inv)
            else:
                scale = float(1 << (-exp2_inv))
            
            q = torch.clamp(torch.round(tensor[:, c] / scale), qmin, qmax)
            result[:, c] = q.int()
        
        return result.tolist()
    
    # 获取位宽配置
    W_bitwidth = gru._bitwidth_config.W_
    R_bitwidth = gru._bitwidth_config.R_
    bx_bitwidth = gru._bitwidth_config.bx_
    br_bitwidth = gru._bitwidth_config.br_
    
    weights = {
        "W": quantize_per_channel(W, list(params.exp2_inv_W_), W_bitwidth, True),
        "R": quantize_per_channel(R, list(params.exp2_inv_R_), R_bitwidth, True),
    }
    
    if gru.bias:
        # 偏置是 1D，需要 unsqueeze
        bx_2d = bx.unsqueeze(0)  # [1, 3*H]
        br_2d = br.unsqueeze(0)
        weights["bx"] = quantize_per_channel(bx_2d, list(params.exp2_inv_bx_), bx_bitwidth, True)[0]
        weights["br"] = quantize_per_channel(br_2d, list(params.exp2_inv_br_), br_bitwidth, True)[0]
    
    return weights


def _load_quant_params_impl(
    gru: 'QuantGRU',
    import_path: str,
    verbose: bool = False
) -> None:
    """从 JSON 文件加载量化参数（内部实现）"""
    with open(import_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证模型配置
    model_info = data.get("model_info", {})
    if model_info.get("input_size") != gru.input_size:
        raise ValueError(
            f"input_size 不匹配: 文件中为 {model_info.get('input_size')}, "
            f"模型为 {gru.input_size}"
        )
    if model_info.get("hidden_size") != gru.hidden_size:
        raise ValueError(
            f"hidden_size 不匹配: 文件中为 {model_info.get('hidden_size')}, "
            f"模型为 {gru.hidden_size}"
        )
    if model_info.get("bidirectional", False) != gru.bidirectional:
        raise ValueError(
            f"bidirectional 不匹配: 文件中为 {model_info.get('bidirectional')}, "
            f"模型为 {gru.bidirectional}"
        )
    # batch_first 不影响量化参数，但不一致可能导致用户困惑，给出警告
    if model_info.get("batch_first", False) != gru.batch_first:
        import warnings
        warnings.warn(
            f"batch_first 不匹配: 文件中为 {model_info.get('batch_first')}, "
            f"模型为 {gru.batch_first}。这不影响量化参数，但请确保输入数据格式正确。",
            UserWarning
        )
    
    # 验证版本和格式
    if "operators" not in data:
        raise ValueError(
            f"不支持的量化参数格式，缺少 'operators' 字段。\n"
            f"当前包含字段: {list(data.keys())}"
        )
    
    # 解析 operators 字典
    gru._bitwidth_config = gru_ops.OperatorQuantConfig()
    gru.quant_params = gru_ops.GRUQuantitativeParameters()
    gru.quant_params.hidden_ = gru.hidden_size
    
    _parse_operators_dict(data["operators"], gru._bitwidth_config, gru.quant_params)
    
    # 设置位宽配置到量化参数中（pybind11 值复制语义，这里是复制而非引用）
    gru.quant_params.bitwidth_config_ = gru._bitwidth_config
    
    # 注意：LUT 会在 forward 时通过 to_cpp() 自动生成，无需在此显式调用
    
    # 加载反向参数（双向）
    # 设计说明：正向和反向共用同一个 _bitwidth_config，这是有意为之：
    #   1. 硬件实现：正向/反向 GRU 计算单元通常时分复用，相同位宽简化硬件设计
    #   2. 模型对称性：双向 GRU 的正反向是对称结构，应使用对称的量化配置
    #   3. 导出时 operators 和 operators_reverse 的 bitwidth 来自同一 _bitwidth_config，
    #      所以导入时解析两次 bitwidth 是等价的（值相同）
    if gru.bidirectional and "operators_reverse" in data:
        gru.quant_params_reverse = gru_ops.GRUQuantitativeParameters()
        gru.quant_params_reverse.hidden_ = gru.hidden_size
        
        # 从 operators_reverse 解析 exp2_inv/zp，bitwidth 与正向相同（共用 _bitwidth_config）
        _parse_operators_dict(data["operators_reverse"], gru._bitwidth_config, gru.quant_params_reverse)
        gru.quant_params_reverse.bitwidth_config_ = gru._bitwidth_config
    
    # 清除脏标志
    gru._quant_params_dirty = False
    
    if verbose:
        print(f"\n[QuantGRU] 量化参数已从 {import_path} 加载")
        print(f"  - 模型配置: input_size={gru.input_size}, hidden_size={gru.hidden_size}")
        print(f"  - 双向: {gru.bidirectional}")
        print(f"  - is_calibrated(): {gru.is_calibrated()}")


def _adjust_quant_config_impl(
    gru: 'QuantGRU',
    operator: str,
    bitwidth: int = None,
    is_symmetric: bool = None,
    exp2_inv: int = None,
    zero_point: int = None,
    verbose: bool = False
) -> None:
    """手动调整指定算子的量化配置（内部实现）"""
    # 验证算子名称（使用模块级常量）
    if operator not in _OPERATOR_SHORT_NAME_MAP:
        valid_ops = sorted(_OPERATOR_SHORT_NAME_MAP.keys())
        raise ValueError(
            f"无效的算子名称: '{operator}'。\n"
            f"有效的算子名称: {valid_ops}"
        )
    
    attrs = _OPERATOR_SHORT_NAME_MAP[operator]
    old_values = {}
    new_values = {}
    
    # 获取旧值
    old_bitwidth = getattr(gru._bitwidth_config, attrs['bw_attr'])
    old_symmetric = getattr(gru._bitwidth_config, attrs['sym_attr'])
    old_values['bitwidth'] = old_bitwidth
    old_values['is_symmetric'] = old_symmetric
    
    # 修改位宽配置
    if bitwidth is not None:
        if not (1 <= bitwidth <= 32):
            raise ValueError(f"bitwidth 必须在 [1, 32] 范围内，当前值: {bitwidth}")
        setattr(gru._bitwidth_config, attrs['bw_attr'], bitwidth)
        new_values['bitwidth'] = bitwidth
        
        # 自动计算 exp2_inv 和 zp（当 exp2_inv 未指定时）
        # 
        # 原理：保持相同的数据表示范围，但用更多/更少的量化级别
        # - scale = 2^(-exp2_inv) 
        # - 位宽增加 -> 量化级别增多 -> scale 应减小 -> exp2_inv 应增大
        # - 公式: new_exp2_inv = old_exp2_inv + (new_bitwidth - old_bitwidth)
        #
        # 对于 zp:
        # - 对称量化: zp = 0（固定不变）
        # - 非对称量化: zp_new ≈ zp_old * 2^delta_bits
        #
        if exp2_inv is None and gru.quant_params is not None:
            exp2_attr = f"exp2_inv_{operator}_"
            zp_attr = f"zp_{operator}_"
            is_per_channel = operator in ['W', 'R', 'bx', 'br']
            is_symmetric = old_symmetric  # 使用当前的对称性设置
            
            bitwidth_delta = bitwidth - old_bitwidth
            scale_factor = 1 << abs(bitwidth_delta)  # 2^|delta|
            
            if hasattr(gru.quant_params, exp2_attr):
                if is_per_channel:
                    old_exp2_list = list(getattr(gru.quant_params, exp2_attr))
                    # exp2_inv 增加 delta_bits（scale 减小）
                    new_exp2_list = [max(-32, min(32, e + bitwidth_delta)) for e in old_exp2_list]
                    setattr(gru.quant_params, exp2_attr, new_exp2_list)
                    new_values['exp2_inv'] = f"auto: [{new_exp2_list[0]}, ...] (delta=+{bitwidth_delta})"
                else:
                    old_exp2 = int(getattr(gru.quant_params, exp2_attr))
                    # exp2_inv 增加 delta_bits
                    new_exp2 = max(-32, min(32, old_exp2 + bitwidth_delta))
                    setattr(gru.quant_params, exp2_attr, new_exp2)
                    new_values['exp2_inv'] = f"auto: {new_exp2} (was {old_exp2}, delta=+{bitwidth_delta})"
                    
                    # 同时调整 zp（非 per-channel 情况）
                    if hasattr(gru.quant_params, zp_attr):
                        old_zp = int(getattr(gru.quant_params, zp_attr))
                        
                        if is_symmetric:
                            # 对称量化：zp 恒为 0
                            new_zp = 0
                            if old_zp != 0:
                                new_values['zero_point'] = f"auto: 0 (对称量化, 强制为0)"
                            else:
                                new_values['zero_point'] = f"auto: 0 (对称量化)"
                        else:
                            # 非对称量化：zp 按比例缩放
                            if bitwidth_delta > 0:
                                new_zp = old_zp * scale_factor
                            else:
                                new_zp = old_zp // scale_factor
                            new_values['zero_point'] = f"auto: {new_zp} (was {old_zp}, x{scale_factor if bitwidth_delta > 0 else '/' + str(scale_factor)})"
                        
                        setattr(gru.quant_params, zp_attr, new_zp)
    
    if is_symmetric is not None:
        setattr(gru._bitwidth_config, attrs['sym_attr'], is_symmetric)
        new_values['is_symmetric'] = is_symmetric
    
    # 修改量化参数（如果已校准且未被 auto_scale 处理）
    if gru.quant_params is not None:
        # 确定 exp2_inv 和 zp 的属性名
        exp2_attr = f"exp2_inv_{operator}_"
        zp_attr = f"zp_{operator}_"
        
        # 检查是否是 per-channel 参数
        is_per_channel = operator in ['W', 'R', 'bx', 'br']
        
        if is_per_channel:
            # per-channel 参数只有 exp2_inv，没有 zp
            if hasattr(gru.quant_params, exp2_attr):
                old_exp2 = list(getattr(gru.quant_params, exp2_attr))
                old_values['exp2_inv'] = f"[{old_exp2[0]}, {old_exp2[1]}, ...] (per-channel)"
                
                if exp2_inv is not None:
                    # 将所有 channel 设置为相同的值
                    new_exp2 = [exp2_inv] * len(old_exp2)
                    setattr(gru.quant_params, exp2_attr, new_exp2)
                    new_values['exp2_inv'] = f"[{exp2_inv}, {exp2_inv}, ...] (all channels)"
        else:
            # 标量参数
            if hasattr(gru.quant_params, exp2_attr):
                old_values['exp2_inv'] = int(getattr(gru.quant_params, exp2_attr))
                if exp2_inv is not None:
                    setattr(gru.quant_params, exp2_attr, exp2_inv)
                    new_values['exp2_inv'] = exp2_inv
            
            if hasattr(gru.quant_params, zp_attr):
                old_values['zero_point'] = int(getattr(gru.quant_params, zp_attr))
                if zero_point is not None:
                    setattr(gru.quant_params, zp_attr, zero_point)
                    new_values['zero_point'] = zero_point
    
    # 同步更新 quant_params 中的 bitwidth_config
    if gru.quant_params is not None:
        gru.quant_params.bitwidth_config_ = gru._bitwidth_config
    
    if verbose:
        print(f"\n[QuantGRU] 调整算子 '{operator}' 配置:")
        print(f"  修改前: {old_values}")
        print(f"  修改后: {new_values if new_values else '(无修改)'}")


def _get_quant_config_impl(gru: 'QuantGRU', operator: str = None) -> dict:
    """获取量化配置信息（内部实现）"""
    def get_single_config(op_name):
        if op_name not in _OPERATOR_SHORT_NAME_MAP:
            raise ValueError(f"无效的算子名称: '{op_name}'")
        
        attrs = _OPERATOR_SHORT_NAME_MAP[op_name]
        config = {
            'bitwidth': getattr(gru._bitwidth_config, attrs['bw_attr']),
            'is_symmetric': getattr(gru._bitwidth_config, attrs['sym_attr']),
        }
        
        if gru.quant_params is not None:
            exp2_attr = f"exp2_inv_{op_name}_"
            zp_attr = f"zp_{op_name}_"
            
            is_per_channel = op_name in ['W', 'R', 'bx', 'br']
            
            if is_per_channel:
                if hasattr(gru.quant_params, exp2_attr):
                    config['exp2_inv'] = list(getattr(gru.quant_params, exp2_attr))
                    # 计算对应的 scale
                    config['scale'] = [_exp2_inv_to_scale(e) for e in config['exp2_inv']]
            else:
                if hasattr(gru.quant_params, exp2_attr):
                    exp2 = int(getattr(gru.quant_params, exp2_attr))
                    config['exp2_inv'] = exp2
                    config['scale'] = _exp2_inv_to_scale(exp2)
                if hasattr(gru.quant_params, zp_attr):
                    config['zero_point'] = int(getattr(gru.quant_params, zp_attr))
        
        return config
    
    if operator is not None:
        return get_single_config(operator)
    else:
        return {op: get_single_config(op) for op in _OPERATOR_SHORT_NAME_MAP}


# ============================================================
#                      调试打印函数（续）
# ============================================================
#
# 以下函数依赖 _get_quant_config_impl，故放在其后。

def print_quant_config(gru: 'QuantGRU', operators: list = None):
    """
    打印量化配置（便于查看和调整）
    
    Args:
        gru: QuantGRU 实例
        operators: 要打印的算子列表，None 表示全部打印
        
    Example:
        >>> print_quant_config(gru)  # 打印所有
        >>> print_quant_config(gru, ["x", "h", "z_out"])  # 只打印指定算子
    """
    all_config = _get_quant_config_impl(gru)
    
    if operators is not None:
        config = {k: v for k, v in all_config.items() if k in operators}
    else:
        config = all_config
    
    # 分组显示
    groups = {
        '输入': ['x'],
        '输出': ['h'],
        '权重': ['W', 'R', 'bx', 'br'],
        'GEMM': ['Wx', 'Rh'],
        '门控(pre)': ['z_pre', 'r_pre', 'g_pre'],
        '门控(out)': ['z_out', 'r_out', 'g_out'],
        '中间': ['Rh_add_br', 'rRh', 'old_contrib', 'new_contrib'],
    }
    
    print("\n" + "=" * 80)
    print("GRU 量化配置详情")
    print("=" * 80)
    
    for group_name, op_list in groups.items():
        ops_in_group = [op for op in op_list if op in config]
        if not ops_in_group:
            continue
        
        print(f"\n[{group_name}]")
        print("-" * 80)
        
        for op in ops_in_group:
            cfg = config[op]
            bw = cfg.get('bitwidth', '?')
            sym = "对称" if cfg.get('is_symmetric', True) else "非对称"
            
            if 'exp2_inv' in cfg:
                exp2 = cfg['exp2_inv']
                if isinstance(exp2, list):
                    # per-channel
                    exp2_str = f"[{exp2[0]}, {exp2[1]}, ...] (per-channel, len={len(exp2)})"
                    scale_str = f"[{cfg['scale'][0]:.6f}, ...]"
                else:
                    exp2_str = str(exp2)
                    scale_str = f"{cfg.get('scale', '?'):.6f}"
            else:
                exp2_str = "N/A"
                scale_str = "N/A"
            
            zp = cfg.get('zero_point', 'N/A')
            
            print(f"  {op:15s}: {bw:2}bit, {sym:4s}, exp2_inv={exp2_str:30s}, scale={scale_str}, zp={zp}")
    
    print("=" * 80)
    print("\n💡 使用 gru.adjust_quant_config('x', bitwidth=16) 可调整配置")


def _format_bitwidth(val: int) -> str:
    """格式化位宽值: 8 -> '8bit'"""
    return f"{abs(val)}bit"


def _format_symmetric(is_symmetric: bool) -> str:
    """格式化对称量化: True -> '对称'"""
    return "对称" if is_symmetric else "非对称"


def print_bitwidth_config(config: gru_ops.OperatorQuantConfig,
                          config_file: str = None,
                          verbose: bool = True) -> None:
    """
    打印 OperatorQuantConfig 的配置详情
    
    Args:
        config: 配置对象
        config_file: 配置文件路径(可选，仅用于显示来源)
        verbose: 是否打印详情(默认 True)
    """
    if not verbose:
        return

    print("\n" + "=" * 70)
    print("🔧 GRU 量化配置(位宽 + 对称量化)")
    print("=" * 70)
    if config_file:
        print(f"📄 配置来源: {config_file}")
    print("-" * 70)
    print(f"  [输入]  x: {_format_bitwidth(config.x_):6s} ({_format_symmetric(config.x_symmetric_)})")
    print(f"  [输出]  h: {_format_bitwidth(config.h_):6s} ({_format_symmetric(config.h_symmetric_)})")
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
