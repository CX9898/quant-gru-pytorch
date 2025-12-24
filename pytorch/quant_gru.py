"""
QuantGRU - 支持量化的 GRU 实现

功能特性:
    - 兼容 nn.GRU 接口（支持 batch_first、bidirectional 等参数）
    - 支持 INT8/INT16/INT32 量化推理
    - 支持 MinMax 和 AIMET 风格直方图校准
    - 延迟初始化设计，支持 pickle/deepcopy 序列化
    - 支持 ONNX 导出（使用纯 PyTorch 实现）
    - 量化模式下使用纯定点计算，与 CUDA 实现完全一致

关键属性:
    - use_quantization: 是否启用量化（默认 False）
    - export_mode: 是否使用 ONNX 导出模式（默认 False）
    - export_format: 导出格式 'float'|'qdq'|'fixedpoint'（高级选项，默认 'float'）

典型用法:
    >>> from quant_gru import QuantGRU
    >>>
    >>> # 创建并校准模型
    >>> gru = QuantGRU(64, 128, batch_first=True).cuda()
    >>> gru.calibrate(calibration_data)
    >>> gru.use_quantization = True
    >>>
    >>> # 正常推理（CUDA 量化模式）
    >>> output = gru(x)
    
ONNX 导出:
    >>> # 启用导出模式（默认使用浮点格式）
    >>> gru.export_mode = True
    >>> torch.onnx.export(gru, x, "model.onnx")
    >>> gru.export_mode = False  # 恢复
    >>> 
    >>> # 量化模型导出需指定格式
    >>> gru.export_format = 'qdq'  # 'float' | 'qdq' | 'fixedpoint'
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
#                      定点运算辅助函数
# ============================================================

def rshift_round(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    带四舍五入的右移操作（与 CUDA 实现一致）
    
    Args:
        x: 输入张量（整数类型）
        n: 移位位数（可以为负，表示左移）
        
    Returns:
        移位后的张量
    """
    if n <= 0:
        return x * (1 << (-n))  # 左移
    
    # 右移带四舍五入
    offset = 1 << (n - 1)
    # 处理正数和负数
    positive_mask = x >= 0
    result = torch.where(
        positive_mask,
        (x + offset) >> n,
        -(((-x) + offset) >> n)
    )
    return result


def rshift_round_i64(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    带四舍五入的右移操作（int64 版本，用于 16 位量化的乘积）
    """
    if n <= 0:
        return x * (1 << (-n))
    
    offset = 1 << (n - 1)
    positive_mask = x >= 0
    result = torch.where(
        positive_mask,
        (x + offset) >> n,
        -(((-x) + offset) >> n)
    )
    return result


def rshift_round_per_channel(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """
    Per-channel 带四舍五入的右移操作（ONNX 可导出）
    
    Args:
        x: 输入张量 [B, C] 或 [B*T, C]，int64 类型
        n: 每个 channel 的移位量 [C]，int8 类型
        
    Returns:
        移位后的张量
    """
    # 将 n 扩展为与 x 相同维度 [1, C] 便于广播
    n_expanded = n.unsqueeze(0).to(torch.int64)  # [1, C]
    
    # 计算 offset = 2^(n-1)，对于 n <= 0 设为 0
    # 使用 clamp 确保 n-1 >= 0 时才计算 offset
    n_clamped = torch.clamp(n_expanded - 1, min=0)
    offset = (torch.ones_like(x) << n_clamped)
    # 对于 n <= 0，offset 应该为 0
    offset = torch.where(n_expanded > 0, offset, torch.zeros_like(offset))
    
    # 计算右移或左移
    # 右移：(x + offset) >> n 或 -(-x + offset) >> n
    # 左移：x << (-n)
    positive_mask = x >= 0
    
    # 右移结果
    rshift_pos = (x + offset) >> n_expanded
    rshift_neg = -(((-x) + offset) >> n_expanded)
    rshift_result = torch.where(positive_mask, rshift_pos, rshift_neg)
    
    # 左移结果
    lshift_result = x << (-n_expanded)
    
    # 根据 n 的正负选择结果
    result = torch.where(n_expanded > 0, rshift_result, lshift_result)
    
    return result


def clamp_to_int8(x: torch.Tensor) -> torch.Tensor:
    """截断到 INT8 范围 [-128, 127]"""
    return torch.clamp(x, -128, 127).to(torch.int32)


def clamp_to_int16(x: torch.Tensor) -> torch.Tensor:
    """截断到 INT16 范围 [-32768, 32767]"""
    return torch.clamp(x, -32768, 32767).to(torch.int32)


def clamp_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """截断到 UINT8 范围 [0, 255]"""
    return torch.clamp(x, 0, 255).to(torch.int32)


def clamp_to_uint16(x: torch.Tensor) -> torch.Tensor:
    """截断到 UINT16 范围 [0, 65535]"""
    return torch.clamp(x, 0, 65535).to(torch.int32)


def quantize(x: torch.Tensor, exp2_inv: int, zp: int = 0, 
             bitwidth: int = 8, symmetric: bool = True) -> torch.Tensor:
    """
    量化张量（与 CUDA quantize<QuantT> 函数一致）
    
    量化公式: q = clamp(round(x / scale) + zp, qmin, qmax)
    其中 scale = 2^(-exp2_inv)
    
    Args:
        x: 浮点输入张量
        exp2_inv: scale = 2^(-exp2_inv)
        zp: zero point（默认0，对称量化）
        bitwidth: 目标位宽 (8, 16, 32)
        symmetric: 是否对称量化
        
    Returns:
        量化后的整数张量 (int32)
    """
    # 根据位宽确定量化范围
    if bitwidth == 8:
        qmin, qmax = (-128, 127) if symmetric else (0, 255)
    elif bitwidth == 16:
        qmin, qmax = (-32768, 32767) if symmetric else (0, 65535)
    else:  # INT32
        qmin, qmax = (-2147483648, 2147483647)
    
    if exp2_inv >= 0:
        scale = 1.0 / (1 << exp2_inv)
    else:
        scale = float(1 << (-exp2_inv))
    
    # q = round(x / scale) + zp
    q = torch.round(x / scale).to(torch.int32) + zp
    q = torch.clamp(q, qmin, qmax)
    
    return q


def dequantize(q: torch.Tensor, exp2_inv: int, zp: int = 0) -> torch.Tensor:
    """
    反量化张量（与 CUDA dequantize 函数一致）
    
    反量化公式: x = (q - zp) * scale
    其中 scale = 2^(-exp2_inv)
    
    Args:
        q: 量化整数张量
        exp2_inv: scale = 2^(-exp2_inv)
        zp: zero point
        
    Returns:
        反量化后的浮点张量
    """
    v = q.to(torch.int32) - zp
    
    if exp2_inv >= 0:
        return v.float() / float(1 << exp2_inv)
    else:
        return v.float() * float(1 << (-exp2_inv))


def quantize_per_channel(x: torch.Tensor, exp2_invs: list, zp: int = 0,
                         bitwidth: int = 8, symmetric: bool = True) -> torch.Tensor:
    """
    Per-channel 量化（与 CUDA quantificationPerChannel 一致）
    
    Args:
        x: 浮点输入张量，shape [..., channel_size]
        exp2_invs: 每个 channel 的 exp2_inv 列表
        zp: zero point（默认0，对称量化）
        bitwidth: 目标位宽 (8, 16, 32)
        symmetric: 是否对称量化
        
    Returns:
        量化后的整数张量 (int32)
    """
    # 根据位宽确定量化范围
    if bitwidth == 8:
        qmin, qmax = (-128, 127) if symmetric else (0, 255)
    elif bitwidth == 16:
        qmin, qmax = (-32768, 32767) if symmetric else (0, 65535)
    else:  # INT32
        qmin, qmax = (-2147483648, 2147483647)
    
    device = x.device
    channel_size = len(exp2_invs)
    q = torch.zeros_like(x, dtype=torch.int32, device=device)
    
    for c in range(channel_size):
        exp2_inv = exp2_invs[c]
        if exp2_inv >= 0:
            scale = 1.0 / (1 << exp2_inv)
        else:
            scale = float(1 << (-exp2_inv))
        
        q[..., c] = torch.clamp(
            torch.round(x[..., c] / scale).to(torch.int32) + zp,
            qmin, qmax
        )
    
    return q


# ============================================================
#                      QDQ (Quantize-Dequantize) 辅助函数
#                      用于 ONNX 导出的伪量化操作
# ============================================================

def fake_quantize(x: torch.Tensor, exp2_inv: int, zp: int = 0,
                  bitwidth: int = 8, symmetric: bool = True) -> torch.Tensor:
    """
    伪量化（Fake Quantize）: 量化后立即反量化，保持浮点格式
    
    用于 ONNX 导出，推理引擎会识别 QDQ 模式并优化
    
    [与 CUDA 一致] 量化参数 (exp2_inv, zp) 与 CUDA 端完全一致
    [ONNX 兼容] 使用浮点运算模拟量化效果
    """
    # 计算 scale
    if exp2_inv >= 0:
        scale = 1.0 / (1 << exp2_inv)
    else:
        scale = float(1 << (-exp2_inv))
    
    # 确定量化范围
    if bitwidth == 8:
        qmin, qmax = (-128, 127) if symmetric else (0, 255)
    elif bitwidth == 16:
        qmin, qmax = (-32768, 32767) if symmetric else (0, 65535)
    else:
        qmin, qmax = (-2147483648, 2147483647)
    
    # 量化: q = clamp(round(x / scale) + zp, qmin, qmax)
    q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
    
    # 反量化: x' = (q - zp) * scale
    x_dequant = (q - zp) * scale
    
    return x_dequant


def fake_quantize_per_channel(x: torch.Tensor, exp2_invs: list, zp: int = 0,
                               bitwidth: int = 8, symmetric: bool = True) -> torch.Tensor:
    """
    Per-channel 伪量化
    
    [与 CUDA 一致] per-channel 量化参数与 CUDA quantificationPerChannel 一致
    [ONNX 兼容] 使用浮点运算模拟量化效果
    """
    if bitwidth == 8:
        qmin, qmax = (-128, 127) if symmetric else (0, 255)
    elif bitwidth == 16:
        qmin, qmax = (-32768, 32767) if symmetric else (0, 65535)
    else:
        qmin, qmax = (-2147483648, 2147483647)
    
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
#                      分段线性 LUT 实现
# ============================================================

class SegmentParams:
    """分段线性参数（与 CUDA SegmentParams_INT16/INT8 对应）"""
    def __init__(self, q_b: int, n_BX_total: int, term_c_precomputed: int, threshold: int):
        self.q_b = q_b                           # 量化斜率
        self.n_BX_total = n_BX_total             # 融合移位位数
        self.term_c_precomputed = term_c_precomputed  # 预计算常数项
        self.threshold = threshold               # 段阈值


class PiecewiseLUT:
    """分段线性查找表（与 CUDA SigmoidLUT_INT16/INT8 对应）"""
    NUM_SEGMENTS = 16
    
    def __init__(self, zp_x: int, shift_bits_x: int, shift_bits_y: int, zp_y: int):
        self.segments = []  # List[SegmentParams]
        self.zp_x = zp_x
        self.shift_bits_x = shift_bits_x
        self.shift_bits_y = shift_bits_y
        self.zp_y = zp_y


def find_segment(q_x: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """
    段查找函数（ONNX 可导出的向量化实现）
    
    Args:
        q_x: 量化输入张量 [N]
        thresholds: 各段阈值张量 [num_segments]
        
    Returns:
        每个元素对应的段索引 [N]
    """
    # 扩展维度进行比较: q_x [N, 1] >= thresholds [1, S] -> [N, S]
    q_x_expanded = q_x.unsqueeze(-1)  # [N, 1]
    thresholds_expanded = thresholds.unsqueeze(0)  # [1, S]
    
    # 比较并累加得到段索引
    # 每个元素 >= threshold[i] 则累加 1
    comparisons = (q_x_expanded >= thresholds_expanded).to(torch.long)  # [N, S]
    seg_ids = comparisons.sum(dim=-1)  # [N]
    
    # 确保不超过最后一段
    seg_ids = torch.clamp(seg_ids, 0, thresholds.shape[0] - 1)
    return seg_ids


def rshift_round_by_index(bx: torch.Tensor, n_BXs: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
    """
    根据段索引进行 rshift_round（ONNX 可导出）
    
    Args:
        bx: 输入张量 [N] (int64)
        n_BXs: 每段的移位量 [num_segments] (int8)
        seg_ids: 段索引 [N]
        
    Returns:
        移位后的张量 [N]
    """
    # 选择对应的 n 值
    n_selected = n_BXs[seg_ids].to(torch.int64)  # [N]
    
    # 计算 offset = 2^(n-1)，对于 n <= 0 设为 0
    n_clamped = torch.clamp(n_selected - 1, min=0)
    offset = torch.ones_like(bx) << n_clamped
    offset = torch.where(n_selected > 0, offset, torch.zeros_like(offset))
    
    # 右移结果
    positive_mask = bx >= 0
    rshift_pos = (bx + offset) >> n_selected
    rshift_neg = -(((-bx) + offset) >> n_selected)
    rshift_result = torch.where(positive_mask, rshift_pos, rshift_neg)
    
    # 左移结果
    lshift_result = bx << (-n_selected)
    
    # 根据 n 的正负选择结果
    result = torch.where(n_selected > 0, rshift_result, lshift_result)
    
    return result


def piecewise_linear_forward(q_x: torch.Tensor, lut: PiecewiseLUT, 
                             output_signed: bool = True, bitwidth: int = 16) -> torch.Tensor:
    """
    分段线性近似前向计算（ONNX 可导出，与 CUDA sigmoid/tanh_piecewise_linear 一致）
    
    Args:
        q_x: 量化输入张量（int32）
        lut: 分段线性查找表
        output_signed: 输出是否有符号（sigmoid=False, tanh=True）
        bitwidth: 输出位宽 (8 或 16)
        
    Returns:
        量化输出张量
    """
    device = q_x.device
    original_shape = q_x.shape
    q_x_flat = q_x.flatten()
    
    # 收集 LUT 参数为张量
    thresholds = torch.tensor([seg.threshold for seg in lut.segments], device=device, dtype=torch.int32)
    q_bs = torch.tensor([seg.q_b for seg in lut.segments], device=device, dtype=torch.int64)
    n_BXs = torch.tensor([seg.n_BX_total for seg in lut.segments], device=device, dtype=torch.int8)
    term_cs = torch.tensor([seg.term_c_precomputed for seg in lut.segments], device=device, dtype=torch.int64)
    
    # Step 1: 段查找（向量化）
    seg_ids = find_segment(q_x_flat, thresholds)
    
    # Step 2: x_offset = q_x - zp_x
    x_offset = q_x_flat.to(torch.int64) - lut.zp_x
    
    # Step 3: bx = q_b * x_offset
    q_b_selected = q_bs[seg_ids]
    bx = q_b_selected * x_offset
    
    # Step 4: term_bx = bx >> n_BX_total（向量化）
    term_bx = rshift_round_by_index(bx, n_BXs, seg_ids)
    
    # Step 5: q_y = term_bx + term_c
    term_c_selected = term_cs[seg_ids]
    y = term_bx + term_c_selected
    
    # Step 6: clamp
    if bitwidth == 16:
        if output_signed:
            y = torch.clamp(y, -32768, 32767)
        else:
            y = torch.clamp(y, 0, 65535)
    else:  # INT8
        if output_signed:
            y = torch.clamp(y, -128, 127)
        else:
            y = torch.clamp(y, 0, 255)
    
    return y.to(torch.int32).view(original_shape)


def generate_sigmoid_lut(exp2_inv_x: int, zp_x: int, exp2_inv_y: int, zp_y: int,
                         x_min: float = -8.0, x_max: float = 8.0, 
                         bitwidth: int = 16) -> PiecewiseLUT:
    """
    生成 Sigmoid 分段线性 LUT（与 CUDA generate_sigmoid_lut_int16/int8 一致）
    
    Args:
        exp2_inv_x: 输入量化参数
        zp_x: 输入零点
        exp2_inv_y: 输出量化参数
        zp_y: 输出零点
        x_min, x_max: sigmoid 有效范围
        bitwidth: 位宽 (8 或 16)
        
    Returns:
        PiecewiseLUT 对象
    """
    import math
    
    lut = PiecewiseLUT(zp_x, exp2_inv_x, exp2_inv_y, zp_y)
    
    # 计算 scale
    scale_x = 1.0 / (1 << exp2_inv_x) if exp2_inv_x >= 0 else float(1 << (-exp2_inv_x))
    scale_y = 1.0 / (1 << exp2_inv_y) if exp2_inv_y >= 0 else float(1 << (-exp2_inv_y))
    
    # 限制范围
    x_min = max(x_min, -8.0)
    x_max = min(x_max, 8.0)
    
    # 确定量化范围
    if bitwidth == 16:
        q_min, q_max = -32768, 32767
        y_min, y_max = (0, 65535)  # sigmoid 输出无符号
    else:
        q_min, q_max = -128, 127
        y_min, y_max = (0, 255)
    
    # 分段边界
    num_segments = PiecewiseLUT.NUM_SEGMENTS
    segment_width = (x_max - x_min) / num_segments
    
    for i in range(num_segments):
        # 段边界
        seg_start = x_min + i * segment_width
        seg_end = seg_start + segment_width
        seg_mid = (seg_start + seg_end) / 2
        
        # 计算该段的线性近似: y = b * x + c
        # sigmoid(x) 在 seg_mid 处的斜率
        sigmoid_mid = 1.0 / (1.0 + math.exp(-seg_mid))
        b_fp = sigmoid_mid * (1.0 - sigmoid_mid)  # sigmoid 导数
        c_fp = sigmoid_mid - b_fp * seg_mid
        
        # 量化参数
        # q_y = q_b * (q_x - zp_x) >> n_BX + term_c
        # 需要满足: (q_y - zp_y) * scale_y = b_fp * (q_x - zp_x) * scale_x + c_fp
        
        # 计算 q_b 和移位
        shift_bits_b = exp2_inv_y  # 近似
        q_b = int(round(b_fp * (1 << shift_bits_b) / scale_x * scale_y))
        n_BX_total = shift_bits_b + exp2_inv_x - exp2_inv_y
        
        # 计算 term_c (包含 zp_y)
        c_adjusted = c_fp + zp_y * scale_y
        term_c = int(round(c_adjusted / scale_y))
        
        # 阈值（量化后的段边界）
        threshold = int(round(seg_end / scale_x)) + zp_x
        if bitwidth == 16:
            threshold = max(-32768, min(32767, threshold))
        else:
            threshold = max(-128, min(127, threshold))
        
        seg_params = SegmentParams(q_b, n_BX_total, term_c, threshold)
        lut.segments.append(seg_params)
    
    return lut


def generate_tanh_lut(exp2_inv_x: int, zp_x: int, exp2_inv_y: int, zp_y: int,
                      x_min: float = -4.0, x_max: float = 4.0,
                      bitwidth: int = 16) -> PiecewiseLUT:
    """
    生成 Tanh 分段线性 LUT（与 CUDA generate_tanh_lut_int16/int8 一致）
    """
    import math
    
    lut = PiecewiseLUT(zp_x, exp2_inv_x, exp2_inv_y, zp_y)
    
    scale_x = 1.0 / (1 << exp2_inv_x) if exp2_inv_x >= 0 else float(1 << (-exp2_inv_x))
    scale_y = 1.0 / (1 << exp2_inv_y) if exp2_inv_y >= 0 else float(1 << (-exp2_inv_y))
    
    x_min = max(x_min, -4.0)
    x_max = min(x_max, 4.0)
    
    if bitwidth == 16:
        q_min, q_max = -32768, 32767
    else:
        q_min, q_max = -128, 127
    
    num_segments = PiecewiseLUT.NUM_SEGMENTS
    segment_width = (x_max - x_min) / num_segments
    
    for i in range(num_segments):
        seg_start = x_min + i * segment_width
        seg_end = seg_start + segment_width
        seg_mid = (seg_start + seg_end) / 2
        
        # tanh 及其导数
        tanh_mid = math.tanh(seg_mid)
        b_fp = 1.0 - tanh_mid ** 2  # tanh 导数
        c_fp = tanh_mid - b_fp * seg_mid
        
        shift_bits_b = exp2_inv_y
        q_b = int(round(b_fp * (1 << shift_bits_b) / scale_x * scale_y))
        n_BX_total = shift_bits_b + exp2_inv_x - exp2_inv_y
        
        c_adjusted = c_fp + zp_y * scale_y
        term_c = int(round(c_adjusted / scale_y))
        
        threshold = int(round(seg_end / scale_x)) + zp_x
        if bitwidth == 16:
            threshold = max(-32768, min(32767, threshold))
        else:
            threshold = max(-128, min(127, threshold))
        
        seg_params = SegmentParams(q_b, n_BX_total, term_c, threshold)
        lut.segments.append(seg_params)
    
    return lut


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
        h0_tensor = ensure_cuda_float32(h0, device) if h0 is not None else torch.empty(0, device=device,
                                                                                       dtype=torch.float32)

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
        3. gru.export_mode = False  # 恢复 CUDA 模式
    
    高级：指定导出格式:
        gru.export_format = 'float'      # 浮点（默认，与 Haste 一致）
        gru.export_format = 'qdq'        # QDQ 伪量化（量化模型推荐）
        gru.export_format = 'fixedpoint' # 纯定点（与 CUDA 量化一致）

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

        # ONNX 导出开关：True 时使用纯 PyTorch 实现，可被 ONNX 追踪
        self.export_mode = False
        # 导出格式（高级选项，仅在 export_mode=True 时有效）
        # 'float': 浮点（默认，与 Haste GRU 行为一致）
        # 'qdq': QDQ 伪量化（推荐用于量化模型）
        # 'fixedpoint': 纯定点（与 CUDA 量化一致，用于验证）
        self._export_format = 'float'

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
        self.quant_ranges = None  # calibrate() 时创建
        self.quant_params = None  # finalize_calibration() 时创建
        if bidirectional:
            self.quant_ranges_reverse = None
            self.quant_params_reverse = None

        self._calibration_dirty = False  # 校准数据更新标志
        self._bitwidth_config_dict = None  # 位宽配置（Python 字典，可序列化）
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

    # -------------------- ONNX 导出模式：纯 PyTorch 实现 --------------------

    def _get_quant_param(self, param_name: str, quant_params) -> Tuple[int, int]:
        """获取量化参数 (exp2_inv, zero_point)"""
        exp2_inv = getattr(quant_params, f'exp2_inv_{param_name}_', 0)
        zp = getattr(quant_params, f'zp_{param_name}_', 0)
        return exp2_inv, zp

    def _get_bitwidth(self, op_name: str) -> int:
        """获取指定操作的位宽"""
        if self._bitwidth_config_dict is not None:
            return self._bitwidth_config_dict.get(f'{op_name}_', 8)
        return 8

    def _get_symmetric(self, op_name: str) -> bool:
        """获取指定操作是否对称量化"""
        if self._bitwidth_config_dict is not None:
            return self._bitwidth_config_dict.get(f'{op_name}_symmetric_', True)
        return True

    @property
    def export_format(self) -> str:
        """
        获取导出格式（高级选项，仅在 export_mode=True 时有效）
        
        Returns:
            'float': 浮点格式（默认，与 Haste GRU 行为一致）
            'qdq': QDQ 伪量化格式（推荐用于量化模型 ONNX 导出）
            'fixedpoint': 纯定点格式（与 CUDA 量化完全一致，用于精度验证）
        """
        return self._export_format
    
    @export_format.setter
    def export_format(self, mode: str):
        """
        设置导出格式（高级用法，大多数用户不需要修改）
        
        Args:
            mode: 'qdq' | 'fixedpoint' | 'float'
        """
        valid_modes = ('qdq', 'fixedpoint', 'float')
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
        纯 PyTorch 实现的单向 GRU 前向传播（可被 ONNX 追踪）

        GRU 公式（Haste 格式，门顺序为 z, r, g）：
            z = sigmoid(W_z @ x + R_z @ h + bx_z + br_z)  # update gate
            r = sigmoid(W_r @ x + R_r @ h + bx_r + br_r)  # reset gate
            g = tanh(W_g @ x + r * (R_g @ h + br_g) + bx_g)  # candidate gate
            h' = z * h + (1 - z) * g

        量化模式下根据 ONNX 导出模式选择实现：
            - 'qdq': QDQ 格式，使用标准算子 + 伪量化
            - 'fixedpoint': 纯定点，与 CUDA 完全一致
            - 'float': 标准浮点计算（Haste 格式）

        Args:
            input: [T, B, I] 输入序列
            h0: [B, H] 初始隐藏状态 或 None
            weight_ih: [3*H, I] 输入权重 (PyTorch r,z,n 格式，内部自动转换)
            weight_hh: [3*H, H] 循环权重 (PyTorch r,z,n 格式，内部自动转换)
            bias_ih: [3*H] 输入偏置 或 None (PyTorch 格式，内部自动转换)
            bias_hh: [3*H] 循环偏置 或 None (PyTorch 格式，内部自动转换)
            quant_params: 量化参数（来自 finalize_calibration）

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
        
        # qdq/fixedpoint 需要量化参数
        if quant_params is None:
            raise RuntimeError(
                f"export_format='{self._export_format}' 需要量化参数，"
                f"请先调用 calibrate() 和 finalize_calibration()"
            )
        
        if self._export_format == 'qdq':
            return self._forward_onnx_qdq_single_direction(
                input, h0, weight_ih, weight_hh, bias_ih, bias_hh, quant_params
            )
        else:  # 'fixedpoint'
            return self._forward_python_fixedpoint_single_direction(
                input, h0, weight_ih, weight_hh, bias_ih, bias_hh, quant_params
            )

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
        浮点实现的单向 GRU 前向传播（Haste 格式）
        
        与 HasteGRU CUDA 浮点推理行为一致
        门控顺序：Haste 格式 (z, r, g)
        
        公式（与 gru_forward_gpu.cu 一致）：
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

        # ========== 循环外一次性计算 Wx GEMM（与 CUDA 一致）==========
        # input: [T, B, I] -> x_flat: [T*B, I]
        # W: [3*H, I] -> W.t(): [I, 3*H]
        # Wx_all: [T*B, 3*H] -> reshape: [T, B, 3*H]
        x_flat = input.reshape(T * B, I)
        Wx_all = torch.mm(x_flat, W.t())  # [T*B, 3*H]
        Wx_all = Wx_all.reshape(T, B, 3 * H)  # [T, B, 3*H]

        # 预分割偏置（循环外完成）
        bx_z, bx_r, bx_g = bx.chunk(3)
        br_z, br_r, br_g = br.chunk(3)

        outputs = []

        for t in range(T):
            # 获取当前时间步的 Wx（已在循环外计算好）
            Wx = Wx_all[t]  # [B, 3*H]
            
            # Rh = h @ R.T, shape [B, 3H]（依赖上一步的 h，必须在循环内）
            Rh = torch.mm(h, R.t())

            # 分割门控（Haste 格式：z, r, g）
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

    # -------------------- 纯定点实现（与 CUDA 完全一致）--------------------

    def _build_rescale_params(self, quant_params, H: int, device: torch.device):
        """
        从 quant_params 构建 rescale 参数（与 CUDA ForwardPassQuant::set_parms 一致）
        
        Returns:
            dict: 包含所有 rescale 参数的字典
        """
        params = {}
        
        # 基础参数
        params['zp_x'] = quant_params.zp_x_
        params['zp_h'] = quant_params.zp_h_
        params['zp_Wx'] = quant_params.zp_Wx_
        params['zp_Rh'] = quant_params.zp_Rh_
        params['exp2_inv_x'] = quant_params.exp2_inv_x_
        params['exp2_inv_h'] = quant_params.exp2_inv_h_
        params['exp2_inv_Wx'] = quant_params.exp2_inv_Wx_
        params['exp2_inv_Rh'] = quant_params.exp2_inv_Rh_
        
        # GEMM rescale 参数 (per-channel)
        # n_W_mul_x_div_Wx[c] = (exp2_inv_W[c] + exp2_inv_x) - exp2_inv_Wx
        # n_R_mul_h_div_Rh[c] = (exp2_inv_R[c] + exp2_inv_h) - exp2_inv_Rh
        exp2_inv_W = list(quant_params.exp2_inv_W_)
        exp2_inv_R = list(quant_params.exp2_inv_R_)
        params['n_W_mul_x_div_Wx'] = [(exp2_inv_W[c] + quant_params.exp2_inv_x_) - quant_params.exp2_inv_Wx_ 
                                      for c in range(3 * H)]
        params['n_R_mul_h_div_Rh'] = [(exp2_inv_R[c] + quant_params.exp2_inv_h_) - quant_params.exp2_inv_Rh_ 
                                      for c in range(3 * H)]
        
        # z 门
        params['zp_z_pre'] = quant_params.zp_z_pre_
        params['zp_z_out'] = quant_params.zp_z_out_
        params['exp2_inv_Wx_div_z_pre'] = quant_params.exp2_inv_Wx_ - quant_params.exp2_inv_z_pre_
        params['exp2_inv_Rh_div_z_pre'] = quant_params.exp2_inv_Rh_ - quant_params.exp2_inv_z_pre_
        
        # per-channel bias rescale for z gate
        exp2_inv_bx = list(quant_params.exp2_inv_bx_)
        exp2_inv_br = list(quant_params.exp2_inv_br_)
        params['n_bx_div_z'] = [exp2_inv_bx[i] - quant_params.exp2_inv_z_pre_ for i in range(H)]
        params['n_br_div_z'] = [exp2_inv_br[i] - quant_params.exp2_inv_z_pre_ for i in range(H)]
        
        # r 门
        params['zp_r_pre'] = quant_params.zp_r_pre_
        params['zp_r_out'] = quant_params.zp_r_out_
        params['exp2_inv_Wx_div_r_pre'] = quant_params.exp2_inv_Wx_ - quant_params.exp2_inv_r_pre_
        params['exp2_inv_Rh_div_r_pre'] = quant_params.exp2_inv_Rh_ - quant_params.exp2_inv_r_pre_
        params['n_bx_div_r'] = [exp2_inv_bx[H + i] - quant_params.exp2_inv_r_pre_ for i in range(H)]
        params['n_br_div_r'] = [exp2_inv_br[H + i] - quant_params.exp2_inv_r_pre_ for i in range(H)]
        
        # g 门 (new gate)
        params['zp_g_pre'] = quant_params.zp_g_pre_
        params['zp_g_out'] = quant_params.zp_g_out_
        params['n_Rh_div_Rh_add_br'] = quant_params.exp2_inv_Rh_ - quant_params.exp2_inv_Rh_add_br_
        params['n_br_div_Rh_add_br'] = [exp2_inv_br[2*H + i] - quant_params.exp2_inv_Rh_add_br_ for i in range(H)]
        params['zp_Rh_add_br'] = quant_params.zp_Rh_add_br_
        params['n_r_mul_Rh_add_br_div_rRh'] = (quant_params.exp2_inv_r_out_ + quant_params.exp2_inv_Rh_add_br_) - quant_params.exp2_inv_rRh_
        params['zp_rRh'] = quant_params.zp_rRh_
        params['n_Wx_div_g_pre'] = quant_params.exp2_inv_Wx_ - quant_params.exp2_inv_g_pre_
        params['n_rRh_div_g_pre'] = quant_params.exp2_inv_rRh_ - quant_params.exp2_inv_g_pre_
        params['n_bx_div_g_pre'] = [exp2_inv_bx[2*H + i] - quant_params.exp2_inv_g_pre_ for i in range(H)]
        
        # h_new
        # one_in_z_scale = round(1.0 * 2^exp2_inv_z_out) + zp_z_out
        exp2_z_out = quant_params.exp2_inv_z_out_
        if exp2_z_out >= 0:
            one_scaled = 1 << exp2_z_out
        else:
            one_scaled = 1  # 近似处理
        params['one_in_z_scale'] = one_scaled + quant_params.zp_z_out_
        
        params['zp_new_contrib'] = quant_params.zp_new_contrib_
        params['n_z_out_mul_g_div_new_contrib'] = (quant_params.exp2_inv_z_out_ + quant_params.exp2_inv_g_out_) - quant_params.exp2_inv_new_contrib_
        params['zp_old_contrib'] = quant_params.zp_old_contrib_
        params['n_z_mul_h_div_old_contrib'] = (quant_params.exp2_inv_z_out_ + quant_params.exp2_inv_h_) - quant_params.exp2_inv_old_contrib_
        params['n_new_contrib_div_h'] = quant_params.exp2_inv_new_contrib_ - quant_params.exp2_inv_h_
        params['n_old_contrib_div_h'] = quant_params.exp2_inv_old_contrib_ - quant_params.exp2_inv_h_
        
        # LUT 参数
        params['exp2_inv_z_pre'] = quant_params.exp2_inv_z_pre_
        params['exp2_inv_z_out'] = quant_params.exp2_inv_z_out_
        params['exp2_inv_r_pre'] = quant_params.exp2_inv_r_pre_
        params['exp2_inv_r_out'] = quant_params.exp2_inv_r_out_
        params['exp2_inv_g_pre'] = quant_params.exp2_inv_g_pre_
        params['exp2_inv_g_out'] = quant_params.exp2_inv_g_out_
        params['exp2_inv_h'] = quant_params.exp2_inv_h_
        
        return params

    def _compute_z_fixedpoint(self, Wx_z: torch.Tensor, Rh_z: torch.Tensor, 
                               bx_z: torch.Tensor, br_z: torch.Tensor,
                               channel_idx: int, rescale: dict, 
                               sigmoid_lut: PiecewiseLUT, bitwidth: int) -> torch.Tensor:
        """
        计算 z 门（纯定点，与 CUDA computeZ 一致）
        
        z = sigmoid(Wx + Rh + bx + br)
        """
        # Rescale 各项到 z_pre 的量化空间
        Wx_shifted = rshift_round(Wx_z - rescale['zp_Wx'], rescale['exp2_inv_Wx_div_z_pre'])
        Rh_shifted = rshift_round(Rh_z - rescale['zp_Rh'], rescale['exp2_inv_Rh_div_z_pre'])
        bx_shifted = rshift_round(bx_z, rescale['n_bx_div_z'][channel_idx])
        br_shifted = rshift_round(br_z, rescale['n_br_div_z'][channel_idx])
        
        z_pre = Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale['zp_z_pre']
        
        # 通过 LUT 计算 sigmoid
        if bitwidth == 16:
            z_pre = clamp_to_int16(z_pre)
        else:
            z_pre = clamp_to_int8(z_pre)
        
        z = piecewise_linear_forward(z_pre, sigmoid_lut, output_signed=False, bitwidth=bitwidth)
        return z

    def _compute_r_fixedpoint(self, Wx_r: torch.Tensor, Rh_r: torch.Tensor,
                               bx_r: torch.Tensor, br_r: torch.Tensor,
                               channel_idx: int, rescale: dict,
                               sigmoid_lut: PiecewiseLUT, bitwidth: int) -> torch.Tensor:
        """
        计算 r 门（纯定点，与 CUDA computeR 一致）
        
        r = sigmoid(Wx + Rh + bx + br)
        """
        Wx_shifted = rshift_round(Wx_r - rescale['zp_Wx'], rescale['exp2_inv_Wx_div_r_pre'])
        Rh_shifted = rshift_round(Rh_r - rescale['zp_Rh'], rescale['exp2_inv_Rh_div_r_pre'])
        bx_shifted = rshift_round(bx_r, rescale['n_bx_div_r'][channel_idx])
        br_shifted = rshift_round(br_r, rescale['n_br_div_r'][channel_idx])
        
        r_pre = Wx_shifted + Rh_shifted + bx_shifted + br_shifted + rescale['zp_r_pre']
        
        if bitwidth == 16:
            r_pre = clamp_to_int16(r_pre)
        else:
            r_pre = clamp_to_int8(r_pre)
        
        r = piecewise_linear_forward(r_pre, sigmoid_lut, output_signed=False, bitwidth=bitwidth)
        return r

    def _compute_g_fixedpoint(self, Wx_g: torch.Tensor, Rh_g: torch.Tensor,
                               bx_g: torch.Tensor, br_g: torch.Tensor, r: torch.Tensor,
                               channel_idx: int, rescale: dict,
                               tanh_lut: PiecewiseLUT, bitwidth: int) -> torch.Tensor:
        """
        计算 g 门（纯定点，与 CUDA computeG 一致）
        
        g = tanh(Wx + r * (Rh + br) + bx)
        """
        # Rh_add_br = Rh + br (rescale 到 Rh_add_br 空间)
        Rh_shifted = rshift_round(Rh_g - rescale['zp_Rh'], rescale['n_Rh_div_Rh_add_br'])
        br_shifted = rshift_round(br_g, rescale['n_br_div_Rh_add_br'][channel_idx])
        Rh_add_br = Rh_shifted + br_shifted + rescale['zp_Rh_add_br']
        
        # rRh = r * Rh_add_br (整数乘法，然后 rescale)
        r_diff = (r - rescale['zp_r_out']).to(torch.int64)
        Rh_add_br_diff = (Rh_add_br - rescale['zp_Rh_add_br']).to(torch.int64)
        rRh_mul = r_diff * Rh_add_br_diff
        rRh = rshift_round_i64(rRh_mul, rescale['n_r_mul_Rh_add_br_div_rRh']).to(torch.int32) + rescale['zp_rRh']
        
        # g_pre = Wx + rRh + bx
        Wx_shifted = rshift_round(Wx_g - rescale['zp_Wx'], rescale['n_Wx_div_g_pre'])
        rRh_shifted = rshift_round(rRh - rescale['zp_rRh'], rescale['n_rRh_div_g_pre'])
        bx_shifted = rshift_round(bx_g, rescale['n_bx_div_g_pre'][channel_idx])
        
        g_pre = Wx_shifted + rRh_shifted + bx_shifted + rescale['zp_g_pre']
        
        if bitwidth == 16:
            g_pre = clamp_to_int16(g_pre)
        else:
            g_pre = clamp_to_int8(g_pre)
        
        g = piecewise_linear_forward(g_pre, tanh_lut, output_signed=True, bitwidth=bitwidth)
        return g

    def _compute_h_fixedpoint(self, z: torch.Tensor, g: torch.Tensor, h_old: torch.Tensor,
                               rescale: dict, bitwidth: int) -> torch.Tensor:
        """
        计算新隐藏状态（纯定点，与 CUDA computeH 一致）
        
        h_new = z * h_old + (1 - z) * g
        """
        # old_contrib = z * h_old
        z_diff = (z - rescale['zp_z_out']).to(torch.int64)
        h_diff = (h_old - rescale['zp_h']).to(torch.int64)
        old_contrib_mul = z_diff * h_diff
        old_contrib = rshift_round_i64(old_contrib_mul, rescale['n_z_mul_h_div_old_contrib']).to(torch.int32) + rescale['zp_old_contrib']
        
        # new_contrib = (1 - z) * g
        # (1 - z) 在量化空间: one_in_z_scale - z
        one_minus_z_diff = rescale['one_in_z_scale'] - z.to(torch.int64)
        g_diff = (g - rescale['zp_g_out']).to(torch.int64)
        new_contrib_mul = one_minus_z_diff * g_diff
        new_contrib = rshift_round_i64(new_contrib_mul, rescale['n_z_out_mul_g_div_new_contrib']).to(torch.int32) + rescale['zp_new_contrib']
        
        # h_new = old_contrib + new_contrib (rescale 到 h 空间)
        old_shifted = rshift_round(old_contrib - rescale['zp_old_contrib'], rescale['n_old_contrib_div_h'])
        new_shifted = rshift_round(new_contrib - rescale['zp_new_contrib'], rescale['n_new_contrib_div_h'])
        h_new = old_shifted + new_shifted + rescale['zp_h']
        
        # clamp 到目标位宽
        if bitwidth == 16:
            h_new = clamp_to_int16(h_new)
        else:
            h_new = clamp_to_int8(h_new)
        
        return h_new

    # -------------------- 向量化门控计算（ONNX 可导出）--------------------
    
    def _compute_z_vectorized(self, Wx_z: torch.Tensor, Rh_z: torch.Tensor,
                               bx_z: torch.Tensor, br_z: torch.Tensor,
                               n_bx_div_z: torch.Tensor, n_br_div_z: torch.Tensor,
                               rescale: dict, sigmoid_lut: PiecewiseLUT, bitwidth: int) -> torch.Tensor:
        """
        向量化计算 z 门（ONNX 可导出）
        
        Args:
            Wx_z: [B, H], Rh_z: [B, H]
            bx_z: [H], br_z: [H] (per-channel 偏置)
            n_bx_div_z: [H], n_br_div_z: [H] (per-channel shift)
        """
        # Wx, Rh: 全局 shift（标量）
        Wx_shifted = rshift_round(Wx_z - rescale['zp_Wx'], rescale['exp2_inv_Wx_div_z_pre'])
        Rh_shifted = rshift_round(Rh_z - rescale['zp_Rh'], rescale['exp2_inv_Rh_div_z_pre'])
        
        # 偏置: per-channel shift（向量化）
        # bx_z: [H], n_bx_div_z: [H] -> 广播到 [1, H] 然后对 [B, H] 操作
        bx_shifted = rshift_round_per_channel(bx_z.unsqueeze(0).to(torch.int64), n_bx_div_z).squeeze(0).to(torch.int32)
        br_shifted = rshift_round_per_channel(br_z.unsqueeze(0).to(torch.int64), n_br_div_z).squeeze(0).to(torch.int32)
        
        # 广播加法: [B, H] + [H] -> [B, H]
        z_pre = Wx_shifted + Rh_shifted + bx_shifted.unsqueeze(0) + br_shifted.unsqueeze(0) + rescale['zp_z_pre']
        
        if bitwidth == 16:
            z_pre = clamp_to_int16(z_pre)
        else:
            z_pre = clamp_to_int8(z_pre)
        
        z = piecewise_linear_forward(z_pre, sigmoid_lut, output_signed=False, bitwidth=bitwidth)
        return z

    def _compute_r_vectorized(self, Wx_r: torch.Tensor, Rh_r: torch.Tensor,
                               bx_r: torch.Tensor, br_r: torch.Tensor,
                               n_bx_div_r: torch.Tensor, n_br_div_r: torch.Tensor,
                               rescale: dict, sigmoid_lut: PiecewiseLUT, bitwidth: int) -> torch.Tensor:
        """向量化计算 r 门（ONNX 可导出）"""
        Wx_shifted = rshift_round(Wx_r - rescale['zp_Wx'], rescale['exp2_inv_Wx_div_r_pre'])
        Rh_shifted = rshift_round(Rh_r - rescale['zp_Rh'], rescale['exp2_inv_Rh_div_r_pre'])
        
        bx_shifted = rshift_round_per_channel(bx_r.unsqueeze(0).to(torch.int64), n_bx_div_r).squeeze(0).to(torch.int32)
        br_shifted = rshift_round_per_channel(br_r.unsqueeze(0).to(torch.int64), n_br_div_r).squeeze(0).to(torch.int32)
        
        r_pre = Wx_shifted + Rh_shifted + bx_shifted.unsqueeze(0) + br_shifted.unsqueeze(0) + rescale['zp_r_pre']
        
        if bitwidth == 16:
            r_pre = clamp_to_int16(r_pre)
        else:
            r_pre = clamp_to_int8(r_pre)
        
        r = piecewise_linear_forward(r_pre, sigmoid_lut, output_signed=False, bitwidth=bitwidth)
        return r

    def _compute_g_vectorized(self, Wx_g: torch.Tensor, Rh_g: torch.Tensor,
                               bx_g: torch.Tensor, br_g: torch.Tensor, r: torch.Tensor,
                               n_br_div_Rh_add_br: torch.Tensor, n_bx_div_g_pre: torch.Tensor,
                               rescale: dict, tanh_lut: PiecewiseLUT, bitwidth: int) -> torch.Tensor:
        """向量化计算 g 门（ONNX 可导出）"""
        # Rh_add_br = Rh + br
        Rh_shifted = rshift_round(Rh_g - rescale['zp_Rh'], rescale['n_Rh_div_Rh_add_br'])
        br_shifted = rshift_round_per_channel(br_g.unsqueeze(0).to(torch.int64), n_br_div_Rh_add_br).squeeze(0).to(torch.int32)
        Rh_add_br = Rh_shifted + br_shifted.unsqueeze(0) + rescale['zp_Rh_add_br']
        
        # rRh = r * Rh_add_br
        r_diff = (r - rescale['zp_r_out']).to(torch.int64)
        Rh_add_br_diff = (Rh_add_br - rescale['zp_Rh_add_br']).to(torch.int64)
        rRh_mul = r_diff * Rh_add_br_diff
        rRh = rshift_round_i64(rRh_mul, rescale['n_r_mul_Rh_add_br_div_rRh']).to(torch.int32) + rescale['zp_rRh']
        
        # g_pre = Wx + rRh + bx
        Wx_shifted = rshift_round(Wx_g - rescale['zp_Wx'], rescale['n_Wx_div_g_pre'])
        rRh_shifted = rshift_round(rRh - rescale['zp_rRh'], rescale['n_rRh_div_g_pre'])
        bx_shifted = rshift_round_per_channel(bx_g.unsqueeze(0).to(torch.int64), n_bx_div_g_pre).squeeze(0).to(torch.int32)
        
        g_pre = Wx_shifted + rRh_shifted + bx_shifted.unsqueeze(0) + rescale['zp_g_pre']
        
        if bitwidth == 16:
            g_pre = clamp_to_int16(g_pre)
        else:
            g_pre = clamp_to_int8(g_pre)
        
        g = piecewise_linear_forward(g_pre, tanh_lut, output_signed=True, bitwidth=bitwidth)
        return g

    def _compute_h_vectorized(self, z: torch.Tensor, g: torch.Tensor, h_old: torch.Tensor,
                               rescale: dict, bitwidth: int) -> torch.Tensor:
        """向量化计算新隐藏状态（ONNX 可导出）"""
        # old_contrib = z * h_old
        z_diff = (z - rescale['zp_z_out']).to(torch.int64)
        h_diff = (h_old - rescale['zp_h']).to(torch.int64)
        old_contrib_mul = z_diff * h_diff
        old_contrib = rshift_round_i64(old_contrib_mul, rescale['n_z_mul_h_div_old_contrib']).to(torch.int32) + rescale['zp_old_contrib']
        
        # new_contrib = (1 - z) * g
        one_minus_z_diff = rescale['one_in_z_scale'] - z.to(torch.int64)
        g_diff = (g - rescale['zp_g_out']).to(torch.int64)
        new_contrib_mul = one_minus_z_diff * g_diff
        new_contrib = rshift_round_i64(new_contrib_mul, rescale['n_z_out_mul_g_div_new_contrib']).to(torch.int32) + rescale['zp_new_contrib']
        
        # h_new = old_contrib + new_contrib
        old_shifted = rshift_round(old_contrib - rescale['zp_old_contrib'], rescale['n_old_contrib_div_h'])
        new_shifted = rshift_round(new_contrib - rescale['zp_new_contrib'], rescale['n_new_contrib_div_h'])
        h_new = old_shifted + new_shifted + rescale['zp_h']
        
        if bitwidth == 16:
            h_new = clamp_to_int16(h_new)
        else:
            h_new = clamp_to_int8(h_new)
        
        return h_new

    def _forward_python_fixedpoint_single_direction(
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
        纯定点实现的单向 GRU 前向传播（用于 ONNX 导出）
        
        所有中间计算都是整数运算，sigmoid/tanh 使用 LUT 查表。
        量化参数和 rescale 逻辑与 CUDA 实现完全一致。
        
        [ONNX兼容说明]
        部分操作的实现方式与 CUDA 端不同，但数学上等价，ONNX 导出后由推理引擎
        选择最优实现。这些差异在代码中用 [ONNX兼容] 标记说明：
        - GEMM: Python 用 float 模拟，CUDA 用 cuBLAS INT8/融合 INT16 kernel
        - Rescale: Python 统一用后处理，CUDA INT16 用融合方式
        
        Args:
            input: [T, B, I] 输入序列（浮点，会被量化）
            h0: [B, H] 初始隐藏状态 或 None
            weight_ih: [3*H, I] 输入权重
            weight_hh: [3*H, H] 循环权重
            bias_ih: [3*H] 输入偏置 或 None
            bias_hh: [3*H] 循环偏置 或 None
            quant_params: 量化参数（来自 finalize_calibration）
            
        Returns:
            output: [T, B, H] 输出序列（浮点，从整数反量化）
            h_n: [1, B, H] 最终隐藏状态
        """
        T, B, I = input.shape
        H = self.hidden_size
        device = input.device
        
        # 获取位宽配置
        h_bitwidth = self._get_bitwidth('h')
        z_out_bitwidth = self._get_bitwidth('z_out')
        r_out_bitwidth = self._get_bitwidth('r_out')
        g_out_bitwidth = self._get_bitwidth('g_out')
        
        # 构建 rescale 参数
        rescale = self._build_rescale_params(quant_params, H, device)
        
        # 生成 LUT
        sigmoid_z_lut = generate_sigmoid_lut(
            rescale['exp2_inv_z_pre'], rescale['zp_z_pre'],
            rescale['exp2_inv_z_out'], rescale['zp_z_out'],
            bitwidth=z_out_bitwidth
        )
        sigmoid_r_lut = generate_sigmoid_lut(
            rescale['exp2_inv_r_pre'], rescale['zp_r_pre'],
            rescale['exp2_inv_r_out'], rescale['zp_r_out'],
            bitwidth=r_out_bitwidth
        )
        tanh_lut = generate_tanh_lut(
            rescale['exp2_inv_g_pre'], rescale['zp_g_pre'],
            rescale['exp2_inv_g_out'], rescale['zp_g_out'],
            bitwidth=g_out_bitwidth
        )
        
        # 量化参数
        exp2_x = quant_params.exp2_inv_x_
        zp_x = quant_params.zp_x_
        exp2_h = quant_params.exp2_inv_h_
        zp_h = quant_params.zp_h_
        
        # 重排序权重/偏置：PyTorch 格式 (r, z, n) -> Haste 格式 (z, r, n)
        # 与 GRUFunction.forward 一致
        W_reordered = reorder_weights_pytorch_to_haste(weight_ih)  # [3*H, I], (z, r, n) 顺序
        R_reordered = reorder_weights_pytorch_to_haste(weight_hh)  # [3*H, H], (z, r, n) 顺序
        
        # 转置权重到 CUDA 格式：[3*H, I] -> [I, 3*H]
        W = W_reordered.t().contiguous()  # [I, 3*H]
        R = R_reordered.t().contiguous()  # [H, 3*H]
        
        # 量化权重（per-channel，zp=0）
        # quantificationPerChannel(W, W_quant, input_size, 3*hidden_size, exp2_inv_W)
        exp2_W = list(quant_params.exp2_inv_W_)
        exp2_R = list(quant_params.exp2_inv_R_)
        W_q = quantize_per_channel(W, exp2_W, zp=0, 
                                   bitwidth=self._get_bitwidth('W'), symmetric=self._get_symmetric('W'))
        R_q = quantize_per_channel(R, exp2_R, zp=0,
                                   bitwidth=self._get_bitwidth('R'), symmetric=self._get_symmetric('R'))
        
        # 重排序偏置：PyTorch 格式 (r, z, n) -> Haste 格式 (z, r, n)
        if bias_ih is not None:
            bx_reordered = reorder_weights_pytorch_to_haste(bias_ih)
        else:
            bx_reordered = None
            
        if bias_hh is not None:
            br_reordered = reorder_weights_pytorch_to_haste(bias_hh)
        else:
            br_reordered = None
        
        # 量化偏置（per-channel，zp=0，使用 INT32）
        exp2_bx = list(quant_params.exp2_inv_bx_)
        exp2_br = list(quant_params.exp2_inv_br_)
        
        if bx_reordered is not None:
            bx_q = quantize_per_channel(bx_reordered.unsqueeze(0), exp2_bx, zp=0, bitwidth=32).squeeze(0)
        else:
            bx_q = torch.zeros(3 * H, device=device, dtype=torch.int32)
            
        if br_reordered is not None:
            br_q = quantize_per_channel(br_reordered.unsqueeze(0), exp2_br, zp=0, bitwidth=32).squeeze(0)
        else:
            br_q = torch.zeros(3 * H, device=device, dtype=torch.int32)
        
        # ========== 循环外一次性量化输入 x ==========
        # input: [T, B, I] -> x_q_all: [T*B, I]
        x_flat = input.reshape(T * B, I)
        x_q_all = quantize(x_flat, exp2_x, zp_x, 
                           bitwidth=self._get_bitwidth('x'), symmetric=self._get_symmetric('x'))
        
        # ========== 循环外一次性计算 Wx GEMM ==========
        # x_q_all: [T*B, I], W_q: [I, 3*H]
        # Wx_all_raw: [T*B, 3*H] (int64)
        # 
        # [ONNX兼容] 与 CUDA 实现差异：
        #   - CUDA INT8: 使用 cuBLAS INT8 GEMM (cublasGemmEx)
        #   - CUDA INT16: 使用融合 kernel (quantizedGemmInt16Fused)
        #   - Python: 使用 float GEMM 模拟（数值等价，int16*int16 在 float32 可精确表示）
        #   - ONNX 导出后由推理引擎选择最优实现 (MatMulInteger/QLinearMatMul)
        Wx_all_raw = torch.mm(x_q_all.to(torch.int64).float(), W_q.to(torch.int64).float()).to(torch.int64)
        
        # ========== Rescale Wx: (Wx_raw - W_sum_mul_x_zp) >> n + zp_Wx ==========
        # [ONNX兼容] 与 CUDA 实现差异：
        #   - CUDA INT8: GEMM 后处理 (rescaleGemmI32 kernel)
        #   - CUDA INT16: 融合在 GEMM kernel 中
        #   - Python: 统一使用后处理方式（数学等价：Σ W[k]*(x[k]-zp) = W@x - W_sum*zp）
        # 计算 W_sum_mul_x_zp[c] = sum_k(W_q[k, c]) * zp_x
        W_sum = W_q.sum(dim=0).to(torch.int64)  # [3*H]
        W_sum_mul_x_zp = W_sum * zp_x  # [3*H]
        
        # n_W_mul_x_div_Wx[c] = (exp2_inv_W[c] + exp2_inv_x) - exp2_inv_Wx (per-channel)
        n_W_mul_x_div_Wx = torch.tensor(rescale['n_W_mul_x_div_Wx'], device=device, dtype=torch.int8)  # [3*H]
        
        # 向量化 per-channel rescale: Wx[b, c] = rshift_round(Wx_raw[b, c] - W_sum_mul_x_zp[c], n[c]) + zp_Wx
        # Wx_all_raw: [T*B, 3*H], W_sum_mul_x_zp: [3*H] -> 广播减法
        Wx_compensated = Wx_all_raw - W_sum_mul_x_zp.unsqueeze(0)  # [T*B, 3*H]
        Wx_all = rshift_round_per_channel(Wx_compensated, n_W_mul_x_div_Wx).to(torch.int32) + quant_params.zp_Wx_
        
        # 重塑为 [T, B, 3*H]
        Wx_all = Wx_all.reshape(T, B, 3 * H)
        
        # 初始化隐藏状态
        if h0 is None:
            h_q = torch.full((B, H), zp_h, device=device, dtype=torch.int32)
        else:
            h_q = quantize(h0, exp2_h, zp_h, bitwidth=h_bitwidth, symmetric=self._get_symmetric('h'))
        
        # R_sum_mul_h_zp 和 n_R_mul_h_div_Rh 预计算
        R_sum = R_q.sum(dim=0).to(torch.int64)  # [3*H]
        R_sum_mul_h_zp = R_sum * zp_h  # [3*H], 常量，循环外计算
        n_R_mul_h_div_Rh = torch.tensor(rescale['n_R_mul_h_div_Rh'], device=device, dtype=torch.int8)  # [3*H]
        
        # 预计算 per-channel bias shift 张量（ONNX 导出需要）
        # z 门: bx[0:H], br[0:H]
        n_bx_div_z = torch.tensor(rescale['n_bx_div_z'], device=device, dtype=torch.int8)  # [H]
        n_br_div_z = torch.tensor(rescale['n_br_div_z'], device=device, dtype=torch.int8)  # [H]
        bx_z = bx_q[:H]  # [H]
        br_z = br_q[:H]  # [H]
        
        # r 门: bx[H:2H], br[H:2H]
        n_bx_div_r = torch.tensor(rescale['n_bx_div_r'], device=device, dtype=torch.int8)  # [H]
        n_br_div_r = torch.tensor(rescale['n_br_div_r'], device=device, dtype=torch.int8)  # [H]
        bx_r = bx_q[H:2*H]  # [H]
        br_r = br_q[H:2*H]  # [H]
        
        # g 门: bx[2H:3H], br[2H:3H]
        n_br_div_Rh_add_br = torch.tensor(rescale['n_br_div_Rh_add_br'], device=device, dtype=torch.int8)  # [H]
        n_bx_div_g_pre = torch.tensor(rescale['n_bx_div_g_pre'], device=device, dtype=torch.int8)  # [H]
        bx_g = bx_q[2*H:3*H]  # [H]
        br_g = br_q[2*H:3*H]  # [H]
        
        # 预分配输出张量（ONNX 导出需要，避免 list append）
        outputs_q = torch.zeros(T, B, H, device=device, dtype=torch.int32)
        
        for t in range(T):
            # 获取当前时间步的 Wx（已在循环外计算好）
            Wx = Wx_all[t]  # [B, 3*H]
            
            # ========== 计算 Rh GEMM（每个时间步依赖上一步的 h）==========
            # h_q: [B, H], R_q: [H, 3*H], Rh_raw: [B, 3*H]
            # [ONNX兼容] 同 Wx GEMM，使用 float 模拟整数 GEMM
            Rh_raw = torch.mm(h_q.to(torch.int64).float(), R_q.to(torch.int64).float()).to(torch.int64)
            
            # Rescale Rh: (Rh_raw - R_sum_mul_h_zp) >> n + zp_Rh (per-channel 向量化)
            # [ONNX兼容] 同 Wx rescale，使用后处理方式
            Rh_compensated = Rh_raw - R_sum_mul_h_zp.unsqueeze(0)  # [B, 3*H]
            Rh = rshift_round_per_channel(Rh_compensated, n_R_mul_h_div_Rh).to(torch.int32) + quant_params.zp_Rh_
            
            # 分割门控（Haste 格式：z, r, n）
            Wx_z, Wx_r, Wx_n = Wx.chunk(3, dim=1)
            Rh_z, Rh_r, Rh_n = Rh.chunk(3, dim=1)
            
            # 向量化门控计算（ONNX 可导出）
            z_out = self._compute_z_vectorized(
                Wx_z, Rh_z, bx_z, br_z,
                n_bx_div_z, n_br_div_z,
                rescale, sigmoid_z_lut, z_out_bitwidth
            )
            
            r_out = self._compute_r_vectorized(
                Wx_r, Rh_r, bx_r, br_r,
                n_bx_div_r, n_br_div_r,
                rescale, sigmoid_r_lut, r_out_bitwidth
            )
            
            g_out = self._compute_g_vectorized(
                Wx_n, Rh_n, bx_g, br_g, r_out,
                n_br_div_Rh_add_br, n_bx_div_g_pre,
                rescale, tanh_lut, g_out_bitwidth
            )
            
            h_new = self._compute_h_vectorized(
                z_out, g_out, h_q,
                rescale, h_bitwidth
            )
            
            h_q = h_new
            
            # 存储量化值（使用索引赋值，避免 list append）
            outputs_q[t] = h_q
        
        # 循环结束后一次性反量化所有时间步（与 CUDA dev::dequantification 一致）
        # outputs_q: [T, B, H] (量化，已预分配)
        output = dequantize(outputs_q, exp2_h, zp_h)  # [T, B, H] (浮点)
        h_n = dequantize(h_q, exp2_h, zp_h).unsqueeze(0)  # [1, B, H]
        
        return output, h_n

    # -------------------- ONNX 导出版本（QDQ 格式）--------------------
    
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
        
        使用伪量化（Fake Quantize）在关键点插入 Q/DQ 操作，
        推理引擎会识别 QDQ 模式并自动优化为量化算子。
        
        设计原则：
        ==========
        [与 CUDA 一致]
          - 量化参数（scale/zp）完全一致
          - 计算图结构一致（门顺序、计算顺序）
          - 权重/偏置的 per-channel 量化参数一致
          
        [ONNX 兼容 - 与 CUDA 实现不同]
          - GEMM: 使用标准 torch.mm（推理引擎会用 MatMulInteger）
          - sigmoid/tanh: 使用标准 torch.sigmoid/tanh（推理引擎会优化）
          - rescale: 通过 QDQ 实现（不用显式 rshift_round）
        
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
        bx_q = fake_quantize_per_channel(bx_reordered.unsqueeze(0), exp2_bx, zp=0,
                                         bitwidth=32, symmetric=True).squeeze(0)
        br_q = fake_quantize_per_channel(br_reordered.unsqueeze(0), exp2_br, zp=0,
                                         bitwidth=32, symmetric=True).squeeze(0)
        
        # 分割偏置（Haste 格式：z, r, n）
        bx_z, bx_r, bx_n = bx_q.chunk(3)  # 各 [H]
        br_z, br_r, br_n = br_q.chunk(3)  # 各 [H]
        
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
        
        # ========== Wx GEMM（循环外一次性计算）==========
        # [与 CUDA 一致] 计算顺序一致
        # [ONNX 兼容] 使用标准 matmul，推理引擎会替换为 MatMulInteger
        # x_q: [T, B, I], W_q: [3*H, I] -> Wx: [T, B, 3*H]
        Wx_all = torch.matmul(x_q, W_q.t())  # [T, B, 3*H]
        
        # [与 CUDA 一致] GEMM 输出量化
        Wx_all = fake_quantize(Wx_all, exp2_Wx, zp_Wx, bitwidth=self._get_bitwidth('Wx'),
                               symmetric=self._get_symmetric('Wx'))
        
        # 预分配输出张量（ONNX 友好，避免动态列表）
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
            # [与 CUDA 一致] Haste 格式 (z, r, n)
            Wx_z, Wx_r, Wx_n = Wx.chunk(3, dim=1)  # 各 [B, H]
            Rh_z, Rh_r, Rh_n = Rh.chunk(3, dim=1)  # 各 [B, H]
            
            # ========== z 门（Update Gate）==========
            # [与 CUDA 一致] z = sigmoid(Wx_z + Rh_z + bx_z + br_z)
            z_pre = Wx_z + Rh_z + bx_z.unsqueeze(0) + br_z.unsqueeze(0)
            
            # [与 CUDA 一致] 激活前量化
            z_pre = fake_quantize(z_pre, exp2_z_pre, zp_z_pre,
                                  bitwidth=self._get_bitwidth('z_pre'),
                                  symmetric=self._get_symmetric('z_pre'))
            
            # [ONNX 兼容] 使用标准 sigmoid（推理引擎会用量化版本或 LUT）
            z = torch.sigmoid(z_pre)
            
            # [与 CUDA 一致] 激活后量化
            z = fake_quantize(z, exp2_z_out, zp_z_out,
                              bitwidth=self._get_bitwidth('z_out'),
                              symmetric=False)  # sigmoid 输出是 [0,1]，非对称
            
            # ========== r 门（Reset Gate）==========
            # [与 CUDA 一致] r = sigmoid(Wx_r + Rh_r + bx_r + br_r)
            r_pre = Wx_r + Rh_r + bx_r.unsqueeze(0) + br_r.unsqueeze(0)
            
            r_pre = fake_quantize(r_pre, exp2_r_pre, zp_r_pre,
                                  bitwidth=self._get_bitwidth('r_pre'),
                                  symmetric=self._get_symmetric('r_pre'))
            
            # [ONNX 兼容] 使用标准 sigmoid
            r = torch.sigmoid(r_pre)
            
            r = fake_quantize(r, exp2_r_out, zp_r_out,
                              bitwidth=self._get_bitwidth('r_out'),
                              symmetric=False)
            
            # ========== g 门（New Gate / Candidate）==========
            # [与 CUDA 一致] g = tanh(Wx_n + r * (Rh_n + br_n) + bx_n)
            Rh_add_br = Rh_n + br_n.unsqueeze(0)
            
            # [与 CUDA 一致] 中间结果量化
            Rh_add_br = fake_quantize(Rh_add_br, quant_params.exp2_inv_Rh_add_br_,
                                      quant_params.zp_Rh_add_br_,
                                      bitwidth=16, symmetric=True)
            
            rRh = r * Rh_add_br
            
            # [与 CUDA 一致] 乘积量化
            rRh = fake_quantize(rRh, quant_params.exp2_inv_rRh_,
                                quant_params.zp_rRh_,
                                bitwidth=16, symmetric=True)
            
            g_pre = Wx_n + rRh + bx_n.unsqueeze(0)
            
            g_pre = fake_quantize(g_pre, exp2_g_pre, zp_g_pre,
                                  bitwidth=self._get_bitwidth('g_pre'),
                                  symmetric=self._get_symmetric('g_pre'))
            
            # [ONNX 兼容] 使用标准 tanh
            g = torch.tanh(g_pre)
            
            g = fake_quantize(g, exp2_g_out, zp_g_out,
                              bitwidth=self._get_bitwidth('g_out'),
                              symmetric=True)  # tanh 输出是 [-1,1]，对称
            
            # ========== 新隐藏状态 ==========
            # [与 CUDA 一致] h_new = z * h + (1 - z) * g
            # CUDA computeH 分别计算并量化 old_contrib 和 new_contrib
            
            # old_contrib = z * h
            old_contrib = z * h
            old_contrib = fake_quantize(old_contrib, quant_params.exp2_inv_old_contrib_,
                                        quant_params.zp_old_contrib_,
                                        bitwidth=16, symmetric=True)
            
            # new_contrib = (1 - z) * g
            new_contrib = (1 - z) * g
            new_contrib = fake_quantize(new_contrib, quant_params.exp2_inv_new_contrib_,
                                        quant_params.zp_new_contrib_,
                                        bitwidth=16, symmetric=True)
            
            # h_new = old_contrib + new_contrib
            h_new = old_contrib + new_contrib
            
            # [与 CUDA 一致] 输出量化
            h_new = fake_quantize(h_new, exp2_h, zp_h,
                                  bitwidth=self._get_bitwidth('h'),
                                  symmetric=self._get_symmetric('h'))
            
            h = h_new
            
            # 使用索引赋值存储（ONNX 友好）
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
        纯 PyTorch 实现的 GRU 前向传播（用于 ONNX 导出）

        支持单向和双向模式
        
        Note:
            量化模式下使用纯定点计算，与 CUDA 量化实现完全一致
        """
        if self.batch_first:
            input = input.transpose(0, 1).contiguous()

        T, B, I = input.shape
        H = self.hidden_size
        device = input.device

        # 初始状态处理
        h0_forward, h0_reverse = None, None
        if hx is not None:
            expected_layers = self.num_layers * self.num_directions
            expected_shape = (expected_layers, B, H)
            if hx.shape != expected_shape:
                raise ValueError(f"hx 形状应为 {expected_shape}，实际 {hx.shape}")
            h0_forward = hx[0]
            if self.bidirectional:
                h0_reverse = hx[1]

        # 前向方向
        output_forward, h_n_forward = self._forward_python_single_direction(
            input, h0_forward,
            self.weight_ih_l0, self.weight_hh_l0,
            self.bias_ih_l0 if self.bias else None,
            self.bias_hh_l0 if self.bias else None,
            self.quant_params
        )

        if self.bidirectional:
            # 反向方向（输入需要翻转）
            output_reverse, h_n_reverse = self._forward_python_single_direction(
                input.flip(0), h0_reverse,
                self.weight_ih_l0_reverse, self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse if self.bias else None,
                self.bias_hh_l0_reverse if self.bias else None,
                self.quant_params_reverse
            )

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
