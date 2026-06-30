"""
QuantGRU - 支持量化的 GRU 实现

功能特性:
    - 兼容 nn.GRU 接口(支持 batch_first、bidirectional 等参数)
    - 支持任意位宽 (1-32 bit) 混合精度量化推理
    - 支持 MinMax / SQNR / Percentile 校准方法
    - 支持 JSON 配置文件指定各算子的位宽和对称量化设置
    - 支持量化参数导出/导入（JSON 格式，便于部署和调试）
    - 支持 ONNX 导出（标准 GRU 节点）

关键属性:
    - use_quantization: 是否启用量化(默认 False)
    - calibrating: 是否在 forward 中收集校准数据(默认 False)
    - calibration_method: 校准方法 'minmax'|'sqnr'|'percentile'(默认 'minmax')
    - export_mode: 是否使用 ONNX 导出模式(默认 False)

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
    >>> print(config)  # {'bitwidth': 16, 'scale': 0.000061, ...}

调试工具（模块级函数）:
    >>> from quant_gru import print_quant_config, print_quant_params
    >>> print_quant_config(gru)  # 打印所有算子的量化配置
    >>> print_quant_params(gru)  # 打印量化参数详情

ONNX 导出:
    >>> from quant_gru import ensure_quant_gru_onnx_registered
    >>> opset = 18  # 支持 >=13；需与 torch.onnx.export(opset_version=...) 一致
    >>> gru.export_mode = True
    >>> ensure_quant_gru_onnx_registered(opset=opset)
    >>> torch.onnx.export(
    ...     gru, x, "model.onnx",
    ...     opset_version=opset,
    ...     dynamo=False,  # PyTorch 2.x 需要使用 legacy exporter
    ...     custom_opsets={"custom_gru": 1}
    ... )
    >>> gru.export_mode = False
"""

import json
import math
import re
import torch
import torch.nn as nn
from typing import Optional, Tuple
from _version import __version__
from torch.onnx import register_custom_op_symbolic, symbolic_helper

try:
    import gru_interface_binding as gru_ops
except ImportError:
    raise ImportError(
        "gru_interface_binding 模块未找到，请先运行 setup.py 编译 C++ 扩展"
    )

# ============================================================
#                   模块级常量与配置映射
# ============================================================

# ============================================================
#                   梯度缩放配置（TF32 精度优化）
# ============================================================
# 用于解决梯度累积时 TF32 精度不足的问题
# 
# 使用方式：
#   - 默认值：GRAD_SCALE_THRESHOLD = 1e-6（适合 grad_accumulate 10-50）
#   - 自定义：在导入模块后修改此值
#     >>> import quant_gru
#     >>> quant_gru.GRAD_SCALE_THRESHOLD = 1e-5  # 更保守，适合 grad_accumulate < 10
#     >>> quant_gru.GRAD_SCALE_THRESHOLD = 1e-7  # 更激进，适合 grad_accumulate > 50
#
# 阈值选择指南：
#   - 1e-5（保守）：适合 grad_accumulate < 10，确保最高精度
#   - 1e-6（平衡，推荐）：适合 grad_accumulate 10-50，平衡精度和性能
#   - 1e-7（激进）：适合 grad_accumulate > 50，最小化性能影响
#
# 原理：
#   - TF32 有效精度约 11 位（3.3 位小数精度）
#   - 对于 1e-6 的数，相对误差可能达到 1-10%
#   - 阈值应在精度开始明显下降之前触发（相对误差 < 1%）
GRAD_SCALE_THRESHOLD = 1e-6  # 梯度缩放阈值（可配置）
GRAD_TARGET_MAX = 0.5        # 目标最大值（放大后的梯度最大值，确保在 TF32 高精度区间）

# ============================================================
# 统一算子映射表（唯一数据源）
# 格式: "算子名" -> {
#   "bw_attr": 位宽属性名,
#   "sym_attr": 对称量化属性名,
#   "scale_attr": scale 属性名 (scale > 0),
#   "zp_attr": zp 属性名 (None 表示无 zp，如 per-channel 权重),
#   "is_per_channel": 是否 per-channel
# }
def _make_op_info(base_name: str, is_per_channel: bool = False, default_unsigned: bool = False):
    """
    生成算子信息字典（减少重复代码）
    
    Args:
        base_name: 基础属性名（如 "x_", "update_gate_output_"）
        is_per_channel: 是否 per-channel 量化
        default_unsigned: C++ 默认是否 unsigned（False=INT, True=UINT）
    
    属性命名规律（与 C++ quantize_ops_helper.h 一致）:
        - bw_attr: "{base_name}" (位宽)
        - sym_attr: "{base_name}symmetric_" (对称量化)
        - unsigned_attr: "{base_name}unsigned_" (无符号量化，只标记例外)
        - scale_attr: "scale_{base_name}" (量化 scale)
        - zp_attr: "zp_{base_name}" (零点，per-channel 为 None)
    """
    return {
        "bw_attr": base_name,
        "sym_attr": f"{base_name}symmetric_",
        "unsigned_attr": f"{base_name}unsigned_",
        "scale_attr": f"scale_{base_name}",
        "zp_attr": None if is_per_channel else f"zp_{base_name}",
        "is_per_channel": is_per_channel,
        "default_unsigned": default_unsigned,
    }


# 算子映射表：JSON 字段名 → C++ 属性名
# 命名与 C++ quantize_ops_helper.h 文档对齐
_OPERATOR_MAP = {
    # 输入
    "input": _make_op_info("x_"),
    # 隐藏状态输出（每时间步的输出，同时作为下一时间步的输入）
    "output": _make_op_info("h_"),
    # 权重（per-channel）
    "weight_ih": _make_op_info("W_", is_per_channel=True),
    "weight_hh": _make_op_info("R_", is_per_channel=True),
    "bias_ih": _make_op_info("bw_", is_per_channel=True),
    "bias_hh": _make_op_info("br_", is_per_channel=True),
    # Linear 输出 (GEMM+bias 融合)
    "weight_ih_linear": _make_op_info("weight_ih_linear_"),  # W*x + bw
    "weight_hh_linear": _make_op_info("weight_hh_linear_"),  # R*h + br
    # 门控（激活前 input / 激活后 output）
    "update_gate_input": _make_op_info("update_gate_input_"),
    "update_gate_output": _make_op_info("update_gate_output_", default_unsigned=True),  # Sigmoid [0,1] → UINT
    "reset_gate_input": _make_op_info("reset_gate_input_"),
    "reset_gate_output": _make_op_info("reset_gate_output_", default_unsigned=True),  # Sigmoid [0,1] → UINT
    "new_gate_input": _make_op_info("new_gate_input_"),
    "new_gate_output": _make_op_info("new_gate_output_"),  # Tanh [-1,1]
    # 中间操作
    "mul_reset_hidden": _make_op_info("mul_reset_hidden_"),        # r * weight_hh_linear
    "mul_old_contribution": _make_op_info("mul_old_contribution_"),  # u * h
    "mul_new_contribution": _make_op_info("mul_new_contribution_"),  # (1-u) * n
}

# 派生常量：从映射表提取的 C++ 属性名集合
_VALID_BITWIDTH_ATTRS = {info["bw_attr"] for info in _OPERATOR_MAP.values()}
_VALID_SYMMETRIC_ATTRS = {info["sym_attr"] for info in _OPERATOR_MAP.values()}
_VALID_UNSIGNED_ATTRS = {info["unsigned_attr"] for info in _OPERATOR_MAP.values()}

# JSON key 到属性映射（从 _OPERATOR_MAP 提取，避免重复构建）
# 由于所有算子名称都不使用前缀，直接使用键名作为 JSON key
# 例如: "input" -> {"bw_attr": "x_", "sym_attr": "x_symmetric_", ...}
# 例如: "weight_ih" -> {"bw_attr": "W_", "sym_attr": "W_symmetric_", ...}
_OPERATOR_SHORT_NAME_MAP = {
    op_name: {  # 直接使用键名，不再需要 split('.')[-1]
        'bw_attr': info["bw_attr"],
        'sym_attr': info["sym_attr"],
        'unsigned_attr': info.get("unsigned_attr"),
        'scale_attr': info["scale_attr"],
        'zp_attr': info["zp_attr"],
        'is_per_channel': info["is_per_channel"],
    }
    for op_name, info in _OPERATOR_MAP.items()
}

# 对称量化属性分类（用于 set_all_bitwidth）
# - 权重/偏置：始终使用对称量化（zero_point=0），计算效率更高
# - 激活值：可配置，非对称量化可能提高精度但增加计算开销
_WEIGHT_SYMMETRIC_ATTRS = {'W_symmetric_', 'R_symmetric_', 'bw_symmetric_', 'br_symmetric_'}
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
#                ONNX 导出（单节点 GRU）辅助逻辑
# ============================================================

_QUANT_GRU_ONNX_LIB = None
QUANT_GRU_ONNX_DOMAIN = "custom_gru"
QUANT_GRU_ONNX_OPSET_VERSION = 1


def get_quant_gru_custom_opsets() -> dict:
    """返回 QuantGRU ONNX 导出所需 custom_opsets。"""
    return {QUANT_GRU_ONNX_DOMAIN: QUANT_GRU_ONNX_OPSET_VERSION}


def ensure_quant_gru_onnx_registered(opset: int = 18) -> None:
    """
    注册 QuantGRU ONNX 导出所需的 custom op 与 symbolic。

    说明：
        - 支持 opset>=13（Squeeze axes 使用输入形式）
        - runtime custom op 只定义一次；symbolic 按 opset 分别注册
        - 注册成功后对同一 opset 重复调用会静默返回
    """
    opset = int(opset)
    if opset < 13:
        raise ValueError(
            f"ensure_quant_gru_onnx_registered 仅支持 opset>=13，当前为 {opset}"
        )

    global _QUANT_GRU_ONNX_LIB
    if _QUANT_GRU_ONNX_LIB is None:
        _QUANT_GRU_ONNX_LIB = torch.library.Library(QUANT_GRU_ONNX_DOMAIN, "FRAGMENT")

    if not getattr(ensure_quant_gru_onnx_registered, "_runtime_done", False):
        _QUANT_GRU_ONNX_LIB.define(
            "quant_gru(Tensor x, Tensor h0, Tensor W, Tensor R, Tensor B, "
            "int hidden_size, int num_layers) -> (Tensor, Tensor)"
        )
        _QUANT_GRU_ONNX_LIB.define(
            "quant_bigru(Tensor x, Tensor h0, Tensor W, Tensor R, Tensor B, "
            "int hidden_size, int num_layers) -> (Tensor, Tensor)"
        )

        def _gru_cpu(x, h0, W, R, B, hidden_size, num_layers):
            t, b = x.shape[0], x.shape[1]
            out = x.new_zeros(t, b, int(hidden_size))
            h_n = x.new_zeros(int(num_layers), b, int(hidden_size))
            return out, h_n

        def _bigru_cpu(x, h0, W, R, B, hidden_size, num_layers):
            t, b = x.shape[0], x.shape[1]
            out = x.new_zeros(t, b, 2 * int(hidden_size))
            h_n = x.new_zeros(2 * int(num_layers), b, int(hidden_size))
            return out, h_n

        try:
            _QUANT_GRU_ONNX_LIB.impl("quant_gru", _gru_cpu, dispatch_key="CPU")
        except TypeError:
            _QUANT_GRU_ONNX_LIB.impl("quant_gru", "CPU", _gru_cpu)

        try:
            _QUANT_GRU_ONNX_LIB.impl("quant_bigru", _bigru_cpu, dispatch_key="CPU")
        except TypeError:
            _QUANT_GRU_ONNX_LIB.impl("quant_bigru", "CPU", _bigru_cpu)

        ensure_quant_gru_onnx_registered._runtime_done = True

    registered_opsets = getattr(ensure_quant_gru_onnx_registered, "_registered_opsets", set())
    if opset in registered_opsets:
        return

    def _symbolic_gru(g, x, h0, W, R, B, hidden_size, num_layers):
        hidden_size_i = symbolic_helper._maybe_get_const(hidden_size, "i")
        num_layers_i = symbolic_helper._maybe_get_const(num_layers, "i")
        if hidden_size_i is None or num_layers_i is None:
            raise RuntimeError(f"{QUANT_GRU_ONNX_DOMAIN}::quant_gru 的 hidden_size/num_layers 必须为常量")

        Y, Y_h = g.op(
            "GRU",
            x,
            W,
            R,
            B,
            symbolic_helper._optional_input_placeholder_tensor(g),
            h0,
            hidden_size_i=int(hidden_size_i),
            linear_before_reset_i=1,
            outputs=2,
        )
        axes = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        out = g.op("Squeeze", Y, axes)
        return out, Y_h

    def _symbolic_bigru(g, x, h0, W, R, B, hidden_size, num_layers):
        hidden_size_i = symbolic_helper._maybe_get_const(hidden_size, "i")
        num_layers_i = symbolic_helper._maybe_get_const(num_layers, "i")
        if hidden_size_i is None or num_layers_i is None:
            raise RuntimeError(f"{QUANT_GRU_ONNX_DOMAIN}::quant_bigru 的 hidden_size/num_layers 必须为常量")

        Y, Y_h = g.op(
            "GRU",
            x,
            W,
            R,
            B,
            symbolic_helper._optional_input_placeholder_tensor(g),
            h0,
            hidden_size_i=int(hidden_size_i),
            direction_s="bidirectional",
            linear_before_reset_i=1,
            outputs=2,
        )

        y_perm = g.op("Transpose", Y, perm_i=[0, 2, 1, 3])
        shape = g.op("Constant", value_t=torch.tensor([0, 0, -1], dtype=torch.long))
        out = g.op("Reshape", y_perm, shape)
        return out, Y_h

    register_custom_op_symbolic(f"{QUANT_GRU_ONNX_DOMAIN}::quant_gru", _symbolic_gru, opset)
    register_custom_op_symbolic(f"{QUANT_GRU_ONNX_DOMAIN}::quant_bigru", _symbolic_bigru, opset)
    registered_opsets.add(opset)
    ensure_quant_gru_onnx_registered._registered_opsets = registered_opsets


def _get_quant_gru_registered_onnx_opsets() -> set:
    """返回当前进程中已注册 QuantGRU symbolic 的 ONNX opset 集合。"""
    return set(getattr(ensure_quant_gru_onnx_registered, "_registered_opsets", set()))


def _rename_onnx_initializer_and_refs(model, old_name: str, new_name: str) -> bool:
    if old_name == new_name:
        return False
    renamed = False
    for init in model.graph.initializer:
        if init.name == old_name:
            init.name = new_name
            renamed = True
            break
    if not renamed:
        return False
    for node in model.graph.node:
        node.input[:] = [new_name if n == old_name else n for n in node.input]
        node.output[:] = [new_name if n == old_name else n for n in node.output]
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == old_name:
            vi.name = new_name
    return True


def _rename_onnx_node_io_refs(model, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    for node in model.graph.node:
        node.input[:] = [new_name if n == old_name else n for n in node.input]
        node.output[:] = [new_name if n == old_name else n for n in node.output]
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == old_name:
            vi.name = new_name


def _onnx_module_value_prefix_for_baseline(gru_name: str) -> str:
    parts = [p for p in str(gru_name).split(".") if p]
    if not parts:
        return "/gru"
    if len(parts) == 1:
        return f"/{parts[0]}"
    prefix = f"/{parts[0]}/{parts[0]}.{parts[1]}"
    if len(parts) > 2:
        prefix += "/" + "/".join(parts[2:])
    return prefix


def _reorder_onnx_initializers_for_unidir_gru(model, gru_name: str) -> None:
    target_order = [
        f"{gru_name}.weight_ih.weight",
        f"{gru_name}.weight_hh.weight",
        f"{gru_name}.bias",
        f"{gru_name}.gru#1.initial_h",
        f"/{gru_name}/gru/Constant_output_0",
    ]
    current = list(model.graph.initializer)
    name_to_init = {i.name: i for i in current}
    if not all(name in name_to_init for name in target_order):
        return
    ordered = [name_to_init[name] for name in target_order]
    remaining = [i for i in current if i.name not in set(target_order)]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(ordered + remaining)


def normalize_quant_gru_onnx_to_optimized_baseline(onnx_path: str) -> None:
    """
    将 QuantGRU 直接导出的 ONNX GRU 局部命名规范化为 OptimizedQuantizableGRU baseline 风格。

    PyTorch legacy ONNX exporter 对节点名和中间 value 名的源头控制有限；这里把
    QuantGRU 自身的导出外观收敛在 QuantGRU 模块内，避免上层导出脚本持有 GRU
    内部命名知识。全模型/AIMET 级后处理仍应留在调用方。
    """
    import onnx

    model = onnx.load(onnx_path)
    changed = False

    for gru_node in model.graph.node:
        if gru_node.op_type != "GRU":
            continue

        gru_name = gru_node.name or "gru"
        value_prefix = _onnx_module_value_prefix_for_baseline(gru_name)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in gru_node.attribute}
        direction = attrs.get("direction", b"forward")
        if isinstance(direction, bytes):
            direction = direction.decode("utf-8")
        is_bidir = direction == "bidirectional"

        if len(gru_node.input) >= 5:
            seq_len_old = gru_node.input[4]
            if seq_len_old and re.match(r"^/Constant_\d+_output_0$", seq_len_old):
                seq_len_old = ""
            if seq_len_old:
                seq_len_new = f"/{gru_name}/Constant_2_output_0" if is_bidir else f"{value_prefix}/gru/Constant_output_0"
                if _rename_onnx_initializer_and_refs(model, seq_len_old, seq_len_new):
                    gru_node.input[4] = seq_len_new
                    changed = True

        if len(gru_node.input) >= 6:
            h0_old = gru_node.input[5]
            if h0_old:
                h0_new = f"{gru_name}#6.initial_h" if is_bidir else f"{gru_name}.gru#1.initial_h"
                if _rename_onnx_initializer_and_refs(model, h0_old, h0_new):
                    gru_node.input[5] = h0_new
                    changed = True

        if is_bidir:
            const_pat = re.compile(rf"^/{re.escape(gru_name)}/Constant_\d+_output_0$")
            for init in model.graph.initializer:
                if const_pat.match(init.name):
                    target = f"/{gru_name}/Constant_2_output_0"
                    if _rename_onnx_initializer_and_refs(model, init.name, target):
                        changed = True
            continue

        x_name = gru_node.input[0] if len(gru_node.input) > 0 else ""
        y_name = gru_node.output[0] if len(gru_node.output) > 0 else ""
        transpose_node = None
        squeeze_node = None
        tail_transpose = None
        for node in model.graph.node:
            if node is gru_node:
                continue
            if x_name and x_name in node.output and node.op_type == "Transpose":
                transpose_node = node
            if y_name and y_name in node.input and node.op_type == "Squeeze":
                squeeze_node = node
        if squeeze_node is not None and len(squeeze_node.output) > 0:
            sq_out = squeeze_node.output[0]
            for node in model.graph.node:
                if node is squeeze_node:
                    continue
                if sq_out and sq_out in node.input and node.op_type == "Transpose":
                    tail_transpose = node
                    break

        if transpose_node is not None and transpose_node.name != f"{gru_name}.gru":
            transpose_node.name = f"{gru_name}.gru"
            changed = True
        if transpose_node is not None and len(transpose_node.output) >= 1:
            t_old = transpose_node.output[0]
            t_new = f"{value_prefix}/gru/Transpose_output_0"
            if t_old and t_old != t_new:
                _rename_onnx_node_io_refs(model, t_old, t_new)
                changed = True

        if squeeze_node is not None:
            if squeeze_node.name != f"{gru_name}.gru#2":
                squeeze_node.name = f"{gru_name}.gru#2"
                changed = True
            if len(squeeze_node.input) >= 2:
                axes_old = squeeze_node.input[1]
                axes_new = f"{value_prefix}/gru/Constant_output_0"
                if axes_old and re.match(r"^/Constant_\d+_output_0$", axes_old):
                    axes_old = ""
                if axes_old and _rename_onnx_initializer_and_refs(model, axes_old, axes_new):
                    squeeze_node.input[1] = axes_new
                    changed = True
                elif axes_old and axes_old != axes_new:
                    _rename_onnx_node_io_refs(model, axes_old, axes_new)
                    squeeze_node.input[1] = axes_new
                    changed = True
            if len(squeeze_node.output) >= 1:
                sq_old = squeeze_node.output[0]
                sq_new = f"{value_prefix}/gru#2/Squeeze_output_0"
                if sq_old and sq_old != sq_new:
                    _rename_onnx_node_io_refs(model, sq_old, sq_new)
                    changed = True

        if tail_transpose is not None and tail_transpose.name != f"{gru_name}.gru#3.end":
            tail_transpose.name = f"{gru_name}.gru#3.end"
            changed = True

        if len(gru_node.output) >= 2:
            y0_old = gru_node.output[0]
            y0_new = f"{value_prefix}/gru#1/GRU_output_0"
            if y0_old and y0_old != y0_new:
                _rename_onnx_node_io_refs(model, y0_old, y0_new)
                changed = True
            y1_old = gru_node.output[1]
            y1_new = f"{value_prefix}/gru#1/GRU_output_1"
            if y1_old and y1_old != y1_new:
                _rename_onnx_node_io_refs(model, y1_old, y1_new)
                changed = True

        if tail_transpose is not None and len(tail_transpose.output) >= 1:
            tp_old = tail_transpose.output[0]
            tp_new = f"{value_prefix}/gru#3.end/Transpose_1_output_0"
            graph_outputs = {o.name for o in model.graph.output}
            if tp_old and tp_old != tp_new and tp_old not in graph_outputs:
                _rename_onnx_node_io_refs(model, tp_old, tp_new)
                changed = True

        if any(a.name == "direction" for a in gru_node.attribute):
            new_attrs = [a for a in gru_node.attribute if a.name != "direction"]
            del gru_node.attribute[:]
            gru_node.attribute.extend(new_attrs)
            changed = True

        _reorder_onnx_initializers_for_unidir_gru(model, gru_name)

    if changed:
        onnx.save(model, onnx_path)


def prune_quant_gru_raw_l0_param_encodings(enc_out_path: str) -> None:
    """删除 AIMET 原生 QuantGRU *_l0 参数键，保留 ONNX baseline 风格参数键。"""
    with open(enc_out_path, "r", encoding="utf-8") as f:
        enc = json.load(f)

    param_encodings = enc.get("param_encodings", {})
    if not isinstance(param_encodings, dict):
        return

    pattern = re.compile(
        r".*\.(weight_ih_l0(_reverse)?|weight_hh_l0(_reverse)?|bias_ih_l0(_reverse)?|bias_hh_l0(_reverse)?)$"
    )
    remove_keys = [key for key in param_encodings.keys() if pattern.match(key)]
    if not remove_keys:
        return

    for key in remove_keys:
        param_encodings.pop(key, None)
    enc["param_encodings"] = param_encodings

    with open(enc_out_path, "w", encoding="utf-8") as f:
        json.dump(enc, f, indent=2, ensure_ascii=False)


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
        (W, R, bw, br): Haste 格式的权重和偏置
            - W: [I, 3*H] 转置后的输入权重
            - R: [H, 3*H] 转置后的循环权重
            - bw: [3*H] 输入偏置 (bias for W)
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
        bw = reorder_weights_pytorch_to_haste(bias_ih).contiguous()
        br = reorder_weights_pytorch_to_haste(bias_hh).contiguous()
    else:
        bw = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)
        br = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)

    return W, R, bw, br


# ============================================================
#                      量化通用工具函数
# ============================================================

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
        
    QAT 支持：
        - 量化训练时，forward 会返回 clamp mask
        - backward 使用 mask 将被 clamp 的梯度置零（Straight-Through Estimator）
    """

    @staticmethod
    def forward(ctx, input, weight_ih, weight_hh, bias_ih, bias_hh, h0, is_training,
                use_quantization=False, quant_params=None, quant_storage_dtype="float32"):
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
        ctx.use_quantization = use_quantization
        ctx.is_training = is_training
        ctx.quant_storage_dtype = quant_storage_dtype

        device = input.device if input.is_cuda else torch.device('cuda')
        input = ensure_cuda_float32(input, device)

        # 权重格式转换(使用统一工具函数)
        W, R, bw, br = convert_weights_to_haste_format(
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
            quant_params = gru_ops.GRUQuantParams()

        # 根据 use_quantization 调用不同的前向函数
        if use_quantization:
            # 量化模式：按 quant_storage_dtype 选择量化值存储类型
            #   float32: 量化值用 float32 存储
            #   int32:   量化值用 int32 存储（纯定点核心）
            if quant_storage_dtype == "int32":
                forward_quant_fn = gru_ops.forward_quant_int_storage
            else:
                forward_quant_fn = gru_ops.forward_quant_float_storage
            (output_full, v,
             W_q, R_q, bw_q, br_q, x_q,
             x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
             weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask) = forward_quant_fn(
                is_training=is_training,
                time_steps=time_steps,
                batch_size=batch_size,
                input_size=input_size,
                hidden_size=hidden_size,
                W=W,
                R=R,
                bw=bw,
                br=br,
                x=input,
                h0=h0_tensor,
                quant_params=quant_params
            )
        else:
            # 浮点模式：调用 forward_fp，不返回 mask
            output_full, v = gru_ops.forward_fp(
                is_training=is_training,
                time_steps=time_steps,
                batch_size=batch_size,
                input_size=input_size,
                hidden_size=hidden_size,
                W=W,
                R=R,
                bw=bw,
                br=br,
                x=input,
                h0=h0_tensor
            )
            # 浮点模式：不保存量化值和 mask，所以不需要创建

        # 分离输出: output_full[0] 是初始状态，[1:] 是时间步输出
        output = output_full[1:]
        h_n = output_full[-1:]

        # 保存反向传播所需的中间结果
        # 根据 use_quantization 区分保存内容，减少内存占用
        if use_quantization:
            # 量化模式：只保存量化值、output_full、v 和 mask
            # backward_quant_wrapper 只需要量化值，不需要原始 W, R, bw, br, input
            # x_q 反量化后就是 input 的量化版本，可以直接使用
            # 顺序：(W_q, R_q, bw_q, br_q, x_q, output_full, v, ...masks)
            ctx.save_for_backward(W_q, R_q, bw_q, br_q, x_q,
                                  output_full, v,
                                  x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
                                  weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask)
        else:
            # 浮点模式：只保存原始值，不保存量化值和 mask
            # 顺序：(W, R, bw, br, input, output_full, v)
            ctx.save_for_backward(W, R, bw, br, input, output_full, v)
        
        # 保存量化参数
        ctx.quant_params = quant_params

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
            
        QAT 说明：
            使用 Straight-Through Estimator (STE)：
            - 被 clamp 的值（mask=1）梯度置零
            - 未被 clamp 的值（mask=0）梯度正常传播
            - 所有 mask 在 C++ 端应用，提高效率
        """
        # 从 saved_tensors 中提取值（根据 use_quantization 区分顺序）
        time_steps, batch_size = ctx.time_steps, ctx.batch_size
        input_size, hidden_size = ctx.input_size, ctx.hidden_size
        
        if ctx.use_quantization:
            # 量化模式：顺序 (W_q, R_q, bw_q, br_q, x_q, output_full, v, ...masks)
            (W_q, R_q, bw_q, br_q, x_q,
             h, v,
             x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
             weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask) = ctx.saved_tensors
        else:
            # 浮点模式：顺序 (W, R, bw, br, input, output_full, v)
            (W, R, bw, br, input, h, v) = ctx.saved_tensors

        # 确保所有张量在 CUDA 上
        device = grad_output.device
        if ctx.use_quantization:
            # 量化模式：只需要处理 h、v（量化值会在 backward_quant 中处理）
            h = h.to(device) if not h.is_cuda else h
            if v is not None and not v.is_cuda:
                v = v.to(device)
        else:
            # 浮点模式：需要处理 W, R, bw, br, input, h
            tensors = [W, R, bw, br, input, h]
            W, R, bw, br, input, h = [t.to(device) if not t.is_cuda else t for t in tensors]
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

        # ========== 梯度缩放处理（解决 TF32 精度问题）==========
        # 问题：梯度累积时梯度被除以 grad_accumulate，导致数值过小，TF32 精度包不住
        # 解决：检测梯度过小时，内部放大梯度进行计算，然后按比例缩小返回
        # 
        # 使用模块级配置：GRAD_SCALE_THRESHOLD 和 GRAD_TARGET_MAX
        # 可在导入后修改：quant_gru.GRAD_SCALE_THRESHOLD = 1e-5
        # 
        # 检测梯度最大值
        # grad_max = torch.max(torch.abs(dh_new)).item()
        # use_grad_scale = grad_max > 0 and grad_max < GRAD_SCALE_THRESHOLD
        # 
        # if use_grad_scale:
        #     # 计算缩放因子：将梯度最大值放大到目标范围（约 0.5）
        #     # 使用 2 的幂次，避免浮点误差累积
        #     import math
        #     # 计算需要的缩放倍数：target / current
        #     scale_ratio = GRAD_TARGET_MAX / grad_max
        #     # 使用 2 的幂次，向上取整到最近的 2 的幂次（避免浮点误差）
        #     # 例如：scale_ratio=1.5e6 -> log2=20.5 -> 2^21=2097152
        #     scale_power = math.ceil(math.log2(scale_ratio))
        #     scale_factor = float(1 << scale_power)  # 2^scale_power
        #     # 放大梯度
        #     dh_new_scaled = dh_new * scale_factor
        # else:
        #     scale_factor = 1.0
        #     dh_new_scaled = dh_new

        # 根据 use_quantization 调用不同的反向函数
        if ctx.use_quantization:
            # 量化模式：使用保存的量化值，调用 backward_quant
            # 确保量化值在正确的设备上（量化模式下一定有量化值，因为已经区分保存了）
            W_q = W_q.to(device) if not W_q.is_cuda else W_q
            R_q = R_q.to(device) if not R_q.is_cuda else R_q
            bw_q = bw_q.to(device) if not bw_q.is_cuda else bw_q
            br_q = br_q.to(device) if not br_q.is_cuda else br_q
            x_q = x_q.to(device) if not x_q.is_cuda else x_q
            
            # 确保 mask 在正确的设备上
            masks = [x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask,
                     weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask]
            x_mask, h0_mask, W_mask, R_mask, bw_mask, br_mask, \
            weight_ih_linear_mask, weight_hh_linear_mask, gate_input_mask, gate_output_mask, h_mask = \
                [m.to(device) if m.numel() > 0 and not m.is_cuda else m for m in masks]
            
            # 按 quant_storage_dtype 选择反向函数：int32 接收 int32 量化值，内部转 float 后复用
            if getattr(ctx, "quant_storage_dtype", "float32") == "int32":
                backward_quant_fn = gru_ops.backward_quant_int_storage
            else:
                backward_quant_fn = gru_ops.backward_quant_float_storage
            dx, dW, dR, dbw, dbr, dh = backward_quant_fn(
                time_steps=time_steps, batch_size=batch_size,
                input_size=input_size, hidden_size=hidden_size,
                W_q=W_q, R_q=R_q, bw_q=bw_q, br_q=br_q, x_q=x_q,
                dh_new=dh_new, h=h, v=v,
                quant_params=ctx.quant_params,
                # QAT masks（C++ 端应用 STE）
                x_mask=x_mask,
                h0_mask=h0_mask,
                W_mask=W_mask,
                R_mask=R_mask,
                bw_mask=bw_mask,
                br_mask=br_mask,
                weight_ih_linear_mask=weight_ih_linear_mask,
                weight_hh_linear_mask=weight_hh_linear_mask,
                gate_input_mask=gate_input_mask,
                gate_output_mask=gate_output_mask,
                h_mask=h_mask
            )
        else:
            # 浮点模式：调用 backward_fp，不需要 mask
            dx, dW, dR, dbw, dbr, dh = gru_ops.backward_fp(
                time_steps=time_steps, batch_size=batch_size,
                input_size=input_size, hidden_size=hidden_size,
                W=W, R=R, bw=bw, br=br, x=input,
                dh_new=dh_new, h=h, v=v
            )

        # 如果使用了梯度缩放，需要将返回的梯度按比例缩小
        # if use_grad_scale:
        #     dx = dx / scale_factor
        #     dW = dW / scale_factor
        #     dR = dR / scale_factor
        #     if dbw is not None:
        #         dbw = dbw / scale_factor
        #     if dbr is not None:
        #         dbr = dbr / scale_factor
        #     if dh is not None:
        #         dh = dh / scale_factor

        # 梯度格式转换: Haste (z,r,n) -> PyTorch (r,z,n)
        dW_pytorch = reorder_weights_haste_to_pytorch(dW.t()).contiguous()
        dR_pytorch = reorder_weights_haste_to_pytorch(dR.t()).contiguous()
        dbw_pytorch = reorder_weights_haste_to_pytorch(dbw).contiguous() if not ctx.bias_ih_is_none else None
        dbr_pytorch = reorder_weights_haste_to_pytorch(dbr).contiguous() if not ctx.bias_hh_is_none else None
        grad_h0 = None if ctx.h0_is_none else dh

        # 返回梯度(对应 forward 的 10 个参数：最后三个 None 对应
        # is_training/use_quantization/quant_params，第 10 个 None 对应 quant_storage_dtype)
        return dx, dW_pytorch, dR_pytorch, dbw_pytorch, dbr_pytorch, grad_h0, None, None, None, None


# ============================================================
#                      QuantGRU 核心模块
# ============================================================
#
# QuantGRU 是本模块的核心类，提供：
#   - 兼容 nn.GRU 的接口
#   - 任意位宽 (1-32 bit) 混合精度量化推理
#   - 多种校准方法（MinMax/SQNR/Percentile）
#   - ONNX 导出支持（float 格式）
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
        export_mode (bool): ONNX 导出模式，True 时使用标准 GRU 节点导出路径
    
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
            use_pot2_scale: bool = False,
            quant_storage_dtype: str = "float32",
    ):
        super(QuantGRU, self).__init__()

        if num_layers != 1:
            raise NotImplementedError("仅支持 num_layers=1")
        if dropout > 0:
            raise NotImplementedError("暂不支持 dropout")
        if quant_storage_dtype not in ("float32", "int32"):
            raise ValueError(
                f"quant_storage_dtype 仅支持 'float32' 或 'int32'，当前为 {quant_storage_dtype}"
            )

        # 基本配置
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_quantization = use_quantization
        self._use_pot2_scale = bool(use_pot2_scale)
        self.num_directions = 2 if bidirectional else 1

        # 量化值存储类型：
        #   'float32': 量化值用 float32 存储（走 forward_quant_float_storage）
        #   'int32':   量化值用 int32 存储（走 forward_quant_int_storage）
        self.quant_storage_dtype = quant_storage_dtype

        # ONNX 导出开关：True 时启用单节点 GRU 导出路径
        self.export_mode = False
        # ONNX 导出临时权重缓存（non-persistent，避免污染 state_dict）
        self.register_buffer('_onnx_export_weight_ih', torch.empty(0), persistent=False)
        self.register_buffer('_onnx_export_weight_hh', torch.empty(0), persistent=False)
        self.register_buffer('_onnx_export_bias', torch.empty(0), persistent=False)

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
        # quant_params 锁：
        # True: 锁定，禁止后续校准重算覆盖；False: 允许覆盖。
        self._quant_params_locked = False

        # 位宽配置对象（直接初始化，避免延迟创建的线程安全问题）
        self._bitwidth_config = gru_ops.OperatorQuantConfig()  # 位宽配置(直接存储 C++ 对象)
        self._bitwidth_config.usePOT2_ = self._use_pot2_scale

        self._cublas_initialized = False  # CUDA 延迟初始化标志
        
        # 模块完整路径名称（用于调试信息和 AIMET/ONNX 名称匹配）
        # 例如 "generator.enc_seqs.0.seq_t" 或 AIMET 在 ONNX 中分配的名称
        self._module_name = None

        # 校准方法:
        #   - 'minmax': 使用 min/max 范围(快速，无直方图)
        #   - 'sqnr': SQNR 优化搜索最优 scale(基于直方图，高精度)
        #   - 'percentile': 百分位裁剪(基于直方图)
        self.calibration_method = 'minmax'

        # Percentile 配置(仅 calibration_method='percentile' 时使用)
        self.percentile_value = 100.0

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
            bitwidth_dict['usePOT2_'] = getattr(self._bitwidth_config, 'usePOT2_', self._use_pot2_scale)
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
        if '_quant_params_locked' not in state:
            if '_quant_params_allow_overwrite' in state:
                state['_quant_params_locked'] = not bool(state['_quant_params_allow_overwrite'])
            else:
                state['_quant_params_locked'] = False
        state.pop('_quant_params_allow_overwrite', None)

        # 兼容旧 pickle：缺省量化值存储类型为 float32
        if 'quant_storage_dtype' not in state:
            state['quant_storage_dtype'] = "float32"
        
        self.__dict__.update(state)
        self._use_pot2_scale = bool(getattr(self._bitwidth_config, 'usePOT2_', getattr(self, '_use_pot2_scale', False)))

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
        # 如果已加载量化参数，跳过此方法
        if self.is_calibrated():
            if verbose:
                module_name_str = f" [{self._module_name}]" if self._module_name else ""
                print(
                    f" [QuantGRU{module_name_str}] 已完成校准或已加载量化参数，跳过 load_bitwidth_config()。\n"
                )
            return

        # 解析 JSON 文件
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 读取 GRU_config 节点下的配置
        gru_config = data.get('GRU_config', {})

        # 读取全局配置
        default_config = gru_config.get('default_config', {})
        if 'disable_quantization' in default_config:
            self.use_quantization = not default_config['disable_quantization']
        if 'use_pot2_scale' in default_config:
            self.use_pot2_scale = bool(default_config['use_pot2_scale'])

        # 直接将配置写入 C++ 对象
        op_config = gru_config.get('operator_config', {})

        # ========================================================================
        # JSON key -> _OPERATOR_MAP key 映射
        # 所有算子名称统一不使用前缀，JSON key 直接对应 _OPERATOR_MAP 的键名
        # 例如: JSON 中的 "input" 对应 _OPERATOR_MAP 的 "input"
        # 例如: JSON 中的 "weight_ih" 对应 _OPERATOR_MAP 的 "weight_ih"
        # 
        # 向后兼容：支持旧的字段名映射
        # 旧字段名 -> 新字段名
        # ========================================================================
        # 字段名映射表（向后兼容旧 JSON 配置）
        FIELD_NAME_MAP = {
            "W": "weight_ih",
            "R": "weight_hh",
            "bw": "bias_ih",
            "br": "bias_hh",
            "x": "input",  # 如果 JSON 中使用 "x" 而不是 "input"
            "h": "output",  # 如果 JSON 中使用 "h" 而不是 "output"
        }
        
        # 应用字段名映射（将旧字段名转换为新字段名）
        normalized_op_config = {}
        for json_key, op_cfg in op_config.items():
            # 如果字段名在映射表中，使用新字段名
            if json_key in FIELD_NAME_MAP:
                new_key = FIELD_NAME_MAP[json_key]
                if verbose:
                    print(f"  [字段名映射] '{json_key}' -> '{new_key}' (向后兼容)")
                normalized_op_config[new_key] = op_cfg
            else:
                normalized_op_config[json_key] = op_cfg
        
        # 使用标准化后的配置
        op_config = normalized_op_config
        
        valid_json_keys = set(_OPERATOR_MAP.keys())
        json_op_names = set(op_config.keys())
        unknown_fields = json_op_names - valid_json_keys
        
        if unknown_fields:
            raise ValueError(
                f"JSON 配置文件 '{config_file}' 包含未知的算子字段:\n"
                f"  {list(unknown_fields)}\n"
                f"有效的字段名:\n"
                f"  {sorted(valid_json_keys)}"
            )

        # 检查 JSON 中缺失的字段并发出警告
        missing_fields = []  # [(json_key, default_bitwidth, default_symmetric, default_unsigned), ...]
        missing_attrs = []   # [(json_key, attr_name, default_value), ...] - 算子存在但属性缺失
        
        for json_key, info in _OPERATOR_MAP.items():
            # json_key 直接就是 _OPERATOR_MAP 的键名，不需要转换
            bw_attr, sym_attr = info["bw_attr"], info["sym_attr"]
            unsigned_attr = info.get("unsigned_attr")
            default_unsigned = info.get("default_unsigned", False)
            
            if json_key in op_config:
                op_cfg = op_config[json_key]
                
                # 检查并记录缺失的属性
                if 'bitwidth' not in op_cfg:
                    missing_attrs.append((json_key, 'bitwidth', 8))
                if 'is_symmetric' not in op_cfg:
                    missing_attrs.append((json_key, 'is_symmetric', True))
                if unsigned_attr and 'is_unsigned' not in op_cfg:
                    unsigned_default_str = "true (UINT)" if default_unsigned else "false (INT)"
                    missing_attrs.append((json_key, 'is_unsigned', unsigned_default_str))
                # 检查量化粒度配置（仅对 weight_ih, weight_hh, bias_ih, bias_hh 有效）
                if json_key in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
                    if 'quantization_granularity' not in op_cfg:
                        missing_attrs.append((json_key, 'quantization_granularity', 'PER_CHANNEL'))
                
                bitwidth = op_cfg.get('bitwidth', 8)
                # 验证位宽范围 (1-32)
                if not (1 <= bitwidth <= 32):
                    raise ValueError(
                        f"Invalid bitwidth {bitwidth} for '{json_key}'. "
                        f"Must be in range [1, 32]."
                    )
                setattr(self._bitwidth_config, bw_attr, bitwidth)
                setattr(self._bitwidth_config, sym_attr, op_cfg.get('is_symmetric', True))
                # 设置 unsigned 属性（只标记 UINT 例外）
                if unsigned_attr:
                    setattr(self._bitwidth_config, unsigned_attr, op_cfg.get('is_unsigned', default_unsigned))
                
                # 设置量化粒度配置（仅对 weight_ih, weight_hh, bias_ih, bias_hh 有效）
                if json_key in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
                    granularity_str = op_cfg.get('quantization_granularity', 'PER_CHANNEL')
                    granularity_map = {
                        'PER_TENSOR': 0,
                        'PER_GATE': 1,
                        'PER_CHANNEL': 2
                    }
                    if granularity_str not in granularity_map:
                        raise ValueError(
                            f"Invalid quantization_granularity '{granularity_str}' for '{json_key}'. "
                            f"Must be one of: PER_TENSOR, PER_GATE, PER_CHANNEL"
                        )
                    # ⚠️ 关键修复：使用 bw_attr 来构造 granularity_attr
                    # 例如：weight_ih -> W_ -> W_granularity_
                    granularity_attr = f"{bw_attr}granularity_"
                    setattr(self._bitwidth_config, granularity_attr, granularity_map[granularity_str])
            else:
                # 记录缺失字段及其默认值
                missing_fields.append((json_key, 8, True, default_unsigned))

        # 报告缺失的算子（整个算子缺失）
        if missing_fields:
            missing_details = []
            for op_name, bw, sym, unsigned in missing_fields:
                unsigned_str = "is_unsigned=true (UINT)" if unsigned else "is_unsigned=false (INT)"
                sym_str = "is_symmetric=true" if sym else "is_symmetric=false"
                missing_details.append(
                    f"    缺少字段: '{op_name}'\n"
                    f"       → 将使用默认值: bitwidth={bw}, {sym_str}, {unsigned_str}"
                )
            
            if verbose:
                print(f"\n  提示：JSON 配置文件 '{config_file}' 缺少以下算子配置，将使用默认值:\n"
                      + "\n".join(missing_details) + "\n")
        
        # 报告缺失的属性（算子存在但属性缺失）
        if missing_attrs:
            # 按算子分组
            attr_by_op = {}
            for op_name, attr_name, default_val in missing_attrs:
                if op_name not in attr_by_op:
                    attr_by_op[op_name] = []
                attr_by_op[op_name].append((attr_name, default_val))
            
            attr_details = []
            for op_name, attrs in attr_by_op.items():
                attr_lines = []
                for attr_name, default_val in attrs:
                    attr_lines.append(f"       → 缺少 '{attr_name}'，将使用默认值: {default_val}")
                attr_details.append(f"    算子 '{op_name}':\n" + "\n".join(attr_lines))
            
            if verbose:
                print(f"\n  提示：JSON 配置文件 '{config_file}' 以下算子缺少部分属性，将使用默认值:\n"
                      + "\n".join(attr_details) + "\n")

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
    
    def set_module_name(self, module_name: str) -> None:
        """
        设置模块完整路径名称（用于调试信息和 AIMET/ONNX 名称匹配）
        
        Args:
            module_name: 模块完整路径名称，例如 "generator.enc_seqs.0.seq_t"
        
        Example:
            >>> gru.set_module_name("generator.enc_seqs.0.seq_t")
        """
        self._module_name = module_name

    def set_quant_params_locked(self, locked: bool) -> None:
        """
        设置 quant_params 是否锁定。

        Args:
            locked: True 锁定（禁止覆盖）；False 解锁（允许覆盖）
        """
        self._quant_params_locked = bool(locked)

    def finalize_calibration(self, verbose: bool = False):
        """
        完成校准，计算量化参数并初始化 LUT
        
        Args:
            verbose: 是否打印校准信息
            
        Raises:
            RuntimeError: 未收集校准数据
        """
        if self._quant_params_locked and self.is_calibrated():
            self._quant_params_dirty = False
            if verbose:
                print("\n[QuantGRU] quant_params 已锁定，跳过 finalize_calibration 覆盖")
            return

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
                    use_percentile=use_percentile,
                    percentile_value=self.percentile_value)
            else:
                if self.quant_ranges_reverse is None:
                    raise RuntimeError("双向 GRU 反向校准数据异常")
                self.quant_params_reverse = gru_ops.calculate_gru_quantitative_parameters(
                    quant_ranges=self.quant_ranges_reverse, bitwidth_config=bitwidth_config)

        # 确保 quant_params 中的 bitwidth_config_ 与当前配置同步
        # 这样 print_quant_params 可以正确读取粒度信息
        if self.quant_params is not None:
            self.quant_params.bitwidth_config_ = self._bitwidth_config
        if self.bidirectional and self.quant_params_reverse is not None:
            self.quant_params_reverse.bitwidth_config_ = self._bitwidth_config
        
        # 量化参数已更新，清除脏标志
        self._quant_params_dirty = False
        
        # verbose=True 时打印量化参数
        if verbose:
            print_quant_params(self)

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

    # -------------------- ONNX 导出模式：标准 GRU 节点 --------------------

    def _get_config_attr(self, op_name: str, suffix: str, valid_set: set, default):
        """
        获取配置属性的通用方法
        
        Args:
            op_name: 操作名称（如 'input', 'output', 'weight_ih_linear', 'update_gate_output' 等）
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
        """获取指定操作的位宽（如 'input', 'output', 'weight_ih_linear' 等），无效返回 8"""
        return self._get_config_attr(op_name, '_', _VALID_BITWIDTH_ATTRS, 8)

    def _get_symmetric(self, op_name: str) -> bool:
        """获取指定操作是否对称量化，无效返回 True"""
        return self._get_config_attr(op_name, '_symmetric_', _VALID_SYMMETRIC_ATTRS, True)

    def _get_unsigned(self, op_name: str) -> bool:
        """获取指定操作是否无符号量化（False=INT, True=UINT），无效返回 False"""
        return self._get_config_attr(op_name, '_unsigned_', _VALID_UNSIGNED_ATTRS, False)

    @property
    def use_pot2_scale(self) -> bool:
        """是否使用 POT2 scale 编码；False 时使用 affine/M+shift 编码。"""
        if hasattr(self, '_bitwidth_config') and self._bitwidth_config is not None:
            return bool(self._bitwidth_config.usePOT2_)
        return bool(getattr(self, '_use_pot2_scale', False))

    @use_pot2_scale.setter
    def use_pot2_scale(self, value: bool):
        self._use_pot2_scale = bool(value)
        if hasattr(self, '_bitwidth_config') and self._bitwidth_config is not None:
            self._bitwidth_config.usePOT2_ = self._use_pot2_scale
            if self.quant_params is not None:
                self.quant_params.bitwidth_config_ = self._bitwidth_config
            # use_pot2_scale 切换会改变目标 scale 语义，需要重新校准
            self._quant_params_dirty = True

    def _resolve_onnx_module_name(self) -> str:
        """解析 ONNX 导出时用于可读命名的模块名称。"""
        if self._module_name:
            return str(self._module_name)
        print("  提示：QuantGRU._module_name 未设置，ONNX 导出将使用默认名称 'gru'。"
              "多 GRU 模型建议先调用 set_quant_gru_module_names(model)。")
        return "gru"

    def _pack_onnx_gru_direction_weights(
            self,
            weight_ih: torch.Tensor,
            weight_hh: torch.Tensor,
            bias_ih: torch.Tensor,
            bias_hh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将单方向 PyTorch GRU 参数打包为 ONNX GRU 方向参数。

        Returns:
            W_dir: [3H, I]  (z,r,n)
            R_dir: [3H, H]  (z,r,n)
            B_dir: [6H]     = concat(bias_ih, bias_hh)，其中每项都为 (z,r,n)
        """
        W_dir = reorder_weights_pytorch_to_haste(weight_ih).contiguous()
        R_dir = reorder_weights_pytorch_to_haste(weight_hh).contiguous()
        b_ih_dir = reorder_weights_pytorch_to_haste(bias_ih).contiguous()
        b_hh_dir = reorder_weights_pytorch_to_haste(bias_hh).contiguous()
        B_dir = torch.cat([b_ih_dir, b_hh_dir], dim=0).contiguous()
        return W_dir, R_dir, B_dir

    def _forward_onnx_unidirectional(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ONNX 导出（单向）：通过 custom_gru::quant_gru symbolic 生成单 GRU 节点。
        """
        if hx is None:
            h0 = input.new_zeros((1, input.size(1), self.hidden_size))
        else:
            expected_shape = (1, input.size(1), self.hidden_size)
            if not torch.onnx.is_in_onnx_export() and tuple(hx.shape) != expected_shape:
                raise ValueError(f"hx 形状应为 {expected_shape}，实际 {tuple(hx.shape)}")
            h0 = hx

        w_f, r_f, b_f = self._pack_onnx_gru_direction_weights(
            self.weight_ih_l0.detach().to(device=input.device, dtype=input.dtype),
            self.weight_hh_l0.detach().to(device=input.device, dtype=input.dtype),
            self.bias_ih_l0.detach().to(device=input.device, dtype=input.dtype),
            self.bias_hh_l0.detach().to(device=input.device, dtype=input.dtype),
        )
        self._onnx_export_weight_ih = w_f.unsqueeze(0).contiguous()
        self._onnx_export_weight_hh = r_f.unsqueeze(0).contiguous()
        self._onnx_export_bias = b_f.unsqueeze(0).contiguous()

        quant_gru_ops = getattr(torch.ops, QUANT_GRU_ONNX_DOMAIN)
        output, h_n = quant_gru_ops.quant_gru(
            input,
            h0,
            self._onnx_export_weight_ih,
            self._onnx_export_weight_hh,
            self._onnx_export_bias,
            int(self.hidden_size),
            int(self.num_layers),
        )
        return output, h_n

    def _prepare_onnx_bidirectional_weights(self, input_tensor: torch.Tensor) -> None:
        """准备双向 custom bigru 导出所需的 ONNX 格式 W/R/B。"""
        w_f, r_f, b_f = self._pack_onnx_gru_direction_weights(
            self.weight_ih_l0.detach().to(device=input_tensor.device, dtype=input_tensor.dtype),
            self.weight_hh_l0.detach().to(device=input_tensor.device, dtype=input_tensor.dtype),
            self.bias_ih_l0.detach().to(device=input_tensor.device, dtype=input_tensor.dtype),
            self.bias_hh_l0.detach().to(device=input_tensor.device, dtype=input_tensor.dtype),
        )
        w_b, r_b, b_b = self._pack_onnx_gru_direction_weights(
            self.weight_ih_l0_reverse.detach().to(device=input_tensor.device, dtype=input_tensor.dtype),
            self.weight_hh_l0_reverse.detach().to(device=input_tensor.device, dtype=input_tensor.dtype),
            self.bias_ih_l0_reverse.detach().to(device=input_tensor.device, dtype=input_tensor.dtype),
            self.bias_hh_l0_reverse.detach().to(device=input_tensor.device, dtype=input_tensor.dtype),
        )

        self._onnx_export_weight_ih = torch.stack([w_f, w_b], dim=0).contiguous()
        self._onnx_export_weight_hh = torch.stack([r_f, r_b], dim=0).contiguous()
        self._onnx_export_bias = torch.stack([b_f, b_b], dim=0).contiguous()

    def _forward_onnx_bidirectional(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ONNX 导出（双向）：通过 custom_gru::quant_bigru symbolic 生成单 GRU 节点。"""
        if hx is None:
            h0 = input.new_zeros((2, input.size(1), self.hidden_size))
        else:
            expected_shape = (2, input.size(1), self.hidden_size)
            if not torch.onnx.is_in_onnx_export() and tuple(hx.shape) != expected_shape:
                raise ValueError(f"hx 形状应为 {expected_shape}，实际 {tuple(hx.shape)}")
            h0 = hx

        self._prepare_onnx_bidirectional_weights(input)
        quant_gru_ops = getattr(torch.ops, QUANT_GRU_ONNX_DOMAIN)
        output, h_n = quant_gru_ops.quant_bigru(
            input,
            h0,
            self._onnx_export_weight_ih,
            self._onnx_export_weight_hh,
            self._onnx_export_bias,
            int(self.hidden_size),
            int(self.num_layers),
        )
        return output, h_n

    def _forward_onnx_gru(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ONNX 导出路径：导出标准 ONNX GRU 节点。
        """
        if not torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "export_mode=True 仅用于 ONNX 导出上下文。"
                "请在 torch.onnx.export(..., dynamo=False) 中调用模型。"
            )
        if not self.bias:
            raise RuntimeError("ONNX 导出暂不支持 bias=False，请使用 bias=True")

        _ = self._resolve_onnx_module_name()
        if not _get_quant_gru_registered_onnx_opsets():
            raise RuntimeError(
                "使用 QuantGRU export_mode=True 导出 ONNX 前，请先调用 "
                "ensure_quant_gru_onnx_registered(opset=...)；该 opset 必须与 "
                "torch.onnx.export(opset_version=...) 一致。"
            )

        if self.bidirectional:
            return self._forward_onnx_bidirectional(input, hx)
        return self._forward_onnx_unidirectional(input, hx)

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
        W, R, bw, br = convert_weights_to_haste_format(
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
            W=W, R=R, bw=bw, br=br, x=input,
            h0=h0_forward if h0_forward is not None else torch.empty(0, device=device),
            calib_method=self.calibration_method,
            bitwidth_config=self._bitwidth_config,
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

            W_rev, R_rev, bw_rev, br_rev = convert_weights_to_haste_format(
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
                W=W_rev, R=R_rev, bw=bw_rev, br=br_rev, x=input_reversed,
                h0=h0_reverse if h0_reverse is not None else torch.empty(0, device=device),
                calib_method=self.calibration_method,
                bitwidth_config=self._bitwidth_config,
                quant_ranges=quant_ranges_rev,
                hist_collectors=hist_collectors_rev
            )

            output_reverse = h_rev[1:].flip(0).contiguous()
            h_n_reverse = h_rev[-1:].contiguous()
        else:
            output_reverse, h_n_reverse = None, None

        # 标记量化参数需要更新
        if (not self._quant_params_locked) or self.quant_params is None:
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
            - export_mode=True: 使用 ONNX 单节点 GRU 导出路径
        """
        # ===== 统一处理 batch_first 输入转换(唯一入口)=====
        if self.batch_first:
            input = input.transpose(0, 1).contiguous()

        # ===== 根据模式选择执行路径 =====
        if self.export_mode:
            # ONNX 导出模式：使用标准 ONNX GRU 节点路径
            output, h_n = self._forward_onnx_gru(input, hx)
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
            if self._quant_params_locked and not self.is_calibrated():
                raise RuntimeError(
                    "quant_params 已锁定但当前量化参数不完整。\n"
                    "请先完成完整校准/加载后再锁定，或先调用 set_quant_params_locked(False) 解锁。"
                )
            if self._quant_params_dirty:
                # 校准数据已更新或配置已修改，需要重新计算量化参数
                if self._quant_params_locked and self.quant_params is not None:
                    self._quant_params_dirty = False
                else:
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
        # LUT 现在存储在 quant_params 中，通过 setRescaleParam 复制到前向传播的 rescale 参数
        output_forward, h_n_forward = GRUFunction.apply(
            input, self.weight_ih_l0, self.weight_hh_l0,
            self.bias_ih_l0 if self.bias else None,
            self.bias_hh_l0 if self.bias else None,
            h0_forward, self.training, self.use_quantization, self.quant_params, self.quant_storage_dtype)

        # 反向方向(双向时)
        output_reverse, h_n_reverse = None, None
        if self.bidirectional:
            # LUT 存储在 quant_params_reverse 中
            output_reverse, h_n_reverse = GRUFunction.apply(
                input.flip(0), self.weight_ih_l0_reverse, self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse if self.bias else None,
                self.bias_hh_l0_reverse if self.bias else None,
                h0_reverse, self.training, self.use_quantization, self.quant_params_reverse, self.quant_storage_dtype)
            # 反转反向输出以对齐时间步
            output_reverse = output_reverse.flip(0)

        # 合并双向输出(统一接口)
        return self._combine_bidirectional_outputs(
            output_forward, h_n_forward, output_reverse, h_n_reverse
        )

    # -------------------- AIMET 黑盒接入契约 v1 --------------------
    #
    # 说明：以下方法供 aimet_torch.fixed_point.quantgru_adapter 黑盒调用。
    # adapter 通过检测 QuantGRU 类是否原生实现这些方法来决定走原生还是 stub。
    # 契约文档：rxmet/doc/QuantGRU_INT16接入计划.md 第 1 章（已冻结 v1）。

    # AIMET ExecutionMode 对应的 mode 字符串（须与 aimet_capabilities 一致）
    _AIMET_SUPPORTED_MODES = (
        "fp32",
        "fp32_qdq",
        "fp16_qdq",
        "int16_fixed_eval",
        "int16_fixed_qat_sim",
        "calibrating",
    )

    def aimet_capabilities(self) -> dict:
        """返回 AIMET 适配能力报告（契约 v1 §1.8）。

        minor 1.1：forward_quantized 为纯定点 int 进 int 出（上游直接传
        scale_x/zp_x 网格上的 int32 x_q），不再接受 fp32 输入。
        """
        return {
            "adapter_version": "1.1",
            "supported_modes": list(self._AIMET_SUPPORTED_MODES),
            "forward_io_dtype": "float32",
            "forward_io_device": "cuda",
            "requires_calibration_for": ["int16_fixed_eval", "int16_fixed_qat_sim"],
            "supports_forward_quantized": True,
            # forward_quantized 接受的输入 dtype（int 进 int 出纯定点链路）
            "forward_quantized_input_dtypes": ["int32"],
        }

    def aimet_configure(self, mode: str) -> None:
        """AIMET 单一入口：把 mode 字符串映射到已有 flag（契约 v1 §1.5）。

        仅复用 use_quantization / calibrating / export_mode 已有语义，
        不引入新的内部计算路径。
        """
        normalized = str(mode).lower()
        if normalized not in self._AIMET_SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported aimet_configure mode: {mode!r}. "
                f"Expected one of: {', '.join(self._AIMET_SUPPORTED_MODES)}."
            )
        if normalized == "calibrating":
            self.calibrating = True
            self.use_quantization = False
            self.export_mode = False
            return
        self.calibrating = False
        self.export_mode = False
        self.use_quantization = normalized in ("int16_fixed_eval", "int16_fixed_qat_sim")

    def get_io_quant_meta(self) -> dict:
        """返回输入/输出/隐藏状态的量化元数据（契约 v1 §1.3）。

        Returns:
            {
              "input":  {"scale", "zp", "bitwidth", "is_symmetric", "is_unsigned"},
              "output": {...},   # 与 hidden 一致（基于 h 网格）
              "hidden": {...},
            }

        Raises:
            RuntimeError: 未校准时抛出。

        Note:
            返回的 scale/zp 必须与 forward_quantized 的整数输出严格一致
            （bit-exact 依赖）。这里读取的是与定点核心相同的 quant_params。
        """
        if not self.is_calibrated() or self.quant_params is None:
            raise RuntimeError("QuantGRU not calibrated")

        qp = self.quant_params
        bw = self._bitwidth_config
        use_pot2 = bool(self.use_pot2_scale)

        def _entry(prefix: str) -> dict:
            raw_scale = float(getattr(qp, f"scale_{prefix}"))
            # 返回核实际使用的有效定点 scale（POT2/M16 编码后），保证与
            # forward_quantized 的整数输出 bit-exact 一致。
            scale = float(gru_ops.decode_effective_scale(raw_scale, use_pot2))
            zp = int(getattr(qp, f"zp_{prefix}"))
            bitwidth = int(getattr(bw, f"{prefix}")) if bw is not None else 16
            is_symmetric = bool(getattr(bw, f"{prefix}symmetric_")) if bw is not None else True
            is_unsigned = bool(getattr(bw, f"{prefix}unsigned_", False)) if bw is not None else False
            if is_unsigned and is_symmetric:
                is_symmetric = False
            return {
                "scale": scale,
                "zp": zp,
                "bitwidth": bitwidth,
                "is_symmetric": is_symmetric,
                "is_unsigned": is_unsigned,
            }

        return {
            "input": _entry("x_"),
            "output": _entry("h_"),
            "hidden": _entry("h_"),
        }

    def _parse_initial_state_quant(
            self,
            hx: Optional[torch.Tensor],
            batch_size: int,
            device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """解析已量化初始隐藏状态（int32，scale_h/zp_h 网格）。"""
        if hx is None:
            return None, None
        expected_shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
        if hx.shape != expected_shape:
            raise ValueError(f"hx 形状应为 {expected_shape}，实际 {hx.shape}")

        def _conv(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device, dtype=torch.int32).contiguous()

        h0_forward = _conv(hx[0])
        h0_reverse = _conv(hx[1]) if self.bidirectional else None
        return h0_forward, h0_reverse

    def _run_forward_quantized_dir(
            self,
            input: torch.Tensor,
            h0: Optional[torch.Tensor],
            quant_params,
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单方向纯定点推理（int 进 int 出），返回 int32 (output_q[T,B,H], h_n_q[1,B,H])。

        input 为已量化 int32 x_q（scale_x/zp_x 网格），核内仅量化权重，跳过输入量化。
        """
        seq_len, batch_size, input_size = input.shape

        # 权重转换为 haste 格式（与 GRUFunction.forward 完全一致，含 None bias 处理）
        W, R, bw, br = convert_weights_to_haste_format(
            self.weight_ih_l0 if quant_params is self.quant_params else self.weight_ih_l0_reverse,
            self.weight_hh_l0 if quant_params is self.quant_params else self.weight_hh_l0_reverse,
            (self.bias_ih_l0 if quant_params is self.quant_params else self.bias_ih_l0_reverse) if self.bias else None,
            (self.bias_hh_l0 if quant_params is self.quant_params else self.bias_hh_l0_reverse) if self.bias else None,
            self.hidden_size, device,
        )

        h0_tensor = h0 if h0 is not None else \
            torch.empty(0, device=device, dtype=torch.int32)
        return gru_ops.forward_quantized_int_io(
            time_steps=seq_len,
            batch_size=batch_size,
            input_size=input_size,
            hidden_size=self.hidden_size,
            W=W, R=R, bw=bw, br=br,
            x_q=input,
            h0_q=h0_tensor,
            quant_params=quant_params,
        )

    def forward_quantized(
            self,
            input: torch.Tensor,
            hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """纯定点推理边界（契约 v1 §1.4，int 进 int 出），仅供 AIMET INT16 mode 调用。

        input 为上游在 scale_x/zp_x 网格上的已量化 int32 x_q，核内不再量化输入，
        仅量化权重。hx 同样为 scale_h/zp_h 网格上的 int32 h0_q（或 None）。

        输出恒为 int32 量化隐藏状态（未反量化），shape 与标准 forward 一致，
        满足 ``(output_q - zp) * scale == 部署核 fp 输出``。

        Args:
            input: [T, B, I] 或 [B, T, I]（batch_first），int32 / CUDA。
            hx: 已量化初始隐藏状态 int32 / CUDA，可为 None。

        Raises:
            RuntimeError: 未校准时抛出。
            TypeError: input 非整数 dtype 时抛出。
        """
        if not self.is_calibrated() or self.quant_params is None:
            raise RuntimeError("QuantGRU not calibrated")
        if self.export_mode:
            raise RuntimeError("forward_quantized 不支持 export_mode")
        if input.is_floating_point():
            raise TypeError(
                "forward_quantized 仅接受已量化的整数输入（int 进 int 出）；"
                f"收到 dtype={input.dtype}。上游须传 scale_x/zp_x 网格上的 int32 x_q。"
            )

        self._ensure_cublas_initialized()

        # batch_first 输入转换（唯一入口，与 forward 对齐）
        if self.batch_first:
            input = input.transpose(0, 1).contiguous()

        seq_len, batch_size, input_size = input.shape
        device = input.device if input.is_cuda else torch.device('cuda')
        input = input.to(device=device, dtype=torch.int32).contiguous()

        h0_forward, h0_reverse = self._parse_initial_state_quant(hx, batch_size, device)

        # 前向方向
        output_q_forward, h_n_q_forward = self._run_forward_quantized_dir(
            input, h0_forward, self.quant_params, device)

        # 反向方向（双向时），整数域翻转/拼接
        output_q_reverse, h_n_q_reverse = None, None
        if self.bidirectional:
            output_q_reverse, h_n_q_reverse = self._run_forward_quantized_dir(
                input.flip(0), h0_reverse, self.quant_params_reverse, device)
            output_q_reverse = output_q_reverse.flip(0)

        output_q, h_n_q = self._combine_bidirectional_outputs(
            output_q_forward, h_n_q_forward, output_q_reverse, h_n_q_reverse)

        # batch_first 输出转换（唯一出口）
        if self.batch_first:
            output_q = output_q.transpose(0, 1).contiguous()

        return output_q, h_n_q

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

    def load_quant_params(
        self,
        import_path: str,
        verbose: bool = False
    ) -> None:
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

    def _fix_zero_point(self, data: dict) -> dict:
        """
        修正 zero_point：如果 scale 是列表，zero_point 也应该是相同长度的列表
        
        参考 transfer_params_to_encodings.py 的实现
        
        Args:
            data: 单个参数的编码字典
            
        Returns:
            修正后的编码字典
        """
        if isinstance(data, dict) and "scale" in data:
            scale = data["scale"]
            
            # AIMET encodings 要求 per-channel/per-gate 的 zero_point 与 scale 等长。
            # 对称量化权重通常没有独立 zp 属性，导出时需要显式补 0。
            if isinstance(scale, list):
                zero_point = data.get("zero_point", 0)
                if isinstance(zero_point, list):
                    if len(zero_point) == len(scale):
                        return data
                    if len(zero_point) == 1:
                        data["zero_point"] = zero_point * len(scale)
                    elif len(zero_point) == 0:
                        data["zero_point"] = [0] * len(scale)
                    else:
                        raise ValueError(
                            f"zero_point 长度不匹配: 期望 {len(scale)}，实际 {len(zero_point)}"
                        )
                else:
                    data["zero_point"] = [zero_point] * len(scale)
        
        return data

    def _convert_to_encoding_format(self, data: dict) -> dict:
        """
        将参数文件格式转换为 encodings 格式
        
        参考 transfer_params_to_encodings.py 的实现
        
        Args:
            data: 参数文件中的单个参数字典
            
        Returns:
            转换后的编码字典（添加 bitwidth, is_symmetric 等字段）
        """
        import copy

        # 双向：收集反向方向算子（权重/偏置拆分 + internal_ops_reverse）
        operators_reverse = {} if self.bidirectional else None
        if not isinstance(data, dict):
            return data
        
        encoding = copy.deepcopy(data)
        
        # 添加 bitwidth（根据 dtype 判断）
        if "dtype" in encoding:
            dtype = encoding["dtype"]
            if "INT8" in dtype or "UINT8" in dtype:
                encoding["bitwidth"] = 8
            elif "INT16" in dtype or "UINT16" in dtype:
                encoding["bitwidth"] = 16
            elif "INT32" in dtype or "UINT32" in dtype:
                encoding["bitwidth"] = 32
        
        # 转换 symmetric 为 is_symmetric（字符串格式），并删除原 symmetric 字段
        if "symmetric" in encoding:
            encoding["is_symmetric"] = "True" if encoding["symmetric"] else "False"
            del encoding["symmetric"]
        
        # 确保 is_symmetric 字段的值是字符串格式（如果已经存在）
        if "is_symmetric" in encoding and isinstance(encoding["is_symmetric"], bool):
            encoding["is_symmetric"] = "True" if encoding["is_symmetric"] else "False"

        # scale-only 导出：清理历史 runtime 编码字段
        for legacy_key in ("multiplier", "shift", "scale_encoding", "n", "exp2_inv"):
            if legacy_key in encoding:
                del encoding[legacy_key]
        
        return encoding

    def export_quant_params_to_aimet_format(
        self,
        encodings_dict: dict,
        module_name: str = None,
        verbose: bool = False
    ) -> dict:
        """
        将 QuantGRU 的量化参数导出为 AIMET encodings 格式并合并到 encodings_dict
        
        参考 transfer_params_to_encodings.py 的实现
        
        Args:
            encodings_dict: AIMET encodings 字典（会被修改）
            module_name: AIMET 分配的模块名称（如果为 None，使用 self._module_name）
            verbose: 是否打印详细信息
        
        Returns:
            更新后的 encodings_dict
        
        """
        import copy

        def _concat_operator_fields(dst: dict, src: dict, *, list_fields: list, verbose_prefix: str = "") -> dict:
            """
            将 src 的 list_fields 追加到 dst 对应字段末尾（用于双向：正向在前、反向在后）。
            """
            if dst is None:
                return copy.deepcopy(src)
            if src is None:
                return dst

            for field in list_fields:
                if field not in src:
                    continue
                if field not in dst:
                    dst[field] = copy.deepcopy(src[field])
                    continue

                a = dst[field]
                b = src[field]
                if isinstance(a, list) and isinstance(b, list):
                    a.extend(b)
                elif isinstance(a, list):
                    a.append(b)
                elif isinstance(b, list):
                    dst[field] = [a] + b
                else:
                    dst[field] = [a, b]

            if verbose and verbose_prefix:
                for k in ["dtype", "symmetric", "enc_type"]:
                    if k in dst and k in src and dst[k] != src[k]:
                        print(f"  ⚠️ 警告：{verbose_prefix} 字段 '{k}' 正反向不一致："
                              f"forward={dst[k]}, reverse={src[k]}。将保留 forward 值。")
            return dst
        
        # 检查是否已校准
        if not self.is_calibrated():
            if verbose:
                print(f"  ⚠️ 警告：模块 {module_name or 'unknown'} 未校准，跳过导出")
            return encodings_dict

        # 配置变更后 quant_params 可能过期（懒同步），导出前强制对齐
        _ensure_quant_params_fresh_for_export(self, verbose=verbose)
        
        # 确定模块名称
        if module_name is None:
            module_name = self._module_name
            if not module_name:
                if verbose:
                    print(f"  _module_name 未设置，使用默认名称 'gru'")
                module_name = "gru"
        
        # 获取量化参数
        if self.quant_params is None:
            if verbose:
                print(f"  ⚠️ 警告：模块 {module_name} 的 quant_params 为空，跳过导出")
            return encodings_dict
        
        # 使用 _build_operators_dict 从 quant_params 构建 operators 字典
        operators = _build_operators_dict(self._bitwidth_config, self.quant_params, verbose)
        if not operators:
            if verbose:
                print(f"  ⚠️ 警告：模块 {module_name} 的 operators 为空，跳过导出")
            return encodings_dict

        # 双向：构建反向 operators（用于权重/偏置拼接 + internal_ops_reverse）
        operators_reverse = None
        if self.bidirectional:
            if self.quant_params_reverse is None:
                if verbose:
                    print(f"  ⚠️ 警告：模块 {module_name} 为双向，但 quant_params_reverse 为空。将仅导出正向参数。")
            else:
                operators_reverse = _build_operators_dict(self._bitwidth_config, self.quant_params_reverse, verbose)
        
        # 确保 encodings_dict 结构存在
        # 设置 schema_version（如果不存在）
        if "schema_version" not in encodings_dict:
            encodings_dict["schema_version"] = 3
        
        if "activation_encodings" not in encodings_dict:
            encodings_dict["activation_encodings"] = {}
        if module_name not in encodings_dict["activation_encodings"]:
            encodings_dict["activation_encodings"][module_name] = {
                "is_GRU": True,
                "input": [],
                "output": [],
                "internal_ops": {
                    "add_final_hidden": {"output": []},
                    "mul_new_contribution": {"output": []},
                    "mul_old_contribution": {"output": []},
                    "mul_reset_hidden": {"output": []},
                    "new_gate_input": {"output": []},
                    "new_gate_output": {"output": []},
                    "reset_gate_input": {"output": []},
                    "reset_gate_output": {"output": []},
                    "sub_one_minus_update": {"output": []},
                    "update_gate_input": {"output": []},
                    "update_gate_output": {"output": []},
                    "weight_ih_linear": {"output": []},
                    "weight_hh_linear": {"output": []},
                },
            }
            # 双向：新增 internal_ops_reverse（结构与 internal_ops 相同）
            if self.bidirectional:
                encodings_dict["activation_encodings"][module_name]["internal_ops_reverse"] = {
                    "add_final_hidden": {"output": []},
                    "mul_new_contribution": {"output": []},
                    "mul_old_contribution": {"output": []},
                    "mul_reset_hidden": {"output": []},
                    "new_gate_input": {"output": []},
                    "new_gate_output": {"output": []},
                    "reset_gate_input": {"output": []},
                    "reset_gate_output": {"output": []},
                    "sub_one_minus_update": {"output": []},
                    "update_gate_input": {"output": []},
                    "update_gate_output": {"output": []},
                    "weight_ih_linear": {"output": []},
                    "weight_hh_linear": {"output": []},
                }
        
        if "param_encodings" not in encodings_dict:
            encodings_dict["param_encodings"] = {}
        
        if "tensor_encodings" not in encodings_dict:
            encodings_dict["tensor_encodings"] = {}
        
        gru_encodings = encodings_dict["activation_encodings"][module_name]

        # 双向：如果 module 已存在但缺少 internal_ops_reverse，则补齐结构
        if self.bidirectional and "internal_ops_reverse" not in gru_encodings:
            gru_encodings["internal_ops_reverse"] = {
                "add_final_hidden": {"output": []},
                "mul_new_contribution": {"output": []},
                "mul_old_contribution": {"output": []},
                "mul_reset_hidden": {"output": []},
                "new_gate_input": {"output": []},
                "new_gate_output": {"output": []},
                "reset_gate_input": {"output": []},
                "reset_gate_output": {"output": []},
                "sub_one_minus_update": {"output": []},
                "update_gate_input": {"output": []},
                "update_gate_output": {"output": []},
                "weight_ih_linear": {"output": []},
                "weight_hh_linear": {"output": []},
            }
        
        # 1. 填充 activation_encodings 的 input 和 output
        if verbose:
            print(f"\n处理 activation_encodings[{module_name}]:")
        
        # input
        if "input" in operators:
            data = self._fix_zero_point(operators["input"])
            data = self._convert_to_encoding_format(data)
            if isinstance(data, dict):
                gru_encodings["input"] = [data]
            else:
                gru_encodings["input"] = data
            if verbose:
                print(f"  ✅ input <- input")
        else:
            if verbose:
                print(f"  ⚠️ 警告：未找到参数 'input'")
        
        # output
        if "output" in operators:
            data = self._fix_zero_point(operators["output"])
            data = self._convert_to_encoding_format(data)
            if isinstance(data, dict):
                gru_encodings["output"] = [data]
            else:
                gru_encodings["output"] = data
            if verbose:
                print(f"  ✅ output <- output")
        else:
            if verbose:
                print(f"  ⚠️ 警告：未找到参数 'output'")
        
        # 2. 填充 param_encodings
        if verbose:
            print(f"\n处理 param_encodings:")
        
        # 直接使用 ONNX 标准命名规则（不需要推断）：
        # - {module_name}.weight_ih.weight (对应 weight_ih)
        # - {module_name}.weight_hh.weight (对应 weight_hh)
        # - {module_name}.bias (合并的 bias，对应 bias_ih+bias_hh)
        
        # weight_ih - ONNX 标准命名：{module_name}.weight_ih.weight
        param_key = f"{module_name}.weight_ih.weight"
        if "weight_ih" in operators:
            data = self._fix_zero_point(operators["weight_ih"])
            if operators_reverse is not None and "weight_ih" in operators_reverse:
                data = _concat_operator_fields(
                    data,
                    self._fix_zero_point(operators_reverse["weight_ih"]),
                    list_fields=["scale", "zero_point", "real_min", "real_max"],
                    verbose_prefix=f"{param_key}/weight_ih"
                )
            data = self._convert_to_encoding_format(data)
            encodings_dict["param_encodings"][param_key] = data
            if verbose:
                print(f"  ✅ {param_key} <- weight_ih")
        else:
            if verbose:
                print(f"  ⚠️ 警告：未找到参数 'weight_ih'")
        
        # weight_hh - ONNX 标准命名：{module_name}.weight_hh.weight
        param_key = f"{module_name}.weight_hh.weight"
        if "weight_hh" in operators:
            data = self._fix_zero_point(operators["weight_hh"])
            if operators_reverse is not None and "weight_hh" in operators_reverse:
                data = _concat_operator_fields(
                    data,
                    self._fix_zero_point(operators_reverse["weight_hh"]),
                    list_fields=["scale", "zero_point", "real_min", "real_max"],
                    verbose_prefix=f"{param_key}/weight_hh"
                )
            data = self._convert_to_encoding_format(data)
            encodings_dict["param_encodings"][param_key] = data
            if verbose:
                print(f"  ✅ {param_key} <- weight_hh")
        else:
            if verbose:
                print(f"  ⚠️ 警告：未找到参数 'weight_hh'")
        
        # bias - 特殊处理，需要拼接 bias_ih 和 bias_hh 两个来源的各个字段
        # ONNX 标准命名：{module_name}.bias（合并的 bias）
        # 必须同时有 bias_ih 和 bias_hh，否则报错
        param_key = f"{module_name}.bias"
        
        # 检查 bias_ih 和 bias_hh 是否都存在
        if "bias_ih" not in operators:
            error_msg = f"❌ 错误：未找到参数 'bias_ih'（模块 {module_name} 的 bias 需要同时有 bias_ih 和 bias_hh）"
            if verbose:
                print(f"  {error_msg}")
            raise ValueError(error_msg)
        
        if "bias_hh" not in operators:
            error_msg = f"❌ 错误：未找到参数 'bias_hh'（模块 {module_name} 的 bias 需要同时有 bias_ih 和 bias_hh）"
            if verbose:
                print(f"  {error_msg}")
            raise ValueError(error_msg)
        
        # 拼接多个来源的数据（bias_ih 在前，bias_hh 在后；双向时在末尾追加反向）
        # 处理 bias_ih（第一个源）
        bw_data = self._fix_zero_point(operators["bias_ih"])
        merged_dict = copy.deepcopy(bw_data)
        if verbose:
            print(f"  ✅ {param_key} <- bw (初始化)")
        
        # 处理 bias_hh（第二个源，需要拼接）
        br_data = self._fix_zero_point(operators["bias_hh"])
        # 需要拼接的字段：scale, zero_point, real_min, real_max
        list_fields = [
            "scale",
            "zero_point",
            "real_min",
            "real_max",
        ]
        for field in list_fields:
            if field in br_data and field in merged_dict:
                if isinstance(merged_dict[field], list) and isinstance(br_data[field], list):
                    merged_dict[field].extend(br_data[field])
                    if verbose:
                        print(f"      拼接字段 {field}: {len(merged_dict[field])} 元素")
                elif isinstance(merged_dict[field], list):
                    merged_dict[field].append(br_data[field])
                elif isinstance(br_data[field], list):
                    merged_dict[field] = [merged_dict[field]] + br_data[field]
                else:
                    # 都是标量，转为列表
                    merged_dict[field] = [
                        merged_dict[field],
                        br_data[field],
                    ]
        if verbose:
            print(f"  ✅ {param_key} <- br (拼接)")

        # 双向：把反向 bias_ih/bias_hh 继续拼接到末尾
        if operators_reverse is not None:
            if "bias_ih" in operators_reverse and "bias_hh" in operators_reverse:
                merged_dict = _concat_operator_fields(
                    merged_dict,
                    self._fix_zero_point(operators_reverse["bias_ih"]),
                    list_fields=list_fields,
                    verbose_prefix=f"{param_key}/bias_ih_reverse"
                )
                merged_dict = _concat_operator_fields(
                    merged_dict,
                    self._fix_zero_point(operators_reverse["bias_hh"]),
                    list_fields=list_fields,
                    verbose_prefix=f"{param_key}/bias_hh_reverse"
                )
                if verbose:
                    print(f"  ✅ {param_key} <- (追加反向 bw/br)")
            else:
                if verbose:
                    print(f"  ⚠️ 警告：未找到反向 bias_ih/bias_hh，跳过反向 bias 拼接")
        
        # 转换格式并保存
        merged_dict = self._convert_to_encoding_format(merged_dict)
        encodings_dict["param_encodings"][param_key] = merged_dict
        
        # 3. 填充 tensor_encodings
        if verbose:
            print(f"\n处理 tensor_encodings:")
        
        # X <- input
        if "input" in operators:
            data = self._fix_zero_point(operators["input"])
            data = self._convert_to_encoding_format(data)
            encodings_dict["tensor_encodings"]["X"] = data
            if verbose:
                print(f"  ✅ X <- input")
        else:
            if verbose:
                print(f"  ⚠️ 警告：未找到参数 'input' (用于 tensor_encodings['X'])")
        
        # Y <- output
        if "output" in operators:
            data = self._fix_zero_point(operators["output"])
            data = self._convert_to_encoding_format(data)
            encodings_dict["tensor_encodings"]["Y"] = data
            if verbose:
                print(f"  ✅ Y <- output")
        else:
            if verbose:
                print(f"  ⚠️ 警告：未找到参数 'output' (用于 tensor_encodings['Y'])")
        
        # /gru/GRU_output_1 <- output
        if "output" in operators:
            data = self._fix_zero_point(operators["output"])
            data = self._convert_to_encoding_format(data)
            encodings_dict["tensor_encodings"]["/gru/GRU_output_1"] = data
            if verbose:
                print(f"  ✅ /gru/GRU_output_1 <- output")
        else:
            if verbose:
                print(f"  ⚠️ 警告：未找到参数 'output' (用于 tensor_encodings['/gru/GRU_output_1'])")
        
        # 4. 填充 internal_ops
        if verbose:
            print(f"\n处理 internal_ops:")
        
        internal_ops = gru_encodings["internal_ops"]
        
        # 定义中间算子列表
        internal_op_names = [
            "weight_ih_linear",
            "weight_hh_linear",
            "update_gate_input",
            "update_gate_output",
            "reset_gate_input",
            "reset_gate_output",
            "new_gate_input",
            "new_gate_output",
            "mul_reset_hidden",
            "mul_old_contribution",
            "mul_new_contribution",
        ]
        
        for op_name in internal_op_names:
            if op_name in operators:
                data = self._fix_zero_point(operators[op_name])
                data = self._convert_to_encoding_format(data)
                if "output" in internal_ops[op_name]:
                    internal_ops[op_name]["output"] = [data]
                else:
                    internal_ops[op_name] = data
                if verbose:
                    print(f"  ✅ {op_name} <- {op_name} (output)")
            else:
                if verbose:
                    print(f"  ⚠️ 警告：未找到参数 '{op_name}'")
        
        # 特殊处理：add_final_hidden 使用 output 的参数
        if "add_final_hidden" in internal_ops:
            if "output" in operators:
                data = self._fix_zero_point(operators["output"])
                data = self._convert_to_encoding_format(data)
                if "output" in internal_ops["add_final_hidden"]:
                    internal_ops["add_final_hidden"]["output"] = [data]
                else:
                    internal_ops["add_final_hidden"] = data
                if verbose:
                    print(f"  ✅ add_final_hidden <- output (使用 output 的量化参数)")
            else:
                if verbose:
                    print(f"  ⚠️ 警告：未找到参数 'output' (用于 add_final_hidden)")
        
        # 特殊处理：sub_one_minus_update 复用 update_gate_output 的参数
        if "sub_one_minus_update" in internal_ops:
            if "update_gate_output" in operators:
                data = self._fix_zero_point(operators["update_gate_output"])
                data = self._convert_to_encoding_format(data)
                if "output" in internal_ops["sub_one_minus_update"]:
                    internal_ops["sub_one_minus_update"]["output"] = [data]
                else:
                    internal_ops["sub_one_minus_update"] = data
                if verbose:
                    print(f"  ✅ sub_one_minus_update <- update_gate_output (复用 update_gate_output)")
            else:
                if verbose:
                    print(f"  ⚠️ 警告：未找到参数 'update_gate_output' (用于 sub_one_minus_update)")

        # 双向：填充 internal_ops_reverse（与 internal_ops 相同逻辑，但读取 operators_reverse）
        if self.bidirectional and operators_reverse is not None and "internal_ops_reverse" in gru_encodings:
            if verbose:
                print(f"\n处理 internal_ops_reverse:")
            internal_ops_rev = gru_encodings["internal_ops_reverse"]

            for op_name in internal_op_names:
                if op_name in operators_reverse:
                    data = self._fix_zero_point(operators_reverse[op_name])
                    data = self._convert_to_encoding_format(data)
                    if "output" in internal_ops_rev[op_name]:
                        internal_ops_rev[op_name]["output"] = [data]
                    else:
                        internal_ops_rev[op_name] = data
                    if verbose:
                        print(f"  ✅ {op_name} <- {op_name} (reverse/output)")
                else:
                    if verbose:
                        print(f"  ⚠️ 警告：未找到反向参数 '{op_name}'")

            # add_final_hidden: 使用 reverse 的 output
            if "add_final_hidden" in internal_ops_rev:
                if "output" in operators_reverse:
                    data = self._fix_zero_point(operators_reverse["output"])
                    data = self._convert_to_encoding_format(data)
                    if "output" in internal_ops_rev["add_final_hidden"]:
                        internal_ops_rev["add_final_hidden"]["output"] = [data]
                    else:
                        internal_ops_rev["add_final_hidden"] = data
                    if verbose:
                        print(f"  ✅ add_final_hidden <- output (reverse)")
                else:
                    if verbose:
                        print(f"  ⚠️ 警告：未找到反向参数 'output' (用于 add_final_hidden/reverse)")

            # sub_one_minus_update: 复用 reverse 的 update_gate_output
            if "sub_one_minus_update" in internal_ops_rev:
                if "update_gate_output" in operators_reverse:
                    data = self._fix_zero_point(operators_reverse["update_gate_output"])
                    data = self._convert_to_encoding_format(data)
                    if "output" in internal_ops_rev["sub_one_minus_update"]:
                        internal_ops_rev["sub_one_minus_update"]["output"] = [data]
                    else:
                        internal_ops_rev["sub_one_minus_update"] = data
                    if verbose:
                        print(f"  ✅ sub_one_minus_update <- update_gate_output (reverse)")
                else:
                    if verbose:
                        print(f"  ⚠️ 警告：未找到反向参数 'update_gate_output' (用于 sub_one_minus_update/reverse)")
        
        # 删除 tensor_encodings（避免旧版本 load_quantizer_encodings 报错）
        # tensor_encodings 主要用于 ONNX 导出，不影响量化器加载
        # 旧版本的 load_quantizer_encodings 会遍历顶层键，当遍历到 tensor_encodings 时会报 KeyError
        if "tensor_encodings" in encodings_dict:
            del encodings_dict["tensor_encodings"]
            if verbose:
                print(f"\n⚠️  已移除 tensor_encodings（避免与旧版本 load_quantizer_encodings 不兼容）")
                print(f"   tensor_encodings 主要用于 ONNX 导出，不影响量化器加载")
        
        return encodings_dict

    def load_quant_params_from_aimet_format(
        self,
        encodings_dict: dict,
        module_name: str = None,
        verbose: bool = False
    ) -> bool:
        """
        从 AIMET encodings 格式加载量化参数
        
        参考 transfer_params_to_encodings.py 的反向映射
        
        Args:
            encodings_dict: AIMET encodings 字典
            module_name: AIMET 分配的模块名称（如果为 None，使用 self._module_name）
            verbose: 是否打印详细信息
        
        Returns:
            是否成功加载
        
        """
        import copy

        # 双向：收集反向方向算子（由导出时的“追加拼接”拆分得到）
        operators_reverse = {} if self.bidirectional else None

        def _split_list_in_half(x):
            if not isinstance(x, list):
                return x, x
            mid = len(x) // 2
            return x[:mid], x[mid:]

        def _split_operator_for_bidirectional(op_data: dict) -> tuple:
            """
            将导出时拼接的权重算子数据拆回 (forward, reverse) 两份。
            仅拆分数组字段：scale/real_min/real_max（以及可能为列表的 zero_point）。
            """
            if op_data is None or not isinstance(op_data, dict):
                return op_data, None

            out_f = copy.deepcopy(op_data)
            out_r = copy.deepcopy(op_data)
            for field in ["scale", "multiplier", "shift", "real_min", "real_max"]:
                if field in op_data and isinstance(op_data[field], list):
                    a, b = _split_list_in_half(op_data[field])
                    out_f[field] = a
                    out_r[field] = b

            if "zero_point" in op_data and isinstance(op_data["zero_point"], list):
                a, b = _split_list_in_half(op_data["zero_point"])
                out_f["zero_point"] = a
                out_r["zero_point"] = b

            return out_f, out_r
        
        # 确定模块名称
        if module_name is None:
            module_name = self._module_name
            if not module_name:
                if verbose:
                    print(f"  _module_name 未设置，使用默认名称 'gru'")
                module_name = "gru"
        
        # 保存模块名称（用于调试信息和后续使用）
        self._module_name = module_name
        
        # 检查 encodings_dict 结构
        if "activation_encodings" not in encodings_dict:
            if verbose:
                print(f"  ❌ 错误：encodings_dict 中缺少 'activation_encodings'")
            return False
        
        if module_name not in encodings_dict["activation_encodings"]:
            if verbose:
                print(f"  ❌ 错误：未找到模块 '{module_name}' 的 activation_encodings")
            return False
        
        gru_encodings = encodings_dict["activation_encodings"][module_name]
        
        # 检查 is_GRU 标记
        if gru_encodings.get("is_GRU") != True:
            if verbose:
                print(f"  ⚠️ 警告：模块 '{module_name}' 的 is_GRU 标记不是 True")
        
        # 构建 operators 字典
        operators = {}
        
        # 1. 从 activation_encodings 读取 input 和 output
        if verbose:
            print(f"\n从 activation_encodings[{module_name}] 读取:")
        
        # input
        if "input" in gru_encodings:
            input_data = gru_encodings["input"]
            if isinstance(input_data, list) and len(input_data) > 0:
                operators["input"] = input_data[0]
            elif isinstance(input_data, dict):
                operators["input"] = input_data
            if verbose:
                print(f"  ✅ input <- input")
            # 双向 GRU 的反向分支与正向分支使用同一个输入张量，
            # 若不显式同步，operators_reverse["input"] 会落回默认量化参数。
            if self.bidirectional and operators_reverse is not None:
                operators_reverse["input"] = copy.deepcopy(operators["input"])
                if verbose:
                    print(f"  ✅ input <- input (reverse 同步)")
        else:
            if verbose:
                print(f"  ❌ 错误：未找到 'input'")
            return False
        
        # output
        if "output" in gru_encodings:
            output_data = gru_encodings["output"]
            if isinstance(output_data, list) and len(output_data) > 0:
                operators["output"] = output_data[0]
            elif isinstance(output_data, dict):
                operators["output"] = output_data
            if verbose:
                print(f"  ✅ output <- output")
        else:
            if verbose:
                print(f"  ❌ 错误：未找到 'output'")
            return False
        
        # 2. 从 param_encodings 读取权重和偏置
        if verbose:
            print(f"\n从 param_encodings 读取:")
        
        if "param_encodings" not in encodings_dict:
            if verbose:
                print(f"  ⚠️ 警告：encodings_dict 中缺少 'param_encodings'")
        else:
            param_encodings = encodings_dict["param_encodings"]
            
            # 直接使用 ONNX 标准命名规则（不需要推断）：
            # - {module_name}.weight_ih.weight (对应 weight_ih)
            # - {module_name}.weight_hh.weight (对应 weight_hh)
            # - {module_name}.bias (合并的 bias，对应 bias_ih+bias_hh)
            
            # weight_ih - ONNX 标准命名：{module_name}.weight_ih.weight
            param_key = f"{module_name}.weight_ih.weight"
            if param_key in param_encodings:
                if self.bidirectional:
                    fwd, rev = _split_operator_for_bidirectional(param_encodings[param_key])
                    operators["weight_ih"] = fwd
                    if rev is not None:
                        operators_reverse["weight_ih"] = rev
                else:
                    operators["weight_ih"] = param_encodings[param_key]
                if verbose:
                    print(f"  ✅ weight_ih <- {param_key}")
            else:
                if verbose:
                    print(f"  ❌ 错误：未找到 '{param_key}'")
                return False
            
            # weight_hh - ONNX 标准命名：{module_name}.weight_hh.weight
            param_key = f"{module_name}.weight_hh.weight"
            if param_key in param_encodings:
                if self.bidirectional:
                    fwd, rev = _split_operator_for_bidirectional(param_encodings[param_key])
                    operators["weight_hh"] = fwd
                    if rev is not None:
                        operators_reverse["weight_hh"] = rev
                else:
                    operators["weight_hh"] = param_encodings[param_key]
                if verbose:
                    print(f"  ✅ weight_hh <- {param_key}")
            else:
                if verbose:
                    print(f"  ❌ 错误：未找到 '{param_key}'")
                return False
            
            # bias - ONNX 标准命名：{module_name}.bias（合并的 bias，需要拆分为 bias_ih 和 bias_hh）
            param_key = f"{module_name}.bias"
            if param_key in param_encodings:
                merged_data = param_encodings[param_key]
                
                # 需要拆分的字段：scale, zero_point, real_min, real_max
                list_fields = ["scale", "zero_point", "real_min", "real_max"]
                
                # 计算每个字段应该拆分的长度（假设 bias_ih 和 bias_hh 长度相同）
                split_length = None
                for field in list_fields:
                    if field in merged_data and isinstance(merged_data[field], list):
                        if split_length is None:
                            split_length = len(merged_data[field]) // 2
                        break
                
                if split_length is None:
                    # 如果无法确定长度，尝试从其他字段推断
                    if "scale" in merged_data and isinstance(merged_data["scale"], list):
                        split_length = len(merged_data["scale"]) // 2
                    else:
                        if verbose:
                            print(f"  ⚠️ 警告：无法确定 split_length，假设 bias_ih 和 bias_hh 长度相同")
                        split_length = None
                
                if not self.bidirectional:
                    # 拆分 bias_ih（前半部分）
                    bias_ih_data = copy.deepcopy(merged_data)
                    # 拆分 bias_hh（后半部分）
                    bias_hh_data = copy.deepcopy(merged_data)
                    
                    for field in list_fields:
                        if field in merged_data:
                            if isinstance(merged_data[field], list):
                                if split_length is not None:
                                    bias_ih_data[field] = merged_data[field][:split_length]
                                    bias_hh_data[field] = merged_data[field][split_length:]
                                else:
                                    # 无法确定长度，尝试平均分割
                                    mid = len(merged_data[field]) // 2
                                    bias_ih_data[field] = merged_data[field][:mid]
                                    bias_hh_data[field] = merged_data[field][mid:]
                            else:
                                # 标量值，两个都使用相同的值
                                bias_ih_data[field] = merged_data[field]
                                bias_hh_data[field] = merged_data[field]
                    
                    operators["bias_ih"] = bias_ih_data
                    operators["bias_hh"] = bias_hh_data
                else:
                    # 双向：merged = [bw_fwd, br_fwd, bw_rev, br_rev]（按导出拼接规则）
                    # 先按方向拆半，再在方向内拆 bw/br
                    dir_len = None
                    for field in list_fields:
                        if field in merged_data and isinstance(merged_data[field], list):
                            dir_len = len(merged_data[field]) // 2
                            break
                    if dir_len is None or dir_len % 2 != 0:
                        if verbose:
                            print(f"  ❌ 错误：无法从 bias 推断双向拆分长度（dir_len={dir_len}）")
                        return False
                    ih_len = dir_len // 2

                    bias_ih_fwd = copy.deepcopy(merged_data)
                    bias_hh_fwd = copy.deepcopy(merged_data)
                    bias_ih_rev = copy.deepcopy(merged_data)
                    bias_hh_rev = copy.deepcopy(merged_data)

                    for field in list_fields:
                        if field in merged_data and isinstance(merged_data[field], list):
                            v = merged_data[field]
                            bias_ih_fwd[field] = v[:ih_len]
                            bias_hh_fwd[field] = v[ih_len:dir_len]
                            bias_ih_rev[field] = v[dir_len:dir_len + ih_len]
                            bias_hh_rev[field] = v[dir_len + ih_len:dir_len * 2]
                        else:
                            bias_ih_fwd[field] = merged_data.get(field)
                            bias_hh_fwd[field] = merged_data.get(field)
                            bias_ih_rev[field] = merged_data.get(field)
                            bias_hh_rev[field] = merged_data.get(field)

                    operators["bias_ih"] = bias_ih_fwd
                    operators["bias_hh"] = bias_hh_fwd

                    operators_reverse["bias_ih"] = bias_ih_rev
                    operators_reverse["bias_hh"] = bias_hh_rev
                if verbose:
                    print(f"  ✅ bias_ih, bias_hh <- {param_key} (从合并的 bias 拆分)")
            else:
                if verbose:
                    print(f"  ❌ 错误：未找到 '{param_key}'")
                return False
        
        # 3. 从 internal_ops 读取中间算子
        if verbose:
            print(f"\n从 internal_ops 读取:")
        
        if "internal_ops" in gru_encodings:
            internal_ops = gru_encodings["internal_ops"]
            
            # 定义中间算子列表
            internal_op_names = [
                "weight_ih_linear",
                "weight_hh_linear",
                "update_gate_input",
                "update_gate_output",
                "reset_gate_input",
                "reset_gate_output",
                "new_gate_input",
                "new_gate_output",
                "mul_reset_hidden",
                "mul_old_contribution",
                "mul_new_contribution",
            ]
            
            for op_name in internal_op_names:
                if op_name in internal_ops:
                    op_data = internal_ops[op_name]
                    if isinstance(op_data, dict) and "output" in op_data:
                        output_data = op_data["output"]
                        if isinstance(output_data, list) and len(output_data) > 0:
                            operators[op_name] = output_data[0]
                        elif isinstance(output_data, dict):
                            operators[op_name] = output_data
                    elif isinstance(op_data, dict):
                        operators[op_name] = op_data
                    if verbose:
                        print(f"  ✅ {op_name} <- internal_ops[{op_name}]")
                else:
                    if verbose:
                        print(f"  ⚠️ 警告：未找到中间算子 '{op_name}'")
            
            # 特殊处理：add_final_hidden 和 sub_one_minus_update 可以忽略
            # （因为它们分别使用 h 和 update_gate_output 的参数）
            if "add_final_hidden" in internal_ops:
                if verbose:
                    print(f"  跳过 'add_final_hidden'（使用 h 的参数）")
            if "sub_one_minus_update" in internal_ops:
                if verbose:
                    print(f"  跳过 'sub_one_minus_update'（使用 update_gate_output 的参数）")
        else:
            if verbose:
                print(f"  ⚠️ 警告：未找到 'internal_ops'")

        # 双向：从 internal_ops_reverse 读取反向中间算子
        if self.bidirectional and operators_reverse is not None and "internal_ops_reverse" in gru_encodings:
            internal_ops_reverse = gru_encodings["internal_ops_reverse"]
            if verbose:
                print(f"\n从 internal_ops_reverse 读取:")
            for op_name in internal_op_names:
                if op_name in internal_ops_reverse:
                    op_data = internal_ops_reverse[op_name]
                    if isinstance(op_data, dict) and "output" in op_data:
                        output_data = op_data["output"]
                        if isinstance(output_data, list) and len(output_data) > 0:
                            operators_reverse[op_name] = output_data[0]
                        elif isinstance(output_data, dict):
                            operators_reverse[op_name] = output_data
                    elif isinstance(op_data, dict):
                        operators_reverse[op_name] = op_data
                    if verbose:
                        print(f"  ✅ {op_name} <- internal_ops_reverse[{op_name}]")
                else:
                    if verbose:
                        print(f"  ⚠️ 警告：未找到反向中间算子 '{op_name}'")
            # add_final_hidden 在导出时使用的是 reverse 的 output 参数，加载时需回填 operators_reverse["output"]
            if "add_final_hidden" in internal_ops_reverse:
                op_data = internal_ops_reverse["add_final_hidden"]
                if isinstance(op_data, dict) and "output" in op_data:
                    output_data = op_data["output"]
                    if isinstance(output_data, list) and len(output_data) > 0:
                        operators_reverse["output"] = output_data[0]
                    elif isinstance(output_data, dict):
                        operators_reverse["output"] = output_data
                elif isinstance(op_data, dict):
                    operators_reverse["output"] = op_data
                if verbose:
                    print(f"  ✅ output <- internal_ops_reverse[add_final_hidden] (reverse 输出)")
        elif self.bidirectional and verbose:
            print(f"  ⚠️ 警告：未找到 'internal_ops_reverse'（反向中间算子将无法加载）")
        
        # 4. 转换格式（从 AIMET 格式转换为 QuantGRU 格式）
        # 将 is_symmetric 字符串转换回 symmetric 布尔值
        for key, value in operators.items():
            if isinstance(value, dict):
                if "is_symmetric" in value:
                    is_symmetric_str = value["is_symmetric"]
                    if isinstance(is_symmetric_str, str):
                        value["symmetric"] = is_symmetric_str.lower() == "true"
                        del value["is_symmetric"]
                    elif isinstance(is_symmetric_str, bool):
                        value["symmetric"] = is_symmetric_str
                        del value["is_symmetric"]
        
        # 5. 直接解析 operators 并设置到 quant_params（双向则额外解析 operators_reverse）
        try:
            # 初始化 bitwidth_config 和 quant_params
            # gru_ops 已在模块顶部导入
            self._bitwidth_config = gru_ops.OperatorQuantConfig()
            self.quant_params = gru_ops.GRUQuantParams()
            self.quant_params.hidden_ = self.hidden_size
            
            # 解析 operators 字典
            _parse_operators_dict(operators, self._bitwidth_config, self.quant_params, verbose)
            
            # 设置位宽配置到量化参数中
            self.quant_params.bitwidth_config_ = self._bitwidth_config

            if self.bidirectional:
                if operators_reverse is not None and len(operators_reverse) > 0:
                    self.quant_params_reverse = gru_ops.GRUQuantParams()
                    self.quant_params_reverse.hidden_ = self.hidden_size
                    _parse_operators_dict(operators_reverse, self._bitwidth_config, self.quant_params_reverse, verbose)
                    self.quant_params_reverse.bitwidth_config_ = self._bitwidth_config
                else:
                    self.quant_params_reverse = None
                    if verbose:
                        print("  ⚠️ 警告：双向模型未找到反向 operators，quant_params_reverse 将保持为空")
            
            # 清除脏标志
            self._quant_params_dirty = False
            
            if verbose:
                module_name_str = f" [{self._module_name}]" if self._module_name else ""
                print(f"  ✅ 成功加载量化参数{module_name_str}")
            return True
        except Exception as e:
            if verbose:
                module_name_str = f" [{self._module_name}]" if self._module_name else ""
                print(f"  ❌ 错误：加载量化参数失败{module_name_str}: {e}")
            return False

    def adjust_quant_config(
        self,
        operator: str,
        bitwidth: int = None,
        is_symmetric: bool = None,
        scale: float = None,
        zero_point: int = None,
        verbose: bool = False
    ) -> None:
        """
        手动调整指定算子的量化配置
        
        Args:
            operator: 算子名称 ("x", "h", "W", "z_out" 等)
            bitwidth: 新的位宽 (1-32)
            is_symmetric: 是否对称量化
            scale: 量化 scale（正数）；None 表示不修改 scale
            zero_point: 零点
            verbose: 是否打印详情
            
        Example:
            >>> gru.adjust_quant_config("z_out", bitwidth=16, verbose=True)
        """
        # 调用模块级实现
        _adjust_quant_config_impl(self, operator, bitwidth, is_symmetric, scale, zero_point, verbose)

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
    打印 QuantGRU 的量化参数和配置（合并版本）
    
    显示内容：bitwidth、is_symmetric、shift、scale、zp
    格式与 C++ quantize_ops_helper.h 对齐
    
    Args:
        gru: QuantGRU 实例（如果未校准，只显示配置信息，不显示参数）
    """
    # 获取 bitwidth_config（总是存在）
    if hasattr(gru, '_bitwidth_config') and gru._bitwidth_config is not None:
        bitwidth_config = gru._bitwidth_config
    elif gru.is_calibrated() and gru.quant_params is not None:
        bitwidth_config = gru.quant_params.bitwidth_config_
    else:
        bitwidth_config = None
    
    if bitwidth_config is None:
        raise RuntimeError("无法获取 bitwidth_config，请先设置量化配置")
    
    # 获取 quant_params（可能为 None，如果未校准）
    params = gru.quant_params if gru.is_calibrated() else None
    
    print("=" * 60)
    print("GRUQuantParams (量化参数和配置)")
    print("=" * 60)
    
    if params is not None:
        print(f"  hidden_ = {params.hidden_}")
    else:
        print(f"  hidden_ = {gru.hidden_size} (未校准)")
    
    # 辅助函数：格式化单个算子的信息
    def format_op_info(op_name: str, scale_val, zp_val, bitwidth: int, is_symmetric: bool, is_unsigned: bool = False):
        """格式化单个算子的打印信息"""
        sym_str = "对称" if is_symmetric else "非对称"
        
        # 根据 is_unsigned 和 bitwidth 生成数据类型字符串
        dtype_str = f"UINT{bitwidth}" if is_unsigned else f"INT{bitwidth}"
        
        shift_str = "N/A"
        if scale_val is not None:
            if isinstance(scale_val, (list, tuple)):
                if len(scale_val) > 0:
                    if len(scale_val) >= 2:
                        scale_str = f"[{float(scale_val[0]):.6f}, {float(scale_val[1]):.6f}, ...] (len={len(scale_val)})"
                    else:
                        scale_str = f"[{float(scale_val[0]):.6f}] (len={len(scale_val)})"
                else:
                    scale_str = "[]"
            else:
                scale_str = f"{float(scale_val):.6f}"
        else:
            scale_str = "N/A"
        
        zp_str = f"{zp_val}" if zp_val is not None else "N/A"
        
        return f"  [{op_name:23s}] {dtype_str:6s}, {sym_str:4s}, shift={shift_str:10s}, scale={scale_str:30s}, zp={zp_str}"
    
    # 打印基础算子（input, output, weight_ih_linear, weight_hh_linear）
    for op_name in ['input', 'output', 'weight_ih_linear', 'weight_hh_linear']:
        if op_name not in _OPERATOR_SHORT_NAME_MAP:
            continue
        attrs = _OPERATOR_SHORT_NAME_MAP[op_name]
        bitwidth = getattr(bitwidth_config, attrs['bw_attr'])
        is_symmetric = getattr(bitwidth_config, attrs['sym_attr'])
        # 获取 unsigned 属性（如果存在）
        unsigned_attr = attrs.get('unsigned_attr')
        is_unsigned = getattr(bitwidth_config, unsigned_attr) if unsigned_attr and hasattr(bitwidth_config, unsigned_attr) else False
        
        if params is not None:
            shift_attr = attrs['scale_attr']
            zp_attr = attrs.get('zp_attr')
            shift_val = getattr(params, shift_attr) if hasattr(params, shift_attr) else None
            zp_val = getattr(params, zp_attr) if zp_attr and hasattr(params, zp_attr) else None
        else:
            shift_val = None
            zp_val = None
        
        print(format_op_info(op_name, shift_val, zp_val, bitwidth, is_symmetric, is_unsigned))
    
    print("-" * 60)
    
    # 打印门控算子（input）
    for op_name in ['update_gate_input', 'reset_gate_input', 'new_gate_input']:
        if op_name not in _OPERATOR_SHORT_NAME_MAP:
            continue
        attrs = _OPERATOR_SHORT_NAME_MAP[op_name]
        bitwidth = getattr(bitwidth_config, attrs['bw_attr'])
        is_symmetric = getattr(bitwidth_config, attrs['sym_attr'])
        # 获取 unsigned 属性（如果存在）
        unsigned_attr = attrs.get('unsigned_attr')
        is_unsigned = getattr(bitwidth_config, unsigned_attr) if unsigned_attr and hasattr(bitwidth_config, unsigned_attr) else False
        
        if params is not None:
            shift_attr = attrs['scale_attr']
            zp_attr = attrs.get('zp_attr')
            shift_val = getattr(params, shift_attr) if hasattr(params, shift_attr) else None
            zp_val = getattr(params, zp_attr) if zp_attr and hasattr(params, zp_attr) else None
        else:
            shift_val = None
            zp_val = None
        
        print(format_op_info(op_name, shift_val, zp_val, bitwidth, is_symmetric, is_unsigned))
    
    # 打印门控算子（output）
    for op_name in ['update_gate_output', 'reset_gate_output', 'new_gate_output']:
        if op_name not in _OPERATOR_SHORT_NAME_MAP:
            continue
        attrs = _OPERATOR_SHORT_NAME_MAP[op_name]
        bitwidth = getattr(bitwidth_config, attrs['bw_attr'])
        is_symmetric = getattr(bitwidth_config, attrs['sym_attr'])
        # 获取 unsigned 属性（如果存在）
        unsigned_attr = attrs.get('unsigned_attr')
        is_unsigned = getattr(bitwidth_config, unsigned_attr) if unsigned_attr and hasattr(bitwidth_config, unsigned_attr) else False
        
        if params is not None:
            shift_attr = attrs['scale_attr']
            zp_attr = attrs.get('zp_attr')
            shift_val = getattr(params, shift_attr) if hasattr(params, shift_attr) else None
            zp_val = getattr(params, zp_attr) if zp_attr and hasattr(params, zp_attr) else None
        else:
            shift_val = None
            zp_val = None
        
        print(format_op_info(op_name, shift_val, zp_val, bitwidth, is_symmetric, is_unsigned))
    
    print("-" * 60)
    
    # 打印中间算子
    for op_name in ['mul_reset_hidden', 'mul_new_contribution', 'mul_old_contribution']:
        if op_name not in _OPERATOR_SHORT_NAME_MAP:
            continue
        attrs = _OPERATOR_SHORT_NAME_MAP[op_name]
        bitwidth = getattr(bitwidth_config, attrs['bw_attr'])
        is_symmetric = getattr(bitwidth_config, attrs['sym_attr'])
        # 获取 unsigned 属性（如果存在）
        unsigned_attr = attrs.get('unsigned_attr')
        is_unsigned = getattr(bitwidth_config, unsigned_attr) if unsigned_attr and hasattr(bitwidth_config, unsigned_attr) else False
        
        if params is not None:
            shift_attr = attrs['scale_attr']
            zp_attr = attrs.get('zp_attr')
            shift_val = getattr(params, shift_attr) if hasattr(params, shift_attr) else None
            zp_val = getattr(params, zp_attr) if zp_attr and hasattr(params, zp_attr) else None
        else:
            shift_val = None
            zp_val = None
        
        print(format_op_info(op_name, shift_val, zp_val, bitwidth, is_symmetric, is_unsigned))
    
    print("-" * 60)
    
    # 打印权重和偏置（根据 granularity）
    if params is not None:
        # weight_ih 权重（权重通常是 INT 类型，不是 UINT）
        is_unsigned_W = getattr(bitwidth_config, 'W_unsigned_', False) if hasattr(bitwidth_config, 'W_unsigned_') else False
        if bitwidth_config.W_granularity_ == 0:  # PER_TENSOR
            bitwidth = bitwidth_config.W_
            is_symmetric = bitwidth_config.W_symmetric_
            shift_val = params.scale_W_
            print(format_op_info("weight_ih (per-tensor)", shift_val, 0, bitwidth, is_symmetric, is_unsigned_W))
        elif bitwidth_config.W_granularity_ == 1:  # PER_GATE
            bitwidth = bitwidth_config.W_
            is_symmetric = bitwidth_config.W_symmetric_
            shift_val = list(params.scale_W_)
            print(format_op_info("weight_ih (per-gate)", shift_val, 0, bitwidth, is_symmetric, is_unsigned_W))
        elif bitwidth_config.W_granularity_ == 2:  # PER_CHANNEL
            bitwidth = bitwidth_config.W_
            is_symmetric = bitwidth_config.W_symmetric_
            if params.scale_W_ and len(params.scale_W_) > 0:
                # 传递完整列表，格式化函数会处理显示
                shift_val = list(params.scale_W_)
                print(format_op_info("weight_ih (per-channel)", shift_val, None, bitwidth, is_symmetric, is_unsigned_W))
            else:
                print(format_op_info("weight_ih (per-channel)", None, None, bitwidth, is_symmetric, is_unsigned_W))
        
        # weight_hh 权重
        is_unsigned_R = getattr(bitwidth_config, 'R_unsigned_', False) if hasattr(bitwidth_config, 'R_unsigned_') else False
        if bitwidth_config.R_granularity_ == 0:  # PER_TENSOR
            bitwidth = bitwidth_config.R_
            is_symmetric = bitwidth_config.R_symmetric_
            shift_val = params.scale_R_
            print(format_op_info("weight_hh (per-tensor)", shift_val, 0, bitwidth, is_symmetric, is_unsigned_R))
        elif bitwidth_config.R_granularity_ == 1:  # PER_GATE
            bitwidth = bitwidth_config.R_
            is_symmetric = bitwidth_config.R_symmetric_
            shift_val = list(params.scale_R_)
            print(format_op_info("weight_hh (per-gate)", shift_val, 0, bitwidth, is_symmetric, is_unsigned_R))
        elif bitwidth_config.R_granularity_ == 2:  # PER_CHANNEL
            bitwidth = bitwidth_config.R_
            is_symmetric = bitwidth_config.R_symmetric_
            if params.scale_R_ and len(params.scale_R_) > 0:
                shift_val = list(params.scale_R_)
                print(format_op_info("weight_hh (per-channel)", shift_val, None, bitwidth, is_symmetric, is_unsigned_R))
            else:
                print(format_op_info("weight_hh (per-channel)", None, None, bitwidth, is_symmetric, is_unsigned_R))
        
        # bias_ih 偏置
        is_unsigned_bw = getattr(bitwidth_config, 'bw_unsigned_', False) if hasattr(bitwidth_config, 'bw_unsigned_') else False
        if bitwidth_config.bw_granularity_ == 0:  # PER_TENSOR
            bitwidth = bitwidth_config.bw_
            is_symmetric = bitwidth_config.bw_symmetric_
            shift_val = params.scale_bw_
            print(format_op_info("bias_ih (per-tensor)", shift_val, 0, bitwidth, is_symmetric, is_unsigned_bw))
        elif bitwidth_config.bw_granularity_ == 1:  # PER_GATE
            bitwidth = bitwidth_config.bw_
            is_symmetric = bitwidth_config.bw_symmetric_
            shift_val = list(params.scale_bw_)
            print(format_op_info("bias_ih (per-gate)", shift_val, 0, bitwidth, is_symmetric, is_unsigned_bw))
        elif bitwidth_config.bw_granularity_ == 2:  # PER_CHANNEL
            bitwidth = bitwidth_config.bw_
            is_symmetric = bitwidth_config.bw_symmetric_
            if params.scale_bw_ and len(params.scale_bw_) > 0:
                shift_val = list(params.scale_bw_)
                print(format_op_info("bias_ih (per-channel)", shift_val, None, bitwidth, is_symmetric, is_unsigned_bw))
            else:
                print(format_op_info("bias_ih (per-channel)", None, None, bitwidth, is_symmetric, is_unsigned_bw))
        
        # bias_hh 偏置
        is_unsigned_br = getattr(bitwidth_config, 'br_unsigned_', False) if hasattr(bitwidth_config, 'br_unsigned_') else False
        if bitwidth_config.br_granularity_ == 0:  # PER_TENSOR
            bitwidth = bitwidth_config.br_
            is_symmetric = bitwidth_config.br_symmetric_
            shift_val = params.scale_br_
            print(format_op_info("bias_hh (per-tensor)", shift_val, 0, bitwidth, is_symmetric, is_unsigned_br))
        elif bitwidth_config.br_granularity_ == 1:  # PER_GATE
            bitwidth = bitwidth_config.br_
            is_symmetric = bitwidth_config.br_symmetric_
            shift_val = list(params.scale_br_)
            print(format_op_info("bias_hh (per-gate)", shift_val, 0, bitwidth, is_symmetric, is_unsigned_br))
        elif bitwidth_config.br_granularity_ == 2:  # PER_CHANNEL
            bitwidth = bitwidth_config.br_
            is_symmetric = bitwidth_config.br_symmetric_
            if params.scale_br_ and len(params.scale_br_) > 0:
                shift_val = list(params.scale_br_)
                print(format_op_info("bias_hh (per-channel)", shift_val, None, bitwidth, is_symmetric, is_unsigned_br))
            else:
                print(format_op_info("bias_hh (per-channel)", None, None, bitwidth, is_symmetric, is_unsigned_br))
    else:
        # 未校准时，只显示配置信息
        is_unsigned_W = getattr(bitwidth_config, 'W_unsigned_', False) if hasattr(bitwidth_config, 'W_unsigned_') else False
        is_unsigned_R = getattr(bitwidth_config, 'R_unsigned_', False) if hasattr(bitwidth_config, 'R_unsigned_') else False
        is_unsigned_bw = getattr(bitwidth_config, 'bw_unsigned_', False) if hasattr(bitwidth_config, 'bw_unsigned_') else False
        is_unsigned_br = getattr(bitwidth_config, 'br_unsigned_', False) if hasattr(bitwidth_config, 'br_unsigned_') else False
        print(format_op_info("weight_ih", None, None, bitwidth_config.W_, bitwidth_config.W_symmetric_, is_unsigned_W))
        print(format_op_info("weight_hh", None, None, bitwidth_config.R_, bitwidth_config.R_symmetric_, is_unsigned_R))
        print(format_op_info("bias_ih", None, None, bitwidth_config.bw_, bitwidth_config.bw_symmetric_, is_unsigned_bw))
        print(format_op_info("bias_hh", None, None, bitwidth_config.br_, bitwidth_config.br_symmetric_, is_unsigned_br))
    
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
    print(f"  [input]  min={r.min_x_:12.6f}, max={r.max_x_:12.6f}")
    print(f"  [output]  min={r.min_h_:12.6f}, max={r.max_h_:12.6f}")
    print(f"  [Wx] min={r.min_Wx_:12.6f}, max={r.max_Wx_:12.6f}")
    print(f"  [Rh] min={r.min_Rh_:12.6f}, max={r.max_Rh_:12.6f}")
    print("-" * 60)
    print(f"  [update_gate_input]  min={r.min_update_gate_input_:12.6f}, max={r.max_update_gate_input_:12.6f}")
    print(f"  [reset_gate_input]   min={r.min_reset_gate_input_:12.6f}, max={r.max_reset_gate_input_:12.6f}")
    print(f"  [new_gate_input]     min={r.min_new_gate_input_:12.6f}, max={r.max_new_gate_input_:12.6f}")
    print(f"  [update_gate_output] min={r.min_update_gate_output_:12.6f}, max={r.max_update_gate_output_:12.6f}")
    print(f"  [reset_gate_output]  min={r.min_reset_gate_output_:12.6f}, max={r.max_reset_gate_output_:12.6f}")
    print(f"  [new_gate_output]    min={r.min_new_gate_output_:12.6f}, max={r.max_new_gate_output_:12.6f}")
    print("-" * 60)
    print(f"  [mul_reset_hidden]      min={r.min_mul_reset_hidden_:12.6f}, max={r.max_mul_reset_hidden_:12.6f}")
    print(f"  [mul_new_contribution]  min={r.min_mul_new_contribution_:12.6f}, max={r.max_mul_new_contribution_:12.6f}")
    print(f"  [mul_old_contribution]  min={r.min_mul_old_contribution_:12.6f}, max={r.max_mul_old_contribution_:12.6f}")
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
#     - scale: 浮点 scale
#     - zero_point: 零点
#     - enc_type: 量化粒度 (PER_TENSOR/PER_GATE/PER_CHANNEL)
#   - model_info: 模型元信息

# ============================================================
#                   辅助函数：数据类型转换
# ============================================================

def _bitwidth_to_dtype(bitwidth: int, is_unsigned: bool = False) -> str:
    """将位宽转换为 AIMET 风格的 dtype 字符串"""
    prefix = "UINT" if is_unsigned else "INT"
    return f"{prefix}{bitwidth}"


# ============================================================
#                   辅助函数：粒度处理
# ============================================================

def _get_weight_granularity(bitwidth_config, json_key: str) -> int:
    """
    获取权重算子的量化粒度
    
    Args:
        bitwidth_config: OperatorQuantConfig 对象
        json_key: 算子名称 ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')
        
    Returns:
        粒度值: 0=PER_TENSOR, 1=PER_GATE, 2=PER_CHANNEL, None=未配置
    """
    granularity_attr = f"{json_key}_granularity_"
    if hasattr(bitwidth_config, granularity_attr):
        return getattr(bitwidth_config, granularity_attr)
    return None


def _extract_weight_scale_for_export(bitwidth_config, quant_params, op_name: str, op_info: dict, verbose: bool = False) -> tuple:
    """提取权重/偏置算子的 scale 列表与 enc_type。"""
    granularity = _get_weight_granularity(bitwidth_config, op_name)
    scale_attr = op_info["scale_attr"]
    if not hasattr(quant_params, scale_attr):
        return None

    raw_value = getattr(quant_params, scale_attr)
    scale_list = list(raw_value) if hasattr(raw_value, '__iter__') and not isinstance(raw_value, str) else [float(raw_value)]
    scale_list = [float(v) for v in scale_list]

    if granularity == 0:
        enc_type = "PER_TENSOR"
    elif granularity == 1:
        enc_type = "PER_GATE"
    elif granularity == 2:
        enc_type = "PER_CHANNEL"
    else:
        if verbose:
            print(f"  算子 '{op_name}' 未配置 quantization_granularity，按默认 PER_CHANNEL 导出")
        enc_type = "PER_CHANNEL"

    return scale_list, enc_type


def _extract_non_weight_scale_for_export(quant_params, op_info: dict) -> tuple:
    """提取非权重算子的 scale（统一数组格式）与 enc_type。"""
    scale_attr = op_info["scale_attr"]
    if not hasattr(quant_params, scale_attr):
        return None

    raw_value = getattr(quant_params, scale_attr)
    if op_info["is_per_channel"]:
        scale_list = list(raw_value) if hasattr(raw_value, '__iter__') and not isinstance(raw_value, str) else [float(raw_value)]
        return [float(v) for v in scale_list], "PER_CHANNEL"

    if hasattr(raw_value, '__iter__') and not isinstance(raw_value, str):
        scale_list = [float(v) for v in list(raw_value)]
        if len(scale_list) == 0:
            scale_list = [1.0]
        return scale_list, "PER_TENSOR"
    return [float(raw_value)], "PER_TENSOR"


def _scalar_or_list(values: list):
    """导出量化字段时，单元素用标量，多元素保留列表。"""
    if len(values) == 1:
        return values[0]
    return values


def _encode_mshift_py(scale: float) -> tuple:
    if not (scale > 0.0):
        raise ValueError(f"scale 必须 > 0，当前值: {scale}")
    mant, exp2 = math.frexp(float(scale))
    m = int(round(mant * 65536.0))
    if m == 65536:
        m = 32768
        exp2 += 1
    if m < 32768:
        m = 32768
    shift = 16 - exp2
    if shift < -128 or shift > 127:
        raise ValueError(f"encodeMShift shift 超范围: shift={shift}, scale={scale}")
    return int(m), int(shift)


def _decode_fixed_py(multiplier: int, shift: int) -> float:
    return float(multiplier) * math.ldexp(1.0, -int(shift))


def _encode_effective_scale_list(scale_list: list, use_pot2_scale: bool) -> tuple:
    multipliers = []
    shifts = []
    effective_scales = []
    for s in scale_list:
        s = float(s)
        if use_pot2_scale:
            sh = int(round(-math.log2(s)))
            m = 1
            eff = math.ldexp(1.0, -sh)
        else:
            m, sh = _encode_mshift_py(s)
            eff = _decode_fixed_py(m, sh)
        multipliers.append(int(m))
        shifts.append(int(sh))
        effective_scales.append(float(eff))
    return effective_scales, multipliers, shifts


# ============================================================
#                   导出函数：构建 operators 字典
# ============================================================

def _build_single_operator_data(bitwidth_config, quant_params, op_name: str, op_info: dict, verbose: bool = False) -> dict:
    """
    构建单个算子的导出数据
    
    Args:
        bitwidth_config: OperatorQuantConfig 对象
        quant_params: GRUQuantParams 对象
        op_name: 算子名称（如 'weight_ih', 'update_gate_output'）
        op_info: 算子信息字典
        verbose: 是否输出详细信息
        
    Returns:
        算子数据字典，如果无法提取数据则返回 None
    """
    # 读取基本配置
    bitwidth = getattr(bitwidth_config, op_info["bw_attr"])
    is_symmetric = getattr(bitwidth_config, op_info["sym_attr"])
    
    # 读取 is_unsigned
    unsigned_attr = op_info.get("unsigned_attr")
    if unsigned_attr and hasattr(bitwidth_config, unsigned_attr):
        is_unsigned = getattr(bitwidth_config, unsigned_attr)
    else:
        is_unsigned = False
    
    # 计算量化范围
    qmin, qmax = get_quant_range(bitwidth, is_unsigned)
    
    # 提取 scale 值
    if op_name in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
        result = _extract_weight_scale_for_export(bitwidth_config, quant_params, op_name, op_info, verbose)
    else:
        result = _extract_non_weight_scale_for_export(quant_params, op_info)
    
    if result is None:
        return None
    
    scale_values, enc_type = result
    scale_values = [float(v) for v in scale_values]
    
    # 构建算子数据（按 AIMET 字段顺序）
    zp_value = int(getattr(quant_params, op_info["zp_attr"])) if op_info["zp_attr"] and hasattr(quant_params, op_info["zp_attr"]) else 0
    zp_values = [zp_value for _ in scale_values]
    real_min_values = [s * (qmin - zp) for s, zp in zip(scale_values, zp_values)]
    real_max_values = [s * (qmax - zp) for s, zp in zip(scale_values, zp_values)]
    
    op_data = {
        "dtype": _bitwidth_to_dtype(bitwidth, is_unsigned=is_unsigned),
        "symmetric": is_symmetric,
        "scale": _scalar_or_list(scale_values),
        "zero_point": _scalar_or_list(zp_values),
        "enc_type": enc_type,
    }
    
    # 计算 real_min 和 real_max
    op_data["real_min"] = _scalar_or_list(real_min_values)
    op_data["real_max"] = _scalar_or_list(real_max_values)
    
    return op_data


def _build_operators_dict(bitwidth_config, quant_params, verbose: bool = False) -> dict:
    """
    构建统一的 operators 字典（AIMET 兼容格式）
    
    Args:
        bitwidth_config: OperatorQuantConfig 对象
        quant_params: GRUQuantParams 对象
        verbose: 是否输出详细信息
        
    Returns:
        operators 字典（per-channel/per-gate 算子放在最后）
        
    输出字段顺序（AIMET 风格）:
        1. dtype: "INT8" 等
        2. symmetric: true/false
        3. scale: 浮点数或列表
        4. zero_point: 整数
        5. real_min: 量化表示的最小实际值
        6. real_max: 量化表示的最大实际值
        7. enc_type: "PER_TENSOR"/"PER_GATE"/"PER_CHANNEL"
        8. enc_type: 量化粒度标记
    """
    operators = {}
    per_channel_ops = {}  # 存放 per-channel/per-gate 算子，最后再添加
    
    for op_name, op_info in _OPERATOR_MAP.items():
        # 构建单个算子的数据
        op_data = _build_single_operator_data(bitwidth_config, quant_params, op_name, op_info, verbose)
        if op_data is None:
            continue
        
        # 直接使用算子名称（所有算子名称都不使用前缀）
        # 分类：per-tensor 算子放在前面，per-channel/per-gate 算子放在最后
        enc_type = op_data.get("enc_type")
        if enc_type and enc_type in ["PER_CHANNEL", "PER_GATE"]:
            per_channel_ops[op_name] = op_data
        else:
            operators[op_name] = op_data
    
    # 添加 per-channel/per-gate 算子到最后
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


def _parse_runtime_scale_lists(op_name: str, op_data: dict) -> tuple:
    """解析 scale 列表：优先 scale-only，其次兼容旧 multiplier/shift 与 POT2 字段。"""
    if "scale" in op_data:
        exported_scale = op_data["scale"]
        if isinstance(exported_scale, list):
            scales = [float(v) for v in exported_scale]
        else:
            scales = [float(exported_scale)]
        if len(scales) == 0:
            raise ValueError(f"算子 '{op_name}' 的 scale 不能为空")
        if any(v <= 0 for v in scales):
            raise ValueError(f"算子 '{op_name}' 的 scale 必须全部 > 0")
        return scales, "scale_only"

    if "multiplier" in op_data and "shift" in op_data:
        multiplier = op_data["multiplier"]
        shift = op_data["shift"]
        if not isinstance(multiplier, list):
            multiplier = [multiplier]
        if not isinstance(shift, list):
            shift = [shift]
        if len(multiplier) != len(shift):
            raise ValueError(f"算子 '{op_name}' 的 multiplier/shift 长度不一致")
        scales = [_decode_fixed_py(int(m), int(s)) for m, s in zip(multiplier, shift)]
        return scales, "new"

    # 旧 POT2 fallback
    if "n" in op_data or "exp2_inv" in op_data or "shift" in op_data:
        old_shift = op_data.get("n", op_data.get("exp2_inv", op_data.get("shift")))
        if isinstance(old_shift, list):
            shifts = [int(v) for v in old_shift]
        else:
            shifts = [int(old_shift)]
        scales = [math.ldexp(1.0, -s) for s in shifts]
        return scales, "legacy_pot2"

    raise ValueError(f"算子 '{op_name}' 缺少 scale/runtime 编码字段")


# ============================================================
#                   导入函数：解析 operators 字典
# ============================================================

def _parse_weight_operator(bitwidth_config, quant_params, op_name: str, op_data: dict, verbose: bool = False) -> None:
    """
    解析权重算子（weight_ih/weight_hh/bias_ih/bias_hh）的量化参数
    
    Args:
        bitwidth_config: OperatorQuantConfig 对象（会被修改）
        quant_params: GRUQuantParams 对象（会被修改）
        op_name: 算子名称 ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh')
        op_data: 算子数据字典
        verbose: 是否输出详细信息
    """
    if op_name not in _OPERATOR_MAP:
        raise ValueError(f"未知的权重算子名称: {op_name}")
    
    op_info = _OPERATOR_MAP[op_name]
    scale_attr = op_info["scale_attr"]
    enc_type = op_data.get("enc_type")
    if enc_type is None:
        if verbose:
            print(f"  算子 '{op_name}' 在 JSON 中缺少 'enc_type' 字段，按默认 PER_CHANNEL 解析")
        enc_type = "PER_CHANNEL"
    scale_list, _ = _parse_runtime_scale_lists(op_name, op_data)

    if enc_type == "PER_TENSOR":
        setattr(bitwidth_config, f"{op_info['bw_attr']}granularity_", 0)
        if len(scale_list) == 0:
            raise ValueError(f"算子 '{op_name}' 的 PER_TENSOR scale 不能为空")
        tensor_attr = f"scale_{op_info['bw_attr']}tensor_"
        if not hasattr(quant_params, tensor_attr):
            raise AttributeError(f"GRUQuantParams 缺少属性 '{tensor_attr}'")
        setattr(quant_params, tensor_attr, float(scale_list[0]))
    elif enc_type == "PER_GATE":
        setattr(bitwidth_config, f"{op_info['bw_attr']}granularity_", 1)
        if len(scale_list) != 3:
            raise ValueError(
                f"算子 '{op_name}' 的 PER_GATE scale 长度必须为 3，当前为 {len(scale_list)}"
            )
        gate_attr = f"scale_{op_info['bw_attr']}gate_"
        if not hasattr(quant_params, gate_attr):
            raise AttributeError(f"GRUQuantParams 缺少属性 '{gate_attr}'")
        setattr(quant_params, gate_attr, [float(v) for v in scale_list])
    else:
        setattr(bitwidth_config, f"{op_info['bw_attr']}granularity_", 2)
        setattr(quant_params, scale_attr, [float(v) for v in scale_list])


def _parse_non_weight_operator(bitwidth_config, quant_params, op_info: dict, op_data: dict) -> None:
    """
    解析非权重算子的量化参数
    
    注意：现在导出时 per-tensor 也是数组格式，所以导入时需要支持数组格式。
    
    Args:
        bitwidth_config: OperatorQuantConfig 对象（会被修改）
        quant_params: GRUQuantParams 对象（会被修改）
        op_info: 算子信息字典
        op_data: 算子数据字典
    """
    scale_attr = op_info["scale_attr"]
    
    # 调试信息：检查属性是否存在
    if not hasattr(quant_params, scale_attr):
        print(f"\n[DEBUG _parse_non_weight_operator] ❌ 错误：quant_params 没有属性 '{scale_attr}'")
        print(f"  op_info: {op_info}")
        print(f"  quant_params 的所有相关属性 (包含 'weight_ih'):")
        for attr in dir(quant_params):
            if not attr.startswith('_') and 'weight_ih' in attr:
                print(f"    - {attr}")
        raise AttributeError(f"'gru_interface_binding.GRUQuantParams' object has no attribute '{scale_attr}'")

    scale_list, _ = _parse_runtime_scale_lists(op_info["bw_attr"].rstrip("_"), op_data)
    if op_info["is_per_channel"]:
        setattr(quant_params, scale_attr, [float(v) for v in scale_list])
    else:
        if len(scale_list) == 0:
            raise ValueError("PER_TENSOR granularity requires non-empty scale list")
        setattr(quant_params, scale_attr, float(scale_list[0]))


def _parse_single_operator(bitwidth_config, quant_params, op_name: str, op_data: dict, verbose: bool = False) -> None:
    """
    解析单个算子的量化参数
    
    Args:
        bitwidth_config: OperatorQuantConfig 对象（会被修改）
        quant_params: GRUQuantParams 对象（会被修改）
        op_name: 算子名称（JSON key，如 'weight_ih', 'update_gate_output'）
        op_data: 算子数据字典
        verbose: 是否输出详细信息
    """
    # 直接使用 op_name 作为 _OPERATOR_MAP 的键（所有算子名称都不使用前缀）
    if op_name not in _OPERATOR_MAP:
        print(f"[DEBUG _parse_single_operator] ⚠️  op_name '{op_name}' 不在 _OPERATOR_MAP 中")
        return
    
    op_info = _OPERATOR_MAP[op_name]
    
    # 错误检查：如果属性不存在，打印详细信息（仅在出错时）
    if not hasattr(quant_params, op_info['scale_attr']):
        print(f"\n[DEBUG _parse_single_operator] ❌ 错误：处理算子 '{op_name}' 时发现属性缺失")
        print(f"  op_info['scale_attr']: {op_info['scale_attr']}")
        print(f"  op_info['zp_attr']: {op_info.get('zp_attr')}")
        print(f"  quant_params 的所有相关属性 (包含 '{op_name.split('_')[0]}'):")
        for attr in dir(quant_params):
            if not attr.startswith('_') and op_name.split('_')[0] in attr:
                print(f"    - {attr}")
    
    # 设置 bitwidth 和 is_unsigned（从 dtype 解析）
    if "dtype" in op_data:
        dtype_str = op_data["dtype"]
        setattr(bitwidth_config, op_info["bw_attr"], _dtype_to_bitwidth(dtype_str))
        unsigned_attr = op_info.get("unsigned_attr")
        if unsigned_attr:
            setattr(bitwidth_config, unsigned_attr, _dtype_to_is_unsigned(dtype_str))
    
    # 设置 symmetric
    if "symmetric" in op_data:
        setattr(bitwidth_config, op_info["sym_attr"], op_data["symmetric"])
    
    # 设置 shift 值（根据算子类型选择不同的解析方法）
    if op_name in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
        _parse_weight_operator(bitwidth_config, quant_params, op_name, op_data, verbose)
    else:
        _parse_non_weight_operator(bitwidth_config, quant_params, op_info, op_data)
    
    # 设置 zero_point
    if op_info["zp_attr"] and "zero_point" in op_data:
        zp_value = op_data["zero_point"]
        # 支持列表格式（per-tensor 量化可能也是数组格式，取第一个值）
        if isinstance(zp_value, list):
            if len(zp_value) > 0:
                # 取第一个值（所有值应该相同）
                setattr(quant_params, op_info["zp_attr"], int(zp_value[0]))
            else:
                raise ValueError(f"zero_point 列表为空")
        else:
            # 标量值
            setattr(quant_params, op_info["zp_attr"], int(zp_value))


def _parse_operators_dict(operators: dict, bitwidth_config, quant_params, verbose: bool = False) -> None:
    """
    从 operators 字典解析并设置 bitwidth_config 和 quant_params
    
    AIMET 风格字段名：
        - dtype: "INT8" 等 → 位宽
        - symmetric: true/false → 对称量化
        - n: 量化指数（优先）
        - scale: 浮点 scale（当无 n 时使用）
        - zero_point: 零点
        - enc_type: 量化粒度 (PER_TENSOR/PER_GATE/PER_CHANNEL)
    
    Args:
        operators: operators 字典
        bitwidth_config: OperatorQuantConfig 对象（会被修改）
        quant_params: GRUQuantParams 对象（会被修改）
        verbose: 是否输出详细信息
    """
    for op_name, op_data in operators.items():
        _parse_single_operator(bitwidth_config, quant_params, op_name, op_data, verbose)


def _ensure_quant_params_fresh_for_export(gru: 'QuantGRU', verbose: bool = False) -> None:
    """导出前确保 quant_params 与当前配置一致（消除懒同步导致的 stale 导出）。

    QuantGRU 采用懒同步：改配置（use_pot2_scale / 位宽 / granularity / 重新收集
    校准数据等）只标记 _quant_params_dirty=True，真正重算 quant_params 推迟到下一次
    forward。导出路径不经过 forward，若不在此显式重算，会静默导出与当前配置不一致的
    旧 scale。此函数与 forward 中的 dirty 处理保持相同语义。
    """
    if not getattr(gru, '_quant_params_dirty', False):
        return

    # 锁定且已有参数：与 forward 路径一致，视为有效并清脏，不覆盖
    if getattr(gru, '_quant_params_locked', False) and gru.quant_params is not None:
        gru._quant_params_dirty = False
        return

    # 判断是否仍持有可重算的校准数据（与 finalize_calibration 的前置检查一致）
    try:
        if gru._use_histogram_collection():
            has_calib_data = gru.hist_collectors is not None and gru.hist_collectors.is_valid()
        else:
            has_calib_data = gru.quant_ranges is not None
    except Exception:
        has_calib_data = gru.quant_ranges is not None

    if not has_calib_data:
        raise RuntimeError(
            "导出失败：量化参数已过期（配置变更后未重新计算），且校准数据已丢失，"
            "无法安全导出。\n"
            "请重新校准，或在改动配置后、导出前调用 finalize_calibration()。\n"
            "提示：pickle/deepcopy 后校准数据会丢失，需要重新校准。"
        )

    if verbose:
        print("[QuantGRU] 导出前检测到 _quant_params_dirty=True，"
              "自动重新 finalize_calibration() 以保证导出与当前配置一致 ...")
    gru.finalize_calibration(verbose=verbose)


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

    # 配置变更后 quant_params 可能过期（懒同步），导出前强制对齐
    _ensure_quant_params_fresh_for_export(gru, verbose=verbose)
    
    # 构建导出数据（统一的 operators 结构）
    export_data = {
        "model_info": {
            "input_size": gru.input_size,
            "hidden_size": gru.hidden_size,
            "bias": gru.bias,
            "batch_first": gru.batch_first,
            "bidirectional": gru.bidirectional,
            "use_pot2_scale": bool(gru.use_pot2_scale),
        },
        "operators": _build_operators_dict(gru._bitwidth_config, gru.quant_params, verbose),
    }
    
    # 双向 GRU 导出反向参数
    if gru.bidirectional and gru.quant_params_reverse is not None:
        export_data["operators_reverse"] = _build_operators_dict(
            gru._bitwidth_config, gru.quant_params_reverse, verbose
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
    W, R, bw, br = convert_weights_to_haste_format(
        gru.weight_ih_l0, gru.weight_hh_l0,
        gru.bias_ih_l0 if gru.bias else None,
        gru.bias_hh_l0 if gru.bias else None,
        gru.hidden_size, device
    )
    
    # 量化权重（使用 per-channel 参数）
    def quantize_per_channel(tensor, scale_list, bitwidth, symmetric):
        """对每个 channel 应用量化（权重使用有符号量化）"""
        qmin, qmax = get_quant_range(bitwidth)  # 权重使用有符号量化（默认）
        result = torch.zeros_like(tensor, dtype=torch.int32)
        
        for c in range(len(scale_list)):
            scale = float(scale_list[c])
            q = torch.clamp(torch.round(tensor[:, c] / scale), qmin, qmax)
            result[:, c] = q.int()
        
        return result.tolist()
    
    # 获取位宽配置
    W_bitwidth = gru._bitwidth_config.W_
    R_bitwidth = gru._bitwidth_config.R_
    bw_bitwidth = gru._bitwidth_config.bw_
    br_bitwidth = gru._bitwidth_config.br_
    
    weights = {
        "W": quantize_per_channel(W, list(params.scale_W_), W_bitwidth, True),
        "R": quantize_per_channel(R, list(params.scale_R_), R_bitwidth, True),
    }
    
    if gru.bias:
        # 偏置是 1D，需要 unsqueeze
        bw_2d = bw.unsqueeze(0)  # [1, 3*H]
        br_2d = br.unsqueeze(0)
        weights["bw"] = quantize_per_channel(bw_2d, list(params.scale_bw_), bw_bitwidth, True)[0]
        weights["br"] = quantize_per_channel(br_2d, list(params.scale_br_), br_bitwidth, True)[0]
    
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
        if verbose:
            print(f"  ⚠️ 警告：batch_first 不匹配: 文件中为 {model_info.get('batch_first')}, "
                  f"模型为 {gru.batch_first}。这不影响量化参数，但请确保输入数据格式正确。")
    
    # 验证版本和格式
    if "operators" not in data:
        raise ValueError(
            f"不支持的量化参数格式，缺少 'operators' 字段。\n"
            f"当前包含字段: {list(data.keys())}"
        )
    
    # 解析 operators 字典
    gru._bitwidth_config = gru_ops.OperatorQuantConfig()
    gru.use_pot2_scale = bool(model_info.get("use_pot2_scale", False))
    gru._bitwidth_config.usePOT2_ = gru.use_pot2_scale
    gru.quant_params = gru_ops.GRUQuantParams()
    gru.quant_params.hidden_ = gru.hidden_size
    
    _parse_operators_dict(data["operators"], gru._bitwidth_config, gru.quant_params, verbose)
    
    # 设置位宽配置到量化参数中（pybind11 值复制语义，这里是复制而非引用）
    gru.quant_params.bitwidth_config_ = gru._bitwidth_config
    
    # 注意：LUT 会在 forward 时通过 to_cpp() 自动生成，无需在此显式调用
    
    # 加载反向参数（双向）
    # 设计说明：正向和反向共用同一个 _bitwidth_config，这是有意为之：
    #   1. 硬件实现：正向/反向 GRU 计算单元通常时分复用，相同位宽简化硬件设计
    #   2. 模型对称性：双向 GRU 的正反向是对称结构，应使用对称的量化配置
    #   3. 导出时 operators 和 operators_reverse 的 bitwidth 来自同一 _bitwidth_config，
    #      所以导入时解析两次 bitwidth 是等价的（值相同）
    if gru.bidirectional and "operators_reverse" not in data:
        raise ValueError("bidirectional=True 时导入文件必须包含 operators_reverse")

    if gru.bidirectional and "operators_reverse" in data:
        gru.quant_params_reverse = gru_ops.GRUQuantParams()
        gru.quant_params_reverse.hidden_ = gru.hidden_size
        
        # 从 operators_reverse 解析 scale/zp，bitwidth 与正向相同（共用 _bitwidth_config）
        _parse_operators_dict(data["operators_reverse"], gru._bitwidth_config, gru.quant_params_reverse, verbose)
        gru.quant_params_reverse.bitwidth_config_ = gru._bitwidth_config
        # 输入量化参数在双向中必须一致
        fwd_input = data["operators"].get("input", {})
        rev_input = data["operators_reverse"].get("input", {})
        for field in ("multiplier", "shift", "zero_point"):
            if field in fwd_input or field in rev_input:
                if fwd_input.get(field) != rev_input.get(field):
                    raise ValueError(f"双向导入失败：operators.input 与 operators_reverse.input 的 {field} 不一致")
    
    # 清除脏标志
    gru._quant_params_dirty = False
    
    if verbose:
        print(f"\n[QuantGRU] 量化参数已从 {import_path} 加载")
        print(f"  - 模型配置: input_size={gru.input_size}, hidden_size={gru.hidden_size}")
        print(f"  - 双向: {gru.bidirectional}")
        print(f"  - is_calibrated(): {gru.is_calibrated()}")
        # 打印详细的量化配置信息
        print_quant_params(gru)


def _adjust_quant_config_impl(
    gru: 'QuantGRU',
    operator: str,
    bitwidth: int = None,
    is_symmetric: bool = None,
    scale: float = None,
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
    
    if is_symmetric is not None:
        setattr(gru._bitwidth_config, attrs['sym_attr'], is_symmetric)
        new_values['is_symmetric'] = is_symmetric
    
    # 修改量化参数
    if gru.quant_params is not None:
        scale_attr = attrs['scale_attr']
        zp_attr = attrs['zp_attr']
        is_per_channel = attrs['is_per_channel']
        
        if hasattr(gru.quant_params, scale_attr):
            old_scale = getattr(gru.quant_params, scale_attr)
            old_values['scale'] = list(old_scale) if hasattr(old_scale, '__iter__') and not isinstance(old_scale, str) else float(old_scale)
            if scale is not None:
                if is_per_channel:
                    old_scale_list = list(old_scale)
                    if not isinstance(scale, (list, tuple)):
                        raise ValueError("per-channel/per-gate 算子调整 scale 必须传 list，禁止 scalar 广播")
                    if len(scale) != len(old_scale_list):
                        raise ValueError(f"scale 长度不匹配: 期望 {len(old_scale_list)}，实际 {len(scale)}")
                    requested = [float(v) for v in scale]
                    if any(v <= 0 for v in requested):
                        raise ValueError("scale 列表中的所有元素必须 > 0")
                    setattr(gru.quant_params, scale_attr, requested)
                    new_values['scale'] = requested
                else:
                    scalar_scale = float(scale[0]) if isinstance(scale, (list, tuple)) else float(scale)
                    if scalar_scale <= 0:
                        raise ValueError(f"scale 必须 > 0，当前值: {scalar_scale}")
                    setattr(gru.quant_params, scale_attr, scalar_scale)
                    new_values['scale'] = scalar_scale
            
            if zp_attr and hasattr(gru.quant_params, zp_attr):
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
            scale_attr = attrs['scale_attr']
            zp_attr = attrs['zp_attr']
            is_per_channel = attrs['is_per_channel']
            
            if is_per_channel:
                if hasattr(gru.quant_params, scale_attr):
                    scales = [float(v) for v in list(getattr(gru.quant_params, scale_attr))]
                    config['scale'] = scales
                    if zp_attr and hasattr(gru.quant_params, zp_attr):
                        config['zero_point'] = [int(getattr(gru.quant_params, zp_attr)) for _ in scales]
            else:
                if hasattr(gru.quant_params, scale_attr):
                    config['scale'] = float(getattr(gru.quant_params, scale_attr))
                if zp_attr and hasattr(gru.quant_params, zp_attr):
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
    打印量化配置（已合并到 print_quant_params）
    
    为了向后兼容，此函数现在调用 print_quant_params。
    如果指定了 operators 参数，会给出警告（当前实现不支持选择性打印）。
    
    Args:
        gru: QuantGRU 实例
        operators: 已废弃，保留以兼容旧代码，当前会忽略此参数
        
    Example:
        >>> print_quant_config(gru)  # 调用 print_quant_params
    """
    if operators is not None:
        import warnings
        warnings.warn(
            "print_quant_config 的 operators 参数已废弃，当前会打印所有算子。"
            "请使用 print_quant_params 函数。",
            DeprecationWarning
        )
    
    # 直接调用合并后的函数
    print_quant_params(gru)


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
    print(f"  [输入]  input: {_format_bitwidth(config.x_):6s} ({_format_symmetric(config.x_symmetric_)})")
    print(f"  [输出]  output: {_format_bitwidth(config.h_):6s} ({_format_symmetric(config.h_symmetric_)})")
    print(f"  [权重]  weight_ih: {_format_bitwidth(config.W_):6s} ({_format_symmetric(config.W_symmetric_)})")
    print(f"          weight_hh: {_format_bitwidth(config.R_):6s} ({_format_symmetric(config.R_symmetric_)})")
    print(f"          bias_ih: {_format_bitwidth(config.bw_):6s} ({_format_symmetric(config.bw_symmetric_)})")
    print(f"          bias_hh: {_format_bitwidth(config.br_):6s} ({_format_symmetric(config.br_symmetric_)})")
    print(f"  [矩阵]  weight_ih_linear: {_format_bitwidth(config.weight_ih_linear_):6s} ({_format_symmetric(config.weight_ih_linear_symmetric_)})")
    print(f"          weight_hh_linear: {_format_bitwidth(config.weight_hh_linear_):6s} ({_format_symmetric(config.weight_hh_linear_symmetric_)})")
    print(f"  [门控]  update_gate_input: {_format_bitwidth(config.update_gate_input_):6s} ({_format_symmetric(config.update_gate_input_symmetric_)})")
    print(f"          update_gate_output: {_format_bitwidth(config.update_gate_output_):6s} ({_format_symmetric(config.update_gate_output_symmetric_)})")
    print(f"          reset_gate_input: {_format_bitwidth(config.reset_gate_input_):6s} ({_format_symmetric(config.reset_gate_input_symmetric_)})")
    print(f"          reset_gate_output: {_format_bitwidth(config.reset_gate_output_):6s} ({_format_symmetric(config.reset_gate_output_symmetric_)})")
    print(f"          new_gate_input: {_format_bitwidth(config.new_gate_input_):6s} ({_format_symmetric(config.new_gate_input_symmetric_)})")
    print(f"          new_gate_output: {_format_bitwidth(config.new_gate_output_):6s} ({_format_symmetric(config.new_gate_output_symmetric_)})")
    print(f"  [运算]  mul_reset_hidden: {_format_bitwidth(config.mul_reset_hidden_):6s} ({_format_symmetric(config.mul_reset_hidden_symmetric_)})")
    print(
        f"  [输出]  mul_old_contribution: {_format_bitwidth(config.mul_old_contribution_):6s} ({_format_symmetric(config.mul_old_contribution_symmetric_)})")
    print(
        f"          mul_new_contribution: {_format_bitwidth(config.mul_new_contribution_):6s} ({_format_symmetric(config.mul_new_contribution_symmetric_)})")
    print("=" * 70 + "\n")


def set_quant_gru_module_names(model: torch.nn.Module, prefix: str = "") -> None:
    """
    为模型中的所有 QuantGRU 模块自动设置模块名称
    
    遍历模型的所有子模块，找到 QuantGRU 实例，并设置它们的模块名称。
    模块名称使用完整的路径（例如 "generator.enc_seqs.0.seq_t"）。
    
    Args:
        model: PyTorch 模型
        prefix: 模块名称前缀（可选，用于递归调用）
    
    Example:
        >>> model = MyModel()
        >>> set_quant_gru_module_names(model)
        >>> # 现在所有 QuantGRU 模块都有了模块名称，调试信息会显示模块名称
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantGRU):
            module.set_module_name(name)
