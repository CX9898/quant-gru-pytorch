"""
QuantGRU 量化库使用示例

示例编号与 README「常见场景索引」一致（1–11），另提供 debug 调试示例。

运行方式:
    python example_usage.py              # 运行全部（部分示例含 8bit/16bit）
    python example_usage.py --list       # 列出可运行示例
    python example_usage.py -e onnx      # 仅运行 ONNX 导出
    python example_usage.py -e manual --bitwidth 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_gru import (
    QuantGRU,
    ensure_quant_gru_onnx_registered,
    get_quant_gru_custom_opsets,
    print_quant_config,
    print_quant_params,
)

# 默认张量尺寸（各示例共用）
INPUT_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 8
SEQ_LEN = 20


def _pytorch_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_gru(*, hidden_size: int = HIDDEN_SIZE, device: str = "cuda", **kwargs) -> QuantGRU:
    gru = QuantGRU(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        batch_first=True,
        **kwargs,
    )
    return gru.to(device)


def _copy_gru_weights(dst: QuantGRU, src: QuantGRU) -> None:
    dst.weight_ih_l0.data.copy_(src.weight_ih_l0.data)
    dst.weight_hh_l0.data.copy_(src.weight_hh_l0.data)
    dst.bias_ih_l0.data.copy_(src.bias_ih_l0.data)
    dst.bias_hh_l0.data.copy_(src.bias_hh_l0.data)


def _calibrate(gru: QuantGRU, data: torch.Tensor, *, batches: int = 1) -> None:
    gru.calibrating = True
    for _ in range(batches):
        batch = data if batches == 1 else torch.randn_like(data)
        gru(batch)
    gru.calibrating = False


def _banner(example_no: int | str, title: str) -> None:
    print("\n" + "=" * 60)
    print(f"示例 {example_no}: {title}")
    print("=" * 60)


def example_basic_usage():
    """示例 1: 基本使用（非量化）"""
    _banner(1, "基本使用（非量化）")

    gru = _make_gru()
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=gru.weight_ih_l0.device)
    output, h_n = gru(x)

    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {output.shape}")
    print(f"隐藏状态:   {h_n.shape}")
    print("✅ 基本使用完成！")


def example_quantization_with_json():
    """
    示例 2: 使用 JSON 配置进行量化

    权重/偏置可在 JSON 中设置 quantization_granularity，详见示例 11。
    """
    _banner(2, "使用 JSON 配置进行量化")

    gru = _make_gru()
    config_path = os.path.join(_pytorch_dir(), "config/gru_quant_bitwidth_config.json")
    gru.load_bitwidth_config(config_path)
    print(f"✅ 加载配置: {config_path}")
    print(f"   量化开关: use_quantization = {gru.use_quantization}")

    print("\n📊 开始校准...")
    calibration_data = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=gru.weight_ih_l0.device)
    _calibrate(gru, calibration_data)
    print("✅ 校准完成！")

    print("\n🚀 开始推理...")
    gru.use_quantization = True
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=gru.weight_ih_l0.device)
    output, h_n = gru(x)

    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {output.shape}")
    print(f"隐藏状态:   {h_n.shape}")
    print("✅ 量化推理完成！")


def example_quantization_manual(bitwidth: int = 8):
    """示例 3: 手动配置量化参数（set_all_bitwidth，支持 1–32 bit）"""
    _banner(3, f"手动配置量化参数 ({bitwidth}bit)")

    gru = _make_gru()
    gru.set_all_bitwidth(bitwidth)
    print(f"✅ 设置位宽: {bitwidth}bit 对称量化")

    print("\n📊 开始校准...")
    calibration_data = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=gru.weight_ih_l0.device)
    _calibrate(gru, calibration_data)
    print("✅ 校准完成！")

    gru.use_quantization = True
    print(f"   量化开关: use_quantization = {gru.use_quantization}")

    print("\n🚀 开始推理...")
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=gru.weight_ih_l0.device)
    output, h_n = gru(x)

    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {output.shape}")
    print(f"隐藏状态:   {h_n.shape}")
    print(f"✅ {bitwidth}bit 量化推理完成！")


def example_compare_precision(bitwidth: int = 8):
    """示例 4: 比较量化前后的精度差异"""
    _banner(4, f"比较量化前后的精度差异 ({bitwidth}bit)")

    device = "cuda"
    gru_float = _make_gru(device=device, use_quantization=False)
    quant_gru = _make_gru(device=device)
    _copy_gru_weights(quant_gru, gru_float)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)
    quant_gru.set_all_bitwidth(bitwidth)
    _calibrate(quant_gru, x)
    quant_gru.use_quantization = True

    gru_float.eval()
    quant_gru.eval()
    with torch.no_grad():
        output_float, _ = gru_float(x)
        output_quant, _ = quant_gru(x)

    mse = torch.mean((output_float - output_quant) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        output_float.flatten().unsqueeze(0),
        output_quant.flatten().unsqueeze(0),
    ).item()

    print(f"📊 {bitwidth}bit 精度比较结果:")
    print(f"   MSE (均方误差):     {mse:.6f}")
    print(f"   余弦相似度:         {cos_sim:.6f}")
    print(f"✅ {bitwidth}bit 精度比较完成！")


def example_training(bitwidth: int = 8):
    """示例 5: 量化感知训练（QAT）"""
    _banner(5, f"量化感知训练 ({bitwidth}bit)")

    hidden_size = 64
    num_epochs = 5
    gru = _make_gru(hidden_size=hidden_size)
    device = gru.weight_ih_l0.device

    torch.manual_seed(42)
    x_train = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device) * 0.5
    target_train = x_train * 0.5

    gru.set_all_bitwidth(bitwidth)
    _calibrate(gru, x_train)
    gru.use_quantization = True

    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    gru.train()
    print(f"\n🏋️ 开始 {bitwidth}bit 量化训练...")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output, _ = gru(x_train)
        loss = torch.mean((output - target_train) ** 2)
        loss.backward()
        optimizer.step()
        print(f"   Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

    print(f"✅ {bitwidth}bit 训练完成！（Loss 应持续下降）")


def example_calibration_method():
    """示例 6: 校准方法选择（MinMax / SQNR / Percentile）"""
    _banner(6, "校准方法选择")

    device = "cuda"
    gru_base = _make_gru(device=device, use_quantization=False)

    torch.manual_seed(42)
    test_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)

    gru_base.eval()
    with torch.no_grad():
        fp32_output, _ = gru_base(test_input)

    print("\n📊 对比三种校准方法:")
    print("-" * 50)

    for method in ("minmax", "sqnr", "percentile"):
        quant_gru = _make_gru(device=device)
        _copy_gru_weights(quant_gru, gru_base)
        quant_gru.calibration_method = method
        if method == "percentile":
            quant_gru.percentile_value = 99.99
        quant_gru.set_all_bitwidth(16)
        _calibrate(quant_gru, test_input, batches=3)
        quant_gru.use_quantization = True
        quant_gru.eval()

        with torch.no_grad():
            quant_output, _ = quant_gru(test_input)

        cos_sim = torch.nn.functional.cosine_similarity(
            fp32_output.flatten().unsqueeze(0),
            quant_output.flatten().unsqueeze(0),
        ).item()
        method_desc = {
            "minmax": "MinMax (快速)",
            "sqnr": "SQNR (高精度)",
            "percentile": "Percentile (抗异常值)",
        }[method]
        print(f"   {method_desc:<25} 余弦相似度: {cos_sim:.6f}")

    print("-" * 50)
    print("\n💡 选择建议:")
    print("   • minmax:     校准速度快，适合快速迭代和调试")
    print("   • sqnr:       精度更高，搜索最优 scale（推荐）")
    print("   • percentile: 对异常值鲁棒，适合含噪声数据")
    print("   默认使用 'minmax' 方法")
    print("✅ 校准方法对比完成！")


def example_bidirectional():
    """示例 7: 双向 GRU"""
    _banner(7, "双向 GRU")

    gru = _make_gru(bidirectional=True)
    device = gru.weight_ih_l0.device
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)

    gru.set_all_bitwidth(8)
    _calibrate(gru, x)
    gru.use_quantization = True
    output, h_n = gru(x)

    print(f"输入形状:   {x.shape}")
    print(f"输出形状:   {output.shape}  (hidden_size * 2 = {HIDDEN_SIZE * 2})")
    print(f"隐藏状态:   {h_n.shape}  (num_directions = 2)")
    print("✅ 双向 GRU 完成！")


def example_onnx_export():
    """示例 8: ONNX 导出（标准 GRU 单节点）"""
    _banner(8, "ONNX 导出")

    gru = _make_gru(device="cpu")
    opset = 18
    gru.export_mode = True
    ensure_quant_gru_onnx_registered(opset=opset)
    gru.eval()
    print(f"   export_mode = {gru.export_mode}, opset = {opset}")

    dummy_input = torch.randn(1, SEQ_LEN, INPUT_SIZE)
    onnx_path = os.path.join(_pytorch_dir(), "quant_gru_example.onnx")

    torch.onnx.export(
        gru,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output", "hidden"],
        dynamic_axes={
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset,
        dynamo=False,
        custom_opsets=get_quant_gru_custom_opsets(),
        verbose=False,
    )
    print(f"   ✅ 导出成功: {onnx_path}")

    try:
        import onnx

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        gru_nodes = [n for n in model.graph.node if n.op_type == "GRU"]
        print(f"   ✅ ONNX 校验通过，GRU 节点数: {len(gru_nodes)}")
    except ImportError:
        print("   ⚠️ 未安装 onnx 库，跳过验证")
    except Exception as exc:
        print(f"   ⚠️ 验证失败: {exc}")

    gru.export_mode = False
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    print("✅ ONNX 导出示例完成！")


def example_quant_params_export_import():
    """示例 9: 量化参数导出/导入"""
    _banner(9, "量化参数导出/导入")

    device = "cuda"
    gru_train = _make_gru(device=device)
    gru_train.set_all_bitwidth(8)
    gru_train.calibration_method = "sqnr"

    calibration_data = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)
    _calibrate(gru_train, calibration_data)
    gru_train.finalize_calibration()
    print("   ✅ 校准完成")

    quant_params_path = os.path.join(_pytorch_dir(), "example_quant_params.json")
    weights_path = os.path.join(_pytorch_dir(), "example_weights.pth")
    gru_train.export_quant_params(quant_params_path, verbose=True)
    torch.save(gru_train.state_dict(), weights_path)

    with open(quant_params_path, encoding="utf-8") as f:
        model_info = json.load(f)["model_info"]

    gru_deploy = QuantGRU(
        input_size=model_info["input_size"],
        hidden_size=model_info["hidden_size"],
        batch_first=model_info["batch_first"],
        bidirectional=model_info["bidirectional"],
    ).to(device)
    gru_deploy.load_state_dict(torch.load(weights_path, weights_only=True))
    gru_deploy.load_quant_params(quant_params_path, verbose=True)
    gru_deploy.use_quantization = True

    gru_train.use_quantization = True
    gru_train.eval()
    gru_deploy.eval()

    test_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)
    with torch.no_grad():
        output_train, _ = gru_train(test_input)
        output_deploy, _ = gru_deploy(test_input)

    mse = torch.mean((output_train - output_deploy) ** 2).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        output_train.flatten().unsqueeze(0),
        output_deploy.flatten().unsqueeze(0),
    ).item()
    print(f"   训练 vs 部署 MSE: {mse:.10f}, 余弦相似度: {cos_sim:.6f}")

    for path in (quant_params_path, weights_path):
        if os.path.exists(path):
            os.remove(path)
    print("✅ 量化参数导出/导入示例完成！")


def example_adjust_quant_config():
    """示例 10: 调整量化配置"""
    _banner(10, "调整量化配置")

    gru = _make_gru()
    device = gru.weight_ih_l0.device
    gru.set_all_bitwidth(8)

    calibration_data = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)
    _calibrate(gru, calibration_data)
    gru.finalize_calibration()

    print("\n📋 update_gate_output 配置:")
    print(f"   {gru.get_quant_config('update_gate_output')}")
    print("\n📊 所有量化配置:")
    print_quant_config(gru)

    gru.use_quantization = True
    gru.eval()
    test_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)

    with torch.no_grad():
        output_before, _ = gru(test_input)

    gru.adjust_quant_config("update_gate_output", bitwidth=16, verbose=True)

    with torch.no_grad():
        output_after, _ = gru(test_input)

    diff = torch.mean((output_before - output_after) ** 2).item()
    print(f"\n   调整前后输出差异 (MSE): {diff:.8f}")
    print(f"   调整后配置: {gru.get_quant_config('update_gate_output')}")
    print("✅ 量化配置调整示例完成！")


def example_weight_bias_granularity():
    """
    示例 11: 权重和偏置的量化粒度（JSON quantization_granularity）

    通过临时 JSON 配置 PER_TENSOR / PER_GATE / PER_CHANNEL，勿使用私有 _bitwidth_config 字段。
    """
    _banner(11, "权重和偏置的量化粒度设置")

    device = "cuda"
    gru_base = _make_gru(device=device, use_quantization=False)

    torch.manual_seed(42)
    test_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)
    gru_base.eval()
    with torch.no_grad():
        fp32_output, _ = gru_base(test_input)

    granularity_configs = [
        ("PER_TENSOR", "整个 tensor 一个 scale"),
        ("PER_GATE", "每个门一个 scale（z/r/n）"),
        ("PER_CHANNEL", "每个输出通道一个 scale（默认）"),
    ]

    print("\n📊 对比三种量化粒度:")
    print("-" * 60)
    results = {}

    weight_ops = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    with tempfile.TemporaryDirectory() as tmpdir:
        for granularity_name, description in granularity_configs:
            print(f"\n🔧 {granularity_name}: {description}")

            config_path = os.path.join(tmpdir, f"granularity_{granularity_name}.json")
            operator_config = {
                op: {
                    "bitwidth": 8,
                    "is_symmetric": True,
                    "is_unsigned": False,
                    "quantization_granularity": granularity_name,
                }
                for op in weight_ops
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "GRU_config": {
                            "default_config": {"disable_quantization": False},
                            "operator_config": operator_config,
                        }
                    },
                    f,
                    indent=2,
                )

            quant_gru = _make_gru(device=device)
            _copy_gru_weights(quant_gru, gru_base)
            quant_gru.load_bitwidth_config(config_path)
            quant_gru.calibration_method = "minmax"
            _calibrate(quant_gru, test_input)
            quant_gru.finalize_calibration()
            quant_gru.use_quantization = True
            quant_gru.eval()

            with torch.no_grad():
                quant_output, _ = quant_gru(test_input)

            mse = torch.mean((fp32_output - quant_output) ** 2).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                fp32_output.flatten().unsqueeze(0),
                quant_output.flatten().unsqueeze(0),
            ).item()
            results[granularity_name] = (mse, cos_sim)
            print(f"   MSE: {mse:.8f}, 余弦相似度: {cos_sim:.6f}")

    print("\n📊 量化粒度对比总结:")
    for granularity_name, _ in granularity_configs:
        mse, cos_sim = results[granularity_name]
        print(f"   {granularity_name:<15} MSE: {mse:.8f}, 余弦相似度: {cos_sim:.6f}")
    print("✅ 量化粒度设置示例完成！")


def example_debug_tools():
    """附加示例: 调试工具（print_quant_ranges / print_quant_params / print_quant_config）"""
    _banner("debug", "调试工具使用")

    from quant_gru import print_quant_ranges

    gru = _make_gru()
    device = gru.weight_ih_l0.device
    gru.set_all_bitwidth(8)
    gru.calibration_method = "minmax"

    calibration_data = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE, device=device)
    _calibrate(gru, calibration_data)

    print("\n📊 量化范围 (print_quant_ranges):")
    print_quant_ranges(gru)

    gru.finalize_calibration()

    print("\n📊 量化参数 (print_quant_params):")
    print_quant_params(gru)

    print("\n📊 量化配置 (print_quant_config):")
    print_quant_config(gru)
    print("✅ 调试工具示例完成！")


# CLI 名称 -> (说明, 可调用对象)；bitwidth 变体在 run_examples 中处理
EXAMPLE_REGISTRY: dict[str, tuple[str, object]] = {
    "basic": ("1 基本浮点使用", example_basic_usage),
    "json": ("2 JSON 配置 + PTQ", example_quantization_with_json),
    "manual": ("3 手动统一位宽", example_quantization_manual),
    "compare": ("4 浮点 vs 量化精度对比", example_compare_precision),
    "training": ("5 QAT 训练", example_training),
    "calibration": ("6 校准方法对比", example_calibration_method),
    "bidirectional": ("7 双向 GRU", example_bidirectional),
    "onnx": ("8 ONNX 单节点导出", example_onnx_export),
    "quant_params": ("9 量化参数导入导出", example_quant_params_export_import),
    "adjust": ("10 单算子配置调整", example_adjust_quant_config),
    "granularity": ("11 per-tensor/gate/channel 权重", example_weight_bias_granularity),
    "debug": ("附加 调试工具", example_debug_tools),
}

BITWIDTH_EXAMPLES = frozenset({"manual", "compare", "training"})


def list_examples() -> None:
    print("可运行示例 (-e / --example):")
    for name, (desc, _) in EXAMPLE_REGISTRY.items():
        suffix = "  [支持 --bitwidth 8|16]" if name in BITWIDTH_EXAMPLES else ""
        print(f"  {name:<14} {desc}{suffix}")
    print("  all            运行全部（含 8bit/16bit 变体）")


def run_examples(selected: str, *, bitwidth: int = 8) -> None:
    if selected == "all":
        example_basic_usage()
        example_quantization_with_json()
        for bw in (8, 16):
            example_quantization_manual(bitwidth=bw)
            example_compare_precision(bitwidth=bw)
            example_training(bitwidth=bw)
        example_calibration_method()
        example_bidirectional()
        example_onnx_export()
        example_quant_params_export_import()
        example_adjust_quant_config()
        example_weight_bias_granularity()
        example_debug_tools()
        return

    if selected not in EXAMPLE_REGISTRY:
        raise ValueError(f"未知示例: {selected}")

    _, fn = EXAMPLE_REGISTRY[selected]
    if selected in BITWIDTH_EXAMPLES:
        fn(bitwidth=bitwidth)
    else:
        fn()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantGRU 使用示例")
    parser.add_argument(
        "-e",
        "--example",
        default="all",
        help="示例名称，默认 all；使用 --list 查看全部",
    )
    parser.add_argument(
        "--bitwidth",
        type=int,
        default=8,
        choices=[8, 16],
        help="用于 manual / compare / training（默认 8）",
    )
    parser.add_argument("--list", action="store_true", help="列出可运行示例")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        list_examples()
        return

    print("=" * 60)
    print("  QuantGRU 量化库使用示例")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ 错误: 需要 CUDA 支持")
        return

    try:
        run_examples(args.example, bitwidth=args.bitwidth)
        print("\n" + "=" * 60)
        print("  示例运行完成！")
        print("=" * 60)
    except Exception as exc:
        print(f"\n❌ 错误: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
