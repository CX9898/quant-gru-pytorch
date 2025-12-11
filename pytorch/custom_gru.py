"""
自定义 GRU 类，继承自 PyTorch 的 nn.GRU
支持量化和非量化两种前向传播模式
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import gru_interface_binding as gru_ops
except ImportError:
    raise ImportError(
        "gru_interface_binding module not found. "
        "Please compile the C++ extension first using setup.py"
    )


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

    支持两种校准方式：
    1. 立即校准：在构造函数中提供 calibration_data，立即进行校准（向后兼容）
    2. 延迟校准：构造函数中 calibration_data=None，后续调用 calibrate() 方法（推荐）

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: GRU 层数（目前仅支持单层）
        bias: 是否使用偏置
        batch_first: 如果为 True，输入形状为 [batch, seq, feature]
        dropout: 层间 dropout 概率
        bidirectional: 是否双向（目前不支持）
        use_quantization: 是否使用量化
        quant_type: 量化类型，'int8' 或 'int16'
        calibration_data: 用于校准量化参数的输入数据（可选）
            - 如果提供：立即进行校准（向后兼容）
            - 如果为 None：延迟校准，需要后续调用 calibrate() 方法

    Examples:
        # 方式1：立即校准（向后兼容）
        gru = CustomGRU(..., use_quantization=True, calibration_data=data)

        # 方式2：延迟校准（推荐，可以在设置权重后校准）
        gru = CustomGRU(..., use_quantization=True, calibration_data=None)
        # ... 设置权重 ...
        gru.calibrate(calibration_data)
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
        calibration_data: Optional[torch.Tensor] = None
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

        # 量化参数初始化
        if self.use_quantization:
            if calibration_data is not None:
                # 立即校准（向后兼容）
                self._initialize_quantization(calibration_data)
            else:
                # 延迟校准
                self.quant_params = None
        else:
            self.quant_params = None

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

    def is_calibrated(self) -> bool:
        """检查量化参数是否已校准"""
        return self.quant_params is not None

    def calibrate(self, calibration_data: torch.Tensor):
        """
        显式校准量化参数（公共方法）

        权重将在每次前向传播时实时量化，以支持训练时权重的更新。

        Args:
            calibration_data: 用于校准的输入数据，形状为 [seq_len, batch, input_size] 或 [batch, seq_len, input_size]

        Raises:
            RuntimeError: 如果未启用量化
        """
        if not self.use_quantization:
            raise RuntimeError("Cannot calibrate: quantization is not enabled. Set use_quantization=True first.")
        self._initialize_quantization(calibration_data)

    def _initialize_quantization(self, calibration_data: torch.Tensor):
        """
        初始化量化参数（内部方法）

        权重将在每次前向传播时实时量化，以支持训练时权重的更新。

        Args:
            calibration_data: 用于校准的输入数据，形状为 [seq_len, batch, input_size] 或 [batch, seq_len, input_size]
        """
        # 确保校准数据在 CUDA 上
        device = calibration_data.device if calibration_data.is_cuda else torch.device('cuda')
        if not calibration_data.is_cuda:
            calibration_data = calibration_data.to(device)

        # 确保模型参数在 GPU 上（手动移动，避免触发 flatten_parameters）
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

        # 校准量化参数（使用默认的 INT8 位宽配置）
        self.quant_params = gru_ops.calibrate_gru_scales(
            time_steps=time_steps,
            batch_size=batch_size,
            input_size=input_size,
            hidden_size=hidden_size,
            W=W,
            R=R,
            bx=bx,
            br=br,
            x=calibration_data
        )
        torch.cuda.synchronize()

        # 初始化量化 LUT 表
        gru_ops.initialize_quantization_lut(quant_params=self.quant_params)
        torch.cuda.synchronize()

        # 确保权重连续性并重置 flatten_parameters 状态
        self.weight_ih_l0.data = self.weight_ih_l0.data.contiguous()
        self.weight_hh_l0.data = self.weight_hh_l0.data.contiguous()
        if self.bias:
            self.bias_ih_l0.data = self.bias_ih_l0.data.contiguous()
            self.bias_hh_l0.data = self.bias_hh_l0.data.contiguous()

        # 重置 flatten_parameters 的内部状态，避免后续 .to(device) 时出现问题
        if hasattr(self, '_flat_weights'):
            self._flat_weights = None

        # 标记量化已初始化，用于后续的 _apply 方法
        self._quantization_initialized = True

    def _apply(self, fn):
        """
        重写 _apply 方法，在量化初始化后正确处理设备迁移

        主要用于向后兼容立即校准方式（方式1）：
        - 如果使用延迟校准（方式2，推荐），此方法不会被触发
        - 如果使用立即校准（方式1），在量化初始化后调用 .to(device) 时会触发此方法
        - 手动应用函数，避免触发 flatten_parameters()，防止 CUDA 状态冲突

        注意：延迟校准方式不需要此方法，因为 .to(device) 在量化初始化之前调用
        """
        if hasattr(self, '_quantization_initialized') and self._quantization_initialized:
            # 量化已初始化：手动应用函数，避免触发 flatten_parameters()
            # 这主要用于向后兼容立即校准方式
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
        else:
            # 量化未初始化：使用父类的默认行为（正常触发 flatten_parameters()）
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
        # 检查量化是否已校准
        if self.use_quantization and not self.is_calibrated():
            raise RuntimeError(
                "Quantization is enabled but not calibrated. "
                "Please call calibrate(calibration_data) before forward pass, "
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
