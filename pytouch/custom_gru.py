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


class GRUFunction(torch.autograd.Function):
    """
    GRU 的自定义 autograd Function，支持反向传播

    支持量化和非量化两种模式，反向传播统一使用 float32 权重调用 haste_gru_backward
    """

    @staticmethod
    def forward(ctx, input, W, R, bx, br, h0, is_training,
                use_quantization=False, quant_type='int8', quant_params=None):
        """
        前向传播

        Args:
            ctx: 上下文对象，用于保存中间结果
            input: 输入序列 [time_steps, batch_size, input_size]
            W: 输入权重 [input_size, hidden_size * 3] (float32，用于反向传播)
            R: 循环权重 [hidden_size, hidden_size * 3] (float32，用于反向传播)
            bx: 输入偏置 [hidden_size * 3] (float32，用于反向传播)
            br: 循环偏置 [hidden_size * 3] (float32，用于反向传播)
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
        hidden_size = R.shape[0]

        # 保存上下文信息
        ctx.time_steps = time_steps
        ctx.batch_size = batch_size
        ctx.input_size = input_size
        ctx.hidden_size = hidden_size

        # 准备量化参数（如果使用量化）
        if use_quantization:
            if quant_params is None:
                raise RuntimeError("quant_params is required when use_quantization=True")
            use_int16 = (quant_type == 'int16')
        else:
            # 非量化模式也需要 quant_params，创建一个空的
            use_int16 = False
            quant_params = gru_ops.GRUQuantitativeParameters()

        # 准备 h0 参数（转换为正确的格式或空张量）
        h0_tensor = h0 if h0 is not None else torch.empty(0, device=input.device)

        # 调用 forward_interface 统一接口
        output_full, v = gru_ops.forward_interface(
            is_training=is_training,
            is_quant=use_quantization,
            use_int16=use_int16,
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

        # 提取时间步输出（去掉初始状态）
        output = output_full[1:]  # [time_steps, batch_size, hidden_size]

        # 获取最终隐藏状态
        h_n = output_full[-1:]  # [1, batch_size, hidden_size]

        # 保存中间结果用于反向传播
        # 保存 float32 权重（无论是否量化，反向传播都使用 float32 权重）
        # 保存 output_full 和 v（量化和非量化都返回反量化后的 float32 值）
        ctx.save_for_backward(W, R, bx, br, input, output_full, v)

        return output, h_n

    @staticmethod
    def backward(ctx, grad_output, grad_h_n):
        """
        反向传播

        Args:
            ctx: 上下文对象，包含前向传播保存的中间结果
            grad_output: 输出序列的梯度 [time_steps, batch_size, hidden_size]
            grad_h_n: 最终隐藏状态的梯度 [1, batch_size, hidden_size]

        Returns:
            grad_input: 输入序列的梯度
            grad_W: 输入权重的梯度
            grad_R: 循环权重的梯度
            grad_bx: 输入偏置的梯度
            grad_br: 循环偏置的梯度
            grad_h0: 初始隐藏状态的梯度（None）
            None: is_training 的梯度（None）
        """
        # 恢复保存的中间结果
        W, R, bx, br, input, h, v = ctx.saved_tensors
        time_steps = ctx.time_steps
        batch_size = ctx.batch_size
        input_size = ctx.input_size
        hidden_size = ctx.hidden_size

        # 构建 dh_new: [time_steps + 1, batch_size, hidden_size]
        # dh_new[0] = grad_h_n (初始状态的梯度)
        # dh_new[1:] = grad_output (时间步输出的梯度)
        dh_new = torch.zeros(
            (time_steps + 1, batch_size, hidden_size),
            device=grad_output.device,
            dtype=grad_output.dtype
        )

        # 设置时间步输出的梯度
        dh_new[1:] = grad_output

        # 设置最终隐藏状态的梯度（如果有）
        if grad_h_n is not None and grad_h_n.numel() > 0:
            dh_new[-1] = dh_new[-1] + grad_h_n[0]

        # 调用 C++ 反向传播
        dx, dW, dR, dbx, dbr, dh = gru_ops.haste_gru_backward(
            time_steps=time_steps,
            batch_size=batch_size,
            input_size=input_size,
            hidden_size=hidden_size,
            W=W,
            R=R,
            bx=bx,
            br=br,
            x=input,
            dh_new=dh_new,
            h=h,
            v=v
        )

        # 返回梯度
        # 注意：h0 和 is_training 不需要梯度
        return dx, dW, dR, dbx, dbr, None, None


class CustomGRU(nn.GRU):
    """
    继承自 PyTorch nn.GRU 的自定义类，支持量化前向传播

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
        calibration_data: 用于校准量化参数的输入数据，形状为 [seq_len, batch, input_size]
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
        # 目前仅支持单层、单向、无 dropout 的情况
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

        self.use_quantization = use_quantization
        self.quant_type = quant_type.lower()
        if self.quant_type not in ['int8', 'int16']:
            raise ValueError(f"quant_type must be 'int8' or 'int16', got {self.quant_type}")

        # 初始化 cublas handle
        gru_ops.init_gru_cublas()

        # 量化相关参数
        self.quant_params = None
        # 注意：不再存储固定的量化权重，而是在每次前向传播时实时量化
        # 这样可以支持训练时权重的更新

        # 性能优化：确保权重在 CUDA 上且为 float32
        # 这样可以减少前向传播时的设备检查和类型转换
        self._ensure_weights_on_cuda_float32()

        # 如果使用量化，需要校准量化参数（不量化权重）
        if self.use_quantization:
            if calibration_data is None:
                raise ValueError(
                    "calibration_data is required when use_quantization=True. "
                    "Please provide sample input data for calibration."
                )
            self._initialize_quantization(calibration_data)

    def _reorder_weights_pytorch_to_haste(self, w):
        """
        将 PyTorch GRU 权重格式 (r, z, n) 转换为 Haste GRU 权重格式 (z, r, n)

        数学原理：
        - PyTorch GRU 使用门控顺序 (r, z, n)：reset gate, update gate, new gate
        - Haste GRU 使用门控顺序 (z, r, n)：update gate, reset gate, new gate
        - 转换方法：将第一维（3*hidden_size）分成三块，然后重新排列顺序

        转换等价性说明：
        - 方法1（Haste）：先转置 [3*hidden, input] -> [input, 3*hidden]，再在最后一维重排序
        - 方法2（本实现）：先在第一维重排序，再转置 [3*hidden, input] -> [input, 3*hidden]
        - 两种方法数学上等价，因为转置和重排序是独立的操作

        Args:
            w: 权重张量，第一维是 3*hidden_size，顺序为 r, z, n
               - 对于权重矩阵：形状为 [3*hidden, input] 或 [3*hidden, hidden]
               - 对于偏置向量：形状为 [3*hidden]

        Returns:
            重排序后的权重张量，顺序为 z, r, n
               - 对于权重矩阵：形状保持不变 [3*hidden, input] 或 [3*hidden, hidden]
               - 对于偏置向量：形状保持不变 [3*hidden]
        """
        # 重排序在 3*hidden 维度上进行（第一维）
        # chunk 将第一维分成三块：r, z, n（每块大小为 hidden_size）
        r, z, n = torch.chunk(w, 3, dim=0)
        # 重新组合为 z, r, n 的顺序
        return torch.cat([z, r, n], dim=0)

    def _quantize_weights(self, W, R, bx, br, device):
        """
        实时量化权重和偏置

        在每次前向传播时调用，将当前的浮点权重量化为整数权重。
        这样可以支持训练时权重的更新。

        Args:
            W: 输入权重，形状 [input_size, 3*hidden_size]
            R: 循环权重，形状 [hidden_size, 3*hidden_size]
            bx: 输入偏置，形状 [3*hidden_size]
            br: 循环偏置，形状 [3*hidden_size]
            device: 设备

        Returns:
            W_quant, R_quant, bx_quant, br_quant: 量化后的权重和偏置
        """
        input_size, hidden_size = W.shape[0], R.shape[0]

        if self.quant_type == 'int8':
            # 创建量化权重张量
            W_quant = torch.empty(
                (input_size, 3 * hidden_size),
                dtype=torch.int8,
                device=device
            )
            R_quant = torch.empty(
                (hidden_size, 3 * hidden_size),
                dtype=torch.int8,
                device=device
            )
            bx_quant = torch.empty(
                3 * hidden_size,
                dtype=torch.int32,
                device=device
            )
            br_quant = torch.empty(
                3 * hidden_size,
                dtype=torch.int32,
                device=device
            )

            gru_ops.quantitative_weight_int8(
                input_size=input_size,
                hidden_size=hidden_size,
                W=W,
                R=R,
                bx=bx,
                br=br,
                quant_params=self.quant_params,
                W_quant=W_quant,
                R_quant=R_quant,
                bx_quant=bx_quant,
                br_quant=br_quant
            )
        else:  # int16
            # 创建量化权重张量
            W_quant = torch.empty(
                (input_size, 3 * hidden_size),
                dtype=torch.int16,
                device=device
            )
            R_quant = torch.empty(
                (hidden_size, 3 * hidden_size),
                dtype=torch.int16,
                device=device
            )
            bx_quant = torch.empty(
                3 * hidden_size,
                dtype=torch.int32,
                device=device
            )
            br_quant = torch.empty(
                3 * hidden_size,
                dtype=torch.int32,
                device=device
            )

            gru_ops.quantitative_weight_int16(
                input_size=input_size,
                hidden_size=hidden_size,
                W=W,
                R=R,
                bx=bx,
                br=br,
                quant_params=self.quant_params,
                W_quant=W_quant,
                R_quant=R_quant,
                bx_quant=bx_quant,
                br_quant=br_quant
            )

        return W_quant, R_quant, bx_quant, br_quant

    def _ensure_weights_on_cuda_float32(self):
        """
        确保所有权重和偏置在 CUDA 上且为 float32 类型

        性能优化：在初始化时统一处理，避免在前向传播时重复检查和转换
        """
        # 确保权重在 CUDA 上且为 float32
        if not self.weight_ih_l0.is_cuda:
            self.weight_ih_l0.data = self.weight_ih_l0.data.cuda()
        if self.weight_ih_l0.dtype != torch.float32:
            self.weight_ih_l0.data = self.weight_ih_l0.data.float()

        if not self.weight_hh_l0.is_cuda:
            self.weight_hh_l0.data = self.weight_hh_l0.data.cuda()
        if self.weight_hh_l0.dtype != torch.float32:
            self.weight_hh_l0.data = self.weight_hh_l0.data.float()

        # 确保偏置在 CUDA 上且为 float32（如果使用偏置）
        if self.bias:
            if not self.bias_ih_l0.is_cuda:
                self.bias_ih_l0.data = self.bias_ih_l0.data.cuda()
            if self.bias_ih_l0.dtype != torch.float32:
                self.bias_ih_l0.data = self.bias_ih_l0.data.float()

            if not self.bias_hh_l0.is_cuda:
                self.bias_hh_l0.data = self.bias_hh_l0.data.cuda()
            if self.bias_hh_l0.dtype != torch.float32:
                self.bias_hh_l0.data = self.bias_hh_l0.data.float()

    def _get_haste_weights(self, device: Optional[torch.device] = None):
        """
        获取 Haste 格式的权重和偏置

        这个方法统一处理权重格式转换，避免在多个地方重复代码。
        转换过程：
        1. 获取 PyTorch 格式的权重和偏置
        2. 重排序：从 (r, z, n) 转为 (z, r, n)
        3. 转置权重矩阵：从 [3*hidden, input] 转为 [input, 3*hidden]

        Args:
            device: 目标设备（可选），如果为 None，使用权重的当前设备

        Returns:
            W: 输入权重，形状 [input_size, 3*hidden_size]，顺序 (z, r, n)
            R: 循环权重，形状 [hidden_size, 3*hidden_size]，顺序 (z, r, n)
            bx: 输入偏置，形状 [3*hidden_size]，顺序 (z, r, n)
            br: 循环偏置，形状 [3*hidden_size]，顺序 (z, r, n)
        """
        # 获取权重（已经在 CUDA 上且为 float32，由 _ensure_weights_on_cuda_float32 保证）
        weight_ih = self.weight_ih_l0  # [3*hidden, input]
        weight_hh = self.weight_hh_l0  # [3*hidden, hidden]

        # 将 PyTorch 格式转换为 Haste 格式：
        # 1. 重排序：在 3*hidden 维度上从 (r, z, n) 转为 (z, r, n)
        # 2. 转置权重：从 [3*hidden, input] 转为 [input, 3*hidden]
        W = self._reorder_weights_pytorch_to_haste(weight_ih).t().contiguous()  # [input, 3*hidden], 顺序 (z, r, n)
        R = self._reorder_weights_pytorch_to_haste(weight_hh).t().contiguous()  # [hidden, 3*hidden], 顺序 (z, r, n)

        # 处理偏置
        if self.bias:
            bias_ih = self.bias_ih_l0  # [3*hidden]
            bias_hh = self.bias_hh_l0  # [3*hidden]
            # 将 PyTorch 格式 (r, z, n) 转换为 Haste 格式 (z, r, n)
            bx = self._reorder_weights_pytorch_to_haste(bias_ih).contiguous()  # [3*hidden], 顺序 (z, r, n)
            br = self._reorder_weights_pytorch_to_haste(bias_hh).contiguous()  # [3*hidden], 顺序 (z, r, n)
        else:
            # 如果没有偏置，创建零偏置
            target_device = device if device is not None else weight_ih.device
            bx = torch.zeros(3 * self.hidden_size, device=target_device, dtype=torch.float32)
            br = torch.zeros(3 * self.hidden_size, device=target_device, dtype=torch.float32)

        return W, R, bx, br

    def _initialize_quantization(self, calibration_data: torch.Tensor):
        """
        初始化量化参数（不量化权重）

        注意：只校准量化参数，不量化权重。权重将在每次前向传播时实时量化，
        这样可以支持训练时权重的更新。

        Args:
            calibration_data: 用于校准的输入数据，形状为 [seq_len, batch, input_size] 或 [batch, seq_len, input_size]
        """
        # 确保数据在 CUDA 上
        if not calibration_data.is_cuda:
            calibration_data = calibration_data.cuda()

        # 处理 batch_first
        if self.batch_first:
            # [batch, seq_len, input_size] -> [seq_len, batch, input_size]
            calibration_data = calibration_data.transpose(0, 1).contiguous()

        time_steps, batch_size, input_size = calibration_data.shape
        hidden_size = self.hidden_size

        # 使用统一的权重转换方法（权重已经在 CUDA 上且为 float32）
        W, R, bx, br = self._get_haste_weights(device=calibration_data.device)

        # 只校准量化参数，不量化权重
        # 权重将在每次前向传播时实时量化，以支持训练时权重的更新
        use_int16 = (self.quant_type == 'int16')
        self.quant_params = gru_ops.calibrate_gru_scales(
            use_int16=use_int16,
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
        """
        # 处理 batch_first
        if self.batch_first:
            # [batch, seq_len, input_size] -> [seq_len, batch, input_size]
            input = input.transpose(0, 1)

        seq_len, batch_size, input_size = input.shape
        hidden_size = self.hidden_size

        # 处理初始隐藏状态
        h0 = None  # C++ 接口需要的格式：[batch, hidden_size]
        if hx is not None:
            expected_shape = (self.num_layers, batch_size, hidden_size)
            if hx.shape != expected_shape:
                raise ValueError(
                    f"Expected hx shape {expected_shape} (num_layers={self.num_layers}, "
                    f"batch_size={batch_size}, hidden_size={hidden_size}), "
                    f"got {hx.shape}"
                )
            # 确保初始状态在正确的设备上
            if not hx.is_cuda:
                hx = hx.cuda()
            if hx.dtype != torch.float32:
                hx = hx.float()
            # 提取第一层的初始状态（因为我们只支持单层）
            # hx 形状: [num_layers, batch, hidden_size] -> h0: [batch, hidden_size]
            h0 = hx[0]  # [batch, hidden_size]

        # 确保输入在 CUDA 上
        if not input.is_cuda:
            input = input.cuda()

        # 使用统一的权重转换方法（权重已经在 CUDA 上且为 float32）
        W, R, bx, br = self._get_haste_weights(device=input.device)

        # 统一使用 GRUFunction 进行前向传播（支持量化和非量化模式的反向传播）
        # 注意：GRUFunction 需要输入为 [time_steps, batch_size, input_size]
        # 而 input 已经是这个格式了（之前已经处理了 batch_first）
        output, h_n_from_func = GRUFunction.apply(
            input,              # [time_steps, batch_size, input_size]
            W,                  # [input_size, hidden_size * 3] (float32)
            R,                  # [hidden_size, hidden_size * 3] (float32)
            bx,                 # [hidden_size * 3] (float32)
            br,                 # [hidden_size * 3] (float32)
            h0,                 # [batch_size, hidden_size] 或 None
            self.training,      # 是否处于训练模式
            self.use_quantization,  # 是否使用量化
            self.quant_type,    # 量化类型
            self.quant_params   # 量化参数
        )
        # output 形状: [time_steps, batch_size, hidden_size]
        # h_n_from_func 形状: [1, batch_size, hidden_size]

        # 处理 batch_first
        if self.batch_first:
            # [seq_len, batch, hidden_size] -> [batch, seq_len, hidden_size]
            output = output.transpose(0, 1)

        # 获取最终隐藏状态
        # GRUFunction 统一返回最终隐藏状态
        # h_n_from_func 形状: [1, batch_size, hidden_size]
        h_n = h_n_from_func

        # 确保 h_n 的形状为 [num_layers, batch, hidden_size]
        # 由于我们只支持单层，所以 h_n 已经是 [1, batch, hidden_size]
        # 但为了保持接口一致性，确保形状正确
        assert h_n.shape[0] == 1, f"Expected h_n shape [1, batch, hidden_size], got {h_n.shape}"

        return output, h_n
