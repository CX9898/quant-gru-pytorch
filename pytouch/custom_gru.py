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

    所有格式转换操作统一在此类中处理：
    - forward: 接收 PyTorch 格式权重，转换为 Haste 格式后调用 C++ 接口
    - backward: 接收 Haste 格式梯度，转换为 PyTorch 格式后返回
    """

    @staticmethod
    def _reorder_weights_pytorch_to_haste(w):
        """
        将 PyTorch GRU 权重格式 (r, z, n) 转换为 Haste GRU 权重格式 (z, r, n)

        Args:
            w: 权重张量，第一维是 3*hidden_size，顺序为 r, z, n
               - 对于权重矩阵：形状为 [3*hidden, input] 或 [3*hidden, hidden]
               - 对于偏置向量：形状为 [3*hidden]

        Returns:
            重排序后的权重张量，顺序为 z, r, n
               - 对于权重矩阵：形状保持不变 [3*hidden, input] 或 [3*hidden, hidden]
               - 对于偏置向量：形状保持不变 [3*hidden]
        """
        r, z, n = torch.chunk(w, 3, dim=0)
        return torch.cat([z, r, n], dim=0)

    @staticmethod
    def _reorder_weights_haste_to_pytorch(w):
        """
        将 Haste GRU 权重格式 (z, r, n) 转换回 PyTorch GRU 权重格式 (r, z, n)

        Args:
            w: 权重张量，第一维是 3*hidden_size，顺序为 z, r, n
               - 对于权重矩阵：形状为 [3*hidden, input] 或 [3*hidden, hidden]
               - 对于偏置向量：形状为 [3*hidden]

        Returns:
            重排序后的权重张量，顺序为 r, z, n
               - 对于权重矩阵：形状保持不变 [3*hidden, input] 或 [3*hidden, hidden]
               - 对于偏置向量：形状保持不变 [3*hidden]
        """
        z, r, n = torch.chunk(w, 3, dim=0)
        return torch.cat([r, z, n], dim=0)

    @staticmethod
    def forward(ctx, input, weight_ih, weight_hh, bias_ih, bias_hh, h0, is_training,
                use_quantization=False, quant_type='int8', quant_params=None):
        """
        前向传播

        Args:
            ctx: 上下文对象，用于保存中间结果
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

        # 格式转换：从 PyTorch 格式转换为 Haste 格式
        # PyTorch: [3*hidden, input] (r, z, n) -> Haste: [input, 3*hidden] (z, r, n)
        W = GRUFunction._reorder_weights_pytorch_to_haste(weight_ih).t().contiguous()
        R = GRUFunction._reorder_weights_pytorch_to_haste(weight_hh).t().contiguous()

        # 处理偏置
        if bias_ih is not None and bias_hh is not None:
            bx = GRUFunction._reorder_weights_pytorch_to_haste(bias_ih).contiguous()
            br = GRUFunction._reorder_weights_pytorch_to_haste(bias_hh).contiguous()
        else:
            # 如果没有偏置，创建零偏置
            device = input.device
            bx = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)
            br = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)

        # 准备量化参数（如果使用量化）
        if use_quantization:
            if quant_params is None:
                raise RuntimeError("quant_params is required when use_quantization=True")
            use_int16 = (quant_type == 'int16')
        else:
            # 非量化模式也需要 quant_params，创建一个空的
            use_int16 = False
            quant_params = gru_ops.GRUQuantitativeParameters()

        # 保存偏置是否为 None 的信息（用于 backward 中判断是否需要返回梯度）
        ctx.bias_ih_is_none = (bias_ih is None)
        ctx.bias_hh_is_none = (bias_hh is None)

        # 准备 h0 参数（转换为正确的格式或空张量）
        # C++ 接口期望 h0 的形状是 [batch_size, hidden_size] 或空张量
        # 当 h0 为 None 时，创建空张量，形状为 [0] 以匹配接口期望
        # 保存 h0 是否为 None 的信息，以便在 backward 中正确处理梯度
        ctx.h0_is_none = (h0 is None)
        if h0 is not None:
            h0_tensor = h0
        else:
            # 创建空张量，C++ 接口会检查 h0.numel() > 0 来判断是否为空
            h0_tensor = torch.empty(0, device=input.device, dtype=torch.float32)

        # 调用 forward_interface 统一接口
        # 注意：此时 W, R, bx, br 已经是 Haste 格式 (z, r, n)
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

        # 将 output_full 分成两个返回值，符合 PyTorch 标准 GRU 接口
        #
        # 原因：
        # 1. PyTorch nn.GRU 标准接口返回 (output, h_n) 两个值
        # 2. 语义不同：
        #    - output: 时间步输出序列，不包含初始状态（初始状态是输入，不是输出）
        #    - h_n: 最终隐藏状态，单独返回方便后续使用（多层网络、序列拼接等）
        # 3. C++ 接口返回完整序列 [time_steps + 1, ...]（包含初始状态），
        #    但 PyTorch 接口期望 output 不包含初始状态 [time_steps, ...]
        #
        # output_full 结构：
        # - output_full[0]: 初始隐藏状态 h0（不是输入序列产生的输出）
        # - output_full[1:time_steps+1]: 时间步 1 到 time_steps 的隐藏状态（输入序列产生的输出）
        output = output_full[1:]  # [time_steps, batch_size, hidden_size] - 时间步输出序列
        h_n = output_full[-1:]   # [1, batch_size, hidden_size] - 最终隐藏状态

        # 保存中间结果用于反向传播
        # 只保存 backward 中实际需要的张量：
        # - W, R, bx, br: Haste 格式的权重和偏置（用于调用 C++ 反向传播接口）
        # - input: 输入序列（用于反向传播计算）
        # - output_full: 完整输出序列（包含初始状态，用于反向传播计算）
        # - v: 中间激活值（用于反向传播计算）
        # 注意：不保存 PyTorch 格式的权重，因为 backward 中不需要它们
        # 注意：不保存 bias_ih/bias_hh 张量，只保存布尔标志（ctx.bias_ih_is_none/bias_hh_is_none）
        ctx.save_for_backward(W, R, bx, br, input, output_full, v)

        return output, h_n

    @staticmethod
    def backward(ctx, grad_output, grad_h_n):
        """
        反向传播

        Args:
            ctx: 上下文对象，包含前向传播保存的中间结果
            grad_output: 输出序列的梯度 [time_steps, batch_size, hidden_size]
                        对应 forward 返回的 output = output_full[1:] 的梯度
                        即时间步 1 到 time_steps 的隐藏状态梯度
            grad_h_n: 最终隐藏状态的梯度 [1, batch_size, hidden_size]
                      对应 forward 返回的 h_n = output_full[-1:] 的梯度
                      注意：h_n[0] 和 output[-1] 指向同一个隐藏状态（output_full[time_steps]）
                      因此 grad_output[-1] 和 grad_h_n[0] 需要相加

        Returns:
            grad_input: 输入序列的梯度
            grad_W: 输入权重的梯度
            grad_R: 循环权重的梯度
            grad_bx: 输入偏置的梯度
            grad_br: 循环偏置的梯度
            grad_h0: 初始隐藏状态的梯度 [batch_size, hidden_size] 或 None
            None: is_training 的梯度（None）
            None: use_quantization 的梯度（None）
            None: quant_type 的梯度（None）
            None: quant_params 的梯度（None）
        """
        # 恢复保存的中间结果
        # saved_tensors = [W, R, bx, br, input, output_full, v]
        W, R, bx, br, input, h, v = ctx.saved_tensors
        time_steps = ctx.time_steps
        batch_size = ctx.batch_size
        input_size = ctx.input_size
        hidden_size = ctx.hidden_size

        # 构建 dh_new: [time_steps + 1, batch_size, hidden_size]
        # dh_new 包含所有时间步隐藏状态的梯度，用于 C++ 反向传播接口
        #
        # 注意：C++ 代码从后往前遍历时间步（i = time_steps-1 到 0），
        # 对于每个时间步 i，使用指针偏移 dh_new + (i + 1) * NH（NH = batch_size * hidden_size）
        #
        # 指针偏移说明（NH 是一个时间步的大小，以元素为单位）：
        # - dh_new + 0 * NH = dh_new + 0：指向 dh_new[0]（时间步 0，初始状态）
        # - dh_new + 1 * NH = dh_new + NH：指向 dh_new[NH]（时间步 1）
        # - dh_new + 2 * NH = dh_new + 2*NH：指向 dh_new[2*NH]（时间步 2）
        #
        # 在循环中：
        # - 当 i = time_steps-1 时，使用 dh_new + (time_steps-1+1) * NH = dh_new + time_steps * NH（最后一个时间步）
        # - 当 i = 0 时，使用 dh_new + (0+1) * NH = dh_new + 1 * NH = dh_new + NH（时间步 1）
        # - dh_new[0]（时间步 0）：从未被访问，保持为 0（正确）
        #
        # 初始状态 h0 的梯度通过 C++ 返回的 dh 计算（在最后一次迭代 i=0 后），
        # 而不是通过 dh_new[0]。dh 包含了从时间步 1 传播回来的梯度。
        dh_new = torch.zeros(
            (time_steps + 1, batch_size, hidden_size),
            device=grad_output.device,
            dtype=grad_output.dtype
        )

        # 设置时间步输出的梯度
        # grad_output 对应 output = output_full[1:] 的梯度
        # 即时间步 1 到 time_steps 的隐藏状态梯度
        dh_new[1:] = grad_output

        # 处理最终隐藏状态的梯度
        #
        # 核心问题：同一个隐藏状态被返回了两次
        # - output[-1] = output_full[time_steps]（最后一个时间步的隐藏状态）
        # - h_n[0] = output_full[time_steps]（同一个隐藏状态）
        #
        # 在 PyTorch 的 autograd 机制中，如果一个值被多个地方使用，
        # 反向传播时梯度会从多个路径传回，需要将这些梯度相加。
        #
        # 梯度来源：
        # - grad_output[-1]: 损失函数对 output[-1] 的梯度
        # - grad_h_n[0]: 损失函数对 h_n[0] 的梯度
        # 两者都对应 output_full[time_steps]，需要相加得到总梯度
        #
        # 执行过程：
        # 1. dh_new[1:] = grad_output 后，dh_new[-1] = grad_output[time_steps-1] = grad_output[-1]
        # 2. dh_new[-1] = dh_new[-1] + grad_h_n[0] 将两个来源的梯度相加
        #
        # 尺寸验证：
        # - dh_new[-1] 的形状是 [batch_size, hidden_size]（对应 dh_new[time_steps]）
        # - grad_h_n[0] 的形状是 [batch_size, hidden_size]
        # - 尺寸匹配，可以相加
        # - 相加后，dh_new 的形状仍然是 [time_steps + 1, batch_size, hidden_size]，没有改变
        if grad_h_n is not None and grad_h_n.numel() > 0:
            dh_new[-1] = dh_new[-1] + grad_h_n[0]

        # 调用 C++ 反向传播（使用 Haste 格式的权重）
        # 注意：W, R, bx, br 已经是 Haste 格式 (z, r, n)
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

        # 格式转换：将梯度从 Haste 格式转换为 PyTorch 格式
        # dW 和 dR 的形状是 [input_size, 3*hidden_size] 和 [hidden_size, 3*hidden_size]，顺序 (z, r, n)
        # 需要转换为 [3*hidden_size, input_size] 和 [3*hidden_size, hidden_size]，顺序 (r, z, n)
        # 步骤：1) 先转置 2) 再重排序
        dW_pytorch = GRUFunction._reorder_weights_haste_to_pytorch(dW.t()).contiguous()
        dR_pytorch = GRUFunction._reorder_weights_haste_to_pytorch(dR.t()).contiguous()

        # dbx 和 dbr 的形状是 [3*hidden_size]，顺序 (z, r, n)
        # 需要转换为 [3*hidden_size]，顺序 (r, z, n)
        dbx_pytorch = GRUFunction._reorder_weights_haste_to_pytorch(dbx).contiguous()
        dbr_pytorch = GRUFunction._reorder_weights_haste_to_pytorch(dbr).contiguous()

        # 处理偏置梯度：如果原始没有偏置，梯度应该为 None
        # 使用保存的布尔标志判断，而不是保存整个张量
        if ctx.bias_ih_is_none:
            dbx_pytorch = None
        if ctx.bias_hh_is_none:
            dbr_pytorch = None

        # 返回梯度（已转换为 PyTorch 格式）
        # 注意：backward 必须返回与 forward 输入参数数量相同的梯度值
        # forward 有 10 个参数：input, weight_ih, weight_hh, bias_ih, bias_hh, h0, is_training, use_quantization, quant_type, quant_params
        # h0 的梯度：C++ 返回的 dh 形状为 [batch_size, hidden_size]
        # 如果 forward 时 h0 是 None，则 backward 也应该返回 None
        if ctx.h0_is_none:
            grad_h0 = None
        else:
            # dh 形状为 [batch_size, hidden_size]，已经是正确格式
            grad_h0 = dh

        # 返回所有梯度（对应 forward 的 10 个参数）
        return dx, dW_pytorch, dR_pytorch, dbx_pytorch, dbr_pytorch, grad_h0, None, None, None, None


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

        # 格式转换：从 PyTorch 格式转换为 Haste 格式（用于校准）
        # 使用 GRUFunction 的静态方法进行转换
        weight_ih = self.weight_ih_l0  # [3*hidden, input]
        weight_hh = self.weight_hh_l0  # [3*hidden, hidden]
        W = GRUFunction._reorder_weights_pytorch_to_haste(weight_ih).t().contiguous()
        R = GRUFunction._reorder_weights_pytorch_to_haste(weight_hh).t().contiguous()

        if self.bias:
            bias_ih = self.bias_ih_l0
            bias_hh = self.bias_hh_l0
            bx = GRUFunction._reorder_weights_pytorch_to_haste(bias_ih).contiguous()
            br = GRUFunction._reorder_weights_pytorch_to_haste(bias_hh).contiguous()
        else:
            device = calibration_data.device
            bx = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)
            br = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)

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

        # 确保输入在 CUDA 上且为 float32
        if not input.is_cuda:
            input = input.cuda()
        if input.dtype != torch.float32:
            input = input.float()

        # 获取权重和偏置（已经在 CUDA 上且为 float32，由 _ensure_weights_on_cuda_float32 保证）
        weight_ih = self.weight_ih_l0  # [3*hidden, input] (PyTorch 格式: r, z, n)
        weight_hh = self.weight_hh_l0  # [3*hidden, hidden] (PyTorch 格式: r, z, n)
        bias_ih = self.bias_ih_l0 if self.bias else None  # [3*hidden] (PyTorch 格式: r, z, n) 或 None
        bias_hh = self.bias_hh_l0 if self.bias else None  # [3*hidden] (PyTorch 格式: r, z, n) 或 None

        # 统一使用 GRUFunction 进行前向传播（支持量化和非量化模式的反向传播）
        # GRUFunction 内部会处理格式转换：PyTorch 格式 -> Haste 格式
        # 注意：GRUFunction 需要输入为 [time_steps, batch_size, input_size]
        # 而 input 已经是这个格式了（之前已经处理了 batch_first）
        output, h_n_from_func = GRUFunction.apply(
            input,              # [time_steps, batch_size, input_size]
            weight_ih,          # [3*hidden_size, input_size] (PyTorch 格式: r, z, n)
            weight_hh,          # [3*hidden_size, hidden_size] (PyTorch 格式: r, z, n)
            bias_ih,            # [3*hidden_size] (PyTorch 格式: r, z, n) 或 None
            bias_hh,            # [3*hidden_size] (PyTorch 格式: r, z, n) 或 None
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
