import torch.nn as nn
from torch.nn.utils import weight_norm

#用于裁剪输入张量的时间维度，去除多余的 padding 部分。
class Crop(nn.Module):

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        #裁剪张量以去除额外的填充
        return x[:, :, :-self.crop_size].contiguous()

#实现了一个膨胀卷积层，由两个膨胀卷积块组成。每个膨胀卷积块包含一个带有权重归一化的卷积层、裁剪模块、ReLU激活函数和 Dropout 正则化。此外，还包括了一个用于快捷连接的卷积层
class TemporalCasualLayer(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation
        }

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2)
        #快捷连接
        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # 应用因果卷积和快捷连接
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)

#通过堆叠多个 TemporalCasualLayer 组成了一个完整的 TCN 网络。每个 TemporalCasualLayer 具有不同的膨胀系数，并根据输入和输出通道的数量进行设置。
class TemporalConvolutionNetwork(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            # tcl = self.relu(tcl)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

#封装了 TemporalConvolutionNetwork，并添加了一个线性层用于最终的预测。在前向传播中，先经过 TCN 网络，然后将输出的最后一个时间步传入线性层，并通过 ReLU 激活函数进行非线性变换。
class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 应用TCN和线性层，然后使用ReLU激活函数
        y = self.tcn(x)  # [N,C_out,L_out=L_in]
        return self.relu(self.linear(y[:, :, -1]))