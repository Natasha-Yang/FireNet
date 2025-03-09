import torch 
import torch.nn as nn
import torch.nn.functional as F
from .convlstm import ConvLSTMCell, ConvLSTM, ConvLSTM_Seg

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2,
                              out_channels=1,
                              kernel_size=7,
                              stride=1,
                              padding=3,
                              dilation=1,
                              bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels,
                      out_features=self.channels//self.r,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r,
                      out_features=self.channels,
                      bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

class CBAM_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding, bias):
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_dim + hidden_dim,
                      out_channels=2 * hidden_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(2 * hidden_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * hidden_dim,
                      out_channels=4 * hidden_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(4 * hidden_dim),
            nn.ReLU(),
            CBAM(4 * hidden_dim, 1)
        )
    def forward(self, x):
        return self.conv_block(x)


class ConvLSTM_CBAM_Cell(ConvLSTMCell):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM CBAM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = CBAM_Encoder(input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 bias=self.bias
        )
class ConvLSTM_CBAM(ConvLSTM):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTM_CBAM_Cell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

class ConvLSTM_CBAM_Seg(ConvLSTM_Seg):
    def __init__(
        self, num_classes, input_size, input_dim, hidden_dim, kernel_size, pad_value=0
    ):
        super(ConvLSTM_Seg, self).__init__()
        self.convlstm_encoder = ConvLSTM_CBAM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
        )
        self.classification_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=1,
        )
        self.pad_value = pad_value

