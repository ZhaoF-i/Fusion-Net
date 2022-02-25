import torch.nn as nn
import torch
from torch.nn import functional as F


class DenseBlock(nn.Module):
    def __init__(self,input_size,depth=5,in_channels=64):
        super(DenseBlock, self).__init__()

        self.conv_lst = nn.ModuleList()
        for i in range(depth):
            self.conv_lst.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels * (i + 1), out_channels=in_channels,
                          kernel_size=(2, 3), dilation=(2 ** i, 1)),
                nn.LayerNorm(input_size),
                nn.PReLU(in_channels)
            ))

    def forward(self, input):
        o = input
        tmp = None
        for i in range(5):
            pad = F.pad(o, [2 ** (i+1), 0, 1, 1], mode='constant', value=0)
            tmp = self.conv_lst[i](pad)
            o = torch.cat([tmp, o], dim=1)

        return tmp

class DenseBlock_origin(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock_origin, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil-1)*(self.twidth-1)-1
            setattr(self, 'pad{}'.format(i+1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i+1),
                    nn.Conv2d(self.in_channels*(i+1), self.in_channels, kernel_size=self.kernel_size, dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i+1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i+1), nn.PReLU(self.in_channels))
    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i+1))(skip)
            out = getattr(self, 'conv{}'.format(i+1))(out)
            out = getattr(self, 'norm{}'.format(i+1))(out)
            out = getattr(self, 'prelu{}'.format(i+1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out