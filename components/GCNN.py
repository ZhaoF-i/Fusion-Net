import torch
from torch import nn


class G_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super(G_Conv2d, self).__init__()
        self.conv_l = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.conv_s = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding)

    def forward(self, input):
        return self.conv_l(input) * torch.sigmoid(self.conv_s(input))


class G_ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), output_padding=(0, 0), padding=(0, 0)):
        super(G_ConvTranspose2d, self).__init__()
        self.conv_l = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
                                         stride=stride, output_padding=output_padding, padding=padding)
        self.conv_s = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
                                         stride=stride, output_padding=output_padding, padding=padding)

    def forward(self, input):
        return self.conv_l(input) * torch.sigmoid(self.conv_s(input))




class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out
