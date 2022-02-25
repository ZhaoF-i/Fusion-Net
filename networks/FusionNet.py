import torch.nn as nn
import torch
from thop import profile, clever_format
from components.DenseBlock import DenseBlock_origin
from components.SPConv import SPConvTranspose2d
import torch.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.in_channel = 1
        self.out_channel = 1
        self.width = 64
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.) # 左右上下
        # input conv
        self.inp_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(512)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense = DenseBlock_origin(512, 5, self.width)

        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm1 = nn.LayerNorm(256)
        self.enc_prelu1 = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock_origin(256, 5, self.width)

        self.enc_conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm2 = nn.LayerNorm(128)
        self.enc_prelu2 = nn.PReLU(self.width)
        self.enc_dense2 = DenseBlock_origin(128, 5, self.width)

        self.enc_conv3 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm3 = nn.LayerNorm(64)
        self.enc_prelu3 = nn.PReLU(self.width)
        self.enc_dense3 = DenseBlock_origin(64, 5, self.width)

        self.enc_conv4 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm4 = nn.LayerNorm(32)
        self.enc_prelu4 = nn.PReLU(self.width)
        self.enc_dense4 = DenseBlock_origin(32, 5, self.width)

        self.enc_conv5 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm5 = nn.LayerNorm(16)
        self.enc_prelu5 = nn.PReLU(self.width)
        self.enc_dense5 = DenseBlock_origin(16, 5, self.width)

        self.enc_conv6 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm6 = nn.LayerNorm(8)
        self.enc_prelu6 = nn.PReLU(self.width)
        self.enc_dense6 = DenseBlock_origin(8, 5, self.width)

    def forward(self, input):

        e1 = self.enc_dense(self.inp_prelu(self.inp_norm(self.inp_conv(input.unsqueeze(1)))))
        e2 = self.enc_dense1(self.enc_prelu1(self.enc_norm1(self.enc_conv1(self.pad1(e1)))))
        e3 = self.enc_dense2(self.enc_prelu2(self.enc_norm2(self.enc_conv1(self.pad1(e2)))))
        e4 = self.enc_dense3(self.enc_prelu3(self.enc_norm3(self.enc_conv1(self.pad1(e3)))))
        e5 = self.enc_dense4(self.enc_prelu4(self.enc_norm4(self.enc_conv1(self.pad1(e4)))))
        e6 = self.enc_dense5(self.enc_prelu5(self.enc_norm5(self.enc_conv1(self.pad1(e5)))))
        e7 = self.enc_dense6(self.enc_prelu6(self.enc_norm6(self.enc_conv1(self.pad1(e6)))))

        return e7, [e2, e3, e4, e5, e6]


class T_Decoder(nn.Module):
    def __init__(self):
        super(T_Decoder, self).__init__()

        self.width = 64

        self.dec_conv6 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm6 = nn.LayerNorm(16)
        self.dec_prelu6 = nn.PReLU(self.width)
        self.dec_dense6 = DenseBlock_origin(16, 5, self.width)

        self.dec_conv5 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm5 = nn.LayerNorm(32)
        self.dec_prelu5 = nn.PReLU(self.width)
        self.dec_dense5 = DenseBlock_origin(32, 5, self.width)

        self.dec_conv4 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm4 = nn.LayerNorm(64)
        self.dec_prelu4 = nn.PReLU(self.width)
        self.dec_dense4 = DenseBlock_origin(64, 5, self.width)

        self.dec_conv3 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm3 = nn.LayerNorm(128)
        self.dec_prelu3 = nn.PReLU(self.width)
        self.dec_dense3 = DenseBlock_origin(128, 5, self.width)

        self.dec_conv2 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm2 = nn.LayerNorm(256)
        self.dec_prelu2 = nn.PReLU(self.width)
        self.dec_dense2 = DenseBlock_origin(256, 5, self.width)

        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(512)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.dec_dense1 = DenseBlock_origin(512, 5, self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=1, kernel_size=(1, 1))

        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)

    def forward(self, encoder_output, skip_connection):

        b6 = self.dec_dense6(self.dec_prelu6(self.dec_norm6(self.dec_conv6(self.pad1(encoder_output)))))
        b6 = torch.cat([b6, skip_connection[-1]], 1)

        b5 = self.dec_dense5(self.dec_prelu5(self.dec_norm5(self.dec_conv5(self.pad1(b6)))))
        b5 = torch.cat([b5, skip_connection[-2]], 1)

        b4 = self.dec_dense4(self.dec_prelu4(self.dec_norm4(self.dec_conv4(self.pad1(b5)))))
        b4 = torch.cat([b4, skip_connection[-3]], 1)

        b3 = self.dec_dense3(self.dec_prelu3(self.dec_norm3(self.dec_conv3(self.pad1(b4)))))
        b3 = torch.cat([b3, skip_connection[-4]], 1)

        b2 = self.dec_dense2(self.dec_prelu2(self.dec_norm2(self.dec_conv2(self.pad1(b3)))))
        b2 = torch.cat([b2, skip_connection[-5]], 1)

        b1 = self.dec_dense1(self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad1(b2)))))

        t_encoder_outp = self.out_conv(b1)

        return t_encoder_outp


class F_Decoder(nn.Module):
    def __init__(self):
        super(F_Decoder, self).__init__()

        self.width = 64

        self.dec_conv6 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm6 = nn.LayerNorm(16)
        self.dec_prelu6 = nn.PReLU(self.width)
        self.dec_dense6 = DenseBlock_origin(16, 5, self.width)

        self.dec_conv5 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm5 = nn.LayerNorm(32)
        self.dec_prelu5 = nn.PReLU(self.width)
        self.dec_dense5 = DenseBlock_origin(32, 5, self.width)

        self.dec_conv4 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm4 = nn.LayerNorm(64)
        self.dec_prelu4 = nn.PReLU(self.width)
        self.dec_dense4 = DenseBlock_origin(64, 5, self.width)

        self.dec_conv3 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm3 = nn.LayerNorm(128)
        self.dec_prelu3 = nn.PReLU(self.width)
        self.dec_dense3 = DenseBlock_origin(128, 5, self.width)

        self.dec_conv2 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm2 = nn.LayerNorm(256)
        self.dec_prelu2 = nn.PReLU(self.width)
        self.dec_dense2 = DenseBlock_origin(256, 5, self.width)

        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(512)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.dec_dense1 = DenseBlock_origin(512, 5, self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=1, kernel_size=(1, 1))

        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)

    def forward(self, encoder_output, skip_connection):

        b6 = self.dec_dense6(self.dec_prelu6(self.dec_norm6(self.dec_conv6(self.pad1(encoder_output)))))
        b6 = torch.cat([b6, skip_connection[-1]], 1)

        b5 = self.dec_dense5(self.dec_prelu5(self.dec_norm5(self.dec_conv5(self.pad1(b6)))))
        b5 = torch.cat([b5, skip_connection[-2]], 1)

        b4 = self.dec_dense4(self.dec_prelu4(self.dec_norm4(self.dec_conv4(self.pad1(b5)))))
        b4 = torch.cat([b4, skip_connection[-3]], 1)

        b3 = self.dec_dense3(self.dec_prelu3(self.dec_norm3(self.dec_conv3(self.pad1(b4)))))
        b3 = torch.cat([b3, skip_connection[-4]], 1)

        b2 = self.dec_dense2(self.dec_prelu2(self.dec_norm2(self.dec_conv2(self.pad1(b3)))))
        b2 = torch.cat([b2, skip_connection[-5]], 1)

        b1 = self.dec_dense1(self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad1(b2)))))

        f_encoder_outp = self.out_conv(b1)

        return f_encoder_outp

class FusionNet(nn.Module):
    """
    一个 encoder，两个decoder
    """

    def __init__(self):
        super(FusionNet, self).__init__()

        self.encoder = Encoder()
        self.t_decoder = T_Decoder()
        self.f_decoder = F_Decoder()
        # self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))

    def forward(self, input):
        # 分帧 加窗
        encoder_outp, skip_connection = self.encoder(input)
        t_decoder_outp = self.t_decoder(encoder_outp, skip_connection)
        t_decoder_outp = t_decoder_outp.squeeze(1)
        f_decoder_outp = self.f_decoder(encoder_outp, skip_connection)
        f_decoder_outp = f_decoder_outp.squeeze(1)

        # return 1
        return t_decoder_outp, f_decoder_outp

import numpy as np
if __name__ == '__main__':
    input = Variable(torch.FloatTensor(torch.rand(1, 63, 512))).cuda(0)
    net = FusionNet().cuda()
    macs, params = profile(net, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
    print("%s | %.2f | %.2f" % ('elephantstudent', params, macs))
