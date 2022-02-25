import torch
import torch.nn as nn
from thop import profile, clever_format
from torch.autograd import Variable
from components.DenseBlock import DenseBlock_origin
from components.SPConv import SPConvTranspose2d
from utils.FrameOptions import frames_data, frames_overlap, pad_input


class NET_Wrapper(nn.Module):
    def __init__(self, win_len, win_offset):
        super(NET_Wrapper, self).__init__()
        self.win_len = win_len
        self.win_offset = win_offset
        self.in_channel = 1
        self.out_channel = 1
        self.width = 64
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        # input conv
        self.inp_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.width, kernel_size=(1, 1))
        self.inp_norm = nn.LayerNorm(512)
        self.inp_prelu = nn.PReLU(self.width)

        self.enc_dense1 = DenseBlock_origin(512, 5, self.width)
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm1 = nn.LayerNorm(256)
        self.enc_prelu1 = nn.PReLU(self.width)

        self.enc_dense2 = DenseBlock_origin(256, 5, self.width)
        self.enc_conv2 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm2 = nn.LayerNorm(128)
        self.enc_prelu2 = nn.PReLU(self.width)

        self.enc_dense3 = DenseBlock_origin(128, 5, self.width)
        self.enc_conv3 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm3 = nn.LayerNorm(64)
        self.enc_prelu3 = nn.PReLU(self.width)

        self.enc_dense4 = DenseBlock_origin(64, 5, self.width)
        self.enc_conv4 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm4 = nn.LayerNorm(32)
        self.enc_prelu4 = nn.PReLU(self.width)

        self.enc_dense5 = DenseBlock_origin(32, 5, self.width)
        self.enc_conv5 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm5 = nn.LayerNorm(16)
        self.enc_prelu5 = nn.PReLU(self.width)

        self.enc_dense6 = DenseBlock_origin(16, 5, self.width)
        self.enc_conv6 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))
        self.enc_norm6 = nn.LayerNorm(8)
        self.enc_prelu6 = nn.PReLU(self.width)

        self.dec_dense6 = DenseBlock_origin(8, 5, self.width)
        self.dec_conv6 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm6 = nn.LayerNorm(16)
        self.dec_prelu6 = nn.PReLU(self.width)

        self.dec_dense5 = DenseBlock_origin(16, 5, self.width)
        self.dec_conv5 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm5 = nn.LayerNorm(32)
        self.dec_prelu5 = nn.PReLU(self.width)

        self.dec_dense4 = DenseBlock_origin(32, 5, self.width)
        self.dec_conv4 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm4 = nn.LayerNorm(64)
        self.dec_prelu4 = nn.PReLU(self.width)

        self.dec_dense3 = DenseBlock_origin(64, 5, self.width)
        self.dec_conv3 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm3 = nn.LayerNorm(128)
        self.dec_prelu3 = nn.PReLU(self.width)

        self.dec_dense2 = DenseBlock_origin(128, 5, self.width)
        self.dec_conv2 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm2 = nn.LayerNorm(256)
        self.dec_prelu2 = nn.PReLU(self.width)

        self.dec_dense1 = DenseBlock_origin(256, 5, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width * 2, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(512)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channel, kernel_size=(1, 1))

    def forward(self, input):
        pad, rest = pad_input(input, self.win_len)
        frames = frames_data(pad, self.win_offset, False)

        enc_list = []

        out = self.inp_prelu(self.inp_norm(self.inp_conv(frames.unsqueeze(1))))

        out = self.enc_dense1(out)
        out = self.enc_prelu1(self.enc_norm1(self.enc_conv1(self.pad1(out))))
        enc_list.append(out)

        out = self.enc_dense2(out)
        out = self.enc_prelu2(self.enc_norm2(self.enc_conv2(self.pad1(out))))
        enc_list.append(out)

        out = self.enc_dense3(out)
        out = self.enc_prelu3(self.enc_norm3(self.enc_conv3(self.pad1(out))))
        enc_list.append(out)

        out = self.enc_dense4(out)
        out = self.enc_prelu4(self.enc_norm4(self.enc_conv4(self.pad1(out))))
        enc_list.append(out)

        out = self.enc_dense5(out)
        out = self.enc_prelu5(self.enc_norm5(self.enc_conv5(self.pad1(out))))
        enc_list.append(out)

        out = self.enc_dense6(out)
        out = self.enc_prelu6(self.enc_norm6(self.enc_conv6(self.pad1(out))))
        enc_list.append(out)

        out = self.dec_dense6(out)
        out = torch.cat([out, enc_list[-1]], dim=1)
        out = self.dec_prelu6(self.dec_norm6(self.dec_conv6(self.pad1(out))))

        out = self.dec_dense5(out)
        out = torch.cat([out, enc_list[-2]], dim=1)
        out = self.dec_prelu5(self.dec_norm5(self.dec_conv5(self.pad1(out))))

        out = self.dec_dense4(out)
        out = torch.cat([out, enc_list[-3]], dim=1)
        out = self.dec_prelu4(self.dec_norm4(self.dec_conv4(self.pad1(out))))

        out = self.dec_dense3(out)
        out = torch.cat([out, enc_list[-4]], dim=1)
        out = self.dec_prelu3(self.dec_norm3(self.dec_conv3(self.pad1(out))))

        out = self.dec_dense2(out)
        out = torch.cat([out, enc_list[-5]], dim=1)
        out = self.dec_prelu2(self.dec_norm2(self.dec_conv2(self.pad1(out))))

        out = self.dec_dense1(out)
        out = torch.cat([out, enc_list[-6]], dim=1)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad1(out))))


        out = self.out_conv(out)
        wav = frames_overlap(out.squeeze(1), self.win_offset, False)
        output = wav[:, self.win_offset:-(rest + self.win_offset)].contiguous()

        return output


if __name__ == '__main__':
    input = Variable(torch.FloatTensor(torch.rand(1, 16000))).cuda(0)
    net = NET_Wrapper(320, 160).cuda()
    macs, params = profile(net, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
    print("%s | %.2f | %.2f" % ('elephantstudent', params, macs))
