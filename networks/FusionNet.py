import torch.nn as nn
import torch
import torch.functional as F

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, dilation):  # num_input_features特征层数
        super(_DenseLayer, self).__init__()
        self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate,
                                           kernel_size=(2, 3), dilation=dilation,
                                           ))
        self.pad = nn.ConstantPad2d((2 * dilation, 0, dilation, 0), value=0.)

    def forward(self, prev_features):

        new_features = self.conv1(self.pad(prev_features))
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super(_DenseBlock, self).__init__()  # num_layers层重复次数
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,  # for一次层数增加64
                growth_rate=growth_rate,
                dilation=2 ** i
            )
            self.add_module('denselayer%d' % (i + 1), layer)  # 追加denselayer层到字典里面

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():  # 依次遍历添加的6个layer层，
            new_features = layer(torch.cat(features, 1))  # 计算特征
            features.append(new_features)  # 追加特征
        return features[5]


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, r=2):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=(1, 3), stride=(1, 1))
        self.pad = nn.ConstantPad2d((2,0,0,0), value=0.)
        self.r = r

    def forward(self, x):
        out = self.pad(self.conv(x))
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.prelu = nn.PReLU()
        self.LN1 = nn.LayerNorm([63,256])
        self.dense1 = _DenseBlock(5, 64, 64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.LN2 = nn.LayerNorm([63,128])
        self.dense2 = _DenseBlock(5, 64, 64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.LN3 = nn.LayerNorm([63,64])
        self.dense3 = _DenseBlock(5, 64, 64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.LN4 = nn.LayerNorm([63,32])
        self.dense4 = _DenseBlock(5, 64, 64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.LN5 = nn.LayerNorm([63,16])
        self.dense5 = _DenseBlock(5, 64, 64)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.LN6 = nn.LayerNorm([63,8])
        self.dense6 = _DenseBlock(5, 64, 64)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.LN7 = nn.LayerNorm([63,4])
        self.dense7 = _DenseBlock(5, 64, 64)

    def forward(self, input):
        input = input.unsqueeze(1)
        b1 = self.conv(input)

        b1 = self.dense1(self.prelu(self.LN1(self.conv1(b1))))
        b2 = self.dense2(self.prelu(self.LN2(self.conv2(b1))))
        b3 = self.dense3(self.prelu(self.LN3(self.conv2(b2))))
        b4 = self.dense4(self.prelu(self.LN4(self.conv2(b3))))
        b5 = self.dense5(self.prelu(self.LN5(self.conv2(b4))))
        b6 = self.dense6(self.prelu(self.LN6(self.conv2(b5))))
        b7 = self.dense7(self.prelu(self.LN7(self.conv2(b6))))

        return b7, [b2, b3, b4, b5, b6]


class T_Decoder(nn.Module):
    def __init__(self):
        super(T_Decoder, self).__init__()
        self.sp6 = SPConvTranspose2d()
        self.prelu = nn.PReLU()
        self.LN6 = nn.LayerNorm([63, 8])
        self.dense6 = _DenseBlock(5, 64, 64)

        self.sp5 = SPConvTranspose2d(r=1)
        self.LN5 = nn.LayerNorm([63, 16])
        self.dense5 = _DenseBlock(5, 64, 64)

        self.sp4 = SPConvTranspose2d(r=1)
        self.LN4 = nn.LayerNorm([63, 32])
        self.dense4 = _DenseBlock(5, 64, 64)

        self.sp3 = SPConvTranspose2d(r=1)
        self.LN3 = nn.LayerNorm([63, 64])
        self.dense3 = _DenseBlock(5, 64, 64)

        self.sp2 = SPConvTranspose2d(r=1)
        self.LN2 = nn.LayerNorm([63, 128])
        self.dense2 = _DenseBlock(5, 64, 64)

        self.sp1 = SPConvTranspose2d(r=2)
        self.LN1 = nn.LayerNorm([63, 512])
        self.dense1 = _DenseBlock(5, 64, 64)

        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))

    def forward(self, encoder_output, skip_connection):
        b6 = self.dense6(self.prelu(self.LN6(self.sp6(encoder_output))))

        b6 = torch.cat([b6, skip_connection[-1]], -1)

        b5 = self.dense5(self.prelu(self.LN5(self.sp5(b6))))
        b5 = torch.cat([b5, skip_connection[-2]], -1)

        b4 = self.dense4(self.prelu(self.LN4(self.sp4(b5))))
        b4 = torch.cat([b4, skip_connection[-3]], -1)

        b3 = self.dense3(self.prelu(self.LN3(self.sp3(b4))))
        b3 = torch.cat([b3, skip_connection[-4]], -1)

        b2 = self.dense2(self.prelu(self.LN2(self.sp2(b3))))
        b2 = torch.cat([b2, skip_connection[-5]], -1)

        b1 = self.dense1(self.prelu(self.LN1(self.sp1(b2))))

        t_encoder_outp = self.conv(b1)

        return t_encoder_outp


class F_Decoder(nn.Module):
    def __init__(self):
        super(F_Decoder, self).__init__()
        self.sp6 = SPConvTranspose2d()
        self.prelu = nn.PReLU()
        self.LN6 = nn.LayerNorm([63, 8])
        self.dense6 = _DenseBlock(5, 64, 64)

        self.sp5 = SPConvTranspose2d(r=1)
        self.LN5 = nn.LayerNorm([63, 16])
        self.dense5 = _DenseBlock(5, 64, 64)

        self.sp4 = SPConvTranspose2d(r=1)
        self.LN4 = nn.LayerNorm([63, 32])
        self.dense4 = _DenseBlock(5, 64, 64)

        self.sp3 = SPConvTranspose2d(r=1)
        self.LN3 = nn.LayerNorm([63, 64])
        self.dense3 = _DenseBlock(5, 64, 64)

        self.sp2 = SPConvTranspose2d(r=1)
        self.LN2 = nn.LayerNorm([63, 128])
        self.dense2 = _DenseBlock(5, 64, 64)

        self.sp1 = SPConvTranspose2d(r=2)
        self.LN1 = nn.LayerNorm([63, 512])
        self.dense1 = _DenseBlock(5, 64, 64)

        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))

    def forward(self, encoder_output, skip_connection):
        b6 = self.dense6(self.prelu(self.LN6(self.sp6(encoder_output))))
        b6 = torch.cat([b6, skip_connection[-1]], -1)

        b5 = self.dense5(self.prelu(self.LN5(self.sp5(b6))))
        b5 = torch.cat([b5, skip_connection[-2]], -1)

        b4 = self.dense4(self.prelu(self.LN4(self.sp4(b5))))
        b4 = torch.cat([b4, skip_connection[-3]], -1)

        b3 = self.dense3(self.prelu(self.LN3(self.sp3(b4))))
        b3 = torch.cat([b3, skip_connection[-4]], -1)

        b2 = self.dense2(self.prelu(self.LN2(self.sp2(b3))))
        b2 = torch.cat([b2, skip_connection[-5]], -1)

        b1 = self.dense1(self.prelu(self.LN1(self.sp1(b2))))

        f_encoder_outp = self.conv(b1)

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
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))

    def forward(self, input):
        # 分帧 加窗
        encoder_outp, skip_connection = self.encoder(input)
        t_decoder_outp =  self.t_decoder(encoder_outp, skip_connection)
        t_decoder_outp = t_decoder_outp.squeeze(1)
        f_decoder_outp =  self.f_decoder(encoder_outp, skip_connection)
        f_decoder_outp = f_decoder_outp.squeeze(1)

        return t_decoder_outp, f_decoder_outp
        # input = input.unsqueeze(0)
        # input = self.conv(input)
        # input = input.squeeze(0)
        # return input, input
