import torch.nn as nn
import torch.autograd.variable


from torch.autograd.variable import *

from utils.stft_istft_real_imag import STFT


class NET_Wrapper(nn.Module):
    def __init__(self,win_len,win_offset):
        self.win_len = win_len
        self.win_offset = win_offset
        super(NET_Wrapper, self).__init__()
        self.lstm_input_size = 256 * 4
        self.lstm_layers = 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2))
        self.conv1_relu = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2))
        self.conv2_relu = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv3_relu = nn.ELU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2))
        self.conv4_relu = nn.ELU()
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2))
        self.conv5_relu = nn.ELU()
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_input_size,
                            num_layers=self.lstm_layers,
                            batch_first=True)

        self.conv5_t = nn.ConvTranspose2d(in_channels=256 * 2, out_channels=128, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv5_t_relu = nn.ELU()
        self.conv4_t = nn.ConvTranspose2d(in_channels=128 * 2, out_channels=64, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv4_t_relu = nn.ELU()
        self.conv3_t = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=32, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv3_t_relu = nn.ELU()
        self.conv2_t = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=16, kernel_size=(2, 3), stride=(1, 2),
                                          output_padding=(0, 1), padding=(1, 0))
        self.conv2_t_relu = nn.ELU()
        self.conv1_t = nn.ConvTranspose2d(in_channels=16 * 2, out_channels=1, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv1_t_relu = nn.Softplus()

        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.conv5_t_bn = nn.BatchNorm2d(128)
        self.conv4_t_bn = nn.BatchNorm2d(64)
        self.conv3_t_bn = nn.BatchNorm2d(32)
        self.conv2_t_bn = nn.BatchNorm2d(16)
        self.pad = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        self.STFT = STFT(self.win_len, self.win_offset).cuda()

    def forward(self, input_data_c1):
        # input_data_c1 = train_info_.mix_feat_b
        STFT_C1, phase_C1 = self.STFT.transform(input_data_c1)
        STFT_C1 = STFT_C1.permute(0, 2, 1)
        input_feature = STFT_C1

        e1 = self.conv1_relu(self.conv1_bn(self.conv1(self.pad(torch.stack([input_feature], 1)))))
        e2 = self.conv2_relu(self.conv2_bn(self.conv2(self.pad(e1))))
        e3 = self.conv3_relu(self.conv3_bn(self.conv3(self.pad(e2))))
        e4 = self.conv4_relu(self.conv4_bn(self.conv4(self.pad(e3))))
        e5 = self.conv5_relu(self.conv5_bn(self.conv5(self.pad(e4))))

        self.lstm.flatten_parameters()
        out_real = e5.contiguous().transpose(1, 2)
        out_real = out_real.contiguous().view(out_real.size(0), out_real.size(1), -1)
        lstm_out, _ = self.lstm(out_real)
        lstm_out_real = lstm_out.contiguous().view(lstm_out.size(0), lstm_out.size(1), 256, 4)
        lstm_out_real = lstm_out_real.contiguous().transpose(1, 2)

        t5 = self.conv5_t_relu(self.conv5_t_bn(self.conv5_t(self.pad(torch.cat((lstm_out_real, e5), dim=1)))))
        t4 = self.conv4_t_relu(self.conv4_t_bn(self.conv4_t(self.pad(torch.cat((t5, e4), dim=1)))))
        t3 = self.conv3_t_relu(self.conv3_t_bn(self.conv3_t(self.pad(torch.cat((t4, e3), dim=1)))))
        t2 = self.conv2_t_relu(self.conv2_t_bn(self.conv2_t(self.pad(torch.cat((t3, e2), dim=1)))))
        t1 = self.conv1_t_relu(self.conv1_t(self.pad(torch.cat((t2, e1), dim=1))))
        out = torch.squeeze(t1, 1)
        return [out, STFT_C1, phase_C1]
