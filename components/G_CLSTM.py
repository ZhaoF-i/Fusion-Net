import torch
from torch import nn
import numpy as np


class G_CLstm(nn.Module):
    def __init__(self, channel, feature, groups, bidirectional=False):
        super(G_CLstm, self).__init__()
        self.channel = channel
        self.feature = feature
        self.groups = groups
        self.bidir = bidirectional
        self.step = feature * channel // groups
        self.sublstmlist = nn.ModuleList()
        for l in range(2):
            for g in range(groups):
                self.sublstmlist.append(
                    nn.LSTM(input_size=self.step, hidden_size=self.step, num_layers=1, batch_first=True,
                            bidirectional=bidirectional))
        snd_layer_index = []
        for i in range(channel // groups):
            for g in range(groups):
                for f in range(feature):
                    snd_layer_index.append(i * feature + g * self.step + f)
        self.register_buffer('snd_layer_index', torch.from_numpy(np.array(snd_layer_index, dtype=np.long)))
        if self.bidir:
            self.linear = nn.Linear(in_features=640, out_features=320)

    def forward(self, input):
        [batch, channel, time, feature] = input.shape
        output = input.transpose(1, 2).contiguous().view(batch, time, -1)
        # first layer
        out = []
        for g in range(self.groups):
            self.sublstmlist[g].flatten_parameters()
            lstm_out, _ = self.sublstmlist[g](output[:, :, g * self.step:(g + 1) * self.step])
            out.append(lstm_out)
        output = torch.cat(out, dim=2)
        if self.bidir:
            output = self.linear(output)
        # second layer
        out = []
        for g in range(self.groups):
            self.sublstmlist[g + self.groups].flatten_parameters()
            output1 = torch.index_select(output, dim=2, index=self.snd_layer_index[g * self.step:(g + 1) * self.step])
            lstm_out, _ = self.sublstmlist[g + self.groups](output1)
            out.append(lstm_out)
        output = torch.cat(out, dim=2)
        if self.bidir:
            output = self.linear(output)

        lstm_out = output.contiguous().view(batch, time, channel, feature)
        lstm_out = lstm_out.transpose(1, 2)

        return lstm_out
