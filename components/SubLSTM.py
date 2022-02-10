import torch
import torch.nn as nn


class SubLSTM(nn.Module):
    def __init__(self, feature, groups=8, dims=20):
        super(SubLSTM, self).__init__()
        self.groups = groups
        self.dims = dims
        self.num_layers = 2
        self.sub_lstms = []
        for i in range(groups):
            self.sub_lstms.append(
                nn.LSTM(input_size=self.dims, hidden_size=self.dims, num_layers=self.num_layers, batch_first=True)
            )
        self.feedforward = nn.Sequential(
            nn.Linear(feature, 4 * feature),
            nn.ReLU(),
            nn.Linear(4 * feature, feature)
        )

    def split(self, input, groups):
        out = input
        return out,out

    def forward(self, input):
        sub_components, rest_component = self.split(input, self.groups)
        sub_res = [rest_component]
        for i in range(self.groups):
            sub_res.append(self.sub_lstms[i](sub_components[i]))
        fullband = torch.cat(sub_res, dim=3) + input
        out = self.feedforward(fullband)
        return out
