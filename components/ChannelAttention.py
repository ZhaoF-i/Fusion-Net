from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid

    def forward(self, input):
        #
        # return a weight matrix
        #
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(input))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(input))))
        out = avg_out + max_out
        return self.sigmoid(out)
