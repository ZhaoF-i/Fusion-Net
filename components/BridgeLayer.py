import torch
from torch import nn
import numpy as np
import math

class RealSpec0(nn.Module):
    def __init__(self, len):
        super(RealSpec0, self).__init__()
        self.len = len

        basis = np.fft.fft(np.eye(self.len))
        basis = np.real(basis[0:len, 0:len])
        win = np.ones([1, 1, len])
        self.fbasis = nn.Parameter(torch.FloatTensor(basis), requires_grad=True)
        self.bbasis = nn.Parameter(torch.FloatTensor(basis), requires_grad=True)
        # self.fwin = nn.Parameter(torch.FloatTensor(win),requires_grad=True)
        # self.bwin = nn.Parameter(torch.FloatTensor(win),requires_grad=True)
        # self.fbasis = nn.Parameter(torch.FloatTensor(len,len),requires_grad=True)
        # self.bbasis = nn.Parameter(torch.FloatTensor(len,len),requires_grad=True)
        self.reset_parameters()
        # self.fmask = nn.Parameter(torch.FloatTensor(np.zeros([len, 1])), requires_grad=True)
        # self.bmask = nn.Parameter(torch.FloatTensor(np.zeros([1, len])), requires_grad=True)

    def forward(self, input, fft=True):
        if fft is True:
            output = torch.matmul(input, self.fbasis)
        else:
            output = torch.matmul(input, self.bbasis) / self.len
        return output

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fbasis.size(1))
        self.fbasis.data.uniform_(-stdv, stdv)
        self.bbasis.data.uniform_(-stdv, stdv)