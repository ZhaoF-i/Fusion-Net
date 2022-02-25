import torch
from torch import nn
import numpy as np

class RealSpec(nn.Module):
    def __init__(self, len, requires_grad=True):
        super(RealSpec, self).__init__()
        self.len = len
        if requires_grad is True:
            self.forward_basis = nn.Parameter(torch.FloatTensor(len, len))
            self.backward_basis = self.forward_basis  # nn.Parameter(torch.FloatTensor(len,len))
            self.reset_parameters()
        else:
            basis = np.fft.fft(np.eye(self.len * 2 - 1))
            basis = np.real(basis[0:len, 0:len])
            self.forward_basis = nn.Parameter(torch.FloatTensor(basis), requires_grad=False)
            self.backward_basis = nn.Parameter(torch.FloatTensor(basis), requires_grad=False)
        window = np.hamming(self.len)
        self.win = nn.Parameter(torch.FloatTensor(window).reshape(1, 1, self.len), requires_grad=False)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.forward_basis.size(1))
        self.forward_basis.data.uniform_(-stdv, stdv)
        self.backward_basis.data.uniform_(-stdv, stdv)

    def forward(self, input, fft=True, usewindow=False):
        if fft is True:
            if usewindow is True:
                input = input * self.win
            output = torch.matmul(input, self.forward_basis)
        else:
            output = torch.matmul(input[..., 1:], self.backward_basis[1:, :]) * 2 + \
                     torch.matmul(input[..., :1], self.backward_basis[:1, :])
            output = output / (self.len * 2 - 1) * 2
        return output



class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""

    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device,
                          requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones

