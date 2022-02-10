import torch
from utils.stft_istft_real_imag_hamming import STFT as complex_STFT
from utils.stft_istft import STFT as mag_STFT


class stftm_loss(object):
    def __init__(self, frame_size=320, frame_shift=160, loss_type='mae'):
        super(stftm_loss, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.complex_stft = complex_STFT(self.frame_size, self.frame_shift).cuda()

    def __call__(self, est, data_info):
        # est : est waveform
        mask = data_info[3].cuda()
        est_spec = self.complex_stft.transform(est)
        raw_spec = self.complex_stft.transform(data_info[1].cuda())

        est_spec = torch.abs(est_spec[..., 0]) + torch.abs(est_spec[..., 1])
        raw_spec = torch.abs(raw_spec[..., 0]) + torch.abs(raw_spec[..., 1])

        if self.loss_type == 'mse':
            loss = torch.sum((est_spec - raw_spec) ** 2) / torch.sum(mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(est_spec - raw_spec)) / torch.sum(mask)

        return loss


class mag_loss(object):
    def __init__(self, frame_size=320, frame_shift=160, loss_type='mae'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.mag_stft = mag_STFT(self.frame_size, self.frame_shift)

    def __call__(self, est, data_info):
        # est : [est_mag,noisy_phase]
        # data_info : [mixture,speech,noise,mask,nframe,len_speech]
        mask = data_info[3].cuda()

        raw_mag = self.mag_stft.transform(data_info[1])[0].permute(0, 2, 1).cuda()
        est_mag = est[0]
        if self.loss_type == 'mse':
            loss = torch.sum((est_mag - raw_mag) ** 2) / torch.sum(mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(est_mag - raw_mag)) / torch.sum(mask)
        return loss


class wavemse_loss(object):
    def __call__(self, est, data_info):
        mask = data_info[3]
        raw = data_info[1]
        loss = torch.sum((est - raw) ** 2) / torch.sum(mask)
        return loss


class sisdr_loss(object):
    def __init__(self):
        self.EPSILON = 1e-7

    def __call__(self, est, data_info):
        raw = data_info[1]
        batch = raw.size(0)
        raw = raw.contiguous().view(-1).unsqueeze(-1)
        est = est.contiguous().view(-1).unsqueeze(-1)
        Rss = torch.mm(raw.T, raw)
        a = (self.EPSILON + torch.mm(raw.T, est)) / (Rss + self.EPSILON)
        e_true = a * raw
        e_res = est - e_true
        Sss = (e_true ** 2).sum()
        Snn = (e_res ** 2).sum()
        sisdr = 10 * torch.log10((self.EPSILON + Sss) / (self.EPSILON + Snn))
        return sisdr
