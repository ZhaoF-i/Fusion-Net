import numpy as np
import struct
import soundfile as sf
import os

from torch.nn.utils.rnn import *
from torch.autograd.variable import *
from torch.utils.data import Dataset, DataLoader


class SpeechMixDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.speech_lst = config['TRAIN_SPEECH_LST'] if mode == 'train' else config['CV_SPEECH_LST']
        self.mix_lst = config['TRAIN_MIX_LST'] if mode == 'train' else config['CV_MIX_LST']

    def __len__(self):
        return self.len

    def mix2signal(self, sig1, sig2, snr):
        alpha = np.sqrt((np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + self.config['EPSILON'])) / 10.0 ** (snr / 10.0))
        return alpha

    def __getitem__(self, idx):
        speech_wav, _ = sf.read(str(self.speech_lst[idx]))
        alpha_pow = 1 / ((np.sqrt(np.sum(speech_wav ** 2)) / ((speech_wav.size) + 1e-7)) + 1e-7)
        speech_wav = speech_wav * alpha_pow

        mix_wav, _ = sf.read(str(self.mix_lst[idx]))
        alpha_pow = 1 / ((np.sqrt(np.sum(mix_wav ** 2)) / ((mix_wav.size) + 1e-7)) + 1e-7)
        mix_wav = mix_wav * alpha_pow

        sample = (Variable(torch.FloatTensor(speech_wav.astype('float32'))),
                  Variable(torch.FloatTensor(mix_wav.astype('float32')))
                  )

        return sample


class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=workers_num,
                                     collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        speech, noise, mask_for_loss, nframe, nsample = zip(*batch)
        speech = pad_sequence(speech, batch_first=True)
        noise = pad_sequence(noise, batch_first=True)
        mixture = speech + noise
        mask_for_loss = pad_sequence(mask_for_loss, batch_first=True)
        return [mixture, speech, noise, mask_for_loss, nframe, nsample]
