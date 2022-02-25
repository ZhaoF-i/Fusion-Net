import numpy as np
import struct
import soundfile as sf
import os
import torch
from torch.nn.utils.rnn import *
from torch.autograd.variable import *
from torch.utils.data import Dataset, DataLoader
from utils.util import convert_to_frame


class SpeechMixDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.speech_path = config['TRAIN_SPEECH_LST'] if mode == 'train' else config['CV_SPEECH_LST']
        self.mix_path = config['TRAIN_MIX_LST'] if mode == 'train' else config['CV_MIX_LST']

        self.speech_lst = np.load(self.speech_path, allow_pickle=True)
        self.mix_lst = np.load(self.mix_path, allow_pickle=True)

    def __len__(self):
        return len(self.speech_lst)-1

    def __getitem__(self, idx):
        speech_wav, _ = sf.read(self.config['SPEECH_PATH']+str(self.speech_lst[idx])+'.wav')
        alpha_pow = 1 / ((np.sqrt(np.sum(speech_wav ** 2)) / ((speech_wav.size) + 1e-7)) + 1e-7)
        speech_wav = speech_wav * alpha_pow

        mix_wav, _ = sf.read(self.config['MIX_PATH']+str(self.mix_lst[idx])+'.wav')
        alpha_pow = 1 / ((np.sqrt(np.sum(mix_wav ** 2)) / ((mix_wav.size) + 1e-7)) + 1e-7)
        mix_wav = mix_wav * alpha_pow

        t_mask = torch.ones(len(mix_wav), dtype=torch.float32)
        f_mask = torch.ones((len(mix_wav)//self.config['FFT_SIZE']-1, self.config['FFT_SIZE']+1), dtype=torch.float32)

        sample = (Variable(torch.FloatTensor(speech_wav.astype('float32'))),
                  Variable(torch.FloatTensor(mix_wav.astype('float32'))),
                  t_mask,
                  f_mask
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
        speech, mix, t_mask, f_mask = zip(*batch)
        speech = pad_sequence(speech, batch_first=True)
        mix = pad_sequence(mix, batch_first=True)
        mix = convert_to_frame(mix, hop_size=256, use_window=False)
        t_mask = pad_sequence(t_mask, batch_first=True)
        f_mask = pad_sequence(f_mask, batch_first=True)

        return [mix, speech, t_mask, f_mask]
