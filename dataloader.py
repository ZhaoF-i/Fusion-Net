import scipy.io as sio
import numpy as np
import struct
import mmap
import os

from torch.nn.utils.rnn import *
from torch.autograd.variable import *
from torch.utils.data import Dataset, DataLoader


class SpeechMixDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        lst = config['TRAIN_LST'] if mode == 'train' else config['CV_LST']
        self.train_lst = sio.loadmat(os.path.join(config['CONFIG_PATH'], lst))
        # noise
        self.f = open(config['NOISE_PATH'], 'r+b')
        self.mm = mmap.mmap(self.f.fileno(), 0)
        # speech
        self.f_speech = open(config['WSJ0_PATH'], 'r+b')
        self.mm_speech = mmap.mmap(self.f_speech.fileno(), 0)

        self.SNR = self.train_lst['snr'][:, 0]
        self.SNR_len = len(self.SNR)
        self.speech_idx = list(map(int, (list(self.train_lst['speech_idx'][:, 0]))))
        self.noise_idx = list(map(int, (list(self.train_lst['noise_idx'][:, 0]))))
        self.snr_idx = self.train_lst['snr_idx'][:, 0]
        self.len = self.train_lst['num_utter'][0, 0]
        self.SPEECH_LEN = self.train_lst['speech_len'][0, 0]

    def __len__(self):
        return self.len

    def mix2signal(self, sig1, sig2, snr):
        alpha = np.sqrt((np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + self.config['EPSILON'])) / 10.0 ** (snr / 10.0))
        return alpha

    def __getitem__(self, idx):
        nframe = (self.SPEECH_LEN - self.config['WIN_LEN']) // self.config['WIN_OFFSET'] + 1
        len_speech = (nframe + 1) * self.config['WIN_OFFSET']
        Snr = self.SNR[self.snr_idx[idx]]
        # noise
        noise_loc = self.noise_idx[idx]
        noise = np.array(list(struct.unpack('f' * len_speech, self.mm[noise_loc * 4:noise_loc * 4 + len_speech * 4])))
        # speech
        speech_loc = self.speech_idx[idx]
        speech = np.array(
            list(struct.unpack('f' * len_speech, self.mm_speech[speech_loc * 4:speech_loc * 4 + len_speech * 4])))

        speech = speech[:len_speech]
        noise = noise[:len_speech]
        alpha = self.mix2signal(speech, noise, Snr)
        noise = alpha * noise
        mixture = noise + speech
        alpha_pow = 1 / (np.sqrt(np.sum(mixture ** 2) / len_speech) + self.config['EPSILON'])
        speech = alpha_pow * speech
        noise = alpha_pow * noise
        mask_for_loss = np.ones((nframe, self.config['FFT_SIZE']), dtype=np.float32)
        sample = (Variable(torch.FloatTensor(speech.astype('float32'))),
                  Variable(torch.FloatTensor(noise.astype('float32'))),
                  Variable(torch.FloatTensor(mask_for_loss)),
                  nframe,
                  len_speech
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
