from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import *
import torch
from tqdm import tqdm
import soundfile as sf
from torch.autograd.variable import *
import torchaudio
import numpy as np
import pickle, os

class Dataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.lst=np.load("./"+mode+"_1s.npy",allow_pickle=True)
        self.mode=mode
    def __getitem__(self, index):
        wav,_=sf.read(str(self.lst[index]))
        wav=wav[:,0]
        alpha_pow = 1 / ((np.sqrt(np.sum(wav** 2)) / ((wav.size) + 1e-7)) + 1e-7)
        wav=wav*alpha_pow

        # label=np.load('/data01/spj/ai_shell4_vad/TRAIN/seg_label/'+self.lst[index].stem+'.npy')
        label=np.load('/data01/zhaofei/data/asr_dataset/ai_shell4_vad/TRAIN/seg_label/'+self.lst[index].stem+'.npy')
        label=np.minimum(label, 2)
        label=frame_level_label(label, 400, 200)

        sample=(
            Variable(torch.FloatTensor(wav.astype('float32'))),
            label,
            alpha_pow
        )
        return sample

    def __len__(self):
        return len(self.lst)


class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                     num_workers=workers_num, collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        wav, tag, alpha_pow = zip(*batch)


        wav_batch = pad_sequence(wav, batch_first=True)
        tag_batch=pad_sequence(tag,batch_first=True)

        return [wav_batch, tag_batch,alpha_pow]
        # return [wav, tag_batch]


if __name__ == '__main__':
    data_train = Dataset()
    tr_batch_dataloader = BatchDataLoader(data_train, 32, is_shuffle=True, workers_num=8)
    for i_batch, batch_data in enumerate(tr_batch_dataloader.get_dataloader()):
        print(i_batch)  # 打印batch编号
        # print(batch_data[0])  # 打印该batch里面src （会死掉的）
        print(batch_data[0].shape)
        print(batch_data[1])  # 打印该batch里面trg
