import random
import soundfile as sf
from tqdm import tqdm
import path
import numpy as np
import os
from pathlib import Path

if __name__ == '__main__':
    # name_lst=list(Path("/data01/spj/ai_shell4_vad/TRAIN/seg_wav/").rglob('*.wav'))
    speech_path = '/data01/zhaofei/data/Fusion-Net/cut_speech/'
    npy_path = '/data01/zhaofei/data/Fusion-Net/npy/'
    mix_lst=list(Path("/data01/zhaofei/data/Fusion-Net/cut_mix/").rglob('*.wav'))
    random.shuffle(mix_lst)

    len_lst=len(mix_lst)

    noisy_lst = []
    speech_lst = []

    for i in tqdm(mix_lst):
        noisy_lst.append(i.stem)
        speech_name = (i.stem).split('&')[0]+ '-' + (i.stem).split('-')[1]
        speech_lst.append(speech_name)
        # a,_ = sf.read(speech_name)
    # print("over")

    train_mix_lst = noisy_lst[:int(0.95*len_lst)]
    eval_mix_lst = noisy_lst[int(0.95 * len_lst):]
    train_speech_lst = speech_lst[:int(0.95*len_lst)]
    eval_speech_lst = speech_lst[int(0.9*len_lst):]

    np.save(npy_path+"train_mix.npy",train_mix_lst)
    np.save(npy_path+"train_speech.npy",train_speech_lst)
    np.save(npy_path+"val_mix.npy", eval_mix_lst)
    np.save(npy_path+"val_speech.npy", eval_speech_lst)
    print("列表已经生成")