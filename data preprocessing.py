"""
在训练阶段，给定的语音信号被分割成波形块，其滑动窗口约为1秒的语音（16384个样本），重叠度为50%。
测试集在整个信号持续时间内没有重叠。
"""
#
import random
import tqdm
import soundfile as sf
import numpy as np
from pathlib import Path
import librosa

EPSILON = 1e-7

# train data, 生成 noise + clean，未切割


def mix2signal(sig1, sig2, snr):
    alpha = np.sqrt((np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + EPSILON)) / 10.0 ** (snr / 10.0))
    return alpha


## noise + speech
def mix_speech(speech_lst, noise_lst, uncut_mix_path):
    for i in tqdm.tqdm(speech_lst):
        speech, _ = sf.read(str(i))
        speech = librosa.resample(speech, _, 16000)
        for j in noise_lst:
            noise, _ = sf.read(str(j))
            # 随机取一段与speech等长的噪音
            start = random.randint(0, noise.shape[0] - speech.shape[0])
            noise = noise[start: speech.shape[0] + start]
            for k in test_snr_list:
                alpha = mix2signal(speech, noise, k)
                noise = alpha * noise
                mixture = noise + speech
                mix_name = i.stem + '&' + j.stem + '&' + str(k) + 'dB.wav'
                sf.write(uncut_mix_path + mix_name, mixture, 16000)

def cut_mix(mix_lst, train_path):
    # 对mix按照16384个样本进行切割
    for i in tqdm.tqdm(mix_lst):
        mixture, _ = sf.read(str(i))

        label = 0
        t = 0
        while (True):
            if label + 16384 > mixture.shape[0]:
                ones = np.zeros(16384)
                ones[:mixture.shape[0] - label] = mixture[label:]
                sf.write(train_path + i.stem + '-' + str(t) + '.wav', ones, samplerate=16000)
                break
            split_mix = mixture[label: label + 16384]
            sf.write(train_path + i.stem + '-' + str(t) + '.wav', split_mix, samplerate=16000)
            label = label + 16384
            t = t + 1

# 对clean按照16384个样本进行切割  /data01/zhaofei/data/Fusion-Net/cut_speech/p270_034-0.wav
def cut_speech(speech_lst, train_speech_path):
    sum = 0
    for i in tqdm.tqdm(speech_lst):
        speech, _ = sf.read(str(i))
        speech = librosa.resample(speech, _, 16000)
        label = 0
        t = 0
        while (True):
            if label + 16384 > speech.shape[0]:
                ones = np.zeros(16384)
                ones[:speech.shape[0] - label] = speech[label:]
                sf.write(train_speech_path + i.stem + '-' + str(t) + '.wav', ones, samplerate=16000)
                break
            split_mix = speech[label: label + 16384]
            sf.write(train_speech_path + i.stem + '-' + str(t) + '.wav', split_mix, samplerate=16000)
            label = label + 8192
            t = t + 1
        sum = sum + t + 1
    print(sum)

if __name__ == '__main__':
    # train_noise_path = '/data01/zhaofei/data/Fusion-Net/noise'
    # train_speech_path = '/data01/zhaofei/data/Voice-Bank/clean_trainset_28spk_wav'
    test_noise_path = '/data01/zhaofei/data/Fusion-Net/test_noise'
    test_speech_path = '/data01/zhaofei/data/Voice-Bank/clean_testset_wav'
    uncut_train_mix_path = '/data01/zhaofei/data/Fusion-Net/uncut_train_mix/'
    uncut_test_mix_path = '/data01/zhaofei/data/Fusion-Net/uncut_test_mix/'
    train_noisy_path = '/data01/zhaofei/data/Fusion-Net/cut_mix/'
    cut_test_mix_path = '/data01/zhaofei/data/Fusion_Net/cut_test_mix/'
    cut_test_speech_path = '/data01/zhaofei/data/Fusion_Net/cut_test_speech/'
    train_snr_list = [0, 5, 10, 15]
    test_snr_list = [2.5, 7.5, 12.5, 17.5]

    noise_lst = list(Path(test_noise_path).rglob('*.wav'))
    speech_lst = list(Path(test_speech_path).rglob('*.wav'))
    mix_lst = list(Path(uncut_test_mix_path).rglob('*.wav'))
    # mix_speech(speech_lst, noise_lst, uncut_test_mix_path)
    cut_mix(mix_lst, cut_test_mix_path)