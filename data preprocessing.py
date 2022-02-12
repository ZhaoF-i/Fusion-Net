"""
在训练阶段，给定的语音信号被分割成波形块，其滑动窗口约为1秒的语音（16384个样本），重叠度为50%。
测试集在整个信号持续时间内没有重叠。
"""
import numpy as np
from pathlib import Path
import librosa

EPSILON = 1e-7

# train data, 生成 noise + clean，未切割
noise_path = '/data01/zhaofei/data/Fusion-Net/noise'
speech_path = '/data01/zhaofei/data/Voice-Bank/clean_trainset_28spk_wav'
snr_list = [0, 5, 10, 15]

def mix2signal(sig1, sig2, snr):
    alpha = np.sqrt((np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + EPSILON)) / 10.0 ** (snr / 10.0))
    return alpha

noise_lst = list(Path(noise_path).rglob('*.wav'))
speech_lst = list(Path(speech_path).rglob('*.wav'))

for i in speech_lst:
    speech = librosa.load(str(i), sr=None)
    speech = librosa.resample(speech, 48000, 16000)
    for j in noise_lst:
        noise = librosa.load(str(j), sr=None)