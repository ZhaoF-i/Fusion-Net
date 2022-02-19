import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

a, _ = sf.read('/data01/zhaofei/data/Fusion-Net/cut_speech/p254_105-3.wav')

def frames_data(noisy_speech, hop_size=256, use_window=True):

    frame_speech = noisy_speech.reshape(noisy_speech.size(0), -1, hop_size)
    frame_speech2 = frame_speech.clone()[:, 1:]
    frames = torch.cat([frame_speech[:, :-1], frame_speech2[:, :]], dim=-1)

    return frames

def overlap_data(data, frame_size=256, use_window=True):
    # hamming_win = torch.from_numpy((np.hamming(frame_size * 2)).astype(np.float32)).reshape(1, 1, frame_size * 2).cuda()
    # if use_window is True:
    #     data = data * hamming_win

    left_data = data[:, :, :frame_size]
    left_data = torch.cat([left_data, torch.zeros_like(left_data[:, -1:, :])], dim=1)
    right_data = data[:, :, frame_size:]
    right_data = torch.cat([torch.zeros_like(right_data[:, 0:1, :]), right_data], dim=1)
    # [:,:left_data.size(1), :]
    overlap_res = (left_data + right_data).reshape(data.size(0), -1)
    overlap_res[:, 256: -256] = overlap_res[:, 256: -256]/2
    return overlap_res

ar = np.arange(16384).reshape(1, 16384)
ar = torch.Tensor(ar)
ar = frames_data(ar, use_window=False)
out = overlap_data(ar, use_window=False)




def process_zly(input_data_data):

    speech_frame = []
    #window = torch.from_numpy(np.hamming(320).astype(np.float32)).reshape(1, 1, 320).cuda()
    window = torch.from_numpy(np.ones(320).astype(np.float32)).reshape(1, 1, 320).cuda()
    #window = np.ones(320)
    print('shape[1]', input_data_data.shape[1])
    print('整除', input_data_data.shape[1] // 320)
    for i in range(int(input_data_data.shape[1]//320)):  #  测试的时候送的是一条数据，所以
        print('看看有多少帧数：')
        print(int(len(input_data_data/320)))
        speech_frame.append(input_data_data[:, i * 320:(i + 1) * 320])

    speech_frame = torch.stack(speech_frame).permute(1, 0, 2)  # batch,帧数，采样点320
    frames = speech_frame * window



