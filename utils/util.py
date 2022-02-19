import numpy as np
import os
import re
import torch
from torch.autograd import Variable
import torch.nn.functional as F



def convert_to_frame(batch_wav, hop_size=256, use_window=True):

    # hamming_win = torch.from_numpy(np.hamming(hop_size * 2).astype(np.float32)).reshape(1, 1, hop_size * 2)
    frame_speech = batch_wav.reshape(batch_wav.size(0), -1, hop_size)
    frame_speech2 = frame_speech.clone()[:, 1:]
    frames = torch.cat([frame_speech[:, :-1], frame_speech2[:, :]], dim=-1)
    # if use_window is True:
    #     frames = frames * hamming_win
    return frames

def overlap_add(data, frame_size=256, use_window=True):
    # hamming_win = torch.from_numpy((np.hamming(frame_size * 2)).astype(np.float32)).reshape(1, 1, frame_size * 2).cuda()
    # if use_window is True:
    #     data = data * hamming_win

    left_data = data[:, :, :frame_size]
    left_data = torch.cat([left_data, torch.zeros_like(left_data[:, -1:, :])], dim=1)
    right_data = data[:, :, frame_size:]
    right_data = torch.cat([torch.zeros_like(right_data[:, 0:1, :]), right_data], dim=1)
    # [:,:left_data.size(1), :]
    overlap_res = (left_data + right_data).reshape(data.size(0), -1)
    overlap_res[:, frame_size: -frame_size] = overlap_res[:, frame_size: -frame_size] / 2
    return overlap_res

def expandWindow(data, left, right):
    data = data.detach().cpu().numpy()
    sp = data.shape
    idx = 0
    exdata = np.zeros([sp[0], sp[1], sp[2] * (left + right + 1)])
    for i in range(-left, right+1):
        exdata[:, :, sp[2] * idx : sp[2] * (idx + 1)] = np.roll(data, shift=-i, axis=1)
        idx = idx + 1
    return Variable(torch.FloatTensor(exdata)).cuda(CUDA_ID[0])

def context_window(data, left, right):
    sp = data.data.shape
    exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
    for i in range(1, left + 1):
        exdata[:, i:, sp[2] * (left - i) : sp[2] * (left - i + 1)] = data.data[:, :-i,:]
    for i in range(1, right+1):
        exdata[:, :-i, sp[2] * (left + i):sp[2]*(left+i+1)] = data.data[:, i:, :]
    exdata[:, :, sp[2] * left : sp[2] * (left + 1)] = data.data
    return Variable(exdata)
def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig
def gen_list(wav_dir, append):
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l

def write_log(file,name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')

def makedirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def saveYAML(yaml,save_path):
    f_params = open(save_path, 'w')
    for k, v in yaml.items():
        f_params.write('{}:\t{}\n'.format(k, v))