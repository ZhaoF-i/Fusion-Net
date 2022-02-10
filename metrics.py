import soundfile as sf
from joblib import Parallel, delayed
# from pypesq import pesq  # 和matlab有0.005左右的差距  pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
# from pystoi import stoi  # pip install pystoi
from pathlib import Path
from tqdm import tqdm
import numpy as np

N_JOBS = 20
noise_type = ['babble', 'caffe']


class Metrics(object):
    def __init__(self, tt_path):
        self.tt_path = tt_path

    def getWavLst(self):
        lst = list(Path(self.tt_path).rglob('*_est.wav'))
        wavlst = [line.stem for line in lst]
        length = len(lst)
        return wavlst, length

    def getSnrLst(self, wavlst, length):
        snr = []
        idxs = []
        for i in range(length):
            name = wavlst[i]
            idx = [j for j in range(len(name)) if name[j] == '_']
            idxs.append(idx)
            snr.append(name[idx[1] + 1:idx[2]])
        SNR = list(set(snr))
        SNR.sort()
        return snr, SNR, idxs

    def compute_stoi(self, name, snr, idx):
        clean, sr = sf.read(name + 'clean.wav')
        mix, _ = sf.read(name + 'mix.wav')
        est, _ = sf.read(name + 'est.wav')

        snr = int(snr)
        noise = int(name.split('/')[-1][idx[2] + 1:idx[3]])
        stoi_mix = stoi(clean, mix, sr)
        stoi_es = stoi(clean, est, sr)

        return [snr, noise, stoi_mix, stoi_es]

    def compute_pesq(self, name, snr, idx):
        clean, sr = sf.read(name + 'clean.wav')
        mix, _ = sf.read(name + 'mix.wav')
        est, _ = sf.read(name + 'est.wav')

        snr = int(snr)
        noise = int(name.split('/')[-1][idx[2] + 1:idx[3]])
        pesq_mix = pesq(clean, mix, sr)
        pesq_es = pesq(clean, est, sr)

        return [snr, noise, pesq_mix, pesq_es]

    def forward(self):
        wavlst, length = self.getWavLst()
        snrlst, SNR, idxlst = self.getSnrLst(wavlst=wavlst, length=length)
        stois = Parallel(n_jobs=N_JOBS)(
            delayed(self.compute_stoi)(name=self.tt_path + wavlst[i][:-3], snr=snrlst[i], idx=idxlst[i])
            for i in tqdm(range(length)))
        pesqs = Parallel(n_jobs=N_JOBS)(
            delayed(self.compute_pesq)(name=self.tt_path + wavlst[i][:-3], snr=snrlst[i], idx=idxlst[i])
            for i in tqdm(range(length)))
        rs_stoi, rs_pesq = self.avgRes(stois=stois, pesqs=pesqs, SNR=SNR)
        # print
        self.print(rs_stoi, 'stoi')
        self.print(rs_pesq, 'pesq')

    def avgRes(self, stois, pesqs, SNR):
        rt_stoi = []
        rt_pesq = []
        for noise_idx in range(1, 3, 1):
            for snr in SNR:
                sts = np.array([m for m in stois if m[0] == int(snr) and m[1] == noise_idx])
                pes = np.array([m for m in pesqs if m[0] == int(snr) and m[1] == noise_idx])
                if len(sts):
                    avg_stoi = [np.mean(sts[:, -2]), np.mean(sts[:, -1])]
                    rt_stoi.append(avg_stoi)
                    avg_pesq = [np.mean(pes[:, -2]), np.mean(pes[:, -1])]
                    rt_pesq.append(avg_pesq)
                else:
                    rt_stoi.append([0.0, 0.0])
                    rt_pesq.append([0.0, 0.0])
        return rt_stoi, rt_pesq

    def print(self, list, type):
        print()
        print(type, end='\t')
        for noise_idx in range(1, 3, 1):
            for snr in range(-5, 10, 5):
                print('(' + noise_type[noise_idx - 1] + ',' + str(snr) + ')', end='\t')
        print()
        print('mix:', end='\t')
        for i in range(len(list)):
            print(round(list[i][0], 4), end='\t')
        print()
        print('es :', end='\t')
        for i in range(len(list)):
            print(round(list[i][1], 4), end='\t')


if __name__ == '__main__':
    path = ''
    metrics = Metrics(path)
