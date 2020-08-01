# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:48:30 2020

@author: piotr
"""


import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import time
from tqdm import tqdm

from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root
stft_options = dict(size=512, shift=128)
channels = 4
sampling_rate = 16000
delay = 3
alpha=0.9999
taps = 12
taps2 = [a for a in range(1,11)] + [a for a in range(11,21) if a % 2 == 0]
frequency_bins = stft_options['size'] // 2 + 1
file_template = 'AMI_WSJ20-Array1-{}_T10c0201.wav'
signal_list = [
    sf.read(str(project_root / 'data' / file_template.format(d + 1)))[0]
    for d in range(channels)
]
y = np.stack(signal_list, axis=0)
IPython.display.Audio(y[0], rate=sampling_rate)
Y = stft(y, **stft_options).transpose(1, 2, 0)
T, _, _ = Y.shape
for i in range(len(taps2)):
    print("\n{} Taps: ".format(taps2[i]))
    def aquire_framebuffer():
        buffer = list(Y[:taps2[i]+delay+1, :, :])
        for t in range(taps2[i]+delay+1, T):
            yield np.array(buffer)
            buffer.append(Y[t, :, :])
            buffer.pop(0)
    Z_list = []
    Q = np.stack([np.identity(channels * taps2[i]) for a in range(frequency_bins)])
    G = np.zeros((frequency_bins, channels * taps2[i], channels))
    
    for Y_step in tqdm(aquire_framebuffer()):
        Z, Q, G = online_wpe_step(
            Y_step,
            get_power_online(Y_step.transpose(1, 2, 0)),
            Q,
            G,
            alpha=alpha,
            taps=taps2[i],
            delay=delay
        )
        Z_list.append(Z)

Z_stacked = np.stack(Z_list)
z = istft(np.asarray(Z_stacked).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])

IPython.display.Audio(z[0], rate=sampling_rate)