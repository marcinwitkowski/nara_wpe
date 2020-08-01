# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:12:44 2020

@author: piotr
"""


import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
from timeit import default_timer as timer

from nara_wpe.wpe import wpe
from nara_wpe.wpe import get_power
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root
stft_options = dict(size=512, shift=128)
frequency_bins = stft_options['size'] // 2 + 1
def aquire_audio_data():
    D, T = 4, 10000
    y = np.random.normal(size=(D, T))
    return y
y = aquire_audio_data()
Y = stft(y, **stft_options)
Y = Y.transpose(2, 0, 1)
stft(np.zeros(19), **stft_options).shape

Z = wpe(Y)
z_np = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
channels = 4
sampling_rate = 16000
delay = 3
iterations = 5
taps = 4
taps2 = [a for a in range(1,11)] + [a for a in range(11,21) if a % 2 == 0]
alpha=0.9999
file_template = 'AMI_WSJ20-Array1-{}_T10c0201.wav'
signal_list = [
    sf.read(str(project_root / 'data' / file_template.format(d + 1)))[0]
    for d in range(channels)
]
y = np.stack(signal_list, axis=0)
IPython.display.Audio(y[0], rate=sampling_rate)
Y = stft(y, **stft_options).transpose(2, 0, 1)
for i in range(len(taps2)):
    start = timer()
    Z = wpe(
        Y,
        taps=taps2[i],
        delay=delay,
        iterations=iterations,
        statistics_mode='full'
    ).transpose(1, 2, 0)
    end =  timer()
    print("Taps = {}, Time = {}".format(taps2[i], end - start))
z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])
IPython.display.Audio(z[0], rate=sampling_rate)