# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:39:48 2020

@author: piotr
"""

import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import time
from tqdm import tqdm
import tensorflow as tf

from nara_wpe.tf_wpe import wpe
from nara_wpe.tf_wpe import online_wpe_step, get_power_online
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root

stft_options = dict(
    size=512,
    shift=128,
    window_length=None,
    fading=True,
    pad=True,
    symmetric_window=False
)
channels = 4
sampling_rate = 16000
delay = 3
alpha=0.99
taps = 10
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

def aquire_framebuffer():
    buffer = list(Y[:taps+delay+1, :, :])
    for t in range(taps+delay+1, T):
        yield np.array(buffer)
        buffer.append(Y[t, :, :])
        buffer.pop(0)
Z_list = []

Q = np.stack([np.identity(channels * taps) for a in range(frequency_bins)])
G = np.zeros((frequency_bins, channels * taps, channels))

with tf.Session() as session:
    Y_tf = tf.placeholder(tf.complex128, shape=(taps + delay + 1, frequency_bins, channels))
    Q_tf = tf.placeholder(tf.complex128, shape=(frequency_bins, channels * taps, channels * taps))
    G_tf = tf.placeholder(tf.complex128, shape=(frequency_bins, channels * taps, channels))
    
    results = online_wpe_step(Y_tf, get_power_online(tf.transpose(Y_tf, (1, 0, 2))), Q_tf, G_tf, alpha=alpha, taps=taps, delay=delay)
    for Y_step in tqdm(aquire_framebuffer()):
        feed_dict = {Y_tf: Y_step, Q_tf: Q, G_tf: G}
        Z, Q, G = session.run(results, feed_dict)
        Z_list.append(Z)

Z_stacked = np.stack(Z_list)
z = istft(np.asarray(Z_stacked).transpose(2, 0, 1), size=stft_options['size'], shift=stft_options['shift'])

IPython.display.Audio(z[0], rate=sampling_rate)