# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:28:17 2020

@author: piotr
"""


import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
from timeit import default_timer as timer

from nara_wpe.tf_wpe import wpe
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe import project_root
stft_options = dict(size=512, shift=128)
def aquire_audio_data():
    D, T = 4, 10000
    y = np.random.normal(size=(D, T))
    return y
y = aquire_audio_data()
Y = stft(y, **stft_options).transpose(2, 0, 1)
with tf.Session() as session:
    Y_tf = tf.placeholder(
        tf.complex128, shape=(None, None, None))
    Z_tf = wpe(Y_tf)
    Z = session.run(Z_tf, {Y_tf: Y})
z_tf = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
channels = 4
sampling_rate = 16000
delay = 3
iterations = 10
taps = 5
file_template = 'AMI_WSJ20-Array1-{}_T10c0201.wav'
signal_list = [
    sf.read(str(project_root / 'data' / file_template.format(d + 1)))[0]
    for d in range(channels)
]
y = np.stack(signal_list, axis=0)
IPython.display.Audio(y[0], rate=sampling_rate)
Y = stft(y, **stft_options).transpose(2, 0, 1)
from nara_wpe.tf_wpe import get_power
with tf.Session()as session:
    Y_tf = tf.placeholder(tf.complex128, shape=(None, None, None))
    Z_tf = wpe(Y_tf, taps=taps, iterations=iterations)
    start = timer()
    Z = session.run(Z_tf, {Y_tf: Y})
    end = timer()
z = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
print(end - start)
IPython.display.Audio(z[0], rate=sampling_rate)