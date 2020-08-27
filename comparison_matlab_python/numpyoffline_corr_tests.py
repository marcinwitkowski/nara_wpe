# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:12:44 2020

@author: piotr
"""


import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from srmrpy import srmr
import scipy.io
from cepsdist import cepsdist
from pesq import pesq
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft
from nara_wpe import project_root
stft_options = dict(size=512, shift=128, fading=False, pad=False)

# =============================================================================
# CGG with different parameters from matlab - time real audio dereverberated
# Change path to .mat files from root folder
# =============================================================================

mat_derever_05 = scipy.io.loadmat('/Users/piotr/time_derv_05.mat')['y2']
mat_derever_00 = scipy.io.loadmat('/Users/piotr/time_derv_wpe.mat')['y1']
mat_derever_19 = scipy.io.loadmat('/Users/piotr/time_derv_19.mat')['y2']
mat_derever_05_sim = scipy.io.loadmat('/Users/piotr/time_derv_simulated_05.mat')['y2']
mat_derever_05 = np.transpose(mat_derever_05,(1,0))
mat_derever_00 = np.transpose(mat_derever_00,(1,0))
mat_derever_19 = np.transpose(mat_derever_19,(1,0))
mat_derever_05_sim = np.transpose(mat_derever_05_sim,(1,0))

channels = 4
sampling_rate = 16000
delay = 2
iterations = 5
taps = 4
alpha=0.9999
file_template1 = 'AMI_WSJ20-Array1-{}_T10c0201.wav'
file_template2 = 'DR3_FEME0_SX335_x2_{}.wav'
file_template_dir = 'DR3_FEME0_SX335_x2_dir{}.wav'

# =============================================================================
# file_template = file_template1 for correlation test with real audio(matlab vs python)
# file_template = file_template2 for quality test with simulated audio(matlab vs python)
# =============================================================================

file_template = file_template1
signal_list = [
    sf.read(str(project_root / 'data' / file_template.format(d + 1)))[0]
    for d in range(channels)
]
signal_list_dir = [
    sf.read(str(project_root / 'data' / file_template_dir.format(d + 1)))[0]
    for d in range(channels)
]

y = np.stack(signal_list, axis=0)
y_dir = np.stack(signal_list_dir, axis=0)
IPython.display.Audio(y[0], rate=sampling_rate)
Y = stft(y, **stft_options).transpose(2, 0, 1)

Z1 = wpe(
    Y,
    taps=taps,
    delay=delay,
    iterations=iterations,
    statistics_mode='full',
    param=0
).transpose(1, 2, 0)
z1 = istft(Z1, size=stft_options['size'], shift=stft_options['shift'], fading=False)

Z2 = wpe(
    Y,
    taps=taps,
    delay=delay,
    iterations=iterations,
    statistics_mode='full',
    param=0.5
).transpose(1, 2, 0)
z2 = istft(Z2, size=stft_options['size'], shift=stft_options['shift'], fading=False)

Z3 = wpe(
    Y,
    taps=taps,
    delay=delay,
    iterations=iterations,
    statistics_mode='full',
    param=1.9
).transpose(1, 2, 0)
z3 = istft(Z3, size=stft_options['size'], shift=stft_options['shift'], fading=False)

z1[0] = z1[0]/np.sqrt((np.sum(z1[0]*z1[0])/len(z1[0])))
z2[0] = z2[0]/np.sqrt((np.sum(z2[0]*z2[0])/len(z2[0])))
z3[0] = z3[0]/np.sqrt((np.sum(z3[0]*z3[0])/len(z3[0])))

mat_derever_00[0] = mat_derever_00[0]/np.sqrt((np.sum(mat_derever_00[0]*mat_derever_00[0])/len(mat_derever_00[0])))
mat_derever_05[0] = mat_derever_05[0]/np.sqrt((np.sum(mat_derever_05[0]*mat_derever_05[0])/len(mat_derever_05[0])))
mat_derever_19[0] = mat_derever_19[0]/np.sqrt((np.sum(mat_derever_19[0]*mat_derever_19[0])/len(mat_derever_19[0])))

corr1 = np.correlate(z1[0],mat_derever_00[0],"same")
corr1 = corr1 / max(sum(z1[0]*z1[0]),sum(mat_derever_00[0]*mat_derever_00[0]))

corr2 = np.correlate(z2[0],mat_derever_05[0],"same")
corr2 = corr2 / max(sum(z2[0]*z2[0]),sum(mat_derever_05[0]*mat_derever_05[0]))

corr3 = np.correlate(z3[0],mat_derever_19[0],"same")
corr3 = corr3 / max(sum(z3[0]*z3[0]),sum(mat_derever_19[0]*mat_derever_19[0]))

if(file_template == file_template1):
    print("Corr1(WPE) = {}".format(max(corr1)))
    print("Corr2(WPE-CGG 0.5) max = {}".format(max(corr2)))
    print("Corr3(WPE-CGG 1.9) max = {}\n".format(max(corr3)))


if(file_template == file_template2):
    
    print("cepsdist ref: {}".format(cepsdist(y[0],y_dir[0],16000)[0]))
    print("pesq ref: {}".format(pesq(16000, y_dir[0], y[0], 'wb')))
    print("srmr ref: {}\n".format(srmr(y[0],16000, fast=False)[0]))
    
    print("cepsdist python data: {}".format(cepsdist(z2[0],y_dir[0],16000)[0]))
    print("pesq python data: {}".format(pesq(16000, y_dir[0], z2[0], 'wb')))
    print("srmr python data: {}\n".format(srmr(z2[0],16000, fast=False)[0]))
    
    print("cepsdist matlab data: {}".format(cepsdist(mat_derever_05_sim[0],y_dir[0],16000)[0]))
    print("pesq matlab data: {}".format(pesq(16000, y_dir[0], mat_derever_05_sim[0], 'wb')))
    print("srmr matlab data: {}\n".format(srmr(mat_derever_05_sim[0],16000, fast=False)[0]))
