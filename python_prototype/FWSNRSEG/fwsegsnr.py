import math
import scipy.signal
import matplotlib.pyplot as plot

import numpy as np


def fwsegsnr(x, y, fs=4800):
    param_frame = 0.025
    param_shift = 0.01
    param_numband = 23

    # Normalization
    print(x)
    xDiv = np.sqrt(np.sum(np.square(x),axis=1))
    yDiv = np.sqrt(np.sum(np.square(y),axis=1))

    x,x2,x3,x4=scipy.linalg.lstsq(x,xDiv)
    y,y2,y3,y4=scipy.linalg.lstsq(y,yDiv)       #mrdivide in matlab

    # STFT
    frame = np.fix(param_frame * fs)
    shift = np.fix(param_shift * fs)
    win = np.hanning(frame)                 #Cannot use win in plot (error with dimensions)
    noverlap = frame - shift
    fftpt = 2 ** int(np.ceil(math.log2(frame)))
    powerSpectrum_x, freqencies_x, time_x, imageAxis_x = plot.specgram(x, Fs=fs, noverlap=noverlap, NFFT=fftpt )
    plot.show()
    powerSpectrum_y, freqencies_y, time_y, imageAxis_y = plot.specgram(x, Fs=fs, noverlap=noverlap,
                                                                       NFFT=fftpt)

