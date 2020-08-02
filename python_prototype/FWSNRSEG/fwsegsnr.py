import math
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np


# Implemented parts: Normalization, STFT (without hanning window), SNR Calculation
# Missing parts: window in STFT, Mel-scale frequency warping

def fwsegsnr(x, y, fs=4800):
    param_frame = 0.025
    param_shift = 0.01
    param_numband = 23

    # Normalization
    xDiv = np.sqrt(np.sum(np.square(x)))
    yDiv = np.sqrt(np.sum(np.square(y)))

    #x, x2, x3, x4 = scipy.linalg.lstsq(x, xDiv)
    #x=np.vstack(x)
    x=x/xDiv
    y=y/yDiv
    #y, y2, y3, y4 = scipy.linalg.lstsq(y, yDiv)  # mrdivide in matlab

    # STFT
    frame = np.fix(param_frame * fs)
    shift = np.fix(param_shift * fs)
    win = np.hanning(frame)  # Cannot use win in plot (error with dimensions)

    noverlap = frame - shift
    fftpt = 2 ** int(np.ceil(math.log2(frame)))

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    powerSpectrum_x, freqencies_x, time_x, imageAxis_x = ax1.specgram(x, Fs=fs, noverlap=noverlap, NFFT=fftpt)
    powerSpectrum_y, freqencies_y, time_y, imageAxis_y = ax2.specgram(x, Fs=fs, noverlap=noverlap, NFFT=fftpt)
    plt.show()

    X = np.abs(powerSpectrum_x)
    Y = np.abs(powerSpectrum_y)

    # Mel Scale frequency warping -> I omit this part, but it should be probably implemented
    #
    #
    #

    # Calculate SNR
    W = np.power(Y, 0.2)
    E = X - Y + 0.1
    dssum1 = np.sum(np.log10(np.divide((np.square(Y)), (np.square(E)))), axis=1)  # Jukic
    ds = np.divide(10 * dssum1, np.sum(W, axis=1))

    for i in range(len(ds)):  # min, max
        if ds[i] > 35:
            ds[i] = 35
        elif ds[i] < -10:
            ds[i] = -10

    d = np.mean(ds)
    e = np.median(ds)
