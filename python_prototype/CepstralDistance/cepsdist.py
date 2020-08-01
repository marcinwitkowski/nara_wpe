from math import sqrt
from realceps import realceps
import numpy as np


# assumption: input is horizontal

def cepsdist(x, y, fs=4800):
    # param values:
    param_cmn = "y"
    param_frame = 0.025
    param_shift = 0.01

    # Calculate number of frames
    if len(x) > len(y):
        x = x[:len(y)]
    else:
        y = y[:len(x)]

    # Normalization
    if param_cmn != "y":
        xDiv = sqrt(np.sum(np.square(x)))
        yDiv = sqrt(np.sum(np.square(y)))
        for v in range(len(x)):
            x[v] = x[v] / xDiv
            y[v] = y[v] / yDiv

    frame = np.fix(param_frame * fs)
    shift = np.fix(param_shift * fs)

    num_sample = len(x)
    num_frame = np.fix((num_sample - frame + shift) / shift)

    # Break up the signals into frames
    #Not sure how to make windowing, now this part is omitted, but has to be implemented
    win = np.hanning(frame)
    # X=x
    # Y=y

    # Apply the cepstrum analysis
    x = np.vstack(x)  # vertical
    y = np.vstack(y)
    ceps_x = realceps(x)
    ceps_y = realceps(y)

    param_order = 24  # default param in "score_sim"

    ceps_x = ceps_x[1:param_order + 1]
    ceps_y = ceps_y[1:param_order + 1]

    # Perform cepstral mean normalization
    if param_cmn == "y":
        mean_ceps_x = np.mean(ceps_x, 0)
        mean_ceps_y = np.mean(ceps_y, 0)
        for i in range(len(ceps_x)):
            ceps_x[i] = ceps_x[i] - mean_ceps_x
            ceps_y[i] = ceps_y[i] - mean_ceps_y

    # Calculate the cepstral distances
    err = np.empty(len(ceps_x))
    for i in range(len(ceps_x)):
        err[i] = (np.square((ceps_x[i] - ceps_y[i])))

    errsum = np.sum(err[2: len(err)])

    ds = np.empty(len(err))
    for i in range(len(err)):
        ds[i] = 10 / np.log(10) * np.sqrt(2 * errsum + err[i])

    for i in range(len(ds)):  # min, max
        if ds[i] > 10:
            ds[i] = 10
        elif ds[i] < 0:
            ds[i] = 0

    d = np.mean(ds)
    e = np.median(ds)
