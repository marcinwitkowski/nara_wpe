#Calculation of Cepstral Distance, written and tested according to matlab implementation
# assumption: input is horizontal, x and y are one dimensional

from math import sqrt
from realceps import realceps
import numpy as np
import numpy.matlib

def cepsdist(x, y, fs):
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
    # Not sure how to make windowing, now this part is omitted, but has to be implemented
    win = np.hanning(frame)

    idx = numpy.matlib.repmat(np.vstack(range(1, int(frame + 1))), 1, int(num_frame)) + \
          numpy.matlib.repmat(range(0, int(num_frame)) * shift, int(frame), 1)

    idx=idx.astype(int)
    x_idx=x[idx]
    y_idx=y[idx]

    X = x_idx*win[:,np.newaxis]
    Y = y_idx*win[:,np.newaxis]

    # Apply the cepstrum analysis
    X = np.vstack(X)  # vertical
    Y = np.vstack(Y)
    ceps_x = realceps(X)
    ceps_y = realceps(Y)

    param_order = 24  # default param in "score_sim"

    ceps_x = ceps_x[0:param_order + 1]
    ceps_y = ceps_y[0:param_order + 1]

    # Perform cepstral mean normalization
    if param_cmn == "y":
        mean_ceps_x = np.vstack(np.mean(ceps_x, 1))
        mean_ceps_y = np.vstack(np.mean(ceps_y, 1))
        ceps_x = ceps_x-mean_ceps_x
        ceps_y = ceps_y - mean_ceps_y

    # Calculate the cepstral distances
    err = np.square(ceps_x - ceps_y)
    ds = 10 / np.log(10) *np.sqrt((2 * np.sum(err[1: len(err)],axis=0))+np.hstack(err[0]))

    for i in range(len(ds)):  # min, max
        if ds[i] > 10:
            ds[i] = 10
        elif ds[i] < 0:
            ds[i] = 0

    d = np.mean(ds)
    e = np.median(ds)
    return [d, e]
