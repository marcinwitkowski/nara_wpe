from math import sqrt
from realceps import realceps
import numpy as np

#assumption: input is horizontal
def cepsdist(x, y, fs=4800):
    #Calculate number of frames - works
    if len(x) > len(y):
        x = x[:len(y)]
    else:
        y = y[:len(x)]

    #Normalization - works
    xDiv = sqrt(np.sum(np.square(x)))
    yDiv= sqrt(np.sum(np.square(y)))
    for v in range(len(x)):
        x[v] = x[v] / xDiv
        y[v] = y[v] / yDiv

    param_frame=0.025 #default param in "score_sim"
    param_shift=0.01  #default param in "score_sim"
    frame = np.fix(param_frame*fs)
    shift = np.fix(param_shift*fs)

    num_sample = len(x)
    num_frame = np.fix((num_sample-frame+shift)/shift)

    #Break up the signals into frames -> Need to find how to replace bsxfun in python
    #win = np.hanning(frame)
    #X=x
    #Y=y


    #Apply the cepstrum analysis
    x=np.vstack(x)          #vertical
    y=np.vstack(y)
    ceps_x = realceps(x)
    ceps_y = realceps(y)

    param_order = 24    #default param in "score_sim"

    ceps_x=ceps_x[1:param_order+1]
    ceps_y=ceps_y[1:param_order+1]


    #Perform cepstral mean normalization ->Need to find how to replace bsxfun in python


    #Calculate the cepstral distances








