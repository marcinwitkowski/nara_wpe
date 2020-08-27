from scipy import signal
from scipy import fftpack
from numpy.ma import floor
import numpy as np


def istft_kkmw(X ,M,N,R):
    y = 1
    X =np.concatenate((X[0,:,:],np.conjugate(X[0,np.shape(X)[1]-2:0:-1])))
    sizeX3=1
    out = np.zeros((M,1,sizeX3))
    x= np.zeros((np.shape(X)[1]*R,sizeX3))

    win = np.transpose(np.matlib.repmat(signal.tukey(M), 1, 1))

    for kk in range(0, np.shape(X)[1]):
        xt = fftpack.ifft(X[:,kk],N,0)
       # X[:, kk, 0:nr_mics] = Xt[0:int(N_out)]

#Not finished