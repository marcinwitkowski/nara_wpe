import numpy as np
from scipy.io.wavfile import read, write
from fwsegsnr import fwsegsnr


Fs, data = read('test.wav')
data=np.array(data)
data=np.transpose(data)

#fwsegsnr(data, data2, Fs)

