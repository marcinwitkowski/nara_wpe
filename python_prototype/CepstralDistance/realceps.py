#Calculation of Cepstral Distance, written and tested according to matlab implementation
# assumption: input is horizontal, x and y are one dimensional
import math
import numpy as np

#assumption: input is vertical
def realceps(x, flr=-100):
    # Calc power spectra of input frames
    xLength = len(x)
    NextPow2 = int(np.ceil(math.log2(xLength)))
    pt = 2 ** NextPow2
    xFFT = np.fft.fft(x,pt,axis=0)  #vertical

    Px = np.absolute(xFFT)

    # Flooring
    flr = np.amax(Px.all()) * (10 ** (flr / 20))
    for i in range(len(Px)):
        for j in range(len(Px[0])):
            if Px[i][j] < flr:
                Px[i][j] = flr

    # Calc cepstral coeffs
    c = np.real(np.fft.ifft(np.log(Px),axis=0))
    return c
