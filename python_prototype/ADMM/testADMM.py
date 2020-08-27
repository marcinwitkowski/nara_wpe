import scipy.io
import soundfile as sf
import numpy as np
from ADMM import ADMM

Nmic=4
data=np.empty([127523, Nmic])
for v in range(1, Nmic+1):
    [data[:,v-1],fs]=sf.read('AMI_WSJ20-Array1-'+str(v)+'_T10c0201.wav')

ckMatlab = scipy.io.loadmat('ckMatlab.mat')['ck']
dkMatlab = scipy.io.loadmat('dkMatlab.mat')['dk']

[dk,ck] = ADMM(data,fs,Nmic)

cktest=np.amax(ck-ckMatlab)
dktest=np.amax(dk[:,:,0]-dkMatlab)

#To add:
#IFFT
#STFT plots
#Metrics calculations
#More arrays test
#Tests for other voice samples (?)
#Integration with Nara_WPE code

print("Ck test: ",cktest)
print("Dk test: ",dktest)