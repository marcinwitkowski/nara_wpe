
from fwsegsnr import fwsegsnr
import soundfile as sf

data, Fs = sf.read('speech.wav')
data2, Fs2 = sf.read('speech_bab_0dB.wav')

print(fwsegsnr(data, data2, Fs))

