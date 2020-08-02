from cepsdist import cepsdist
import soundfile as sf

data, Fs = sf.read('speech.wav')
data2, Fs2 = sf.read('speech_bab_0dB.wav')

print(cepsdist(data, data2,Fs))