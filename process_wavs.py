import os
from nara_wpe.utils import istft, stft
from nara_wpe.wpe import wpe
import soundfile as sf
import numpy as np
import time
from tqdm import tqdm


def wpe_mock(y):
    return y

def process_wav(wavpath):
    #Setup
    channels = 1
    #sampling_rate = 16000
    delay = 3
    iterations = 20
    taps = 15
    #alpha=0.9999
    stft_options = dict(size=512, shift=160)
    
    y, sr = sf.read(wavpath);
    #print(np.shape(y))
    Y = stft(y, **stft_options)#.transpose(2,0, 1)
    Y = Y.reshape(Y.shape[0], Y.shape[1],1)
    Y = Y.transpose(2,0,1)
    D,T,F = Y.shape
    X = np.copy(Y)
    #for f in tqdm(range(F), total=F):
    for f in range(F):
        #taps = get_taps(f)
        X[ :,:, f] = wpe(
            Y[:, :, f],
            taps=taps,
            delay=delay,
            iterations=iterations
    )
    X = X.transpose(1, 2, 0)
    x = istft(X.reshape(X.shape[0], X.shape[1]),
        size=stft_options['size'], shift=stft_options['shift'])
    dirsplit = os.path.split(wavpath)
    output_dir = os.path.join('nara_wpe_orig',dirsplit[0])
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir,dirsplit[1]), x, sr)
    return np.size(x)/sr


wav_main_paths = ['sid_dev', 'sid_eval']

# make filenamelist
src_wavpaths = {}
for wp in wav_main_paths:
    src_wavpaths[wp] = []
    for root, dirs, files in os.walk(wp):
        for f in files:
            if f.lower().endswith('.wav'):
                src_wavpaths[wp].append(os.path.join(root, f))
    if not src_wavpaths[wp]:
        raise Exception('No wavs found in directory \'{}\''.format(wp))


for wp in wav_main_paths:
    N = len(src_wavpaths[wp])
    print(wp)
    #start = time.time()
    for f in tqdm(src_wavpaths[wp], total=N):
        process_wav(f)
    #end = time.time()
        #print('{} - processed in {} s'.format(f,np.round(end - start,3)))
