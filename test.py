import torch 
import torchaudio

# waveform, sample_rate = torchaudio.load('data/01.wav', normalization=True)
# mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)  # (channel, n_mels, time)

import ipdb; ipdb.set_trace()

import librosa
import numpy as np

sample_data,sr =librosa.load( 'data/01.wav')

frame_length = 0.025
frame_stride = 0.010

def Mel_S(wav_file):
    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr=16000)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
    
    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))
    return S

d1 = Mel_S('data/01.wav')
d2 = Mel_S('data/02.wav')

d1_ = librosa.amplitude_to_db(d1)
d2_ = librosa.amplitude_to_db(d2)