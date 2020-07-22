import librosa
import matplotlib.pyplot as plt

import librosa.display


WAV_PATH = "/Users/sooyoungmoon/git/coretech/sampledata/t2_sample 2/file001_e.wav"

y, sr = librosa.load(WAV_PATH, sr=None)

def divide_list(l, sr, ti, stride):
    n = int(sr * ti)
    move = int(sr * stride)
    for i in range(0, len(l), move):
        yield l[i:i + n]

result = list(divide_list(y, sr, 1, 1))

a = result[6]
b = result[7]
c = result[8]
d = result[9]

mel_total = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# mel_total에서 컬럼 760~796 사이에 beep음 보일거예요! 

logmel = librosa.amplitude_to_db(abs(mel_total))
fig = plt.Figure()
ax = fig.add_subplot(111)
p = librosa.display.specshow(logmel)
plt.show()

mel_a = librosa.feature.melspectrogram(a, sr=sr, n_mels=128)
mel_b = librosa.feature.melspectrogram(b, sr=sr, n_mels=128)
mel_c = librosa.feature.melspectrogram(c, sr=sr, n_mels=128)
mel_d = librosa.feature.melspectrogram(d, sr=sr, n_mels=128)


