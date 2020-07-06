import librosa
import numpy as np
import librosa.display
import glob
from pathlib import Path
import matplotlib.pyplot as plt

filedir = glob.glob("../chunkdata/wav/*.wav")
outdir = "../chunkdata/img/"

# for wavdir in filedir:
#     chunks = glob.glob(wavdir + "/*")
for chunk in filedir:
    subdir = outdir + Path(chunk).parent.name
    outfile = Path(chunk).name.rstrip("wav") + "png"
    # if not Path(subdir).is_dir():
    y, sr = librosa.load(chunk)
    S = librosa.feature.melspectrogram(y=y, sr=44100, n_mels=32,
                                    fmax=16000)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    Path(subdir).mkdir(parents=True, exist_ok=True)

    plt.interactive(False)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    librosa.display.specshow(S_dB)
    plt.savefig(outdir + "/" + outfile, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')