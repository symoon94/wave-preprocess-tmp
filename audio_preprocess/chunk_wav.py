from pydub import AudioSegment
from pydub.utils import make_chunks
import glob
from pathlib import Path

filedir = "../data/"  # the directory path where wave files exist
outdir = "../chunkdata/wav/"

wavelist = glob.glob("../data/*") # ['../data/02.wav', '../data/03.wav']

chunk_length_ms = 1000 # pydub calculates in millisec

for wave in wavelist:
    myaudio = AudioSegment.from_file(wave , "wav") 
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    subdir = wave.lstrip(filedir).rstrip(".wav")
    # subpath = outdir + subdir
    # Path(subpath).mkdir(parents=True, exist_ok=True)
    #Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = outdir + "/" + subdir + "_chunk{0}.wav".format(str(i).zfill(2))
        chunk.export(chunk_name, format="wav")

