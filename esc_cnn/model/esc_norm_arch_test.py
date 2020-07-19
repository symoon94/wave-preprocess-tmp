import os
import glob
import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import add, Lambda, LeakyReLU, SpatialDropout2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from random import shuffle
from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling2D
import keras.backend as K

from keras.regularizers import l2


# 파일경로
wavFiles = glob.glob("../ESC-50-master/audio/*.wav")
table = pd.read_csv("../ESC-50-master/meta/esc50.csv")

# mel-spec 변수
sr = 44100
n_mels = 256
frame_length = 0.030
frame_stride = 0.020

# training 변수
batch_size = 24
spatial_dropout_rate_1 = 0.07
spatial_dropout_rate_2 = 0.14
l2_rate = 0.0005
learning_rate = 1e-4
epochs = 1000

# model type
model_type = 'CNN_only'


def melS(wav_file, n_mels, sr):
    ### wav file to mel-spectrogram 
    y, sr = librosa.load(wav_file, sr=sr)
    # Normalize audio data between -1 and 1
    y = librosa.util.normalize(y)

    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=n_mels, n_fft=input_nfft, hop_length=input_stride)
    # S = librosa.power_to_db(S, ref=np.max)

    # Convert sound intensity to log amplitude:
    S = librosa.amplitude_to_db(abs(S))

    # Normalize between -1 and 1
    S = librosa.util.normalize(S)

    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))

    return S

esc10Table = table.loc[table.esc10 == True]
esc10WavFiles = []
for path in wavFiles:
    fName = os.path.basename(path)
    esc10Lists = esc10Table.filename.values.astype(np.str)
    if fName in esc10Lists:
        esc10WavFiles.append(path)

# 이미지 저장
for f in esc10WavFiles: 
  mel_spec = melS(f, n_mels, sr)
  directory = os.path.dirname(f)
  id = os.path.basename(f).split('.')[0]
  print (os.path.basename(f))
  target_size = mel_spec.shape
  plt.imsave("%s/norm_mel_S/%s.jpg"%(directory, id), mel_spec)
  plt.imshow(mel_spec)

nclass = len(esc10Table.target.unique())
esc10MelFiles = glob.glob("../ESC-50-master/audio/norm_mel_S/*.jpg")
esc10MelFiles = sorted(esc10MelFiles)
esc10Table['path'] = esc10MelFiles

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=esc10Table,
    directory=None,
    x_col="path",
    y_col="category",
    subset="training",
    batch_size=batch_size,
    seed=42,
    shuffle=False,
    class_mode="categorical",
    color_mode="grayscale",
    target_size=target_size)

valid_generator = datagen.flow_from_dataframe(
    dataframe=esc10Table,
    directory=None,
    x_col="path",
    y_col="category",
    subset="validation",
    batch_size=batch_size,
    seed=42,
    shuffle=False,
    class_mode="categorical",
    color_mode="grayscale",
    target_size=target_size)

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

if model_type == 'CNN_only':
  # Create a secquential object
    model = Sequential()
  # Conv 1
    model.add(Conv2D(filters=32, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate), 
                     input_shape=(256,250,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=32, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    # Max Pooling #1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=64, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_2))
    model.add(Conv2D(filters=64, 
                     kernel_size=(3,3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
   
    # Reduces each h×w feature map to a single number by taking the average of all h,w values.
    model.add(GlobalAveragePooling2D())

    # Softmax output
    model.add(Dense(10, activation='softmax'))

    adam = optimizers.Adam(lr=learning_rate, beta_1=0.99, beta_2=0.999)
    model.compile(
    loss='categorical_crossentropy', 
    metrics=['accuracy'], 
    optimizer=adam)


model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=epochs, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID, max_queue_size=60)

