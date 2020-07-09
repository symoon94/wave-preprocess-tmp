import os
import glob
import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D, Reshape, LSTM, add, concatenate, Lambda
from sklearn.model_selection import train_test_split
from random import shuffle
from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Permute, Bidirectional
import keras.backend as K
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed


def melS(wav_file, n_mels, sr):
    ### wav file to mel-spectrogram 
    y, sr = librosa.load(wav_file, sr=sr)
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=n_mels, n_fft=input_nfft, hop_length=input_stride)
    S = librosa.power_to_db(S, ref=np.max)
    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))

    return S



wavFiles = glob.glob("../ESC-50-master/audio/*.wav")
table = pd.read_csv("../ESC-50-master/meta/esc50.csv")


esc10Table = table.loc[table.esc10 == True]
esc10WavFiles = []
for path in wavFiles:
    fName = os.path.basename(path)
    esc10Lists = esc10Table.filename.values.astype(np.str)
    if fName in esc10Lists:
        esc10WavFiles.append(path)



sr = 44100
n_mels = 256
frame_length = 0.030
frame_stride = 0.020

# 이미지 저장
# for f in esc10WavFiles: 
#   mel_spec = melS(f, n_mels, sr)
#   directory = os.path.dirname(f)
#   id = os.path.basename(f).split('.')[0]
#   print (os.path.basename(f))
#   plt.imsave("%s/mel_S/%s.jpg"%(directory, id), mel_spec)
#   plt.imshow(mel_spec)



nclass = len(esc10Table.target.unique())
esc10MelFiles = glob.glob("../ESC-50-master/audio/mel_S/*.jpg")
esc10MelFiles = sorted(esc10MelFiles)
esc10Table['path'] = esc10MelFiles


batch_size = 32
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)

train_generator=datagen.flow_from_dataframe(
    dataframe=esc10Table,
    directory=None,
    x_col="path",
    y_col="category",
    subset="training",
    batch_size=1,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(250,256))

valid_generator=datagen.flow_from_dataframe(
    dataframe=esc10Table,
    directory=None,
    x_col="path",
    y_col="category",
    subset="validation",
    batch_size=1,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(250,256))


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


# model_type = 'CNN_only'
model_type = '2Dconv+LSTM'
# model_type = 'TD+BiLSTM'

if model_type == 'CNN_only':
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same',
                  input_shape=(250,256,3)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.1))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.1))
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.1))
  #model.add(Conv2D(64, (3, 3)))
  #model.add(BatchNormalization())
  #model.add(Activation('relu'))
  #model.add(MaxPooling2D(pool_size=(2, 2)))
  #model.add(Dropout(0.1))
  model.add(Conv2D(128, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  #model.add(Dense(128))
  #model.add(Activation('relu'))
  #model.add(Dropout(0.2))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(10, activation='softmax'))

  model.summary()
  model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

elif model_type == '2Dconv+LSTM':
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same',
                  input_shape=(250,256,3)))
  model.add(Conv2D(32, (3, 3)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.1))
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.1))
  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(BatchNormalization())
  
  frequency_axis = 1
  time_axis = 2
  channel_axis = 3

  model.add(Permute((time_axis, frequency_axis, channel_axis)))
  resize_shape = model.output_shape[2] * model.output_shape[3]
  model.add(Reshape((model.output_shape[1], resize_shape)))
  ## recurrent layer
  model.add(GRU(32, return_sequences=True))
  model.add(GRU(32, return_sequences=False))
  #model.add(Bidirectional(LSTM(256, return_sequences=True)))
  #model.add(Bidirectional(LSTM(256, return_sequences=True)))
  model.add(Dropout(0.3))
  ## output layer
  model.add(Dense(nclass, activation='softmax'))
  model.summary()
  model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

elif model_type == 'TD+BiLSTM':
  initialiser = 'glorot_uniform'
  input_shape=(1,250,256,3)
  model = Sequential()
  model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same', kernel_initializer=initialiser), input_shape=input_shape))
  model.add(TimeDistributed(BatchNormalization()))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser)))
  model.add(TimeDistributed(BatchNormalization()))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
  # model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer="glorot_uniform")))
  # model.add(TimeDistributed(BatchNormalization()))
  # model.add(TimeDistributed(Activation('relu')))
  # model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer="glorot_uniform")))
  # model.add(TimeDistributed(BatchNormalization()))
  # model.add(TimeDistributed(Activation('relu')))
  # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

  model.add(TimeDistributed(Flatten()))
  model.add(LSTM(64, return_sequences=False, dropout=0.5))
  model.add(Dense(nclass, activation='softmax'))
  
  model.summary()
  model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

## Validation
import ipdb; ipdb.set_trace()
model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=200,
                    validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                   use_multiprocessing=True, workers=8, max_queue_size=60,
                    callbacks=[ModelCheckpoint("baseline_cnn_mel.h5", monitor="val_acc", save_best_only=True),
                               EarlyStopping(patience=5, monitor="val_acc")])

a = plt.imread(esc10MelFiles[0])
