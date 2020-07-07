import logging
import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd 
import numpy as np
import glob
from keras_preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt

import re

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


metadata=pd.read_csv('../metadata/audio_metadata.csv',dtype=str)
imgpath = '../chunkdata/img/*.png'

traindf = {'ID':[], 'class':[]}

# traindf['ID']에 200ms 이미지들의 파일 경로 리스트를 넣어줌
imgs = glob.glob(imgpath)
tmp = []
for img in imgs:
    if int(img.split("chunk")[-1].rstrip('.png')) < 200:    
        tmp.append(img)
tmp.sort(key=natural_keys)
traindf['ID'] += tmp

import ipdb; ipdb.set_trace()
for index, row in metadata.iterrows():
    ID = row['file_name']
    if len(ID) == 1:
        ID = "0" + ID
    
    start = int(row['start']) * 5
    end = int(row['end']) * 5
    tmp = np.zeros(200)
    if end >= 200:
        tmp[start:] = 1
    elif end != 0:
        tmp[start:end+1] = 1
    traindf['class'] += list(tmp)

import ipdb; ipdb.set_trace()
traindf = pd.DataFrame(data=traindf)
traindf['class'] = traindf['class'].astype(str)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=None,
    x_col="ID",
    y_col="class",
    subset="training",
    batch_size=1,
    class_mode="categorical",
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=None,
    x_col="ID",
    y_col="class",
    subset="validation",
    batch_size=1,
    class_mode="categorical",
    target_size=(64,64))

import ipdb; ipdb.set_trace()

input_shape = (train_generator.shape[1], train_generator.shape[2])
print("Build LSTM RNN model ...")
model = Sequential()

model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=2, activation="softmax"))

print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()