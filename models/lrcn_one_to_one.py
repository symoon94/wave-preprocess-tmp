'''
200ms 이미지를 timestep 1로 해서 -> lrcn -> 폭력/비폭력 classification
TODO:
- autoencoder로 representation learning 하고나서
data -> encoder -> feature 뽑아내서 그 모아둔 feature을 lstm에 태우기
'''

import logging
import os

from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM

from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam

from keras_preprocessing.image import ImageDataGenerator

import pandas as pd 
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

import re

metadata = pd.read_csv('../metadata/audio_metadata.csv',dtype=str)
imgpath = '../chunkdata/img/*.png'
# TODO: 더 효율적으로 input 넣어주는 방법 쓰기.
# image로 변환 후 넣기보다는 training 하면서 wav -> input matrix로 바로 바꿔서 넣어준다거나.


# 숫자 들어가 있는 string 형태인 파일명 sort하기 위한 함수들.
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]


# data set 만들기.
def create_dataframe(imgpath):
    '''
    TODO: 
    one to one lstm 말고 many to one을 위한 학습 데이터셋 만들기

    traindf['ID']: a list of lists where each has five 200ms images.
    ex) [[chunk1, chunk2 ..., chunk5], [chunk2, chunk3 ..., chunk4], ..., [chunk100, chunk101, ..., chunk104]]
    ex) chunk104개 까지 있을 때 -> number of ID: 100
    class: list ex) [0, 0, 0, 1, ...]
    ex) chunk104개 까지 있을 때 -> number of class: 100
    '''

    traindf = {'ID':[], 'class':[]} # a list of all the images
    # traindf['ID']에 200ms 이미지들의 파일 경로 리스트를 넣어줌

    # TODO: 40초 이하의 음원도 이용할 수 있게 만들기
    imgs = glob.glob(imgpath)
    tmp = [] # a list of 5 chunks
    for img in imgs:
        if int(img.split("chunk")[-1].rstrip('.png')) < 200: # 40초이상의 음원데이터를 40초까지만 취급
            tmp.append(img)
    tmp.sort(key=natural_keys)
    traindf['ID'] += tmp

    for index, row in metadata.iterrows():
        ID = row['file_name']
        if len(ID) == 1:
            ID = "0" + ID  # '1' -> '01', '2' -> '02'
        
        start = int(row['start']) * 5
        end = int(row['end']) * 5
        tmp = np.zeros(200)
        if end >= 200:
            tmp[start:] = 1
        elif end != 0:
            tmp[start:end+1] = 1
        traindf['class'] += list(tmp)

    traindf = pd.DataFrame(data=traindf)
    traindf['class'] = traindf['class'].astype(str)

    return traindf


def create_data_generator(train_dataframe):
    datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.25)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_dataframe,
        directory=None,
        x_col="ID",
        y_col="class",
        subset="training",
        batch_size=1,
        class_mode="categorical",
        target_size=(1,64,64))

    valid_generator = datagen.flow_from_dataframe(
        dataframe=train_dataframe,
        directory=None,
        x_col="ID",
        y_col="class",
        subset="validation",
        batch_size=1,
        class_mode="categorical",
        target_size=(1,64,64))
    
    return train_generator, valid_generator


def lrcn(train_generator, valid_generator):
    input_shape = (1, 64, 64, 3)
    initialiser = 'glorot_uniform'

    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same', kernel_initializer=initialiser), input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer=initialiser)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer="glorot_uniform")))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer="glorot_uniform")))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(2, activation='softmax'))

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    model.fit_generator(generator=train_generator,
                        steps_per_epoch=200,
                        validation_data=valid_generator,
                        validation_steps=200,
                        epochs=10
    )


# TODO:
# test 하는 부분 코드 짜기.


def main():
    traindf = create_dataframe(imgpath)
    train_generator, valid_generator = create_data_generator(traindf)
    lrcn(train_generator, valid_generator)
    

if __name__ == "__main__":
    main()