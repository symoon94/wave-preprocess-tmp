import glob
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from tqdm import tqdm
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils import np_utils

from keras.models import model_from_json
from keras.models import load_model

TIMESTEPS = 5
CLASSES = 10
EPOCHS = 10

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


with open('./ae_lstm_mnist_save_weights/model.json', "r") as json_file2:
  loaded_model_json = json_file2.read()
  json_file2.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('./ae_lstm_mnist_save_weights/model.h5')

encoder = loaded_model.layers[0]
encoder.add(keras.layers.Flatten())

features = encoder.predict(X_train)
valid_features = encoder.predict(X_valid)

def create_dataset(X, Y, look_back=TIMESTEPS):
    dataX = []
    for i in range(len(X)-look_back):
        dataX.append(X[i:(i+look_back)])
    dataX = np.reshape(dataX, (-1, TIMESTEPS, 576))
    dataY = Y[5:]
    dataY = np_utils.to_categorical(dataY, CLASSES)
    dataY = np.reshape(dataY, (-1, CLASSES))
    return dataX, dataY

x_train, y_train = create_dataset(features, y_train, look_back=TIMESTEPS)
x_valid, y_valid = create_dataset(valid_features, y_valid, look_back=TIMESTEPS)

input_shape = (TIMESTEPS, 576)
lstm = keras.models.Sequential([
                          keras.layers.LSTM(128, input_shape=input_shape, return_sequences=False),
                          keras.layers.Dense(CLASSES, activation='softmax')
])

lstm.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

lstm.fit(x_train, y_train,validation_data=(x_valid,y_valid), epochs = EPOCHS)



