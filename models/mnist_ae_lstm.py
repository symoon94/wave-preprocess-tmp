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
EPOCHS = 2000

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid, X_test = X_train_full[:36000], X_train_full[36000:48000], X_train_full[48000:]
y_train, y_valid, y_test = y_train_full[:36000], y_train_full[36000:48000], y_train_full[48000:]


with open('./ae/mnist_cae_save_weights/model.json', "r") as json_file2:
  loaded_model_json = json_file2.read()
  json_file2.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('./ae/mnist_cae_save_weights/model.h5')

encoder = loaded_model.layers[0]
encoder.add(keras.layers.Flatten())

features = encoder.predict(X_train)
valid_features = encoder.predict(X_valid)
test_features = encoder.predict(X_test)

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
x_test, y_test = create_dataset(test_features, y_test, look_back=TIMESTEPS)

input_shape = (TIMESTEPS, 576)
lstm = keras.models.Sequential([
                          keras.layers.LSTM(128, input_shape=input_shape, return_sequences=False),
                          keras.layers.Dense(CLASSES, activation='softmax')
])

lstm.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

lstm.fit(x_train, y_train,validation_data=(x_valid,y_valid), epochs = EPOCHS)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = lstm.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)


######## 모델 저장 (weight 저장)###########
#json 파일 저장
json_save = './mnist_ae_lstm_weight_save/'

model_json = lstm.to_json()
with open(json_save+"model.json", "w") as json_file : 
    json_file.write(model_json)

#h5 파일 저장
lstm.save_weights(json_save+"model.h5")
print("Saved model to disk")