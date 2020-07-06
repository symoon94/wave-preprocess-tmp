import pandas as pd 
import numpy as np
import glob
from keras_preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, Flatten, Dropout, BatchNormalization
import tensorflow as tf

# ----- ae version -----
def make_batch(img_list, batch_size):
    i = 0
    train_dataset = []
    for each in img_list:
        img = cv2.imread(each)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        train_dataset.append(img.astype("float32")/255.0)
        i += 1
    return train_dataset

train_imgs = glob.glob("../chunkdata/img/*.png")
test_imgs = glob.glob("../chunkdata/img/*.png")
train_loader = make_batch(train_imgs, 16)

train_loader = tf.convert_to_tensor(train_loader)

import ipdb; ipdb.set_trace()
input_img = Input(shape=(64, 64, 3))
x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')


model.fit(train_loader,train_loader, epochs = 10, batch)

import ipdb; ipdb.set_trace()


ex = "../chunkdata/img/173_chunk05.png"
img = cv2.imread(ex)
img = cv2.resize(img, (64, 64))
img = img.astype("float32")/255.0
img = np.reshape(img, (1,64,64,3))

output = model.predict(img)

for i in range(2):
    plt.subplot(1,2,1)
    img = img.reshape(64,64,3)
    plt.imshow(img)
    plt.subplot(1,2,2)
    output = output.reshape(64,64,3)
    plt.imshow(output)
    plt.show()  

