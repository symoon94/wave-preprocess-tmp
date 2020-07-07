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

import ipdb; ipdb.set_trace()


# ----- ae version -----
def make_batch(img_list):
    train_dataset = []
    for each in img_list:
        img = cv2.imread(each)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        train_dataset.append(img.astype("float32")/255.0)
    return train_dataset

train_imgs = glob.glob("../../chunkdata/img/*.png")
test_imgs = glob.glob("../../chunkdata/img/*.png")
train_loader = make_batch(train_imgs)

train_loader = tf.convert_to_tensor(train_loader)

input_img = Input(shape=(128, 128, 3))
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


model.fit(train_loader,train_loader, epochs = 10, batch_size = 10)

import ipdb; ipdb.set_trace()

ex = "../../chunkdata/img/01_chunk05.png"
img = cv2.imread(ex)
img = cv2.resize(img, (64, 64))
img = img.astype("float32")/255.0
img = np.reshape(img, (1,64,64,3))

output = model.predict(img)

for i in range(1):
    plt.subplot(1,2,1)
    img = img.reshape(64,64,3)
    plt.imshow(img)
    plt.subplot(1,2,2)
    output = output.reshape(64,64,3)
    plt.imshow(output)
    plt.show()  


######## 모델 저장 (weight 저장)###########
#json 파일 저장
json_save = './cae_weight_save/'

model_json = model.to_json()
with open(json_save+"model.json", "w") as json_file : 
    json_file.write(model_json)

#h5 파일 저장
model.save_weights(json_save+"model.h5")
print("Saved model to disk")

