import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
import cv2
from keras.utils import np_utils



(x_train, y_train), (x_test, y_test) = mnist.load_data()

classes = 10

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)

# x_train = np.reshape(x_train, (len(x_train), 784))
# x_test = np.reshape(x_test, (len(x_test), 784))

# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

def resize(data):
     train_data = []
     for img in data:
            resized_img = cv2.resize(img, (32, 32))
            train_data.append(resized_img)
     return train_data

new_x_train = np.array(resize(x_train))
import ipdb; ipdb.set_trace()
new_x_train = np.reshape(new_x_train, (len(new_x_train), 32, 32, 1))

new_x_test = np.array(resize(x_test))
new_x_test = np.reshape(new_x_test, (len(new_x_test), 32, 32, 1))

    


model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 1),
    pooling=None,
    classes=classes,
)

model.compile(optimizer='adam', loss='categorical_crossentropy')

# 모형 학습
model.fit(new_x_train, y_train, 
          batch_size=256,
          epochs=50)

import ipdb; ipdb.set_trace()

# 결과 확인
decoded_imgs = model.predict(x_test)
# decoded_imgs = model.predict(x_test)

n = 20
for i in range(0, n):
    lst = list(decoded_imgs[i])
    print(lst)
    print(lst.index(max(lst)))

plt.figure(figsize=(20, 6))
for i in range(0, n):
    # display original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # # display reconstruction
    # ax = plt.subplot(3, n, i+2*n+1)
    # plt.imshow(decoded_imgs[i].reshape(28, 28))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
plt.show()