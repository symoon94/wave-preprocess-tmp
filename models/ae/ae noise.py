import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.utils import np_utils


# MNIST 로딩 (라벨은 필요없기 때문에 버림)
(x_train, _), (x_test, _) = mnist.load_data()

# 데이터 정규화 및 Reshape
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))

# 원본데이터에 Noise 추가
noise_factor = 0.1
x_train_noisy1 = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy1 = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
# x_train_noisy1 = x_train 
# x_test_noisy1 = x_test 
x_train_noisy = np.clip(x_train_noisy1, 0., 1.)
x_test_noisy = np.clip(x_test_noisy1, 0., 1.)
# plt.hist(x_test_noisy, normed=False)



# Noise가 추가된 데이터 확인
n = 10
plt.figure(figsize=(20, 2))

for i in range(n):    
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))#plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 모형 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=784))
# w0 = model.layers[0].get_weights()
# model.layers[0].set_weights([w0[0]*1, w0[1]+0.1])
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# w2 = model.layers[2].get_weights()
# model.layers[2].set_weights([w2[0]*1, w2[1]+0.1])
model.add(Dense(32, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(784, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

# 모형 학습
model.fit(x_train, x_train, 
          batch_size=256,
          epochs=50)
          # shuffle=True,
          # validation_data=(x_test_noisy, x_test))


∑



# 결과 확인
decoded_imgs = model.predict(x_test)
n = 10
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

    # display reconstruction
    ax = plt.subplot(3, n, i+2*n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# %matplotlib qt
