import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.models import model_from_json, Model
from keras.utils import np_utils
from keras.layers.core import Dense, Activation



# MNIST 로딩 (라벨은 필요없기 때문에 버림)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train[y_train!=4] = 0
y_train[y_train==4] = 1
y_test[y_test!=4] = 0
y_test[y_test==4] = 1

y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)

# 데이터 정규화 및 Reshape
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))

# 원본데이터에 Noise 추가
noise_factor = 0.1
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Noise가 추가된 데이터 확인
n = 10
plt.figure(figsize=(20, 2))

for i in range(n):    
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))#plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# input_size = 784
# batch_size = 100    
# hidden_neurons = 400    
# epochs = 30
# classes = 10

# model = Sequential()     
# model.add(Dense(hidden_neurons, input_dim=input_size)) 
# model.add(Activation('relu'))     
# model.add(Dense(classes, input_dim=hidden_neurons)) 
# model.add(Activation('softmax'))
 
# model.compile(loss='categorical_crossentropy', 
#     metrics=['accuracy'], optimizer='adadelta')
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

json_save = './weight_save/'
json_file = open(json_save+"model.json", "r") 
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights(json_save+"model.h5") 
import ipdb; ipdb.set_trace()
loaded_model.pop()
loaded_model.pop()
loaded_model.pop()
loaded_model.pop()

loaded_model.layers[0].trainable = False
loaded_model.layers[1].trainable = False
loaded_model.layers[2].trainable = False

new_layer1 = Dense(32, activation = 'relu', name='my_dense1')
new_layer2 = Dense(2, activation = 'softmax', name='my_dense2')

inp = loaded_model.input
out = new_layer1(loaded_model.layers[-1].output)
out = new_layer2(out)

new_model = Model(inp, out)


new_model.compile(optimizer='adam', loss='categorical_crossentropy')

# # 모형 학습
new_model.fit(x_train, y_train, batch_size=256, epochs=50)



# 결과 확인
decoded_imgs = new_model.predict(x_test)
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


# %matplotlib qt
