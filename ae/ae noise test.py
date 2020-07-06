import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.models import model_from_json



# MNIST 로딩 (라벨은 필요없기 때문에 버림)
(x_train, _), (x_test, _) = mnist.load_data()

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
    plt.imshow(x_test_noisy[i].reshape(28, 28))#plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 모형 구성
# model = Sequential()
# model.add(Dense(128, activation='relu', input_dim=784))
# w0 = model.layers[0].get_weights()
# model.layers[0].set_weights([w0[0]*1, w0[1]+0.1])
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# w2 = model.layers[2].get_weights()
# model.layers[2].set_weights([w2[0]*1, w2[1]+0.1])
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(784, activation='sigmoid'))
json_save = 'C:/Users/CT_NT_CNY/Desktop/python test1/denoising AE/weight save/'
json_file = open(json_save+"model.json", "r") 
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights(json_save+"model.h5") 
# print("Loaded model from disk")
# w0 = loaded_model.layers[5].get_weights()
# print(w0)
# loaded_model.layers[5].set_weights([w0[0]*1, w0[1]+0.1])

loaded_model.layers[3].trainable = False

loaded_model.compile(optimizer='adam', loss='binary_crossentropy')

# # 모형 학습
loaded_model.fit(x_train, x_train, 
          batch_size=256,
          epochs=50)
#           # shuffle=True,
#           # validation_data=(x_test_noisy, x_test))
# # model.fit(x_train_noisy, x_train, 
# #           nb_epoch=100,
# #           batch_size=256, 
# #           shuffle=True,
# #           validation_data=(x_test_noisy, x_test))         


#  -------------------------
# import cv2
# import glob

# def make_batch(img_list, batch_size):
#     i = 0
#     train_dataset = []
#     tmp = []
#     for each in img_list:
#         if i == batch_size:
#             i = 0
#             train_dataset.append(tmp)
#             tmp = []
#         img = cv2.imread(each,  cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
#         # img = np.moveaxis(img, -1, 0)
#         tmp.append(img.astype("float32")/255.0)
#         i += 1
#     # import ipdb; ipdb.set_trace()
#     # Y_data = np.vstack(train_dataset)
#     # Z_data = copy.deepcopy(Y_data)
#     # Z_data = (Z_data - Z_data.mean()) / Z_data.std()
#     return train_dataset

# train_imgs = glob.glob("C:/Users/CT_NT_CNY/Desktop/python test1/mel ae/image/image cut/*.png")
# test_imgs = glob.glob("C:/Users/CT_NT_CNY/Desktop/python test1/mel ae/image/image cut/*.png")
# train_loader = make_batch(train_imgs, 128)
# test_loader = make_batch(test_imgs, 128)

# test = np.array(train_loader)[0][0]
# test = np.reshape(test, (1,784))
# # 결과 확인
# decoded_imgs = loaded_model.predict(test)
# n = 1
# plt.figure(figsize=(20, 6))
# for i in range(0, n):
#     # display original
#     ax = plt.subplot(3, n, i+1)
#     plt.imshow(test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # # display noisy
#     # ax = plt.subplot(3, n, i+n+1)
#     # plt.imshow(x_test_noisy[i].reshape(28, 28))
#     # plt.gray()
#     # ax.get_xaxis().set_visible(False)
#     # ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(3, n, i+2*n+1)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()



# 결과 확인
decoded_imgs = loaded_model.predict(x_test)
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
