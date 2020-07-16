import pandas as pd 
import numpy as np
import glob
from keras_preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, MaxPool2D, Reshape, UpSampling2D, Activation, Flatten, Dropout, BatchNormalization, Conv2DTranspose
import tensorflow as tf
from keras.utils import np_utils, to_categorical, plot_model

import os

import librosa
import librosa.display

from sklearn import metrics 

import struct
import random
import math


esc_path = os.path.abspath('../../ESC-50-master' )
metadata_path = os.path.join(esc_path, 'meta/esc50.csv')
audio_path = os.path.join(esc_path, 'audio')

# Generates/extracts Log-MEL Spectrogram coefficients with LibRosa 
def get_mel_spectrogram(file_path, x_data, mfcc_max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        result = list(divide_list(y, sr))
        result= result[0:5]

        for i, n_sam in enumerate(result):

            # Normalize audio data between -1 and 1
            normalized_y = librosa.util.normalize(n_sam)

            # Generate mel scaled filterbanks
            mel = librosa.feature.melspectrogram(normalized_y, sr=sr, n_mels=n_mels)

            # Convert sound intensity to log amplitude:
            mel_db = librosa.amplitude_to_db(abs(mel))

            # Normalize between -1 and 1
            normalized_mel = librosa.util.normalize(mel_db)

            x_data += [normalized_mel]
            # normalized_mel
        # import ipdb; ipdb.set_trace()

    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return x_data

# Reads a file's header data and returns a list of wavefile properties
def read_header(filename):
    wave = open(filename,"rb")
    riff = wave.read(12)
    fmat = wave.read(36)
    num_channels_string = fmat[10:12]
    num_channels = struct.unpack('<H', num_channels_string)[0]
    sample_rate_string = fmat[12:16]
    sample_rate = struct.unpack("<I",sample_rate_string)[0]
    bit_depth_string = fmat[22:24]
    bit_depth = struct.unpack("<H",bit_depth_string)[0]
    return (num_channels, sample_rate, bit_depth)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    return train_score, test_score


def model_evaluation_report(model, X_train, y_train, X_test, y_test, calc_normal=True):
    dash = '-' * 38

    # Compute scores
    train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Pint Train vs Test report
    print('{:<10s}{:>14s}{:>14s}'.format("", "LOSS", "ACCURACY"))
    print(dash)
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Training:", train_score[0], 100 * train_score[1]))
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Test:", test_score[0], 100 * test_score[1]))


    # Calculate and report normalized error difference?
    if (calc_normal):
        max_err = max(train_score[0], test_score[0])
        error_diff = max_err - min(train_score[0], test_score[0])
        normal_diff = error_diff * 100 / max_err
        print('{:<10s}{:>13.2f}{:>1s}'.format("Normal diff ", normal_diff, ""))



# Expects a NumPy array with probabilities and a confusion matrix data, retuns accuracy per class
def acc_per_class(np_probs_array):    
    accs = []
    for idx in range(0, np_probs_array.shape[0]):
        correct = np_probs_array[idx][idx].astype(int)
        total = np_probs_array[idx].sum().astype(int)
        acc = (correct / total) * 100
        accs.append(acc)
    return accs




"""
    Plotting
"""

def plot_train_history(history, x_ticks_vertical=False):
    history = history.history

    # min loss / max accs
    min_loss = min(history['loss'])
    min_val_loss = min(history['val_loss'])
    max_accuracy = max(history['accuracy'])
    max_val_accuracy = max(history['val_accuracy'])

    # x pos for loss / acc min/max
    min_loss_x = history['loss'].index(min_loss)
    min_val_loss_x = history['val_loss'].index(min_val_loss)
    max_accuracy_x = history['accuracy'].index(max_accuracy)
    max_val_accuracy_x = history['val_accuracy'].index(max_val_accuracy)

    # summarize history for loss, display min
    plt.figure(figsize=(16,8))
    plt.plot(history['loss'], color="#1f77b4", alpha=0.7)
    plt.plot(history['val_loss'], color="#ff7f0e", linestyle="--")
    plt.plot(min_loss_x, min_loss, marker='o', markersize=3, color="#1f77b4", alpha=0.7, label='Inline label')
    plt.plot(min_val_loss_x, min_val_loss, marker='o', markersize=3, color="#ff7f0e", alpha=7, label='Inline label')
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train', 
                'Test', 
                ('%.3f' % min_loss), 
                ('%.3f' % min_val_loss)], 
                loc='upper right', 
                fancybox=True, 
                framealpha=0.9, 
                shadow=True, 
                borderpad=1)

    if (x_ticks_vertical):
        plt.xticks(np.arange(0, len(history['loss']), 5.0), rotation='vertical')
    else:
        plt.xticks(np.arange(0, len(history['loss']), 5.0))

    plt.show()

    # summarize history for accuracy, display max
    plt.figure(figsize=(16,6))
    plt.plot(history['accuracy'], alpha=0.7)
    plt.plot(history['val_accuracy'], linestyle="--")
    plt.plot(max_accuracy_x, max_accuracy, marker='o', markersize=3, color="#1f77b4", alpha=7)
    plt.plot(max_val_accuracy_x, max_val_accuracy, marker='o', markersize=3, color="orange", alpha=7)
    plt.title('Model accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train', 
                'Test', 
                ('%.2f' % max_accuracy), 
                ('%.2f' % max_val_accuracy)], 
                loc='upper left', 
                fancybox=True, 
                framealpha=0.9, 
                shadow=True, 
                borderpad=1)
    plt.figure(num=1, figsize=(10, 6))

    if (x_ticks_vertical):
        plt.xticks(np.arange(0, len(history['accuracy']), 5.0), rotation='vertical')
    else:
        plt.xticks(np.arange(0, len(history['accuracy']), 5.0))

    plt.show()


def compute_confusion_matrix(y_true, 
               y_pred, 
               classes, 
               normalize=False):

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


# Plots a confussion matrix
def plot_confusion_matrix(cm,
                          classes, 
                          normalized=False, 
                          title=None, 
                          cmap=plt.cm.Blues,
                          size=(10,10)):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalized else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

print("MetaData CSV file {}".format(metadata_path))

# Load metadata as a Pandas dataframe
metadata = pd.read_csv(metadata_path)
metadata = metadata.loc[metadata.esc10 == True]


# Examine dataframe's head
metadata.head()


metadata['category'].value_counts()

# # Read every file header to collect audio properties
# audiodata = []
# for index, row in metadata.iterrows():
#     cat = str(row["category"])
#     fold = str(row["fold"])
#     name = str(row["filename"])
#     file_name = os.path.join(audio_path, name)
#     audio_props = read_header(file_name)
#     audiodata.append((name, fold, cat) + audio_props)

# # Convert into a Pandas dataframe
# audiodatadf = pd.DataFrame(audiodata, columns=['file', 'fold', 'category', 'channels','sample_rate','bit_depth'])


# Iterate through all audio files and extract MFCC
features = []
labels = []
frames_max = 0
counter = 0
total_samples = len(metadata)
n_mels=87

def divide_list(l, n): 
  for i in range(0, len(l), n): 
      yield l[i:i + n]

for index, row in metadata.iterrows():
    file_path = os.path.join(os.path.abspath(audio_path), str(row["filename"]))
    # class_label = row["category"]
    # start_v = row['start']
    # end_v = row['end']

    # Extract Log-Mel Spectrograms (do not add padding)
    features = get_mel_spectrogram(file_path=file_path, x_data = features, mfcc_max_padding=0, n_mels=n_mels)
    # features += [norm_mel.tolist()]

    # tmp = np.zeros(39)
    # if end_v >= len(tmp):
    #   tmp[start_v+1:] = 1
    # elif end_v != 0:
    #   tmp[start_v:end_v] = 1
    # labels += list(tmp)

    # # Notify update every N files
    # if (counter == 500):
    #     print("Status: {}/{}".format(index+1, total_samples))
    #     counter = 0

    # counter += 1
    
# print("Finished: {}/{}".format(index, total_samples))
print("Raw features length: {}".format(len(features)))
# print("Feature labels length: {}".format(len(labels)))

X = np.array(features)
# y = np.array(labels)

# np.save("./dataframe/X-mel_spec", X)
# np.save("./dataframe/y-mel_spec", y)



# Pre-processed MFCC coefficients
# X = np.load("./dataframe/X-mel_spec.npy")
# y = np.load("./dataframe/y-mel_spec.npy")


# total = len(metadata) * 39
# indexes = list(range(0, total))
# random.shuffle(indexes)


# test_split_pct = 10
# split_offset = math.floor(test_split_pct * total / 100)

# # Split the metadata
# test_split_idx = indexes[0:split_offset]
# train_split_idx = indexes[split_offset:total]


# len(test_split_idx)


# len(train_split_idx)


# X_test = np.take(X, test_split_idx, axis=0)
# y_test = np.take(y, test_split_idx, axis=0)
# X_train = np.take(X, train_split_idx, axis=0)
# y_train = np.take(y, train_split_idx, axis=0)

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)



# print("Test split: {} \t\t Train split: {} ".format(len(test_meta)*40, len(train_meta)*40))
# print("X test shape: {} \t\t X train shape: {} ".format(X_test.shape, X_train.shape))
# print("y test shape: {} \t\t y train shape: {}".format(y_test.shape, y_train.shape))



# le = LabelEncoder()
# y_test_encoded = to_categorical(le.fit_transform(y_test))
# y_train_encoded = to_categorical(le.fit_transform(y_train))


# How data should be structured
num_rows = n_mels
num_columns = 87
num_channels = 1
import ipdb; ipdb.set_trace()
# Reshape to fit the network input (channel last)
X_train = X.reshape(X.shape[0], num_rows, num_columns)
# X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

# model
input_img = Input(shape=(num_rows, num_columns, num_channels))
x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(16, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)

# x = Conv2D(16, (3, 3), padding='same')(encoded)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(32, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((3, 3))(x)
x = Conv2D(1, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')
import ipdb; ipdb.set_trace()

history = model.fit(X_train, X_train, epochs=50)

# model
# input_img = Input(shape=(num_rows, num_columns, num_channels))
# x = Conv2D(64, (3, 3), padding='same')(input_img)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(32, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(16, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)

# x = Conv2D(16, (3, 3), padding='same')(encoded)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(32, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(64, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(3, (3, 3), padding='same')(x)
# x = BatchNormalization()(x)
# decoded = Activation('sigmoid')(x)

# model = Model(input_img, decoded)
# model.compile(optimizer='adam', loss='binary_crossentropy')


# model.fit(X_train,X_train, epochs = 100, batch_size = 24)

import ipdb; ipdb.set_trace()

# ex = "../../chunkdata/img/01_chunk05.png"
# img = cv2.imread(ex)
# img = cv2.resize(img, (64, 64))
# img = img.astype("float32")/255.0
# img = np.reshape(img, (1,64,64,3))

# output = model.predict(img)

# for i in range(1):
#     plt.subplot(1,2,1)
#     img = img.reshape(64,64,3)
#     plt.imshow(img)
#     plt.subplot(1,2,2)
#     output = output.reshape(64,64,3)
#     plt.imshow(output)
#     plt.show()  


######## 모델 저장 (weight 저장)###########
#json 파일 저장
json_save = './cae_weight_save/'

model_json = model.to_json()
with open(json_save+"model.json", "w") as json_file : 
    json_file.write(model_json)

#h5 파일 저장
model.save_weights(json_save+"model.h5")
print("Saved model to disk")

