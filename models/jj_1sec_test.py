import sys
import os
import IPython as IP
import pandas as pd
import IPython as IP
import struct
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


esc_path = os.path.abspath('/Users/sooyoungmoon/git/coretech/ESC-50-master' )
metadata_path = os.path.join(esc_path, 'meta/esc50.csv')
audio_path = os.path.join(esc_path, 'audio')


from sklearn import metrics 
import pickle
import time
import struct

from keras.callbacks import EarlyStopping


# Generates/extracts Log-MEL Spectrogram coefficients with LibRosa 
def get_mel_spectrogram(file_path, mfcc_max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
    try:
        # import ipdb; ipdb.set_trace()

        # Load audio file
        y, sr = librosa.load(file_path)

        # Normalize audio data between -1 and 1
        normalized_y = librosa.util.normalize(y)

        # Generate mel scaled filterbanks
        mel = librosa.feature.melspectrogram(normalized_y, sr=sr, n_mels=n_mels)

        # Convert sound intensity to log amplitude:
        mel_db = librosa.amplitude_to_db(abs(mel))

        # Normalize between -1 and 1
        normalized_mel = librosa.util.normalize(mel_db)

        # Should we require padding
        shape = normalized_mel.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            normalized_mel = np.pad(normalized_mel, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return normalized_mel


# Generates/extracts MFCC coefficients with LibRosa 
def get_mfcc(file_path, mfcc_max_padding=0, n_mfcc=40):
    try:
        # Load audio file
        y, sr = librosa.load(file_path)

        # Normalize audio data between -1 and 1
        normalized_y = librosa.util.normalize(y)

        # Compute MFCC coefficients
        mfcc = librosa.feature.mfcc(y=normalized_y, sr=sr, n_mfcc=n_mfcc)

        # Normalize MFCC between -1 and 1
        normalized_mfcc = librosa.util.normalize(mfcc)

        # Should we require padding
        shape = normalized_mfcc.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            normalized_mfcc = np.pad(normalized_mfcc, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return normalized_mfcc


# Given an numpy array of features, zero-pads each ocurrence to max_padding
def add_padding(features, mfcc_max_padding=174):
    padded = []

    # Add padding
    for i in range(len(features)):
        px = features[i]
        size = len(px[0])
        # Add padding if required
        if (size < mfcc_max_padding):
            xDiff = mfcc_max_padding - size
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            px = np.pad(px, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        
        padded.append(px)

    return padded


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


# # Given a dataset row it returns an audio player and prints the audio properties
# def play_dataset_sample(dataset_row, audio_path):
#     fold_num = dataset_row.iloc[0]['fold']
#     file_name = dataset_row.iloc[0]['file']
#     file_path = os.path.join(audio_path, fold_num, file_name)
#     file_path = os.path.join(audio_path, dataset_row.iloc[0]['fold'], dataset_row.iloc[0]['file'])

#     print("Class:", dataset_row.iloc[0]['class'])
#     print("File:", file_path)
#     print("Sample rate:", dataset_row.iloc[0]['sample_rate'])
#     print("Bit depth:", dataset_row.iloc[0]['bit_depth'])
#     print("Duration {} seconds".format(dataset_row.iloc[0]['duration']))
    
#     # Sound preview
#     return IP.display.Audio(file_path)



"""
    Prediction and analisys
"""

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

print("ESC50 CSV file {}".format(metadata_path))

# Load metadata as a Pandas dataframe
metadata = pd.read_csv(metadata_path)
metadata = metadata.loc[metadata.esc10 == True]

# Examine dataframe's head
metadata.head()


metadata['category'].value_counts()

# Read every file header to collect audio properties
audiodata = []
for index, row in metadata.iterrows():
    cat = str(row["category"])
    fold = str(row["fold"])
    name = str(row["filename"])
    file_name = os.path.join(audio_path, name)
    audio_props = read_header(file_name)
    audiodata.append((name, fold, cat) + audio_props)

# Convert into a Pandas dataframe
audiodatadf = pd.DataFrame(audiodata, columns=['file', 'fold', 'category', 'channels','sample_rate','bit_depth'])


print(audiodatadf.channels.value_counts(normalize=True))


print("Bit depths:\n")
print(audiodatadf.bit_depth.value_counts(normalize=True))


print("Sample rates:\n")
print(audiodatadf.sample_rate.value_counts(normalize=True))


# 램덤 파일
row = metadata.sample(1)
file_path = audio_path + '/' + str(row.iloc[0,0])

# Window 사이즈
n_fft=2048
hop_length=512

# 오디오 읽고 ->  y(audion time-series), sr(sample_rate)
y, sr = librosa.load(file_path)

# Normalize between -1 and 1
normalized_y = librosa.util.normalize(y)

# Compute STFT
stft = librosa.core.stft(normalized_y, n_fft=n_fft, hop_length=hop_length)

# Convert sound intensity to log amplitude:
stft_db = librosa.amplitude_to_db(abs(stft))


# Plot spectrogram from STFT
plt.figure(figsize=(12, 4))
librosa.display.specshow(stft_db, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB');
plt.title('STFT Spectrogram')
plt.tight_layout()
plt.show()


n_mels = 128

# 멜스펙트로그램 
mel = librosa.feature.melspectrogram(S=stft, n_mels=n_mels)

# 변환 log amplitude:
mel_db = librosa.amplitude_to_db(abs(mel))

# 노멀라이즈 -1 ~ 1
normalized_mel = librosa.util.normalize(mel_db)

# 사진
plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_db, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB');
plt.title('MEL-Scaled Spectrogram')
plt.tight_layout()
plt.show()


# Iterate through all audio files and extract MFCC
features = []
labels = []
frames_max = 0
counter = 0
total_samples = len(metadata)
n_mels=40

for index, row in metadata.iterrows():
    file_path = os.path.join(os.path.abspath(audio_path), str(row["filename"]))
    class_label = row["category"]

    # Extract Log-Mel Spectrograms (do not add padding)
    mels = get_mel_spectrogram(file_path, 0, n_mels=n_mels)
    
    # Save current frame count
    num_frames = mels.shape[1]
    
    # Add row (feature / label)
    # features.append(mels)
    mels = mels[:,:215]
    features = features + np.hsplit(mels, 5)

    # labels.append(class_label)
    labels = labels + [class_label]*5

    # Update frames maximum
    if (num_frames > frames_max):
        frames_max = num_frames

    # Notify update every N files
    if (counter == 500):
        print("Status: {}/{}".format(index+1, total_samples))
        counter = 0

    counter += 1
    
print("Finished: {}/{}".format(index, total_samples))


# Add padding to features with less than frames than frames_max
padded_features = features
# padded_features = add_padding(features, frames_max)


print("Raw features length: {}".format(len(features)))
print("Padded features length: {}".format(len(padded_features)))
print("Feature labels length: {}".format(len(labels)))


# Convert features (X) and labels (y) to Numpy arrays
X = np.array(padded_features)
y = np.array(labels)

np.save("/Users/sooyoungmoon/git/coretech/ESC-50-master/X-mel_1sec_spec", X)
np.save("/Users/sooyoungmoon/git/coretech/ESC-50-master/y-mel_1sec_spec", y)


import sys
import os
import IPython
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import random
from datetime import datetime

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, SpatialDropout2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import ModelCheckpoint 
from keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix


# Pre-processed MFCC coefficients
X = np.load("/Users/sooyoungmoon/git/coretech/ESC-50-master/X-mel_spec.npy")
y = np.load("/Users/sooyoungmoon/git/coretech/ESC-50-master/y-mel_spec.npy")

# Metadata
metadata = pd.read_csv(metadata_path)
metadata = metadata.loc[metadata.esc10 == True]
metadata['category'].unique()
# metadata.head(100)

TIMESTEP = 1
total = len(metadata) * TIMESTEP
indexes = list(range(0, total))
random.shuffle(indexes)


test_split_pct = 10
split_offset = math.floor(test_split_pct * total / 100)

# Split the metadata
test_split_idx = indexes[0:split_offset]
train_split_idx = indexes[split_offset:total]


len(test_split_idx)


len(train_split_idx)


X_test = np.take(X, test_split_idx, axis=0)
y_test = np.take(y, test_split_idx, axis=0)
X_train = np.take(X, train_split_idx, axis=0)
y_train = np.take(y, train_split_idx, axis=0)


# Also split metadata
# test_meta = metadata.iloc[test_split_idx]
# train_meta = metadata.iloc[train_split_idx]


# print("Test split: {} \t\t Train split: {} ".format(len(test_meta), len(train_meta)))
print("X test shape: {} \t\t X train shape: {} ".format(X_test.shape, X_train.shape))
print("y test shape: {} \t\t y train shape: {}".format(y_test.shape, y_train.shape))


le = LabelEncoder()
y_test_encoded = to_categorical(le.fit_transform(y_test))
y_train_encoded = to_categorical(le.fit_transform(y_train))


# How data should be structured
num_rows = 40
num_columns = 43 
num_channels = 1

import ipdb; ipdb.set_trace()   
# Reshape to fit the network input (channel last)
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)


# Total number of labels to predict (equal to the network output nodes)
num_labels = y_train_encoded.shape[1]


def create_model(spatial_dropout_rate_1=0, spatial_dropout_rate_2=0, l2_rate=0):

    # Create a secquential object
    model = Sequential()


    # Conv 1
    model.add(Conv2D(filters=32, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate), 
                     input_shape=(num_rows, num_columns, num_channels)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=32, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())


    # Max Pooling #1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=64, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_2))
    model.add(Conv2D(filters=64, 
                     kernel_size=(3,3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    
   
    # Reduces each h×w feature map to a single number by taking the average of all h,w values.
    model.add(GlobalAveragePooling2D())


    # Softmax output
    model.add(Dense(num_labels, activation='softmax'))
    
    return model

# Regularization rates
spatial_dropout_rate_1 = 0.07
spatial_dropout_rate_2 = 0.14
l2_rate = 0.0005

model = create_model(spatial_dropout_rate_1, spatial_dropout_rate_2, l2_rate)
import ipdb; ipdb.set_trace()


adam = Adam(lr=1e-4, beta_1=0.99, beta_2=0.999)
model.compile(
    loss='categorical_crossentropy', 
    metrics=['accuracy'], 
    optimizer=adam)

# Display model architecture summary 
model.summary()


num_epochs = 1000
num_batch_size = 24


# Save checkpoints
checkpointer = ModelCheckpoint(filepath="/Users/sooyoungmoon/git/coretech/ESC-50-master",
                               verbose=1, 
                               save_best_only=True)
start = datetime.now()
history = model.fit(X_train, 
                    y_train_encoded, 
                    batch_size=num_batch_size, 
                    epochs=num_epochs,
                    validation_split=1/12.
                    ,callbacks=[
                               EarlyStopping(patience=5, monitor="val_accuracy")])
                    # callbacks=[checkpointer], 
                    # verbose=1)

#json 파일 저장
json_save = './esc_1sec_weight_save/'

model_json = model.to_json()
with open(json_save+"_model.json", "w") as json_file : 
    json_file.write(model_json)

#h5 파일 저장
model.save_weights(json_save+"_model.h5")
print("Saved model to disk")

import ipdb; ipdb.set_trace()

duration = datetime.now() - start
print("Training completed in time: ", duration)


model_evaluation_report(model, X_train, y_train_encoded, X_test, y_test_encoded)


# Predict probabilities for test set
y_probs = model.predict(X_test, verbose=0)

# Get predicted labels
yhat_probs = np.argmax(y_probs, axis=1)
y_trues = np.argmax(y_test_encoded, axis=1)

# # Add "pred" column
# test_meta['pred'] = yhat_probs


import importlib

labels = [
          'dog', 
          'chainsaw', 
          'crackling_fire', 
          'helicopter', 
          'rain',
          'crying_baby', 
          'clock_tick', 
          'sneezing', 
          'rooster', 
          'sea_waves'
        ]

# labels = [
#           'dog', 
#           'chainsaw', 
#           'crackling_fire', 
#           'helicopter', 
#           'rain',
#           'crying_baby', 
#           'clock_tick', 
#           'sneezing', 
#           'rooster', 
#           'sea_waves', 'ambiguous'
#         ]


# Sets decimal precision (for printing output only)
np.set_printoptions(precision=2)

# Compute confusion matrix data
cm = confusion_matrix(y_trues, yhat_probs)

plot_confusion_matrix(cm,
                      labels, 
                      normalized=False, 
                      title="Model Performance", 
                      cmap=plt.cm.Blues,
                      size=(12,12))



# Find per-class accuracy from the confusion matrix data
accuracies = acc_per_class(cm)

pd.DataFrame({
    'CLASS': labels,
    'ACCURACY': accuracies
}).sort_values(by="ACCURACY", ascending=False)

indexes = [0,1,2,3,4,5,6,7,8,9]
# indexes = [0,1,2,3,4,5,6,7,8,9,10]

# Build classification report
re = classification_report(y_trues, yhat_probs, labels=[0,1,2,3,4,5,6,7,8,9], target_names=labels)

print(re)

import ipdb; ipdb.set_trace()