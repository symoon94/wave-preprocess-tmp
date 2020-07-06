import pandas as pd 
import numpy as np
import glob
from keras_preprocessing.image import ImageDataGenerator


metadata=pd.read_csv('../metadata/audio_metadata.csv',dtype=str)
imgpath = '../chunkdata/img/'

traindf = {'ID':[], 'class':[]}

for index, row in metadata.iterrows():
    ID = row['file_name']
    if len(ID) == 1:
        ID = "0" + ID

    imgs = glob.glob(imgpath + ID + "/*")
    tmp = []
    for img in imgs:
        if int(img.split("chunk")[-1].rstrip('.png')) < 40:    
            tmp.append(img)
    tmp.sort()
    traindf['ID'] += tmp

    start = int(row['start'])
    end = int(row['end'])
    tmp = np.zeros(40)
    if end >= 40:
        tmp[start:] = 1
    elif end != 0:
        tmp[start:end+1] = 1
    traindf['class'] += list(tmp)

traindf = pd.DataFrame(data=traindf)
traindf['class'] = traindf['class'].astype(str)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=None,
    x_col="ID",
    y_col="class",
    subset="training",
    batch_size=1,
    seed=42,
    shuffle=True,
    class_mode=None,
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=None,
    x_col="ID",
    y_col="class",
    subset="validation",
    batch_size=1,
    seed=42,
    shuffle=True,
    class_mode=None,
    target_size=(64,64))

# ----model----
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D


# --- cnn version ---
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same',
#                  input_shape=(64,64,3)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(128, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))

# model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
# model.summary()

# ----- ae version -----
def make_batch(img_list, batch_size):
    i = 0
    train_dataset = []
    tmp = []
    for each in img_list:
        if i == batch_size:
            i = 0
            train_dataset.append(tmp)
            tmp = []
        img = cv2.imread(each)
        img = cv2.resize(img, (20, 100), interpolation=cv2.INTER_AREA)
        
        img = np.moveaxis(img, -1, 0)
        tmp.append(img.astype("float32")/255.0)
        i += 1
    return train_dataset

train_imgs = glob.glob("../*.png")
test_imgs = glob.glob("C:/Users/CT_NT_CNY/Desktop/python test1/mel ae/image/image cut/*.png")
train_loader = make_batch(train_imgs, 128)
test_loader = make_batch(test_imgs, 128)

# train_loader = torch.FloatTensor(train_loader)
# test_loader = torch.FloatTensor(test_loader)



model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), padding='same'))
model.add(Activation('sigmoid'))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.summary()



# #Fitting keras model, no test gen for now
# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
# STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# #STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
# model.fit_generator(generator=train_generator,
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     validation_data=valid_generator,
#                     validation_steps=STEP_SIZE_VALID,
#                     epochs=20
# )

model.fit(train_loader,train_loader)

import ipdb; ipdb.set_trace()
import cv2
model.evaluate(valid_generator, steps=STEP_SIZE_VALID)

ex = "../chunkdata/img/173/173_chunk05.png"
img = cv2.imread(ex)
img = cv2.resize(img, (64, 64))
# img = np.moveaxis(img, -1, 0)
img = img.astype("float32")/255.0
img = np.reshape(img, (1,64,64,3))

model.predict(img)

