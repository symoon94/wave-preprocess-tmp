import pandas as pd 
import numpy as np
import glob
from keras_preprocessing.image import ImageDataGenerator


metadata=pd.read_csv('../metadata/audio_metadata.csv',dtype=str)
imgpath = '/Users/sooyoungmoon/git/coretech/chunkdata/img/'

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
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=None,
    x_col="ID",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

# ----model----
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

#Fitting keras model, no test gen for now
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20
)

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

