from __future__ import print_function, division
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import cv2
import matplotlib.pyplot as plt
import os
import keras
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Input, Lambda, LSTM, GRU, Bidirectional, Lambda, Concatenate
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from keras.applications.resnet50 import ResNet50
from builtins import range, input

#face_cascade = cv2.CascadeClassifier('haar.xml')
dirs = "data/train/"
img_size = 60
n_classes=7

data = []
for name in os.listdir(dirs):
    for f in os.listdir(dirs+name):
        f = cv2.imread(os.path.join(dirs+name, f))
        #data.append((f[::2,::2] /255.0).reshape(img_size,img_size))
        img = cv2.resize(f, (img_size,img_size))
        #img = img.reshape(-1)
        data.append((img, name))

#df = pd.DataFrane(data)            
df = pd.DataFrame(data, columns=["image", "name"])
#df = pd.DataFrame.from_records(data)
print("Length:",len(df))

##################################import-val_data##############################
dirs = "data/val/"
data = []
for name in os.listdir(dirs):
    for f in os.listdir(dirs+name):
        f = cv2.imread(os.path.join(dirs+name, f))
        img = cv2.resize(f, (img_size,img_size))
        #img = img.reshape(-1)
        data.append((img, name))

df_test = pd.DataFrame(data, columns=["image", "name"])
#df_test = pd.DataFrame.from_records(data)
print("Test size: ", len(df_test))

le = LabelEncoder()
le.fit(df["name"].values)
X_train = list(df.image.values)
X_train = np.array(X_train)
X_train = np.mean(X_train, axis=-1,keepdims=True)
#X_train = X_train.reshape(len(X_train), -1)
X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)
#X_train = X_train.reshape(X_train.shape[0], -1)
X_train = X_train.astype('float32')
X_train = X_train / 255

y_train = le.transform(df["name"].values)
y_train = to_categorical(y_train)
print(X_train.shape)
print(y_train.shape)



X_test = list(df_test.image.values)
X_test = np.array(X_test)
X_test = np.mean(X_test, axis=-1,keepdims=True)
#X_test = X_test.reshape(len(X_test), -1)
X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 1)
#X_test = X_test.reshape(X_test.shape[0], -1)
X_test = X_test.astype('float32')
X_test = X_test / 255

y_test = le.transform(df_test["name"].values)
y_test = to_categorical(y_test)

print(X_test.shape)
print(y_test.shape)

########---Stacked-LSTM---######
model = Sequential()
model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(3600, ), activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Embedding(10000, 32))
model.add(Dropout(0.8))
# Convert the 2D ouput into one sequential vector as input of the LSTM layer
# define LSTM model
model.add(LSTM(10))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(n_classes, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1, validation_split=0.05)
