from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import os.path
import projecttools as pt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# get path to images
p = Path('.')
jpgpath = list(p.glob("test/**/**/*.jpeg"))
print(jpgpath[0:5])

# map label to images
jpglabels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], jpgpath))
print("native: ", jpglabels.count("native"))
print("non-native: ", jpglabels.count("nonnative"))

# separate images and labels
jpgpathseries = pd.Series(jpgpath, name="path").astype(str)
jpglabelseries = pd.Series(jpglabels, name="label").astype(str)

# df of all data
df_all = pd.concat([jpgpathseries, jpglabelseries], axis=1)
df_all = df_all.sample(frac=1).reset_index(drop=True)

#print(df_all.head(-1))
Train_Data, Test_Data = train_test_split(df_all, train_size=0.8, shuffle=True)

xtrain = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
xtest = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

train = xtrain.flow_from_dataframe(dataframe=Train_Data,
                                   x_col="path",
                                   y_col="label",
                                   color_mode="rgb",
                                   class_mode="categorical",
                                   subset="training",
                                   batch_size=16,
                                   target_size=(1280, 720))

test = xtest.flow_from_dataframe(dataframe=Test_Data,
                                 x_col="path",
                                 y_col="label",
                                 color_mode="rgb",
                                 class_mode="categorical",
                                 batch_size=16,
                                 target_size=(1280, 720))
# shape of input
print(train.image_shape)
input_sh = (1280, 720, 3)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df_all["path"][i]))
    ax.set_title(df_all["label"][i])
plt.tight_layout()
plt.show()

# Model A; the 'simpler' model
def runModelA():
    model = Sequential()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience = 10)

    # Conv layer, max pool, dropout
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu",
                     input_shape=input_sh, strides=(3, 3), padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    # Conv layer, max pool, dropout
    model.add(Conv2D(16, kernel_size=(3, 3), activation="relu",
                     strides= (3,3), padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    # Flatten, followed by dense layers and output layer
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    A_hist = model.fit(train, validation_data=test, epochs=50,callbacks=callback)

    # plot accuracy and loss for model
    pt.plot_acc(A_hist.history['accuracy'], A_hist.history['val_accuracy'], "Model_A_native")
    pt.plot_loss(A_hist.history['loss'], A_hist.history['val_loss'], "Model_A_native")

    print("Model A")
    print(model.summary())

# Model B; the 'complexer' model
def runModelB():
    model = Sequential()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience = 10)

    # Conv layer, max pool, dropout
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu",
                     input_shape=input_sh, strides=(3, 3), padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    # Conv layer, max pool, dropout
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu",
                     strides=(3, 3), padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    # Conv layer, max pool, dropout
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu",
                     strides=(3, 3), padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    # Conv layer, max pool, dropout
    model.add(Conv2D(16, kernel_size=(3,3), activation="relu",
                     strides=(3, 3), padding="same"))
    model.add(Dropout(0.2))
    # Flatten, followed by dense layers and output layer
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    B_hist = model.fit(train, validation_data=test, epochs=50,callbacks=callback)

    # plot accuracy and loss for model
    pt.plot_acc(B_hist.history['accuracy'], B_hist.history['val_accuracy'], "Model_B_native")
    pt.plot_loss(B_hist.history['loss'], B_hist.history['val_loss'], "Model_B_native")

    print("Model B")
    print(model.summary())


# runModelA()
# runModelB()

