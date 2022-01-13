
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

from sklearn.metrics import classification_report, confusion_matrix

# deep learning libraries
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image

import cv2
import csv

import warnings
warnings.filterwarnings('ignore')


train_path = 'train'
test_path = 'test'
valid_path = 'test_useless'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(100, 100), classes=['Amazon', 'Apple', 'Disney', 'Facebook', 'Google', 'IBM', 'Intel', 'Netflix'], batch_size=8)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(100, 100), classes=['Amazon', 'Apple', 'Disney', 'Facebook', 'Google', 'IBM', 'Intel', 'Netflix'], batch_size=8)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(100, 100), classes=None, batch_size=8, shuffle=False)


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


def run_test_harness():
    model = define_model()
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    history = model.fit(
        x=train_batches, validation_data=valid_batches, epochs=8, verbose=2)
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("Training Accuracy:"), print(history.history['acc'][-1])
    print("Testing Accuracy:"), print(history.history['val_acc'][-1])


def read_predictions(predictions):
    output_list = []
    for mypred in predictions:
        if mypred[0] == 1.0:
            output = '0'
            output_list.append(output)
        elif mypred[1] == 1.0:
            output = '1'
            output_list.append(output)
        elif mypred[2] == 1.0:
            output = '2'
            output_list.append(output)
        elif mypred[3] == 1.0:
            output = '3'
            output_list.append(output)
        elif mypred[4] == 1.0:
            output = '4'
            output_list.append(output)
        elif mypred[5] == 1.0:
            output = '5'
            output_list.append(output)
        elif mypred[6] == 1.0:
            output = '6'
            output_list.append(output)
        elif mypred[7] == 1.0:
            output = '7'
            output_list.append(output)
    return output_list


run_test_harness()
test_images, test_labels = next(test_batches)

model = define_model()
predictions = model.predict(x=test_batches, verbose=0)

old_predict = np.round(predictions)

os.chdir('test/unknown')
file_list = glob.glob('*.png')

my_sub = []
i = 0

my_predict = read_predictions(old_predict)

while i < len(file_list):
    my_sub.append(file_list[i] + ',' + str(my_predict[i]))
    i += 1

os.chdir('C:/Users/tijil/github/ExML_2022')
np.savetxt("submission.csv", my_sub, delimiter=" ", fmt='% s')
