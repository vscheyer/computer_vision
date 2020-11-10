# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
# from dataset_processing import DatasetProcessing

import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
from tensorflow.keras import layers

# data = input_data.read_data_sets('data/fashion',one_hot=True,\
#                                  source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

class ConvNeuralNet():
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.model = None

    def load_data(self, data_dir):
        batch_size = 32
        img_height = 20
        img_width = 20

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)


        class_names = self.train_ds.class_names
        print(class_names)

        print("==============ALL THE SHIT======================")
        print(self.train_ds)

    def convert_to_greyscale(self):
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    
    def one_hot_encode(self):
        #one-hot encode target column
        self.y_train = to_categorical(self.train_ds)
        self.y_test = to_categorical(self.val_ds)

        # y_train[0]
    
    def make_model(self):
        #create model
        self.model = Sequential()
        #add model layers
        self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))

        #compile model using accuracy to measure model performance
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=3)

    def predict(self, num_images):
        #predict first num_images in the test set
        return self.model.predict(self.X_test[:num_images])

if __name__ == "__main__":
    # dataset = DatasetProcessing('/home/vscheyer/catkin_ws/src/computer_vision/scripts/images/', training_div = 2)
    print("make cnn")
    cnn = ConvNeuralNet()
    print("load data")
    cnn.load_data('/home/vscheyer/catkin_ws/src/computer_vision/scripts/images/')
    print("encode")
    cnn.one_hot_encode()
    print("make model")
    cnn.make_model()
    print("train model")
    # cnn.train_model()
    print("predict")
    print(cnn.predict(4))
    print(cnn.y_test[:4])
    print("done")
