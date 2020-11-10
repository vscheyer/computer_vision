# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from dataset_processing import DatasetProcessing

import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

# data = input_data.read_data_sets('data/fashion',one_hot=True,\
#                                  source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

class ConvNeuralNet():
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.model = None

    def load_data(self, dataset):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = dataset

        #reshape data to fit model
        self.X_train = self.X_train.reshape(60000,28,28,1)sudo make install
        self.X_test = self.X_test.reshape(10000,28,28,1)

        #mnist.load_data()
    
    def one_hot_encode(self):
        #one-hot encode target column
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

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
    dataset = DatasetProcessing('/home/vscheyer/Desktop/traffic_sign_dataset/archive/', training_div = 2)
    print("make cnn")
    cnn = ConvNeuralNet()
    print("load data")
    cnn.load_data(dataset)
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
