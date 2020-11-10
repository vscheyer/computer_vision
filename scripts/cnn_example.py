# Example source https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# data = input_data.read_data_sets('data/fashion',one_hot=True,\
#                                  source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plot the first image in the dataset
# print(X_train)
plt.imshow(X_train[0])
plt.show()

#check image shape
print(X_train[0].shape)

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

print("BEFORE CATEGORY")
print(y_train)
print("PRE TEST")
print(y_test)

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]

print("AFTER CATEGORY")
print(y_train)
print("TEST")
print(y_test)

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 4 images in the test set
print(model.predict(X_test[:4]))
plt.figure(1)


#actual results for first 4 images in test set
print(y_test[:4])