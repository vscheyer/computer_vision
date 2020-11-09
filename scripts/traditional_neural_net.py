import os
import cv2
from dataset_processing import DatasetProcessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_data(df):
    image_paths = []
    labels = []
    for r in range(0,train_df.shape[0]):
        row = train_df.iloc[r]
        image_label = row["category"]
        path = train_path + "/" + str(row['file_name'])
        image_path = tf.convert_to_tensor(path, dtype=tf.string)
        label = tf.convert_to_tensor(image_label)
        image_paths.append(image_path)
        labels.append(label)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    

downloaded_data_path = '/home/abbymfry/Desktop/chinese_traffic_signs/'
selected_categories = [1,50]
train_path = os.getcwd() + "/train"

size_x=20
size_y=20
image_size_flat = size_x * size_y
n_class = len(selected_categories) #number of classes

if __name__ == "__main__":
    dsp = DatasetProcessing(downloaded_data_path, training_div = 2)
    train_df, test_df = dsp.create_train_and_test(selected_categories,
        resize=True, size_x=size_x, size_y=size_y)

    print(train_df)
    print(train_df.dtypes)

    load_data(train_df)

    # for r in range(0,train_df.shape[0]):
    #     row = train_df.iloc[r]
    #     label = row["category"]
    #     image_path = train_path + "/" + str(row['file_name'])
    #     print(image_path, label)
