# import torch
# import torch.nn as nn
import tensorflow as tf
# import keras
from dataset_processing import DatasetProcessing
import os
import cv2



downloaded_data_path = '/home/abbymfry/Desktop/chinese_traffic_signs/'
selected_categories = [1,50]
train_path = os.getcwd() + "/train"

if __name__ == "__main__":
    dsp = DatasetProcessing(downloaded_data_path, training_div = 2)
    train_df, test_df = dsp.create_train_and_test(selected_categories,
        resize=True, size_x=20, size_y=20)

    img_path = train_path + "/" + str(train_df.iloc[0]['file_name'])

    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
