"""
This script organizes https://www.kaggle.com/dmitryyemelyanov/chinese-traffic-signs into
test and training data.

-------------------------------------------------------------------------------
The dataset is a converted version of publicly available Chinese Traffic Sign Recognition Database.
This work is supported by National Nature Science Foundation of China(NSFC) Grant 61271306

Credits:
LinLin Huang,Prof.Ph.D
School of Electronic and Information Engingeering,
Beijing Jiaotong University,
Beijing 100044,China
Email: huangll@bjtu.edu.cn

All images originally collected by camera under nature scenes or from BAIDU Street View. """

import pandas as pd
import os
import shutil

class DatasetProcessing():
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.annotations_path = self.folder_path +'annotations.csv'
        self.images_path = self.folder_path + "/images/"

        self.all_annotations = pd.read_csv(self.annotations_path)
        self.image_annotations = pd.concat([self.all_annotations['file_name'],
            self.all_annotations['category']], axis = 1,
            keys = [ "file_name", "category"])
        self.current_dir = os.getcwd()
        self.train_path = self.current_dir + "/train"
        self.test_path = self.current_dir + "/test"
        self.cat_df = [] #list of dataframes that each contain one of the categories selected

    def select_category(self,val):
        """Returns a pd.DataFrame that is a subset of the image_annotations
        DataFrame rows that are part of the category specified in val
        """
        df = self.image_annotations.loc[self.image_annotations['category'] == val]
        return df

    def create_folders(self):
        """Create directory for test & train.  Deletes existing test & train and
        folder as well as contents if needed"""
        if os.path.isdir(self.train_path):
            shutil.rmtree(self.train_path)
            shutil.rmtree(self.test_path)
        os.mkdir(self.train_path)
        os.mkdir(self.test_path)

    def split_category_df(self, df, train_size = 2):
        size = list(df.shape)[0]
        train_num = (int(round(size/train_size)))
        test_num = size - train_num

    def create_train_and_test(self, cat_vals):
        self.create_folders()
        for v in cat_vals:
            df = self.select_category(v)
            self.cat_df.append(df)
        self.split_category_df(self.cat_df[2])

if __name__ == "__main__":
    downlonaded_data_path = '/home/abbymfry/Desktop/chinese_traffic_signs/'
    ds = DatasetProcessing(downlonaded_data_path)
    classes = [0,4,56]
    ds.create_train_and_test(classes)
