"""
This script organizes https://www.kaggle.com/dmitryyemelyanov/chinese-traffic-signs into
test and training data.  To run the script, a copy of the dataset must be downloaded
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

import os
import shutil
import math
import pandas as pd
import cv2

class DatasetProcessing():
    def __init__(self,folder_path, training_div = 2):
        """
        folder_path (str): the path to the folder containing just the downloaded dataset
        Training div (int): the portion of a given category will go into the training data.
        For example, with a training div of 2, 1/2 of a category will be training.  With a
        training div of 6, 1/6 of the category will be part of the training dataset
        """
        self.folder_path = folder_path
        self.training_div = training_div
        self.annotations_path = self.folder_path +'annotations.csv'
        self.images_path = self.folder_path + "images/"
        self.all_annotations = pd.read_csv(self.annotations_path)
        self.image_annotations = pd.concat([self.all_annotations['file_name'],
            self.all_annotations['category']], axis = 1,
            keys = [ "file_name", "category"])
        self.current_dir = os.getcwd()
        self.train_path = self.current_dir + "/train"
        self.test_path = self.current_dir + "/test"
        self.cat_df = []
        self.selected_train_df = pd.DataFrame(columns=['file_name','category'])
        self.selected_test_df = pd.DataFrame(columns=['file_name','category'])

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

    def split_category_df_to_folder(self, df):
        """ splits up a single dataframe containg a split_category_df_to_folder
        into test and training images
        """
        size = list(df.shape)[0]
        train_num = (int(math.ceil(size/self.training_div)))
        test_num = size - train_num
        cat_df_train = (df.iloc[0:train_num])
        cat_df_test = (df.iloc[train_num:train_num + test_num])
        self.selected_train_df = pd.concat([self.selected_train_df,cat_df_train])
        self.selected_test_df = pd.concat([self.selected_test_df,cat_df_test])
        cat_train_files_names = cat_df_train['file_name']
        cat_test_files_names = cat_df_test['file_name']
        for file in cat_train_files_names:
            src_path = self.images_path + str(file)
            shutil.copy(src_path, self.train_path)
        for file in cat_test_files_names:
            src_path = self.images_path + str(file)
            shutil.copy(src_path, self.test_path)

    def resize_image(self,size, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        out = cv2.resize(img, size)
        cv2.imwrite(path,out)

    def resize_test_and_train_images(self,size_x, size_y):
        size = (size_x, size_y)
        for i in range(0,self.selected_train_df.shape[0]):
            row = self.selected_train_df.iloc[i]
            path = self.train_path + "/" + str(row['file_name'])
            self.resize_image(size, path)
        for i in range(0,self.selected_test_df.shape[0]):
            row = self.selected_test_df.iloc[i]
            path = self.test_path + "/" + str(row['file_name'])
            self.resize_image(size, path)

    def create_train_and_test(self, cat_vals, resize = False, size_x = 100,
            size_y = 100):
        """Creates a training set of images and a testing set of images from the
        categories provided in cat_vals.  For the Chinese Traffic Sign
        Recognition Database there are 58 categories
        cat_vals (list): a list of the category number to be sorted into test
        and training data
        """
        self.create_folders()
        for v in cat_vals:
            df = self.select_category(v)
            self.cat_df.append(df)
        for df in self.cat_df:
            self.split_category_df_to_folder(df)
        self.selected_train_df.reset_index(inplace = True, drop = True)
        self.selected_test_df.reset_index(inplace = True, drop = True)
        if resize == True:
            self.resize_test_and_train_images(size_x, size_y)
        return self.selected_train_df, self.selected_test_df
