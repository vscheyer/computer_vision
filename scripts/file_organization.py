"""
This script organizes https://www.kaggle.com/dmitryyemelyanov/chinese-traffic-signs into
directory where each sub-directory is it's own class.  To run the script, a copy of the dataset must be downloaded
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
import pandas as pd


class FileOrganizing():
    def __init__(self,folder_path):
        """
        folder_path (str): the path to the folder containing just the downloaded dataset
        """
        self.folder_path = folder_path
        self.annotations_path = self.folder_path +'annotations.csv'
        self.org_images_path = self.folder_path + "images/"
        self.all_annotations = pd.read_csv(self.annotations_path)
        self.image_annotations = pd.concat([self.all_annotations['file_name'],
            self.all_annotations['category']], axis = 1,
            keys = [ "file_name", "category"])
        self.current_dir = os.getcwd()
        self.copied_images_path = self.current_dir + "/images"

    def copy_class_to_folder(self, class_val):
        """Creates a folder for a class and copies all images from that class
        into the folder"""
        path = self.copied_images_path + "/class_{}".format(class_val)
        os.mkdir(path) #make folder
        df = self.image_annotations.loc[self.image_annotations['category'] == class_val] #make_df
        file_names = df['file_name']
        for file in file_names:
                src_path = self.org_images_path + str(file)
                shutil.copy(src_path, path)

    def organize_data(self, class_vals):
        """Creates a directory containg subdirectories for all classes in the
        class_vals
        """
        if os.path.isdir(self.copied_images_path):
            shutil.rmtree(self.copied_images_path)
        os.mkdir(self.copied_images_path)
        for c in class_vals:
            self.copy_class_to_folder(c)

downloaded_data_path = '/home/abbymfry/Desktop/chinese_traffic_signs/'
selected_categories = [1,4,6,50,9]
if __name__ == "__main__":
    dp = FileOrganizing(downloaded_data_path)
    dp.organize_data(selected_categories)
