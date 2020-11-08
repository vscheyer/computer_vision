import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import datasets
from dataset_processing import DatasetProcessing
import os
import cv2
import torch
from torchvision import datasets



class TradNet(nn.Module):
    def __init__(self):
        super(TradNet, self).__init__()
        self.fc1 = nn.Linear(20 * 20, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


downloaded_data_path = '/home/abbymfry/Desktop/chinese_traffic_signs/'
selected_categories = [1,50]
train_path = os.getcwd() #+ "/train/"

if __name__ == "__main__":
    dsp = DatasetProcessing(downloaded_data_path, training_div = 2)
    train_df, test_df = dsp.create_train_and_test(selected_categories,
        resize=True, size_x=20, size_y=20)

    dataset = ImageFolderWithPaths(train_path)
    dataloader = torch.utils.DataLoader(dataset)


    # img_path = train_path + "/" + str(train_df.iloc[0]['file_name'])
    # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    # net = TradNet()
    # print(net)
    #
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # criterion = nn.NLLLoss()
    #
    # epochs = 10
    # for e in range(epochs):

    for inputs, labels, paths in dataloader:
        # use the above variables freely
        print(inputs, labels, paths)
