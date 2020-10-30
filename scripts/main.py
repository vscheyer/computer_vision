from dataset_processing import DatasetProcessing


downloaded_data_path = '/home/abbymfry/Desktop/chinese_traffic_signs/'
selected_categories = []

if __name__ == "__main__":
    ds = DatasetProcessing(downloaded_data_path, training_div = 2)
    selected_dataframe = ds.create_train_and_test(categories)
    print(selected_dataframe)
