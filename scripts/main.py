from dataset_processing import DatasetProcessing


downloaded_data_path = '/home/abbymfry/Desktop/chinese_traffic_signs/'
selected_categories = [0,1,2,4,57]

if __name__ == "__main__":
    ds = DatasetProcessing(downloaded_data_path, training_div = 2)
    train_df, test_df = ds.create_train_and_test(selected_categories)
    print(train_df)
    print(test_df)
