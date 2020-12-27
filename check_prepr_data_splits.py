import numpy as np


data = np.load('preprocessed_data_splits/07_2018-04-04/split_0/y_train.npz', allow_pickle=True)
lst = data.files

for item in lst:
    print(item)
    print(data[item].shape)
