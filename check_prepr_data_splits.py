import numpy as np


data = np.load('preprocessed_data_splits/12_2018-04-12/split_0/X_test.npz', allow_pickle=True)
lst = data.files

for item in lst:
    print(item)
    print(data[item].shape)
