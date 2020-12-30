import numpy as np
import os

data = np.load('preprocessed_data_splits/12_2018-04-12/split_0/X_test.npz', allow_pickle=True)
lst = data.files

for item in lst:
    print(item)
    print(data[item].shape)

os.system(f'spd-say "hello babe"')
a = 3
