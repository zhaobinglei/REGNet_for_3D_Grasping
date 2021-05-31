import os
# import glob
# import pickle
# import torch
# import math
# import torch.utils.data
# import torch.nn as nn
import numpy as np
import random

path = '/data1/cxg6/eval_data'

np.random.seed(1)
p_path = os.listdir(path)
p_path.sort()
p_path = np.array(p_path)

index = np.random.choice(len(p_path), int(len(p_path)*0.8), replace=False)
data_names = p_path[index]
print(len(data_names))
num = 0
length = 0
score = 0
for data_name in data_names: 
    data_path = os.path.join(path, data_name)
    data = np.load(data_path, allow_pickle=True)

    length += len(data['frame'])
    score  += (data['antipodal_score']).sum()
    num += 1
    if num % 10 == 0:
        print("rest:", len(data_names)-num)
        print(score / length, length/num)
    
print(length/num)
print(score/length)