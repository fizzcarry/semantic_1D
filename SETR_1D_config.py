import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
from torch.nn import functional as F
from torch.utils import data
from sklearn.model_selection import train_test_split
class MyDataset(data.Dataset):
    def __init__(self, feature,target):
        self.feature = feature
        self.target = target
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        # 返回第index个数据样本
            #即为迭代器中读取的内容，一般为数据标签
        return self.feature[index],self.target[index]
gesture_number=3
step_len=256
spilt_len=2048
sample_frequent=500
