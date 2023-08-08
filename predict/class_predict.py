import matplotlib.pyplot as plt
import torch
from SETR.transformer_seg import Vit
import torchvision
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from SETR_1D_tools import *
#加载模型
model_path = r"C:\Users\DELL\PycharmProjects\SETR_1D\checkpoints/classification_model.pkl"
model = Vit(patch_size=16,
            in_channels=6,
            out_class=2,
            hidden_size=2048,
            num_hidden_layers=1,
            num_attention_heads=16,
            )
model.load_state_dict(torch.load(model_path))
model.eval()


#加载数据
IMU_data=get_test_data(r"D:\my_data\cam\classification\test",0,2)
print(IMU_data.shape)
plt.plot(IMU_data[0,0:2048])
plt.show()
plt.plot(IMU_data[1,0:2048])
plt.show()
plt.plot(IMU_data[2,0:2048])
plt.show()
input_tensor=torch.Tensor(IMU_data[:,0:2048])
#获取预测结果
out = model(input_tensor.unsqueeze(0))
print(out)
