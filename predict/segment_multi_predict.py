import matplotlib.pyplot as plt
import torch
from SETR.transformer_seg import Vit
import torchvision
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from SETR.transformer_seg import SETRModel
from SETR_1D_tools import *


def plot_segmentation_multi(tensor):
    # 将张量转换为numpy数组
    image_np = tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor

    discretization = segmentation_result_discretization(image_np)
    # 绘制黑白图像
    discretization=discretization[0]
    for i in range(discretization.shape[0]):
        image_np = pd.DataFrame(discretization[i].reshape(1, -1))
        sns.heatmap(image_np, cmap='coolwarm', cbar=False)
        plt.show()
def plot_segmentation_binary(tensor):
    image_np = tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor
    discretization = segmentation_result_discretization(image_np)
    binary_result=multi_to_binary(discretization)
    image_np = pd.DataFrame(binary_result.reshape(1, -1))
    sns.heatmap(image_np, cmap='coolwarm', cbar=False)
    plt.show()
#加载模型
model_path = r"C:\Users\DELL\PycharmProjects\SETR_1D\checkpoints\segmentation_multi_splice_model/0.984_6.83_.pkl"
model = SETRModel(patch_size=16,
                  in_channels=6,
                  out_channels=gesture_number,
                  hidden_size=1024,
                  num_hidden_layers=6,
                  num_attention_heads=16,
                  decode_features=[512, 256, 128, 64])
model.load_state_dict(torch.load(model_path))
model.eval()


#加载数据
IMU_data=get_test_data(r"D:\my_data\cam\segmentation_multi\test","01",0)
print(IMU_data.shape)
start=0
end=2048
# plt.plot(IMU_data[0,start:end])
# plt.show()
# plt.plot(IMU_data[1,start:end])
# plt.show()
# plt.plot(IMU_data[2,start:end])
# plt.show()
#获取预测结果
input_tensor=torch.Tensor(IMU_data[:,start:end])
out = model(input_tensor.unsqueeze(0))
# plot_segmentation_multi(out)
plot_segmentation_binary(out)
#获取预测结果
# input_tensor=torch.Tensor(IMU_data[:,start:end])
# out = model(input_tensor.unsqueeze(0))
# plot_segmentation_multi(out)
# #获取预测结果
# input_tensor=torch.Tensor(IMU_data[:,start:end])
# out = model(input_tensor.unsqueeze(0))
# plot_segmentation_multi(out)



