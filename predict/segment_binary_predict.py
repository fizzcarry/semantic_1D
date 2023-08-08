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
def plot_grayscale_image(tensor,binary=False):
    # 将张量转换为numpy数组
    image_np = tensor.cpu().detach().numpy() if tensor.is_cuda else tensor.cpu().detach().numpy()

    # 确保图像数据范围在0到1之间（如果不在这个范围，将其归一化）
    if np.min(image_np) < 0 or np.max(image_np) > 1:
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))

    #数据进行二元化
    if binary:
        image_np = (image_np > 0.5)
    # 绘制黑白图像
    image_np = pd.DataFrame(image_np.reshape(1, -1))
    sns.heatmap(image_np, cmap='coolwarm', cbar=False)
    plt.show()
#加载模型
model_path = r"C:\Users\DELL\PycharmProjects\SETR_1D\checkpoints/segmentation_model.pkl"
model = SETRModel(patch_size=16,
                  in_channels=6,
                  out_channels=1,
                  hidden_size=2048,
                  num_hidden_layers=6,
                  num_attention_heads=16,
                  decode_features=[512, 256, 128, 64])
model.load_state_dict(torch.load(model_path))
model.eval()


#加载数据
IMU_data=get_test_data(r"D:\my_data\cam\segmentation\test",0,2)
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
plot_grayscale_image(out,binary=False)
plot_grayscale_image(out,binary=True)