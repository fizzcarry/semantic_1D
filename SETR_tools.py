import numpy as np
import matplotlib.pyplot as plt

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
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')
    plt.show()
def plot_RGB_image(tensor):
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    # 假设您有一个三通道的张量

    # 将张量转换为NumPy数组，并调整通道顺序
    # 从 (3, 256, 256) 转换为 (256, 256, 3)
    image_np = np.transpose(tensor.numpy(), (1, 2, 0))

    # 将像素值从 [0, 1] 范围映射到 [0, 255] 范围
    image_np = (image_np * 255).astype(np.uint8)

    # 使用matplotlib显示图像
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()






