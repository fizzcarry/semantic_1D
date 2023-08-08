# data_url : https://www.kaggle.com/c/carvana-image-masking-challenge/data
#二元分割
import torch 
import numpy as np 
from SETR.transformer_seg import SETRModel
from PIL import Image
from SETR_1D_config import *
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from SETR_1D_tools import *
# X_all, y_all=get_segmentation_multi()
X_all, y_all=splice_multi()
x_train, x_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25)
train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)
train_iter = data.DataLoader(train_dataset, 3, shuffle=True,
                            )
test_iter = data.DataLoader(test_dataset, 2, shuffle=True,
                            )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))
epoches = 10
out_channels = gesture_number

def build_model():
    model = SETRModel(patch_size=16,
                    in_channels=6,
                    out_channels=gesture_number,
                    hidden_size=1024,
                    num_hidden_layers=6, 
                    num_attention_heads=16, 
                    decode_features=[512, 256, 128, 64])
    return model


def compute_dice(input, target):
    eps = 0.0001
    # input 是经过了sigmoid 之后的输出。
    input = (input > 0.5).float()
    target = (target > 0.5).float()

    # inter = torch.dot(input.view(-1), target.view(-1)) + eps
    inter = torch.sum(target.view(-1) * input.view(-1)) + eps

    # print(self.inter)
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float()) / union.float()
    return t
if __name__ == "__main__":

    model = build_model()
    model.to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    step = 0
    report_loss = 0.0
    for epoch in range(epoches):
        print("epoch is " + str(epoch))

        for img, mask in tqdm(train_iter, total=len(train_iter)):
            optimizer.zero_grad()
            step += 1

            img = img.to(device)
            mask = mask.to(device)

            pred_img = model(img) ## pred_img (batch, len, channel, W, H)

            if out_channels == 1:
                pred_img = pred_img.squeeze(1) # 去掉通道维度

            loss = loss_func(pred_img, mask)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()
            if step%300==0:
                dice = 0.0
                n = 0
                model.eval()
                with torch.no_grad():
                    print("report_loss is " + str(report_loss))
                    train_loss = report_loss
                    report_loss = 0.0
                    for val_img, val_mask in tqdm(test_iter, total=len(test_iter)):
                        n += 1
                        val_img = val_img.to(device)
                        val_mask = val_mask.to(device)
                        pred_img = torch.sigmoid(model(val_img))
                        if out_channels == 1:
                            pred_img = pred_img.squeeze(1)
                        discretization=segmentation_result_discretization(pred_img)
                        cur_dice=get_acc_segmentation_multi(discretization,pred_img)
                        # cur_dice = compute_dice(pred_img, val_mask)
                        dice += cur_dice
                    dice = dice / n
                    print("mean dice is " + str(dice))
                    str_dice="{:.3f}".format(dice)
                    str_train_loss="{:.2f}".format(train_loss)
                    torch.save(model.state_dict(), "./checkpoints/segmentation_multi_splice_model/"+str_dice+"_"+str_train_loss+"_.pkl")
                    model.train()
