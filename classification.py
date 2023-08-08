from SETR_1D_config import *
from SETR.transformer_seg import Vit
import torchvision
import torch
import torch.nn as nn 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from SETR_1D_tools import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is " + str(device))

def compute_acc(model, test_dataloader):
    with torch.no_grad():
        right_num = 0
        total_num = 0
        for in_data, label in tqdm(test_dataloader, total=len(test_dataloader)):
            in_data = in_data.to(device)
            label = label.to(device)
            total_num += len(in_data)
            out = model(in_data)
            pred = out.argmax(dim=-1)
            for i, each_pred in enumerate(pred):
                if int(each_pred) == int(label[i]):
                    right_num += 1
        
        return (right_num / total_num)

if __name__ == "__main__":

    model = Vit(patch_size=16,
                    in_channels=6,
                    out_class=2,
                    hidden_size=2048,
                    num_hidden_layers=1, 
                    num_attention_heads=16, 
                    )
    print(model)
    model.to(device)

    X_all, y_all = get_spilt_data()
    print(X_all.shape)
    x_train, x_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25)
    train_dataset = MyDataset(x_train, y_train)
    test_dataset = MyDataset(x_test, y_test)
    train_iter = data.DataLoader(train_dataset, 16, shuffle=True,
                                )
    test_iter = data.DataLoader(test_dataset, 8, shuffle=True,
                                )

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    report_loss = 0
    step = 0
    best_acc = 0.0

    for in_data, label in tqdm(train_iter, total=len(test_iter)):
        batch_size = len(in_data)
        # plot_grayscale_image(in_data[0][0])
        # print(label[0])
        in_data = in_data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        step += 1
        out = model(in_data)
        # print(out[0])
        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()
        report_loss += loss.item()
        if step % 10 == 0:
            print("report_loss is : " + str(report_loss))
            report_loss = 0
            acc = compute_acc(model, test_iter)
            if acc > best_acc:
                best_acc = acc 
                torch.save(model.state_dict(), "./checkpoints/classification_model.pkl")

            print("acc is " + str(acc) + ", best acc is " + str(best_acc))
        