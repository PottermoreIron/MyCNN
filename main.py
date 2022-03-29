import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model import CNN
import config
from train import train
from test import test_acc
from test import test_image
import torch

device = config.device
if os.path.exists("minist/train"):
    DOWNLOAD_TRAIN_DATA = False
if os.path.exists("minist/test"):
    DOWNLOAD_TEST_DATA = False
data = torchvision.datasets.MNIST(root='./minist/train', transform=torchvision.transforms.ToTensor(), train=True,
                                  download=config.DOWNLOAD_TRAIN_DATA)
train_data, valid_data = train_test_split(data, test_size=0.1, random_state=1)

test_data = torchvision.datasets.MNIST(root='./minist/test', transform=torchvision.transforms.ToTensor(), train=False,
                                       download=config.DOWNLOAD_TEST_DATA)
train_loader = DataLoader(dataset=train_data, batch_size=config.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=config.BATCH_SIZE, shuffle=True)

cnn = CNN()
cnn = cnn.to(device)
optimizer = Adam(cnn.parameters(), lr=config.LR)
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

# 测试
# train(train_loader, valid_loader, config.EPOCH, cnn, config.device, loss_func, optimizer)
# 载入训好的模型
cnn.load_state_dict(torch.load('cnn.pth', map_location=torch.device('cpu')))
# test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)
# test_y = test_data.targets
# 预测
# test(test_x, test_y, cnn)
# 预测图像
test_image(cnn)
