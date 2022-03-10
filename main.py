import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from model import CNN
import config
from train import train

if os.path.exists("minist"):
    DOWNLOAD_DATA = False
train_data = torchvision.datasets.MNIST(root='./minist/train', transform=torchvision.transforms.ToTensor(), train=True,
                                        download=config.DOWNLOAD_DATA)
# test_data = torchvision.datasets.MNIST(root='./minist/test', train=False)
train_loader = DataLoader(dataset=train_data, batch_size=config.BATCH_SIZE, shuffle=True)
cnn = CNN()
optimizer = Adam(cnn.parameters(), lr=config.LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()
train(train_loader, config.EPOCH, cnn, loss_func, optimizer)
