import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

cuda = torch.device('cpu') 


train_data=datasets.MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())
validate_data=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())

d=train_data[0][1]

train_loader=DataLoader(train_data,batch_size=100)
validate_loader=DataLoader(validate_data,batch_size=1000)

z = torch.tensor([[[1,2,3,-4],[0.0,2.0,-3.0,0],[0,2,3,1],[0,0,0,0]]])
pool=torch.nn.MaxPool2d(2, stride=2)
print(pool(z))


class Conv(nn.Module):
    def __init__(self):
        super(Conv,self).__init__()
        self.conv=nn.Conv2d(1,1,4,2)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self,x_data):
        py=self.conv(x_data)
        return py

model=Conv()

print(model(train_data[0][0]))