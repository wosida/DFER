import torch
import torch.nn as nn

class spa_CNN(nn.Module):
    def __init__(self,dim=64):
        super(spa_CNN, self).__init__()
        self.conv1 = nn.Conv2d(dim,2*dim,1,2,0)
        self.bn1 = nn.BatchNorm2d(2*dim)

        self.conv2=nn.Conv2d(dim,2*dim,3,2,1)
        self.bn2 = nn.BatchNorm2d(2*dim)
        self.relu= nn.ReLU(inplace=True)
        self.conv3=nn.Conv2d(2*dim,2*dim,3,1,1)
        self.bn3 = nn.BatchNorm2d(2*dim)

        self.conv4=nn.Conv2d(2*dim,2*dim,3,1,1)
        self.bn4 = nn.BatchNorm2d(2*dim)

        self.conv5=nn.Conv2d(2*dim,2*dim,3,1,1)
        self.bn5 = nn.BatchNorm2d(2*dim)

    def forward(self,x):
        t=x.size(1)
        x=x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        x1=self.conv1(x)
        x1=self.bn1(x1)
        x2=self.conv2(x)
        x2=self.bn2(x2)
        x2=self.relu(x2)
        x2=self.conv3(x2)
        x2=self.bn3(x2)
        x=x1+x2
        x=self.relu(x)
        x3=x
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu(x)
        x=self.conv5(x)
        x=self.bn5(x)
        x=x3+x
        x=self.relu(x)
        x=x.view(x.size(0)//t,t,x.size(1),x.size(2),x.size(3))
        return x


