import torch
import torch.nn as nn
class FENet(nn.Module):
    def __init__(self):
        #input(3,224,224)
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
    def forward(self, x):
        t=x.size(1)
        x=x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x1=x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x=x+x1
        x=self.relu(x)
        x2=x
        x = self.conv4(x)
        x = self.bn4(x)
        x= self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x=x+x2
        x = self.relu(x)
        x=x.view(x.size(0)//t,t,x.size(1),x.size(2),x.size(3))
        return x #64,56,56
