import torch.nn as nn
from FENet import FENet
from spa_CNN import spa_CNN
from T_former import TransformerEncoder
import torch
class T_mean(nn.Module):
    #T帧的特征平均
    #（b,t,c,h,w）->(b,c,h,w)
    def __init__(self):
        super(T_mean, self).__init__()
    def forward(self,x):
        return x.mean(dim=1)

class DFER(nn.Module):
    def __init__(self, num_classes=7,):
        super(DFER, self).__init__()
        self.fenet = FENet()
        self.spa_cnn1 = spa_CNN(dim=64)
        self.t_former1 = TransformerEncoder(128,28,28,768)
        self.spa_cnn2 = spa_CNN(dim=128)
        self.t_former2 = TransformerEncoder(256,14,14,768)
        self.spa_cnn3 = spa_CNN(dim=256)
        self.t_former3 = TransformerEncoder(512,7,7,768)
        self.t_mean = T_mean()
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.fenet(x)
        x = self.spa_cnn1(x)
        x = self.t_former1(x)
        x = self.spa_cnn2(x)
        x = self.t_former2(x)
        x = self.spa_cnn3(x)
        x = self.t_former3(x)
        x = self.t_mean(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    x=torch.randn(1,16,3,224,224)
    model=DFER()
    y=model(x)
    print(y.shape)

