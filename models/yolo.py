import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import shy

class YOLO(nn.Module):
    def __init__(self, S, B, C):
        super(YOLO, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        
        self.layer1 = nn.Sequential(
            *self.make_conv(3, 64, 7, 2, 3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            *self.make_conv(64, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            *self.make_conv(192, 128, 1, 1, 0),
            *self.make_conv(128, 256, 3, 1, 1),
            *self.make_conv(256, 256, 1, 1, 0),
            *self.make_conv(256, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            *self.make_conv(512, 256, 1, 1, 0),
            *self.make_conv(256, 512, 3, 1, 1),
            *self.make_conv(512, 256, 1, 1, 0),
            *self.make_conv(256, 512, 3, 1, 1),
            *self.make_conv(512, 256, 1, 1, 0),
            *self.make_conv(256, 512, 3, 1, 1),
            *self.make_conv(512, 256, 1, 1, 0),
            *self.make_conv(256, 512, 3, 1, 1),
            
            *self.make_conv(512, 512, 1, 1, 0),
            *self.make_conv(512, 1024, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            *self.make_conv(1024, 512, 1, 1, 0),
            *self.make_conv(512, 1024, 3, 1, 1),
            *self.make_conv(1024, 512, 1, 1, 0),
            *self.make_conv(512, 1024, 3, 1, 1),
            *self.make_conv(1024, 1024, 3, 1, 1),
            *self.make_conv(1024, 1024, 3, 2, 1),
        )
        self.layer6 = nn.Sequential(
            *self.make_conv(1024, 1024, 3, 1, 1),
            *self.make_conv(1024, 1024, 3, 1, 1),
        )
        self.fc1 = nn.Linear(7*7*1024, 4096)
        self.fc2 = nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        self.relu = nn.LeakyReLU(0.1)

    def make_conv(self, _in, _out, _k, _s, _p):
        layer = []
        layer.append(nn.Conv2d(_in, _out, kernel_size=_k, stride=_s, padding=_p))
        layer.append(nn.LeakyReLU(0.1))
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1, 7*7*1024)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x

if __name__ == '__main__':
    x = torch.zeros((10, 3, 448, 448))
    net = YOLO(7, 2, 20)
    x = net(x)
    print(x.shape)