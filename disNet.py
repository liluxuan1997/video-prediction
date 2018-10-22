import torch.nn as nn
import torch
import math


__all__ = ['disNet']


class MultiScaleConv2d(nn.Module):    
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=1):
        super(MultiScaleConv2d, self).__init__()
        #to keep same size with input when stride is 1, set padding to (kernel_size-1)*dilation/2
        self.conv1 = nn.Conv2d(inplanes, int(planes/2), kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = nn.Conv2d(inplanes, planes-int(planes/2), kernel_size=kernel_size, stride=stride,
                               padding=2*padding, dilation=2, bias=False)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        out = torch.cat((a,b),1)
        return out

class disNet(nn.Module):
    def __init__(self,num_classes=65):
        super(disNet, self).__init__()
        self.layer1 = self._normal_seq(1,16,5,2,2)
        self.layer2 = self._normal_seq(16,64,3,2,1)
        self.layer3 = self._normal_seq(64,64,3,2,1)
        self.layer4 = self._normal_seq(64,128,3,2,1)
        self.avgpool = nn.AvgPool2d(4,stride=1)
        self.short_layer1 = self._short_seq(1,64,7,2,3)
        self.short_layer2 = self._short_seq(64,128,3,2,1)
        self.fc = nn.Linear(128,1)
        self.activation = nn.Sigmoid()
        self.weight_init()

    def _normal_seq(self,inplanes,planes,kernel_size,stride,padding):
        return nn.Sequential(
            MultiScaleConv2d(inplanes,planes,kernel_size,stride,padding),
            nn.BatchNorm2d(planes),nn.ReLU(inplace=True),nn.Dropout(p=0.2))

    def _short_seq(self,inplanes,planes,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(inplanes,planes,kernel_size,stride,padding),
            nn.AvgPool2d(2,stride=2),
            nn.BatchNorm2d(planes),nn.ReLU(inplace=True))

    def forward(self,x):
        shortcut = x
        x = self.layer1(x)
        x = self.layer2(x)
        x += self.short_layer1(shortcut)
        shortcut = x
        x = self.layer3(x)
        x = self.layer4(x)
        x += self.short_layer2(shortcut)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.activation(x)

        return x
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features*m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
