import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.autograd import Variable
import sys,os
import numpy as np
import random

class Aux_Layer1(nn.Module):
    def __init__(self, inplanes, num_classes=1000, reduction=8):
        super(Aux_Layer1, self).__init__()
        
        self.preBlock = nn.Sequential(
            nn.Conv2d(inplanes, 2*inplanes, kernel_size=5, padding=2),
            nn.BatchNorm2d(2*inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*inplanes, 2*inplanes, kernel_size=3, groups=16),
            nn.BatchNorm2d(2*inplanes),
            nn.ReLU(inplace=True)
        )

        channel = 2*inplanes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel/reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.FC = nn.Linear(channel, num_classes)
        
    
    def forward(self, x):
        #x1 = self.conv(x)
        x2 = self.preBlock(x)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.dropout(out)
        out = self.FC(out)

        return out

        
class Aux_Layer2(nn.Module):
    def __init__(self, inplanes, num_classes=1000, reduction=8):
        super(Aux_Layer2, self).__init__()
        
        self.preBlock = nn.Sequential(
            nn.Conv2d(inplanes, 2*inplanes, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*inplanes, 2*inplanes, kernel_size=3, groups=16),
            nn.BatchNorm2d(2*inplanes),
            nn.ReLU(inplace=True)
        )

        channel = 2*inplanes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel/reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.FC = nn.Linear(channel, num_classes)
        
    
    def forward(self, x):
        x2 = self.preBlock(x)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.dropout(out)
        
        out = self.FC(out)

        return out
        
class Aux_Layer3(nn.Module):
    def __init__(self, inplanes, num_classes=1000, reduction=8):
        super(Aux_Layer3, self).__init__()
        
        self.preBlock = nn.Sequential(
            nn.Conv2d(inplanes, 2*inplanes, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*inplanes, 2*inplanes, kernel_size=1, groups=16),
            nn.BatchNorm2d(2*inplanes),
            nn.ReLU(inplace=True)
        )

        channel = 2*inplanes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel/reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.FC = nn.Linear(channel, num_classes)
        
    
    def forward(self, x):
        x2 = self.preBlock(x)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.dropout(out)
        out = self.FC(out)

        return out
        

class Aux_Layer4(nn.Module):
    def __init__(self, inplanes, num_classes=1000, reduction=8):
        super(Aux_Layer4, self).__init__()
        
        self.preBlock = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(inplanes, 2*inplanes, kernel_size=1, groups=16),
            nn.BatchNorm2d(2*inplanes),
            nn.ReLU(inplace=True)
        )
        
        channel = 2*inplanes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel/reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel/reduction), channel, bias=False),
            nn.Sigmoid()
        )
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.FC = nn.Linear(channel, num_classes)
    
    def forward(self, x, targets=None, lam=1.0):
        x2 = self.preBlock(x)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.dropout(out)
        out = self.FC(out)

        return out

class ResNeXtBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride, cardinality, base_width, widen_factor):
        super(ResNeXtBottleneck, self).__init__()

        width_ratio = planes /(widen_factor*64.)
        D = cardinality * int(base_width*width_ratio)

        self.conv_reduce = nn.Conv2d(inplanes, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)

        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)

        self.conv_expand = nn.Conv2d(D, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if inplanes != planes:
            self.shortcut.add_module('shortcut_conv',
                                    nn.Conv2d(inplanes, planes, kernel_size=1,stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(planes))

    def forward(self, x):
        
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)

        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)

        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        residual = self.shortcut.forward(x)
        
        return F.relu(residual + bottleneck, inplace=True)
