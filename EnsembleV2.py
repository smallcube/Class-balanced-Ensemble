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


class ResNeXtBlock(nn.Module):
    def __init__(self, name, inplanes, planes, block_depth, cardinality, base_width, widen_factor=4, pool_stride=2):
        super(ResNeXtBlock, self).__init__()
        self.block = nn.Sequential()
        for bottleneck in range(block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck==0:
                self.block.add_module(name_, ResNeXtBottleneck(inplanes, planes, pool_stride, cardinality, base_width, widen_factor))
            else:
                self.block.add_module(name_, ResNeXtBottleneck(planes, planes, 1, cardinality, base_width, widen_factor))
    
    def forward(self, x):
        count = 0
        
        
        for layer in self.block.modules():
            if isinstance(layer, ResNeXtBottleneck):
                #print("",count)
                if count==0:
                    x1 = layer(x)
                    x2 = x1
                else:
                    x2 = layer(x2)
                count = count+1
        '''
        x1 = self.block(x)
        x2 = x1
        '''
        return x2, x1


class CifarResNeXt(nn.Module):
    def __init__(self, depth, cardinality, base_width, widen_factor=4, num_classes=1000):
        super(CifarResNeXt, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.depth = depth
        self.output_size = 64
        self.stages = [64, 64*widen_factor, 128*widen_factor, 256*widen_factor]
        
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = ResNeXtBlock('stage_1', self.stages[0], self.stages[1], layer_blocks, cardinality=cardinality, base_width=base_width, widen_factor=widen_factor, pool_stride=1)
        self.stage_2 = ResNeXtBlock('stage_2', self.stages[1], self.stages[2], layer_blocks, cardinality=cardinality, base_width=base_width, widen_factor=widen_factor, pool_stride=2)
        self.stage_3 = ResNeXtBlock('stage_3', self.stages[2], self.stages[3], layer_blocks, cardinality=cardinality, base_width=base_width, widen_factor=widen_factor, pool_stride=2)
        
        #1: aux_layer for the layers before stage_1
        #self.aux_layer1 = Aux_Layer1(64, num_classes)
        #2: aux_layer for the layers in stage_1
        #self.aux_layer2 = Aux_Layer1(self.stages[1], num_classes)
        self.aux_layer3 = Aux_Layer1(self.stages[1], num_classes)
        #3: aux_layer for the layers in stage_2
        self.aux_layer4 = Aux_Layer2(self.stages[2], num_classes)
        self.aux_layer5 = Aux_Layer2(self.stages[2], num_classes)
        #4: aux_layer for the layers in stage 3
        self.aux_layer6 = Aux_Layer3(self.stages[3], num_classes)
        self.aux_layer7 = Aux_Layer4(self.stages[3], num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(self.stages[3], num_classes)
        init.kaiming_normal(self.classifier.weight)

        self.weights = None

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    
    def forward(self, x, targets= None, lam=1.0):
        out = x
        
        out = self.conv_1_3x3.forward(out)
        out = F.relu(self.bn_1.forward(out), inplace=True)
        #aux_result1 = self.aux_layer1(out, targets, lam)
        
        out, aux_input2 = self.stage_1.forward(out)
        #aux_result2 = self.aux_layer2(aux_input2)
        aux_result3 = self.aux_layer3(out)
        
        out, aux_input4 = self.stage_2.forward(out)
        aux_result4 = self.aux_layer4(aux_input4)
        aux_result5 = self.aux_layer5(out)
        
        out, aux_input6 = self.stage_3.forward(out)
        aux_result6 = self.aux_layer6(aux_input6)
        aux_result7 = self.aux_layer7(out)
        
        out = self.avgpool(out)
        out = out.view(-1, self.stages[3])
        out = self.classifier(out)
    
        #return out,aux_result1,aux_result2,aux_result3,aux_result4,aux_result5,aux_result6,aux_result7
        return out,aux_result3,aux_result4,aux_result5,aux_result6,aux_result7

    def setWeights(self, w):
        self.weights = w

    def getWeights(self):
        return self.weights


def resnext29_16_64(num_classes=10):
    model = CifarResNeXt(29, 16, 64, 4, num_classes)
    return model

def resnext29_8_64(num_classes=10, dropout=False):
    model = CifarResNeXt(29, 8, 64, 4, num_classes)
    return model

def resnext101_16_64(num_classes=10):
    model = CifarResNeXt(101, 16, 64, 4, num_classes)
    return model

def resnext47_16_64(num_classes=10):
    model = CifarResNeXt(47, 16, 64, 4, num_classes)
    return model

def resnext56_16_64(num_classes=10):
    model = CifarResNeXt(56, 16, 64, 4, num_classes)
    return model  

def resnext20_8_64(num_classes=10, dropout=False):
    model = CifarResNeXt(20, 8, 64, 4, num_classes)
    return model
