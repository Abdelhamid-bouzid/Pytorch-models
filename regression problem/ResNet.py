# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:15:21 2021

@author: Admin
"""
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1      = conv3x3(in_channels, out_channels, stride)
        self.bn1        = nn.BatchNorm2d(out_channels)
        self.relu       = nn.LeakyReLU(0.1)
        self.conv2      = conv3x3(out_channels, out_channels)
        self.bn2        = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_points=15):
        super(ResNet, self).__init__()
        self.num_points  = num_points
        self.in_channels = 16
        self.conv        = conv3x3(3, 16)
        self.bn          = nn.BatchNorm2d(16)
        self.relu        = nn.LeakyReLU(0.1)
        self.layer1      = self.make_layer(block, 16, layers[0])
        self.layer2      = self.make_layer(block, 32, layers[1], 1)
        self.layer3      = self.make_layer(block, 64, layers[2], 1)
        self.layer4      = self.make_layer(block, 128, layers[3], 1)
        self.layer5      = self.make_layer(block, 256, layers[4], 1)
        self.fc1         = nn.Linear(6400, 2048)
        self.fc2         = nn.Linear(2048, 512)
        self.fc3         = nn.Linear(512, 2*self.num_points)
        
        self.Ap1         = nn.AdaptiveAvgPool2d((240,320))
        self.Ap2         = nn.AdaptiveAvgPool2d((120,160))
        self.Ap3         = nn.AdaptiveAvgPool2d((60,80))
        self.Ap4         = nn.AdaptiveAvgPool2d((30,40))
        self.Ap5         = nn.AdaptiveAvgPool2d((15,20))
        self.Ap6         = nn.AdaptiveAvgPool2d((5,5))
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def determine_dimnesion(self,out):
        return out.shape[-1]
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.Ap1(out)
        out = self.layer1(out)
        out = self.Ap2(out)
        out = self.layer2(out)
        out = self.Ap3(out)
        out = self.layer3(out)
        out = self.Ap4(out)
        out = self.layer4(out)
        out = self.Ap5(out)
        out = self.layer5(out)
        out = self.Ap6(out)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.reshape(-1, self.num_points, 2)
        #print(out.shape)
        return out
