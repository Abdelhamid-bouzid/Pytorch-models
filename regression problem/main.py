# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:18:51 2021

@author: Admin
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchsummary import summary

from ResNet import ResNet, ResidualBlock
from load_process_data import load_data,process_data
from learning_function import learning_function
from plot import plot

images_mat_path   = 'data/images.mat'
labels_mat_path   = 'data/centers_labels.mat'

data_path         = 'data/data.npy'
labels_path       = 'data/labels.npy'

#process_data(images_mat_path,labels_mat_path)
images_train, labels_train, images_test, labels_test = load_data(data_path,labels_path)

model = ResNet(ResidualBlock, [1, 1, 1])
model = model.float()
#summary(model, (3, 640 ,480))

Loss_train,Loss_test = learning_function(model,images_train, labels_train,images_test,labels_test)

plot(Loss_train,Loss_test)
