# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:03:34 2021

@author: Admin
"""
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance
import torch.nn as nn

class loss_function(Function):
    def __init__(self):
        self.pdist   = PairwiseDistance(2)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, pred, truth):
        loss = self.loss_fn(pred,truth)
        return loss