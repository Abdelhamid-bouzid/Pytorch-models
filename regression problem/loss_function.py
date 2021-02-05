# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:03:34 2021

@author: Admin
"""
import torch
from config import config
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance
import torch.nn as nn

class loss_function(Function):
    def __init__(self):
        self.pdist   = PairwiseDistance(2)
        self.loss_fn = nn.MSELoss()
        self.b       = config["threshold loss"]
    
    def forward(self, pred, truth):
        
        cdist   = torch.cdist(pred,truth)**2
        loss1,_ = torch.min(cdist,axis=1)
        loss1   = loss1.mean()
        
        pdist   = torch.cdist(pred,pred)
        tdist   = torch.cdist(truth,truth)
        std_all = 0
        for i in range(pred.shape[0]):
            pstd = pdist[i].std()
            tstd = tdist[i].std()
            std_all += torch.clamp(pstd,min=0,max=tstd)
        loss2 = std_all/pred.shape[0]
        
        pcenter = pred.mean(1)
        tcenter = truth.mean(1)
        loss3   = self.loss_fn(pcenter,tcenter)
            
        loss = loss1 - config["std_weight"]*loss2 + loss3
        
        loss   = torch.abs(loss-self.b) + self.b
# =============================================================================
#         loss   = self.loss_fn(pred,truth)
#         loss   = torch.abs(loss-self.b) + self.b
# =============================================================================
        return loss