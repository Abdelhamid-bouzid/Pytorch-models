# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:23:32 2021

@author: Admin
"""

from loss_function import loss_function
import torch
import numpy as np
import scipy.io as sio
from config import config


def learning_function(model,images_train, labels_train,images_test,labels_test):
    
    ''' #################################################  set up optim  ################################################### '''
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device=device, dtype=torch.float)
    model.eval()
    
    optimizer     = torch.optim.Adam(model.parameters(),lr = config['learning_rate'])
    
    ''' #################################################  initialization  ################################################### '''
    Epochs              = config['Epochs']
    batch_size          = config['batch_size']
        
    Loss_train,Loss_test = [],[]
    best_loss = float('inf')
    for i in range(Epochs):
        
        model.train()
        ll         = 0
        list_inds  = [s for s in range(0,images_train.shape[0],batch_size)]
        for s in list_inds:
            if s+batch_size<images_train.shape[0]:
                targets = images_train[s:s+batch_size]
                labels  = labels_train[s:s+batch_size]
            else:
                targets = images_train[s:]
                labels  = labels_train[s:]
            targets    = torch.from_numpy(targets).to(device=device, dtype=torch.float)
            labels     = torch.from_numpy(labels).to(device=device, dtype=torch.float)
            
            output  = model(targets)
            loss   = loss_function().forward(output, labels,device).to(device)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ll = ll + loss.item()
            
        Loss_train.append(ll/len(list_inds))
        
        if (i+1)%config["decay_lr_epoch"]==0:
            for param_group in optimizer.param_groups:
                if param_group['lr']<config["min_lr"]:
                    param_group['lr'] *=config["decay_lr"]
                else:
                    param_group['lr'] = config["min_lr"]
        

        model.eval()
        ll         = 0
        list_inds  = [s for s in range(0,images_test.shape[0],batch_size)]
        for s in list_inds:
            if s+batch_size<images_test.shape[0]:
                targets = images_test[s:s+batch_size]
                labels  = labels_test[s:s+batch_size]
            else:
                targets = images_test[s:]
                labels  = labels_test[s:]
            targets    = torch.from_numpy(targets).to(device=device, dtype=torch.float)
            labels     = torch.from_numpy(labels).to(device=device, dtype=torch.float)
            
            output  = model(targets)
            loss   = loss_function().forward(output, labels,device).to(device)
            
            ll = ll + loss.item()
            
        Loss_test.append(ll/len(list_inds))
        print("   #####  Train Epoch: {} train_loss: {:0.4f} test_loss: {:0.4f}".format(i,Loss_train[-1],Loss_test[-1]))
        
        print("   #####  Train Epoch: {} train_loss: {:0.4f} test_loss: {:0.4f}".format(i,Loss_train[-1],0))
        if Loss_test[-1]<best_loss:
            best_loss = Loss_train[-1]
            torch.save(model,'models/model.pth')
    return Loss_train,Loss_test