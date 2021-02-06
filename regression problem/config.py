# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:22:10 2021

@author: Admin
"""

config = {
    
    "Epochs"         : 5000,       # Number of epochs
    "learning_rate"  : 10**-3,     # learning rate
    "decay_lr"       : 0.1,        # deacy learing
    "decay_lr_epoch" : 50,         # deacy learning rate
    "min_lr"         : 10**-4,     # min learing
    "batch_size"     : 32,         # batch size
    "optimizer_flag" : 'Adam',     # Optimizer
    
    "number_points"  :15,          # number of output points
    "0_1_mapping"    : False,      # map output to [0,..,1]
    "train points"   : 330,        # train data points
    "random"         : True,       # randomize data points
    
    "std_weight"     : 0.01,       # std loss weight
    "threshold loss" : 10,         # threshold loss
    
    "images_mat_path": 'data/images.mat',
    "labels_mat_path": 'data/centers_labels.mat',
    
    "data_path"      : 'data/data.npy',
    "labels_path"    : 'data/labels.npy',
}