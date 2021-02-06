# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:07:46 2021

@author: Admin
"""
import numpy as np
import scipy.io as sio
import mat73
from config import config

def load_data():
    
    data   = np.load(config["data_path"])
    labels = np.load(config["labels_path"])[:,:config["number_points"],:] 

    if config["0_1_mapping"]:
        labels[:,:,0] = labels[:,:,0]/640
        labels[:,:,1] = labels[:,:,1]/480

    if config["random"]:
        arr = np.arange(data.shape[0])
        np.random.shuffle(arr)
         
        data   = data[arr]
        labels = labels[arr]
    
    images_train = data[:config["train points"]]
    labels_train = labels[:config["train points"]]
    images_test  = data[config["train points"]:]
    labels_test  = labels[config["train points"]:]
    
    np.save('data/images_train.npy', images_train)
    np.save('data/labels_train.npy', labels_train)
    
    np.save('data/images_test.npy', images_test)
    np.save('data/labels_test.npy', labels_test)
    
    return images_train, labels_train,images_test,labels_test

def process_data():
    
    data   = np.moveaxis(mat73.loadmat(config["images_mat_path"])['images'],-1,1)
    labels = sio.loadmat(config["labels_mat_path"])['centers_labels']
    
    for i in range(data.shape[0]):
        data[i] = (data[i]-data[i].mean())/data[i].std()
        
    np.save('data/data.npy', data)
    np.save('data/labels.npy', labels)