# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:07:46 2021

@author: Admin
"""
import numpy as np
import scipy.io as sio
import mat73

def load_data(data_path,labels_path):
    
    data   = np.load(data_path)
    labels = np.load(labels_path) 

# =============================================================================
#     labels[:,:,0] = labels[:,:,0]/640
#     labels[:,:,1] = labels[:,:,1]/480
# =============================================================================
    
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    
    data   = data[arr]
    labels = labels[arr]
    
    images_train = data[:330]
    labels_train = labels[:330]
    images_test  = data[330:]
    labels_test  = labels[330:]
    
    return images_train, labels_train,images_test,labels_test

def process_data(images_mat_path,labels_mat_path):
    
    data   = np.moveaxis(mat73.loadmat(images_mat_path)['images'],-1,1)
    labels = sio.loadmat(labels_mat_path)['centers_labels']
    
    for i in range(data.shape[0]):
        data[i] = (data[i]-data[i].mean())/data[i].std()
        
    np.save('data/data.npy', data)
    np.save('data/labels.npy', labels)
    