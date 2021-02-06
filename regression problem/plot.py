# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:24:38 2021

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
from config import config

def plot(Loss_train,Loss_test):
    x = np.arange(len(Loss_train))
    plt.plot(x,Loss_train,label='Train loss', c='r')
    plt.plot(x,Loss_test,label='Test loss', c='b')
    plt.hline(config["threshold loss"],0,len(Loss_train),label='loss threshold',c='k')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()