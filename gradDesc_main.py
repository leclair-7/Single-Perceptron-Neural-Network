# -*- coding: utf-8 -*-

"""
Created on Wed Mar  2 20:26:05 2016

@author: Lucas
"""

import numpy as np
from NeuralNetwork import *

'''
data = np.loadtxt("train2-win.dat",skiprows=1)

fh = open("train2-win.dat",'r')
for i,line in enumerate(fh):
    if i is 1: break
    titles = line.split()
fh.close()
'''

NN_uno = NeuralNetwork("train5-win.dat","test5-win.dat",.9, 60)
NN_uno.trainNN()
NN_uno.test()

    
            