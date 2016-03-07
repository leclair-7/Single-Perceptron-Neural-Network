# -*- coding: utf-8 -*-

"""
Created on Wed Mar  2 20:26:05 2016

@author: Lucas
"""

import numpy as np
import sys

from NeuralNetwork import *

assert len(sys.argv) == 5, "Invalid number of arguments"


try:
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
except ValueError:
    print("Invalid input files")
    raise SystemExit

try:
    alpha =float( sys.argv[3])
    numIterations = int(sys.argv[4])
except ValueError:
    print("alpha or number of iterations entered could",)
    print(" not be converted to integers")
    raise SystemExit

NN_uno = NeuralNetwork(trainFile, testFile, alpha, numIterations)
NN_uno.trainNN()
NN_uno.test()
    
            
