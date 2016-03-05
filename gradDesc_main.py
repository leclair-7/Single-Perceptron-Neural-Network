# -*- coding: utf-8 -*-

"""
Created on Wed Mar  2 20:26:05 2016

@author: Lucas
"""

import numpy as np
import NeuralNetwork

titles=[]
data = np.array([])

count =0

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line
            
with open("train2-win.dat",'r') as ofo:    
    for line in nonblank_lines(ofo):        
        if len(line.split()) != 0 and count ==0:
            count+=1
            titles = line.split()
            column = len(titles)            
        else:
            count += 1
            currLine = line.split()            
            row = len(data)
            data = np.append( data, currLine)
            data = np.resize( data, (row + 1, column))
            currLine = []
row = len(data)
column = len(data[0])
data = [ float(i) for line in data for i in line ]
data = np.resize( data, ( row ,column ))            
            
def sigmoid(z):
        #applies the sigmoid activation function
        return 1.0/(1.0 + np.exp(-z))
def sigmoindPrime(z):
        #derivative of sigmoid function        
        return sigmoid(z) * (1.0- sigmoid(z))
def forward(w1, row, data):        
        #input layer matrix + hidden layer matrix                
        z2 = np.dot( data[row][:-1], w1)
        a2 =sigmoid(z2)
        if a2 > .5:
            return 1.0
        else:
            return 0.0

'''
for i in range(1000):
    alphaBest = .00001
'''

w1 = np.zeros( len( data[0]) -1 )
w1 = [ float(i) for i in w1]
alpha = .9

numIterations = 3
for iteration in range(numIterations):
    for k in range(len(data)):
        for i in range(len(w1)):            
            yHypoth = sigmoid(  (np.dot( data[k][:-1], w1) ) )
           
            yActual = data[k][-1]
            data_k_i =  ( data[k][i] ) 
            sigma_wx = sigmoid(np.dot( data[k][:-1], w1) ) 
            #w1[i] = w1[i] + alpha * ( yActual - yHypoth) *(data_k_i)*(sigma_wx)*(sigma_wx - 1) #+ (data[k][i])**2  added to the last bit in
            w1[i] = w1[i] + alpha * ( yActual - yHypoth) *(data_k_i)*sigmoindPrime( np.dot(data[k][:-1], w1))
    printCount = 0
    for t in range(len(titles)):
        if t == 0:
            print("After iteration",t+1,": w(", titles[t],")= %.3f" % w1[t],end="",sep = '')
        elif t == len(titles)-1:
            print(", output= %.3f" % np.dot( data[k][:-1], w1),sep='' )
        else:
            print(", w(", titles[t],")= %.3f" % w1[t],end="",sep='' )
        

#(np.dot( data[k][:-1], w1)     
    
numRight =0
numWrong = 0
for j in range(len(data)):
    yHypoth = forward(w1,j,data)
    if data[j][-1] == yHypoth:
        numRight +=1
    else:
        numWrong += 1
print("\nAccuracy on training set", numRight/(numRight + numWrong)," for accuracy" )
    
    
    
            