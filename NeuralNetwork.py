# - Created by Lucas Hagel - 
#this nonworking iteration of the Neuralnetwork class
#is a first take of a few to get this project working
#

import numpy as np


class NeuralNetwork(object, fileName):
    def __init__(self):
        self.alpha = alphaUpdate
        self.data = dataSet
        #defines hyperparameters
        self.inputLayerSize = NumInputs - 1
        self.outputLayerSize = 1
        self.hiddenLayerSize = 1
        
        #NN's learn the larameters
        self.w1 =np.zeros( self.inputLayerSize ,1)
        #self.w2 =np.array( self.hiddenLayerSize ,1)
        self.w2 =np.array( [1])
        '''
        self.w1 =np.random.randn( self.inputLayerSize, \
                                  self.hiddenLayerSize)
        self.w2 = np.random.randn( self.hiddenLayerSize, \
                                    self.outputLayerSize)
        '''
    def sigmoid(self,z):
        #applies the sigmoid activation function
        return 1/(1 + np.exp(-z))

    def sigmoindPrime(self,z):
        #derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)


        
    def forward(self,InputVector):        
        #input layer matrix + hidden layer matrix                
        self.z2 = np.dot( InputVector, self.w1)
        self.a2 = self.sigmoid(self.z2)
        if self.a2 > .5:
            return 1
        else:
            return 0
        '''
        #these 3 lines below are for increasing NN complexity 
        self.z3 = np.dot( self.a2, self.w2)
        yHat = self.sigmoid(self.z3)
        return yHat
        '''
    def costFunctionPrime( self, X, y):
        #Note  dot is matric multiplication, multiply is elementwise
        self.actual = self.forward(X)
        delta3 = np.multiply( - (y - self.actual), self.sigmoindPrime(self.z3) )
        dJdW2 = np.dot( self.a2.T, delta3)
        
        delta2 = np.dot( delta3, self.w2.T * self.sigmoindPrime(self.z2))
        dJdW1 = np.dot( X.T, delta2)
        
        return dJdW1, dJdW2
    def gradientDescentWithMessage( ):
        for x_in in range(len(data[:-1])):
            for i in self.w1:
                theY =  forward(data[x_in][:-1])
                actual = data[:-1]
                for k in 


        

