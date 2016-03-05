# - Created by Lucas Hagel - 
#this nonworking iteration of the Neuralnetwork class
#is a first take of a few to get this project working
#

import numpy as np


class NeuralNetwork(object):
    def __init__(self):
        self.alpha = alphaUpdate
        self.data = np.array([])
        #defines hyperparameters
        self.inputLayerSize = NumInputs - 1
        self.outputLayerSize = 1
        self.hiddenLayerSize = 1
        
        #NN's learn the larameters
        self.w1 =np.zeros( self.inputLayerSize ,1)
        #self.w2 =np.array( self.hiddenLayerSize ,1)
        self.w2 =np.array( [1])
        
        # block comment under here allows the NN to initialize to random numbers
        # in its matrices instead of zeros         
        '''
        self.w1 =np.random.randn( self.inputLayerSize, \
                                  self.hiddenLayerSize)
        self.w2 = np.random.randn( self.hiddenLayerSize, \
                                    self.outputLayerSize)
        '''
    '''
    Under here are the file read operations
    '''
    def nonblank_lines(self,f):
        for l in f:
            line = l.rstrip()
            if line:
                yield line
    def getDataFromFile(self,filename):
        count = 0        
        with open(filename,'r') as ofo:    
            for line in self.nonblank_lines(ofo):        
                if len(line.split()) != 0 and count ==0:
                    count+=1
                    self.titles = line.split()
                    column = len(self.titles)            
                else:
                    count += 1
                    currLine = line.split()            
                    row = len(self.data)
                    self.data = np.append( self.data, currLine)
                    self.data = np.resize( self.data, (row + 1, column))
                    currLine = []
    '''
    End of File/dataset entry functions
    '''
    def sigmoid(self,z):
        #applies the sigmoid activation function
        return 1.0/(1.0 + np.exp(-z))
        
    def sigmoindPrime(self,z):
        #derivative of sigmoid function        
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

        
    def forward(self, w1, row, data):        
        #input layer matrix + hidden layer matrix                
        z2 = self.sigmoid( np.dot( self.data[row][:-1], w1) )        
        if z2 > .5:
            return 1.0
        else:
            return 0.0

    