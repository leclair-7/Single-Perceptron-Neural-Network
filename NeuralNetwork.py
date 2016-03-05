# - Created by Lucas Hagel - 
#this nonworking iteration of the Neuralnetwork class
#is a first take of a few to get this project working
#

import numpy as np


class NeuralNetwork(object):
    def __init__(self,trainFileName, alphaUpdate,iteration):
        self.alpha = alphaUpdate
        self.data = np.array([])
        
        #row and column are rewritten in getDataFromFile
        #these two are just here for awaremess
        self.row = 0
        self.column = 3
        self.NumIterations = iteration
        self.getDataFromFile(trainFileName)

        #defines hyperparameters
        self.inputLayerSize = self.column - 1
        self.outputLayerSize = 1
        self.hiddenLayerSize = 1
        
        #NN's learn the larameters
        self.w1 =np.zeros( [self.inputLayerSize ,1])
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
        self.row = len(self.data)
        self.column = len(self.data[0])
        self.data = [ float(i) for line in self.data for i in line ]
        self.data = np.resize( self.data, ( self.row ,self.column ))  
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
    def trainNN(self):
        for iteration in range(self.NumIterations):
            for k in range(len(self.data)):
                for i in range(len(self.w1)):            
                    yHypoth = self.sigmoid(  (np.dot( self.data[k][:-1], self.w1) ) )
           
                    yActual = self.data[k][-1]
                    data_k_i =  ( self.data[k][i] ) 
                    #sigma_wx = self.sigmoid(np.dot( self.data[k][:-1], self.w1) ) 
                    #w1[i] = w1[i] + alpha * ( yActual - yHypoth) *(data_k_i)*(sigma_wx)*(sigma_wx - 1) #+ (data[k][i])**2  added to the last bit in
                    self.w1[i] = self.w1[i] + self.alpha * ( yActual - yHypoth) *(data_k_i)*self.sigmoindPrime( np.dot(self.data[k][:-1], self.w1))
            for t in range(len(self.titles)):
                if t == 0:
                    print("After iteration",t+1,": w(", self.titles[t],")= %.3f" % self.w1[t],end="",sep = '')
                elif t == len(self.titles)-1:
                    print(", output= %.3f" % np.dot( self.data[k][:-1], self.w1),sep='' )
                else:
                    print(", w(", self.titles[t],")= %.3f" % self.w1[t],end="",sep='' )
        numRight =0.0
        numWrong = 0.0
        for j in range(len(self.data)):
            yHypoth = self.forward(self.w1,j,self.data)
            if self.data[j][-1] == yHypoth:
                numRight +=1.0
            else:
                numWrong += 1.0
        print("\nAccuracy on training set %.1f" % ((numRight/(numRight + numWrong)) * 100.0),"%",sep='' )


    