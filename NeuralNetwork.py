# - Created by Lucas Hagel - 
#this iteration of the Neuralnetwork class
#is a single perceptron gradient descent NN
#

import numpy as np

class NeuralNetwork(object):
    def __init__(self,trainFileName, testFile, alphaUpdate,iteration):
        self.alpha = alphaUpdate
        
        #row and column are rewritten in getDataFromFile
        #these two are just here for awaremess
        #self.row = 0
        #self.column = 3
        self.NumIterations = iteration
        
        
        self.data = np.loadtxt(trainFileName,skiprows=1)

        fh = open(trainFileName,'r')
        for i,line in enumerate(fh):
            if i is 1: break
            self.titles = line.split()
            
        self.testSet = np.loadtxt(testFile,skiprows=1)
    
        assert len(self.testSet[0]) == len(self.data[0]), "Training set and test vectors dont match"
        self.inputLayerSize = len(self.data[0]) - 1
        self.outputLayerSize = 1
        self.hiddenLayerSize = 1
        
        #NN's learn the larameters
        self.w1 =np.zeros( [self.inputLayerSize ,1])
        #print(self.w1)        
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
        data = np.array([])
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
                    row = len(data)
                    data = np.append( data, currLine)
                    data = np.resize( data, (row + 1, column))
                    currLine = []
        #self.row = len(data)
        #self.column = len(self.data[0])
        data = [ float(i) for line in data for i in line ]
        data = np.resize( data, ( row ,column ))
        print(data)        
        return data
    '''
    End of File/dataset entry functions
    '''
    def sigmoid(self,z):
        #applies the sigmoid activation function
        return 1.0/(1.0 + np.exp(-z))
        
    def sigmoindPrime(self,z):
        #derivative of sigmoid function        
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))

        
    def forward(self, weightVector, row, dataset):        
        #input layer matrix + hidden layer matrix                
        z2 = self.sigmoid( np.dot( dataset[row][:-1], weightVector) )        
        if z2 > .5:
            return 1.0
        else:
            return 0.0
    def trainNN(self):
        #for iteration in range(self.NumIterations):
        #len(self.data)
        for k in range(self.NumIterations):
            newWeight = self.w1.copy()                
            for i in range(len(self.w1)):            
                yHypoth = self.sigmoid(  (np.dot( self.data[k % len(self.data)][:-1], self.w1) ) )           
                yActual = self.data[k % len(self.data)][-1]
                data_k_i =  ( self.data[k % len(self.data)][i] ) 
                #sigma_wx = self.sigmoid(np.dot( self.data[k][:-1], self.w1) ) 
                #w1[i] = w1[i] + alpha * ( yActual - yHypoth) *(data_k_i)*(sigma_wx)*(sigma_wx - 1) #+ (data[k][i])**2  added to the last bit in
                newWeight[i] = self.w1[i] + self.alpha * (yActual - yHypoth) *(data_k_i)*self.sigmoindPrime( np.dot(self.data[k% len(self.data)][:-1], self.w1))
                #print(self.w1)
            self.w1 = newWeight.copy()
            #print( self.w1)
            for t in range(len(self.titles)):
                if t == 0:
                    print("After iteration",k+1,": w(", self.titles[t],")= %.3f" % self.w1[t],end="",sep = '')
                elif t == len(self.titles)-1:
                    print(", output= %.3f" % self.sigmoid(np.dot( self.data[k % len(self.data)][:-1], self.w1) ),sep='' )
                else:
                    print(", w(", self.titles[t],")= %.3f" % self.w1[t],end="",sep='' )
        numRight =0.0
        numWrong = 0.0
        for j in range(len(self.data)):
            yHypoth = self.forward(self.w1,j,self.data)
            #if self.data[j][-1] == yHypoth:
            if int(self.data[j][-1]) == int(yHypoth):
                numRight +=1.0
            else:
                numWrong += 1.0
        print("\nAccuracy on training set (",self.NumIterations," instances): ","%.1f" % ((numRight/(numRight + numWrong)) * 100.0),"%",sep='' )
    #it would be cool if this were written to take a numpy input as an argument
    def test(self):
        numRight =0.0
        numWrong = 0.0
        for j in range(len(self.testSet)):
            yHypoth = self.forward(self.w1,j,self.testSet)
            #print("Hypothe", int(yHypoth), " actual is: ", int(self.testSet[j][-1])  )
            if ((int(self.testSet[j][-1]) ) == (int(yHypoth)) ):
                numRight +=1.0
            else:
                numWrong += 1.0
        print("\nAccuracy on test set (",len(self.testSet)," instances): ", "%.1f" % ((numRight/(numRight + numWrong)) * 100.0),"%",sep='' )
        


    
