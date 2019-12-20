import random
import math
import numpy as np
from MatrixOperations import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanH(x):
    return np.tanh(x)





class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr, String=None):
        if String == None:
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes
            self.learning_rate = lr
            self.IHWeights = self.CreateIHWeights()
            self.HOWeights = self.CreateHOWeights()
            self.HBias = [random.uniform(-1,1) for i in range(self.hidden_nodes)]
            self.OBias = [random.uniform(-1,1) for i in range(self.output_nodes)]

        else:
            self.learning_rate = lr
            self.ProccessNNStrinng(String)



    def ProccessNNStrinng(self, string):
        st_split = string.split('|')

        # Get the first | - NN structure
        NN_struct = st_split[0]
        self.input_nodes = int(NN_struct.split(',')[0])
        self.hidden_nodes = int(NN_struct.split(',')[1])
        self.output_nodes = int(NN_struct.split(',')[2])

        # Get the input - hidden weights
        input_hidden = st_split[1]
        self.IHWeights = []
        for part in input_hidden.split('?'):
            if part == "": break
            tmp = []
            for weight in part.split(','):
                if weight == "": break
                tmp.append(float(weight))
            self.IHWeights.append(tmp)

        # Get the hidden - output weights
        hidden_output = st_split[2]
        self.HOWeights = []
        for part in hidden_output.split('?'):
            if part == "": break
            tmp1 = []
            for weight in part.split(','):
                if weight == "": break
                tmp1.append(float(weight))
            self.HOWeights.append(tmp1)


        hidden_bias = st_split[3]
        self.HBias = []
        for b in hidden_bias.split(","):
            if b == "": break
            self.HBias.append(float(b))

        output_bias = st_split[4]
        self.OBias = []
        for b in output_bias.split(","):
            if b == "": break
            self.OBias.append(float(b))

    def GenerateString(self):
        result = ""
        # NN Structure
        result += str(self.input_nodes) + "," + str(self.hidden_nodes) + "," + str(self.output_nodes) + "|"

        # Input Hidden weights
        for part in self.IHWeights:
            for weight in part:
                result += str(weight) + ","
            result += "?"

        result += "|"

        # Hidden output weights
        for part in self.HOWeights:
            for weight in part:
                result += str(weight) + ","
            result += "?"

        result += "|"


        #Hidden Bias
        for b in self.HBias:
            result += str(b) +","

        result += "|"

        #Output bias
        for b in self.OBias:
            result += str(b) +","

        return result

    def Mutate(self, rate):
        rate = 0.3
        # Input - hidden weights
        for i in range(self.hidden_nodes):
            for j in range(self.input_nodes):
                r = random.uniform(0, 1)
                if r < rate:
                    self.IHWeights[i][j] = random.uniform(-1, 1)

        # Hidden - output weights
        for i in range(self.output_nodes):
            for j in range(self.hidden_nodes):
                r = random.uniform(0, 1)
                if r < rate:
                    self.HOWeights[i][j] = random.uniform(-1, 1)


        #Hidden Bias
        for i in range(len(self.HBias)):
            r = random.uniform(0,1)
            if r < rate:
                self.HBias[i] = random.uniform(-1,1)

        #output Bias
        for i in range(len(self.OBias)):
            r = random.uniform(0, 1)
            if r < rate:
                self.OBias[i] = random.uniform(-1, 1)

    def ActivationFunction(self, vec):
        ret = []
        for val in vec:
            v = sigmoid(val)
            ret.append(v)
        return ret

    def PrintWeights(self):
        print('Input-Hidden weights: ')
        for i in range(self.hidden_nodes):
            tmp = "[ "
            for j in range(self.input_nodes):
                tmp += str(self.IHWeights[i][j]) + ' , '
            tmp += "]"
            print(tmp)
        print('Hidden-Output weights: ')
        for i in range(self.output_nodes):
            tmp = "[ "
            for j in range(self.hidden_nodes):
                tmp += str(self.HOWeights[i][j]) + ' , '
            tmp += "]"
            print(tmp)

    def CreateIHWeights(self):
        IH = []
        for i in range(self.hidden_nodes):
            tmp = []
            for j in range(self.input_nodes):
                val = random.uniform(-1, 1)
                tmp.append(val)
            IH.append(tmp)
        return IH

    def CreateHOWeights(self):
        HO = []
        for i in range(self.output_nodes):
            tmp = []
            for j in range(self.hidden_nodes):
                val = random.uniform(-1, 1)
                tmp.append(val)
            HO.append(tmp)
        return HO

    def Train(self,input_vector,target_vector):
		#Training and optimizting the Nerual Network's weights using back-propegation and gradient descent method
        #Make a prediction. Didn't use self.Predict method because i need to keep hidden_guess value
        hidden_guess = MatrixByVector(self.IHWeights, input_vector)
        hidden_guess = AddVecToVec(self.HBias, hidden_guess)
        hidden_guess = self.ActivationFunction(hidden_guess)


        output_guess = MatrixByVector(self.HOWeights, hidden_guess)
        output_guess = AddVecToVec(self.OBias, output_guess)
        output_guess = self.ActivationFunction(output_guess)


        #Changing each weight by the Formula:
        #Delta = Error function gradient * input * learning rate

        ###Hidden - output weights####

        output_errors = SubVecToVec(target_vector,output_guess)

        # update the weights, based on gradient descent method:
        tmp = VecByVecMultiplication(output_errors,output_guess)
        SubVecToVec(tmp,VecByVecMultiplication(VecByVecMultiplication(output_errors,output_guess),output_guess))
        tmp = MatrixByVector(Transpose(self.HOWeights),tmp)
        hidden_output_deltas = VecByConst(tmp,self.learning_rate)

        self.HOWeights = AddVecToMatrix(self.HOWeights,hidden_output_deltas)


        #####input - Hidden weights#######
        hidden_errors = MatrixByVector(Transpose(self.HOWeights),output_errors)

        tmp = VecByVecMultiplication(hidden_errors, hidden_guess)
        SubVecToVec(tmp,VecByVecMultiplication(VecByVecMultiplication(output_errors, hidden_guess), hidden_guess))
        tmp = MatrixByVector(Transpose(self.IHWeights), tmp)
        hidden_output_deltas = VecByConst(tmp, self.learning_rate)

        self.IHWeights = AddVecToMatrix(self.IHWeights, hidden_output_deltas)



		
		
    def Predict(self, data):
        # Function gets vector (list) of data
        # And feed-forwards it, and return vector (list) of results

        # Input - hidden
        output_vector = MatrixByVector(self.IHWeights, data)
        output_vector = AddVecToVec(self.HBias,output_vector)
        output_vector = self.ActivationFunction(output_vector)

        # Hidden - output
        output_vector = MatrixByVector(self.HOWeights, output_vector)
        output_vector = AddVecToVec(self.OBias,output_vector)
        output_vector = self.ActivationFunction(output_vector)

        return output_vector







