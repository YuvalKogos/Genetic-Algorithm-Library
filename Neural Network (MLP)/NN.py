import random
import math
import numpy as np
from MatrixOperations import *
import csv


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
            self.IHWeights = self.__CreateIHWeights()
            self.HOWeights = self.__CreateHOWeights()
            self.HBias = [random.uniform(-1,1) for i in range(self.hidden_nodes)]
            self.OBias = [random.uniform(-1,1) for i in range(self.output_nodes)]

        else:
            self.learning_rate = lr
            self.__ProccessNNStrinng(String)

    def __CreateIHWeights(self):
        IH = []
        for i in range(self.hidden_nodes):
            tmp = []
            for j in range(self.input_nodes):
                val = random.uniform(-1, 1)
                tmp.append(val)
            IH.append(tmp)
        return IH

    def __CreateHOWeights(self):
        HO = []
        for i in range(self.output_nodes):
            tmp = []
            for j in range(self.hidden_nodes):
                val = random.uniform(-1, 1)
                tmp.append(val)
            HO.append(tmp)
        return HO

    def __ProccessNNStrinng(self, string):
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

        return result

    def __ActivationFunction(self, vec):
        ret = []
        for val in vec:
            v = sigmoid(val)
            ret.append(v)
        return ret

    def PrintWeights(self):
        print('Input-Hidden weights: ')
        for i in range(self.hidden_nodes):
            tmp = "[ "
            for j in range(self.input_nodes + 1):
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

    def Accuracy(self,data_samples,true_labels):
        if len(data_samples) != len(true_labels):
            raise ValueError("Data samples length and labels length must be equal")
        correct_predictions = 0
        for i in range(len(data_samples)):
            result = self.Predict(data_samples[i])
            prediction = result.index(max(result))
            if prediction == true_labels[i].index(max(true_labels[i])):
                correct_predictions += 1

        return correct_predictions / len(data_samples)

    def Mutate(self, rate):
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

    def Train(self, input_vector, target_vector):
        #Hidden Prediction
        hidden_pred = MatrixByVector(self.IHWeights,input_vector)
        hidden_pred = AddVecToVec(hidden_pred,self.HBias)
        hidden_pred = self.__ActivationFunction(hidden_pred)

        #Output prediction
        output_pred = MatrixByVector(self.HOWeights,hidden_pred)
        output_pred = AddVecToVec(output_pred,self.OBias)
        output_pred = self.__ActivationFunction(output_pred)

        #Calculate the output errors
        output_errors = SubVecToVec(target_vector,output_pred)

        #Calculate the gradient (for deltas)
        #Gradient  = errors * preds (1 - preds)
        gradients = SubVecToVec(output_pred,VecByVecMultiplication(output_pred,output_pred))
        gradients = VecByVecMultiplication(gradients,output_errors)
        gradients = VecByConst(gradients,self.learning_rate)

        #Multiply the gradients by the hidden tranposed
        #to get the delta weight matrices (same dimensions)
        HOWeihts_deltas = VecByTVec(gradients,hidden_pred)
        #Adjust the weights
        self.HOWeights = AddMatrixToMatrix(self.HOWeights,HOWeihts_deltas)
        #Adjust the bias by its deltas - just the gradients
        self.OBias = AddVecToVec(self.OBias,gradients)


        #Calculate the hidden errors
        IHWeights_T = Transpose(self.HOWeights)
        hidden_errors = MatrixByVector(IHWeights_T,output_errors)

        #Calculate the hidden gradients
        gradients_hidden = SubVecToVec(hidden_pred,VecByVecMultiplication(hidden_pred,hidden_pred))
        gradients_hidden = VecByVecMultiplication(gradients_hidden,hidden_errors)
        gradients_hidden = VecByConst(gradients_hidden,self.learning_rate)

        #Calculete the hidden deltas
        IHWeights_deltas = VecByTVec(gradients_hidden,input_vector)

        #Adjust the weights based on the deltas
        self.IHWeights = AddMatrixToMatrix(self.IHWeights,IHWeights_deltas)
        self.HBias = AddVecToVec(self.HBias,gradients_hidden)

    def Predict(self, data):

        # Function gets vector (list) of data
        # And feed-forwards it, and return vector (list) of results

        # Input - hidden
        output_vector = MatrixByVector(self.IHWeights, data)
        output_vector = AddVecToVec(output_vector,self.HBias)
        output_vector = self.__ActivationFunction(output_vector)


        # Hidden - output
        output_vector = MatrixByVector(self.HOWeights, output_vector)
        output_vector = AddVecToVec(output_vector,self.OBias)
        output_vector = self.__ActivationFunction(output_vector)

        return output_vector




