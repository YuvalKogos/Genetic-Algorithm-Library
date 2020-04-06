import numpy as np
import copy
import random
from MatrixOperations import *

class NN:

    @staticmethod
    def sigmoid(value):
        return (1 / (1 + np.exp(-value)))
 

    @staticmethod
    def Sigmoid(array):
        for i in range(array.shape[0]):
            array[i] = (1 / (1 + np.exp(-array[i])))
        return array
    

    def ActivationFunction(self,vec):
        return [NN.sigmoid(val) for val in vec]

     

    def __init__(self,input_nodes,output_nodes,hidden_layers = [20,10],constructor_string = None, learning_rate = 0.01,last_layer_bias = False):
        
        self.learning_rate = learning_rate
        self.last_layer_bias = last_layer_bias
        
        if constructor_string:
            self.Metrices,self.biases = self.ProcessString(constructor_string)
        else:
            #List of all the layers metrices
            self.Metrices = []
            self.biases = []
            #If there are no hidden layers
            if hidden_layers == []:
                self.Metrices.append([[random.uniform(-1,1) for i in range(input_nodes)] for j in range(output_nodes)])
            else:
                #The first layer (input -> hidden) weights
                self.Metrices.append([[random.uniform(-1,1) for i in range(input_nodes)] for j in range(hidden_layers[0])])
                self.biases.append([random.uniform(-1,1) for i in range(hidden_layers[0])])
                #The hidden layers
                for i in range(0,len(hidden_layers)-1):
                    self.Metrices.append([[random.uniform(-1,1) for i in range(hidden_layers[i])] for j in range(hidden_layers[i+1])])
                    self.biases.append([random.uniform(-1,1) for i in range(hidden_layers[i+1])])
                #The output layer (last hidden -> output)
                self.Metrices.append([[random.uniform(-1,1) for i in range(hidden_layers[len(hidden_layers)-1])] for j in range(output_nodes)])
                if last_layer_bias:
                    self.biases.append([random.uniform(-1,1) for i in range(output_nodes)]) 
    

    def Predict(self,inputs,return_predictions_list = False):
        assert len(inputs) == len(self.Metrices[0][0]), "input vector dimensions doesn't match first layer dimensions, should be {0}".format(len(self.Metrices[0][0]))
        
        #List for the back-prop training
        predictions = []
        #Go over each matrix in multiply, add bias, and activate
        outputs = inputs
        predictions.append(outputs.copy())
        for i in range(0,len(self.Metrices)-1):
            outputs = MatrixByVector(self.Metrices[i],outputs)
            outputs = AddVecToVec(outputs,self.biases[i])
            outputs = self.ActivationFunction(outputs)
            predictions.append(outputs.copy())

        #Output layer
        outputs = MatrixByVector(self.Metrices[-1],outputs)
        if self.last_layer_bias:
            outputs = AddVecToVec(outputs,self.biases[-1])
        
        outputs = self.ActivationFunction(outputs)
        predictions.append(outputs.copy())

        if return_predictions_list:
            return predictions

        return outputs

    def Train(self,input_vector,target_vector):
        #List that contains the prediction for each layer
        predictions = self.Predict(input_vector,return_predictions_list = True)  
        #List that'll contains the errors of each layer (*from end to begining!)
        errors = []
        #Output errors
        output_pred = predictions[-1]
        output_errors = SubVecToVec(target_vector,output_pred)
        errors.append(output_errors.copy())
        #Calculate the gradients
        gradients = SubVecToVec(output_pred,VecByVecMultiplication(output_pred,output_pred))
        gradients = VecByVecMultiplication(gradients,output_errors)
        gradients = VecByConst(gradients,self.learning_rate)
        #Multiply the gradients by the hidden tranposed
        #to get the delta weight matrices (same dimensions)
        weights_deltas = VecByTVec(gradients,predictions[-2])
        self.Metrices[-1] = AddMatrixToMatrix(self.Metrices[-1],weights_deltas)
        if self.last_layer_bias:
            self.biases[-1] = AddVecToVec(self.biases[-1],gradients)

        #Loop over every other layer include inputs-hidden1
        for i in reversed(range(0,len(self.Metrices)-1)):
            #Calculate hidden errors
            next_layer_weights_tranposed = Transpose(self.Metrices[i+1])
            hidden_error = MatrixByVector(next_layer_weights_tranposed,errors[-1])
            errors.append(hidden_error.copy())

            #Calculate hidden gradients
            gradients_hidden = SubVecToVec(predictions[i+1],VecByVecMultiplication(predictions[i+1],predictions[i+1]))
            gradients_hidden = VecByVecMultiplication(gradients_hidden,hidden_error)
            gradients_hidden = VecByConst(gradients_hidden,self.learning_rate)

            #Calculate
            weights_deltas = VecByTVec(gradients_hidden,predictions[i])
            
            self.Metrices[i] = AddMatrixToMatrix(self.Metrices[i],weights_deltas)
            self.biases[i] = AddVecToVec(self.biases[i],gradients_hidden)

    def Accuracy(self, data_samples, true_labels):
        if len(data_samples) != len(true_labels):
            raise ValueError("Data samples length and labels length must be equal")
        correct_predictions = 0
        for i in range(len(data_samples)):
            result = self.Predict(data_samples[i])
            prediction = result.index(max(result))
            if prediction == true_labels[i].index(max(true_labels[i])):
                correct_predictions += 1

        return correct_predictions / len(data_samples)

    def ProcessString(self,string):
        Metrices = []
        Biases =[]
        for i in range(len(string.split('|'))-1):
            layer = string.split('|')[i]
            #Process the matrix string
            tmp = layer.split('?')[0]
            #Process each row
            mat_lst = []
            for j in range(len(tmp.split('/'))-1):
                row = tmp.split('/')[j]
                values = []
                for k in range(len(row.split(','))-1):
                    val = row.split(',')[k]
                    values.append(float(val))
                mat_lst.append(values)
            #Add the bias vector
            bias = []
            for j in range(len(layer.split('?')[1].split(','))-1):
                val = layer.split('?')[1].split(',')[j]
                bias.append(float(val))

            Metrices.append(mat_lst)
            Biases.append(bias)




        return Metrices,Biases


    def Mutate(self,mutation_rate):
        #Mutate each weight matrix
        for matrix in self.Metrices:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    rate = np.random.uniform(0,1)
                    if rate < mutation_rate:
                        matrix[i][j] = np.random.uniform(-1,1)

        #Mutate each bias vector
        for vec in self.biases:
            for i in range(vec.shape[0]):
                rate = np.random.uniform(0,1)
                if rate < mutation_rate:
                    vec[i] = np.random.uniform(-1,1)


    def GenerateString(self):
        res = ''
        for i in range(len(self.Metrices)):
            for row in range(self.Metrices[i].shape[0]):
                for col in range(self.Metrices[i].shape[1]):
                    res += str(self.Metrices[i][row][col]) + ','
                res += '/'
            res += '?'
            for val in self.biases[i]:
                res += str(val) + ','

            res += '|'

        return res


    def Copy(self):
        return copy.deepcopy(self)


    def Crossover(self,other):
        assert len(self.Metrices) == len(other.Metrices) and len(self.biases) == len(other.biases),"Weights arrays dimensions ar not equal."
        ret_nn = self.Copy()

        #Loop over the weight metrices
        for k in range(len(ret_nn.Metrices)):
            matrix = ret_nn.Metrices[k]
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    r = random.uniform(0,1)
                    if r < 0.5:
                        matrix[i][j] = other.Metrices[k][i][j]


        for k in range(len(ret_nn.biases)):
            vec = ret_nn.biases[k]
            for i in range(len(vec.shape[0])):
                r = random.uniform(0,1)
                if r < 0.5:
                    vec[i] = other.biases[k][i]


        return ret_nn





