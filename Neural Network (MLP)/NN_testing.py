import csv
from NN import *

def PrepareDataset():
    reader = csv.reader(open(r"C:\Users\yuval\Desktop\GitHub projects\Genetic_Algorithm_example\iris.csv", 'r'))
    dataset = []
    labels = []
    next(reader)
    for line in reader:
        dataset.append([float(line[i]) for i in range(len(line) - 1)])
        if line[len(line) - 1] == "Setosa":
            labels.append([1,0,0])
        elif line[len(line) - 1] == "Versicolor":
            labels.append([0,1,0])
        elif line[len(line) - 1] == "Virginica":
            labels.append([0,0,1])

    return dataset, labels




def XOR_Problem_example(number_of_epochs):
    # XOR problem dataset
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [[1,0], [0,1], [0,1], [1,0]]
    # Neural Network model
    nn = NeuralNetwork(2, 2, 2, 0.2)
    # Initial prediction
    print(nn.Predict(data[0]))
    print("Accuracy : ", nn.Accuracy(data,labels))
    # Training
    for i in range(number_of_epochs):
        sample = random.choice(data)
        idx = data.index(sample)
        nn.Train(sample, labels[idx])

    # Predictions after training
    print("Prediction for data {0} is {1}".format(data[0], nn.Predict(data[0])))
    print("Prediction for data {0} is {1}".format(data[1], nn.Predict(data[1])))
    print("Prediction for data {0} is {1}".format(data[2], nn.Predict(data[2])))
    print("Prediction for data {0} is {1}".format(data[3], nn.Predict(data[3])))
    print("Accuracy : " , nn.Accuracy(data,labels))


def SplitDataset(dataset,labels,fraqtion_for_train  = 0.7):
    idx_split = int(fraqtion_for_train * len(dataset))
    x_train = dataset[:idx_split]
    x_test = dataset[idx_split:]
    y_train = labels[:idx_split]
    y_test = labels[idx_split:]

    return x_train,x_test,y_train,y_test



def IRIS_dataset_example():
    dataset, labels = PrepareDataset()

    x_train, x_test, y_train, y_test = SplitDataset(dataset, labels)

    nn = NeuralNetwork(len(dataset[0]), 20, len(labels[0]), 0.1)
    print("initial accuracy : ", nn.Accuracy(dataset, labels))

    for i in range(300):
        if i%100 == 0:
            print("Epoch : [{0}/{1}] , accuracy (train) : {2}".format(i,300,nn.Accuracy(x_train, y_train)))
        for idx in range(len(x_train)):
            nn.Train(x_train[idx], y_train[idx])

    print("Accuracy train : ", nn.Accuracy(x_train, y_train))
    print("Accuracy test : ", nn.Accuracy(x_test, y_test))

def main():
    IRIS_dataset_example()
    print("Done.")


if __name__ == "__main__":
    main()



