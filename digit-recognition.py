from mnist import MNIST
import numpy as np
import random

#read the data from the mnist database
mndata = MNIST('.\digitSamples')

images, labels = mndata.load_training()
testImgs, testLabels = mndata.load_testing()
#convert labels into vectors
output=np.zeros([len(labels),10],dtype=np.float64)
for i in range(len(labels)):
    output[i,labels[i]]=1

testOutput=np.zeros([len(testLabels),10],dtype=np.float64)
for i in range(len(testLabels)):
    testOutput[i,testLabels[i]]=1

#using the seed so results can be reproduced
np.random.seed(1)

#setting up the weights for the network:
#784 inputs + 1 bias -> 40 neurons + 1 bias -> 16 neurons + 1 bias -> 10 outputs

weights0 = 2 * np.random.random((785, 40)) - 1
weights1 = 2 * np.random.random((41, 16)) - 1
weights2 = 2 * np.random.random((17, 10)) - 1

def sigmoid(x, deriv = False):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def predict(inputs): #takes 1x784 Vector as input returns a one hot 1x10 vector
    inputs = np.insert(np.array(inputs), 784, 1, axis = 1)

    pred0 = np.insert(sigmoid(inputs.dot(weights0)), 40, 1, axis = 1)
    pred1 = np.insert(sigmoid(pred0.dot(weights1)), 16, 1, axis = 1)
    pred2 = sigmoid(pred1.dot(weights2))
    return pred2

def softmax(vector): #returns the index of highest value of a vector (one hot vector)
    return np.argmax(vector, axis = 1)

def backprop(expectedOut, inputs, samples, learnRate):
    #back propagates the error and adjusts weights

    global weights0, weights1, weights2
    #np.insert to insert the bias into the the input
    inputs = np.insert(np.array(inputs).reshape(samples,784), 784, 1, axis = 1).reshape(samples,785)
    expectedOut = np.array(expectedOut).reshape((samples,10))

    pred0 = np.insert(sigmoid(inputs.dot(weights0)), 40, 1, axis = 1)
    pred1 = np.insert(sigmoid(pred0.dot(weights1)), 16, 1, axis = 1)
    pred2 = sigmoid(pred1.dot(weights2))

    meanError = np.mean(abs(expectedOut - pred2))

    dError2 = (expectedOut - pred2)
    delta2 = dError2 * sigmoid(pred2, deriv = True)
    #np.delete to delete the bias for backpropagation
    dError1 = delta2.dot(weights2.T)
    delta1 = np.delete(dError1 * sigmoid(pred1, deriv = True), 16, axis = 1)
    dError0 = delta1.dot(weights1.T)
    delta0 = np.delete(dError0 * sigmoid(pred0, deriv = True), 40, axis = 1)

    #adjust weights
    weights0 += inputs.T.dot(delta0) * learnRate
    weights1 += pred0.T.dot(delta1) * learnRate
    weights2 += pred1.T.dot(delta2) * learnRate
    return meanError

def train(inputs, outputs, sampleSize, epochSize, iterations, learningRate):

    inputs = np.array(inputs).reshape(sampleSize, 784)
    outputs = np.array(outputs).reshape(sampleSize, 10)
    for i in range(iterations):
        #calc derivatives for [epochsize] samples and subtract them from the weights and biases
        error = 0
        for epoch in range(epochSize, sampleSize, epochSize):
            error = backprop(outputs[int(epoch-epochSize):epoch], inputs[int(epoch-epochSize):epoch], epochSize, learningRate)
        """
        if i % 20 == 0:
            print("Iteration: " + str(i))
            print("Error: " + str(error))
        """
def test(outputsEnabled = False):
    pred = predict(testImgs)
    testPred = softmax(pred) # returns the "guesses" of the network
    errorsAt = np.array(np.where(np.not_equal(testLabels,testPred))).T #the samples the network was mistaken on
    if(outputsEnabled):
        for i in np.nditer(errorsAt):
            print(MNIST.display(testImgs[i]))
            print("network Output: " + str(testPred[i]))
            print("correct Output: " + str(testLabels[i]))
            print("networks whole Output: ")
            print(pred[i])

    return (1 - (len(errorsAt) / len(testPred))) * 100
print("before training: ")
print("Right guesses: " + str(test()) + "%")

train(images, output, 60000, 1000, 1000, 0.002)
print("after training: ")
print("Right guesses: " + str(test()) + "%")
