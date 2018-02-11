# Python-MNIST
This Python Script reads in the data from the MNIST database and trains
a four layer feed forward network via backpropagation and stochastic gradient descent
to recognize the digit displayed in a given Picture (28x28 pixels).

I only used numpy and python-mnist for the script to understand the maths
behind the backpropagation itself better.

Architecture:

The network has four layers.
Input Layer with 784 neurons (28x28 pixel images) + 1 bias
Hidden Layer 1 with 40 neurons + 1 bias
Hidden Layer 2 with 16 neurons + 1 bias
Output layer with 10 neurons, each corresponding to one digit

I used the sigmoid activation function for all layers, maybe results could be
improved by using other activation functions for the hidden layers.

Results:

The training Parameters I used(learning rate = 0.002, iterations = 1000, batchsize = 1000)
trained to network well enough to correctly identify about 94.7% of all test inputs
after around 15mins of training on my laptop.
