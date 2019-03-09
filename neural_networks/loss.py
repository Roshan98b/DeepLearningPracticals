import numpy as np
from math import log

# For output layer and hidden layer
# Derivative Chain Rule
# delta = d(E)/d(net_sum_output) = d(E)/d(output) * d(output)/d(net_sum_output)

# Mean Square Error Loss
def mse(true, pred):
    return ((true-pred)**2)/2

# Cross Entropy
def cross_entropy(true, pred):
    return -true*log(pred)

# For output layer
# d(E)/d(output) = derivative of loss function
def output_loss_derivative(loss, true, pred):
    if loss == 'mse':
        return -(true-pred)
    elif loss == 'cross_entropy':
        return -true/pred 
    else:
        pass

# For hidden layer
# d(E)/d(output) = dotproduct(weights_ji, delta_o)
def hidden_loss_derivative(self, grad_next, layer_number, num_of_neurons_next):
    grad = []
    for i in range(num_of_neurons_next):
        weights_next = self.weights[layer_number+1][:, i]
        grad.append(np.dot(weights_next, grad_next))
    return np.array(grad)
