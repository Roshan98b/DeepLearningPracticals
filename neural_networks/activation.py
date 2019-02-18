import numpy as np

# For output layer and hidden layer
# Derivative Chain Rule
# delta = d(E)/d(net_sum_output) = d(E)/d(output) * d(output)/d(net_sum_output)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + (np.e ** -x))

# ReLU - Rectified Linear Unit
def relu(x):
    return max(0, x)

# Softmax
def softmax(x):
    exp = np.exp(x)
    return np.true_divide(exp, sum(exp)).transpose()

# d(output)/d(net_sum_output)
def activation_derivative(output_vector_i, net_sum_i, activation):
    if activation == 'relu':
        return np.array([1 if i>0 else 0 for i in net_sum_i])
    elif activation == 'sigmoid':
        return output_vector_i * (1 - output_vector_i)
    elif activation == 'softmax':
        return output_vector_i * (1 - output_vector_i)
    else:
        return output_vector_i * (1 - output_vector_i)
