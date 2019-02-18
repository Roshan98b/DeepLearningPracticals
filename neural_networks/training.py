import numpy as np
from neural_networks.loss import output_loss_derivative, hidden_loss_derivative
from neural_networks.activation import sigmoid, relu, softmax, activation_derivative

# Creation of Jacobian Matrix
def initialize_jacobian(self, X_train):
    # initializing Jacobian
    for i in range(self.num_layers):
        size = self.layers_info['Layer_'+str(i+1)]['size']            
        input_size = 0
        self.jacobian_biases.append(np.zeros((size)))
        if i == 0:
            input_size = X_train.shape[1]
        else:
            input_size = len(self.layers[i-1])
        self.jacobian_weights.append(np.zeros((size, input_size)))

# Create Training data based on branches    
def create_batches(self, X_train, y_train, batch_size):
    # Creation of batches
    self.x = X_train
    self.y = y_train
    batches = int(abs(len(X_train) / batch_size))
    if batches == 1:
        return X_train, y_train
    else:    
        X_train = np.array_split(X_train, batches)
        y_train = np.array_split(y_train, batches)
        return X_train, y_train, batches

# Activation function selection        
def activation(net_sum, activation):
    if activation == 'sigmoid':
        activation_function = sigmoid
        output_vector = np.array([activation_function(i) for i in net_sum]).transpose()
    elif activation == 'relu':
        activation_function = relu
        output_vector = np.array([activation_function(i) for i in net_sum]).transpose()
    elif activation == 'softmax':
        activation_function = softmax
        output_vector = activation_function(net_sum)
    return output_vector

# Forward Propogation
def neural_network_output(self, record):
    input_vector = record.transpose()
    output_vector = None
    for i in range(len(self.layers)):
        # y = aW + b
        self.net_sum[i] = np.matmul(self.weights[i], input_vector) + self.biases[i].transpose()
        self.layers[i] = activation(self.net_sum[i], self.layers_info['Layer_'+str(i+1)]['activation'])
        input_vector = self.layers[i]
    output_vector = self.layers[len(self.layers)-1]
    return output_vector

# Training using Backpropogation
def evaluate_gradient(self, record, label): 
                    
    true_output = np.array(label)
    predicted_output = neural_network_output(self, np.array(record))

    # gradient of output layer
    output_vector_i = self.layers[self.num_layers-1]
    net_sum_i = self.net_sum[self.num_layers-1]
    activation = self.layers_info['Layer_'+str(self.num_layers)]['activation']
    # derivative of output loss * derivetive of activation
    grad_o = output_loss_derivative(self.loss, true_output, predicted_output) * activation_derivative(output_vector_i, net_sum_i, activation)
    
    # Jacobian Output Layer - Jw = input * grad_o and Jb = 1.0 * grad_o
    if self.num_layers-2 < 0:
        self.jacobian_weights[self.num_layers-1] += grad_o[:, None] @ np.array(record)[None, :]
    else:
        self.jacobian_weights[self.num_layers-1] += grad_o[:, None] @ self.layers[self.num_layers-2][None, :]
    self.jacobian_biases[self.num_layers-1] += grad_o
    
    # gradient of hidden layer
    grad_next = grad_o
    for i in range(self.num_layers-2, -1, -1):
        output_vector_i = self.layers[i]
        net_sum_i = self.layers[i]
        activation = self.layers_info['Layer_'+str(i+1)]['activation']
        num_of_neurons_next = self.layers_info['Layer_'+str(i+2)]['size']
        # derivative of hidden loss * derivetive of activation
        loss_grad = hidden_loss_derivative(self, grad_next, i, num_of_neurons_next)
        activation_grad = activation_derivative(output_vector_i, net_sum_i, activation) 
        grad_h = np.tensordot(loss_grad, activation_grad, axes=0)[0]
        
        # Jacobian Hidden layer - Jw = input * grad_next and Jb = 1.0 * grad_next
        if i-1 < 0:
            self.jacobian_weights[i] += grad_h[:, None] @ np.array(record)[None, :]
        else:
            self.jacobian_weights[i] += grad_h[:, None] @ self.layers[i-1][None, :]
        self.jacobian_biases[i] += grad_h
        
        # change the gradient
        grad_next = grad_h

# Divide accumulated jacobian by batch size
def accumulated_jacobian_average(self, record):
    # Divide accumulated jacobian by number of records in the batch
    for i in range(self.num_layers):
        self.jacobian_weights[i] = self.jacobian_weights[i] / len(record)
        self.jacobian_biases[i] = self.jacobian_biases[i] / len(record)
        
# Print loss and accuracy for current epoch
def print_info(self, epoch):
    # Accuracy and Loss
    print('Epoch ',epoch)
    self.evaluate(self.x, self.y)
    print('')
