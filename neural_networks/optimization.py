import numpy as np
from neural_networks.training import initialize_jacobian, create_batches, evaluate_gradient, accumulated_jacobian_average, print_info

# Stochastic Gradient Descent (mini batches)
def sgd(self, X_train, y_train, epochs, batch_size):
    initialize_jacobian(self, X_train)
    X_train, y_train, batches = create_batches(self, X_train, y_train, batch_size)
    for epoch in range(epochs):
        for batch in range(batches):
            for record, label in zip(X_train[batch], y_train[batch]):
                evaluate_gradient(self, record, label)
            
            # Update weights and biases
            accumulated_jacobian_average(self, record)            
            for i in range(self.num_layers-1, -1, -1):
                self.weights[i] += self.lr * self.jacobian_weights[i]
                self.biases[i] += self.lr * self.jacobian_biases[i]
        print_info(self, epoch)    

# Gradient Descent (1 Batch)
def gd(self, X_train, y_train, epochs = 1):
    initialize_jacobian(self, X_train)
    X_train, y_train = create_batches(self, X_train, y_train, len(X_train))
    for epoch in range(epochs):
        for record, label in zip(X_train, y_train):
            evaluate_gradient(self, record, label)
        
        # Update weights and biases
        accumulated_jacobian_average(self, record)            
        for i in range(self.num_layers-1, -1, -1):
            self.weights[i] += self.lr * self.jacobian_weights[i]
            self.biases[i] += self.lr * self.jacobian_biases[i]
        print_info(self, epoch)