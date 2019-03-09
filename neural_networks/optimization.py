import numpy as np
from neural_networks.training import initialize_jacobian, initialize_velocity, create_batches, evaluate_gradient, accumulated_jacobian_average, print_info

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

# Stochastic Gradient Descent (mini batches)
def sgd(self, X_train, y_train, epochs, batch_size):
    x = X_train
    X_train, y_train, batches = create_batches(self, X_train, y_train, batch_size)
    for epoch in range(epochs):
        for batch in range(batches):
            initialize_jacobian(self, x)
            for record, label in zip(X_train[batch], y_train[batch]):
                evaluate_gradient(self, record, label)
            
            # Update weights and biases
            accumulated_jacobian_average(self, record)            
            for i in range(self.num_layers-1, -1, -1):
                self.weights[i] += self.lr * self.jacobian_weights[i]
                self.biases[i] += self.lr * self.jacobian_biases[i]
        print_info(self, epoch)    

# Nestrov's Accelerated Momentum
def nam(self, X_train, y_train, epochs, batch_size):
    x = X_train
    X_train, y_train, batches = create_batches(self, X_train, y_train, batch_size)
    for epoch in range(epochs):
        for batch in range(batches):
            initialize_velocity(self, x)
            initialize_jacobian(self, x)
            for record, label in zip(X_train[batch], y_train[batch]):
                evaluate_gradient(self, record, label)
            
            # Update weights and biases with new velocity
            accumulated_jacobian_average(self, record)            
            for i in range(self.num_layers-1, -1, -1):
                # weights
                velocity_w = self.velocity_weights[i]
                self.velocity_weights[i] = (self.m * self.velocity_weights[i]) + (self.lr * self.jacobian_weights[i])
                self.weights[i] += (-self.m * velocity_w) + ((1+self.m) * self.velocity_weights[i])
                # biases
                velocity_b = self.velocity_biases[i]
                self.velocity_biases[i] = (self.m * self.velocity_biases[i]) + (self.lr * self.jacobian_biases[i])
                self.biases[i] += (-self.m * velocity_b) + ((1+self.m) * self.velocity_biases[i])
        print_info(self, epoch)

# Adagrad
# Adadelta
# RMSProp
# Adam