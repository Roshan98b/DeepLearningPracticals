import numpy as np
from neural_networks.training import initialize_jacobian, initialize_velocity, initialize_squared_gradient, create_batches, evaluate_gradient, accumulated_jacobian_average, print_info

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
            self.weights[i] += -self.lr * self.jacobian_weights[i]
            self.biases[i] += -self.lr * self.jacobian_biases[i]
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
                self.weights[i] += -self.lr * self.jacobian_weights[i]
                self.biases[i] += -self.lr * self.jacobian_biases[i]
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
   
            # Update weights and biases with velocity
            # velocity_prev = velocity
            # velocity = m*velocity - lr*jacobian
            # weights += -m*velocity_prev + (1+m)*velocity
            accumulated_jacobian_average(self, record)            
            for i in range(self.num_layers-1, -1, -1):
                # weights
                velocity_w = self.velocity_weights[i]
                self.velocity_weights[i] = (self.m * self.velocity_weights[i]) - (self.lr * self.jacobian_weights[i])
                self.weights[i] += (-self.m * velocity_w) + ((1+self.m) * self.velocity_weights[i])
                # biases
                velocity_b = self.velocity_biases[i]
                self.velocity_biases[i] = (self.m * self.velocity_biases[i]) - (self.lr * self.jacobian_biases[i])
                self.biases[i] += (-self.m * velocity_b) + ((1+self.m) * self.velocity_biases[i])
        print_info(self, epoch)

# Adagrad
def adagrad(self, X_train, y_train, epochs, batch_size):
    x = X_train
    eps = np.e**-7
    X_train, y_train, batches = create_batches(self, X_train, y_train, batch_size)
    for epoch in range(epochs):
        for batch in range(batches):
            initialize_squared_gradient(self, x)
            initialize_jacobian(self, x)
            for record, label in zip(X_train[batch], y_train[batch]):
                evaluate_gradient(self, record, label)
   
            # Update weights and biases with sum of squared gradient
            # sum_squared_gradient += jacobian**2
            # weights += -lr*jacobian / (sqrt(sum_squared_gradient)+eps)
            accumulated_jacobian_average(self, record)            
            for i in range(self.num_layers-1, -1, -1):
                # Second order gradient (jacobian**2)
                self.squared_gradient_weights[i] += self.jacobian_weights[i]**2
                self.squared_gradient_biases[i] += self.jacobian_biases[i]**2
                # First order gradient (jacobian) 
                self.weights[i] += -self.lr * self.jacobian_weights[i] / (np.sqrt(self.squared_gradient_weights[i]) + eps)
                self.biases[i] += -self.lr * self.jacobian_biases[i] / (np.sqrt(self.squared_gradient_biases[i]) + eps)
        print_info(self, epoch)

# RMSProp
def rmsprop(self, X_train, y_train, epochs, batch_size):
    x = X_train
    eps = np.e**-6
    X_train, y_train, batches = create_batches(self, X_train, y_train, batch_size)
    for epoch in range(epochs):
        for batch in range(batches):
            initialize_squared_gradient(self, x)
            initialize_jacobian(self, x)
            for record, label in zip(X_train[batch], y_train[batch]):
                evaluate_gradient(self, record, label)
   
            # Update weights and biases with exponential moving average of squared gradient
            # exp_moving_average_squared_gradient = beta*exp_moving_average_squared_gradient + (1+beta)*jacobian**2
            # weights += -lr*jacobian / (sqrt(exp_moving_average_squared_gradient)+eps)
            accumulated_jacobian_average(self, record)            
            for i in range(self.num_layers-1, -1, -1):
                # Second order gradient (jacobian**2)
                self.squared_gradient_weights[i] = (self.d * self.squared_gradient_weights[i]) + ((1-self.d) * self.jacobian_weights[i]**2)
                self.squared_gradient_biases[i] = (self.d * self.squared_gradient_biases[i]) + ((1-self.d) * self.jacobian_biases[i]**2)
                # First oredr gradient (jacobian)
                self.weights[i] += -self.lr * self.jacobian_weights[i] / (np.sqrt(self.squared_gradient_weights[i]) + eps)
                self.biases[i] += -self.lr * self.jacobian_biases[i] / (np.sqrt(self.squared_gradient_biases[i]) + eps)
        print_info(self, epoch)

# Adam
def adam(self, X_train, y_train, epochs, batch_size):
    x = X_train
    eps = np.e**-8
    t=0
    X_train, y_train, batches = create_batches(self, X_train, y_train, batch_size)
    for epoch in range(epochs):
        t += 1
        for batch in range(batches):
            initialize_velocity(self, x)
            initialize_squared_gradient(self, x)
            initialize_jacobian(self, x)
            for record, label in zip(X_train[batch], y_train[batch]):
                evaluate_gradient(self, record, label)
   
            # Update weights and biases with velocity of gradient and exponential moving average of squared gradient
            # velocity_gradient = m*velocity_gradient + (1+m)*jacobian
            # exp_moving_average_squared_gradient = d*exp_moving_average_squared_gradient + (1+d)*jacobian**2
            # weights += -lr * velocity_gradient / (sqrt(exp_moving_average_squared_gradient)+eps)
            accumulated_jacobian_average(self, record)            
            for i in range(self.num_layers-1, -1, -1):
                # Second order gradient (jacobian**2)
                self.squared_gradient_weights[i] = (self.d * self.squared_gradient_weights[i]) + ((1-self.d) * self.jacobian_weights[i]**2)
                self.squared_gradient_biases[i] = (self.d * self.squared_gradient_biases[i]) + ((1-self.d) * self.jacobian_biases[i]**2)
                # Squared gradient corrected
                sw_corrected = self.squared_gradient_weights[i]/(1-self.d**t)
                sb_corrected = self.squared_gradient_biases[i]/(1-self.d**t)
                # First oredr gradient (jacobian)
                self.velocity_weights[i] = (self.m * self.velocity_weights[i]) + ((1-self.m) * self.jacobian_weights[i])
                self.velocity_biases[i] = (self.m * self.velocity_biases[i]) + ((1-self.m) * self.jacobian_biases[i])
                # Velocity correction
                vw_corrected = self.velocity_weights[i]/(1-self.m**t)
                vb_corrected = self.velocity_biases[i]/(1-self.m**t)
                # Update
                self.weights[i] += (-self.lr * vw_corrected / (np.sqrt(sw_corrected) + eps))
                self.biases[i] += (-self.lr * vb_corrected / (np.sqrt(sb_corrected) + eps))
        print_info(self, epoch)