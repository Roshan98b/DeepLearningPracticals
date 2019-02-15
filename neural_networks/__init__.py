import numpy as np
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoded(y, classes = 0):
    values = y
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = values.reshape(len(values), 1)
    one_hot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return one_hot_encoded

class Model:
    
    # Constructor
    def __init__(self):
        self.x = []
        self.y = []
        self.num_layers = 0
        self.layers_info = {}
        self.layers = []
        self.net_sum = []
        self.weights = []
        self.biases = []
        self.jacobian_weights = []
        self.jacobian_biases = []
    
    # Add a fully connected layer
    def add_dense_layer(self, size, input_size = 0, activation = 'linear'):
        self.num_layers += 1
        self.layers_info['Layer_'+str(self.num_layers)] = {
            'size': size,
            'activation': activation
        }
        self.layers.append(np.ones(size))
        self.net_sum.append(np.ones(size))
        if input_size != 0:
            self.weights.append(np.random.randn(size, input_size) * np.sqrt(2/(size+input_size)))
        else:
            input_size = self.layers_info['Layer_'+str(self.num_layers-1)]['size']
            self.weights.append(np.random.randn(size, input_size) * np.sqrt(2/(size+input_size)))
        self.biases.append(np.random.randn(size) * np.sqrt(1/size))
        
    def mse(self, true, pred):
        return ((true-pred)**2)/2
        
    # Selection of Loss Function and Optimization function
    def set_parameters(self, lr = 0.01, loss = 'mse'):
        if loss == 'mse':
            self.loss = loss
            self.loss_function = self.mse
        self.lr = lr
    
    # Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + (np.e ** -x))
    
    # ReLU - Rectified Linear Unit
    def relu(self, x):
        return max(0, x)
    
    # Softmax
    def softmax(self, x):
        exp = np.exp(x)
        return np.true_divide(exp, sum(exp)).transpose()
    
    def activation(self, net_sum, activation):
        if activation == 'sigmoid':
            activation_function = self.sigmoid
            output_vector = np.array([activation_function(i) for i in net_sum]).transpose()
        elif activation == 'relu':
            activation_function = self.relu
            output_vector = np.array([activation_function(i) for i in net_sum]).transpose()
        elif activation == 'softmax':
            activation_function = self.softmax
            output_vector = activation_function(net_sum)
        return output_vector
    
    # Forward Propogation
    def neural_network_output(self, record):
        input_vector = record.transpose()
        output_vector = None
        for i in range(len(self.layers)):
            # y = aW + b
            self.net_sum[i] = np.matmul(self.weights[i], input_vector) + self.biases[i].transpose()
            self.layers[i] = self.activation(self.net_sum[i], self.layers_info['Layer_'+str(i+1)]['activation'])
            input_vector = self.layers[i]
        output_vector = self.layers[len(self.layers)-1]
        return output_vector
    
    # For output layer
    # delta = dE/dnet_sum_output = dE/doutput * doutput/dnet_sum_output
    # dE/doutput
    def output_loss_derivative(self, true, pred):
        if self.loss == 'mse':
            return (true - pred)
    
    # doutput/dnet_sum_output
    def activation_derivative(self, output_vector_i, net_sum_i, activation):
        if activation == 'relu':
            return np.array([1 if i>0 else 0 for i in net_sum_i])
        elif activation == 'sigmoid':
            return output_vector_i * (1 - output_vector_i)
        elif activation == 'softmax':
            return output_vector_i * (1 - output_vector_i)
        else:
            return output_vector_i * (1 - output_vector_i)
    
    # For hidden layer
    # dE/dnet_sum_hidden = dotproduct(weights_ji, delta_o)
    def hidden_loss_derivative(self, grad_next, layer_number, num_of_neurons_next):
        grad = []
        for i in range(num_of_neurons_next):
            weights_next = self.weights[layer_number+1][:, i]
            grad.append(np.dot(weights_next, grad_next))
        return np.array(grad)
    
    # Training using Backpropogation
    def train(self, X_train, y_train, epochs = 1, batch_size = 0): 
        
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
        
        # Creation of batches
        self.x = X_train
        self.y = y_train
        if batch_size == 0:
            batch_size = len(y_train)
        batches = int(abs(len(X_train) / batch_size))
        X_train = np.array_split(X_train, batches)
        y_train = np.array_split(y_train, batches)
        
        for epoch in range(epochs): 
            for batch in range(batches):
                for record, label in zip(X_train[batch], y_train[batch]):                    
                    true_output = np.array(label)
                    predicted_output = self.neural_network_output(np.array(record))

                    # gradient of output layer
                    output_vector_i = self.layers[self.num_layers-1]
                    net_sum_i = self.net_sum[self.num_layers-1]
                    activation = self.layers_info['Layer_'+str(self.num_layers)]['activation']
                    # derivative of output loss * derivetive of activation
                    grad_o = self.output_loss_derivative(true_output, predicted_output) * self.activation_derivative(output_vector_i, net_sum_i, activation)
                    
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
                        loss_grad = self.hidden_loss_derivative(grad_next, i, num_of_neurons_next)
                        activation_grad = self.activation_derivative(output_vector_i, net_sum_i, activation) 
                        grad_h = np.tensordot(loss_grad, activation_grad, axes=0)[0]
                        
                        # Jacobian Hidden layer - Jw = input * grad_next and Jb = 1.0 * grad_next
                        if i-1 < 0:
                            self.jacobian_weights[i] += grad_h[:, None] @ np.array(record)[None, :]
                        else:
                            self.jacobian_weights[i] += grad_h[:, None] @ self.layers[i-1][None, :]
                        self.jacobian_biases[i] += grad_h
                        
                        # change the gradient
                        grad_next = grad_h
                
                # Divide accumulated jacobian by number of records in the batch
                for i in range(self.num_layers):
                    self.jacobian_weights[i] = self.jacobian_weights[i] / len(record)
                    self.jacobian_biases[i] = self.jacobian_biases[i] / len(record)
                
                # Update weights and biases
                for i in range(self.num_layers-1, -1, -1):
                    self.weights[i] += self.lr * self.jacobian_weights[i]
                    self.biases[i] += self.lr * self.jacobian_biases[i]
            
            # Accuracy and Loss
            print('Epoch ',epoch)
            self.evaluate(self.x, self.y)
            print('')
            
    
    # Predict
    def predict(self, X_test):
        predictions = []
        classes = self.layers_info['Layer_'+str(self.num_layers)]['size']
        for record in X_test:
            predicted_output = self.neural_network_output(record)
            if classes == 1:
                if predicted_output[0] >= 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)
            else:
                predict = np.argmax(predicted_output)
                predictions.append(predict)
        return predictions
    
    # Accuracy and Loss
    def evaluate(self, X_test, y_test):       
        # Loss
        loss = 0
        for record, label in zip(X_test, y_test):
            true_output = np.array(label)
            predicted_output = self.neural_network_output(record)
            loss += self.loss_function(true_output, predicted_output)        
        print('Loss: ',sum(loss)/(len(X_test) * len(loss)))
        
        # Accuracy
        true_output = []
        if len(y_test.shape) > 1:
            for i in y_test:
                true_output.append(np.argmax(i))
        true_output = np.array(true_output)
        predicted_output = self.predict(X_test)
        count = 0
        for true, pred in zip(true_output, predicted_output):
            if (true == pred):
                count += 1
        accuracy = count/len(true_output)
        print('Accuracy: ',accuracy)
