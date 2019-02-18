import numpy as np
from neural_networks.activation import sigmoid, relu, softmax
from neural_networks.loss import mse
from neural_networks.preprocessing import one_hot_encoded
from neural_networks.optimization import neural_network_output

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
        
    # Selection of Loss Function and Optimization function
    def set_parameters(self, lr = 0.01, loss = 'mse', optimization = 'gd'):
        self.loss = loss
        self.optimization = optimization
        if loss == 'mse':
            self.loss_function = classmethod(mse)
        elif loss == 'categorical_crossentropy':
            pass
        if optimization == 'gd':
            pass
        elif optimization == 'sgd':
            pass
        elif optimization == 'rmsprop':
            pass
        elif optimization == 'adagrad':
            pass
        elif optimization == 'adam':
            pass
        self.lr = lr
        
    def activation(self, net_sum, activation):
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
    
    # Predict
    def predict(self, X_test):
        predictions = []
        classes = self.layers_info['Layer_'+str(self.num_layers)]['size']
        for record in X_test:
            predicted_output = neural_network_output(self, record)
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
            predicted_output = neural_network_output(self, record)
            loss += self.loss_function(true_output, predicted_output)        
        print('Loss: ',sum(loss)/(len(X_test) * len(loss)))
        
        # Accuracy
        true_output = []
        if len(y_test.shape) > 1:
            for i in y_test:
                true_output.append(np.argmax(i))
        else:
            true_output = y_test
        true_output = np.array(true_output)
        predicted_output = self.predict(X_test)
        count = 0
        for true, pred in zip(true_output, predicted_output):
            if (true == pred):
                count += 1
        accuracy = count/len(true_output)
        print('Accuracy: ',accuracy)
