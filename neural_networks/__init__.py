import numpy as np
from neural_networks.loss import mse
from neural_networks.preprocessing import one_hot_encoded
from neural_networks.training import neural_network_output
from neural_networks.optimization import gd, sgd

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
        self.lr = lr
        if loss == 'mse':
            self.loss_function = mse
        elif loss == 'categorical_crossentropy':
            pass

    # Training
    def train(self, X_train, y_train, epochs=1, batch_size=1):
        if self.optimization == 'sgd':
            sgd(self, X_train, y_train, epochs, batch_size)
        elif self.optimization == 'gd':
            if (batch_size != 1):
                print('Number of batches by default in Gradient descent is set to 1')
            gd(self, X_train, y_train, epochs)

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
