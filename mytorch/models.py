#load pacakge
import numpy as np
#creat class as Model
class Model:
    #initialize model parameter , # model parameter,# input features,# output
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.params = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros(output_dim) #initialize bias
    #forward function
    def forward(self, X):
        return np.dot(X, self.params) + self.bias
    #get parameters
    def get_params(self):
        return self.params, self.bias
    #update parameters
    def update_params(self, params, bias):
        self.params = params
        self.bias = bias

#linear regression model class
class LinearModel(Model):
    def forward(self, X):
        return super().forward(X)  

#logistic regression model class
class LogisticRegression(Model):
    def forward(self, X):
        logits = super().forward(X)
        return 1 / (1 + np.exp(-logits)) #sigmoid

# create class DenseNetwork with random weight and activation function relu first
class DenseNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu"):
        self.hidden_weights = np.random.randn(input_dim, hidden_dim) * 0.01
        self.hidden_bias = np.zeros(hidden_dim)
        self.output_weights = np.random.randn(hidden_dim, output_dim) * 0.01
        self.output_bias = np.zeros(output_dim)
        self.activation = activation.lower()
    #relu function
    def relu(self, x):
        return np.maximum(0, x)
    #sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    #softmax function
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # computing forward pass of network
    def forward(self, X):
       #compute hidden layer
        self.hidden_layer = np.dot(X, self.hidden_weights) + self.hidden_bias
        if self.activation == "relu": #apply relu activation function
            self.hidden_layer = self.relu(self.hidden_layer) 
        elif self.activation == "sigmoid": #apply sigmoid activation function 
            self.hidden_layer = self.sigmoid(self.hidden_layer)

        self.output_layer = np.dot(self.hidden_layer, self.output_weights) + self.output_bias
        return self.softmax(self.output_layer) #apply softmax function 
    
    #get parameter using function
    def get_params(self):
        return self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias
    # get network parameters: new hidden layer,new hidden bias,new output layer weight, new output layer bias 
    def update_params(self, hidden_weights, hidden_bias, output_weights, output_bias):
        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias