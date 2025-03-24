
import numpy as np
#package import
# define class Optimizer: Gradient Descent Optimizer with Backpropagation for Linear, Logistic Regression, and MLP.
class Optimizer:
  
    
    def __init__(self, model, loss_type='mse', learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_type = loss_type

    # Loss function for regression and classification(all models eventually)
    def loss_fn(self, y_pred, y_true):

        if self.loss_type == 'mse':  # Mean Squared Error (for Linear Regression)
            return np.mean((y_pred - y_true.reshape(-1, 1)) ** 2)
        elif self.loss_type == 'bce':  # Binary Cross-Entropy (for Logistic Regression)
            y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)  # Avoid log(0) errors
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_type == 'cross_entropy':  # Cross-Entropy Loss (for multi-class classification)
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[np.arange(len(y_true)), y_true] = 1
            return -np.sum(y_true_one_hot * np.log(y_pred + 1e-7)) / len(y_true)
        else:
            raise ValueError("not vaild loss function")
    # Compute gradients using backpropagation for models.
    def compute_gradient(self, X, y):
        params = self.model.get_params()

        if len(params) == 2:  # Linear and  Logistic Regression (weights, bias)
            weights, bias = params

            # Forward pass
            logits = np.dot(X, weights) + bias

            # check if it's Logistic Regression (logistic regression - sigmoid)
            if hasattr(self.model, "forward") and "logistic" in self.model.__class__.__name__.lower():
                logits = 1 / (1 + np.exp(-logits))  # Apply sigmoid activation

            loss_grad = logits - y.reshape(-1, 1)  # Derivative of MSE or BCE loss

            # Compute gradients
            grad_weights = np.dot(X.T, loss_grad) / len(y)
            grad_bias = np.mean(loss_grad, axis=0)

            return grad_weights, grad_bias

        elif len(params) == 4:  # MLP (hidden & output layers)
            hidden_weights, hidden_bias, output_weights, output_bias = params

            # Forward pass
            hidden_layer = np.maximum(0, np.dot(X, hidden_weights) + hidden_bias)  # ReLU activation
            output_layer = self.model.softmax(np.dot(hidden_layer, output_weights) + output_bias)  # Softmax

            # Compute loss derivative (cross-entropy loss)
            y_true_one_hot = np.zeros_like(output_layer)
            y_true_one_hot[np.arange(len(y)), y] = 1
            d_output = output_layer - y_true_one_hot

            # Gradients for output layer
            d_output_weights = np.dot(hidden_layer.T, d_output) / len(y)
            d_output_bias = np.mean(d_output, axis=0)

            # Backpropagate through ReLU
            d_hidden = np.dot(d_output, output_weights.T)
            d_hidden[hidden_layer <= 0] = 0  # ReLU derivative

            # Gradients for hidden layer
            d_hidden_weights = np.dot(X.T, d_hidden) / len(y)
            d_hidden_bias = np.mean(d_hidden, axis=0)

            return d_hidden_weights, d_hidden_bias, d_output_weights, d_output_bias

        else:
            raise ValueError("Model type not supported by optimizer.")

    # "Perform a gradient descent step to update model parameters
    def step(self, X, y):
        
        grads = self.compute_gradient(X, y)
        params = self.model.get_params()

        if len(params) == 2:  # Linear , Logistic Regression
            weights, bias = params
            grad_weights, grad_bias = grads

            # Gradient descent update
            weights -= self.learning_rate * grad_weights
            bias -= self.learning_rate * grad_bias

            self.model.update_params(weights, bias)

        elif len(params) == 4:  # MLP
            hidden_weights, hidden_bias, output_weights, output_bias = params
            d_hidden_weights, d_hidden_bias, d_output_weights, d_output_bias = grads

            # Gradient descent update
            hidden_weights -= self.learning_rate * d_hidden_weights
            hidden_bias -= self.learning_rate * d_hidden_bias
            output_weights -= self.learning_rate * d_output_weights
            output_bias -= self.learning_rate * d_output_bias

            self.model.update_params(hidden_weights, hidden_bias, output_weights, output_bias)
