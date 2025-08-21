import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.n_input = n_input
        self.n_output = n_output
        
        # TODO Create necessary layers
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLU = ReLULayer()
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params()
        for param in params.values():
            param.grad = np.zeros_like(param.grad)
            
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        preds = self.second_layer.forward(self.ReLU.forward(self.first_layer.forward(X)))
        #--------------
        # print("[debug] max|preds| of this batch =", np.max(np.abs(preds)))
        #--------------
        loss, d_preds = softmax_with_cross_entropy(predictions=preds, target_index=y)
        d_second = self.second_layer.backward(d_preds)
        d_relu = self.ReLU.backward(d_second)
        d_first = self.first_layer.backward(d_relu)
        
        # params['W_2'] =  
        # params['B_2'] = 
        # params['W_1'] =  
        # params['B_1'] = 
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in [params['W_2'], params['W_1']]: # in params.values(): 
            param_loss, param_grad = l2_regularization(W=param.value, reg_strength=self.reg) 
            param.grad += param_grad
            loss += param_loss
            
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        # pred = np.zeros(X.shape[0], np.int)

        preds = np.argmax(self.second_layer.forward(self.ReLU.forward(self.first_layer.forward(X))), axis=-1)
        return preds

    def params(self):
        result = {'W_1': self.first_layer.W, 
                  'W_2': self.second_layer.W, 
                  'B_1': self.first_layer.B, 
                  'B_2': self.second_layer.B
                 }

        # TODO Implement aggregating all of the params

        return result
