import numpy as np
np.random.seed(None)

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    
    # loss = reg_strength * np.sum(W**2)
    # grad = 2 * reg_strength * W
    loss = 0.5 * reg_strength * np.sum(W**2)
    grad = reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    
    logits = predictions - np.max(predictions, axis=-1, keepdims=True) 
    exp = np.exp(logits)
    soft_max = exp/np.sum(exp, axis=-1, keepdims=True)  

    # p = np.zeros(predictions.shape)
    
    # if isinstance(target_index, int):
    #     p[target_index] = 1
    # else:
    #     row_index = np.arange(target_index.shape[0])
    #     p[row_index, target_index.flatten()] = 1

    batch_size = predictions.shape[0]
    
    row_index = np.arange(target_index.shape[0])
    loss = -np.sum(np.log(soft_max[row_index, target_index]))/batch_size
    d_preds = soft_max
    d_preds[row_index, target_index] -= 1
    
    # #averaging gradient by batch
    # d_preds /= batch_size
    
    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.old_x = X.copy
        X[X < 0] == 0
        return X
        
        
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        # grad = np.zeros_like(d_out)
        # d_result = (d_out > 0).astype(d_out.dtype)
        # d_result = d_out * grad
        d_result = d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output)) #
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        grad_w = self.X.T @ d_out
        self.W.grad += grad_w
        
        grad_b = np.ones((1, d_out.shape[0])) @ d_out
        self.B.grad += grad_b
        d_result = d_out @ self.W.value.T
        
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
