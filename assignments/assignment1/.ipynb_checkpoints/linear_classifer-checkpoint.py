import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    predictions -= np.max(predictions, axis=-1, keepdims=True)
    exp_preds = np.exp(predictions)
    probs = exp_preds/np.sum(exp_preds, axis=-1, keepdims=True)
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    p = np.zeros(probs.shape)
    p[target_index] = 1
    return -np.sum(p * np.log(probs))


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops

    predictions = predictions.copy()
    predictions -= np.max(predictions, axis=-1, keepdims=True) 
    exp_preds = np.exp(predictions)
    soft_max = exp_preds/np.sum(exp_preds, axis=-1, keepdims=True)  

    p = np.zeros(predictions.shape)
    
    if isinstance(target_index, int):
        p[target_index] = 1
    else:
        row_index = np.arange(target_index.shape[0])
        p[row_index, target_index.flatten()] = 1
    
    loss = -np.sum(p * np.log(soft_max))
    
    dprediction = soft_max - p
    
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    predictions -= np.max(predictions, axis=-1, keepdims=True) 
    exp_preds = np.exp(predictions)
    soft_max = exp_preds/np.sum(exp_preds, axis=-1, keepdims=True)  

    p = np.zeros(predictions.shape)
    row_index = np.arange(target_index.shape[0])
    p[row_index, target_index.flatten()] = 1
    
    loss = -np.sum(p * np.log(soft_max))
    
    dW = np.matmul(X.T, soft_max - p)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(int(epochs)):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size, dtype=int)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for i in range(len(batches_indices)):
                loss, dW = linear_softmax(X[batches_indices[i]], self.W, y[batches_indices[i], None])
                l2_loss, l2_grad = l2_regularization(self.W, reg)
                self.W = self.W - learning_rate * dW - learning_rate * l2_grad #not sure if l2_grad should be multiplied by learnign rate
                loss = loss + l2_loss
                loss_history.append(loss)
            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
      

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        y_pred = np.argmax(np.dot(X, self.W), axis=-1)
        
        return y_pred



                
                                                          

            

                
