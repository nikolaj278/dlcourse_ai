import itertools as it
import numpy as np
from metrics import multiclass_accuracy

class diy_GridSearch:
    def __init__(self, clf, *params):
        # epochs, batch_size, learning_rates, reg_strengths
        self.param_grid = np.array(list(it.product(*params)))
        self.clf = clf
        self.clf_acc = np.zeros(self.param_grid.shape[0])
        self.best_classifier = None
        self.best_val_accuracy = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        for i, node in enumerate(self.param_grid):
            epochs, lrates, batch_size, reg = node
            self.clf.fit(X_train, y_train, epochs=int(epochs), learning_rate=lrates , batch_size=batch_size, reg=reg)
            self.clf_acc[i] = multiclass_accuracy(self.clf.predict(X_val), y_val)

        self.best_val_accuracy = self.clf_acc.max()
        self.clf.fit(X_train, y_train, *self.param_grid[np.argmax(self.clf_acc)])
        self.best_classifier = self.clf