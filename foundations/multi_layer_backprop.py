import numpy as np
from typing import List


class Solution:
    def forward_pass(self, W: NDArray[np.float64],x: NDArray[np.float64], b: NDArray[np.float64]):
        return x@W.T + b

    def relu(self, x: NDArray[np.float64]):
        return np.maximum(0, x) 

    def loss(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]):
        error = y_true - y_pred
        return np.dot(error, error)/len(y_true)

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # return dict
        results = dict()

        # convert lists to numpy arrays
        x_np = np.array(x)
        w1_np = np.array(W1)
        w2_np = np.array(W2)
        b1_np = np.array(b1)
        b2_np = np.array(b2)
        y_true_np = np.array(y_true)

        
        # first linear layer
        ll_1 = self.forward_pass(w1_np, x_np, b1_np)

        # first ReLu
        relu_ll_1 = self.relu(ll_1)

        # second linear layer (prediction of the ML perceptron)
        y_pred = self.forward_pass(w2_np, relu_ll_1, b2_np)

        # record the loss
        results['loss'] = np.round(self.loss(y_true, y_pred), 4)

        # dW2
        # apply chain rule 
        N = len(y_true)
        dL_dz2 = 2.0/N*(y_pred-y_true)
        dz2_dw2 = relu_ll_1
        results['dW2'] = np.round(np.outer(dL_dz2,dz2_dw2), 4)

        # db2
        # dL_dz2 * dz2_db2 = dL_dz2
        results['db2'] = np.round(dL_dz2, 4)

        # dW1
        dL_da1 = dL_dz2@w2_np
        mask = (ll_1 > 0).astype(float) 
        dL_dz1 = dL_da1 * mask
        results['dW1'] = np.round(np.outer(dL_dz1, x_np), 4)
        results['db1'] = np.round(dL_dz1, 4)




        return results
        





        
