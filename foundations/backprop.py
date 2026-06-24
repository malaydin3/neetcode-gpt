import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:

    def forward_pass(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float):
            return np.dot(w,x)+b
    def sigmoid(self, score: float):
            return 1.0/(1.0 + np.exp(-score))
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        # apply chain rule to get the derivatives
        # remember that derivative of sigmoid(x) = sigmoid(x) * (1-sigmoid(x))
        y_pred =  self.sigmoid(self.forward_pass(x, w, b))
        dl_db = (y_pred - y_true)*y_pred*(1.0-y_pred)
        dl_dw = dl_db*x
        return (np.round(dl_dw, 5), np.round(dl_db, 5))
