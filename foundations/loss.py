import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        n = len(y_true)
        loss = -1/n * (np.dot(y_true, np.log(y_pred + eps)) + np.dot(1.0 - y_true, np.log(1.0 - y_pred + eps)))
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n_samples, n_classes = y_true.shape
        eps = 1e-7
        loss = 0.0
        for sample in range(n_samples):
            loss += np.dot(y_true[sample], np.log(y_pred[sample] + eps))
        return np.round(-1.0/n_samples*loss, 4)


