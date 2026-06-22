import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    def loss(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]):
        # compute cross entropy loss
        return -1.0/len(model_prediction)*np.dot(model_prediction, np.log(ground_truth)) + np.dot(1.0 - model_prediction, np.log(1.0 - ground_truth))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)

        # number of weights
        n_weights = len(initial_weights)
        
        for iter in range(num_iterations):
            # forward layer to get the predictions
            x_pred = self.get_model_prediction(X, initial_weights)

            # get the derivative
            for i in range(n_weights):
                initial_weights[i] -= self.learning_rate*self.get_derivative(x_pred, Y, len(x_pred), X, i)

        return np.round(initial_weights, 5)

