class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x_val = init
        for iter in range(iterations):
            derivative = 2.0*x_val
            x_val -= learning_rate * derivative

        return round(x_val, 5)
