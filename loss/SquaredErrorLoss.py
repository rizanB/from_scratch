import numpy as np


class SquaredErrorLoss:
    def __init__(self):
        pass

    def forward(self, y_pred, y):
        """calculates squared error loss between predicted and actual output

        Args:
            y_pred (np.ndarray): predicted output
            y (np.ndarray): true output

        Returns:
            loss (np.ndarray): loss
        """

        loss = (y_pred - y) ** 2

        return loss

    def backward(self, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y)
