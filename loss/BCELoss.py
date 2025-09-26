import numpy as np


class BCELoss:
    def __init__(self):
        pass

    def forward(self, y_pred, y):
        """calculates binary cross entropy error loss between predicted and actual output

        Args:
            y_pred (np.ndarray): predicted output
            y (np.ndarray): true output

        Returns:
            (np.ndarray): loss
        """

        raise NotImplementedError("bceloss not implemented")

    def backward(self, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("back prop not implemented for bceloss")
