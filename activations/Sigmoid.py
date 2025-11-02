import numpy as np

from utils.logging_helper import get_logger

logger = get_logger(__name__)


class Sigmoid:
    def __init__(self):
        self.sigmoid_out = None

    def forward(self, z: np.ndarray, verbose=False):
        """
        applies sigmoid activation fn

        Args:
        z (np.ndarray): output from conv layer
        verbose (bool): flag

        Returns:
        sigmoid (np.ndarray): sigmoid of z, with values between 0 and 1
        """

        logger.debug(f"Sigmoid forward - input shape: {z.shape}")

        self.sigmoid_out = np.where(
            z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z))
        )
        return self.sigmoid_out

    def backward(self, grad_output) -> np.ndarray:
        """back pass through sigmoid

        Args:
            y_pred: predicted output
            y: true output

        Returns:
            Loss gradient wrt z
        """
        if self.sigmoid_out is None:
            raise ValueError("forward method hasn't been called before backward")

        return grad_output * self.sigmoid_out * (1 - self.sigmoid_out)
