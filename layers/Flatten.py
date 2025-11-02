import numpy as np

from utils.logging_helper import setup_logger

logger = setup_logger(__name__)


class Flatten:
    def __init__(self):
        self.x_shape = None
        logger.info("Flatten layer initialized")

    def forward(self, x: np.ndarray, verbose=False):
        """
        flattens input array

        Args:
            x (np.ndarray): input array with shape num_filters, feature_map_size
            verbose (bool): flag

        Returns:
            np.ndarray: flattened 1d array

        Raises:
            ValueError: if input array is empty
        """

        self.x_shape = x.shape
        logger.debug(f"Flatten forward - input shape: {self.x_shape}")

        if x.size == 0:
            logger.error(
                f"Flatten forward - received empty input with shape {self.x_shape}"
            )
            raise ValueError("Cannot flatten empty array")

        flattened_x = x.flatten()
        logger.debug(f"Flatten forward- output shape: {flattened_x.shape}")

        return flattened_x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """reshapes grad_output to work with layer before it

        Args:
            grad_output (np.ndarray): grad from layer after this

        Returns:
            np.ndarray: reshaped grads

        Raises:
            RuntimeError: if backward() is called before forward()
        """

        if self.x_shape is None:
            logger.error(
                "Flatten backward reshape error: x_shape is None. call forward() first"
            )
            raise RuntimeError("forward() must be called before backward()")

        logger.debug(
            f"Flatten backward - reshaping from {grad_output.shape} to {self.x_shape}"
        )

        return grad_output.reshape(self.x_shape)
