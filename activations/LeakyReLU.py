import numpy as np

from utils.logging_helper import get_logger

logger = get_logger(__name__)


class LeakyReLU:
    def __init__(self):
        pass

    def forward(z: np.ndarray, alpha=0.01, verbose=False) -> np.ndarray:
        """applies leaky relu activation fn

        Args:
            z (np.ndarray): input array
            alpha (float): slope of neg section of fn

        Returns:
            (np.ndarray)
        """

        logger.debug(f"Leakyrelu forward - input shape: {z.shape}")

        return np.where(z < 0, alpha * z, z)

    def backward():
        """back prop for leaky relu activation fn

        Args:

        Returns:
        """

        raise NotImplementedError("back prop not implemented for leaky relu")
