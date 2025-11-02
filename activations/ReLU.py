import numpy as np

from utils.logging_helper import get_logger

logger = get_logger(__name__)


class ReLU:
    def __init__(self):
        pass

    def forward(z: np.ndarray, verbose=False) -> np.ndarray:
        """
            applies relu activation fn

        Args:
            z (np.ndarray): output from conv layer
            verbose (bool): flag

        Returns:
            (np.ndarray): relu of z, with +ve values only
        """

        logger.debug(f"Relu forward - input shape: {z.shape}")

        # takes in feature maps like [ [1,2] [2,3] ]
        return np.maximum(0, z)

    def backward():
        """
            back prop for relu activation fn

        Args:

        Returns:
        """

        raise NotImplementedError("back prop not implemented for relu")
