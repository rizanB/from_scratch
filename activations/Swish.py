import numpy as np
from activations.Sigmoid import Sigmoid


class Swish:
    def __init__(self):
        self.sigmoid = Sigmoid()

    def forward(self, z: np.ndarray, verbose=False) -> np.ndarray:
        """
        swish activation function

        Args:
        x (np.ndarray): input array

        Returns:
        swish (np.ndarray): swish activated array
        """

        return z * self.sigmoid.forward(z)

    def backward():
        """
            back prop for swish activation fn

        Args:

        Returns:
        """

        raise NotImplementedError("back prop not implemented for swish")
