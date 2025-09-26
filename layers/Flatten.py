import numpy as np

from utils.printv import printv


class Flatten:
    def __init__(self):
        self.x_shape = None

    def forward(self, x: np.ndarray, verbose=False):
        """
        flattens input array

        Args:
            x (np.ndarray): input array with shape num_filters, feature_map_size
            verbose (bool): flag

        Returns:
            np.ndarray: flattened 1d array
        """
        self.x_shape = x.shape

        printv(f"x is: {x}", verbose)
        printv(f"shape of x is: {self.x_shape}", verbose)

        flattened_x = x.flatten()
        printv(
            f"shape after flatten is: {flattened_x.shape}, x after flatten is: {flattened_x}",
            verbose,
        )

        printv(
            f"after reshaping, shape of x is: {flattened_x.reshape(self.x_shape).shape}, reshaped x is: {flattened_x.reshape(self.x_shape)}",
            verbose,
        )
        return flattened_x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """back prop for flatten layer, reshapes grad_output to work with layer before it

        Args:
            grad_output (np.ndarray): grad from layer after this

        Returns:
            np.ndarray: reshaped grads
        """
        return grad_output.reshape(self.x_shape)
