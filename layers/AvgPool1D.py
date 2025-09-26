import numpy as np


class AvgPool1D:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        performs avg pooling

        Args:
            x (np.ndarray): input array of shape (num_features, feature_map_size)

        Returns:
            np.ndarray: pooled output
        """

        pooling_output = []

        for fmap in x:
            pooled = []
            for i in range(0, len(fmap), self.kernel_size):
                pooled.append(np.average(fmap[i : i + self.kernel_size]))
            pooling_output.append(pooled)

        return np.array(pooling_output)

    def backward(self):
        raise NotImplementedError("back prop not implemented for avgpool1d")
