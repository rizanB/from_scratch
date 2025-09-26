import numpy as np

from utils.printv import printv


class MaxPool1D:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size
        self.x_shape = None

        # store indices of max fmap values
        self.mask_fmaps = []

    def forward(self, x: np.ndarray, verbose=False) -> np.ndarray:
        """performs max pooling and tracks pooled indices:

        takes in feature maps like [ [1 2 3 4 5] [2 3 4 3 1] ]
            for each fmap,
                look at a small window: eg. [1 2]
                    tracks index of max value in the window; maxidx
            gets actual index of max value; maxidx + i
            adds actual index of max value to maxpool_mask;

        Args:
            x (np.ndarray): input array of shape (num_features, feature_map_size)
            verbose (bool): flag

        Returns:
            np.ndarray: pooled output
        """
        pooling_output = []
        self.x_shape = x.shape
        # fixes bug: by resetting feature masks for every forward pass
        self.mask_fmaps = []

        printv(f"x going into maxpool1d: {x}, shape: {self.x_shape}", verbose)

        for fmap in x:
            pooled = []

            # store indices of max elements for a single fmap
            mask_fmap = []

            for i in range(0, len(fmap), self.kernel_size):
                window = fmap[i : i + self.kernel_size]
                # print(f"window is: {window}")
                maxidx_in_window = np.argmax(window)
                # print(f"index of max value inside the window is: {maxidx_in_window}")
                # print(f"actual index of max value in fmap is: {i+maxidx_in_window}")
                # print(f"max is: {max(fmap[i: i+self.kernel_size]) }")
                true_maxidx = i + maxidx_in_window

                mask_fmap.append(true_maxidx)
                pooled.append(max(fmap[i : i + self.kernel_size]))
            # print(f"mask_fmap:  {mask_fmap}")

            pooling_output.append(pooled)
            self.mask_fmaps.append(mask_fmap)

        printv(f"mask stored in maxpool class mask_fmaps: {self.mask_fmaps}", verbose)
        printv(
            f"output from maxpool1d: {pooling_output}, shape: {np.array(pooling_output).shape}",
            verbose,
        )

        return np.array(pooling_output)

    def backward(self, grad_output: np.ndarray):
        """back prop for maxpool layer

        reshapes grads by looking at indices of max values stored in self.mask_fmaps

        Args:
            grad_output (np.ndarray): grads from layer after this

        Returns:
            (np.ndarray): grad input for previous layer
        """
        grad_input = np.zeros(self.x_shape)  # same shape as input to maxpool
        for fmap_idx, fmap_mask in enumerate(self.mask_fmaps):
            for out_idx, input_idx in enumerate(fmap_mask):
                grad_input[fmap_idx, input_idx] = grad_output[fmap_idx, out_idx]

        return grad_input
