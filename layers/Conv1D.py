import numpy as np
from utils.pad_input import pad_input
from utils.printv import printv


class Conv1D:
    def __init__(
        self,
        num_filters=2,
        filter_size=3,
        stride=1,
        pooling_kernel_size=2,
    ):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, filter_size)
        self.stride = stride
        self.pooling_kernel_size = pooling_kernel_size
        self.bias = np.zeros(num_filters)

    # x = [1, 2, 3, 4, 5], filters = sth like [[1 2] [ 2 3]]
    def forward(self, x: np.ndarray, verbose=False) -> np.ndarray:
        """forward pass for conv1d

        Args:
            x (np.ndarray): input
            verbose: flag

        Returns:
            np.ndarray: array with feature maps
        """

        self.x = x

        len_x = len(x)
        k = self.filter_size

        # add padding before conv; so output is same size as input after conv
        padded_x = pad_input(x, self.filters[0])
        self.padded_x = padded_x

        pad_total = len(padded_x) - len(x)
        self.pad_left = pad_total // 2
        self.pad_right = pad_total - self.pad_left

        feature_maps = []

        for f_idx, f in enumerate(self.filters):
            conv_out = []

            for i in range(0, len(padded_x) - k + 1, self.stride):
                conv_out.append(np.dot(f, padded_x[i : i + k]) + self.bias[f_idx])

            feature_maps.append(conv_out)

        out = np.array(feature_maps)
        return out

    def backward(self, grad_output, lr=0.01, verbose=False):
        """computes grads wrt filters and returns grads wrt input

        grad_wrt_filter = cross correlation between input windows and grad_output
        grad_wrt_bias = grad_output
        grad_wrt_input = conv of grad_output wrt flipped filter

        Args:
            grad_output (np.ndarray) of shape (num_filters, output_length): grad output from next layer
            lr: learning rate


        Returns:
            grad_wrt_input: grads wrt input, same length as x
        """

        k = self.filter_size
        padded_len = len(self.padded_x)
        dx_padded = np.zeros(padded_len)
        grad_wrt_filter = np.zeros_like(self.filters)
        grad_wrt_bias = np.zeros_like(self.bias)

        # compute gradients
        for f_idx, f in enumerate(self.filters):
            for i in range(grad_output.shape[1]):
                start = i
                end = start + k
                grad_wrt_bias[f_idx] += grad_output[f_idx, i]
                grad_wrt_filter[f_idx] += (
                    grad_output[f_idx, i] * self.padded_x[start:end]
                )
                dx_padded[start:end] += grad_output[f_idx, i] * f

        grad_wrt_input = dx_padded[self.pad_left : len(dx_padded) - self.pad_right]

        # update weights and bias
        self.filters -= lr * grad_wrt_filter
        self.bias -= lr * grad_wrt_bias

        printv(f"grad wrt filters: {grad_wrt_filter}", verbose)
        printv(f"grad wrt bias: {grad_wrt_bias}", verbose)
        printv(
            f"grad wrt input: {grad_wrt_input}, length: {len(grad_wrt_input)}", verbose
        )

        return grad_wrt_input
