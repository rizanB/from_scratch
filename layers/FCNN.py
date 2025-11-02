import numpy as np

from utils.logging_helper import get_logger

logger = get_logger(__name__)


class FCNN:
    def __init__(self, input_size, output_size):

        # as many inputs as flatten layer output, as many outputs as needed
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(
            output_size,
        )
        logger.info("FCNN layer initialized")

    def forward(self, x):
        """forward pass through fully connected nn

        Args:
            x (np.ndarray): input

        Returns:
            z: logits
        """

        # refactor: store x for backprop
        self.x = x

        logger.debug(f"FCNN forward - input shape: {self.x.shape}")

        return np.dot(self.weights, x) + self.bias

    def backward(self, grad_output, lr=0.01, verbose=True) -> None:
        """
        updates fc weights and bias and return loss grad wrt x

        maths: go backward for z = wx + b,
        dz/dw = x
        dz/db = 1

        computes three gradients : dLoss/dw, dLoss/db and dLoss/dx
        dLoss/dw and dLoss/dx used to update weights and bias of fc layer

        Loss grad wrt weight (dLoss/dw = (dLoss/dz) * (dz/dw) )

        Loss grad wrt bias (dLoss/db = (dLoss/dz) * (dz/db) )

        Loss grad wrt input (dLoss/dx) = (dLoss/dz) * w

        Args:
        grad_output (np.ndarray): grad from layer after this
        lr (float): learning rate

        Returns:
        loss_grad_wrt_x (np.ndarray): loss grad wrt input
        """

        loss_grad_wrt_w = np.outer(grad_output, self.x)

        # update fc weights
        self.weights -= lr * loss_grad_wrt_w

        loss_grad_wrt_b = grad_output

        # update fc bias
        self.bias -= lr * loss_grad_wrt_b

        loss_grad_wrt_x = np.dot(self.weights.T, grad_output)

        return loss_grad_wrt_x
