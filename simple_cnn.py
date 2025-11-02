import numpy as np

from activations.Sigmoid import Sigmoid
from layers.Conv1D import Conv1D
from layers.FCNN import FCNN
from layers.Flatten import Flatten
from layers.MaxPool1D import MaxPool1D
from loss.SquaredErrorLoss import SquaredErrorLoss
from utils.logging_helper import setup_logger

logger = setup_logger(__name__)


class Simple1DCNN:

    def __init__(self, num_filters=2, filter_size=3, stride=1, pooling_kernel_size=2):

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, filter_size)
        self.stride = stride
        self.pooling_kernel_size = pooling_kernel_size

        params = {
            "filters": self.num_filters,
            "filter_size": self.filter_size,
            "stride": self.stride,
            "pool_kernel_size": self.pooling_kernel_size,
        }

        logger.debug(f"Simple1DCNN init - {params}")

        # bug fix: fixes fc weights not being random everytime
        self.conv1d = Conv1D(
            num_filters=num_filters, filter_size=filter_size, stride=stride
        )
        self.sigmoid_after_conv = Sigmoid()
        self.maxpool1d = MaxPool1D(kernel_size=pooling_kernel_size)
        self.fc = None
        self.sigmoid_after_fc = Sigmoid()
        self.flatten = Flatten()
        self.loss_fn = SquaredErrorLoss()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, filters: {self.filters}"

    def forward(self, x, verbose=False):
        """performs forward pass for 1dcnn

        Args:
            x (np.ndarray): input
            verbose (bool, optional): flag

        Returns:
            y_pred (np.ndarray): predicted output

        Raises:
            ValueError: if input array is empty
        """

        logger.debug(f"Simple1DCNN forward - input shape: {x.shape}")

        conv_1d_out = self.conv1d.forward(x, verbose=verbose)

        logger.debug(f"Simple1DCNN conv1D forward - output: {conv_1d_out.shape}")

        activation_output = self.sigmoid_after_conv.forward(
            z=conv_1d_out, verbose=verbose
        )

        logger.debug(f"Simple1DCNN sigmoid forward - output: {activation_output.shape}")

        pool_1d_out = self.maxpool1d.forward(activation_output)
        logger.debug(f"Simple1DCNN maxpool1d forward - output: {pool_1d_out.shape}")

        flattened_out = self.flatten.forward(pool_1d_out)
        self.flattened_out = flattened_out
        logger.debug(f"Simple1DCNN flatten forward - output: {flattened_out.shape}")

        # runs only once, fixes bug: fc weights being random everytime
        if self.fc is None:
            self.fc = FCNN(input_size=len(flattened_out), output_size=1)

        fcnn_out = self.fc.forward(flattened_out)

        logger.debug(f"Simple1DCNN fcnn forward - output: {fcnn_out.shape}")

        y_pred = self.sigmoid_after_fc.forward(fcnn_out, verbose)

        logger.debug(f"Simple1DCNN forward - target shape: {y_pred.shape} ")
        return y_pred

    def backward(self, x, y_pred, y, lr, verbose=False):

        loss_grad_wrt_pred = self.loss_fn.backward(y_pred=y_pred, y=y)

        loss_grad_wrt_z = self.sigmoid_after_fc.backward(loss_grad_wrt_pred)

        loss_grad_wrt_x = self.fc.backward(grad_output=loss_grad_wrt_z, verbose=False)

        loss_grad_from_flatten = self.flatten.backward(loss_grad_wrt_x)

        loss_grad_from_maxpool1d = self.maxpool1d.backward(loss_grad_from_flatten)

        loss_grad_from_sigmoid_after_conv = self.sigmoid_after_conv.backward(
            loss_grad_from_maxpool1d
        )

        self.conv1d.backward(loss_grad_from_sigmoid_after_conv)
