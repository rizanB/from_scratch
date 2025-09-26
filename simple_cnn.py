import numpy as np
from layers.Conv1D import Conv1D
from activations.Sigmoid import Sigmoid
from layers.MaxPool1D import MaxPool1D
from layers.FCNN import FCNN
from layers.Flatten import Flatten
from loss.SquaredErrorLoss import SquaredErrorLoss
from utils.printv import printv


class Simple1DCNN:

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
        """

        printv(f"x is {x}", verbose)

        conv_1d_out = self.conv1d.forward(x, verbose=verbose)
        printv(f"x -> conv1d: {conv_1d_out}", verbose)

        activation_output = self.sigmoid_after_conv.forward(
            z=conv_1d_out, verbose=verbose
        )
        printv(f"sigmoid out: {activation_output}", verbose)

        pool_1d_out = self.maxpool1d.forward(activation_output)
        printv(f"maxpool: {pool_1d_out}", verbose)

        flattened_out = self.flatten.forward(pool_1d_out)
        self.flattened_out = flattened_out
        printv(f"flattened out : {flattened_out}", verbose)

        # runs only once, uses self.fc everytime, fixes bug: fc weights being random everytime
        if self.fc is None:
            self.fc = FCNN(input_size=len(flattened_out), output_size=1)

        fcnn_out = self.fc.forward(flattened_out)
        y_pred = self.sigmoid_after_fc.forward(fcnn_out, verbose)

        return y_pred

    def backward(self, x, y_pred, y, lr, verbose=False):

        loss_grad_wrt_pred = self.loss_fn.backward(y_pred=y_pred, y=y)
        printv(
            f"grad of loss wrt prediction, from loss fn is: {loss_grad_wrt_pred}",
            verbose,
        )

        loss_grad_wrt_z = self.sigmoid_after_fc.backward(loss_grad_wrt_pred)

        printv(
            f"grad_z, grad from the sigmoid layer after fc is: {loss_grad_wrt_z}",
            verbose,
        )

        loss_grad_wrt_x = self.fc.backward(grad_output=loss_grad_wrt_z, verbose=False)

        loss_grad_from_flatten = self.flatten.backward(loss_grad_wrt_x)
        printv(
            f"grad from flatten layer after reshaping is: {loss_grad_from_flatten}",
            verbose,
        )

        loss_grad_from_maxpool1d = self.maxpool1d.backward(loss_grad_from_flatten)
        printv(f"grad from maxpool layer: {loss_grad_from_maxpool1d}", verbose)

        loss_grad_from_sigmoid_after_conv = self.sigmoid_after_conv.backward(
            loss_grad_from_maxpool1d
        )
        printv(
            f"loss grad from sigmoid after conv is: {loss_grad_from_sigmoid_after_conv}",
            verbose,
        )

        loss_grad_from_conv1d = self.conv1d.backward(loss_grad_from_sigmoid_after_conv)
        printv(f"loss grad from conv1d:{loss_grad_from_conv1d}", verbose)
