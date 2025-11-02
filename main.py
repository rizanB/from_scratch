from timeit import timeit

import numpy as np

from loss.SquaredErrorLoss import SquaredErrorLoss
from simple_cnn import Simple1DCNN
from utils.logging_helper import setup_logger

logger = setup_logger(__name__)
np.random.seed(42)

x = np.arange(1, 1000)
x = np.array(x)

# normalize x
x = (x - x.min()) / (x.max() - x.min())
y = np.array([0.57])

logger.info(f"Input shape: {x.shape}, Target: {y.shape}")

loss_fn = SquaredErrorLoss()

simplecnn = Simple1DCNN()

epochs = 30
losses = []

start = timeit()

logger.info(f"Starting training for {epochs} epochs")
for epoch in range(epochs):

    y_pred = simplecnn.forward(x, verbose=False)
    l_epoch = loss_fn.forward(y_pred=y_pred, y=y)
    losses.append(l_epoch)

    simplecnn.backward(x=x, y_pred=y_pred, y=y, lr=0.01, verbose=False)

    logger.info(f"epoch: {epoch+1} loss: {l_epoch}")

end = timeit()

logger.info(f"Training completed in {(end - start):.3f} seconds")

# plot_loss_curve(losses=losses)
