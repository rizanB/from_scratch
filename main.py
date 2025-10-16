import numpy as np
from simple_cnn import Simple1DCNN
from loss.SquaredErrorLoss import SquaredErrorLoss
from utils.plot_loss_curve import plot_loss_curve
from timeit import timeit

np.random.seed(42)
x = np.arange(1, 1000)
# x = [1,2,3,4,5]

x = np.array(x)

# normalize x
x = (x - x.min()) / (x.max() - x.min())
y = np.array([0.57])

l = SquaredErrorLoss()

simplecnn = Simple1DCNN()

# training loop
epochs = 30
losses = []

start = timeit()

for epoch in range(epochs):

    y_pred = simplecnn.forward(x, verbose=False)
    l_epoch = l.forward(y_pred=y_pred, y=y)
    losses.append(l_epoch)

    simplecnn.backward(x=x, y_pred=y_pred, y=y, lr=0.01, verbose=False)

    print(f"epoch: {epoch+1} loss: {l_epoch}")

end = timeit()

print(f"time taken to train model: {end - start}")

plot_loss_curve(losses=losses)
