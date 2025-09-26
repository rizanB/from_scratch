import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curve(losses: list[int]):
    plt.plot(losses)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.title("loss curve")
    plt.show()