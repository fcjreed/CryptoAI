import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

def plot_history_graph(history: keras.callbacks.History, plot_on: str):
    plt.plot(history.history[plot_on])
    plt.plot(history.history['val_'+plot_on])
    plt.xlabel("Epochs")
    plt.ylabel(plot_on)
    plt.legend([plot_on, 'val_'+plot_on])
    plt.show()

def plot_learning_rate(history: keras.callbacks.History, xmin: int, xmax: int):
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([xmin, xmax, 0, 20])

def bar_chart(data: np.array, width: int, xlabel: str, ylabel: str):
    num_bars, counts = np.unique(data, return_counts=True)
    ind = [i for i, _ in enumerate(num_bars)]
    plt.bar(ind, counts, width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ind, num_bars)
    plt.show()