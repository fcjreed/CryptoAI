import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_dataset_window(series, window_size, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    # dataset = dataset.shuffle(len(series))
    dataset = dataset.shuffle(shuffle_buffer)
    return dataset.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    return model.predict(dataset)


def seq2seq_window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[1:]))
    return dataset.batch(batch_size).prefetch(1)


def moving_average_forecast(series, window_size):
    maf = np.cumsum(series)
    maf[window_size:] = maf[window_size:] - maf[:-window_size]
    return maf[window_size - 1 : -1] / window_size


def split_series(series, time, split_time):
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    return time_train, x_train, time_valid, x_valid


def build_learning_rate(lr, epoch):
    return keras.callbacks.LearningRateScheduler(lambda e: lr * 10 ** (e / epoch))


def build_early_stopping(patience):
    return keras.callbacks.EarlyStopping(patience=patience)


def build_checkpoint(checkpoint_name):
    return keras.callbacks.ModelCheckpoint(checkpoint_name, save_best_only=True)


def compile_SGD(model, loss, lr, momentum):
    optimizer = keras.optimizers.SGD(lr=lr, momentum=momentum)
    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])


def compile_learning_early_SGD(
    model, epoch, loss=keras.losses.Huber(), lr=1e-6, momentum=0.9, patience=10
):
    compile_SGD(model, loss, lr, momentum)
    return [
        build_learning_rate(lr, epoch),
        build_early_stopping(patience),
    ]  # Return callbacks to use for fitting


def compile_early_checkpoint_SGD(
    model,
    checkpoint_name,
    loss=keras.losses.Huber(),
    lr=1e-6,
    momentum=0.9,
    patience=10,
):
    compile_SGD(model, loss, lr, momentum)
    return [
        build_early_stopping(patience),
        build_checkpoint(checkpoint_name),
    ]  # Return callbacks to use for fitting


def plot_learning_rate(history, xmin, xmax):
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([xmin, xmax, 0, 20])


class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


def compile_stateful_learning_SGD(
    model, epoch, loss=keras.losses.Huber(), lr=1e-6, momentum=0.9, patience=10
):
    compile_SGD(model, loss, lr, momentum)
    return [build_learning_rate(lr, epoch), ResetStatesCallback()]