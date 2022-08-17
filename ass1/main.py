from keras_uncertainty.losses import (
    regression_gaussian_nll_loss,
)
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def create_data(num):
    y = np.tanh(np.linspace(-10, 10, num=num))
    x = y + np.random.normal(0, 0.5, size=num)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.3)
    return x_train, y_train, y, x


def train_n_predict(x, y, pred):
    inl = Input((1,))
    X = Dense(64, activation="relu")(inl)
    X = Dense(64, activation="relu")(X)
    mean_head = Dense(1, activation="linear")(X)
    var_head = Dense(1, activation="softplus")(X)

    model = Model(inl, mean_head)
    predictor = Model(inl, [mean_head, var_head])

    model.compile(Adam(learning_rate=0.00001), regression_gaussian_nll_loss(var_head))
    model.fit(x, y, epochs=500)

    return predictor.predict(pred)


def run_exp():
    x_train, y_train, y, x = create_data(500)
    mean, std = train_n_predict(x_train, y_train, y)
    std = np.sqrt(std.reshape((-1,)))
    return mean.reshape((-1,)), std, x, y


def plot(mean, std, x, y):
    plt.plot(y, label="Tanh")
    plt.plot(mean, label="Predicted")
    plt.fill_between(
        range(500),
        mean + std,
        mean - std,
        alpha=0.3,
        color="yellow",
    )
    plt.scatter(range(len(x)), x, label="Raw data", color="grey")
    plt.legend()
    plt.show()
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    mean, std, x, y = run_exp()
    plot(mean, std, x, y)
