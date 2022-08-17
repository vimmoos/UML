from keras_uncertainty.losses import (
    regression_gaussian_nll_loss,
)
from keras_uncertainty.models import DeepEnsembleRegressor, TwoHeadStochasticRegressor
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def create_data(num, domain=1):
    inp = np.linspace(-domain, domain, num=num)
    y = np.tanh(inp / 2)
    stds = (np.tanh(inp / 2) ** 2) / 2  # max .5 min 0
    print(max(stds), min(stds))
    x = y + np.random.normal(0, stds, size=num)
    return x, y


def make_model():
    inl = Input((1,))
    X = Dense(64, activation="relu")(inl)
    X = Dense(64, activation="relu")(X)
    mean_head = Dense(1, activation="linear")(X)
    var_head = Dense(1, activation="softplus")(X)

    model = Model(inl, mean_head)
    predictor = Model(inl, [mean_head, var_head])

    model.compile(Adam(learning_rate=0.00005), regression_gaussian_nll_loss(var_head))
    return model, predictor


def train(domain=1, epochs=50, num_models=5, length_data=500):
    x, y = create_data(length_data, domain)
    model = DeepEnsembleRegressor(make_model, num_models)
    model.fit(x, y, epochs=epochs)
    return model, x, y


def predict(model, domain=2, length_data=500):
    x, y = create_data(length_data, domain)
    return (
        [x.reshape((-1,)) for x in model.predict(x, disentangle_uncertainty=True)],
        x,
        y,
    )


def plot(mean, ale_std, epi_std, x, y):
    plt.plot(y, label="Tanh")
    plt.plot(mean, label="Predicted")
    plt.fill_between(
        range(len(x)),
        mean + ale_std,
        mean - ale_std,
        alpha=0.3,
        color="blue",
    )
    plt.fill_between(
        range(len(x)),
        mean + (ale_std + epi_std),
        mean - (ale_std + epi_std),
        alpha=0.3,
        color="green",
    )
    plt.scatter(range(len(x)), x, label="Raw data", color="grey")
    plt.legend()
    plt.show()
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    model, x, y = train()
    (y_mean, ale_std, epi_std), x_test, y_test = predict(model)
    plt.plot(y, color="red", label="Train target")
    plt.scatter(range(len(x)), x, label="Train input")
    plt.legend()
    plt.show()

    plt.plot(y_test, color="red", label="Test target")
    plt.scatter(range(len(x_test)), x_test, label="Test input")
    plt.legend()
    plt.show()

    plt.plot(y_test, color="red", label="Test target")
    plt.scatter(range(len(x_test)), y_mean, label="Predicted")
    plt.legend()
    plt.show()

    plt.plot(y_test, color="red", label="Test target")
    plt.fill_between(
        range(len(x_test)),
        y_test + ale_std,
        y_test - ale_std,
        alpha=0.3,
        color="blue",
        label="aleatoric",
    )
    plt.legend()
    plt.show()

    plt.plot(y_test, color="red", label="Test target")
    plt.fill_between(
        range(len(x_test)),
        y_test + epi_std,
        y_test - epi_std,
        alpha=0.3,
        color="blue",
        label="epistemic",
    )
    plt.legend()
    plt.show()
