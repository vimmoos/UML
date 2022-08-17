from tensorflow.keras.datasets import mnist, fashion_mnist
import seaborn
from keras.utils import to_categorical
from dataclasses import dataclass
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import keras as k
import numpy as np
from keras_uncertainty.models import DeepEnsembleClassifier
from keras_uncertainty.utils import (
    numpy_entropy,
    classifier_calibration_curve,
    classifier_calibration_error,
)
from keras_uncertainty.utils.numpy_metrics import accuracy
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


@dataclass
class Data_points:
    x: np.ndarray
    y: np.ndarray
    n_classes: int = 10

    def scale(self):
        self.x = np.expand_dims((self.x.astype("float32") / 255), -1)
        self.y = to_categorical(self.y, self.n_classes)
        return self

    def __len__(self):
        return self.x.shape[0]

    @property
    def shape(self):
        return self.x.shape[1:]


@dataclass
class Data:
    train: Data_points
    test: Data_points


def load_data(flag: bool = True):
    train, test = (mnist if flag else fashion_mnist).load_data()

    return Data(Data_points(*train).scale(), Data_points(*test).scale())


def create_model():
    model = k.models.Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=k.optimizers.Adadelta(),
        metrics=["accuracy"],
    )
    return model


def train_base(data, batch_size=128, epoch=30):
    model = create_model()
    model.fit(
        data.train.x,
        data.train.y,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        validation_data=(data.test.x, data.test.y),
    )
    return model


def train_ensemble(data, n_models=10, batch_size=128, epoch=30):
    model = DeepEnsembleClassifier(create_model, num_estimators=n_models)
    model.fit(
        data.train.x,
        data.train.y,
        batch_size=batch_size,
        epochs=epoch,
        verbose=1,
        validation_data=(data.test.x, data.test.y),
    )
    return model


def cal_acc(pred, target):
    return accuracy(np.argmax(target, 1), np.argmax(pred, 1))


@dataclass
class Stats:
    preds: Any
    accuracy: Any
    entropy: Any
    max_prob: Any


def predict(model, data):
    preds = model.predict(data.test.x)
    acc = cal_acc(preds, data.test.y)
    entropy = numpy_entropy(preds)
    max_prob = np.max(preds, axis=1)
    return Stats(preds, acc, entropy, max_prob)


def plot(id_stats, od_stats, attr="entropy", title=None):
    seaborn.kdeplot(
        data=pd.DataFrame(
            {
                "in distribution": getattr(id_stats, attr),
                "out distribution": getattr(od_stats, attr),
            }
        ),
        fill=True,
    )
    title = title if title else f"{attr} density"
    plt.xlabel(attr)
    plt.title(title)
    plt.show()
    plt.clf()
    plt.cla()


def roc_plot(id_stats, od_stats):
    y_target = np.concatenate((np.ones(10000), np.zeros(10000)))
    y_pred = np.concatenate((id_stats.max_prob, od_stats.max_prob))
    plt_range = [0, 1]
    fp, tp, th = roc_curve(y_target, y_pred)
    auc = roc_auc_score(y_target, y_pred)
    plt.title("ROC")
    plt.plot(fp, tp, "b")
    plt.legend()
    plt.plot(plt_range, plt_range, "r--")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
    return auc


def calibration_curve_per_class(preds, y_classes, name="test"):
    for cl in range(10):
        x, y = classifier_calibration_curve(
            np.full((10000,), cl),
            y_classes,
            preds[:, cl],
            num_bins=20,
        )
        plt.plot(x, y, color="red")
        plt.plot([0, 1], [0, 1], "r--", color="black")
        plt.ylabel("Accuracy")
        plt.xlabel("Confidence")
        plt.savefig(f"{name}_rel_{cl}")
        plt.cla()
        plt.clf()


def train_models():
    id = load_data(1)  # mnist
    od = load_data(0)  # fashion_mnist

    model_mnist = train_ensemble(id)
    base_model_mnist = train_base(id)
    model_fash = train_ensemble(od)
    base_model_fash = train_base(od)
    return model_mnist, base_model_mnist, model_fash, base_model_fash


def report(model, id, od):
    id_stats = predict(model, id)
    od_stats = predict(model, od)
    plot(id_stats, od_stats)
    plot(id_stats, od_stats, "max_prob")
    auc = roc_plot(id_stats, od_stats)
    print(
        f"""ID accuracy = {id_stats.accuracy}
OD accuracy = {od_stats.accuracy}
AUC  = {auc}"""
    )


def calibration_report(model, data, name="test"):
    stats = predict(model, data)
    y_classes = np.argmax(data.test.y, axis=1)
    y_preds = np.argmax(stats.preds, axis=1)
    y_confs = np.max(stats.preds, axis=1)
    x, y = classifier_calibration_curve(
        y_preds,
        y_classes,
        y_confs,
        num_bins=20,
    )
    plt.plot(x, y, color="red")
    plt.plot([0, 1], [0, 1], "r--", color="black")
    plt.ylabel("Accuracy")
    plt.xlabel("Confidence")
    plt.savefig(f"{name}_total_rel")
    plt.cla()
    plt.clf()

    print(classifier_calibration_error(y_preds, y_classes, y_confs, num_bins=20))

    calibration_curve_per_class(stats.preds, y_classes, name)


if __name__ == "__main__":
    models = train_models()

    model_mnist, bmodel_mnist, model_fash, bmodel_fash = models
    id = load_data(1)  # mnist
    od = load_data(0)  # fashion_mnist

    report(model_mnist, id, od)
    report(bmodel_mnist, id, od)
    report(model_fash, od, id)
    report(bmodel_fash, od, id)

    calibration_report(model_mnist, id, "ens_mnist")
    calibration_report(bmodel_mnist, id, "base_mnist")
    calibration_report(model_fash, od, "ens_fash")
    calibration_report(bmodel_fash, od, "base_fash")
