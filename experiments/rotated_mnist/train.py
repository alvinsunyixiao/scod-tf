import math

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa
import typing as T

class RotatedMNIST:
    def __init__(self,
                 digit: int,
                 batch_size: int = 1,
                 rot_mean: float = 0.0,
                 rot_std: float = math.pi / 4):
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tfk.datasets.mnist.load_data()
        assert digit >= 0 and digit <= 10, f"{digit} is not a valid digit"
        self.digit = digit
        self.batch_size = batch_size
        self.rot_mean = rot_mean
        self.rot_std = rot_std
        self.x_train_raw = x_train_raw[y_train_raw == digit]
        self.x_test_raw = x_test_raw[y_test_raw == digit]

        self.train_ds = self.preprocess(self.x_train_raw, train=True)
        self.test_ds = self.preprocess(self.x_test_raw, train=False)

    def augment_sample(self, x, y):
        # augment image
        x = x + tf.random.normal(tf.shape(x), stddev=0.1)
        return x, y

    def preprocess(self, data: np.ndarray, train: bool = True) -> tf.data.Dataset:
        # make HWC wich C = 1
        data = data[..., np.newaxis]
        # apply random rotation
        rotation = tf.random.normal((data.shape[0],), self.rot_mean, self.rot_std)
        data_rot = tfa.image.rotate(data, rotation, interpolation="bilinear")
        # apply normalization
        data_rot = tf.cast(data_rot, tf.float32) / 127.5 - 1

        dataset = tf.data.Dataset.from_tensor_slices((data_rot, rotation))
        if train:
            dataset = dataset.shuffle(dataset.cardinality())
            dataset = dataset.map(self.augment_sample)
        dataset = dataset.batch(self.batch_size)

        return dataset

def model_gen(img_size: T.Tuple[int, int] = (28, 28)) -> tfk.Sequential:
    reg = tfk.regularizers.l2(1e-4)
    return tfk.Sequential([
        tfk.layers.Input(img_size + (1,)),
        tfk.layers.Conv2D(16, 5, 2, "same", activation="relu"),
        tfk.layers.Conv2D(32, 5, 2, "same", activation="relu"),
        tfk.layers.Conv2D(32, 5, 2, "same", activation="relu"),
        tfk.layers.Conv2D(32, 5, 2, "same", activation="relu"),
        tfk.layers.Flatten(),
        tfk.layers.Dense(10, activation="relu"),
        tfk.layers.Dense(1),
    ])

if __name__ == "__main__":
    model = model_gen()
    model.compile("adam", loss=tfk.losses.mse, metrics=tfk.metrics.mse)

    rot_mnist = RotatedMNIST(2, 512)
    model.fit(
        x=rot_mnist.train_ds,
        validation_data=rot_mnist.test_ds,
        epochs=300,
        callbacks=[],
    )
