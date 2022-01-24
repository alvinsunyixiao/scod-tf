import math
import functools

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

        self.train_ds = self.preprocess_train(self.x_train_raw)
        self.test_ds = self.preprocess_test(self.x_test_raw)

    def _apply_random_rotation(self, imgs: tf.Tensor) -> T.Tuple[tf.Tensor, tf.Tensor]:
        rotation = tf.random.normal((tf.shape(imgs)[0],), self.rot_mean, self.rot_std)
        imgs_rot = tfa.image.rotate(imgs, rotation, interpolation="bilinear")

        return imgs_rot, rotation

    def _normalize_images(self, imgs: tf.Tensor) -> tf.Tensor:
        return tf.cast(imgs, tf.float32) / 127.5 - 1

    def _map_func(self, imgs: tf.Tensor) -> T.Tuple[tf.Tensor, tf.Tensor]:
        imgs = self._normalize_images(imgs)
        return self._apply_random_rotation(imgs)

    def preprocess_train(self, data: np.ndarray) -> tf.data.Dataset:
        # make HWC wich C = 1
        data = data[..., np.newaxis]
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(dataset.cardinality())
        dataset = dataset.batch(self.batch_size)

        # apply random rotation
        dataset = dataset.map(self._map_func, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    def preprocess_test(self, data: np.ndarray) -> tf.data.Dataset:
        data = data[..., np.newaxis]
        images, rotation = self._map_func(data)
        dataset = tf.data.Dataset.from_tensor_slices((images, rotation))
        dataset = dataset.batch(self.batch_size)

        return dataset

def model_gen(img_size: T.Tuple[int, int] = (28, 28)) -> tfk.Sequential:
    return tfk.Sequential([
        tfk.layers.Input(img_size + (1,)),
        tfk.layers.Conv2D(16, 5, 2, "same", activation="relu"),
        tfk.layers.Conv2D(32, 5, 2, "same", activation="relu"),
        tfk.layers.Conv2D(64, 5, 2, "same", activation="relu"),
        tfk.layers.Conv2D(128, 5, 2, "same", activation="relu"),
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
        callbacks=[
            tfk.callbacks.TensorBoard("./logs"),
            tfk.callbacks.ReduceLROnPlateau(patience=20),
            tfk.callbacks.ModelCheckpoint("./logs/ckpts/{epoch:03d}-{val_loss:.3f}"),
        ],
    )
