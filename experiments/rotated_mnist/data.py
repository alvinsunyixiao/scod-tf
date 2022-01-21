import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa

class RotatedMNIST:
    def __init__(self, max_abs_rotation: float = math.pi / 4):
        self.max_abs_rotation = max_abs_rotation
        (x_train_raw, _), (x_test_raw, _) = tfk.datasets.mnist.load_data()
        self.x_train_raw = x_train_raw
        self.x_test_raw = x_test_raw

        self.train_ds = self.preprocess(x_train_raw, max_abs_rotation)
        self.test_ds = self.preprocess(x_test_raw, max_abs_rotation)

    def preprocess(self, data: np.ndarray, max_abs_rotation: float) -> tf.data.Dataset:
        rotation = tf.random.uniform((data.shape[0],), -max_abs_rotation, max_abs_rotation)
        data_rot = tfa.image.rotate(data, rotation, interpolation="bilinear")

        return tf.data.Dataset.from_tensor_slices((data_rot, rotation))


