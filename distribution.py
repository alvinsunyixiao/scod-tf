import numpy as np
import tensorflow as tf

class OutputDist(tf.Module):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def apply_sqrt_fisher(self, output):
        raise NotImplementedError

class GaussianFixedDiagVar(OutputDist):
    def __init__(self, sigma_diag: np.ndarray = np.ones(1)):
        super().__init__()
        self.sigma_diag = tf.Variable(sigma_diag, trainable=False)

    def apply_sqrt_fisher(self, output):
        return tf.stop_gradient(output) / tf.sqrt(self.sigma_diag)
