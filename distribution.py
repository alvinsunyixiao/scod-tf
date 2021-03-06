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
        self.sigma_diag = tf.Variable(sigma_diag, trainable=False, dtype=tf.float32)

    def apply_sqrt_fisher(self, output):
        mu = output
        if isinstance(output, dict):
            mu = output["mu"]
        return mu / self.sigma_diag

class GaussianDynamicDiagVar(OutputDist):
    def apply_sqrt_fisher(self, output):
        mu = output["mu"]
        sigma = tf.stop_gradient(output["sigma"])

        return mu / sigma
