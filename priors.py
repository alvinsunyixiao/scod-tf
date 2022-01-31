import math
import typing as T
import tensorflow as tf

class WeightPrior(tf.Module):
    def __init__(self, model_vars: T.List[tf.Variable] = []):
        super().__init__(name=self.__class__.__name__)
        # dummy variable for type inference purposes
        self.log_priors = tf.Variable(0.)

    def broadcast(self) -> tf.Tensor:
        """
        broadcast log priors to the shape of weights
        """
        raise NotImplementedError

class ScalarPrior(WeightPrior):
    def __init__(self, model_vars: T.List[tf.Variable]):
        super().__init__()
        self.log_priors = tf.Variable(
            0.0,
            name="log_priors",
            trainable=True,
        )
        self.num_weights = sum(map(lambda var: math.prod(var.shape)), model_vars)

    def broadcast(self) -> tf.Tensor:
        return tf.tile(tf.exp(self.log_prior[tf.newaxis]), (self.num_weights,))

class PerVarPrior(WeightPrior):
    def __init__(self, model_vars: T.List[tf.Variable]):
        super().__init__()
        self.var_sizes = list(map(lambda var: math.prod(var.shape), model_vars))
        self.log_priors = tf.Variable(
            tf.zeros(len(self.var_sizes)),
            name="log_priors",
            trainable=True,
        )

    def broadcast(self) -> tf.Tensor:
        return tf.exp(tf.repeat(self.log_priors, repeats=self.var_sizes))

class IndependentPrior(WeightPrior):
    def __init__(self, model_vars: T.List[tf.Variable]):
        num_weights = sum(map(math.prod, weight_shapes))
        self.log_priors = tf.Variable(
            tf.zeros(num_weights),
            name="log_priors",
            trainable=True,
        )

    def broadcast(self) -> tf.Tensor:
        return tf.exp(self.log_priors)
