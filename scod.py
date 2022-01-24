import typing as T
import tensorflow as tf
import tensorflow.keras as tfk

from distribution import OutputDist
from sketching import Sketch

class SCOD(tfk.Model):
    def __init__(self,
                 model: tfk.Model,
                 output_dist: OutputDist,
                 dataset: tf.data.Dataset,
                 sketch: Sketch,
                 model_vars: T.Optional[T.List[tf.Variable]] = None,
                 num_samples: T.Optional[int] = None,
                 num_eigs: int = 100,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = model
        self.output_dist = output_dist
        self.model_vars = model_vars
        if model_vars is None:
            self.model_vars = model.trainable_variables
        self.num_samples = num_samples
        if num_samples is None:
            self.num_samples = dataset.cardinality()
        self.dataset = dataset
        self.sketch = sketch
        self._process_dataset()

    def _process_dataset(self):
        for x in dataset:
            batch_L_w = self._compute_sqrt_fisher_w(x)
            self.sketch.batch_update(batch_L_w)
        self.sketch.fixed_rank_eig_approx()

    @tf.function
    def _compute_sqrt_fisher_w(self, x):
        with tf.GradientTape() as tape:
            f = self.base_model(x)
            g = self.output_dist.apply_sqrt_fisher(f)
        batch_jacobian = tape.batch_jacobian(g, self.model_vars)

        def partial_flatten(x):
            shp = tf.shape(x)
            return tf.reshape(x, (shp[0], shp[1], -1))

        batch_jacobian = tf.nest.map_structure(partial_flatten, batch_jacobian)
        batch_jacobian = tf.concat(batch_jacobian, axis=-1)
        batch_L_w = tf.transpose(batch_jacobian, (0, 2, 1))

        return batch_L_w
