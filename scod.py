import typing as T
import tensorflow as tf
import tensorflow.keras as tfk

from tqdm import tqdm

from distribution import OutputDist
from sketching import Sketch, SketchOp, SRFTSketchOp

class SCOD(tfk.Model):
    def __init__(self,
                 model: tfk.Model,
                 output_dist: OutputDist,
                 dataset: tf.data.Dataset,
                 num_samples: int,
                 model_vars: T.Optional[T.List[tf.Variable]] = None,
                 num_eigs: int = 100,
                 num_sketches: T.Optional[int] = None,
                 sketch_op_class: T.Type[SketchOp] = SRFTSketchOp,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = model
        self.output_dist = output_dist

        self.model_vars = model_vars
        if model_vars is None:
            self.model_vars = model.trainable_variables
        model_vars_shps = tf.nest.map_structure(tf.size, self.model_vars)
        self.num_vars = tf.reduce_sum(model_vars_shps).numpy()

        self.num_samples = num_samples
        self.dataset = dataset

        self.sketch = Sketch(
            N=self.num_samples,
            M=self.num_vars,
            k=num_eigs,
            T=num_sketches,
            sketch_op_class=sketch_op_class,
        )

        self._process_dataset()

    def _process_dataset(self):
        for x in tqdm(self.dataset):
            batch_L_w = self._compute_sqrt_fisher_w(x)
            self.sketch.batch_update(batch_L_w)
        self.sketch.fixed_rank_eig_approx()

    #@tf.function
    def _compute_sqrt_fisher_w(self, x):
        with tf.GradientTape() as tape:
            f = self.base_model(x)
            g = self.output_dist.apply_sqrt_fisher(f)
        batch_jacobian = tape.jacobian(g, self.model_vars)

        def partial_flatten(input):
            shp = tf.shape(input)
            return tf.reshape(input, (shp[0], shp[1], -1))

        batch_jacobian = tf.nest.map_structure(partial_flatten, batch_jacobian)
        batch_jacobian = tf.concat(batch_jacobian, axis=-1)
        batch_L_w = tf.transpose(batch_jacobian, (0, 2, 1))

        return batch_L_w
