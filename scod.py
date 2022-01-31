import typing as T
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow_probability import distributions as tfd

from tqdm import tqdm, trange

from distribution import OutputDist
from sketching import Sketch, SketchOp, SRFTSketchOp
from priors import WeightPrior, PerVarPrior

class SCOD(tfk.Model):
    def __init__(self,
                 model: tfk.Model,
                 output_dist: OutputDist,
                 num_samples: int,
                 model_vars: T.Optional[T.List[tf.Variable]] = None,
                 num_eigs: int = 50,
                 sketch_op_class: T.Type[SketchOp] = SRFTSketchOp,
                 prior_class: T.Type[WeightPrior] = PerVarPrior,
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

        self.sketch = Sketch(
            N=self.num_vars,
            M=self.num_samples,
            k=num_eigs,
            sketch_op_class=sketch_op_class,
        )
        self.prior = prior_class(self.model_vars)

    def process_dataset(self, dataset: tf.data.Dataset):
        for x in tqdm(dataset):
            batch_L_w = self._compute_sqrt_fisher_w(x)
            self.sketch.batch_update(batch_L_w)
        self.sketch.fixed_rank_eig_approx()

    @tf.function
    def _compute_sqrt_fisher_w(self, x):
        with tf.GradientTape() as tape:
            f = self.base_model(x)
            g = self.output_dist.apply_sqrt_fisher(f)
        batch_jacobian_list = tape.jacobian(g, self.model_vars)
        batch_jacobian = self._batch_jacobian_concat(batch_jacobian_list)
        batch_L_w = tf.transpose(batch_jacobian, (0, 2, 1))

        return batch_L_w

    def _batch_jacobian_concat(self, batch_jac_list: T.List[tf.Variable]):
        def partial_flatten(input):
            shp = tf.shape(input)
            return tf.reshape(input, (shp[0], shp[1], -1))

        batch_jac_list = tf.nest.map_structure(partial_flatten, batch_jac_list)
        return tf.concat(batch_jac_list, axis=-1)

    @tf.function
    def _optimize_prior_once(self, data, optimizer: tfk.optimizers.Optimizer):
        inputs, labels = data
        with tf.GradientTape() as tape:
            y, S = self(inputs)
            distribution = tfd.MultivariateNormalTriL(y, tf.linalg.cholesky(S))
            if tf.rank(labels) == 1:
                labels = labels[:, tf.newaxis]
            log_ll = distribution.log_prob(labels)
            loss = tf.reduce_mean(-log_ll)

        grads = tape.gradient(loss, self.prior.log_priors)
        optimizer.apply_gradients([(grads, self.prior.log_priors)])

        return loss

    def calibrate_prior(self, dataset: tf.data.Dataset, num_epochs: int = 100):
        optimizer = tfk.optimizers.Adam()
        for epoch in range(num_epochs):
            print(f"-------- Epoch {epoch} --------")
            for data in dataset:
                loss = self._optimize_prior_once(data, optimizer)
                print(loss)

    def call(self, x):
        with tf.GradientTape() as tape:
            y = self.base_model(x)
        batch_jacobian_list = tape.jacobian(y, self.model_vars)
        batch_jacobian = self._batch_jacobian_concat(batch_jacobian_list)
        sqrt_prior = self.prior.broadcast()

        return y, self.sketch.cov_posterior(batch_jacobian, sqrt_prior)

