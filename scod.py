import numpy as np
import typing as T
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow_probability import distributions as tfd

from tqdm import tqdm, trange

from distribution import OutputDist
from sketching import Sketch, SketchOp, SRFTSketchOp

class SCOD(tfk.Model):
    def __init__(self,
                 model: tfk.Model,
                 output_dist: OutputDist,
                 model_vars: T.Optional[T.List[tf.Variable]] = None,
                 num_eigs: int = 50,
                 sketch_op_class: T.Type[SketchOp] = SRFTSketchOp,
                 output_func: T.Optional[T.Callable[..., tf.Tensor]] = None,
                 diag: bool = True,
                 log_prior_init: np.ndarray = np.ones(1),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = model
        self.output_dist = output_dist

        self.output_func = output_func
        if output_func is None:
            self.output_func = lambda x: x

        self.model_vars = model_vars
        if model_vars is None:
            self.model_vars = model.trainable_variables
        model_vars_shps = tf.nest.map_structure(tf.size, self.model_vars)
        self.num_vars = tf.reduce_sum(model_vars_shps).numpy()
        print(f"Calibrating for {self.num_vars} variables:")
        print("\n".join([var.name for var in self.model_vars]))

        self.sketch = Sketch(
            N=self.num_vars,
            k=num_eigs,
            sketch_op_class=sketch_op_class,
        )
        self.diag = diag
        self.log_prior = self.add_weight(name="log_prior", shape=log_prior_init.shape,
                                         initializer=tfk.initializers.constant(log_prior_init))

    def process_dataset(self, dataset: tf.data.Dataset, sketch_dir: str = "/tmp/sketch"):
        for x in tqdm(dataset):
            self._update_sqrt_fisher_w(x)

        self.sketch.fixed_rank_eig_approx()

    @tf.function
    def _update_sqrt_fisher_w(self, x):
        with tf.GradientTape() as tape:
            f = self.base_model(x)
            g = self.output_dist.apply_sqrt_fisher(f)

        batch_jacobian = tape.jacobian(g, self.model_vars)

        def partial_flatten(input):
            shp = tf.shape(input)
            return tf.reshape(input, (shp[0], -1))

        for i in tf.range(tf.shape(g)[0]):
            jacobians = tf.nest.map_structure(lambda x: partial_flatten(x[i]), batch_jacobian)
            LT = tf.concat(jacobians, axis=-1)
            self.sketch.update(LT)

    def _batch_jacobian_concat(self, batch_jac_list: T.List[tf.Variable]):
        def partial_flatten(input):
            shp = tf.shape(input)
            return tf.reshape(input, (shp[0], shp[1], -1))

        batch_jac_list = tf.nest.map_structure(partial_flatten, batch_jac_list)
        return tf.concat(batch_jac_list, axis=-1)

    @tf.function
    def _optimize_prior_once(self, data, sigma_scale):
        inputs, labels = data
        with tf.GradientTape() as tape:
            output, S = self(inputs)
            y = self.output_func(output)
            distribution = tfd.MultivariateNormalTriL(y, sigma_scale * tf.linalg.cholesky(S))
            if tf.rank(labels) == 1:
                labels = labels[:, tf.newaxis]
            log_ll = distribution.log_prob(labels)
            loss = tf.reduce_mean(-log_ll)

        grads = tape.gradient(loss, self.log_prior)

        return loss, grads

    def load_from_saved_model(self, path):
        saved_model = tfk.models.load_model(path)

    def calibrate_prior(self,
        dataset: tf.data.Dataset,
        sigma_scale: float = 1.0,
        num_epochs: int = 100,
        learning_rate: T.Union[float, tfk.optimizers.schedules.LearningRateSchedule] = 1e-3,
    ):
        optimizer = tfk.optimizers.Adam(learning_rate)
        sigma_scale = tf.constant(sigma_scale)
        losses = []
        pbar = trange(num_epochs)
        for epoch in pbar:
            sum_loss = 0.0
            cnt = 0
            for data in dataset:
                loss, grads = self._optimize_prior_once(data, sigma_scale)
                optimizer.apply_gradients([(grads, self.log_prior)])
                sum_loss += loss
                cnt += 1
            losses.append((sum_loss / cnt).numpy())
            pbar.set_postfix({"Loss": losses[-1], "lr": optimizer._decayed_lr(tf.float32).numpy()})

        return losses

    def call(self, x):
        with tf.GradientTape() as tape:
            output = self.base_model(x)
            g = self.output_dist.apply_sqrt_fisher(output)
        batch_jacobian_list = tape.jacobian(g, self.model_vars)
        batch_jacobian = self._batch_jacobian_concat(batch_jacobian_list)

        if self.diag:
            S = self.sketch.diag_var_posterior(batch_jacobian, self.log_prior)
        else:
            S = self.sketch.cov_posterior(batch_jacobian, self.log_prior)

        return output, S

