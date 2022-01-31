import typing

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class SketchOp(tf.Module):
    def __init__(self, d: int, N: int):
        super().__init__(name=self.__class__.__name__)
        self.d = d
        self.N = N

    def __call__(self, M: tf.Tensor):
        raise NotImplementedError

class GaussianSketchOp(SketchOp):
    def __init__(self, d: int, N: int):
        super().__init__(d, N)
        self.op = tf.Variable(tf.random.normal((d, N)), trainable=False)

    def __call__(self, M: tf.Tensor, batched: bool = False):
        return self.op @ M

class SRFTSketchOp(SketchOp):
    def __init__(self, d: int, N: int):
        super().__init__(d, N)
        self.D = tf.Variable(tfp.random.rademacher((N,)), trainable=False)
        self.P = tf.Variable(np.random.choice(N, d, replace=False), trainable=False)

    def __call__(self, M: tf.Tensor):
        # apply diag(D)
        D_M = self.D[:, tf.newaxis] * M
        # perform inverse discrete consine transform to the first axis
        F_D_M = tf.transpose(tf.signal.idct(tf.transpose(D_M)))
        # randomly chooses n rows
        return tf.gather(F_D_M, self.P)


class Sketch(tf.Module):
    def __init__(self, N: int, M: int, k: int,
                 T: typing.Optional[int] = None,
                 sketch_op_class: typing.Type[SketchOp] = SRFTSketchOp):
        super().__init__(name=self.__class__.__name__)
        self.N = N
        self.M = M
        self.k = k
        self.T = T
        if T is None:
            self.T = 6 * k + 4

        self.r = (self.T - 1) // 3
        self.s = self.T - self.r

        self.get_sketch_op = sketch_op_class
        self.Omega = sketch_op_class(self.r, self.N)
        self.Psi = sketch_op_class(self.s, self.N)

        self.Y = tf.Variable(tf.zeros((self.N, self.r)), trainable=False)
        self.W = tf.Variable(tf.zeros((self.s, self.N)), trainable=False)

        self.eigs = tf.Variable(tf.zeros(self.k), trainable=False)
        self.basis = tf.Variable(tf.zeros((self.N, self.k)), trainable=False)

    @tf.function
    def update(self, L: tf.Tensor):
        self.Y.assign_add(tf.matmul(L, self.Omega(L), transpose_b=True))
        self.W.assign_add(tf.matmul(self.Psi(L), L, transpose_b=True))

    # TODO(alvin): vectorize this maybe?
    @tf.function
    def batch_update(self, batch_L: tf.Tensor):
        for L in batch_L:
            self.update(L)

    def fixed_rank_eig_approx(self):
        # A ~= QX
        Q, _ = tf.linalg.qr(self.Y)
        U, T = tf.linalg.qr(self.Psi(Q))
        X = tf.linalg.triangular_solve(T, tf.transpose(U) @ self.W)

        # A ~= U S U_T (where S is symmetric)
        U, T = tf.linalg.qr(tf.concat([Q, tf.transpose(X)], axis=1))
        T1 = T[:, :self.r]
        T2 = T[:, self.r:]
        S = (T1 @ tf.transpose(T2) + T2 @ tf.transpose(T1)) / 2.

        # A ~= U D U_T (where D is diagonal)
        D, V = tf.linalg.eigh(S)
        # preserve k eigen values / basis
        D = D[-self.k:]
        V = V[:, -self.k:]
        D = tf.maximum(D, 0.0) # ensures PSD
        U = U @ V

        self.eigs.assign(D)
        self.basis.assign(U)

    def cov_posterior(self, batch_jac: tf.Tensor, sqrt_prior: tf.Tensor):
        scaled_batch_jac = batch_jac * sqrt_prior[tf.newaxis, tf.newaxis]
        JT_S_J = tf.matmul(scaled_batch_jac, scaled_batch_jac, transpose_b=True)

        UT_S = tf.transpose(self.basis) * sqrt_prior[tf.newaxis]**2
        UT_S_U = UT_S @ self.basis
        A = tf.linalg.diag(1.0 / self.eigs) + UT_S_U
        L = tf.linalg.cholesky(A)
        Linv_UT_S = tf.linalg.triangular_solve(L, UT_S)
        scaled_batch_jac_2 = tf.matmul(batch_jac, Linv_UT_S[tf.newaxis], transpose_b=True)
        JT_S2_J = tf.matmul(scaled_batch_jac_2, scaled_batch_jac_2, transpose_b=True)

        return JT_S_J - JT_S2_J

