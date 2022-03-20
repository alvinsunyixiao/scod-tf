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
        self.op = tf.Variable(tf.random.normal((N, d)), trainable=False)

    def __call__(self, M: tf.Tensor) -> tf.Tensor:
        return M @ self.op

class SRFTSketchOp(SketchOp):
    def __init__(self, d: int, N: int):
        super().__init__(d, N)
        self.D = tf.Variable(tfp.random.rademacher((N,)), trainable=False)
        self.P = tf.Variable(np.random.choice(N, d, replace=False), trainable=False)

    def __call__(self, M: tf.Tensor) -> tf.Tensor:
        # apply diag(D)
        D_M = self.D * M
        # perform inverse discrete consine transform to the first axis
        F_D_M = tf.signal.idct(D_M)
        # randomly chooses n rows
        return tf.gather(F_D_M, self.P, axis=1)


class Sketch(tf.Module):
    def __init__(self, N: int, k: int,
                 T: typing.Optional[int] = None,
                 sketch_op_class: typing.Type[SketchOp] = SRFTSketchOp):
        super().__init__(name=self.__class__.__name__)
        self.N = N
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

    def update(self, LT: tf.Tensor):
        self.Y.assign_add(tf.matmul(LT, self.Omega(LT), transpose_a=True))
        self.W.assign_add(tf.matmul(self.Psi(LT), LT, transpose_a=True))

    def batch_update(self, batch_LT: tf.Tensor):
        for LT in batch_LT:
            self.update(LT)

    def fixed_rank_eig_approx(self):
        # A ~= QX
        Q, _ = tf.linalg.qr(self.Y)
        U, T = tf.linalg.qr(tf.linalg.adjoint(self.Psi(tf.linalg.adjoint(Q))))
        X = tf.linalg.triangular_solve(T, tf.transpose(U) @ self.W)
        tf.print("A ~= QX Done")

        # A ~= U S U_T (where S is symmetric)
        U, T = tf.linalg.qr(tf.concat([Q, tf.transpose(X)], axis=1))
        T1 = T[:, :self.r]
        T2 = T[:, self.r:]
        S = (T1 @ tf.transpose(T2) + T2 @ tf.transpose(T1)) / 2.
        tf.print("A ~= U S U_T Done")

        # A ~= U D U_T (where D is diagonal)
        D, V = tf.linalg.eigh(S)
        # preserve k eigen values / basis
        D = D[-self.k:]
        V = V[:, -self.k:]
        D = tf.maximum(D, 0.0) # ensures PSD
        U = U @ V
        tf.print("A ~= U D U_T Done")

        self.eigs.assign(D)
        self.basis.assign(U)

    def cov_posterior(self, batch_jac: tf.Tensor, log_prior: tf.Tensor):
        J_JT = tf.matmul(batch_jac, batch_jac, transpose_b=True)

        J_U = batch_jac @ self.basis
        J2 = J_U * tf.sqrt(self.eigs / (self.eigs + tf.exp(-2 * log_prior)))
        J2_J2T = tf.matmul(J2, J2, transpose_b=True)

        return tf.exp(2 * log_prior) * (J_JT - J2_J2T)

    def diag_var_posterior(self, batch_jac: tf.Tensor, log_prior: tf.Tensor):
        J_JT = tf.reduce_sum(tf.square(batch_jac), axis=-1)

        J_U = batch_jac @ self.basis
        J_U_S = J_U * (self.eigs / (self.eigs + tf.exp(-2 * log_prior[..., tf.newaxis])))
        J2_J2T = tf.reduce_sum(J_U_S * J_U, axis=-1)

        return tf.exp(2 * log_prior) * (J_JT - J2_J2T)
