# Copyright 2018 Hugh Salimbeni (hrs13@ic.ac.uk), Ching-An Cheng (cacheng@gatech.edu)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose

from gpflow import settings
custom_config = settings.get_settings()
custom_config.numerics.jitter_level = 0.

with settings.temp_settings(custom_config):

    from gpflow.test_util import session_tf
    from gpflow.models.svgp import SVGP
    from gpflow.likelihoods import Gaussian
    from gpflow.kernels import RBF
    from gpflow.training import NatGradOptimizer, ScipyOptimizer
    from odvgp.odvgp import ODVGP, DVGP, HDVGP
    from odvgp.gaussian_bases import DecoupledBasis



    def test_vs_svgp(session_tf):
        N, M, Dx, Dy = 5, 4, 3, 2
        np.random.seed(1)
        X = np.random.randn(N, Dx)
        Y = np.random.randn(N, Dy)
        Z = np.random.randn(M, Dx)
        kern = RBF(1)
        lik = Gaussian()
        lik.variance = 0.1

        q_mu = np.random.randn(M, Dy)
        q_sqrt = np.random.rand(Dy, M, M)

        kern.set_trainable(False)
        lik.set_trainable(False)

        model_svgp = SVGP(X, Y, kern, lik, Z=Z, whiten=False)
        model_odvgp = ODVGP(X, Y, kern, lik, np.empty((0, Dx)), Z)
        model_dvgp = DVGP(X, Y, kern, lik, Z, Z)
        model_hdvgp = HDVGP(X, Y, kern, lik, np.empty((0, Dx)), Z)

        model_svgp.q_mu = q_mu
        model_svgp.q_sqrt = q_sqrt

        model_odvgp.basis.a_beta = q_mu
        model_odvgp.basis.L = q_sqrt

        model_hdvgp.basis.a_beta = q_mu
        model_hdvgp.basis.L = q_sqrt

        L_svgp = model_svgp.compute_log_likelihood()
        L_odvgp = model_odvgp.compute_log_likelihood()
        L_hdvgp = model_hdvgp.compute_log_likelihood()


        print(model_svgp.compute_log_likelihood())
        print(model_odvgp.compute_log_likelihood())
        print(model_dvgp.compute_log_likelihood())
        print(model_hdvgp.compute_log_likelihood())


        # The hybrid, svgp and orth decoupled models should all be the same
        # assert_allclose(L_svgp, L_odvgp)
        # assert_allclose(L_svgp, L_hdvgp)

        # as there is no gamma these three all give the exact answer in a single nat grad step

        NatGradOptimizer(1.).minimize(model_svgp,
                                      var_list=[[model_svgp.q_mu, model_svgp.q_sqrt]],
                                      maxiter=1)


        NatGradOptimizer(1.).minimize(model_odvgp,
                                      var_list=[[model_odvgp.basis.a_beta, model_odvgp.basis.L]],
                                      maxiter=1)


        NatGradOptimizer(1.).minimize(model_hdvgp,
                                      var_list=[[model_hdvgp.basis.a_beta, model_hdvgp.basis.L]],
                                      maxiter=1)

        # for the decoupled model we need to use ordinary optimization
        model_dvgp.basis.alpha.set_trainable(False)
        model_dvgp.basis.beta.set_trainable(False)
        ScipyOptimizer().minimize(model_dvgp, maxiter=10000)

        model_svgp.feature.set_trainable(False)
        ScipyOptimizer().minimize(model_svgp, maxiter=10000)

        L_svgp = model_svgp.compute_log_likelihood()
        L_odvgp = model_odvgp.compute_log_likelihood()
        L_dvgp = model_dvgp.compute_log_likelihood()
        L_hdvgp = model_hdvgp.compute_log_likelihood()

        # The should be exactly the same
        # assert_allclose(L_svgp, L_odvgp)
        # assert_allclose(L_svgp, L_hdvgp)

        print(model_svgp.compute_log_likelihood())
        print(model_odvgp.compute_log_likelihood())
        print(model_dvgp.compute_log_likelihood())
        print(model_hdvgp.compute_log_likelihood())

        # print(model_svgp)
        # print(model_dvgp)

        # must be a lower bound since at optimal
        assert L_dvgp < L_svgp

        # should be vaguely close
        assert_allclose(L_dvgp, L_svgp, atol=1e-4, rtol=1e-4)

test_vs_svgp(2)

class ExactDecoupledBasis(DecoupledBasis):
    """
    For testing purposes, we can analytically compute the optimial variational
    parameters for Gaussian likelihood.
    """
    def __init__(self, num_latent, alpha, beta):
        DecoupledBasis.__init__(self, num_latent, alpha, beta)
        self._X = None
        self._Y = None
        self._likelihood_var = None
        self._kernel = None
        self.a_shape = [len(alpha), num_latent]
        self.chol_B_shape = [num_latent, len(beta), len(beta)]

    def update_data(self, X, Y, likelihood_var, kernel):
        self._X = X
        self._Y = Y
        self._likelihood_var = likelihood_var
        self._kernel = kernel

    def _build_S(self):
        K_beta = self.beta.Kuu(self._kernel, jitter=1e-6)
        K_beta_X = self.beta.Kuf(self._kernel, self._X)
        KK = tf.matmul(K_beta_X, K_beta_X, transpose_b=True)/self._likelihood_var
        KK += K_beta
        return tf.matmul(K_beta, tf.matrix_solve(KK, K_beta))

    def _build_a(self):
        K_alpha = self.alpha.Kuu(self._kernel, jitter=1e-6)
        K_alpha_X = self.alpha.Kuf(self._kernel, self._X)
        KK = tf.matmul(K_alpha_X, K_alpha_X, transpose_b=True) / self._likelihood_var
        KK += K_alpha
        return tf.matrix_solve(KK, tf.matmul(K_alpha_X, self._Y))/self._likelihood_var

    @property
    def a(self):
        return tf.reshape(self._build_a(), self.a_shape)

    @a.setter
    def a(self, val):
        pass

    @property
    def chol_B(self):
        S = self._build_S()
        K = self.beta.Kuu(self._kernel, jitter=1e-6)
        B_inv = tf.matmul(K, tf.matrix_solve(K - S, K)) - K
        B_inv = B_inv + tf.eye(tf.shape(B_inv)[0], dtype=tf.float64)*1e-6
        chol_B_inv = tf.cholesky(B_inv)
        chol_B = tf.matrix_inverse(tf.transpose(chol_B_inv))
        chol_B = tf.tile(chol_B[None, :, :], [self.chol_B_shape[0], 1, 1])
        return tf.reshape(chol_B, self.chol_B_shape)

    @chol_B.setter
    def chol_B(self, val):
        pass


# def test_vs_exact(session_tf):
#     N, M, Dx, Dy = 5, 4, 3, 2
#     np.random.seed(0)
#     X = np.random.randn(N, Dx)
#     Y = np.random.randn(N, Dy)
#     Z = np.random.randn(M, Dx)
#     kern = RBF(1)
#     lik = Gaussian()
#     lik.variance = 0.1
#
#     q_mu = np.random.randn(M, Dy)
#     q_sqrt = np.random.rand(Dy, M, M)
#
#     kern.set_trainable(False)
#     lik.set_trainable(False)
#
#     model_svgp = SVGP(X, Y, kern, lik, Z=Z, whiten=False)
#     model_odvgp = ODVGP(X, Y, kern, lik, np.empty((0, Dx)), Z)
#     model_dvgp = DVGP(X, Y, kern, lik, Z, Z)
#     model_hdvgp = HDVGP(X, Y, kern, lik, np.empty((0, Dx)), Z)
#
#
#
#
