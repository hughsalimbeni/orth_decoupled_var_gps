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
from gpflow import params_as_tensors
from gpflow.params import Parameterized
from gpflow.features import InducingPoints

custom_config = settings.get_settings()
custom_config.numerics.jitter_level = 0.

from odvgp.odvgp import Variational_GP
from odvgp.gaussian_bases import add_jitter

with settings.temp_settings(custom_config):

    from gpflow.test_util import session_tf
    from gpflow.models.svgp import SVGP
    from gpflow.likelihoods import Gaussian
    from gpflow.kernels import RBF
    from gpflow.training import NatGradOptimizer, ScipyOptimizer, AdamOptimizer
    from odvgp.odvgp import ODVGP, DVGP, HDVGP
    from odvgp.gaussian_bases import DecoupledBasis


    def test_vs_svgp(session_tf):
        N, M, Dx, Dy = 5, 4, 3, 2
        np.random.seed(1)
        X = np.random.randn(N, Dx)
        Y = np.random.randn(N, Dy)
        Z = np.random.randn(M, Dx)
        kern = RBF(Dx)
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


        # The hybrid, svgp and orth decoupled models should all be the same
        assert_allclose(L_svgp, L_odvgp)
        assert_allclose(L_svgp, L_hdvgp)

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

        L_svgp = model_svgp.compute_log_likelihood()
        L_odvgp = model_odvgp.compute_log_likelihood()
        L_dvgp = model_dvgp.compute_log_likelihood()
        L_hdvgp = model_hdvgp.compute_log_likelihood()

        # should be exactly the same
        assert_allclose(L_svgp, L_odvgp)
        assert_allclose(L_svgp, L_hdvgp)

        # must be a lower bound since at optimal
        assert L_dvgp < L_svgp + 1e-4

        # should be vaguely close, but might not be exact due to ill-conditioning etc
        assert_allclose(L_dvgp, L_svgp, atol=1e-3, rtol=1e-3)

    class ExactDecoupledBasis(Parameterized):
        def __init__(self, num_latent, alpha, beta):
            Parameterized.__init__(self)
            self.num_latent = num_latent
            self.M_alpha = len(alpha)
            self.M_beta = len(beta)

            self.alpha = InducingPoints(alpha)
            self.beta = InducingPoints(beta)

            self._X = None
            self._Y = None
            self._likelihood_var = None
            self._kernel = None

        def update_data(self, X, Y, likelihood_var, kernel):
            self._X = X
            self._Y = Y
            self._likelihood_var = likelihood_var
            self._kernel = kernel

        def conditional_with_KL(self, kernel, X, full_cov=False):
            K_alpha = self.alpha.Kuu(self._kernel, jitter=settings.jitter)
            K_alpha_X = self.alpha.Kuf(self._kernel, self._X)
            KK = tf.matmul(K_alpha_X, K_alpha_X, transpose_b=True) / self._likelihood_var
            KK += K_alpha
            a = tf.matmul(K_alpha, tf.matrix_solve(KK, tf.matmul(K_alpha_X, self._Y)) / self._likelihood_var)

            K_beta = self.beta.Kuu(self._kernel, jitter=settings.jitter)
            K_beta_X = self.beta.Kuf(self._kernel, self._X)
            KK = tf.matmul(K_beta_X, K_beta_X, transpose_b=True) / self._likelihood_var
            KK += K_beta
            S = tf.matmul(K_beta, tf.matrix_solve(KK, K_beta))
            L = tf.tile(tf.cholesky(add_jitter(S))[None, :, :], [self.num_latent, 1, 1])

            L_beta = tf.cholesky(K_beta)
            L_beta_tiled = tf.tile(L_beta[None, :, :], [self.num_latent, 1, 1])
            K_beta_tiled = tf.tile(K_beta[None, :, :], [self.num_latent, 1, 1])

            KL = -0.5 * tf.cast(self.M_beta * self.num_latent, dtype=tf.float64)
            KL -= tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
            KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(L_beta))) * self.num_latent
            KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(L_beta_tiled, L, lower=True)))
            KL += 0.5 * tf.reduce_sum(a * tf.matrix_solve(K_alpha, a))

            ####### mean
            K_alpha_Xs = self.alpha.Kuf(self._kernel, X)
            K_beta_Xs = self.beta.Kuf(self._kernel, X)
            K_Xs = self._kernel.K(X) if full_cov else self._kernel.Kdiag(X)

            mean = tf.matmul(K_alpha_Xs, tf.matrix_solve(K_alpha, a), transpose_a=True)

            ####### cov
            # delta_cov = K_X_beta ( K_beta^-1 S K_beta^-1 - K_beta^-1 ) K_beta_X
            # call the sqrts of the two terms C and A, so delta_cov = C^T C - A^T A
            A = tf.matrix_triangular_solve(L_beta, K_beta_Xs, lower=True)
            B = tf.matrix_triangular_solve(tf.transpose(L_beta), A, lower=False)
            # C = tf.matmul(self.L, tf.tile(B[None, :, :], [self.num_latent, 1, 1]), transpose_a=True)
            SK = tf.matmul(L, L, transpose_b=True) - K_beta_tiled
            B_tiled = tf.tile(B[None, :, :], [self.num_latent, 1, 1])
            D = tf.matmul(SK, B_tiled)  # might be more stable this way

            if full_cov:
                # (num_latent, num_X, num_X)
                delta_cov = tf.matmul(B_tiled, D, transpose_a=True)  # more stable
            else:
                # (num_latent, num_X)
                delta_cov = tf.reduce_sum(B_tiled * D, 1)

            # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
            var = tf.expand_dims(K_Xs, 0) + delta_cov
            var = tf.transpose(var)

            return mean, var, KL


    class Exact_Variational_GP(Variational_GP):
        @params_as_tensors
        def _build_likelihood(self):
            self.basis.update_data(self.X, self.Y, self.likelihood.variance, self.kern)
            return Variational_GP._build_likelihood(self)

        @params_as_tensors
        def _build_predict(self, Xnew, full_cov=False):
            self.basis.update_data(self.X, self.Y, self.likelihood.variance, self.kern)
            return Variational_GP._build_predict(self, Xnew, full_cov=full_cov)


    def test_vs_exact(session_tf):
        N, M_gamma, M_beta, Dx, Dy = 6, 5, 4, 3, 2
        np.random.seed(0)
        X = np.random.randn(N, Dx)
        Y = np.random.randn(N, Dy)
        kern = RBF(Dx)
        lik = Gaussian()
        lik.variance = 0.1

        kern.set_trainable(False)
        lik.set_trainable(False)

        gamma = np.random.randn(M_gamma, Dx)
        beta = np.random.randn(M_beta, Dx)
        alpha = np.concatenate([gamma, beta], 0)

        model_odvgp = ODVGP(X, Y, kern, lik, gamma, beta)
        model_dvgp = DVGP(X, Y, kern, lik, alpha, beta)
        model_hdvgp = HDVGP(X, Y, kern, lik, gamma, beta)

        basis = ExactDecoupledBasis(Dy, alpha, beta)
        model_exact = Exact_Variational_GP(X, Y, kern, lik, basis)

        model_odvgp.basis.gamma.set_trainable(False)
        model_odvgp.basis.beta.set_trainable(False)

        model_dvgp.basis.alpha.set_trainable(False)
        model_dvgp.basis.beta.set_trainable(False)

        model_hdvgp.basis.gamma.set_trainable(False)
        model_hdvgp.basis.beta.set_trainable(False)

        ScipyOptimizer().minimize(model_odvgp, maxiter=10000)
        ScipyOptimizer().minimize(model_dvgp, maxiter=10000)
        ScipyOptimizer().minimize(model_hdvgp, maxiter=10000)

        L_exact = model_exact.compute_log_likelihood()
        L_odvgp = model_odvgp.compute_log_likelihood()
        L_dvgp = model_dvgp.compute_log_likelihood()
        L_hdvgp = model_hdvgp.compute_log_likelihood()

        # all should be sub optimal or equal to the exact solution
        assert L_odvgp < L_exact + 1e-6
        # assert L_dvgp < L_exact + 1e-6
        assert L_hdvgp < L_exact + 1e-6

        # we expect that these are very close as the optimization is convex
        assert_allclose(L_exact, L_odvgp, atol=1e-5, rtol=1e-5)
        assert_allclose(L_exact, L_hdvgp, atol=1e-5, rtol=1e-5)

        # should be vaguely close
        assert_allclose(L_exact, L_dvgp, atol=1e-2, rtol=1e-2)

        m_exact, v_exact = model_exact.predict_f(X)

        m_odvgp, v_odvgp = model_odvgp.predict_f(X)
        m_dvgp, v_dvgp = model_dvgp.predict_f(X)
        m_hdvgp, v_hdvgp = model_hdvgp.predict_f(X)

        # should beclose
        assert_allclose(m_odvgp, m_exact, atol=1e-4, rtol=1e-4)
        assert_allclose(v_odvgp, v_exact, atol=1e-4, rtol=1e-4)
        assert_allclose(m_hdvgp, m_exact, atol=1e-4, rtol=1e-4)
        assert_allclose(v_hdvgp, v_exact, atol=1e-4, rtol=1e-4)

        # should be vaguely close
        assert_allclose(m_dvgp, m_exact, atol=1e-2, rtol=1e-2)
        assert_allclose(v_dvgp, v_exact, atol=1e-2, rtol=1e-2)
