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

from gpflow.params import Parameterized, Parameter
from gpflow.transforms import positive, LowerTriangular
from gpflow.params.dataholders import Minibatch
from gpflow import params_as_tensors
from gpflow import settings

from gpflow.features import InducingPoints


add_jitter = lambda M: M + tf.eye(tf.shape(M)[1], dtype=settings.float_type) * settings.jitter


class GaussianBasis(Parameterized):
    """
    Base class for the decoupled bases. The KL is implemented together with the conditional
    (in gpflow.models.SVGP these are separate methods) as the required matrices are the same.
    """
    def __init__(self, **kwargs):
        Parameterized.__init__(self, **kwargs)

    def conditional_with_KL(self, X, full_cov=False):
        """
        The predictive mean and variance, and KL to the prior, returned as a tuple (mean, cov, KL)
        :param X: the input locations (N, Dx)
        :param full_cov: bool, whether to compute the full covariance or just the diagonals
        :return: mean, var, KL
        """
        raise NotImplementedError


class OrthogonallyDecoupledBasis(GaussianBasis):
    """
    The basis from

    @inproceedings{salimbeni2018decoupled,
      title={Orthogonally Decoupled Variational Gaussian Processes},
      author={Salimbeni, Hugh and Cheng, Ching-An and Boots, Byron and Deisenroth, Marc},
      booktitle={Advances in Neural Information Processing Systems},
      year={2018}
    }
    """
    def __init__(self, num_latent, gamma, beta,
                 a_gamma=None, a_beta=None, L=None,
                 minibatch_size=None,
                 **kwargs):
        GaussianBasis.__init__(self, **kwargs)
        self.num_latent = num_latent

        self.M_beta = len(beta)
        self.M_gamma = len(gamma)

        self.gamma = InducingPoints(gamma)
        self.beta = InducingPoints(beta)

        if a_gamma is None:
            a_gamma = np.zeros((self.M_gamma, self.num_latent))

        if a_beta is None:
            a_beta = np.zeros((self.M_beta, self.num_latent))

        if L is None:
            L = np.tile(np.eye(self.M_beta)[None, :, :], [self.num_latent, 1, 1])

        self.a_gamma = Parameter(a_gamma)
        self.a_beta = Parameter(a_beta)
        self.L = Parameter(L, transform=LowerTriangular(self.M_beta, self.num_latent))

        if minibatch_size:
            # This so we can take minibatches for the KL term, otherwise we have to compute a (M_gamma, M_gamma) matrix
            # NB we assume the data has seed 0, so the two minibatching objects are independent
            self.gamma_indices = Minibatch(np.arange(len(gamma)), batch_size=minibatch_size, seed=1)
        else:
            self.gamma_indices = None


    @params_as_tensors
    def _build_kernels_matrices(self, kernel, X, full_cov):
        # either (num_X, ) or (num_X, num_X)
        self.K_X = kernel.K(X) if full_cov else kernel.Kdiag(X)

        self.K_beta              = self.beta.Kuu(kernel, jitter=settings.jitter)
        self.L_beta              = tf.cholesky(self.K_beta)
        self.K_beta_X            = self.beta.Kuf(kernel, X)
        self.K_gamma_beta        = self.gamma.Kuf(kernel, self.beta.Z)
        self.K_gamma             = self.gamma.Kuu(kernel, jitter=settings.jitter)
        self.K_gamma_X           = self.gamma.Kuf(kernel, X)

        # (M_beta, M_gamma)
        self.inv_K_beta_K_beta_gamma = tf.cholesky_solve(self.L_beta, tf.transpose(self.K_gamma_beta))

        K_gb_K_b_inv_K_bg_diag = tf.reduce_sum(self.K_gamma_beta * tf.transpose(self.inv_K_beta_K_beta_gamma), 1)
        D = kernel.Kdiag(self.gamma.Z) - K_gb_K_b_inv_K_bg_diag
        # (M_gamma, 1)
        self.D = D[:, None]

        if self.gamma_indices is None:
            # (M_gamma, M_gamma)
            K_gamma_minibatch = self.K_gamma

            # (M_gamma, M_beta)
            K_gamma_beta_minibatch = self.K_gamma_beta

            # (M_gamma, Dy)
            self.a_gamma2 = self.a_gamma

        else:
            # (minibatch_gamma, Dx)
            gamma2 = tf.gather(self.gamma.Z, self.gamma_indices)

            # (minibatch_gamma, M_gamma)
            K_gamma_minibatch = kernel.K(gamma2, self.gamma.Z)

            # (minibatch_gamma, M_beta)
            K_gamma_beta_minibatch = kernel.K(gamma2, self.beta.Z)


        self.K_alpha = tf.concat([tf.concat([K_gamma_minibatch, K_gamma_beta_minibatch], 1),
                                  tf.concat([tf.transpose(self.K_gamma_beta), self.K_beta], 1)],
                                  0)

        self.K_alpha_X = tf.concat([self.K_gamma_X, self.K_beta_X], 0)

        # this is the full K_gamma - K_gamma_beta K_beta_inv K_beta_gamma. This is expensive and potentially unstable
        chol_A = tf.matrix_triangular_solve(self.L_beta, tf.transpose(self.K_gamma_beta))
        A = tf.matmul(chol_A, chol_A, transpose_a=True)
        self.pred_beta_to_gamma = add_jitter(self.K_gamma - A)

        self.K_beta_tiled         = tf.tile(self.K_beta[None, :, :], [self.num_latent, 1, 1])
        self.L_beta_tiled         = tf.tile(self.L_beta[None, :, :], [self.num_latent, 1, 1])
        self.K_beta_X_tiled       = tf.tile(self.K_beta_X[None, :, :], [self.num_latent, 1, 1])
        self.K_gamma_beta_tiled   = tf.tile(self.K_gamma_beta[None, :, :], [self.num_latent, 1, 1])
        self.K_gamma_tiled        = tf.tile(self.K_gamma[None, :, :], [self.num_latent, 1, 1])
        self.K_alpha_tiled        = tf.tile(self.K_alpha[None, :, :], [self.num_latent, 1, 1])

        self.I = tf.tile(tf.eye(self.M_beta, dtype=tf.float64)[None, :, :], [self.num_latent, 1, 1])

    def _build_a(self):
        # this is separated out to distinguish TrulyDecoupled from Decoupled, and for preconditioning gamma
        a_gamma = self.a_gamma / self.D  # preconditioning
        a_beta = tf.cholesky_solve(self.L_beta, self.a_beta) - tf.matmul(self.inv_K_beta_K_beta_gamma, a_gamma)
        # a_beta = self.a_beta - tf.matmul(self.inv_K_beta_K_beta_gamma, a_gamma)
        return a_gamma, a_beta

    @params_as_tensors
    def conditional_with_KL(self, kernel, X, full_cov=False):
        self._build_kernels_matrices(kernel, X, full_cov)

        a_gamma, a_beta = self._build_a()
        a = tf.concat([a_gamma, a_beta], 0)
        if self.gamma_indices is None:
            a2 = a
        else:
            N_batch = tf.shape(self.gamma_indices)[0]
            scale = self.M_gamma / tf.cast(N_batch, dtype=tf.float64)
            a2 = tf.concat([scale * tf.gather(a_gamma, self.gamma_indices), a_beta], 0)


        KL = -0.5 * tf.cast(self.M_beta * self.num_latent, dtype=tf.float64)
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)**2))
        KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L_beta))) * self.num_latent
        KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.L_beta_tiled, self.L, lower=True)))
        KL += 0.5 * tf.reduce_sum(a2 * tf.matmul(self.K_alpha, a))

        ####### mean
        # (M_alpha, num_X)^T * (M_alpha, num_latent) -> (num_X, num_latent)
        mean = tf.matmul(self.K_alpha_X, a, transpose_a=True)

        ####### cov
        # delta_cov = K_X_beta ( K_beta^-1 S K_beta^-1 - K_beta^-1 ) K_beta_X
        # call the sqrts of the two terms C and A, so delta_cov = C^T C - A^T A
        A = tf.matrix_triangular_solve(self.L_beta, self.K_beta_X, lower=True)
        B = tf.matrix_triangular_solve(tf.transpose(self.L_beta), A, lower=False)
        # C = tf.matmul(self.L, tf.tile(B[None, :, :], [self.num_latent, 1, 1]), transpose_a=True)
        SK = tf.matmul(self.L, self.L, transpose_b=True) - self.K_beta_tiled
        B_tiled = tf.tile(B[None, :, :], [self.num_latent, 1, 1])
        D = tf.matmul(SK, B_tiled)  # might be more stable this way

        if full_cov:
            # (num_latent, num_X, num_X)
            # delta_cov = tf.matmul(C, C, transpose_a=True) - tf.matmul(A, A, transpose_a=True)[None, :, :]
            delta_cov = tf.matmul(B_tiled, D, transpose_a=True) # more stable
        else:
            # (num_latent, num_X)
            # delta_cov = tf.reduce_sum(tf.square(C), 1) - tf.reduce_sum(tf.square(A), 0)[None, :]
            delta_cov = tf.reduce_sum(B_tiled * D, 1)

        # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
        var = tf.expand_dims(self.K_X, 0) + delta_cov
        var = tf.transpose(var)

        return mean, var, KL


class DecoupledBasis(GaussianBasis):
    """
    The basis from

    @inproceedings{cheng2017variational,
      title={Variational Inference for Gaussian Process Models with Linear Complexity},
      author={Cheng, Ching-An and Boots, Byron},
      booktitle={Advances in Neural Information Processing Systems},
      year={2017}
    }

    """
    def __init__(self, D, alpha, beta,
                 a=None, chol_B=None,
                 minibatch_size=None,
                 **kwargs):
        GaussianBasis.__init__(self, **kwargs)
        self.num_latent = D

        self.alpha = InducingPoints(alpha)
        self.beta = InducingPoints(beta)

        if minibatch_size:
            self.alpha_indices = Minibatch(np.arange(len(alpha)), batch_size=minibatch_size)
        else:
            self.alpha_indices = None

        M_alpha = len(alpha)
        M_beta = len(beta)

        if a is None:
            a = np.random.normal(size=(M_alpha, D), scale=1e-6/np.sqrt(M_alpha))

        if chol_B is None:
            chol_B = np.random.normal(size=(M_beta, M_beta), scale=1e-6/M_beta)
            chol_B = np.tile(chol_B[None, :, :], [D, 1, 1])

        self.a = Parameter(a)
        self.chol_B = Parameter(chol_B, transform=LowerTriangular(M_beta, D))

    @params_as_tensors
    def conditional_with_KL(self, kernel, X, full_cov=False):
        K_beta = self.beta.Kuu(kernel, jitter=settings.jitter)
        K_X_beta = tf.transpose(self.beta.Kuf(kernel, X))

        a1 = self.a

        if self.alpha_indices is not None:
            a2 = tf.gather(self.a, self.alpha_indices)
            alpha2 = tf.gather(self.alpha.Z, self.alpha_indices)
            scale_a2 = tf.cast(tf.shape(self.a)[0], dtype=settings.float_type)
            scale_a2 /= tf.cast(tf.shape(self.alpha_indices)[0], dtype=settings.float_type)
            K_alpha = tf.transpose(self.alpha.Kuf(kernel, alpha2))
        else:
            a2 = self.a
            scale_a2 = 1.
            K_alpha = self.alpha.Kuu(kernel)

        Dy, M_beta = tf.shape(self.a)[1], tf.shape(K_beta)[0]
        K_beta_tiled = tf.tile(K_beta[None, :, :], [Dy, 1, 1])
        K_X_beta_tiled = tf.tile(K_X_beta[None, :, :], [Dy, 1, 1])

        H = tf.tile(tf.eye(M_beta, dtype=settings.float_type)[None, :, :], [Dy, 1, 1])
        H += tf.matmul(self.chol_B, tf.matmul(K_beta_tiled, self.chol_B), transpose_a=True)

        chol_H = tf.cholesky(H)
        chol_H_inv_LT = tf.matrix_triangular_solve(chol_H, tf.transpose(self.chol_B, [0, 2, 1]))
        # LHinvLT = tf.matmul(chol_H_inv_LT, chol_H_inv_LT, transpose_a=True)

        LHinvLT = tf.matmul(self.chol_B, tf.matrix_solve(H, tf.transpose(self.chol_B, [0, 2, 1])))  # Dy,M,M

        if full_cov:
            KBK = tf.matmul(K_X_beta_tiled, tf.matmul(LHinvLT, K_X_beta_tiled, transpose_b=True))
            Knn = kernel.K(X)[None, :, :]

        else:
            Knn = kernel.Kdiag(X)[None, :]  # 1N
            A = tf.matmul(LHinvLT, K_X_beta_tiled, transpose_b=True)  # Dy,M,N
            KBK = tf.transpose(K_X_beta_tiled, [0, 2, 1]) * A  # Dy,M,N
            KBK = tf.reduce_sum(KBK, 1)  # Dy,N

        var = Knn - KBK  # either Dy,N,N or Dy,N

        var = tf.transpose(var)

        K_alpha_X = self.alpha.Kuf(kernel, X)
        mean = tf.matmul(K_alpha_X, a1, transpose_a=True)

        KL = 0.5 * tf.reduce_sum(a2 * tf.matmul(K_alpha, a1)) * scale_a2
        KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_H)))
        KL -= 0.5 * tf.reduce_sum(K_beta_tiled * LHinvLT)

        return mean, var, KL


class HybridDecoupledBasis(OrthogonallyDecoupledBasis):
    """
    The basis from the appendix of

    @inproceedings{cheng2017variational,
      title={Variational Inference for Gaussian Process Models with Linear Complexity},
      author={Cheng, Ching-An and Boots, Byron},
      booktitle={Advances in Neural Information Processing Systems},
      year={2017}
    }

    """
    def _build_a(self):
        return self.a_gamma, tf.cholesky_solve(self.L_beta, self.a_beta)