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

import pytest

import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose

from gpflow.kernels import Matern52
from gpflow.likelihoods import Gaussian
from gpflow.test_util import session_tf

from odvgp.gaussian_bases import DecoupledBasis, OthogonallyDecoupledBasis


def ref_decoupled_KL(a, B, K_alpha, K_beta):
    """
    eq 9 of Cheng and Boots 2017
    """
    B_inv = np.linalg.inv(B)
    I = np.eye(K_beta.shape[0])

    KL = 0.5 * np.sum(a * (K_alpha @ a))
    KL += 0.5 * np.linalg.slogdet(I + K_beta @ B)[1]
    KL -= 0.5 * np.sum(K_beta * np.linalg.inv(B_inv + K_beta))
    return KL

def ref_hybrid_KL(m, S, a, K_alpha, K_tilde_alpha, K_tilde):
    """
    eq 14 of Cheng and Boots 2017
    """
    K_inv = np.linalg.inv(K_tilde)
    KL = 0.5 * np.sum(m * (K_inv @ m))
    KL += np.sum(m * (K_inv @ K_tilde_alpha @ a))
    KL += 0.5 * np.sum(a * (K_alpha @ a))

    KL -= 0.5 * np.linalg.slogdet(K_inv @ S)[1]  # in eq 14 the factor of 0.5 is missing

    KL += 0.5 * np.sum(S * K_inv)

    KL -= 0.5 * K_tilde.shape[0]
    return KL


def test_decoupled_kl(session_tf):
    Dx = 2
    Dy = 3
    M_alpha = 4
    M_beta = 5

    alpha = np.random.randn(M_alpha, Dx)
    beta = np.random.randn(M_beta, Dx)

    kernel = Matern52(Dx)
    K_alpha = kernel.compute_K_symm(alpha)
    K_beta = kernel.compute_K_symm(beta)

    a = np.random.randn(M_alpha, Dy)
    U = np.random.randn(Dy, M_beta, M_beta)
    B = np.einsum('dnN,dmN->dnm', U, U) + 1e-6 * np.eye(M_beta)[None, :, :]
    chol_B = np.linalg.cholesky(B)

    KL_ref = sum([ref_decoupled_KL(_a, _B, K_alpha, K_beta) for _a, _B in zip(a.T, B)])

    basis = DecoupledBasis(Dy, alpha, beta, a=a, chol_B=chol_B)
    _, _, kl = basis.conditional_with_KL(kernel, np.empty((1, Dx)))
    KL = session_tf.run(kl)

    assert_allclose(KL, KL_ref, rtol=1e-6, atol=1e-6)

def test_orthogonally_decoupled_kl(session_tf):
    Dx = 2
    Dy = 3
    M_alpha = 4
    M_beta = 5

    alpha = np.random.randn(M_alpha, Dx)
    beta = np.random.randn(M_beta, Dx)

    kernel = Matern52(Dx)
    K_alpha = kernel.compute_K_symm(alpha)
    K_beta = kernel.compute_K_symm(beta)
    K_beta_alpha = kernel.K(beta, alpha)

    m = np.random.randn(M_alpha, Dy)
    a = np.random.randn(M_alpha, Dy)
    U = np.random.randn(Dy, M_beta, M_beta)
    S = np.einsum('dnN,dmN->dnm', U, U) + 1e-6 * np.eye(M_beta)[None, :, :]
    chol_S = np.linalg.cholesky(S)

    # def ref_hybrid_KL(m, S, a, K_alpha, K_tilde_alpha, K_tilde):


    KL_ref = sum([ref_hybrid_KL(_a, _S, K_alpha, K_beta_alpha, K_beta) for _a, _S in zip(a.T, S)])

    basis = OthogonallyDecoupledBasis(Dy, alpha, beta, a=a, L=chol_S)
    _, _, kl = basis.conditional_with_KL(kernel, np.empty((1, Dx)))
    KL = session_tf.run(kl)

    assert_allclose(KL, KL_ref, rtol=1e-6, atol=1e-6)



# def test_hybrid_kl(self):
#     Dx = 2
#     Dy = 3
#     M_alpha = 4
#     M_tilde = 4
#
#     alpha = np.random.randn(M_alpha, Dx)
#     X_tilde = np.random.randn(M_tilde, Dx)
#
#     kernel = Matern52(Dx)
#     K_alpha = kernel.compute_K_symm(alpha)
#     K_tilde = kernel.compute_K_symm(X_tilde)
#     K_tilde_alpha = kernel.compute_K(X_tilde, alpha)
#
#     a = np.random.randn(M_alpha, Dy)
#     m = np.random.randn(M_tilde, Dy)
#     U = np.random.randn(Dy, M_tilde, M_tilde)
#     S = np.einsum('dnN,dmN->dnm', U, U) + 1e-6 * np.eye(M_tilde)[None, :, :]
#     chol_S = np.linalg.cholesky(S)
#
#     KL_ref = sum([ref_hybrid_KL(_m, _S, _a, K_alpha, K_tilde_alpha, K_tilde)
#                   for _m, _S, _a, in zip(m.T, S, a.T)])
#
#     with tf.Session() as sess:
#         KL = sess.run(hybrid_gauss_kl(m, tf.identity(chol_S), a,
#                                       K_alpha,
#                                       K_tilde_alpha,
#                                       tf.identity(K_tilde)))
#
#     assert_allclose(KL, KL_ref)
#
