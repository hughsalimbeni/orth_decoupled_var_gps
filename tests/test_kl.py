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

from gpflow import settings
from gpflow.kernels import Matern52
from gpflow.likelihoods import Gaussian
from gpflow.test_util import session_tf

from odvgp.gaussian_bases import DecoupledBasis, OthogonallyDecoupledBasis, HybridDecoupledBasis


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

def ref_orthogonally_decoupled_KL(a_gamma, a_beta, S, K_gamma, K_gamma_beta, K_beta):
    # appendix A1
    KL = -0.5 * K_beta.shape[0]

    K_beta_inv = np.linalg.inv(K_beta)
    KL += 0.5 * np.sum(a_gamma * (K_gamma @ a_gamma))
    KL += 0.5 * np.sum(a_beta * (K_beta @ a_beta))
    KL -= 0.5 * np.sum(a_gamma * (K_gamma_beta @ K_beta_inv @ K_gamma_beta.T @ a_gamma))

    KL += 0.5 * np.linalg.slogdet(K_beta)[1]
    KL -= 0.5 * np.linalg.slogdet(S)[1]

    KL += 0.5 * np.sum(S * K_beta_inv)

    return KL

def ref_hybrid_KL(a, m, S, K_alpha, K_alpha_tilde, K_tilde):
    """
    eq 14 of Cheng and Boots 2017
    """
    K_inv = np.linalg.inv(K_tilde)
    KL = 0.5 * np.sum(m * (K_inv @ m))
    KL += np.sum(m * (K_inv @ K_alpha_tilde.T @ a))
    KL += 0.5 * np.sum(a * (K_alpha @ a))

    KL -= 0.5 * np.linalg.slogdet(K_inv @ S)[1]  # in eq 14 the factor of 0.5 is missing

    KL += 0.5 * np.sum(S * K_inv)

    KL -= 0.5 * K_tilde.shape[0]
    return KL


def test_decoupled_kl(session_tf):
    Dx = 2
    Dy = 3
    M_alpha = 10
    M_beta = 11

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
    M_gamma = 4
    M_beta = 5

    gamma = np.random.randn(M_gamma, Dx)
    beta = np.random.randn(M_beta, Dx)

    kernel = Matern52(Dx)

    K_gamma = kernel.compute_K_symm(gamma)
    K_beta = kernel.compute_K_symm(beta) + np.eye(M_beta) * settings.jitter
    K_gamma_beta = kernel.compute_K(gamma, beta)

    a_gamma = np.random.randn(M_gamma, Dy)
    a_beta = np.random.randn(M_beta, Dy)
    U = np.random.randn(Dy, M_beta, M_beta)
    S = np.einsum('dnN,dmN->dnm', U, U) + settings.jitter * np.eye(M_beta)[None, :, :]
    chol_S = np.linalg.cholesky(S)

    KL_ref = sum([ref_orthogonally_decoupled_KL(_a_gamma, _a_beta, _S, K_gamma, K_gamma_beta, K_beta)
                  for _a_gamma, _a_beta, _S in zip(a_gamma.T, a_beta.T, S)])

    # undo the pre-conditioning
    D = kernel.compute_Kdiag(gamma) - np.sum(K_gamma_beta.T * np.linalg.solve(K_beta, K_gamma_beta.T), 0)

    basis = OthogonallyDecoupledBasis(Dy, gamma, beta,
                                      a_gamma=a_gamma * D[:, None], a_beta=K_beta @ a_beta, L=chol_S)

    _, _, kl = basis.conditional_with_KL(kernel, np.empty((1, Dx)))
    KL = session_tf.run(kl)
    print(KL - KL_ref)
    assert_allclose(KL, KL_ref, rtol=1e-6, atol=1e-6)

def test_hybrid_kl(session_tf):
    Dx = 2
    Dy = 3
    M_gamma = 4
    M_beta = 5

    gamma = np.random.randn(M_gamma, Dx)
    beta = np.random.randn(M_beta, Dx)

    kernel = Matern52(Dx)

    K_gamma = kernel.compute_K_symm(gamma)
    K_beta = kernel.compute_K_symm(beta) + np.eye(M_beta) * 1e-6
    K_gamma_beta = kernel.compute_K(gamma, beta)

    a_gamma = np.random.randn(M_gamma, Dy)
    a_beta = np.random.randn(M_beta, Dy)
    U = np.random.randn(Dy, M_beta, M_beta)
    S = np.einsum('dnN,dmN->dnm', U, U) + 1e-6 * np.eye(M_beta)[None, :, :]
    chol_S = np.linalg.cholesky(S)

    KL_ref = sum([ref_hybrid_KL(_a_gamma, _a_beta, _S, K_gamma, K_gamma_beta, K_beta)
                  for _a_gamma, _a_beta, _S in zip(a_gamma.T, a_beta.T, S)])

    basis = HybridDecoupledBasis(Dy, gamma, beta,
                                 a_gamma=a_gamma, a_beta=a_beta, L=chol_S)

    _, _, kl = basis.conditional_with_KL(kernel, np.empty((1, Dx)))
    KL = session_tf.run(kl)
    assert_allclose(KL, KL_ref, rtol=1e-6, atol=1e-6)
