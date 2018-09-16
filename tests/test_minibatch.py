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
import numpy as np

from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF
from odvgp.odvgp import ODVGP, DVGP, HDVGP

@pytest.mark.parametrize('Model', [ODVGP, DVGP, HDVGP])
def test_minibatch(Model):

    batch_size = 9
    N = 10
    Dx, Dy = 3, 2

    np.random.seed(1)
    X = np.random.randn(N, Dx)
    Y = np.random.randn(N, Dy)
    M_gamma, M_beta = 3, 2

    gamma = np.random.randn(M_gamma, Dx)
    beta = np.random.randn(M_beta, Dx)

    kern = RBF(Dx)
    lik = Gaussian()
    lik.variance = 0.1

    a_gamma = np.random.randn(M_gamma, Dy)
    a_beta = np.random.randn(M_beta, Dy)
    L = np.random.randn(Dy, M_beta, M_beta)

    model_batch = Model(X, Y, kern, lik, gamma, beta,
                        minibatch_size=batch_size, gamma_minibatch_size=batch_size)
    model_full = Model(X, Y, kern, lik, gamma, beta)

    for model in [model_batch, model_full]:
        model.basis.a_beta = a_beta
        model.basis.a_gamma = a_gamma
        model.basis.L = L

    S = 1000
    L_batch = [model_batch.compute_log_likelihood() for _ in range(S)]
    L_full = model_full.compute_log_likelihood()
    print(L_batch)
    mean = np.average(L_batch)
    std_err = np.std(L_batch) / S**0.5

    # check within 2 std, which is about a 95% CI
    assert L_full < mean + std_err * 2
    assert L_full > mean - std_err * 2
