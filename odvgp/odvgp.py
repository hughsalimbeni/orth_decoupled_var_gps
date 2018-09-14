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

from gpflow.models import GPModel
from gpflow.params.dataholders import DataHolder, Minibatch
from gpflow import settings, params_as_tensors

from .gaussian_bases import OthogonallyDecoupledBasis, DecoupledBasis, HybridDecoupledBasis


class Variational_GP(GPModel):
    """
    Similar to gpflow's SVGP model, but with a more general basis. The basis implements one method: conditional_with_KL.
    """
    def __init__(self, X, Y, kernel, likelihood, basis,
                 minibatch_size=None,
                 mean_function=None, **kw):
        if minibatch_size is None or minibatch_size>=X.shape[0]:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        GPModel.__init__(self, X, Y, kernel, likelihood, mean_function, basis.num_latent, **kw)

        self.num_data = X.shape[0]
        self.basis = basis

    @params_as_tensors
    def _build_likelihood(self):
        fmean, fvar, KL = self.basis.conditional_with_KL(self.kern, self.X, full_cov=False)

        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        scale = tf.cast(self.num_data, settings.float_type)
        scale /= tf.cast(tf.shape(self.X)[0], settings.float_type)
        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        mu, var, _ = self.basis.conditional_with_KL(self.kern, Xnew, full_cov=full_cov)
        return mu + self.mean_function(Xnew), var


#### Convenience classses


class ODVGP(Variational_GP):
    """
    Convenience class for a Variational GP with the orthogonally decoupled basis, from
    
    @inproceedings{salimbeni2018decoupled,
      title={Orthogonally Decoupled Variational Gaussian Processes},
      author={Salimbeni, Hugh and Cheng, Ching-An and Boots, Byron and Deisenroth, Marc},
      booktitle={Advances in Neural Information Processing Systems},
      year={2018}
    }
        
    """
    def __init__(self, X, Y, kernel, likelihood, alpha, beta,
                 minibatch_size=None,
                 mean_function=None,
                 gamma_minibatch_size=None,
                 **kwargs):
        num_latent = kwargs['num_latent'] if 'num_latent' in kwargs else Y.shape[1]

        basis = OthogonallyDecoupledBasis(num_latent, alpha, beta, minibatch_size=gamma_minibatch_size)

        Variational_GP.__init__(self, X, Y, kernel, likelihood, basis,
                                minibatch_size=minibatch_size,
                                mean_function=mean_function,
                                **kwargs)


class DVGP(Variational_GP):
    """
    Convenience class for a Variational GP with the decoupled basis, from
    
    @inproceedings{cheng2017variational,
      title={Variational Inference for Gaussian Process Models with Linear Complexity},
      author={Cheng, Ching-An and Boots, Byron},
      booktitle={Advances in Neural Information Processing Systems},
      year={2017}
    }
    """
    def __init__(self, X, Y, kernel, likelihood, alpha, beta,
                 minibatch_size=None,
                 mean_function=None,
                 gamma_minibatch_size=None,
                 **kwargs):
        num_latent = kwargs['num_latent'] if 'num_latent' in kwargs else Y.shape[1]

        basis = DecoupledBasis(num_latent, alpha, beta, minibatch_size=gamma_minibatch_size)

        Variational_GP.__init__(self, X, Y, kernel, likelihood, basis,
                                minibatch_size=minibatch_size,
                                mean_function=mean_function,
                                **kwargs)


class HDVGP(Variational_GP):
    """
    Convenience class for a Variational GP with the hybrid basis, from the appendix of
    
    @inproceedings{cheng2017variational,
      title={Variational Inference for Gaussian Process Models with Linear Complexity},
      author={Cheng, Ching-An and Boots, Byron},
      booktitle={Advances in Neural Information Processing Systems},
      year={2017}
    }
    """
    def __init__(self, X, Y, kernel, likelihood, alpha, beta,
                 minibatch_size=None,
                 mean_function=None,
                 gamma_minibatch_size=None,
                 **kwargs):
        num_latent = kwargs['num_latent'] if 'num_latent' in kwargs else Y.shape[1]

        basis = HybridDecoupledBasis(num_latent, alpha, beta, minibatch_size=gamma_minibatch_size)

        Variational_GP.__init__(self, X, Y, kernel, likelihood, basis,
                                minibatch_size=minibatch_size,
                                mean_function=mean_function,
                                **kwargs)