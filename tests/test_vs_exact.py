# # Copyright 2018 Hugh Salimbeni (hrs13@ic.ac.uk), Ching-An Cheng (cacheng@gatech.edu)
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.
#
#
#
# class ExactDecoupledGaussian(DecoupledGaussian):
#     def __init__(self, D, alpha, beta):
#         DecoupledGaussian.__init__(self, D, alpha, beta)
#         self._X = None
#         self._Y = None
#         self._likelihood_var = None
#         self._kernel = None
#         self.a_shape = [len(alpha), D]
#         self.chol_B_shape = [D, len(beta), len(beta)]
#
#     def update_data(self, X, Y, likelihood_var, kernel):
#         self._X = X
#         self._Y = Y
#         self._likelihood_var = likelihood_var
#         self._kernel = kernel
#
#     def _build_S(self):
#         K_beta = self.beta.Kuu(self._kernel, jitter=1e-6)
#         K_beta_X = self.beta.Kuf(self._kernel, self._X)
#         KK = tf.matmul(K_beta_X, K_beta_X, transpose_b=True)/self._likelihood_var
#         KK += K_beta
#         return tf.matmul(K_beta, tf.matrix_solve(KK, K_beta))
#
#     def _build_a(self):
#         K_alpha = self.alpha.Kuu(self._kernel, jitter=1e-6)
#         K_alpha_X = self.alpha.Kuf(self._kernel, self._X)
#         KK = tf.matmul(K_alpha_X, K_alpha_X, transpose_b=True) / self._likelihood_var
#         KK += K_alpha
#         return tf.matrix_solve(KK, tf.matmul(K_alpha_X, self._Y))/self._likelihood_var
#
#     @property
#     def a(self):
#         return tf.reshape(self._build_a(), self.a_shape)
#
#     @a.setter
#     def a(self, val):
#         pass
#
#     @property
#     def chol_B(self):   # TODO: S might be 3d-tensor
#         S = self._build_S()
#         K = self.beta.Kuu(self._kernel, jitter=1e-6)
#         B_inv = tf.matmul(K, tf.matrix_solve(K - S, K)) - K
#         # B = tf.matrix_inverse(B_inv)
#         B_inv = B_inv + tf.eye(tf.shape(B_inv)[0], dtype=tf.float64)*1e-6
#
#         chol_B_inv = tf.cholesky(B_inv)
#         chol_B = tf.matrix_inverse(tf.transpose(chol_B_inv))
#         # chol_B = tf.cholesky(B)
#         chol_B = tf.tile(chol_B[None, :, :], [self.chol_B_shape[0], 1, 1])
#         return tf.reshape(chol_B, self.chol_B_shape)
#
#     @chol_B.setter
#     def chol_B(self, val):
#         pass
#
#
#
# class ExactDecoupled_a_S(Decoupled_a_S):
#     def __init__(self, D, alpha, beta, **kwargs):
#         Decoupled_a_S.__init__(self, D, alpha, beta, **kwargs)
#         self._X = None
#         self._Y = None
#         self._likelihood_var = None
#         self._kernel = None
#         self.a_shape = [len(alpha), D]
#         self.S_shape = [D, len(beta), len(beta)]
#
#     def update_data(self, X, Y, likelihood_var, kernel):
#         self._X = X
#         self._Y = Y
#         self._likelihood_var = likelihood_var
#         self._kernel = kernel
#
#     def _build_S(self):
#         K_beta = self.beta.Kuu(self._kernel, jitter=1e-6)
#         K_beta_X = self.beta.Kuf(self._kernel, self._X)
#         KK = tf.matmul(K_beta_X, K_beta_X, transpose_b=True)/self._likelihood_var
#         KK += K_beta
#         return tf.matmul(K_beta, tf.matrix_solve(KK, K_beta))
#
#     def _build_a(self):
#         K_alpha = self.alpha.Kuu(self._kernel, jitter=1e-6)
#         K_alpha_X = self.alpha.Kuf(self._kernel, self._X)
#         KK = tf.matmul(K_alpha_X, K_alpha_X, transpose_b=True) / self._likelihood_var
#         KK += K_alpha
#         return tf.matrix_solve(KK, tf.matmul(K_alpha_X, self._Y))/self._likelihood_var
#
#     @property
#     def a(self):
#         return tf.reshape(self._build_a(), self.a_shape)
#
#     @a.setter
#     def a(self, val):
#         pass
#
#     @property
#     def S(self):
#         S = self._build_S()
#         return tf.tile(S[None, :, :], [self.S_shape[0], 1, 1])
#
#     @S.setter
#     def S(self, val):
#         pass
#
#
