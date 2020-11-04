import jax.numpy as np
from utils import scaled_squared_euclid_dist


class StationaryKernel(object):
    """
    Classic form of a stationary kernel, to be used for the spatial part of spatio-temporal priors.

    These kernels are adapted from GPflow: https://github.com/GPflow/GPflow

    LICENSE:

        Copyright The Contributors to the GPflow Project. All Rights Reserved.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.

    """

    def K(self, X, X2, lengthscale=1., variance=1.):
        r2 = scaled_squared_euclid_dist(X, X2, lengthscale)
        return self.K_r2(r2, variance)

    def K_r2(self, r2, variance=1.):
        # Clipping around the (single) float precision which is ~1e-45.
        r = np.sqrt(np.maximum(r2, 1e-36))
        return self.K_r(r, variance)

    @staticmethod
    def K_r(r, variance=1.):
        raise NotImplementedError('kernel not implemented')


class Matern12Kernel(StationaryKernel):
    """
    The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ² is the variance parameter
    """

    @staticmethod
    def K_r(r, variance=1.):
        return variance * np.exp(-r)


class Matern32Kernel(StationaryKernel):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @staticmethod
    def K_r(r, variance=1.):
        sqrt3 = np.sqrt(3.0)
        return variance * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)


class Matern52Kernel(StationaryKernel):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is

    k(r) = σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @staticmethod
    def K_r(r, variance=1.):
        sqrt5 = np.sqrt(5.0)
        return variance * (1.0 + sqrt5 * r + 5.0 / 3.0 * np.square(r)) * np.exp(-sqrt5 * r)
