import jax.numpy as np
from jax.scipy.special import erfc
from jax import partial, jit
pi = 3.141592653589793


def softplus_inv(x_):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """
    y_ = x_
    for i in range(len(x_)):
        if x_[i] is not []:
            y_[i] = np.log(np.exp(x_[i]) - 1)
    return y_


def logphi(z):
    """
    Calculate the log Gaussian CDF, used for closed form moment matching when the EP power is 1,
        logΦ(z) = log[(1 + erf(z / √2)) / 2]
    for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    and its derivative w.r.t. z
        dlogΦ(z)/dz = 𝓝(z|0,1) / Φ(z)
    :param z: input value, typically z = (my) / √(1 + v) [scalar]
    :return:
        lp: logΦ(z) [scalar]
        dlp: dlogΦ(z)/dz [scalar]
    """
    z = np.real(z)
    # erfc(z) = 1 - erf(z) is the complementary error function
    lp = np.log(erfc(-z / np.sqrt(2.0)) / 2.0)  # log Φ(z)
    dlp = np.exp(-z * z / 2.0 - lp) / np.sqrt(2.0 * pi)  # derivative w.r.t. z
    return lp, dlp


@partial(jit, static_argnums=4)
def gaussian_moment_match(y, m, v, hyp=None, site_update=True):
    """
    Closed form Gaussian moment matching.
    Calculates the log partition function of the EP tilted distribution:
        logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
    and its derivatives w.r.t. mₙ, which are required for moment matching.
    :param y: observed data (yₙ) [scalar]
    :param m: cavity mean (mₙ) [scalar]
    :param v: cavity variance (vₙ) [scalar]
    :param hyp: observation noise variance (σ²) [scalar]
    :param site_update: if True, return the derivatives of the log partition function w.r.t. mₙ [bool]
    :return:
        lZ: the log partition function, logZₙ [scalar]
        dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
    """
    # log partition function, lZ:
    # logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #       = log 𝓝(yₙ|mₙ,σ²+vₙ)
    lZ = (
            - (y - m) ** 2 / (hyp + v) / 2
            - np.log(np.maximum(2 * pi * (hyp + v), 1e-10)) / 2
    )
    if site_update:
        # 𝓝(yₙ|fₙ,σ²) = 𝓝(fₙ|yₙ,σ²)
        site_mean = y
        site_var = hyp
        return lZ, site_mean, site_var
    else:
        return lZ
