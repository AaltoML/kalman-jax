import jax.numpy as np
from jax.scipy.special import erfc
from jax import random
import matplotlib.pyplot as plt
pi = 3.141592653589793


def softplus_list(x_):
    """
    Softplus positiviy mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """
    y_ = [np.log(1 + np.exp(-np.abs(x_[0]))) + np.maximum(x_[0], 0)]
    for i in range(1, len(x_)):
        if x_[i] is not []:
            y_ = y_ + [np.log(1 + np.exp(-np.abs(x_[i]))) + np.maximum(x_[i], 0)]
    return y_


def softplus_inv_list(x_):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    Loop over the elements of the paramter list so we can handle the special case
    where an element is empty
    """
    y_ = x_
    for i in range(len(x_)):
        if x_[i] is not []:
            y_[i] = np.log(1-np.exp(-np.abs(x_[i]))) + np.maximum(x_[i], 0)
    return y_


def softplus_inv(x_):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    """
    if x_ is None:
        return x_
    else:
        return np.log(np.exp(x_) - 1)


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


def gaussian_moment_match(y, m, v, hyp=None):
    """
    Closed form Gaussian moment matching.
    Calculates the log partition function of the EP tilted distribution:
        logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
    and its derivatives w.r.t. mₙ, which are required for moment matching.
    :param y: observed data (yₙ) [scalar]
    :param m: cavity mean (mₙ) [scalar]
    :param v: cavity variance (vₙ) [scalar]
    :param hyp: observation noise variance (σ²) [scalar]
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
    # 𝓝(yₙ|fₙ,σ²) = 𝓝(fₙ|yₙ,σ²)
    site_mean = y
    site_var = hyp
    return lZ, site_mean, site_var


def sample_gaussian_noise(latent_mean, likelihood_var):
    lik_std = np.sqrt(likelihood_var)
    gaussian_sample = latent_mean + lik_std[..., np.newaxis] * random.normal(random.PRNGKey(123),
                                                                             shape=latent_mean.shape)
    return gaussian_sample


def rotation_matrix(dt, omega):
    """
    Discrete time rotation matrix
    :param dt: step size [1]
    :param omega: frequency [1]
    :return:
        R: rotation matrix [2, 2]
    """
    R = np.array([
        [np.cos(omega * dt), -np.sin(omega * dt)],
        [np.sin(omega * dt),  np.cos(omega * dt)]
    ])
    return R


def plot(mod, it_num, ax=None):
    post_mean, post_var, _ = mod.predict()
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    lb_ = post_mean[:, 0] - 1.96 * post_var[:, 0] ** 0.5
    ub_ = post_mean[:, 0] + 1.96 * post_var[:, 0] ** 0.5
    ax.plot(mod.t_train, mod.y, 'k.', label='observations')
    ax.plot(mod.t_all, post_mean, 'b', label='posterior mean')
    ax.fill_between(mod.t_all, lb_, ub_, color='b', alpha=0.05, label='95% confidence')
    ax.legend()
    plt.xlim([mod.t_test[0], mod.t_test[-1]])
    plt.title('GP regression via Kalman smoothing')
    plt.xlabel('time - $t$')
    plt.savefig('output/test_%d.png' % it_num)
    plt.close()
