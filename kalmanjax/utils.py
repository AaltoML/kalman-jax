import jax.numpy as np
from jax.scipy.special import erfc
from jax.scipy.linalg import cho_factor, cho_solve
from jax import random
import numpy as nnp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, ListedColormap
pi = 3.141592653589793


def solve(P, Q):
    """
    Compute P^-1 Q, where P is a covariance matrix, using the Cholesky factoristion
    """
    L = cho_factor(P)
    return cho_solve(L, Q)


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


def plot(model, it_num, ax=None):
    post_mean, post_var, _, nlpd = model.predict()
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    lb = post_mean[:, 0] - 1.96 * post_var[:, 0] ** 0.5
    ub = post_mean[:, 0] + 1.96 * post_var[:, 0] ** 0.5
    ax.plot(model.t_train, model.y, 'k.', label='training observations')
    plt.plot(model.t_test, model.y_all[model.test_id], 'r.', alpha=0.4, label='test observations')
    ax.plot(model.t_all, post_mean, 'b', label='posterior mean')
    ax.fill_between(model.t_all[:, 0], lb, ub, color='b', alpha=0.05, label='95% confidence')
    ax.legend(loc=1)
    plt.xlim([model.t_test[0], model.t_test[-1]])
    plt.title('Test NLPD: %1.2f' % nlpd)
    plt.xlabel('time - $t$')
    plt.savefig('output/output_%02d.png' % it_num)
    plt.close()


def plot_2d_classification(m, it_num):
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # # xtest, ytest = np.mgrid[-2.8:2.8:100j, -2.8:2.8:100j]
    # # Xtest = np.vstack((xtest.flatten(), ytest.flatten())).T
    # for i, mark in [[1, 'o'], [0, 'o']]:
    #     ind = m.y[:, 0] == i
    #     # ax.plot(X[ind, 0], X[ind, 1], mark)
    #     ax.scatter(m.t_train[ind, 0], m.t_train[ind, 1], s=100, alpha=.5)
    # mu, var, _, nlpd_test = m.predict_2d()
    # ax.contour(m.t_test, m.y_all[m.test_id], mu.reshape(100, 100), levels=[.5],
    #            colors='k', linewidths=4.)
    # ax.axis('equal')
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    # # ax.axis('off')
    # ax.set_xlim(-2.8, 2.8)
    # ax.set_ylim(-2.8, 2.8)

    mu, var, _, nlpd_test = m.predict_2d()
    mu = np.squeeze(mu)
    lim = 3
    cmap_ = [[1, 0.498039215686275, 0.0549019607843137], [0.12156862745098, 0.466666666666667, 0.705882352941177]]
    cmap = hsv_to_rgb(
        interp1d([-1, 1], rgb_to_hsv(cmap_), axis=0
                 )(m.likelihood.link_fn(nnp.linspace(-3.5, 3.5, num=64))))
    newcmp = ListedColormap(cmap)

    Xtest, Ytest = nnp.mgrid[-3.:3.:100j, -3.:3.:100j]
    plt.figure(2)
    plt.imshow(m.likelihood.link_fn(mu).T, cmap=newcmp, extent=[-lim, lim, -lim, lim], origin='lower')
    plt.contour(Xtest, Ytest, mu, levels=[.0], colors='k', linewidths=1.5)
    # plt.axis('equal')
    for i in [1, 0]:
        ind = m.y[:, 0] == i
        plt.scatter(m.t_train[ind, 0], m.t_train[ind, 1], s=50, alpha=.5, edgecolor='k')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    plt.savefig('output/output_%02d.png' % it_num)
    plt.close()
