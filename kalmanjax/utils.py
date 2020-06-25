import jax.numpy as np
from jax.scipy.special import erfc
from jax.scipy.linalg import cho_factor, cho_solve
from jax import random
from jax.ops import index_add, index
import numpy as nnp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, ListedColormap
from numpy.polynomial.hermite import hermgauss
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
    plt.savefig('output/output_%04d.png' % it_num)
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

    mu, var, _, nlpd_test, _, _ = m.predict_2d()
    mu = np.squeeze(mu)
    lim = 2.8
    label0, label1 = -1., 1.  # class labels are +/-1
    cmap_ = [[1, 0.498039215686275, 0.0549019607843137], [0.12156862745098, 0.466666666666667, 0.705882352941177]]
    cmap = hsv_to_rgb(
        interp1d([label0, label1], rgb_to_hsv(cmap_), axis=0
                 )(m.likelihood.link_fn(nnp.linspace(-3.5, 3.5, num=64))))
    newcmp = ListedColormap(cmap)

    Xtest, Ytest = nnp.mgrid[-2.8:2.8:100j, -2.8:2.8:100j]
    plt.figure()
    im = plt.imshow(m.likelihood.link_fn(mu).T, cmap=newcmp, extent=[-lim, lim, -lim, lim], origin='lower',
                    vmin=label0, vmax=label1)
    cb = plt.colorbar(im)
    cb.set_ticks([cb.vmin, 0, cb.vmax])
    cb.set_ticklabels([-1, 0, 1])
    plt.contour(Xtest, Ytest, mu, levels=[.0], colors='k', linewidths=1.5)
    # plt.axis('equal')
    for label in [1, 0]:
        ind = m.y[:, 0] == label
        plt.scatter(m.t_train[ind, 0], m.t_train[ind, 1], s=50, alpha=.5, edgecolor='k')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    plt.title('Iteration: %02d' % (it_num + 1), loc='right', fontweight='bold')
    plt.savefig('output/output_%04d.png' % it_num)
    plt.close()


def plot_2d_classification_filtering(m, it_num, plot_num, mu_prev=None):
    mu, var, _, nlpd_test, mu_filt, var_filt = m.predict_2d()
    mu, mu_filt = np.squeeze(mu), np.squeeze(mu_filt)
    if mu_prev is None:
        mu_plot = nnp.zeros_like(mu)
    else:
        mu_plot = mu_prev
    lim = 2.8
    label0, label1 = -1., 1.  # class labels are +/-1
    cmap_ = [[1, 0.498039215686275, 0.0549019607843137], [0.12156862745098, 0.466666666666667, 0.705882352941177]]
    cmap = hsv_to_rgb(
        interp1d([label0, label1], rgb_to_hsv(cmap_), axis=0
                 )(m.likelihood.link_fn(nnp.linspace(-3.5, 3.5, num=64))))
    newcmp = ListedColormap(cmap)

    Xtest, Ytest = nnp.mgrid[-lim:lim:100j, -lim:lim:100j]

    for i in range(Xtest.shape[0]):
        mu_plot[i] = mu_filt[i]
        plt.figure()
        im = plt.imshow(m.likelihood.link_fn(mu_plot).T, cmap=newcmp, extent=[-lim, lim, -lim, lim], origin='lower',
                        vmin=label0, vmax=label1)
        cb = plt.colorbar(im)
        cb.set_ticks([cb.vmin, 0, cb.vmax])
        cb.set_ticklabels([-1, 0, 1])
        # plt.contour(Xtest, Ytest, mu_plot, levels=[.0], colors='k', linewidths=1.5)
        # plt.axis('equal')
        for label in [1, 0]:
            ind = m.y[:, 0] == label
            plt.scatter(m.t_train[ind, 0], m.t_train[ind, 1], s=50, alpha=.5, edgecolor='k')
        plt.plot([Xtest[i, 0], Xtest[i, 0]], [-lim, lim], 'k', alpha=0.4)
        plt.title('Iteration: %02d' % (it_num + 1), loc='right', fontweight='bold')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.savefig('output/output_%04d.png' % plot_num)
        plt.close()
        plot_num += 1
    for i in range(Xtest.shape[0] - 1, -1, -1):
        mu_plot[i] = mu[i]
        plt.figure()
        im = plt.imshow(m.likelihood.link_fn(mu_plot).T, cmap=newcmp, extent=[-lim, lim, -lim, lim], origin='lower',
                        vmin=label0, vmax=label1)
        cb = plt.colorbar(im)
        cb.set_ticks([cb.vmin, 0, cb.vmax])
        cb.set_ticklabels([-1, 0, 1])
        # plt.contour(Xtest, Ytest, mu_plot, levels=[.0], colors='k', linewidths=1.5)
        # plt.axis('equal')
        for label in [1, 0]:
            ind = m.y[:, 0] == label
            plt.scatter(m.t_train[ind, 0], m.t_train[ind, 1], s=50, alpha=.5, edgecolor='k')
        plt.plot([Xtest[i, 0], Xtest[i, 0]], [-lim, lim], 'k', alpha=0.4)
        plt.title('Iteration: %02d' % (it_num + 1), loc='right', fontweight='bold')
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.savefig('output/output_%04d.png' % plot_num)
        plt.close()
        plot_num += 1
    return plot_num, mu_plot


def gauss_hermite(dim=1, num_quad_pts=20):
    """
    Return weights and sigma-points for Gauss-Hermite cubature
    """
    sigma_pts, weights = hermgauss(num_quad_pts)  # Gauss-Hermite sigma points and weights
    sigma_pts = np.sqrt(2) * sigma_pts
    weights = weights / np.sqrt(pi)  # scale weights by 1/√π
    return sigma_pts, weights


def symmetric_cubature_third_order(dim=1, kappa=None):
    """
    Return weights and sigma-points for the symmetric cubature rule of order 5, for
    dimension dim with parameter kappa (default 0).
    """
    if kappa is None:
        # kappa = 1 - dim
        kappa = 0  # CKF
    if (dim == 1) and (kappa == 0):
        weights = np.array([0., 0.5, 0.5])
        sigma_pts = np.array([0., 1., -1.])
        # sigma_pts = np.array([-1., 0., 1.])
        # weights = np.array([0.5, 0., 0.5])
        # u = 1
    elif (dim == 2) and (kappa == 0):
        weights = np.array([0., 0.25, 0.25, 0.25, 0.25])
        sigma_pts = np.block([[0., 1.4142,  0., -1.4142, 0.],
                              [0., 0., 1.4142, 0., -1.4142]])
        # u = 1.4142
    elif (dim == 3) and (kappa == 0):
        weights = np.array([0., 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667])
        sigma_pts = np.block([[0., 1.7321, 0.,  0., -1.7321, 0., 0.],
                              [0., 0., 1.7321, 0., 0., -1.7321, 0.],
                              [0., 0., 0., 1.7321, 0., 0., -1.7321]])
        # u = 1.7321
    else:
        # weights
        weights = np.zeros([1, 2 * dim + 1])
        weights = index_add(weights, index[0, 0], kappa / (dim + kappa))
        for j in range(1, 2 * dim + 1):
            wm = 1 / (2 * (dim + kappa))
            weights = index_add(weights, index[0, j], wm)
        # Sigma points
        sigma_pts = np.block([np.zeros([dim, 1]), np.eye(dim), - np.eye(dim)])
        sigma_pts = np.sqrt(dim + kappa) * sigma_pts
        # u = np.sqrt(n + kappa)
    return sigma_pts, weights  # , u


def symmetric_cubature_fifth_order(dim=1):
    """
    Return weights and sigma-points for the symmetric cubature rule of order 5
    """
    if dim == 1:
        weights = np.array([0.6667, 0.1667, 0.1667])
        sigma_pts = np.array([0., 1.7321, -1.7321])
    elif dim == 2:
        weights = np.array([0.4444, 0.1111, 0.1111, 0.1111, 0.1111, 0.0278, 0.0278, 0.0278, 0.0278])
        sigma_pts = np.block([[0., 1.7321, -1.7321, 0., 0., 1.7321, -1.7321, 1.7321, -1.7321],
                              [0., 0., 0., 1.7321, -1.7321, 1.7321, -1.7321, -1.7321, 1.7321]])
    elif dim == 3:
        weights = np.array([0.3333, 0.0556, 0.0556, 0.0556, 0.0556, 0.0556, 0.0556, 0.0278, 0.0278, 0.0278,
                            0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278])
        sigma_pts = np.block([[0., 1.7321, -1.7321, 0., 0., 0., 0., 1.7321, -1.7321, 1.7321, -1.7321, 1.7321,
                               -1.7321, 1.7321, -1.7321, 0., 0., 0., 0.],
                              [0., 0., 0., 1.7321, -1.7321, 0., 0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0.,
                               0., 1.7321, -1.7321, 1.7321, -1.7321],
                              [0., 0., 0., 0., 0., 1.7321, -1.7321, 0., 0., 0., 0., 1.7321, -1.7321, -1.7321,
                               1.7321, 1.7321, -1.7321, -1.7321, 1.7321]])
    # else:
    #     # The weights and sigma-points from McNamee & Stenger
    #     I0 = 1.
    #     I2 = 1.
    #     I4 = 3.
    #     I22 = 1.
    #     u = np.array(np.sqrt(I4 / I2))
    #     A0 = I0 - dim * (I2 / I4) ** 2 * (I4 - 0.5 * (dim - 1) * I22)
    #     A1 = 0.5 * (I2 / I4) ** 2 * (I4 - (dim - 1) * I22)
    #     A11 = 0.25 * (I2 / I4) ** 2 * I22
    #     U0 = sym_set(dim)
    #     U1 = sym_set(dim, u)
    #     U2 = sym_set(dim, np.block([u, u]))
    #     sigma_pts = np.block([U0, U1, U2])
    #     weights = np.block([A0 * np.ones([1, U0.shape[1]]),
    #                         A1 * np.ones([1, U1.shape[1]]),
    #                         A11 * np.ones([1, U2.shape[1]])])
    return sigma_pts, weights


# def sym_set(n, gen=None):
#     # U = sym_set(n, gen)
#     nonzero = 0
#     if gen is None:
#         if nonzero:
#             U = []
#         else:
#             U = np.zeros([n, 1])
#     else:
#         U = []
#         for i in range(n):
#             u = np.zeros([n, 1])
#             u[i] = gen[0]
#             if gen.shape[1] > 1:
#                 if abs(gen(1) - gen(2)) < 1e-16:
#                     V = sym_set(n - i, gen[1:])
#                     for j in range(V.shape[1]):
#                         u[i:] = V[:, j]
#                         U = np.block([U, u - u])
#                 else:
#                     V = sym_set(n - 1, gen[1:])
#                     for j in range(V.shape[1]):
#                         # u([1: i - 1, i + 1: end]) = V(:, j)
#                         U = np.block([U, u - u])
#             else:
#                 U = np.block([U, u - u])
#     return U
