import jax.numpy as np
from jax.scipy.special import erfc
from jax.scipy.linalg import cho_factor, cho_solve
from jax import random
from jax.ops import index_add, index_update, index
import numpy as nnp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, ListedColormap
from numpy.polynomial.hermite import hermgauss
import itertools
pi = 3.141592653589793


def ensure_positive_precision(K):
    """
    Check whether matrix K has positive diagonal elements.
    If not, then replace the negative elements with default value 0.01
    """
    K_diag = np.diag(np.diag(K))
    K = np.where(np.any(np.diag(K) < 0), np.where(K_diag < 0, 1e-2, K_diag), K)
    return K


def solve(P, Q):
    """
    Compute P^-1 Q, where P is a PSD matrix, using the Cholesky factoristion
    """
    L = cho_factor(P)
    return cho_solve(L, Q)


def inv(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P)
    return cho_solve(L, np.eye(P.shape[0]))


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


def softplus(x_):
    # return np.log(1 + np.exp(x_))
    return np.log(1 + np.exp(-np.abs(x_))) + np.maximum(x_, 0)  # safer version


def sigmoid(x_):
    return np.exp(x_) / (np.exp(x_) + 1.)


def softplus_inv(x_):
    """
    Inverse of the softplus positiviy mapping, used for transforming parameters.
    """
    if x_ is None:
        return x_
    else:
        # return np.log(np.exp(x_) - 1)
        return np.log(1 - np.exp(-np.abs(x_))) + np.maximum(x_, 0)  # safer version


def input_admin(t, y, r):
    """
    Order the inputs.
    :param t: training inputs [N, 1]
    :param y: observations at the training inputs [N, 1]
    :param r: training spatial inputs
    :return:
        t_train: training inputs [N, 1]
        y_train: training observations [N, R]
        r_train: training spatial inputs [N, R]
        dt_train: training step sizes, Δtₙ = tₙ - tₙ₋₁ [N, 1]
    """
    assert t.shape[0] == y.shape[0]
    if t.ndim < 2:
        t = nnp.expand_dims(t, 1)  # make 2-D
    if y.ndim < 2:
        y = nnp.expand_dims(y, 1)  # make 2-D
    if r is None:
        r = nnp.nan * t  # np.empty((1,) + x.shape[1:]) * np.nan
    if r.ndim < 2:
        r = nnp.expand_dims(r, 1)  # make 2-D
    ind = nnp.argsort(t[:, 0], axis=0)
    t_train = t[ind, ...]
    y_train = y[ind, ...]
    r_train = r[ind, ...]
    dt_train = nnp.concatenate([np.array([0.0]), nnp.diff(t_train[:, 0])])
    return (np.array(t_train, dtype=np.float64), np.array(y_train, dtype=np.float64),
            np.array(r_train, dtype=np.float64), np.array(dt_train, dtype=np.float64))


def test_input_admin(t, y, r, t_test, y_test, r_test):
    """
    TODO: tidy this function up
    Order the inputs, remove duplicates, and index the train and test input locations.
    :param t: training inputs [N, 1]
    :param y: observations at the training inputs [N, 1]
    :param r: training spatial inputs
    :param t_test: testing inputs [N*, 1]
    :param y_test: observations at the test inputs [N*, 1]
    :param r_test: test spatial inputs
    :return:
        t_all: the combined and sorted training and test inputs [N + N*, 1]
        y_all: an array of observations y augmented with nans at test locations [N + N*, R]
        r_all: spatial inputs with nans at test locations [N + N*, R]
        dt_all: combined training and test step sizes, Δtₙ = tₙ - tₙ₋₁ [N + N*, 1]
        dt_train: training step sizes, Δtₙ = tₙ - tₙ₋₁ [N, 1]
        train_id: an array of indices corresponding to the training inputs [N, 1]
        test_id: an array of indices corresponding to the test inputs [N*, 1]
        mask: boolean array to signify training locations [N + N*, 1]
    """
    assert t.shape[0] == y.shape[0]
    if t.ndim < 2:
        t = np.expand_dims(t, 1)  # make 2-D
    if y.ndim < 2:
        y = np.expand_dims(y, 1)  # make 2-D
    if r is None:
        r = np.nan * t  # np.empty((1,) + x.shape[1:]) * np.nan
    if r.ndim < 2:
        r = np.expand_dims(r, 1)  # make 2-D
    ind = np.argsort(t[:, 0], axis=0)
    t_train = t[ind, ...]
    y_train = y[ind, ...]
    r_train = r[ind, ...]
    if t_test is None:
        t_test = np.empty((1,) + t_train.shape[1:]) * np.nan
        r_test = np.empty((1,) + t_train.shape[1:]) * np.nan
    else:
        if t_test.ndim < 2:
            t_test = np.expand_dims(t_test, 1)  # make 2-D
        test_sort_ind = np.argsort(t_test[:, 0], axis=0)
        t_test = t_test[test_sort_ind, ...]
        if y_test is not None:
            y_test = y_test[test_sort_ind, ...].reshape((-1,) + y.shape[1:])
        if r_test is not None:
            r_test = r_test[test_sort_ind, ...]
        else:
            r_test = np.nan * t_test
    if not (t_test.shape[1] == t_train.shape[1]):
        t_test = np.concatenate([t_test[:, 0][:, None],
                                 np.nan * np.empty([t_test.shape[0], t_train.shape[1]-1])], axis=1)
    # here we use non-JAX numpy to sort out indexing of these static arrays
    t_train_test = np.concatenate([t_train, t_test])
    keep_ind = ~np.isnan(t_train_test[:, 0])
    t_train_test = t_train_test[keep_ind, ...]
    if r_test.shape[1] != r_train.shape[1]:  # do spatial test points have different dimensionality to training points?
        r_test_nan = np.nan * np.zeros([r_test.shape[0], r_train.shape[1]])
    else:
        r_test_nan = r_test
    r_train_test = np.concatenate([r_train, r_test_nan])
    r_train_test = r_train_test[keep_ind, ...]
    t_ind = np.argsort(t_train_test[:, 0])
    t_all = t_train_test[t_ind]
    r_all = r_train_test[t_ind]
    reverse_ind = np.argsort(t_ind)
    n_train = t_train.shape[0]
    train_id = reverse_ind[:n_train]  # index the training locations
    test_id = reverse_ind[n_train:]  # index the test locations
    y_all = np.nan * np.zeros([t_all.shape[0], y_train.shape[1]])  # observation vector with nans at test locations
    # y_all[reverse_ind[:n_train], ...] = y_train  # and the data at the train locations
    y_all = index_update(y_all, index[reverse_ind[:n_train]], y_train)  # and the data at the train locations
    if y_test is not None:
        # y_all[reverse_ind[n_train:], ...] = y_test  # and the data at the train locations
        y_all = index_update(y_all, index[reverse_ind[n_train:]], y_test)  # and the data at the train locations
    mask = np.ones_like(y_all, dtype=bool)
    # mask[train_id] = False
    mask = index_update(mask, index[train_id], False)
    dt_all = np.concatenate([np.array([0.0]), np.diff(t_all[:, 0])])
    return (np.array(t_all, dtype=np.float64), np.array(y_all, dtype=np.float64), np.array(r_all, dtype=np.float64),
            np.array(r_test, dtype=np.float64),
            np.array(dt_all, dtype=np.float64),
            np.array(train_id, dtype=np.int64), np.array(test_id, dtype=np.int64), np.array(mask, dtype=bool))


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


def sample_gaussian_noise(latent_mean, likelihood_cov):
    lik_std, _ = cho_factor(likelihood_cov)
    gaussian_sample = latent_mean + lik_std * random.normal(random.PRNGKey(123), shape=latent_mean.shape)
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


def discretegrid(xy, w, nt):
    """
    Convert spatial observations to a discrete intensity grid
    :param xy: observed spatial locations as a two-column vector
    :param w: observation window, i.e. discrete grid to be mapped to, [xmin xmax ymin ymax]
    :param nt: two-element vector defining number of bins in both directions
    """
    # Make grid
    x = nnp.linspace(w[0], w[1], nt[0] + 1)
    y = nnp.linspace(w[2], w[3], nt[1] + 1)
    X, Y = nnp.meshgrid(x, y)

    # Count points
    N = nnp.zeros([nt[1], nt[0]])
    for i in range(nt[0]):
        for j in range(nt[1]):
            ind = (xy[:, 0] >= x[i]) & (xy[:, 0] < x[i + 1]) & (xy[:, 1] >= y[j]) & (xy[:, 1] < y[j + 1])
            N[j, i] = nnp.sum(ind)
    return X[:-1, :-1].T, Y[:-1, :-1].T, N.T


def plot(model, it_num, ax=None):
    post_mean, post_var, _, nlpd = model.predict()
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    lb = post_mean[:, 0] - 1.96 * post_var[:, 0] ** 0.5
    ub = post_mean[:, 0] + 1.96 * post_var[:, 0] ** 0.5
    ax.plot(model.t_train, model.y, 'k.', label='training observations')
    plt.plot(model.t_all[model.test_id], model.y_all[model.test_id], 'r.', alpha=0.4, label='test observations')
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

    mu, var, _, nlpd_test, _, _ = m.predict(return_full=True)
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
    mu, var, _, nlpd_test, mu_filt, var_filt = m.predict_2d(return_full=True)
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


def mvhermgauss(H: int, D: int):
    """
    This function is taken from GPflow: https://github.com/GPflow/GPflow
    Copied here rather than imported so that users don't need to install gpflow to use kalman-jax

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

    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


def gauss_hermite(dim=1, num_quad_pts=20):
    """
    Return weights and sigma-points for Gauss-Hermite cubature
    """
    # sigma_pts, weights = hermgauss(num_quad_pts)  # Gauss-Hermite sigma points and weights
    sigma_pts, weights = mvhermgauss(num_quad_pts, dim)
    sigma_pts = np.sqrt(2) * sigma_pts.T
    weights = weights.T * pi ** (-0.5 * dim)  # scale weights by 1/√π
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
    elif dim == 6:
        weights = np.array([0.6667, -0.1111, -0.1111, -0.1111, -0.1111, -0.1111, -0.1111, -0.1111, -0.1111, -0.1111,
                            -0.1111, -0.1111, -0.1111, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278,
                            0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278,
                            0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278,
                            0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278,
                            0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278,
                            0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278, 0.0278])
        sigma_pts = np.block([[
            0.,      1.7321, -1.7321, 0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,     0.,
            0.,      1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321,
            -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 0.,      0.,     0.,
            0.,      0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,     0.,
            0.,      0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,     0.,
            0.,      0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,     0.,
            0.],
            [0.,     0.,      0.,     1.7321, -1.7321, 0.,      0.,     0.,     0.,      0.,      0.,     0.,
             0.,     1.7321, -1.7321, -1.7321, 1.7321, 0.,      0.,     0.,     0.,      0.,      0.,     0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1.7321, -1.7321, 1.7321,
             -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321,
             -1.7321, 0.,      0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0.],
            [0.,     0.,      0.,  0.,      0.,     1.7321, -1.7321,     0.,     0.,      0.,      0.,     0.,
             0., 0., 0., 0., 0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1.7321, -1.7321, 1.7321,
             1.7321, 0.,      0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,
             0., 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 1.7321,
             -1.7321, 0.,      0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,
             0.],
            [0.,     0.,      0.,  0.,      0.,     0.,     0.,     1.7321, -1.7321,      0.,      0.,     0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1.7321, -1.7321, 1.7321,
             1.7321, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0., 0., 0., 0., 0.,
             0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0., 0., 0., 0., 0.,
             0., 1.7321, -1.7321, -1.7321, 1.7321, 1.7321, -1.7321, 1.7321, -1.7321, 0., 0., 0.,
             0.],
            [0.,     0.,      0.,  0.,      0.,     0.,     0.,      0.,      0.,   1.7321, -1.7321,  0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0.,
             0., 0., 0., 0., 0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0.,
             0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0., 0., 1.7321, -1.7321, 1.7321,
             -1.7321],
            [0.,      0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,     1.7321,
             -1.7321, 0.,      0.,      0.,     0.,      0.,     0.,      0.,     0.,     0.,      0.,      0.,
             0., 0., 0., 0., 0., 1.7321, -1.7321, -1.7321, 1.7321, 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1.7321, -1.7321, -1.7321,
             1.7321, 0., 0., 0., 0., 0., 0., 0., 0., 1.7321, -1.7321, -1.7321,
             1.7321, 0., 0., 0., 0., 1.7321, -1.7321, -1.7321, 1.7321, 1.7321, -1.7321, -1.7321,
             1.7321]
            ])

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

def scaled_squared_euclid_dist(X, X2, ell):
    """
    Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.

    Adapted from GPflow: https://github.com/GPflow/GPflow
    http://www.apache.org/licenses/LICENSE-2.0

    """
    return square_distance(X / ell, X2 / ell)


def square_distance(X, X2):
    """

    Adapted from GPflow: https://github.com/GPflow/GPflow
    http://www.apache.org/licenses/LICENSE-2.0

    Returns ||X - X2ᵀ||²
    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.

    This function can deal with leading dimensions in X and X2.
    In the sample case, where X and X2 are both 2 dimensional,
    for example, X is [N, D] and X2 is [M, D], then a tensor of shape
    [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D]
    then the output will be [N1, S1, N2, S2].
    """
    Xs = np.sum(np.square(X), axis=-1)
    X2s = np.sum(np.square(X2), axis=-1)
    dist = -2 * np.tensordot(X, X2, [[-1], [-1]])
    dist += broadcasting_elementwise(np.add, Xs, X2s)
    return dist


def broadcasting_elementwise(op, a, b):
    """

    Adapted from GPflow: https://github.com/GPflow/GPflow
    http://www.apache.org/licenses/LICENSE-2.0

    Apply binary operation `op` to every pair in tensors `a` and `b`.

    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(np.reshape(a, [-1, 1]), np.reshape(b, [1, -1]))
    return flatres.reshape(a.shape[0], b.shape[0])
