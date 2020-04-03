import jax.numpy as jnp
from jax.ops import index, index_update
from jax.scipy.linalg import cho_factor, cho_solve
from jax.experimental import loops
from jax import value_and_grad
from jax import random
from jax.nn import softplus
from jax import jit, partial
import numpy as np
from likelihoods import Gaussian
pi = 3.141592653589793


class SDEGP(object):
    """
    The stochastic differential equation (SDE) form of a Gaussian process (GP) model
    """
    def __init__(self, prior, likelihood, x, y, x_test=None):
        # TODO: implement EP
        # TODO: implement lookup table for A
        x, ind = np.unique(x, return_index=True)
        if y.ndim < 2:
            y = np.expand_dims(y, 1)  # make 2D
        y = y[ind, :]
        self.t_train = x
        self.y = jnp.array(y)
        if x_test is None:
            t_test = []
        else:
            t_test = np.unique(np.squeeze(x_test))  # test inputs
        (self.t_all, self.test_id, self.train_id,
         self.y_all, self.mask, self.dt, self.dt_all) = self.input_admin(self.t_train, t_test, self.y)
        self.t_test = jnp.array(t_test)
        self.nlml = 0.0
        self.prior = prior
        self.likelihood = likelihood
        # construct the state space model:
        print('building SDE-GP with', self.prior.name, 'prior and', self.likelihood.name, 'likelihood ...')
        self.F, self.L, self.Qc, self.H, self.Pinf = self.prior.cf_to_ss()
        self.latent_size = self.F.shape[0]
        self.observation_size = self.y.shape[1]
        self.minf = jnp.zeros([self.latent_size, 1])  # stationary state mean
        self.filtered_mean, self.filtered_cov, self.site_mean, self.site_var = [], [], [], []

    @staticmethod
    def input_admin(t_train, t_test, y):
        """
        order the inputs, remove duplicates, and index the train and test input locations
        """
        # here we use non-JAX numpy to sort out indexing of these static arrays
        t, x_ind = np.unique(np.concatenate([t_test, t_train]), return_inverse=True)
        n_test = t_test.shape[0]  # number of test locations
        test_id = x_ind[:n_test]  # index the test locations
        train_id = x_ind[n_test:]  # index the training locations
        y_all = np.nan * np.zeros([t.shape[0], y.shape[1]])  # observation vector with nans at test locations
        y_all[x_ind[n_test:], :] = y  # and the data at the train locations
        mask = np.ones(y_all.shape[0], dtype=bool)
        mask[train_id] = False
        dt = np.concatenate([jnp.array([0.0]), np.diff(t_train)])
        dt_all = np.concatenate([jnp.array([0.0]), np.diff(t)])
        return (jnp.array(t), jnp.array(test_id), jnp.array(train_id), jnp.array(y_all),
                jnp.array(mask), jnp.array(dt), jnp.array(dt_all))

    def neg_log_marg_lik(self, params=None):
        """
        calculate the negative log-marginal likelihood by running the Kalman filter across training locations
        """
        if params is None:
            params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        neg_log_marg_lik, dlZ = value_and_grad(self.kalman_filter, argnums=2)(self.y, self.dt, params,
                                                                              None, None, False)
        return neg_log_marg_lik, dlZ

    def predict(self, y=None, dt=None, mask=None, site_params=None):
        """
        calculate posterior predictive distribution via filtering and smoothing across training & test locations
        """
        y = self.y_all if y is None else y
        dt = self.dt_all if dt is None else dt
        mask = self.mask if mask is None else mask
        params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        # self.update_model(params[0])
        filter_mean, filter_cov, site_mean, site_var = self.kalman_filter(y, dt, params, mask, site_params, True)
        posterior_mean, posterior_var = self.rauch_tung_striebel_smoother(filter_mean, filter_cov, dt, params)
        return posterior_mean, posterior_var, site_mean, site_var

    def update_model(self, theta_prior):
        """
        re-construct the SDE-GP model with latest parameters
        """
        self.F, self.L, self.Qc, self.H, self.Pinf = self.prior.cf_to_ss(theta_prior)

    @partial(jit, static_argnums=(0, 6))  # make jit work with self
    def kalman_filter(self, y, dt, params, mask=None, site_params=None, store=False):
        """
        run the Kalman filter to get p(f_k | y_{1:k})
        """
        theta_prior, theta_lik = softplus(params[0]), softplus(params[1])
        self.update_model(theta_prior)
        if mask is not None:
            mask = mask[..., jnp.newaxis, jnp.newaxis]  # align mask.shape with y.shape
        N = dt.shape[0]
        scalar_obs = self.H.shape[-2] == 1
        with loops.Scope() as s:
            s.neg_log_marg_lik = 0.0  # negative log-marginal likelihood
            s.m, s.P = self.minf, self.Pinf
            if store:
                s.filtered_mean = jnp.zeros([N, self.latent_size, 1])
                s.filtered_cov = jnp.zeros([N, self.latent_size, self.latent_size])
            if site_params is not None:
                s.site_mean, s.site_var = site_params
            else:
                if store:
                    s.site_mean = jnp.zeros([N, self.observation_size])
                    s.site_var = jnp.zeros([N, self.observation_size])
            for k in s.range(N):
                y_k = y[k]
                # -- KALMAN PREDICT --
                #  m_{k|k-1} = A_k m_{k-1}
                #  P_{k|k-1} = A_k P_{k-1} A_k' + Q_k, where Q_k = Pinf - A_k Pinf A_k'
                # A = tf.linalg.expm(self.F * dt[k])  # this is naive but dynamic step size checking is also expensive
                A = self.prior.expm(dt[k], theta_prior)
                m_ = A @ s.m
                P_ = A @ (s.P - self.Pinf) @ A.T + self.Pinf
                # --- KALMAN UPDATE ---
                # Given previous predicted mean m_{k|k-1} and cov P_{k|k-1}, incorporate y_k to get filtered mean m_k &
                # cov P_k and compute the marginal likelihood p(y[k] | y[:k-1])
                mu = self.H @ m_
                var = self.H @ P_ @ self.H.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    y_k = jnp.where(mask[k], mu, y_k)  # fill in masked obs with prior expectation to prevent NaN grads
                if site_params is None:
                    # likelihood-specific moment matching function:
                    log_marg_lik_k, d1, d2 = self.likelihood.moment_match(y=y_k, m=mu, v=var, hyp=theta_lik)
                else:
                    # use supplied site variance (for the smoothing operation in posterior sampling)
                    log_marg_lik_k, d1, d2 = Gaussian.moment_match(y_k, mu, var, hyp=s.site_var[k])
                m_site = mu - d1 / d2  # approximate likelihood (site) mean (see Rasmussen & Williams p75)
                P_site = -var - 1 / d2  # approximate likelihood (site) variance
                # slightly modified Kalman update (see Nickish et. al. ICML 2018 or Wilkinson et. al. ICML 2019):
                S = var + P_site
                if scalar_obs:  # if S is scalar
                    K = P_ @ self.H.T / S
                else:
                    L, low = cho_factor(S)
                    K = (cho_solve((L, low), self.H @ P_)).T
                s.m = m_ + K @ (m_site - mu)
                s.P = P_ - K @ S @ K.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    s.m = jnp.where(mask[k], m_, s.m)
                    s.P = jnp.where(mask[k], P_, s.P)
                    log_marg_lik_k = jnp.where(mask[k][..., 0, 0], jnp.zeros_like(log_marg_lik_k), log_marg_lik_k)
                s.neg_log_marg_lik -= jnp.sum(log_marg_lik_k)
                if store:
                    s.filtered_mean = index_update(s.filtered_mean, index[k, ...], s.m)
                    s.filtered_cov = index_update(s.filtered_cov, index[k, ...], s.P)
                    s.site_mean = index_update(s.site_mean, index[k, ...], jnp.squeeze(m_site.T))
                    s.site_var = index_update(s.site_var, index[k, ...], jnp.squeeze(P_site.T))
        if store:
            # self.filtered_mean = s.filtered_mean
            # self.filtered_cov = s.filtered_cov
            # self.site_mean = s.site_mean
            # self.site_var = s.site_var
            return s.filtered_mean, s.filtered_cov, s.site_mean, s.site_var
        return s.neg_log_marg_lik

    @partial(jit, static_argnums=0)  # make jit work with self
    def rauch_tung_striebel_smoother(self, m_filtered, P_filtered, dt, params):
        """
        run the RTS smoother to get p(f_k | y_{1:N})
        """
        theta_prior, theta_lik = softplus(params[0]), softplus(params[1])
        self.update_model(theta_prior)
        N = dt.shape[0]
        dt = jnp.concatenate([dt[1:], jnp.array([0.0])], axis=0)
        with loops.Scope() as s:
            s.m, s.P = m_filtered[-1, ...], P_filtered[-1, ...]
            s.smoothed_mean = jnp.zeros([N, self.observation_size])
            s.smoothed_var = jnp.zeros([N, self.observation_size])
            for k in s.range(N-1, -1, -1):
                A = self.prior.expm(dt[k], self.prior.hyp)  # closed form integration of transition matrix
                m_predicted = A @ m_filtered[k, ...]
                tmp_gain_cov = A @ P_filtered[k, ...]
                P_predicted = A @ (P_filtered[k, ...] - self.Pinf) @ A.T + self.Pinf
                # backward Kalman gain:
                # G = F * A' * P^{-1}
                # since both F(iltered) and P(redictive) are cov matrices, thus self-adjoint, we can take the transpose:
                #   = (P^{-1} * A * F)'
                P_predicted_chol, low = cho_factor(P_predicted)
                G_transpose = cho_solve((P_predicted_chol, low), tmp_gain_cov)
                s.m = m_filtered[k, ...] + G_transpose.T @ (s.m - m_predicted)
                s.P = P_filtered[k, ...] + G_transpose.T @ (s.P - P_predicted) @ G_transpose
                s.smoothed_mean = index_update(s.smoothed_mean, index[k, ...], jnp.squeeze((self.H @ s.m).T))
                s.smoothed_var = index_update(s.smoothed_var, index[k, ...],
                                              jnp.squeeze(jnp.diag(self.H @ s.P @ self.H.T)))
        return s.smoothed_mean, s.smoothed_var

    def prior_sample(self, num_samps, x=None):
        """
        sample from the prior
        """
        self.update_model(softplus(self.prior.hyp))
        # TODO: sort out prior sampling - currently very unstable
        if x is None:
            dt = jnp.concatenate([jnp.array([0.0]), jnp.diff(self.t_all)])
        else:
            dt = jnp.concatenate([jnp.array([0.0]), jnp.diff(x)])
        N = dt.shape[0]
        with loops.Scope() as s:
            s.f_sample = jnp.zeros([N, self.observation_size, num_samps])
            s.m = jnp.linalg.cholesky(self.Pinf) @ random.normal(random.PRNGKey(99), shape=[self.latent_size, 1])
            for i in s.range(num_samps):
                s.m = jnp.linalg.cholesky(self.Pinf) @ random.normal(random.PRNGKey(i), shape=[self.latent_size, 1])
                for k in s.range(N):
                    A = self.prior.expm(dt[k], self.prior.hyp)  # transition and noise process matrices
                    Q = self.Pinf - A @ self.Pinf @ A.T
                    C = jnp.linalg.cholesky(Q + 2e-6 * jnp.eye(self.latent_size))  # <--- unstable
                    # we need to provide a different PRNG seed every time:
                    s.m = A @ s.m + C @ random.normal(random.PRNGKey(i*k+k), shape=[self.latent_size, 1])
                    f = (self.H @ s.m).T
                    s.f_sample = index_update(s.f_sample, index[k, ..., i], jnp.squeeze(f))
        return s.f_sample

    def posterior_sample(self, num_samps):
        """
        sample from the posterior at the test locations
        """
        post_mean, _, site_mean, site_var = self.predict()
        prior_samp = self.prior_sample(num_samps, x=self.t_all)
        prior_samp_y = self.likelihood.sample_noise(prior_samp, site_var)
        with loops.Scope() as ss:
            ss.smoothed_sample = jnp.zeros(prior_samp_y.shape)
            for i in ss.range(num_samps):
                smoothed_sample_i, _, _, _ = self.predict(prior_samp_y[..., i], self.dt_all, self.mask,
                                                          (site_mean, site_var))
                ss.smoothed_sample = index_update(ss.smoothed_sample, index[..., i], smoothed_sample_i)
        return prior_samp - ss.smoothed_sample + post_mean[..., jnp.newaxis]
