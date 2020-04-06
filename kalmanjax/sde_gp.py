import jax.numpy as jnp
from jax.ops import index, index_update
from jax.scipy.linalg import cho_factor, cho_solve
from jax.experimental import loops
from jax import value_and_grad, jit, partial, random
from jax.nn import softplus
import numpy as np
from likelihoods import Gaussian
pi = 3.141592653589793


class SDEGP(object):
    """
    The stochastic differential equation (SDE) form of a Gaussian process (GP) model
    """
    def __init__(self, prior, likelihood, x, y, x_test=None):
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
        self.prior = prior
        self.likelihood = likelihood
        # construct the state space model:
        print('building SDE-GP with', self.prior.name, 'prior and', self.likelihood.name, 'likelihood ...')
        self.F, self.L, self.Qc, self.H, self.Pinf = self.prior.cf_to_ss()
        self.state_dim = self.F.shape[0]
        self.obs_dim = self.y.shape[1]
        self.minf = jnp.zeros([self.state_dim, 1])  # stationary state mean
        self.site_params = None

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

    def predict(self, y=None, dt=None, mask=None, site_params=None, sampling=False):
        """
        calculate posterior predictive distribution by filtering and smoothing across the training & test locations
        """
        y = self.y_all if y is None else y
        dt = self.dt_all if dt is None else dt
        mask = self.mask if mask is None else mask
        params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        # construct a vector of site parameters that is the full size of the test data
        if site_params is not None and not sampling:
            # test site parameters are 𝓝(0,∞), and will not be used
            site_mean, site_var = jnp.zeros(dt.shape[0]), 1e5*jnp.ones(dt.shape[0])
            # replace parameters at training locations with the supplied sites
            site_mean = index_update(site_mean, index[self.train_id], site_params[0])
            site_var = index_update(site_var, index[self.train_id], site_params[1])
            site_params = (site_mean, site_var)
        filter_mean, filter_cov, site_params_kf = self.kalman_filter(y, dt, params, True, sampling, mask, site_params)
        if site_params is None:
            site_params = site_params_kf
        posterior_mean, posterior_var, _ = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, dt, y)
        return posterior_mean, posterior_var, site_params

    def neg_log_marg_lik(self, params=None):
        """
        a single assumed density filtering step - to be fed to a gradient-based optimiser.
         - calculates the negative log-marginal likelihood and its gradients by running
           the Kalman filter across training locations
        """
        if params is None:
            params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        neg_log_marg_lik, dlZ = value_and_grad(self.kalman_filter, argnums=2)(self.y, self.dt, params, False, False)
        return neg_log_marg_lik, dlZ

    def expectation_propagation(self):
        """
        a single expectation propagation step - to be fed to a gradient-based optimiser.
         - we first update the site parameters (site mean and variance)
         - then compute the marginal lilelihood and its gradient w.r.t. the hyperparameters
        """
        # fetch the model parameters from the prior and the likelihood
        params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        # run the forward filter to calculate the filtering distribution.
        # on the first pass (when self.site_params = None) this initialises the sites too
        filter_mean, filter_cov, self.site_params = self.kalman_filter(self.y, self.dt, params,
                                                                       True, False, None, self.site_params)
        # run the smoother and update the EP sites
        post_mean, post_var, self.site_params = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov,
                                                                                  self.dt, self.y, self.site_params)
        # compute the negative log-marginal likelihood and its gradient in order to update the hyperparameters
        neg_log_marg_lik, dlZ = value_and_grad(self.kalman_filter, argnums=2)(self.y, self.dt, params,
                                                                              False, False, None, self.site_params)
        return neg_log_marg_lik, dlZ

    def update_model(self, theta_prior=None):
        """
        re-construct the SDE-GP model with latest parameters
        """
        self.F, self.L, self.Qc, self.H, self.Pinf = self.prior.cf_to_ss(hyperparams=theta_prior)

    @partial(jit, static_argnums=(0, 4, 5))
    def kalman_filter(self, y, dt, params, store=False, sampling=False, mask=None, site_params=None):
        """
        run the Kalman filter to get p(fₖ|y₁,...,yₖ)
        """
        theta_prior, theta_lik = softplus(params[0]), softplus(params[1])
        self.update_model(theta_prior)  # all model components that are not static must be reset inside the function
        if mask is not None:
            mask = mask[..., jnp.newaxis, jnp.newaxis]  # align mask.shape with y.shape
        N = dt.shape[0]
        with loops.Scope() as s:
            s.neg_log_marg_lik = 0.0  # negative log-marginal likelihood
            s.m, s.P = self.minf, self.Pinf
            if store:
                s.filtered_mean = jnp.zeros([N, self.state_dim, 1])
                s.filtered_cov = jnp.zeros([N, self.state_dim, self.state_dim])
            if site_params is not None:
                s.site_mean, s.site_var = site_params
            else:
                if store:
                    s.site_mean = jnp.zeros([N, self.obs_dim])
                    s.site_var = jnp.zeros([N, self.obs_dim])
            for k in s.range(N):
                y_k = y[k]
                # -- KALMAN PREDICT --
                #  mₖ⁻ = Aₖ mₖ₋₁
                #  Pₖ⁻ = Aₖ Pₖ₋₁ Aₖ' + Qₖ, where Qₖ = Pinf - Aₖ Pinf Aₖ'
                A = self.prior.expm(dt[k], theta_prior)
                m_ = A @ s.m
                P_ = A @ (s.P - self.Pinf) @ A.T + self.Pinf
                # --- KALMAN UPDATE ---
                # Given previous predicted mean mₖ⁻ and cov Pₖ⁻, incorporate yₖ to get filtered mean mₖ &
                # cov Pₖ and compute the marginal likelihood p(yₖ|y₁,...,yₖ₋₁)
                mu = self.H @ m_
                var = self.H @ P_ @ self.H.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    y_k = jnp.where(mask[k], mu, y_k)  # fill in masked obs with prior expectation to prevent NaN grads
                if sampling:
                    log_marg_lik_k, d1, d2 = Gaussian.moment_match([], y_k, mu, var, s.site_var[k], True)
                    m_site = mu - d1 / d2  # approximate likelihood (site) mean (see Rasmussen & Williams p75)
                    P_site = -var - 1 / d2  # approximate likelihood (site) variance
                else:
                    if site_params is None:
                        # likelihood-specific moment matching function:
                        log_marg_lik_k, d1, d2 = self.likelihood.moment_match(y_k, mu, var, theta_lik, True)
                        m_site = mu - d1 / d2  # approximate likelihood (site) mean (see Rasmussen & Williams p75)
                        P_site = -var - 1 / d2  # approximate likelihood (site) variance
                    else:
                        # use supplied site variance (for the smoothing operation in posterior sampling)
                        # log_marg_lik_k, d1, d2 = Gaussian.moment_match([], y_k, mu, var, s.site_var[k], True)
                        # _, d1, d2 = Gaussian.moment_match([], s.site_mean[k], mu, var, s.site_var[k], True)
                        log_marg_lik_k = self.likelihood.moment_match(y_k, mu, var, theta_lik, False)
                        m_site = s.site_mean[k]  # use supplied site parameters
                        P_site = s.site_var[k]
                # modified Kalman update (see Nickish et. al. ICML 2018 or Wilkinson et. al. ICML 2019):
                S = var + P_site
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
            return s.filtered_mean, s.filtered_cov, (s.site_mean, s.site_var)
        return s.neg_log_marg_lik

    @partial(jit, static_argnums=0)
    def rauch_tung_striebel_smoother(self, params, m_filtered, P_filtered, dt, y, site_params=None):
        """
        run the RTS smoother to get p(fₖ|y₁,...,yₙ)
        """
        theta_prior, theta_lik = softplus(params[0]), softplus(params[1])
        self.update_model(theta_prior)  # all model components that are not static must be reset inside the function
        N = dt.shape[0]
        dt = jnp.concatenate([dt[1:], jnp.array([0.0])], axis=0)
        with loops.Scope() as s:
            s.m, s.P = m_filtered[-1, ...], P_filtered[-1, ...]
            s.smoothed_mean = jnp.zeros([N, self.obs_dim])
            s.smoothed_var = jnp.zeros([N, self.obs_dim])
            if site_params is not None:
                s.site_mean, s.site_var = site_params
            for k in s.range(N-1, -1, -1):
                A = self.prior.expm(dt[k], theta_prior)  # closed form integration of transition matrix
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
                if site_params is not None:
                    # extract mean and var from state (we discard cross-covariance for now):
                    mu, var = jnp.squeeze(self.H @ s.m), jnp.squeeze(jnp.diag(self.H @ s.P @ self.H.T))
                    # remove local likelihood approximation to obtain the marginal cavity distribution:
                    var_cav = 1.0 / (1.0 / var - 1.0 / s.site_var[k])  # cavity variance
                    mu_cav = var_cav * (mu / var - s.site_mean[k] / s.site_var[k])  # cavity mean
                    # calculate the log-normaliser of the tilted distribution and its derivatives w.r.t. mu_cav (d1, d2)
                    # lZ = log E_{N(f|m,P)} [p(y|f)] = log int p(y|f) N(f|m,P) df
                    _, d1, d2 = self.likelihood.moment_match(y[k], mu_cav, var_cav, theta_lik, True)
                    m_site = mu_cav - d1 / d2  # approximate likelihood (site) mean (see Rasmussen & Williams p75)
                    P_site = -var_cav - 1 / d2  # approximate likelihood (site) variance
                    s.site_mean = index_update(s.site_mean, index[k, ...], jnp.squeeze(m_site.T))
                    s.site_var = index_update(s.site_var, index[k, ...], jnp.squeeze(P_site.T))
        if site_params is not None:
            site_params = (s.site_mean, s.site_var)
        return s.smoothed_mean, s.smoothed_var, site_params

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
            s.f_sample = jnp.zeros([N, self.obs_dim, num_samps])
            s.m = jnp.linalg.cholesky(self.Pinf) @ random.normal(random.PRNGKey(99), shape=[self.state_dim, 1])
            for i in s.range(num_samps):
                s.m = jnp.linalg.cholesky(self.Pinf) @ random.normal(random.PRNGKey(i), shape=[self.state_dim, 1])
                for k in s.range(N):
                    A = self.prior.expm(dt[k], self.prior.hyp)  # transition and noise process matrices
                    Q = self.Pinf - A @ self.Pinf @ A.T
                    C = jnp.linalg.cholesky(Q + 1e-5 * jnp.eye(self.state_dim))  # <--- unstable
                    # we need to provide a different PRNG seed every time:
                    s.m = A @ s.m + C @ random.normal(random.PRNGKey(i*k+k), shape=[self.state_dim, 1])
                    f = (self.H @ s.m).T
                    s.f_sample = index_update(s.f_sample, index[k, ..., i], jnp.squeeze(f))
        return s.f_sample

    def posterior_sample(self, num_samps):
        """
        sample from the posterior at the test locations
        """
        post_mean, _, (site_mean, site_var) = self.predict()
        prior_samp = self.prior_sample(num_samps, x=self.t_all)
        prior_samp_y = self.likelihood.sample_noise(prior_samp, site_var)
        with loops.Scope() as ss:
            ss.smoothed_sample = jnp.zeros(prior_samp_y.shape)
            for i in ss.range(num_samps):
                smoothed_sample_i, _, _ = self.predict(prior_samp_y[..., i], self.dt_all, self.mask,
                                                       (site_mean, site_var), sampling=True)
                ss.smoothed_sample = index_update(ss.smoothed_sample, index[..., i], smoothed_sample_i)
        return prior_samp - ss.smoothed_sample + post_mean[..., jnp.newaxis]
