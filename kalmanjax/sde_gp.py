import jax.numpy as jnp
from jax.ops import index, index_update
from jax.scipy.linalg import cho_factor, cho_solve
from jax.experimental import loops
from jax import value_and_grad, jit, partial, random
from jax.nn import softplus
from utils import gaussian_moment_match
import numpy as np
from approximate_inference import EP
pi = 3.141592653589793


class SDEGP(object):
    """
    The stochastic differential equation (SDE) form of a Gaussian process (GP) model.
    Implements methods for inference and learning in models with GP priors of the form
        f(t) ~ GP(0,k(t,t'))
    using state space methods, i.e. Kalman filtering and smoothing.
    Constructs a linear time-invariant (LTI) stochastic differential equation (SDE) of the following form:
        dx(t)/dt = F x(t) + L w(t)
              yₙ ~ p(yₙ | f(tₙ)=H x(tₙ))
    where w(t) is a white noise process and where the state x(t) is Gaussian distributed with initial
    state distribution x(t)~𝓝(0,Pinf).
    Currently implemented inference methods:
        - Assumed density filtering (ADF, single sweep EP)
        - Power expectation propagation (PEP)
    """
    def __init__(self, prior, likelihood, x, y, x_test=None, approx_inf=None):
        """
        :param prior: the model prior p(f|0,k(t,t')) object which constructs the required state space model matrices
        :param likelihood: the likelihood model object which performs moment matching and evaluates p(y|f)
        :param x: training inputs
        :param y: training data / observations
        :param x_test: test inputs
        :param approx_inf: the approximate inference algorithm for computing the sites (EP, GHKS, PL, ...)
        """
        assert x.shape[0] == y.shape[0]
        x, ind = np.unique(x, return_index=True)
        if y.ndim < 2:
            y = np.expand_dims(y, 1)  # make 2-D
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
        self.sites = EP() if approx_inf is None else approx_inf

    @staticmethod
    def input_admin(t_train, t_test, y):
        """
        Order the inputs, remove duplicates, and index the train and test input locations.
        :param t_train: training inputs [N, 1]
        :param t_test: testing inputs [N*, 1]
        :param y: observations at the training inputs [N, 1]
        :return:
            t: the combined and sorted training and test inputs [N + N*, 1]
            test_id: an array of indices corresponding to the test inputs [N*, 1]
            train_id: an array of indices corresponding to the training inputs [N, 1]
            y_all: an array observations y augmented with nans at test locations [N + N*, 1]
            mask: boolean array to signify training locations [N + N*, 1]
            dt: training step sizes, Δtₙ = tₙ - tₙ₋₁ [N, 1]
            dt_all: combined training and test step sizes, Δtₙ = tₙ - tₙ₋₁ [N + N*, 1]
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
        Calculate posterior predictive distribution p(f*|f,y) by filtering and smoothing across the
        training & test locations.
        This function is also used during posterior sampling to smooth the auxillary data sampled from the prior.
        :param y: observations (nans at test locations) [M, 1]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [M, 1]
        :param mask: a boolean array signifying which elements are observed and which are nan [M, 1]
        :param site_params: the sites computed during a previous inference proceedure [2, M, obs_dim]
        :param sampling: notify whether we are running posterior sampling
        :return:
            posterior_mean: the posterior predictive mean [M, obs_dim]
            posterior_var: the posterior predictive variance [M, obs_dim]
            site_params: the site parameters. If none are provided then new sites are computed [2, M, obs_dim]
        """
        y = self.y_all if y is None else y
        dt = self.dt_all if dt is None else dt
        mask = self.mask if mask is None else mask
        params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        site_params = self.site_params if site_params is None else site_params
        if site_params is not None and not sampling:
            # construct a vector of site parameters that is the full size of the test data
            # test site parameters are 𝓝(0,∞), and will not be used
            site_mean, site_var = jnp.zeros([dt.shape[0], 1]), 1e5*jnp.ones([dt.shape[0], 1])
            # replace parameters at training locations with the supplied sites
            site_mean = index_update(site_mean, index[self.train_id], site_params[0])
            site_var = index_update(site_var, index[self.train_id], site_params[1])
            site_params = (site_mean, site_var)
        filter_mean, filter_cov, site_params = self.kalman_filter(y, dt, params, True, sampling, mask, site_params)
        posterior_mean, posterior_var, _ = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, dt)
        return posterior_mean, posterior_var, site_params

    def neg_log_marg_lik(self, params=None):
        """
        Calculates the negative log-marginal likelihood and its gradients by running
        the Kalman filter across training locations.
        :param params: the model parameters. If not supplied then defaults to the model's
                       assigned parameters [num_params]
        :return:
            neg_log_marg_lik: the negative log-marginal likelihood -log p(y), i.e. the energy [scalar]
            dlZ: the derivative of the energy w.r.t. the model parameters [num_params]
        """
        if params is None:
            # fetch the model parameters from the prior and the likelihood
            params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        neg_log_marg_lik, dlZ = value_and_grad(self.kalman_filter, argnums=2)(self.y, self.dt, params, False, False)
        return neg_log_marg_lik, dlZ

    def run_model(self, params=None):
        """
        A single parameter update step - to be fed to a gradient-based optimiser.
         - we first update the site parameters (site mean and variance)
         - then compute the marginal lilelihood and its gradient w.r.t. the hyperparameters
        :param params: the model parameters. If not supplied then defaults to the model's
                       assigned parameters [num_params]
        :return:
            neg_log_marg_lik: the negative log-marginal likelihood -log p(y), i.e. the energy [scalar]
            dlZ: the derivative of the energy w.r.t. the model parameters [num_params]
        """
        if params is None:
            # fetch the model parameters from the prior and the likelihood
            params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        # run the forward filter to calculate the filtering distribution
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
        Re-construct the SDE-GP model with latest parameters.
        :param theta_prior: the hyperparameters of the GP prior
        return:
            computes the model matrices F, L, Qc, H, Pinf. See the prior class for details
        """
        self.F, self.L, self.Qc, self.H, self.Pinf = self.prior.cf_to_ss(hyperparams=theta_prior)

    @partial(jit, static_argnums=(0, 4, 5))
    def kalman_filter(self, y, dt, params, store=False, sampling=False, mask=None, site_params=None):
        """
        Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
        The Kalman update step invloves some control flow to work out whether we are
            i) initialising the EP sites / running ADF
            ii) using supplied sites (e.g. in EP)
            iii) running the smoothing operation in posterior sampling
        If store is True then we compute and return the intermediate filtering distributions
        p(fₙ|y₁,...,yₖ) and sites sₙ(fₙ), otherwise we do not store the intermediates and simply
        return the energy / negative log-marginal likelihood, -log p(y).
        :param y: observed data [N, obs_dim]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [N, 1]
        :param params: the model parameters, i.e the hyperparameters of the prior & likelihood
        :param store: flag to notify whether to store the intermediates
        :param sampling: flag to notify whether we are running the posterior sampling operation
        :param mask: boolean array signifying which elements of y are observed [N, obs_dim]
        :param site_params: the Gaussian approximate likelihoods [2, N, obs_dim]
        :return:
            if store is True:
                filtered_mean: intermediate filtering means [N, state_dim, 1]
                filtered_cov: intermediate filtering covariances [N, state_dim, state_dim]
                site_mean: mean of the approximate likelihood sₙ(fₙ) [N, obs_dim]
                site_var: variance of the approximate likelihood sₙ(fₙ) [N, obs_dim]
            otherwise:
                neg_log_marg_lik: the filter energy, i.e. negative log-marginal likelihood -log p(y),
                                  used for hyperparameter optimisation (learning) [scalar]
        """
        theta_prior, theta_lik = softplus(params[0]), softplus(params[1])
        self.update_model(theta_prior)  # all model components that are not static must be computed inside the function
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
            for n in s.range(N):
                y_n = y[n]
                # -- KALMAN PREDICT --
                #  mₙ⁻ = Aₙ mₙ₋₁
                #  Pₙ⁻ = Aₙ Pₙ₋₁ Aₙ' + Qₙ, where Qₙ = Pinf - Aₙ Pinf Aₙ'
                A = self.prior.expm(dt[n], theta_prior)
                m_ = A @ s.m
                P_ = A @ (s.P - self.Pinf) @ A.T + self.Pinf
                # --- KALMAN UPDATE ---
                # Given previous predicted mean mₙ⁻ and cov Pₙ⁻, incorporate yₙ to get filtered mean mₙ &
                # cov Pₙ and compute the marginal likelihood p(yₙ|y₁,...,yₙ₋₁)
                mu = self.H @ m_
                var = self.H @ P_ @ self.H.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    y_n = jnp.where(mask[n], mu, y_n)  # fill in masked obs with prior expectation to prevent NaN grads
                if sampling:  # are we computing posterior samples via smoothing in an auxillary model?
                    log_lik_n, site_mu, site_var = gaussian_moment_match(y_n, mu, var, s.site_var[n], True)
                else:
                    if site_params is None:  # are we computing new sites? then run model-specific update
                        log_lik_n, site_mu, site_var = self.sites.update(self.likelihood, y_n, mu, var, theta_lik,
                                                                         True, None)
                    else:  # are we using supplied sites? then just compute the log-marginal likelihood
                        log_lik_n = self.likelihood.moment_match(y_n, mu, var, theta_lik, False, 1.0)  # lml
                        site_mu, site_var = s.site_mean[n], s.site_var[n]  # use supplied site parameters
                # modified Kalman update (see Nickish et. al. ICML 2018 or Wilkinson et. al. ICML 2019):
                S = var + site_var
                L, low = cho_factor(S)
                K = (cho_solve((L, low), self.H @ P_)).T
                s.m = m_ + K @ (site_mu - mu)
                s.P = P_ - K @ S @ K.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    s.m = jnp.where(mask[n], m_, s.m)
                    s.P = jnp.where(mask[n], P_, s.P)
                    log_lik_n = jnp.where(mask[n][..., 0, 0], jnp.zeros_like(log_lik_n), log_lik_n)
                s.neg_log_marg_lik -= jnp.sum(log_lik_n)
                if store:
                    s.filtered_mean = index_update(s.filtered_mean, index[n, ...], s.m)
                    s.filtered_cov = index_update(s.filtered_cov, index[n, ...], s.P)
                    s.site_mean = index_update(s.site_mean, index[n, ...], jnp.squeeze(site_mu.T))
                    s.site_var = index_update(s.site_var, index[n, ...], jnp.squeeze(site_var.T))
        if store:
            return s.filtered_mean, s.filtered_cov, (s.site_mean, s.site_var)
        return s.neg_log_marg_lik

    @partial(jit, static_argnums=0)
    def rauch_tung_striebel_smoother(self, params, m_filtered, P_filtered, dt, y=None, site_params=None):
        """
        Run the RTS smoother to get p(fₙ|y₁,...,y_N),
        i.e. compute p(f)𝚷ₙsₙ(fₙ) where sₙ(fₙ) are the sites (approx. likelihoods).
        If sites are provided, then it is assumed we are running EP, and we compute
        new sites by first calculating the cavity distribution and then performing moment matching.
        :param params: the model parameters, i.e the hyperparameters of the prior & likelihood
        :param m_filtered: the intermediate distribution means computed during filtering [N, state_dim, 1]
        :param P_filtered: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [N, 1]
        :param y: observed data [N, obs_dim]
        :param site_params: the Gaussian approximate likelihoods [2, N, obs_dim]
        :return:
            smoothed_mean: the posterior marginal means [N, obs_dim]
            smoothed_var: the posterior marginal variances [N, obs_dim]
            site_params: the updated EP sites [2, N, obs_dim]
        """
        theta_prior, theta_lik = softplus(params[0]), softplus(params[1])
        self.update_model(theta_prior)  # all model components that are not static must be computed inside the function
        N = dt.shape[0]
        dt = jnp.concatenate([dt[1:], jnp.array([0.0])], axis=0)
        with loops.Scope() as s:
            s.m, s.P = m_filtered[-1, ...], P_filtered[-1, ...]
            s.smoothed_mean = jnp.zeros([N, self.obs_dim])
            s.smoothed_var = jnp.zeros([N, self.obs_dim])
            if site_params is not None:
                s.site_mean, s.site_var = site_params
            for n in s.range(N-1, -1, -1):
                # --- First compute the smoothing distribution: ---
                A = self.prior.expm(dt[n], theta_prior)  # closed form integration of transition matrix
                m_predicted = A @ m_filtered[n, ...]
                tmp_gain_cov = A @ P_filtered[n, ...]
                P_predicted = A @ (P_filtered[n, ...] - self.Pinf) @ A.T + self.Pinf
                # backward Kalman gain:
                # G = F * A' * P^{-1}
                # since both F(iltered) and P(redictive) are cov matrices, thus self-adjoint, we can take the transpose:
                #   = (P^{-1} * A * F)'
                P_predicted_chol, low = cho_factor(P_predicted)
                G_transpose = cho_solve((P_predicted_chol, low), tmp_gain_cov)
                s.m = m_filtered[n, ...] + G_transpose.T @ (s.m - m_predicted)
                s.P = P_filtered[n, ...] + G_transpose.T @ (s.P - P_predicted) @ G_transpose
                s.smoothed_mean = index_update(s.smoothed_mean, index[n, ...], jnp.squeeze((self.H @ s.m).T))
                s.smoothed_var = index_update(s.smoothed_var, index[n, ...],
                                              jnp.squeeze(jnp.diag(self.H @ s.P @ self.H.T)))
                # --- Now update the site parameters: ---
                if site_params is not None:
                    # extract mean and var from state (we discard cross-covariance for now):
                    mu, var = jnp.squeeze(self.H @ s.m), jnp.squeeze(jnp.diag(self.H @ s.P @ self.H.T))
                    # calculate the new sites
                    _, site_mu, site_var = self.sites.update(self.likelihood, y[n], mu, var, theta_lik,
                                                             True, (s.site_mean[n], s.site_var[n]))
                    s.site_mean = index_update(s.site_mean, index[n, ...], jnp.squeeze(site_mu.T))
                    s.site_var = index_update(s.site_var, index[n, ...], jnp.squeeze(site_var.T))
        if site_params is not None:
            site_params = (s.site_mean, s.site_var)
        return s.smoothed_mean, s.smoothed_var, site_params

    def prior_sample(self, num_samps, x=None):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param x: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        self.update_model(softplus(self.prior.hyp))
        # TODO: sort out prior sampling - currently a bit unstable
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
        Sample from the posterior at the test locations.
        Posterior sampling works by smoothing samples from the prior using the approximate Gaussian likelihood
        model given by the sites computed during inference in the true model (e.g. via EP):
         - compute the approximate likelihood terms, sites (𝓝(f|μ*,σ²*))
         - draw samples (f*) from the prior
         - add Gaussian noise to the prior samples using auxillary model p(y*|f*) = 𝓝(y*|f*,σ²*)
         - smooth the samples by computing the posterior p(f*|y*), i.e. the posterior samples
        See Arnaud Doucet's note "A Note on Efficient Conditional Simulation of Gaussian Distributions" for details.
        :param num_samps: the number of samples to draw [scalar]
        :return:
            the posterior samples [N_test, num_samps]
        """
        post_mean, _, (site_mean, site_var) = self.predict(site_params=self.site_params)
        prior_samp = self.prior_sample(num_samps, x=self.t_all)
        prior_samp_y = self.likelihood.sample_noise(prior_samp, site_var)
        with loops.Scope() as ss:
            ss.smoothed_sample = jnp.zeros(prior_samp_y.shape)
            for i in ss.range(num_samps):
                smoothed_sample_i, _, _ = self.predict(prior_samp_y[..., i], self.dt_all, self.mask,
                                                       (site_mean, site_var), sampling=True)
                ss.smoothed_sample = index_update(ss.smoothed_sample, index[..., i], smoothed_sample_i)
        return prior_samp - ss.smoothed_sample + post_mean[..., jnp.newaxis]
