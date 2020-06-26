import jax.numpy as np
from jax.ops import index, index_update, index_add
from jax.experimental import loops
from jax import value_and_grad, jit, partial, random, vmap
from utils import softplus, softplus_list, sample_gaussian_noise, solve
import numpy as nnp  # "normal" numpy
from approximate_inference import EP
from jax.config import config
config.update("jax_enable_x64", True)
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
        - Power expectation propagation (PEP)
        - Posterior linearisation (PL)
        - Cavity linearisation (CL)
        - Iterated Kalman smoother (IKS)
        - Extended Kalman EP (EKEP)
        - Extended Kalman smoother (EKS)
        - Variational Inference - with natural gradients (VI)
    """
    def __init__(self, prior, likelihood, x, y, x_test=None, y_test=None, r_test=None, approx_inf=None):
        """
        :param prior: the model prior p(f|0,k(t,t')) object which constructs the required state space model matrices
        :param likelihood: the likelihood model object which performs moment matching and evaluates p(y|f)
        :param x: training inputs
        :param y: training data / observations
        :param x_test: test inputs
        :param y_test: test data / observations

        :param approx_inf: the approximate inference algorithm for computing the sites (EP, IKS, PL, ...)
        """
        assert x.shape[0] == y.shape[0]
        x, ind = nnp.unique(x, return_index=True, axis=0)
        if x.ndim < 2:
            x = nnp.expand_dims(x, 1)  # make 2-D
        if y.ndim < 2:
            y = nnp.expand_dims(y, 1)  # make 2-D
        y = y[ind, :]
        self.t_train = x
        self.y = np.array(y)
        if x_test is None:
            t_test = np.empty((1,) + x.shape[1:]) * np.nan
        else:
            t_test, test_sort_ind = nnp.unique(nnp.squeeze(x_test), return_index=True, axis=0)  # test inputs
            if t_test.ndim < 2:
                # t_test = t_test.reshape((-1,) + x.shape[1:])
                t_test = nnp.expand_dims(t_test, 1)
            if y_test is not None:
                y_test = y_test[test_sort_ind].reshape((-1,) + y.shape[1:])
            if r_test is not None:
                r_test = r_test[test_sort_ind]
        if r_test is None:
            r_test = np.empty((1,) + x_test.shape[1:]) * np.nan
        (self.t_all, self.test_id, self.train_id, self.y_all,
         self.mask, self.dt, self.dt_all, self.r_test) = self.input_admin(self.t_train, t_test, self.y, y_test, r_test)
        self.t_test = np.array(t_test)
        self.prior = prior
        self.likelihood = likelihood
        # construct the state space model:
        print('building SDE-GP with', self.prior.name, 'prior and', self.likelihood.name, 'likelihood ...')
        self.F, self.L, self.Qc, self.H, self.Pinf = self.prior.kernel_to_state_space()
        self.state_dim = self.F.shape[0]
        H = self.prior.measurement_model(self.t_train[0, 1:])
        self.f_dim = H.shape[0]
        self.minf = np.zeros([self.state_dim, 1])  # stationary state mean
        self.sites = EP() if approx_inf is None else approx_inf
        print('inference method is', self.sites.name)

    @staticmethod
    def input_admin(t_train, t_test, y, y_test, r_test):
        """
        Order the inputs, remove duplicates, and index the train and test input locations.
        :param t_train: training inputs [N, 1]
        :param t_test: testing inputs [N*, 1]
        :param y: observations at the training inputs [N, 1]
        :param y_test: observations at the test inputs [N*, 1]

        :return:
            t: the combined and sorted training and test inputs [N + N*, 1]
            test_id: an array of indices corresponding to the test inputs [N*, 1]
            train_id: an array of indices corresponding to the training inputs [N, 1]
            y_all: an array observations y augmented with nans at test locations [N + N*, 1]
            mask: boolean array to signify training locations [N + N*, 1]
            dt: training step sizes, Δtₙ = tₙ - tₙ₋₁ [N, 1]
            dt_all: combined training and test step sizes, Δtₙ = tₙ - tₙ₋₁ [N + N*, 1]
        """
        if not (t_test.shape[1] == t_train.shape[1]):
            t_test = np.concatenate([t_test[:, 0][:, None],
                                     np.nan * np.empty([t_test.shape[0], t_train.shape[1]-1])], axis=1)
        #     r_test = t_test.copy()  # spacial test points
        #     t_test = t_test[:, :t_train.shape[1]]  # temporal test points
        # else:
        #     r_test = t_test.copy()
        # here we use non-JAX numpy to sort out indexing of these static arrays
        t_test_train = nnp.concatenate([t_test, t_train])
        t_test_train = t_test_train[~np.isnan(t_test_train[:, 0]), :]
        t, x_ind = nnp.unique(t_test_train, return_inverse=True, axis=0)
        n_test = t_test[~np.isnan(t_test[:, 0]), ...].shape[0]  # number of test locations
        test_id = x_ind[:n_test]  # index the test locations
        train_id = x_ind[n_test:]  # index the training locations
        y_all = nnp.nan * nnp.zeros([t.shape[0], y.shape[1]])  # observation vector with nans at test locations
        y_all[x_ind[n_test:], :] = y  # and the data at the train locations
        if y_test is not None:
            y_all[x_ind[:n_test], :] = y_test  # and the data at the train locations
        mask = nnp.ones(y_all.shape[0], dtype=bool)
        mask[train_id] = False
        dt = nnp.concatenate([np.array([0.0]), nnp.diff(t_train[:, 0])])
        dt_all = nnp.concatenate([np.array([0.0]), nnp.diff(t[:, 0])])
        return (np.array(t), np.array(test_id), np.array(train_id), np.array(y_all),
                np.array(mask), np.array(dt), np.array(dt_all), np.array(r_test))

    def predict(self, y=None, dt=None, mask=None, site_params=None, sampling=False, x=None, return_full=False):
        """
        Calculate posterior predictive distribution p(f*|f,y) by filtering and smoothing across the
        training & test locations.
        This function is also used during posterior sampling to smooth the auxillary data sampled from the prior.
        The output shapes depend on return_full
        :param y: observations (nans at test locations) [M, 1]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [M, 1]
        :param mask: a boolean array signifying which elements are observed and which are nan [M, 1]
        :param site_params: the sites computed during a previous inference proceedure [2, M, obs_dim]
        :param sampling: notify whether we are doing posterior sampling
        :return:
            posterior_mean: the posterior predictive mean [M, state_dim] or [M, obs_dim]
            posterior_cov: the posterior predictive (co)variance [M, M, state_dim] or [M, obs_dim]
            site_params: the site parameters. If none are provided then new sites are computed [2, M, obs_dim]
        """
        y = self.y_all if y is None else y
        x = self.t_all if x is None else x
        dt = self.dt_all if dt is None else dt
        mask = self.mask if mask is None else mask
        params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        site_params = self.sites.site_params if site_params is None else site_params
        if site_params is not None and not sampling:
            # construct a vector of site parameters that is the full size of the test data
            # test site parameters are 𝓝(0,∞), and will not be used
            site_mean, site_var = np.zeros([dt.shape[0], 1]), 1e5 * np.ones([dt.shape[0], 1])
            # replace parameters at training locations with the supplied sites
            site_mean = index_add(site_mean, index[self.train_id], site_params[0])
            site_var = index_update(site_var, index[self.train_id], site_params[1])
            site_params = (site_mean, site_var)
        _, (filter_mean, filter_cov, site_params) = self.kalman_filter(y, dt, params, True, mask, site_params, x)
        _, posterior_mean, posterior_cov = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, dt,
                                                                             True, return_full, None, None, x)
        nlpd_test = self.negative_log_predictive_density(self.t_all[self.test_id], self.y_all[self.test_id],
                                                         posterior_mean[self.test_id],
                                                         posterior_cov[self.test_id],
                                                         softplus_list(params[0]), softplus(params[1]),
                                                         return_full)
        return posterior_mean, posterior_cov, site_params, nlpd_test

    def predict_2d(self, y=None, dt=None, mask=None, site_params=None, sampling=False, x=None):
        """
        Calculate posterior predictive distribution p(f*|f,y) by filtering and smoothing across the
        training & test locations.
        This function is also used during posterior sampling to smooth the auxillary data sampled from the prior.
        The output shapes depend on return_full
        :param y: observations (nans at test locations) [M, 1]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [M, 1]
        :param mask: a boolean array signifying which elements are observed and which are nan [M, 1]
        :param site_params: the sites computed during a previous inference proceedure [2, M, obs_dim]
        :param sampling: notify whether we are doing posterior sampling
        :return:
            posterior_mean: the posterior predictive mean [M, state_dim] or [M, obs_dim]
            posterior_cov: the posterior predictive (co)variance [M, M, state_dim] or [M, obs_dim]
            site_params: the site parameters. If none are provided then new sites are computed [2, M, obs_dim]
        """
        return_full = True
        y = self.y_all if y is None else y
        x = self.t_all if x is None else x
        dt = self.dt_all if dt is None else dt
        mask = self.mask if mask is None else mask
        params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        site_params = self.sites.site_params if site_params is None else site_params
        if site_params is not None and not sampling:
            # construct a vector of site parameters that is the full size of the test data
            # test site parameters are 𝓝(0,∞), and will not be used
            site_mean, site_var = np.zeros([dt.shape[0], 1]), 1e5 * np.ones([dt.shape[0], 1])
            # replace parameters at training locations with the supplied sites
            site_mean = index_add(site_mean, index[self.train_id], site_params[0])
            site_var = index_update(site_var, index[self.train_id], site_params[1])
            site_params = (site_mean, site_var)
        _, (filter_mean, filter_cov, site_params) = self.kalman_filter(y, dt, params, True, mask, site_params, x)
        _, posterior_mean, posterior_cov = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, dt,
                                                                             True, return_full, None, None, x)
        nlpd_test = self.negative_log_predictive_density(self.t_all[self.test_id], self.y_all[self.test_id],
                                                         posterior_mean[self.test_id],
                                                         posterior_cov[self.test_id],
                                                         softplus_list(params[0]), softplus(params[1]),
                                                         return_full)
        # posterior_mean, posterior_cov, site_params, nlpd_test = self.predict(y, dt, mask, site_params, sampling, x, True)
        mean_test_filt, cov_test_filt = filter_mean[self.test_id], filter_cov[self.test_id]
        mean_test, cov_test = posterior_mean[self.test_id], posterior_cov[self.test_id]
        measure_func = vmap(
            self.compute_measurement, (0, 0, 0, None)
        )
        m_test_filt, v_test_filt = measure_func(self.r_test, mean_test_filt, cov_test_filt, softplus_list(self.prior.hyp))
        m_test, v_test = measure_func(self.r_test, mean_test, cov_test, softplus_list(self.prior.hyp))
        return m_test, v_test, site_params, nlpd_test, m_test_filt, v_test_filt

    @partial(jit, static_argnums=0)
    def compute_measurement(self, x, mean, cov, hyp_prior):
        H = self.prior.measurement_model(x, hyp_prior)
        return H @ mean, H @ cov @ H.T

    def negative_log_predictive_density(self, x_test, y_test, m_test, v_test, hyp_prior, hyp_lik, full_cov):
        """
        Compute the (normalised) negative log predictive density (NLPD) of the test data yₙ*:
            NLPD = - ∑ₙ log ∫ p(yₙ*|fₙ*) 𝓝(fₙ*|mₙ*,vₙ*) dfₙ*
        where fₙ* is the function value at the test location.
        The above is equivalent to the quantity used for EP moment matching, so we
        vectorise that method using vmap, and compute it for all test data.
        :param y_test: the test data yₙ*  [N*, 1]
        :param m_test: posterior predictive mean at the test locations, mₙ*  [N*, 1]
        :param v_test: posterior predictive variance at the test locations, vₙ*  [N*, 1]
        :param hyp_lik: the hyperparameters of the likelihood model  [array]
        :return:
            NLPD: the negative log predictive density for the test data
        """
        if full_cov:
            measure_func = vmap(
                self.compute_measurement, (0, 0, 0, None)
            )
            m_test, v_test = measure_func(x_test, m_test, v_test, hyp_prior)
        lpd_func = vmap(
            self.likelihood.moment_match, (0, 0, 0, None, None, None)
        )
        log_predictive_density, _, _ = lpd_func(y_test, m_test, v_test, hyp_lik, 1, None)
        return -np.mean(log_predictive_density)  # mean = normalised sum

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
        neg_log_marg_lik, dlZ = value_and_grad(self.kalman_filter, argnums=2)(self.y, self.dt, params, False)
        return neg_log_marg_lik, dlZ

    def run(self, params=None):
        """
        A single parameter update step - to be fed to a gradient-based optimiser.
         - we first compute the marginal likelihood and its gradient w.r.t. the hyperparameters via filtering
         - then update the site parameters via smoothing (site mean and variance)
        :param params: the model parameters. If not supplied then defaults to the model's
                       assigned parameters [num_params]
        :return:
            neg_log_marg_lik: the negative log-marginal likelihood -log p(y), i.e. the energy [scalar]
            dlZ: the derivative of the energy w.r.t. the model parameters [num_params]
        """
        if params is None:
            # fetch the model parameters from the prior and the likelihood
            params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        # run the forward filter to calculate the filtering distribution and compute the negative
        # log-marginal likelihood and its gradient in order to update the hyperparameters
        (neg_log_marg_lik, aux), dlZ = value_and_grad(self.kalman_filter,
                                                      argnums=2, has_aux=True)(self.y, self.dt, params, True,
                                                                               None, self.sites.site_params,
                                                                               self.t_train)
        filter_mean, filter_cov, self.sites.site_params = aux
        # run the smoother and update the sites
        self.sites.site_params = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, self.dt,
                                                                   False, False, self.y,
                                                                   self.sites.site_params, self.t_train)
        return neg_log_marg_lik, dlZ

    def run_two_stage(self, params=None):
        """
        Note: This 2-stage version has been replaced by the more elegant implementation above, however we
        keep this method because it is more accurate, and faster in practice for small data.
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
        # if self.sites.site_params=None then the filter initialises the sites too
        _, (filter_mean, filter_cov, self.sites.site_params) = self.kalman_filter(self.y, self.dt, params, True,
                                                                                  None, self.sites.site_params,
                                                                                  self.t_train)
        # run the smoother and update the sites
        self.sites.site_params = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, self.dt,
                                                                   False, False, self.y,
                                                                   self.sites.site_params, self.t_train)
        # compute the negative log-marginal likelihood and its gradient in order to update the hyperparameters
        neg_log_marg_lik, dlZ = value_and_grad(self.kalman_filter, argnums=2)(self.y, self.dt, params, False,
                                                                              None, self.sites.site_params,
                                                                              self.t_train)
        return neg_log_marg_lik, dlZ

    def update_model(self, theta_prior=None):
        """
        Re-construct the SDE-GP model with latest parameters.
        :param theta_prior: the hyperparameters of the GP prior
        return:
            computes the model matrices F, L, Qc, H, Pinf. See the prior class for details
        """
        self.F, self.L, self.Qc, self.H, self.Pinf = self.prior.kernel_to_state_space(hyperparams=theta_prior)

    @partial(jit, static_argnums=(0, 4))
    def kalman_filter(self, y, dt, params, store=False, mask=None, site_params=None, x=None):
        """
        Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
        The Kalman update step invloves some control flow to work out whether we are
            i) initialising the sites
            ii) using supplied sites
            iii) performing a Gaussian update with fixed parameters (e.g. in posterior sampling or ELBO calc.)
        If store is True then we compute and return the intermediate filtering distributions
        p(fₙ|y₁,...,yₙ) and sites sₙ(fₙ), otherwise we do not store the intermediates and simply
        return the energy / negative log-marginal likelihood, -log p(y).
        :param y: observed data [N, obs_dim]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [N, 1]
        :param params: the model parameters, i.e the hyperparameters of the prior & likelihood
        :param store: flag to notify whether to store the intermediates
        :param mask: boolean array signifying which elements of y are observed [N, obs_dim]
        :param site_params: the Gaussian approximate likelihoods [2, N, obs_dim]
        :return:
            if store is True:
                neg_log_marg_lik: the filter energy, i.e. negative log-marginal likelihood -log p(y),
                                  used for hyperparameter optimisation (learning) [scalar]
                filtered_mean: intermediate filtering means [N, state_dim, 1]
                filtered_cov: intermediate filtering covariances [N, state_dim, state_dim]
                site_mean: mean of the approximate likelihood sₙ(fₙ) [N, obs_dim]
                site_var: variance of the approximate likelihood sₙ(fₙ) [N, obs_dim]
            otherwise:
                neg_log_marg_lik: the filter energy, i.e. negative log-marginal likelihood -log p(y),
                                  used for hyperparameter optimisation (learning) [scalar]
        """
        theta_prior, theta_lik = softplus_list(params[0]), softplus(params[1])
        self.update_model(theta_prior)  # all model components that are not static must be computed inside the function
        if mask is not None:
            mask = mask[..., np.newaxis, np.newaxis]  # align mask.shape with y.shape
        N = dt.shape[0]
        with loops.Scope() as s:
            s.neg_log_marg_lik = 0.0  # negative log-marginal likelihood
            s.m, s.P = self.minf, self.Pinf
            if store:
                s.filtered_mean = np.zeros([N, self.state_dim, 1])
                s.filtered_cov = np.zeros([N, self.state_dim, self.state_dim])
                s.site_mean = np.zeros([N, self.f_dim])
                s.site_var = np.zeros([N, self.f_dim])
            for n in s.range(N):
                y_n = y[n]
                # -- KALMAN PREDICT --
                #  mₙ⁻ = Aₙ mₙ₋₁
                #  Pₙ⁻ = Aₙ Pₙ₋₁ Aₙ' + Qₙ, where Qₙ = Pinf - Aₙ Pinf Aₙ'
                A = self.prior.state_transition(dt[n], theta_prior)
                m_ = A @ s.m
                P_ = A @ (s.P - self.Pinf) @ A.T + self.Pinf
                # --- KALMAN UPDATE ---
                # Given previous predicted mean mₙ⁻ and cov Pₙ⁻, incorporate yₙ to get filtered mean mₙ &
                # cov Pₙ and compute the marginal likelihood p(yₙ|y₁,...,yₙ₋₁)
                H = self.prior.measurement_model(x[n, 1:], theta_prior)
                mu = H @ m_
                var = H @ P_ @ H.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    y_n = np.where(mask[n], mu, y_n)  # fill in masked obs with prior expectation to prevent NaN grads
                log_lik_n, site_mu, site_var = self.sites.update(self.likelihood, y_n, mu, var, theta_lik, None)
                if site_params is not None:  # use supplied site parameters to perform the update
                    site_mu, site_var = site_params[0][n], site_params[1][n]
                # modified Kalman update (see Nickish et. al. ICML 2018 or Wilkinson et. al. ICML 2019):
                S = var + site_var
                K = solve(S, H @ P_).T  # HP(S^-1)
                s.m = m_ + K @ (site_mu - mu)
                s.P = P_ - K @ S @ K.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    s.m = np.where(mask[n], m_, s.m)
                    s.P = np.where(mask[n], P_, s.P)
                    log_lik_n = np.where(mask[n][..., 0, 0], np.zeros_like(log_lik_n), log_lik_n)
                s.neg_log_marg_lik -= np.sum(log_lik_n)
                if store:
                    s.filtered_mean = index_add(s.filtered_mean, index[n, ...], s.m)
                    s.filtered_cov = index_add(s.filtered_cov, index[n, ...], s.P)
                    s.site_mean = index_add(s.site_mean, index[n, ...], np.squeeze(site_mu.T))
                    s.site_var = index_add(s.site_var, index[n, ...], np.squeeze(site_var.T))
        if store:
            return s.neg_log_marg_lik, (s.filtered_mean, s.filtered_cov, (s.site_mean, s.site_var))
        return s.neg_log_marg_lik

    @partial(jit, static_argnums=(0, 5, 6))
    def rauch_tung_striebel_smoother(self, params, m_filtered, P_filtered, dt, store=False, return_full=False,
                                     y=None, site_params=None, x=None):
        """
        Run the RTS smoother to get p(fₙ|y₁,...,y_N),
        i.e. compute p(f)𝚷ₙsₙ(fₙ) where sₙ(fₙ) are the sites (approx. likelihoods).
        If sites are provided, then it is assumed they are to be updated, which is done by
        calling the site-specific update() method.
        :param params: the model parameters, i.e the hyperparameters of the prior & likelihood
        :param m_filtered: the intermediate distribution means computed during filtering [N, state_dim, 1]
        :param P_filtered: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [N, 1]
        :param y: observed data [N, obs_dim]
        :param site_params: the Gaussian approximate likelihoods [2, N, obs_dim]
        :return:
            var_exp: the sum of the variational expectations [scalar]
            smoothed_mean: the posterior marginal means [N, obs_dim]
            smoothed_var: the posterior marginal variances [N, obs_dim]
            site_params: the updated sites [2, N, obs_dim]
        """
        theta_prior, theta_lik = softplus_list(params[0]), softplus(params[1])
        self.update_model(theta_prior)  # all model components that are not static must be computed inside the function
        N = dt.shape[0]
        dt = np.concatenate([dt[1:], np.array([0.0])], axis=0)
        with loops.Scope() as s:
            s.m, s.P = m_filtered[-1, ...], P_filtered[-1, ...]
            if return_full:
                s.smoothed_mean = np.zeros([N, self.state_dim, 1])
                s.smoothed_cov = np.zeros([N, self.state_dim, self.state_dim])
            else:
                s.smoothed_mean = np.zeros([N, self.f_dim])
                s.smoothed_cov = np.zeros([N, self.f_dim])
            if site_params is not None:
                s.site_mean, s.site_var = np.zeros([N, self.f_dim]), np.zeros([N, self.f_dim])
            for n in s.range(N-1, -1, -1):
                # --- First compute the smoothing distribution: ---
                A = self.prior.state_transition(dt[n], theta_prior)  # closed form integration of transition matrix
                m_predicted = A @ m_filtered[n, ...]
                tmp_gain_cov = A @ P_filtered[n, ...]
                P_predicted = A @ (P_filtered[n, ...] - self.Pinf) @ A.T + self.Pinf
                # backward Kalman gain:
                # G = F * A' * P^{-1}
                # since both F(iltered) and P(redictive) are cov matrices, thus self-adjoint, we can take the transpose:
                #   = (P^{-1} * A * F)'
                G_transpose = solve(P_predicted, tmp_gain_cov)  # (P^-1)AF
                s.m = m_filtered[n, ...] + G_transpose.T @ (s.m - m_predicted)
                s.P = P_filtered[n, ...] + G_transpose.T @ (s.P - P_predicted) @ G_transpose
                H = self.prior.measurement_model(x[n, 1:], theta_prior)
                if store:
                    if return_full:
                        s.smoothed_mean = index_add(s.smoothed_mean, index[n, ...], s.m)
                        s.smoothed_cov = index_add(s.smoothed_cov, index[n, ...], s.P)
                    else:
                        s.smoothed_mean = index_add(s.smoothed_mean, index[n, ...], np.squeeze((H @ s.m).T))
                        s.smoothed_cov = index_add(s.smoothed_cov, index[n, ...],
                                                   np.squeeze(np.diag(H @ s.P @ H.T)))
                # --- Now update the site parameters: ---
                if site_params is not None:
                    # extract mean and var from state (we discard cross-covariance for now):
                    mu, var = H @ s.m, np.diag(H @ s.P @ H.T)
                    # calculate the new sites
                    _, site_mu, site_var = self.sites.update(self.likelihood, y[n], mu, var, theta_lik,
                                                             (site_params[0][n], site_params[1][n]))
                    s.site_mean = index_add(s.site_mean, index[n, ...], np.squeeze(site_mu.T))
                    s.site_var = index_add(s.site_var, index[n, ...], np.squeeze(site_var.T))
        if site_params is not None:
            site_params = (s.site_mean, s.site_var)
        if store:
            return site_params, s.smoothed_mean, s.smoothed_cov
        return site_params

    def prior_sample(self, num_samps, x=None):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param x: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        self.update_model(softplus_list(self.prior.hyp))
        if x is None:
            x = self.t_all
        else:
            x_ind = np.argsort(x[:, 0])
            x = x[x_ind]
        dt = np.concatenate([np.array([0.0]), np.diff(x[:, 0])])
        N = dt.shape[0]
        with loops.Scope() as s:
            s.f_sample = np.zeros([N, self.f_dim, num_samps])
            s.m = np.linalg.cholesky(self.Pinf) @ random.normal(random.PRNGKey(99), shape=[self.state_dim, 1])
            for i in s.range(num_samps):
                s.m = np.linalg.cholesky(self.Pinf) @ random.normal(random.PRNGKey(i), shape=[self.state_dim, 1])
                for k in s.range(N):
                    A = self.prior.state_transition(dt[k], self.prior.hyp)  # transition and noise process matrices
                    Q = self.Pinf - A @ self.Pinf @ A.T
                    C = np.linalg.cholesky(Q + 1e-7 * np.eye(self.state_dim))  # <--- can be a bit unstable
                    # we need to provide a different PRNG seed every time:
                    s.m = A @ s.m + C @ random.normal(random.PRNGKey(i*k+k), shape=[self.state_dim, 1])
                    H = self.prior.measurement_model(x[k, 1:], softplus_list(self.prior.hyp))
                    f = (H @ s.m).T
                    s.f_sample = index_add(s.f_sample, index[k, ..., i], np.squeeze(f))
        return s.f_sample

    def posterior_sample(self, num_samps):
        """
        Sample from the posterior at the test locations.
        Posterior sampling works by smoothing samples from the prior using the approximate Gaussian likelihood
        model given by the sites computed during inference in the true model:
         - compute the approximate likelihood terms, sites (𝓝(f|μ*,σ²*))
         - draw samples (f*) from the prior
         - add Gaussian noise to the prior samples using auxillary model p(y*|f*) = 𝓝(y*|f*,σ²*)
         - smooth the samples by computing the posterior p(f*|y*), i.e. the posterior samples
        See Arnaud Doucet's note "A Note on Efficient Conditional Simulation of Gaussian Distributions" for details.
        :param num_samps: the number of samples to draw [scalar]
        :return:
            the posterior samples [N_test, num_samps]
        """
        post_mean, _, (site_mean, site_var), _ = self.predict(site_params=self.sites.site_params)
        prior_samp = self.prior_sample(num_samps, x=self.t_all)
        prior_samp_y = sample_gaussian_noise(prior_samp, site_var)
        with loops.Scope() as ss:
            ss.smoothed_sample = np.zeros(prior_samp_y.shape)
            for i in ss.range(num_samps):
                smoothed_sample_i, _, _, _ = self.predict(np.zeros_like(prior_samp_y[..., i]), self.dt_all, self.mask,
                                                          (prior_samp_y[..., i], site_var), sampling=True)
                ss.smoothed_sample = index_add(ss.smoothed_sample, index[..., i], smoothed_sample_i)
        return prior_samp - ss.smoothed_sample + post_mean[..., np.newaxis]
