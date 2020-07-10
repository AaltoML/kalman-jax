import jax.numpy as np
from jax.ops import index, index_update, index_add
from jax.experimental import loops
from jax import value_and_grad, jit, partial, random, vmap
from utils import softplus, softplus_list, sample_gaussian_noise, solve, input_admin
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
    def __init__(self, prior, likelihood, t, y, r=None, t_test=None, y_test=None, r_test=None, approx_inf=None):
        """
        :param prior: the model prior p(f|0,k(t,t')) object which constructs the required state space model matrices
        :param likelihood: the likelihood model object which performs moment matching and evaluates p(y|f)
        :param t: training inputs
        :param y: training data / observations
        :param r: training spatial points
        :param t_test: test inputs
        :param y_test: test data / observations
        :param r_test: test spatial points
        :param approx_inf: the approximate inference algorithm for computing the sites (EP, VI, UKS, ...)
        """
        (self.t_all, self.y_all, self.r_all,
         self.t_train, self.y_train, self.r_train,
         self.r_test,
         self.dt_all, self.dt_train,
         self.train_id, self.test_id, self.mask
         ) = input_admin(t, y, r, t_test, y_test, r_test)
        self.prior = prior
        self.likelihood = likelihood
        # construct the state space model:
        print('building SDE-GP with', self.prior.name, 'prior and', self.likelihood.name, 'likelihood ...')
        self.F, self.L, self.Qc, self.H, self.Pinf = self.prior.kernel_to_state_space()
        self.state_dim = self.F.shape[0]
        H = self.prior.measurement_model(self.r_all[0])
        self.func_dim = H.shape[0]
        self.minf = np.zeros([self.state_dim, 1])  # stationary state mean
        self.sites = EP() if approx_inf is None else approx_inf
        print('inference method is', self.sites.name)

    def predict(self, y=None, dt=None, mask=None, site_params=None, sampling=False, r=None, return_full=False, compute_nlpd=True):
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
        r = self.r_all if r is None else r
        dt = self.dt_all if dt is None else dt
        mask = self.mask if mask is None else mask
        params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        site_params = self.sites.site_params if site_params is None else site_params
        if site_params is not None and not sampling:
            # construct a vector of site parameters that is the full size of the test data
            # test site parameters are 𝓝(0,∞), and will not be used
            site_mean = np.zeros([dt.shape[0], self.func_dim, 1])
            site_cov = 1e5 * np.tile(np.eye(self.func_dim), (dt.shape[0], 1, 1))
            # replace parameters at training locations with the supplied sites
            site_mean = index_add(site_mean, index[self.train_id], site_params[0])
            site_cov = index_update(site_cov, index[self.train_id], site_params[1])
            site_params = (site_mean, site_cov)
        _, (filter_mean, filter_cov, site_params) = self.kalman_filter(y, dt, params, True, mask, site_params, r)
        _, posterior_mean, posterior_cov = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, dt,
                                                                             True, return_full, None, None, r)
        if compute_nlpd:
            nlpd_test = self.negative_log_predictive_density(self.t_all[self.test_id], self.y_all[self.test_id],
                                                             posterior_mean[self.test_id],
                                                             posterior_cov[self.test_id],
                                                             softplus_list(params[0]), softplus(params[1]),
                                                             return_full)
        else:
            nlpd_test = np.nan
        return posterior_mean, posterior_cov, site_params, nlpd_test

    def predict_2d(self, y=None, dt=None, mask=None, site_params=None, sampling=False, r=None, compute_nlpd=True):
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
        r = self.r_all if r is None else r
        dt = self.dt_all if dt is None else dt
        mask = self.mask if mask is None else mask
        params = [self.prior.hyp.copy(), self.likelihood.hyp.copy()]
        site_params = self.sites.site_params if site_params is None else site_params
        if site_params is not None and not sampling:
            # construct a vector of site parameters that is the full size of the test data
            # test site parameters are 𝓝(0,∞), and will not be used
            site_mean = np.zeros([dt.shape[0], self.func_dim, 1])
            site_cov = 1e5 * np.tile(np.eye(self.func_dim), (dt.shape[0], 1, 1))
            # replace parameters at training locations with the supplied sites
            site_mean = index_add(site_mean, index[self.train_id], site_params[0])
            site_cov = index_update(site_cov, index[self.train_id], site_params[1])
            site_params = (site_mean, site_cov)
        _, (filter_mean, filter_cov, site_params) = self.kalman_filter(y, dt, params, True, mask, site_params, r)
        _, posterior_mean, posterior_cov = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, dt,
                                                                             True, return_full, None, None, r)
        if compute_nlpd:
            nlpd_test = self.negative_log_predictive_density(self.t_all[self.test_id], self.y_all[self.test_id],
                                                             posterior_mean[self.test_id],
                                                             posterior_cov[self.test_id],
                                                             softplus_list(params[0]), softplus(params[1]),
                                                             return_full)
        else:
            nlpd_test = np.nan
        mean_test_filt, cov_test_filt = filter_mean[self.test_id], filter_cov[self.test_id]
        mean_test, cov_test = posterior_mean[self.test_id], posterior_cov[self.test_id]
        measure_func = vmap(
            self.compute_measurement, (0, 0, 0, None)
        )
        m_test_filt, v_test_filt = measure_func(self.r_test, mean_test_filt,
                                                cov_test_filt, softplus_list(self.prior.hyp))
        m_test, v_test = measure_func(self.r_test, mean_test, cov_test, softplus_list(self.prior.hyp))
        return m_test, v_test, site_params, nlpd_test, m_test_filt, v_test_filt

    @partial(jit, static_argnums=0)
    def compute_measurement(self, r, mean, cov, hyp_prior):
        H = self.prior.measurement_model(r, hyp_prior)
        return H @ mean, H @ cov @ H.T

    def negative_log_predictive_density(self, t_test, y_test, m_test, v_test, hyp_prior, hyp_lik, full_cov):
        """
        Compute the (normalised) negative log predictive density (NLPD) of the test data yₙ*:
            NLPD = - ∑ₙ log ∫ p(yₙ*|fₙ*) 𝓝(fₙ*|mₙ*,vₙ*) dfₙ*
        where fₙ* is the function value at the test location.
        The above is equivalent to the quantity used for EP moment matching, so we
        vectorise that method using vmap, and compute it for all test data.
        :param t_test: the test inputs tₙ*  [N*, 1]
        :param y_test: the test data yₙ*  [N*, 1]
        :param m_test: posterior predictive mean at the test locations, mₙ*  [N*, 1]
        :param v_test: posterior predictive (co)variance at the test locations, vₙ*  [N*, 1]
        :param hyp_prior: the hyperparameters of the prior  [array]
        :param hyp_lik: the hyperparameters of the likelihood model  [array]
        :param full_cov: flag signalling whether m_testa and v_test are the full state dist.
        :return:
            NLPD: the negative log predictive density for the test data
        """
        if full_cov:
            measure_func = vmap(
                self.compute_measurement, (0, 0, 0, None)
            )
            m_test, v_test = measure_func(t_test, m_test, v_test, hyp_prior)
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
        neg_log_marg_lik, dlZ = value_and_grad(self.kalman_filter, argnums=2)(self.y_train, self.dt_train,
                                                                              params, False, None,
                                                                              self.sites.site_params, self.r_train)
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
                                                      argnums=2, has_aux=True)(self.y_train, self.dt_train, params, True,
                                                                               None, self.sites.site_params,
                                                                               self.r_train)
        filter_mean, filter_cov, self.sites.site_params = aux
        # run the smoother and update the sites
        self.sites.site_params = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, self.dt_train,
                                                                   False, False, self.y_train,
                                                                   self.sites.site_params, self.r_train)
        return neg_log_marg_lik, dlZ

    def run_two_stage(self, params=None):
        """
        Note: This 2-stage version has been replaced by the more elegant implementation above, however we
        keep this method because it is more accurate, and faster in practice for small datasets.
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
        _, (filter_mean, filter_cov, self.sites.site_params) = self.kalman_filter(self.y_train, self.dt_train, params, True,
                                                                                  None, self.sites.site_params,
                                                                                  self.r_train)
        # run the smoother and update the sites
        self.sites.site_params = self.rauch_tung_striebel_smoother(params, filter_mean, filter_cov, self.dt_train,
                                                                   False, False, self.y_train,
                                                                   self.sites.site_params, self.r_train)
        # compute the negative log-marginal likelihood and its gradient in order to update the hyperparameters
        neg_log_marg_lik, dlZ = value_and_grad(self.kalman_filter, argnums=2)(self.y_train, self.dt_train, params, False,
                                                                              None, self.sites.site_params,
                                                                              self.r_train)
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
    def kalman_filter(self, y, dt, params, store=False, mask=None, site_params=None, r=None):
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
        :param r: spatial input locations
        :return:
            if store is True:
                neg_log_marg_lik: the filter energy, i.e. negative log-marginal likelihood -log p(y),
                                  used for hyperparameter optimisation (learning) [scalar]
                filtered_mean: intermediate filtering means [N, state_dim, 1]
                filtered_cov: intermediate filtering covariances [N, state_dim, state_dim]
                site_mean: mean of the approximate likelihood sₙ(fₙ) [N, obs_dim]
                site_cov: variance of the approximate likelihood sₙ(fₙ) [N, obs_dim]
            otherwise:
                neg_log_marg_lik: the filter energy, i.e. negative log-marginal likelihood -log p(y),
                                  used for hyperparameter optimisation (learning) [scalar]
        """
        theta_prior, theta_lik = softplus_list(params[0]), softplus(params[1])
        self.update_model(theta_prior)  # all model components that are not static must be computed inside the function
        N = dt.shape[0]
        with loops.Scope() as s:
            s.neg_log_marg_lik = 0.0  # negative log-marginal likelihood
            s.m, s.P = self.minf, self.Pinf
            if store:
                s.filtered_mean = np.zeros([N, self.state_dim, 1])
                s.filtered_cov = np.zeros([N, self.state_dim, self.state_dim])
                s.site_mean = np.zeros([N, self.func_dim, 1])
                s.site_cov = np.zeros([N, self.func_dim, self.func_dim])
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
                H = self.prior.measurement_model(r[n], theta_prior)
                predict_mean = H @ m_
                predict_cov = H @ P_ @ H.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    y_n = np.where(mask[n], predict_mean[:y_n.shape[0], 0], y_n)  # fill in masked obs with expectation
                log_lik_n, site_mean, site_cov = self.sites.update(self.likelihood, y_n, predict_mean, predict_cov,
                                                                   theta_lik, None)
                if site_params is not None:  # use supplied site parameters to perform the update
                    site_mean, site_cov = site_params[0][n], site_params[1][n]
                # modified Kalman update (see Nickish et. al. ICML 2018 or Wilkinson et. al. ICML 2019):
                S = predict_cov + site_cov
                K = solve(S, H @ P_).T  # HP(S^-1)
                s.m = m_ + K @ (site_mean - predict_mean)
                s.P = P_ - K @ S @ K.T
                if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
                    s.m = np.where(np.any(mask[n]), m_, s.m)
                    s.P = np.where(np.any(mask[n]), P_, s.P)
                    log_lik_n = np.where(mask[n][..., 0], np.zeros_like(log_lik_n), log_lik_n)
                s.neg_log_marg_lik -= np.sum(log_lik_n)
                if store:
                    s.filtered_mean = index_add(s.filtered_mean, index[n, ...], s.m)
                    s.filtered_cov = index_add(s.filtered_cov, index[n, ...], s.P)
                    s.site_mean = index_add(s.site_mean, index[n, ...], site_mean)
                    s.site_cov = index_add(s.site_cov, index[n, ...], site_cov)
        if store:
            return s.neg_log_marg_lik, (s.filtered_mean, s.filtered_cov, (s.site_mean, s.site_cov))
        return s.neg_log_marg_lik

    @partial(jit, static_argnums=(0, 5, 6))
    def rauch_tung_striebel_smoother(self, params, m_filtered, P_filtered, dt, store=False, return_full=False,
                                     y=None, site_params=None, r=None):
        """
        Run the RTS smoother to get p(fₙ|y₁,...,y_N),
        i.e. compute p(f)𝚷ₙsₙ(fₙ) where sₙ(fₙ) are the sites (approx. likelihoods).
        If sites are provided, then it is assumed they are to be updated, which is done by
        calling the site-specific update() method.
        :param params: the model parameters, i.e the hyperparameters of the prior & likelihood
        :param m_filtered: the intermediate distribution means computed during filtering [N, state_dim, 1]
        :param P_filtered: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
        :param dt: step sizes Δtₙ = tₙ - tₙ₋₁ [N, 1]
        :param store: a flag determining whether to store and return state mean and covariance
        :param return_full: a flag determining whether to return the full state distribution or just the function(s)
        :param y: observed data [N, obs_dim]
        :param site_params: the Gaussian approximate likelihoods [2, N, obs_dim]
        :param r: spatial input locations
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
                s.smoothed_mean = np.zeros([N, self.func_dim, 1])
                s.smoothed_cov = np.zeros([N, self.func_dim, self.func_dim])
            if site_params is not None:
                s.site_mean = np.zeros([N, self.func_dim, 1])
                s.site_var = np.zeros([N, self.func_dim, self.func_dim])
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
                H = self.prior.measurement_model(r[n], theta_prior)
                if store:
                    if return_full:
                        s.smoothed_mean = index_add(s.smoothed_mean, index[n, ...], s.m)
                        s.smoothed_cov = index_add(s.smoothed_cov, index[n, ...], s.P)
                    else:
                        s.smoothed_mean = index_add(s.smoothed_mean, index[n, ...], H @ s.m)
                        s.smoothed_cov = index_add(s.smoothed_cov, index[n, ...], H @ s.P @ H.T)
                # --- Now update the site parameters: ---
                if site_params is not None:
                    # extract mean and var from state:
                    post_mean, post_cov = H @ s.m, H @ s.P @ H.T
                    # calculate the new sites
                    _, site_mu, site_cov = self.sites.update(self.likelihood, y[n], post_mean, post_cov, theta_lik,
                                                             (site_params[0][n], site_params[1][n]))
                    s.site_mean = index_add(s.site_mean, index[n, ...], site_mu)
                    s.site_var = index_add(s.site_var, index[n, ...], site_cov)
        if site_params is not None:
            site_params = (s.site_mean, s.site_var)
        if store:
            return site_params, s.smoothed_mean, s.smoothed_cov
        return site_params

    def prior_sample(self, num_samps, t=None):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param t: the input locations at which to sample (defaults to train+test set) [N_samp, 1]
        :return:
            f_sample: the prior samples [S, N_samp]
        """
        self.update_model(softplus_list(self.prior.hyp))
        if t is None:
            t = self.t_all
        else:
            x_ind = np.argsort(t[:, 0])
            t = t[x_ind]
        dt = np.concatenate([np.array([0.0]), np.diff(t[:, 0])])
        N = dt.shape[0]
        with loops.Scope() as s:
            s.f_sample = np.zeros([N, self.func_dim, num_samps])
            s.m = np.linalg.cholesky(self.Pinf) @ random.normal(random.PRNGKey(99), shape=[self.state_dim, 1])
            for i in s.range(num_samps):
                s.m = np.linalg.cholesky(self.Pinf) @ random.normal(random.PRNGKey(i), shape=[self.state_dim, 1])
                for k in s.range(N):
                    A = self.prior.state_transition(dt[k], self.prior.hyp)  # transition and noise process matrices
                    Q = self.Pinf - A @ self.Pinf @ A.T
                    C = np.linalg.cholesky(Q + 1e-6 * np.eye(self.state_dim))  # <--- can be a bit unstable
                    # we need to provide a different PRNG seed every time:
                    s.m = A @ s.m + C @ random.normal(random.PRNGKey(i*k+k), shape=[self.state_dim, 1])
                    H = self.prior.measurement_model(t[k, 1:], softplus_list(self.prior.hyp))
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
        post_mean, _, (site_mean, site_cov), _ = self.predict(site_params=self.sites.site_params, compute_nlpd=False)
        prior_samp = self.prior_sample(num_samps, t=self.t_all)
        prior_samp_y = sample_gaussian_noise(prior_samp, site_cov)
        with loops.Scope() as ss:
            ss.smoothed_sample = np.zeros(prior_samp_y.shape)
            for i in ss.range(num_samps):
                smoothed_sample_i, _, _, _ = self.predict(np.zeros_like(prior_samp_y[..., i]), self.dt_all, self.mask,
                                                          (prior_samp_y[..., i], site_cov),
                                                          sampling=True, compute_nlpd=False)
                ss.smoothed_sample = index_add(ss.smoothed_sample, index[..., i], smoothed_sample_i[..., 0])
        return prior_samp - ss.smoothed_sample + post_mean
