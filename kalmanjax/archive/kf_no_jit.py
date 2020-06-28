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
    neg_log_marg_lik = 0.0  # negative log-marginal likelihood
    m, P = self.minf, self.Pinf
    if store:
        filtered_mean = np.zeros([N, self.state_dim, 1])
        filtered_cov = np.zeros([N, self.state_dim, self.state_dim])
        site_mean = np.zeros([N, self.func_dim, 1])
        site_var = np.zeros([N, self.func_dim, self.func_dim])
    for n in range(N):
        y_n = y[n]
        # -- KALMAN PREDICT --
        #  mₙ⁻ = Aₙ mₙ₋₁
        #  Pₙ⁻ = Aₙ Pₙ₋₁ Aₙ' + Qₙ, where Qₙ = Pinf - Aₙ Pinf Aₙ'
        A = self.prior.state_transition(dt[n], theta_prior)
        m_ = A @ m
        P_ = A @ (P - self.Pinf) @ A.T + self.Pinf
        # --- KALMAN UPDATE ---
        # Given previous predicted mean mₙ⁻ and cov Pₙ⁻, incorporate yₙ to get filtered mean mₙ &
        # cov Pₙ and compute the marginal likelihood p(yₙ|y₁,...,yₙ₋₁)
        H = self.prior.measurement_model(x[n, 1:], theta_prior)
        mu = H @ m_
        var = H @ P_ @ H.T
        if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
            y_n = np.where(mask[n], mu, y_n)  # fill in masked obs with prior expectation to prevent NaN grads
        log_lik_n, site_mu_, site_var_ = self.sites.update(self.likelihood, y_n, mu, var, theta_lik, None)
        if site_params is not None:  # use supplied site parameters to perform the update
            site_mu_, site_var_ = site_params[0][n], site_params[1][n]
        # modified Kalman update (see Nickish et. al. ICML 2018 or Wilkinson et. al. ICML 2019):
        S = var + site_var_
        K = solve(S, H @ P_).T  # HP(S^-1)
        m = m_ + K @ (site_mu_ - mu)
        P = P_ - K @ S @ K.T
        if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
            m = np.where(mask[n], m_, m)
            P = np.where(mask[n], P_, P)
            log_lik_n = np.where(mask[n][..., 0, 0], np.zeros_like(log_lik_n), log_lik_n)
        neg_log_marg_lik -= np.sum(log_lik_n)
        if store:
            filtered_mean = index_add(filtered_mean, index[n, ...], m)
            filtered_cov = index_add(filtered_cov, index[n, ...], P)
            site_mean = index_add(site_mean, index[n, ...], site_mu_)
            site_var = index_add(site_var, index[n, ...], site_var_)
    if store:
        return neg_log_marg_lik, (filtered_mean, filtered_cov, (site_mean, site_var))
    return neg_log_marg_lik
