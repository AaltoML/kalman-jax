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
    N = dt.shape[0]
    neg_log_marg_lik = 0.0  # negative log-marginal likelihood
    m, P = self.minf, self.Pinf
    if store:
        filtered_mean = np.zeros([N, self.state_dim, 1])
        filtered_cov = np.zeros([N, self.state_dim, self.state_dim])
        site_mean = np.zeros([N, self.func_dim, 1])
        site_var = np.zeros([N, self.func_dim, self.func_dim])
    for n in range(N):
        y_n = y[n][..., np.newaxis]
        # -- KALMAN PREDICT --
        #  mₙ⁻ = Aₙ mₙ₋₁
        #  Pₙ⁻ = Aₙ Pₙ₋₁ Aₙ' + Qₙ, where Qₙ = Pinf - Aₙ Pinf Aₙ'
        A = self.prior.state_transition(dt[n], theta_prior)
        m_ = A @ m
        P_ = A @ (P - self.Pinf) @ A.T + self.Pinf
        # --- KALMAN UPDATE ---
        # Given previous predicted mean mₙ⁻ and cov Pₙ⁻, incorporate yₙ to get filtered mean mₙ &
        # cov Pₙ and compute the marginal likelihood p(yₙ|y₁,...,yₙ₋₁)
        H = self.prior.measurement_model(r[n], theta_prior)
        mu = H @ m_
        var = H @ P_ @ H.T
        if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
            y_n = np.where(mask[n][..., np.newaxis], mu[:y_n.shape[0]],
                           y_n)  # fill in masked obs with prior expectation
        log_lik_n, site_mu_, site_var_ = self.sites.update(self.likelihood, y_n, mu, var, theta_lik, None)
        if site_params is not None:  # use supplied site parameters to perform the update
            site_mu_, site_var_ = site_params[0][n], site_params[1][n]
        # modified Kalman update (see Nickish et. al. ICML 2018 or Wilkinson et. al. ICML 2019):
        S = var + site_var_
        K = solve(S, H @ P_).T  # HP(S^-1)
        m = m_ + K @ (site_mu_ - mu)
        P = P_ - K @ S @ K.T
        if mask is not None:  # note: this is a bit redundant but may come in handy in multi-output problems
            m = np.where(np.any(mask[n]), m_, m)
            P = np.where(np.any(mask[n]), P_, P)
            log_lik_n = np.where(mask[n][..., 0], np.zeros_like(log_lik_n), log_lik_n)
        neg_log_marg_lik -= np.sum(log_lik_n)
        if store:
            filtered_mean = index_add(filtered_mean, index[n, ...], m)
            filtered_cov = index_add(filtered_cov, index[n, ...], P)
            site_mean = index_add(site_mean, index[n, ...], site_mu_)
            site_var = index_add(site_var, index[n, ...], site_var_)
    if store:
        return neg_log_marg_lik, (filtered_mean, filtered_cov, (site_mean, site_var))
    return neg_log_marg_lik


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
    m, P = m_filtered[-1, ...], P_filtered[-1, ...]
    if return_full:
        smoothed_mean = np.zeros([N, self.state_dim, 1])
        smoothed_cov = np.zeros([N, self.state_dim, self.state_dim])
    else:
        smoothed_mean = np.zeros([N, self.func_dim, 1])
        smoothed_cov = np.zeros([N, self.func_dim, self.func_dim])
    if site_params is not None:
        site_mean = np.zeros([N, self.func_dim, 1])
        site_var = np.zeros([N, self.func_dim, self.func_dim])
    for n in range(N-1, -1, -1):
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
        m = m_filtered[n, ...] + G_transpose.T @ (m - m_predicted)
        P = P_filtered[n, ...] + G_transpose.T @ (P - P_predicted) @ G_transpose
        H = self.prior.measurement_model(r[n], theta_prior)
        if store:
            if return_full:
                smoothed_mean = index_add(smoothed_mean, index[n, ...], m)
                smoothed_cov = index_add(smoothed_cov, index[n, ...], P)
            else:
                smoothed_mean = index_add(smoothed_mean, index[n, ...], H @ m)
                smoothed_cov = index_add(smoothed_cov, index[n, ...], H @ P @ H.T)
        # --- Now update the site parameters: ---
        if site_params is not None:
            # extract mean and var from state:
            post_mean, post_cov = H @ m, H @ P @ H.T
            # calculate the new sites
            _, site_mu, site_cov = self.sites.update(self.likelihood, y[n], post_mean, post_cov, theta_lik,
                                                     (site_params[0][n], site_params[1][n]))
            site_mean = index_add(site_mean, index[n, ...], site_mu)
            site_var = index_add(site_var, index[n, ...], site_cov)
    if site_params is not None:
        site_params = (site_mean, site_var)
    if store:
        return site_params, smoothed_mean, smoothed_cov
    return site_params