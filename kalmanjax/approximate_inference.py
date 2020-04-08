

class EP(object):
    """
    Expectation propagation (EP)
    """
    def __init__(self, site_params=None, power=1.0):
        self.site_params = site_params
        self.power = power

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses moment matching to update the site parameters
        """
        if site_params is None:
            # if no site is provided, use the predictions/posterior as the cavity with ep_fraction=1
            return likelihood.moment_match(y, m, v, hyp, site_update, 1.0)  # calculate new sites via moment matching
        else:
            site_mean, site_var = site_params
            # --- Compute the cavity distribution ---
            # remove local likelihood approximation to obtain the marginal cavity distribution:
            var_cav = 1.0 / (1.0 / v - self.power / site_var)  # cavity variance
            mu_cav = var_cav * (m / v - self.power * site_mean / site_var)  # cav. mean
            # calculate the new sites via moment matching
            return likelihood.moment_match(y, mu_cav, var_cav, hyp, site_update, self.power)


class IKS(object):
    """
    Iterated Kalman smoother (IKS). This uses statistical linearisation to perform the updates.
    A single forward pass using this approximation is called the statistical linearisation filter (SLF),
    which itself is equivalent to the Unscented/Gauss-Hermite filter (UKF/GHKF), depending on the
    quadrature method used.
    """
    def __init__(self, site_params=None):
        self.site_params = site_params

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses statistical linearisation
        to update the site parameters
        """
        log_marg_lik = likelihood.moment_match(y, m, v, hyp, False, 1.0)
        if site_update:
            # SLR gives a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Var[yₙ|fₙ])
            A, b, omega = likelihood.statistical_linear_regression(m, v, hyp)
            # classical iterated smoothers, which are based on statistical linearisation (as opposed to SLR),
            # do not utilise the linearisation error Ω, distinguishing them from posterior linearisation.
            # convert to a Gaussian site in fₙ: sₙ(fₙ) = 𝓝(fₙ|(yₙ-b)/A,Var[yₙ|fₙ]/√A)
            site_mean = A ** -1 * (y - b)  # approx. likelihood (site) mean
            site_var = A ** -0.5 * likelihood.likelihood_variance(m, hyp)  # approx. likelihood variance
            return log_marg_lik, site_mean, site_var
        else:
            return log_marg_lik


class SLF(IKS):
    """
    Statistical linearisation filter (SLF)
    A single forward pass of the IKS is called the statistical linearisation filter.
    """
    pass


class GHKF(IKS):
    """
    Gauss-Hermite Kalman filter (GHKF)
    When Gauss-Hermite is used, the statistical linearisation filter (SLF) is equivalent to the GHKF
    """
    pass


class PL(object):
    """
    Posterior linearisation (PL)
    """
    def __init__(self, site_params=None):
        self.site_params = site_params

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses statistical linear
        regression (SLR) to update the site parameters
        """
        log_marg_lik = likelihood.moment_match(y, m, v, hyp, False, 1.0)
        if site_update:
            # SLR gives a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ])
            A, b, omega = likelihood.statistical_linear_regression(m, v, hyp)
            # convert to a Gaussian site in fₙ: sₙ(fₙ) = 𝓝(fₙ|(yₙ-b)/A,(Ω+Var[yₙ|fₙ])/√A)
            site_mean = A ** -1 * (y - b)  # approx. likelihood (site) mean
            site_var = A ** -0.5 * (omega + likelihood.likelihood_variance(m, hyp))  # approx. likelihood variance
            return log_marg_lik, site_mean, site_var
        else:
            return log_marg_lik


class PrL(PL):
    """
    A single forward pass of the PL filter is called the prior linearisation (PrL) filter
    """
    pass


class CL(object):
    """
    Cavity linearisation (CL) - a version of posterior linearisation that linearises w.r.t. the
    cavity distribution rather than the posterior. Reduces to PL when power=0.
    """
    def __init__(self, site_params=None, power=1.0):
        self.site_params = site_params
        self.power = power

    def update(self, likelihood, y, m, v, hyp=None, site_update=True, site_params=None):
        """
        The update function takes a likelihood as input, and uses statistical linear
        regression (SLR) w.r.t. the cavity distribution to update the site parameters
        """
        log_marg_lik = likelihood.moment_match(y, m, v, hyp, False, 1.0)
        if site_update:
            if site_params is None:
                mu_cav, var_cav = m, v
            else:
                site_mean, site_var = site_params
                # --- Compute the cavity distribution ---
                # remove local likelihood approximation to obtain the marginal cavity distribution:
                var_cav = 1.0 / (1.0 / v - self.power / site_var)  # cavity variance
                mu_cav = var_cav * (m / v - self.power * site_mean / site_var)  # cav. mean
            # SLR gives a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ])
            A, b, omega = likelihood.statistical_linear_regression(mu_cav, var_cav, hyp)
            # convert to a Gaussian site in fₙ: sₙ(fₙ) = 𝓝(fₙ|(yₙ-b)/A,(Ω+Var[yₙ|fₙ])/√A)
            site_mean = A ** -1 * (y - b)  # approx. likelihood (site) mean
            site_var = A ** -0.5 * (omega + likelihood.likelihood_variance(mu_cav, hyp))  # approx. likelihood var.
            return log_marg_lik, site_mean, site_var
        else:
            return log_marg_lik
