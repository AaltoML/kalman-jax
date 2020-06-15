import jax.numpy as np
from jax.scipy.linalg import cholesky
pi = 3.141592653589793


class ApproxInf(object):
    """
    The approximate inference class.
    Each approximate inference scheme implements an 'update' method which is called during
    filtering and smoothing in order to update the local likelihood approximation (the sites).
    """
    def __init__(self, site_params=None):
        self.site_params = site_params

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        raise NotImplementedError('the update function for this approximate inference method is not implemented')


class EP(ApproxInf):
    """
    Expectation propagation (EP)
    """
    def __init__(self, site_params=None, power=1.0):
        self.power = power
        super().__init__(site_params=site_params)
        self.name = 'expectation propagation (EP)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses moment matching to update the site parameters
        """
        if site_params is None:
            # if no site is provided, use the predictions/posterior as the cavity with ep_fraction=1
            return likelihood.moment_match(y, m, v, hyp, 1.0)  # calculate new sites via moment matching
        else:
            site_mean, site_var = site_params
            # --- Compute the cavity distribution ---
            # remove local likelihood approximation to obtain the marginal cavity distribution:
            var_cav = 1.0 / (1.0 / v - self.power / site_var)  # cavity variance
            mu_cav = var_cav * (m / v - self.power * site_mean / site_var)  # cav. mean
            # calculate the new sites via moment matching
            return likelihood.moment_match(y, mu_cav, var_cav, hyp, self.power)


class PEP(EP):
    """
    Power expectation propagation (PEP)
    """
    pass


class EKEP(ApproxInf):
    """
    Extended Kalman expectation propagation (EK-EP)
    """
    def __init__(self, site_params=None, power=1.0):
        self.power = power
        super().__init__(site_params=site_params)
        self.name = 'extended Kalman expectation propagation (EK-EP)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses analytical linearisation
        to update the site parameters
        """
        if site_params is None:
            mu_cav, var_cav = m, v
        else:
            site_mean, site_var = site_params
            # --- Compute the cavity distribution ---
            # remove local likelihood approximation to obtain the marginal cavity distribution:
            var_cav = 1.0 / (1.0 / v - self.power / site_var)  # cavity variance
            mu_cav = var_cav * (m / v - self.power * site_mean / site_var)  # cav. mean
        # calculate the Jacobian of the observation model w.r.t. function fₙ and noise term rₙ
        Jf, Jr = likelihood.analytical_linearisation(mu_cav, hyp)  # evaluated at the mean
        var_obs = 1.0  # observation noise scale is w.l.o.g. 1
        likelihood_expectation, _ = likelihood.conditional_moments(mu_cav, hyp)
        residual = y - likelihood_expectation  # residual, yₙ-E[yₙ|fₙ]
        sigma = Jr * var_obs * Jr + self.power * Jf * var_cav * Jf
        site_var = (Jf * (Jr * var_obs * Jr) ** -1 * Jf) ** -1
        site_mean = m + (site_var + self.power * var_cav) * Jf * sigma**-1 * residual
        # now compute the marginal likelihood approx.
        chol_site_var = cholesky(site_var, lower=True)
        log_marg_lik = -1 * (
                .5 * site_var.shape[0] * np.log(2 * pi)
                + np.sum(np.log(np.diag(chol_site_var)))
                + .5 * (residual * site_var**-1 * residual)
        )
        return log_marg_lik, site_mean, site_var


class EKS(ApproxInf):
    """
    Extended Kalman smoother (EKS)
    """
    def __init__(self, site_params=None):
        super().__init__(site_params=site_params)
        self.name = 'extended Kalman smoother (EKS)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses analytical linearisation
        to update the site parameters
        """
        # calculate the Jacobian of the observation model w.r.t. function fₙ and noise term rₙ
        Jf, Jr = likelihood.analytical_linearisation(m, hyp)  # evaluated at the mean
        var_obs = 1.0  # observation noise scale is w.l.o.g. 1
        likelihood_expectation, _ = likelihood.conditional_moments(m, hyp)
        residual = y - likelihood_expectation  # residual, yₙ-E[yₙ|fₙ]
        sigma = Jr * var_obs * Jr  # + Jf * v * Jf
        site_var = (Jf * sigma ** -1 * Jf) ** -1
        site_mean = m + (site_var + v) * Jf * (sigma + Jf * v * Jf) ** -1 * residual
        # now compute the marginal likelihood approx.
        chol_site_var = cholesky(site_var, lower=True)
        log_marg_lik = -1 * (
                .5 * site_var.shape[0] * np.log(2 * pi)
                + np.sum(np.log(np.diag(chol_site_var)))
                + .5 * (residual * site_var ** -1 * residual)  # TODO: use cholesky for inverse in multi-dim case
        )
        return log_marg_lik, site_mean, site_var


class EKF(EKS):
    """
    Extended Kalman filter (EKF)
    A single forward pass of the EKS.
    """
    # TODO: make is so that when EKF/UKF/SLF/GHKF are chosen, model.run_model() only performs filtering.
    pass


class IKS(ApproxInf):
    """
    Iterated Kalman smoother (IKS). This uses statistical linearisation to perform the updates.
    A single forward pass using this approximation is called the statistical linearisation filter (SLF),
    which itself is equivalent to the Unscented/Gauss-Hermite filter (UKF/GHKF), depending on the
    quadrature method used.
    """
    def __init__(self, site_params=None):
        super().__init__(site_params=site_params)
        self.name = 'iterated Kalman smoother (IKS)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses statistical linearisation
        to update the site parameters
        """
        log_marg_lik, _, _ = likelihood.moment_match(y, m, v, hyp, 1.0)
        # SLR gives a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Var[yₙ|fₙ])
        A, b, omega = likelihood.statistical_linear_regression(m, v, hyp)
        # classical iterated smoothers, which are based on statistical linearisation (as opposed to SLR),
        # do not utilise the linearisation error Ω, distinguishing them from posterior linearisation.
        # convert to a Gaussian site in fₙ: sₙ(fₙ) = 𝓝(fₙ|(yₙ-b)/A,Var[yₙ|fₙ]/√A)
        # TODO: implement case where A is not invertable
        site_mean = A ** -1 * (y - b)  # approx. likelihood (site) mean
        _, likelihood_variance = likelihood.conditional_moments(m, hyp)
        site_var = A ** -0.5 * likelihood_variance  # approx. likelihood variance
        return log_marg_lik, site_mean, site_var


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


class PL(ApproxInf):
    """
    Posterior linearisation (PL)
    An iterated smoothing algorithm based on statistical linear regression (SLR) w.r.t. the approximate posterior.
    This is a special case of cavity linearisation, where power = 0.
    """
    def __init__(self, site_params=None):
        super().__init__(site_params=site_params)
        self.name = 'posterior linearisation (PL)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses statistical linear
        regression (SLR) to update the site parameters
        """
        # TODO: implement PL approximate likelihood
        log_marg_lik, _, _ = likelihood.moment_match(y, m, v, hyp, 1.0)
        # SLR gives a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ])
        A, b, omega = likelihood.statistical_linear_regression(m, v, hyp)
        # convert to a Gaussian site in fₙ: sₙ(fₙ) = 𝓝(fₙ|(yₙ-b)/A,(Ω+Var[yₙ|fₙ])/√A)
        # TODO: implement case where A is not invertable
        site_mean = A ** -1 * (y - b)  # approx. likelihood (site) mean
        _, likelihood_variance = likelihood.conditional_moments(m, hyp)
        site_var = A ** -0.5 * (omega + likelihood_variance)  # approx. likelihood variance
        return log_marg_lik, site_mean, site_var


class CL(ApproxInf):
    """
    Cavity linearisation (CL) - a version of posterior linearisation that linearises w.r.t. the
    cavity distribution rather than the posterior. Reduces to PL when power = 0.
    """
    def __init__(self, site_params=None, power=1.0):
        self.power = power
        super().__init__(site_params=site_params)
        self.name = 'cavity linearisation (CL)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses statistical linear
        regression (SLR) w.r.t. the cavity distribution to update the site parameters.
        """
        # TODO: implement CL approximate likelihood
        log_marg_lik, _, _ = likelihood.moment_match(y, m, v, hyp, 1.0)
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
        # TODO: implement case where A is not invertable
        site_mean = A ** -1 * (y - b)  # approx. likelihood (site) mean
        _, likelihood_variance = likelihood.conditional_moments(mu_cav, hyp)
        site_var = A ** -0.5 * (omega + likelihood_variance)  # approx. likelihood var.
        return log_marg_lik, site_mean, site_var


# class PL(CL):
#     """
#     Posterior linearisation (PL)
#     This is a special case of cavity linearisation, where the power = 0.
#     """
#     def __init__(self, site_params=None):
#         super().__init__(site_params=site_params, power=0.0)


class PrL(PL):
    """
    A single forward pass of the PL filter is called the prior linearisation (PrL) filter
    """
    pass


class VI(ApproxInf):
    """
    Natural gradient VI (using the conjugate-computation VI approach)
    Refs:
        Khan & Lin 2017 "Conugate-computation variational inference - converting inference
                         in non-conjugate models in to inference in conjugate models"
        Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
    """
    def __init__(self, site_params=None, damping=1.):
        self.damping = damping
        super().__init__(site_params=site_params)
        self.name = 'variational inference (VI)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses CVI to update the site parameters
        """
        if site_params is None:
            _, dE_dm, dE_dv = likelihood.variational_expectation(y, m, v, hyp)
            site_var = -0.5 / dE_dv
            site_mean = m + dE_dm * site_var
        else:
            site_mean, site_var = site_params
            log_marg_lik, dE_dm, dE_dv = likelihood.variational_expectation(y, m, v, hyp)
            lambda_t_1 = site_mean / site_var
            lambda_t_2 = 1 / site_var
            lambda_t_1 = (1 - self.damping) * lambda_t_1 + self.damping * (dE_dm - 2 * dE_dv * m)
            lambda_t_2 = (1 - self.damping) * lambda_t_2 + self.damping * (-2 * dE_dv)
            site_mean = lambda_t_1 / lambda_t_2
            site_var = np.abs(1 / lambda_t_2)
        log_marg_lik, _, _ = likelihood.moment_match(y, m, v, hyp, 1.0)
        return log_marg_lik, site_mean, site_var
