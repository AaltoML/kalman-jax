import jax.numpy as np
from jax.scipy.linalg import cho_factor, cho_solve
from utils import symmetric_cubature_third_order, symmetric_cubature_fifth_order, gauss_hermite
pi = 3.141592653589793


def compute_cavity(m_post, v_post, m_site, v_site, power):
    """
    remove local likelihood approximation  from the posterior to obtain the marginal cavity distribution
    """
    var_cav = 1.0 / (1.0 / v_post - power / v_site)  # cavity variance
    mu_cav = var_cav * (m_post / v_post - power * m_site / v_site)  # cav. mean
    return mu_cav, var_cav


class ApproxInf(object):
    """
    The approximate inference class.
    Each approximate inference scheme implements an 'update' method which is called during
    filtering and smoothing in order to update the local likelihood approximation (the sites).
    """
    def __init__(self, site_params=None, intmethod='GH', num_cub_pts=20):
        self.site_params = site_params
        if intmethod == 'GH':
            self.cubature_func = lambda dim: gauss_hermite(dim, num_cub_pts)  # Gauss-Hermite
        elif intmethod == 'UT3':
            self.cubature_func = lambda dim: symmetric_cubature_third_order(dim)  # Unscented transform
        elif (intmethod == 'UT5') or (intmethod == 'UT'):
            self.cubature_func = lambda dim: symmetric_cubature_fifth_order(dim)  # Unscented transform
        else:
            raise NotImplementedError('integration method not recognised')

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        raise NotImplementedError('the update function for this approximate inference method is not implemented')


class ExpectationPropagation(ApproxInf):
    """
    Expectation propagation (EP)
    """
    def __init__(self, site_params=None, power=1.0, intmethod='GH', num_cub_pts=20):
        self.power = power
        super().__init__(site_params=site_params, intmethod=intmethod, num_cub_pts=num_cub_pts)
        self.name = 'expectation propagation (EP)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses moment matching to update the site parameters
        """
        if site_params is None:
            # if no site is provided, use the predictions/posterior as the cavity with ep_fraction=1
            # calculate log marginal likelihood and the new sites via moment matching:
            lml, site_mean, site_var = likelihood.moment_match(y, m, v, hyp, 1.0, self.cubature_func)
            return lml, site_mean, site_var
        else:
            site_mean_prev, site_var_prev = site_params  # previous site params
            # --- Compute the cavity distribution ---
            mu_cav, var_cav = compute_cavity(m, v, site_mean_prev, site_var_prev, self.power)
            # check that the cavity variances are positive
            var_cav = np.where(var_cav > 0, var_cav, 999.)
            # calculate log marginal likelihood and the new sites via moment matching:
            lml, site_mean, site_var = likelihood.moment_match(y, mu_cav, var_cav, hyp, self.power, self.cubature_func)
            # don't update entries whose site variance is not positive
            site_mean = np.where(site_var > 0, site_mean, site_mean_prev)
            site_var = np.where(site_var > 0, site_var, site_var_prev)
            return lml, site_mean, site_var


class PowerExpectationPropagation(ExpectationPropagation):
    pass


class EP(ExpectationPropagation):
    pass


class PEP(ExpectationPropagation):
    pass


class ExtendedEP(ApproxInf):
    """
    Extended expectation propagation (EEP). This is equivalent to the extended Kalman smoother (EKS) but with
    linearisation applied about the cavity mean. Recovers the EKS when power=0.
    """
    def __init__(self, site_params=None, power=1.0):
        self.power = power
        super().__init__(site_params=site_params)
        self.name = 'extended expectation propagation (EEP)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses analytical linearisation (first
        order Taylor series expansion) to update the site parameters
        """
        power = 1. if site_params is None else self.power
        if (site_params is None) or (power == 0):  # avoid cavity calc if power is 0
            mu_cav, var_cav = m, v
        else:
            site_mean, site_var = site_params
            # --- Compute the cavity distribution ---
            mu_cav, var_cav = compute_cavity(m, v, site_mean, site_var, power)
        # calculate the Jacobian of the observation model w.r.t. function fₙ and noise term rₙ
        Jf, Jr = likelihood.analytical_linearisation(mu_cav, hyp)  # evaluated at the mean
        var_obs = np.array([[1.0]])  # observation noise scale is w.l.o.g. 1
        likelihood_expectation, _ = likelihood.conditional_moments(mu_cav, hyp)
        residual = y - likelihood_expectation  # residual, yₙ-E[yₙ|fₙ]
        sigma = Jr * var_obs * Jr + power * Jf * var_cav * Jf
        site_var = (Jf * (Jr * var_obs * Jr) ** -1 * Jf) ** -1
        site_mean = mu_cav + (site_var + power * var_cav) * Jf * sigma**-1 * residual
        # now compute the marginal likelihood approx.
        chol_sigma, low = cho_factor(sigma)
        log_marg_lik = -1 * (
                .5 * site_var.shape[0] * np.log(2 * pi)
                + np.sum(np.log(np.diag(chol_sigma)))
                + .5 * (residual.T @ cho_solve((chol_sigma, low), residual)))
        return log_marg_lik, site_mean, site_var


class EEP(ExtendedEP):
    pass


class ExtendedKalmanSmoother(ExtendedEP):
    """
    Extended Kalman smoother (EKS). Equivalent to EEP when power = 0.
    """
    def __init__(self, site_params=None):
        super().__init__(site_params=site_params, power=0)
        self.name = 'extended Kalman smoother (EKS)'


class EKS(ExtendedKalmanSmoother):
    pass


class ExtendedKalmanFilter(ExtendedKalmanSmoother):
    """
    Extended Kalman filter (EKF)
    A single forward pass of the EKS.
    """
    # TODO: make is so that when EKF/UKF/SLF/GHKF are chosen, model.run_model() only performs filtering.
    pass


class EKF(ExtendedKalmanFilter):
    pass


class StatisticallyLinearisedEP(ApproxInf):
    """
    An iterated smoothing algorithm based on statistical linear regression (SLR) w.r.t. the cavity.
    """
    def __init__(self, site_params=None, power=1.0, intmethod='GH', num_cub_pts=20):
        self.power = power
        super().__init__(site_params=site_params, intmethod=intmethod, num_cub_pts=num_cub_pts)
        self.name = 'statistically linearised expectation propagation (SLEP)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses statistical linear
        regression (SLR) w.r.t. the cavity distribution to update the site parameters.
        """
        power = 1. if site_params is None else self.power
        # TODO: implement SLR approximate likelihood
        log_marg_lik, _, _ = likelihood.moment_match(y, m, v, hyp, 1.0, self.cubature_func)
        if (site_params is None) or (power == 0):
            mu_cav, var_cav = m, v
        else:
            site_mean, site_var = site_params
            # --- Compute the cavity distribution ---
            mu_cav, var_cav = compute_cavity(m, v, site_mean, site_var, power)
        # SLR gives a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ])
        mu, S, C, omega = likelihood.statistical_linear_regression(mu_cav, var_cav, hyp, self.cubature_func)
        # convert to a Gaussian site in fₙ: sₙ(fₙ) = 𝓝(fₙ|(yₙ-b)/A,(Ω+Var[yₙ|fₙ])/√A)
        residual = y - mu
        sigma = S + (power - 1) * C * var_cav ** -1 * C
        osigo = (omega * sigma ** -1 * omega) ** -1
        site_mean = mu_cav + osigo * omega * sigma ** -1 * residual  # approx. likelihood (site) mean
        site_var = -power * var_cav + osigo  # approx. likelihood var.
        return log_marg_lik, site_mean, site_var


class SLEP(StatisticallyLinearisedEP):
    pass


class GaussHermiteEP(StatisticallyLinearisedEP):
    def __init__(self, site_params=None, power=1, num_cub_pts=20):
        super().__init__(site_params=site_params, power=power, intmethod='GH', num_cub_pts=num_cub_pts)
        self.name = 'Gauss-Hermite expectation propagation (GHEP'


class GHEP(GaussHermiteEP):
    pass


class GaussHermiteKalmanSmoother(GaussHermiteEP):
    def __init__(self, site_params=None, num_cub_pts=20):
        super().__init__(site_params=site_params, power=0, num_cub_pts=num_cub_pts)
        self.name = 'Gauss-Hermite Kalman smoother'


class GHKS(GaussHermiteKalmanSmoother):
    pass


class UnscentedEP(StatisticallyLinearisedEP):
    def __init__(self, site_params=None, power=1):
        super().__init__(site_params=site_params, power=power, intmethod='UT')
        self.name = 'Unscented expectation propagation (UEP)'


class UEP(UnscentedEP):
    pass


class UnscentedKalmanSmoother(UnscentedEP):
    def __init__(self, site_params=None):
        super().__init__(site_params=site_params, power=0)
        self.name = 'Unscented Kalman smoother (UKS)'


class UKS(UnscentedKalmanSmoother):
    pass


class PosteriorLinearisation(StatisticallyLinearisedEP):
    pass


class PL(PosteriorLinearisation):
    pass


class GaussHermiteKalmanFilter(GaussHermiteKalmanSmoother):
    """
    Gauss-Hermite Kalman filter (GHKF)
    When Gauss-Hermite is used, the statistical linearisation filter (SLF) is equivalent to the GHKF
    """
    pass


class GHKF(GaussHermiteKalmanFilter):
    pass


class UnscentedKalmanFilter(UnscentedKalmanSmoother):
    """
    Unscented Kalman filter (UKF)
    When the Unscented transform is used, the statistical linearisation filter (SLF) is equivalent to the UKF
    """
    pass


class UKF(UnscentedKalmanFilter):
    pass


class VariationalInference(ApproxInf):
    """
    Natural gradient VI (using the conjugate-computation VI approach)
    Refs:
        Khan & Lin 2017 "Conugate-computation variational inference - converting inference
                         in non-conjugate models in to inference in conjugate models"
        Chang, Wilkinson, Khan & Solin 2020 "Fast variational learning in state space Gaussian process models"
    """
    def __init__(self, site_params=None, damping=1., intmethod='GH', num_cub_pts=20):
        self.damping = damping
        super().__init__(site_params=site_params, intmethod=intmethod, num_cub_pts=num_cub_pts)
        self.name = 'variational inference (VI)'

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses CVI to update the site parameters
        """
        if site_params is None:
            _, dE_dm, dE_dv = likelihood.variational_expectation(y, m, v, hyp, self.cubature_func)
            site_var = -0.5 / dE_dv
            site_mean = m + dE_dm * site_var
        else:
            site_mean, site_var = site_params
            log_marg_lik, dE_dm, dE_dv = likelihood.variational_expectation(y, m, v, hyp, self.cubature_func)
            lambda_t_1 = site_mean / site_var
            lambda_t_2 = 1 / site_var
            lambda_t_1 = (1 - self.damping) * lambda_t_1 + self.damping * (dE_dm - 2 * dE_dv * m)
            lambda_t_2 = (1 - self.damping) * lambda_t_2 + self.damping * (-2 * dE_dv)
            site_mean = lambda_t_1 / lambda_t_2
            site_var = np.abs(1 / lambda_t_2)
        log_marg_lik, _, _ = likelihood.moment_match(y, m, v, hyp, 1.0, self.cubature_func)
        return log_marg_lik, site_mean, site_var


class VI(VariationalInference):
    pass


class CVI(VariationalInference):
    pass
