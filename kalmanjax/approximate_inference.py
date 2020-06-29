import jax.numpy as np
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.linalg import inv as inv_any
from utils import inv, symmetric_cubature_third_order, symmetric_cubature_fifth_order, gauss_hermite
pi = 3.141592653589793


def compute_cavity(post_mean, post_cov, site_mean, site_cov, power):
    """
    remove local likelihood approximation  from the posterior to obtain the marginal cavity distribution
    """
    post_precision, site_precision = inv(post_cov), inv(site_cov)
    cav_cov = inv(post_precision - power * site_precision)  # cavity covariance
    cav_mean = cav_cov @ (post_precision @ post_mean - power * site_precision @ site_mean)  # cavity mean
    return cav_mean, cav_cov


def ensure_positive_variance(K):
    K = np.where(np.any(np.diag(K) < 0), np.diag(np.diag(K)), K)
    K = np.where(K < 0, 99., K)
    return K


class ApproxInf(object):
    """
    The approximate inference class.
    Each approximate inference scheme implements an 'update' method which is called during
    filtering and smoothing in order to update the local likelihood approximation (the sites).
    See the paper for derivations of each update rule.
    """
    def __init__(self, site_params=None, intmethod='GH', num_cub_pts=20):
        self.site_params = site_params
        if intmethod == 'GH':
            self.cubature_func = lambda dim: gauss_hermite(dim, num_cub_pts)  # Gauss-Hermite
        elif intmethod == 'UT3':
            self.cubature_func = lambda dim: symmetric_cubature_third_order(dim)  # Unscented transform (3rd order)
        elif (intmethod == 'UT5') or (intmethod == 'UT'):
            self.cubature_func = lambda dim: symmetric_cubature_fifth_order(dim)  # Unscented transform (5th order)
        else:
            raise NotImplementedError('integration method not recognised')

    def update(self, likelihood, y, m, v, hyp=None, site_params=None):
        raise NotImplementedError('the update function for this approximate inference method is not implemented')


class ExpectationPropagation(ApproxInf):
    """
    Expectation propagation (EP)
    """
    def __init__(self, site_params=None, damping=1., power=1.0, intmethod='GH', num_cub_pts=20):
        self.damping = damping
        self.power = power
        super().__init__(site_params=site_params, intmethod=intmethod, num_cub_pts=num_cub_pts)
        self.name = 'Expectation Propagation (EP)'

    def update(self, likelihood, y, post_mean, post_cov, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses moment matching to update the site parameters
        """
        if site_params is None:
            # if no site is provided, use the predictions/posterior as the cavity with ep_fraction=1
            # calculate log marginal likelihood and the new sites via moment matching:
            lml, site_mean, site_cov = likelihood.moment_match(y, post_mean, post_cov, hyp, 1.0, self.cubature_func)
            site_cov = ensure_positive_variance(site_cov)
            return lml, site_mean, site_cov
        else:
            site_mean_prev, site_cov_prev = site_params  # previous site params
            # --- Compute the cavity distribution ---
            cav_mean, cav_cov = compute_cavity(post_mean, post_cov, site_mean_prev, site_cov_prev, self.power)
            # check that the cavity variances are positive
            cav_cov = ensure_positive_variance(cav_cov)
            # calculate log marginal likelihood and the new sites via moment matching:
            lml, site_mean, site_cov = likelihood.moment_match(y, cav_mean, cav_cov, hyp, self.power, self.cubature_func)
            site_cov = ensure_positive_variance(site_cov)
            site_nat2, site_nat2_prev = inv(site_cov), inv(site_cov_prev)
            site_nat1, site_nat1_prev = site_nat2 @ site_mean, site_nat2_prev @ site_mean_prev
            site_cov = inv((1. - self.damping) * site_nat2_prev + self.damping * site_nat2)
            site_mean = site_cov @ ((1. - self.damping) * site_nat1_prev + self.damping * site_nat1)
            return lml, site_mean, site_cov


class PowerExpectationPropagation(ExpectationPropagation):
    pass


class EP(ExpectationPropagation):
    pass


class PEP(ExpectationPropagation):
    pass


class ExtendedEP(ApproxInf):
    """
    Extended expectation propagation (EEP). This is equivalent to the extended Kalman smoother (EKS) but with
    linearisation applied at the cavity mean. Recovers the EKS when power=0.
    """
    def __init__(self, site_params=None, power=1.0):
        self.power = power
        super().__init__(site_params=site_params)
        self.name = 'Extended Expectation Propagation (EEP)'

    def update(self, likelihood, y, post_mean, post_cov, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses analytical linearisation (first
        order Taylor series expansion) to update the site parameters
        """
        power = 1. if site_params is None else self.power
        if (site_params is None) or (power == 0):  # avoid cavity calc if power is 0
            cav_mean, cav_cov = post_mean, post_cov
        else:
            site_mean, site_cov = site_params
            # --- Compute the cavity distribution ---
            cav_mean, cav_cov = compute_cavity(post_mean, post_cov, site_mean, site_cov, power)
        # calculate the Jacobian of the observation model w.r.t. function fₙ and noise term rₙ
        Jf, Jr = likelihood.analytical_linearisation(cav_mean, hyp)  # evaluated at the mean
        obs_cov = np.eye(y.shape[0])  # observation noise scale is w.l.o.g. 1
        likelihood_expectation, _ = likelihood.conditional_moments(cav_mean, hyp)
        residual = y - likelihood_expectation  # residual, yₙ-E[yₙ|fₙ]
        sigma = Jr @ obs_cov @ Jr.T + power * Jf @ cav_cov @ Jf.T
        site_cov = inv(Jf.T @ inv(Jr @ obs_cov @ Jr.T) @ Jf + 1e-10 * np.eye(Jf.shape[1]))
        site_mean = cav_mean + (site_cov + power * cav_cov) @ Jf.T @ inv(sigma) @ residual
        # now compute the marginal likelihood approx.
        sigma_marg_lik = Jr @ obs_cov @ Jr.T + Jf @ cav_cov @ Jf.T
        chol_sigma, low = cho_factor(sigma_marg_lik)
        log_marg_lik = -1 * (
                .5 * site_cov.shape[0] * np.log(2 * pi)
                + np.sum(np.log(np.diag(chol_sigma)))
                + .5 * (residual.T @ cho_solve((chol_sigma, low), residual)))
        return log_marg_lik, site_mean, site_cov


class EEP(ExtendedEP):
    pass


class ExtendedKalmanSmoother(ExtendedEP):
    """
    Extended Kalman smoother (EKS). Equivalent to EEP when power = 0.
    """
    def __init__(self, site_params=None):
        super().__init__(site_params=site_params, power=0)
        self.name = 'Extended Kalman Smoother (EKS)'


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
    An iterated smoothing algorithm based on statistical linear regression (SLR).
    This method linearises the likelihood model in the region described by the cavity.
    When the power is zero, we recover posterior linearisation, i.e. the statistically
    linearised Kalman smoother. Using Gauss-Hermite / Unscented transform for numerical
    integration results in Gauss-Hermite EP / Unscented EP.
    """
    def __init__(self, site_params=None, power=1.0, intmethod='GH', num_cub_pts=20):
        self.power = power
        super().__init__(site_params=site_params, intmethod=intmethod, num_cub_pts=num_cub_pts)
        self.name = 'Statistically Linearised Expectation Propagation (SLEP)'

    def update(self, likelihood, y, post_mean, post_cov, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses statistical linear
        regression (SLR) w.r.t. the cavity distribution to update the site parameters.
        """
        power = 1. if site_params is None else self.power
        # TODO: implement SLR approximate likelihood
        log_marg_lik, _, _ = likelihood.moment_match(y, post_mean, post_cov, hyp, 1.0, self.cubature_func)
        if (site_params is None) or (power == 0):
            cav_mean, cav_cov = post_mean, post_cov
        else:
            site_mean, site_cov = site_params
            # --- Compute the cavity distribution ---
            cav_mean, cav_cov = compute_cavity(post_mean, post_cov, site_mean, site_cov, power)
        # SLR gives a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ])
        mu, S, C, omega = likelihood.statistical_linear_regression(cav_mean, cav_cov, hyp, self.cubature_func)
        # convert to a Gaussian site (a function of fₙ):
        residual = y - mu
        sigma = S + (power - 1) * C.T @ inv(cav_cov) @ C
        osigo = inv(omega.T @ inv(sigma) @ omega + 1e-10 * np.eye(omega.shape[1]))
        site_mean = cav_mean + osigo @ omega.T @ inv(sigma) @ residual  # approx. likelihood (site) mean
        site_cov = -power * cav_cov + osigo  # approx. likelihood var.
        return log_marg_lik, site_mean, site_cov


class SLEP(StatisticallyLinearisedEP):
    pass


class GaussHermiteEP(StatisticallyLinearisedEP):
    def __init__(self, site_params=None, power=1, num_cub_pts=20):
        super().__init__(site_params=site_params, power=power, intmethod='GH', num_cub_pts=num_cub_pts)
        self.name = 'Gauss-Hermite Expectation Propagation (GHEP)'


class GHEP(GaussHermiteEP):
    pass


class GaussHermiteKalmanSmoother(GaussHermiteEP):
    def __init__(self, site_params=None, num_cub_pts=20):
        super().__init__(site_params=site_params, power=0, num_cub_pts=num_cub_pts)
        self.name = 'Gauss-Hermite Kalman Smoother (GHKS)'


class GHKS(GaussHermiteKalmanSmoother):
    pass


class UnscentedEP(StatisticallyLinearisedEP):
    def __init__(self, site_params=None, power=1):
        super().__init__(site_params=site_params, power=power, intmethod='UT')
        self.name = 'Unscented Expectation Propagation (UEP)'


class UEP(UnscentedEP):
    pass


class UnscentedKalmanSmoother(UnscentedEP):
    def __init__(self, site_params=None):
        super().__init__(site_params=site_params, power=0)
        self.name = 'Unscented Kalman Smoother (UKS)'


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
        self.name = 'Variational Inference (VI)'

    def update(self, likelihood, y, post_mean, post_cov, hyp=None, site_params=None):
        """
        The update function takes a likelihood as input, and uses CVI to update the site parameters
        """
        if site_params is None:
            _, dE_dm, dE_dv = likelihood.variational_expectation(y, post_mean, post_cov, hyp, self.cubature_func)
            dE_dm, dE_dv = np.atleast_2d(dE_dm), np.atleast_2d(dE_dv)
            site_cov = -0.5 * inv_any(dE_dv + 1e-10 * np.eye(dE_dv.shape[0]))
            site_mean = post_mean + site_cov @ dE_dm
            site_cov = ensure_positive_variance(site_cov)
        else:
            site_mean, site_cov = site_params
            log_marg_lik, dE_dm, dE_dv = likelihood.variational_expectation(y, post_mean, post_cov, hyp, self.cubature_func)
            dE_dm, dE_dv = np.atleast_2d(dE_dm), np.atleast_2d(dE_dv)
            # site_cov = -0.5 * inv_any(dE_dv + 1e-10 * np.eye(dE_dv.shape[0]))
            # site_mean = post_mean + site_cov @ dE_dm
            # site_cov = np.where(np.any(np.diag(site_cov) < 0), np.diag(np.diag(site_cov)), site_cov)
            # site_cov = np.where(site_cov < 0, 99., site_cov)

            lambda_t_2 = inv_any(site_cov + 1e-10 * np.eye(site_cov.shape[0]))
            lambda_t_1 = lambda_t_2 @ site_mean
            lambda_t_1 = (1 - self.damping) * lambda_t_1 + self.damping * (dE_dm - 2 * dE_dv @ post_mean)
            lambda_t_2 = (1 - self.damping) * lambda_t_2 + self.damping * (-2 * dE_dv)
            site_cov = inv_any(lambda_t_2 + 1e-10 * np.eye(site_cov.shape[0]))
            site_mean = site_cov @ lambda_t_1
            site_cov = ensure_positive_variance(site_cov)
        log_marg_lik, _, _ = likelihood.moment_match(y, post_mean, post_cov, hyp, 1.0, self.cubature_func)
        return log_marg_lik, site_mean, site_cov


class VI(VariationalInference):
    pass


class CVI(VariationalInference):
    pass
