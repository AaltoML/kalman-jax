import jax.numpy as np
from jax.scipy.special import erf, erfc, gammaln
from jax.nn import softplus
from jax import jit
# from numpy import random as nprandom
from jax import random
pi = 3.141592653589793


class Likelihood(object):
    """
    The likelihood model class, p(yₙ|fₙ). Each likelihood implements its own moment matching method,
    which calculates the log partition function, logZₙ, and its derivatives w.r.t. the cavity mean mₙ,
        logZₙ = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[p(yₙ|fₙ)]
    If no custom moment matching method is provided, Gauss-Hermite quadrature is used by default.
    The requirement for quadrature is simply a method called evaluate_likelihood(), which computes
    the likelihood model p(yₙ|fₙ) for given data and function values
    """
    def __init__(self, hyp=None):
        """
        :param hyp: (hyper)parameters of the likelihood model
        """
        self.hyp = hyp

    def evaluate_likelihood(self, y, f):
        raise NotImplementedError('direct evaluation of this likelihood is not implemented')

    def evaluate_log_likelihood(self, y, f):
        raise NotImplementedError('direct evaluation of this log-likelihood is not implemented')

    # def moment_match(self, y, mu, var, var_y, derivatives=True):
    @staticmethod
    def moment_match(y, mu, var, hyp, derivatives=True):
        """
        Moment matching invloves computing the log partition function, logZₙ, and its derivatives w.r.t. the cavity mean
            logZₙ = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[p(yₙ|fₙ)]
        which is equivalent to the expectation w.r.t the cavity
        """
        raise NotImplementedError('moment matching not implemented for this likelihood')

    @staticmethod
    def link_fn(latent_mean):
        return latent_mean

    @staticmethod
    @jit
    def sample_noise(latent_mean, likelihood_var):
        lik_std = np.sqrt(likelihood_var)
        # gaussian_sample = latent_mean + lik_std[..., np.newaxis] * nprandom.normal(size=latent_mean.shape)
        gaussian_sample = latent_mean + lik_std[..., np.newaxis] * random.normal(random.PRNGKey(123),
                                                                                 shape=latent_mean.shape)
        return gaussian_sample


class Gaussian(Likelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """
    def __init__(self, hyp):
        """
        :param hyp: The observation noise variance, σ²
        """
        super().__init__(hyp=hyp)
        if self.hyp is None:
            print('using default likelihood parameter since none was supplied')
            self.hyp = -2.25  # softplus(-2.25) ~= 0.1

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the Gaussian function 𝓝(yₙ|fₙ,σ²)
        Can be used to evaluate Q quadrature points
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [Q, 1]
        """
        return (2 * pi * self.hyp) ** -0.5 * np.exp(-0.5 * (y - f) ** 2 / self.hyp)

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²)
        Can be used to evaluate Q quadrature points
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [Q, 1]
        """
        return -0.5 * np.log(2 * pi * self.hyp) - 0.5 * (y - f) ** 2 / self.hyp

    @staticmethod
    @jit
    def moment_match(y, m, v, hyp, derivatives=True):
        """
        Closed form Gaussian moment matching.
        Calculates the log partition function of the EP tilted distribution:
            logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
        and its derivatives w.r.t. mₙ, which are required for moment matching.
        :param y: observed data (yₙ) [scalar]
        :param m: cavity mean (mₙ) [scalar]
        :param v: cavity variance (vₙ) [scalar]
        :param hyp: observation noise variance (σ²) [scalar]
        :param derivatives: if True, return the derivatives of the log partition function w.r.t. mₙ [bool]
        :return:
            lZ: the log partition function, logZₙ [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        """
        # log partition function, lZ:
        # logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #       = log 𝓝(yₙ|mₙ,σ²+vₙ)
        lZ = (
                - (y - m) ** 2 / (hyp + v) / 2
                - np.log(np.maximum(2 * pi * (hyp + v), 1e-10)) / 2
        )
        if derivatives:
            # dlogZₙ/dmₙ = (yₙ - mₙ)(σ² + vₙ)⁻¹
            dlZ = (y - m) / (hyp + v)  # 1st derivative w.r.t. mean
            # d²logZₙ/dmₙ² = -(σ² + vₙ)⁻¹
            d2lZ = -1 / (hyp + v)  # 2nd derivative w.r.t. mean
            return lZ, dlZ, d2lZ
        else:
            return lZ


class Probit(Likelihood):
    """
    The Probit Binary Classification likelihood, i.e. the Error Function Likelihood,
    i.e. the Gaussian (Normal) cumulative density function:
        p(yₙ|fₙ) = Φ(yₙfₙ)
                 = ∫ 𝓝(x|0,1) dx, where the integral is over (-∞, fₙyₙ],
    and where we force the data to be +/-1: yₙ ϵ {-1, +1}
    The Normal CDF is calulcated using the error function:
        Φ(yₙfₙ) = (1 + erf(yₙfₙ / √2)) / 2
    for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    """
    def __init__(self, hyp):
        """
        :param hyp: None. This likelihood model has no hyperparameters
        """
        super().__init__(hyp=hyp)

    @staticmethod
    @jit
    def link_fn(latent_mean):
        return erfc(-latent_mean / np.sqrt(2.0)) - 1.0

    def eval(self, mu, var):
        """
        ported from GPML toolbox - not used
        """
        lp, _, _ = self.moment_match(1, mu, var)
        p = np.exp(lp)
        ymu = 2 * p - 1
        yvar = 4 * p * (1 - p)
        return lp, ymu, yvar

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the Gaussian CDF likelihood model,
            Φ(yₙfₙ) = (1 + erf(yₙfₙ / √2)) / 2
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
        Can be used to evaluate Q quadrature points when performing moment matching
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ [Q, 1]
        :return:
            Φ(yₙfₙ) [Q, 1]
        """
        return (1.0 + erf(y * f / np.sqrt(2.0))) / 2.0  # Φ(z)

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the Gaussian CDF log-likelihood,
            log Φ(yₙfₙ) = log[(1 + erf(yₙfₙ / √2)) / 2]
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
        Can be used to evaluate Q quadrature points when performing moment matching
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ [Q, 1]
        :return:
            log Φ(yₙfₙ) [Q, 1]
        """
        return np.log(1.0 + erf(y * f / np.sqrt(2.0)) + 1e-10) - np.log(2)  # logΦ(z)

    @staticmethod
    @jit
    def moment_match(y, m, v, hyp=None, derivatives=True):
        """
        Probit likelihood moment matching.
        Calculates the log partition function of the EP tilted distribution:
            logZₙ = log ∫ Φ(yₙfₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
                  = log Φ(yₙzₙ), where zₙ = mₙ / √(1 + vₙ)   [see Rasmussen & Williams p74]
        and its derivatives w.r.t. mₙ, which are required for moment matching.
        Note: we enforce yₙ ϵ {-1, +1}
        :param y: observed data (yₙ) [scalar]
        :param m: cavity mean (mₙ) [scalar]
        :param v: cavity variance (vₙ) [scalar]
        :param hyp: dummy variable (Probit has no hyperparameters)
        :param derivatives: if True, return the derivatives of the log partition function w.r.t. mₙ [bool]
        :return:
            lZ: the log partition function, logZₙ [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        """
        y = np.sign(y)  # only allow values of +/-1
        # y[np.where(y == 0)] = -1  # set zeros to -1
        y = np.sign(y - 0.01)  # set zeros to -1
        z = m / np.sqrt(1.0 + v)
        z = z * y  # zₙ = yₙmₙ / √(1 + vₙ)
        # logZₙ = log ∫ Φ(yₙfₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #       = log Φ(yₙmₙ/√(1 + vₙ))  [see Rasmussen & Williams p74]
        lZ, dlp = logphi(z)
        if derivatives:
            # dlogZₙ/dmₙ = yₙ dlogΦ(zₙ)/dmₙ / √(1 + vₙ)
            dlZ = y * dlp / np.sqrt(1.0 + v)  # first derivative w.r.t mₙ
            # d²logZₙ/dmₙ² = -dlogΦ(zₙ)/dmₙ (zₙ + dlogΦ(zₙ)/dmₙ) / √(1 + vₙ)
            d2lZ = -dlp * (z + dlp) / (1.0 + v)  # second derivative w.r.t mₙ
            return lZ, dlZ, d2lZ
        else:
            return lZ


class Erf(Probit):
    pass


class Poisson(Likelihood):
    """
    TODO: not JIT'ed and not tested, just here for an example of how quadrature can be implemented
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using quadrature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))            / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self, hyp=None, link='exp'):
        """
        :param hyp: None. This likelihood model has no hyperparameters
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(hyp=hyp)
        if link is 'exp':
            self.link = lambda mu: np.exp(mu)
        elif link is 'logistic':
            self.link = lambda mu: np.log(1.0 + np.exp(mu))
        else:
            raise NotImplementedError('link function not implemented')

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the Poisson likelihood:
            p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
        for μ = g(fₙ), where g() is the link function (exponential or logisitc)
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1)
        Can be used to evaluate Q quadrature points when performing moment matching
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            Poisson(fₙ) = μʸ exp(-μ) / yₙ! [Q, 1]
        """
        mu = self.link(f)
        return mu**y * np.exp(-mu) / np.exp(gammaln(y + 1))

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logisitc)
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1)
        Can be used to evaluate Q quadrature points when performing moment matching
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = self.link(f)
        return y * np.log(mu) - mu - gammaln(y + 1)

    def moment_match(self, y, m, v, hyp=None, derivatives=True, num_quad_points=20):
        """
        Perform moment matching via Gauss-Hermite quadrature
        Moment matching invloves computing the log partition function, logZₙ, and its derivatives w.r.t. the cavity mean
            logZₙ = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[p(yₙ|fₙ)]
        :param y: observed data (yₙ) [scalar]
        :param m: cavity mean (mₙ) [scalar]
        :param v: cavity variance (vₙ) [scalar]
        :param hyp: dummy variable
        :param derivatives: if True, return the derivatives of the log partition function w.r.t. mₙ [bool]
        :param num_quad_points: the number of Gauss-Hermite sigma points to use during quadrature [scalar]
        :return:
            lZ: the log partition function, logZₙ  [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
        """
        x, w = np.polynomial.hermite.hermgauss(num_quad_points)  # Gauss-Hermite sigma points and weights
        w = w / np.sqrt(pi)  # scale weights by 1/√π
        sigma_points = np.sqrt(2) * np.sqrt(v) * x + m  # scale locations according to cavity dist.
        # pre-compute wᵢ p(yₙ|xᵢ√(2vₙ) + mₙ)
        weighted_likelihood_eval = w * self.evaluate_likelihood(y, sigma_points)

        # a different approach, based on the log-likelihood, which can be more stable:
        # ll = self.evaluate_log_likelihood(y, sigma_points)
        # lmax = np.max(ll)
        # weighted_likelihood_eval = np.exp(lmax) * w * np.exp(ll - lmax)

        # Compute partition function via quadrature:
        # Zₙ = ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ p(yₙ|xᵢ√(2vₙ) + mₙ)
        Z = np.sum(
            weighted_likelihood_eval
        )
        lZ = np.log(Z)
        if derivatives:
            Zinv = 1.0 / Z
            # Compute derivative of partition function via quadrature:
            # dZₙ/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
            #         ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ p(yₙ|xᵢ√(2vₙ) + mₙ)
            dZ = np.sum(
                (sigma_points - m) / v
                * weighted_likelihood_eval
            )
            # dlogZₙ/dmₙ = (dZₙ/dmₙ) / Zₙ
            dlZ = Zinv * dZ
            # Compute second derivative of partition function via quadrature:
            # d²Zₙ/dmₙ² = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
            #           ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] p(yₙ|xᵢ√(2vₙ) + mₙ)
            d2Z = np.sum(
                ((sigma_points - m)**2 / v**2 - 1.0 / v)
                * weighted_likelihood_eval
            )
            # d²logZₙ/dmₙ² = d[(dZₙ/dmₙ) / Zₙ]/dmₙ
            #              = (d²Zₙ/dmₙ² * Zₙ - (dZₙ/dmₙ)²) / Zₙ²
            #              = d²Zₙ/dmₙ² / Zₙ - (dlogZₙ/dmₙ)²
            d2lZ = -dlZ**2 + Zinv * d2Z
            return lZ, dlZ, d2lZ
        else:
            return lZ


def logphi(z):
    """
    Calculate the log Gaussian CDF, used for closed form moment matching when the EP power is 1,
        logΦ(z) = log[(1 + erf(z / √2)) / 2]
    for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    and its derivative w.r.t. z
        dlogΦ(z)/dz = 𝓝(z|0,1) / Φ(z)
    :param z: input value, typically z = (my) / √(1 + v) [scalar]
    :return:
        lp: logΦ(z) [scalar]
        dlp: dlogΦ(z)/dz [scalar]
    """
    z = np.real(z)
    # erfc(z) = 1 - erf(z) is the complementary error function
    lp = np.log(erfc(-z / np.sqrt(2.0)) / 2.0)  # log Φ(z)
    dlp = np.exp(-z * z / 2.0 - lp) / np.sqrt(2.0 * pi)  # derivative w.r.t. z
    return lp, dlp
