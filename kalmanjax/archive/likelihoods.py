import jax.numpy as np
from jax.scipy.special import erf, erfc, gammaln
from jax.nn import softplus
from jax import jit, partial, jacrev, random
from jax.scipy.linalg import cholesky
from jax.scipy.stats import beta
from numpy.random import binomial
from numpy.polynomial.hermite import hermgauss
from utils import logphi, gaussian_moment_match, softplus_inv
pi = 3.141592653589793


class Likelihood(object):
    """
    The likelihood model class, p(yₙ|fₙ). Each likelihood implements its own moment matching method,
    which calculates the log partition function, logZₙ, and its derivatives w.r.t. the cavity mean mₙ,
        logZₙ = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[p(yₙ|fₙ)]
    If no custom moment matching method is provided, Gauss-Hermite quadrature is used by default.
    The requirement for quadrature is simply a method called evaluate_likelihood(), which computes
    the likelihood model p(yₙ|fₙ) for given data and function values.
    """
    def __init__(self, hyp=None):
        """
        :param hyp: (hyper)parameters of the likelihood model
        """
        self.hyp = softplus_inv(hyp)

    def evaluate_likelihood(self, y, f, hyp=None):
        raise NotImplementedError('direct evaluation of this likelihood is not implemented')

    def evaluate_log_likelihood(self, y, f, hyp=None):
        raise NotImplementedError('direct evaluation of this log-likelihood is not implemented')

    def conditional_moments(self, f, hyp=None):
        raise NotImplementedError('conditional moments of this likelihood are not implemented')

    @partial(jit, static_argnums=0)
    def moment_match_quadrature(self, y, m, v, hyp=None, power=1.0, num_quad_points=20):
        """
        Perform moment matching via Gauss-Hermite quadrature.
        Moment matching invloves computing the log partition function, logZₙ, and its derivatives w.r.t. the cavity mean
            logZₙ = log ∫ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        with EP power a.
        :param y: observed data (yₙ) [scalar]
        :param m: cavity mean (mₙ) [scalar]
        :param v: cavity variance (vₙ) [scalar]
        :param hyp: likelihood hyperparameter [scalar]
        :param power: EP power / fraction (a) [scalar]
        :param num_quad_points: the number of Gauss-Hermite sigma points to use during quadrature [scalar]
        :return:
            lZ: the log partition function, logZₙ  [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
        """
        x, w = hermgauss(num_quad_points)  # Gauss-Hermite sigma points and weights
        w = w / np.sqrt(pi)  # scale weights by 1/√π
        sigma_points = np.sqrt(2) * np.sqrt(v) * x + m  # scale locations according to cavity dist.
        # pre-compute wᵢ pᵃ(yₙ|xᵢ√(2vₙ) + mₙ)
        weighted_likelihood_eval = w * self.evaluate_likelihood(y, sigma_points, hyp) ** power

        # a different approach, based on the log-likelihood, which can be more stable:
        # ll = self.evaluate_log_likelihood(y, sigma_points)
        # lmax = np.max(ll)
        # weighted_likelihood_eval = np.exp(lmax * power) * w * np.exp(power * (ll - lmax))

        # Compute partition function via quadrature:
        # Zₙ = ∫ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ pᵃ(yₙ|xᵢ√(2vₙ) + mₙ)
        Z = np.sum(
            weighted_likelihood_eval
        )
        lZ = np.log(Z)
        Zinv = 1.0 / Z
        # Compute derivative of partition function via quadrature:
        # dZₙ/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #         ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ pᵃ(yₙ|xᵢ√(2vₙ) + mₙ)
        dZ = np.sum(
            (sigma_points - m) / v
            * weighted_likelihood_eval
        )
        # dlogZₙ/dmₙ = (dZₙ/dmₙ) / Zₙ
        dlZ = Zinv * dZ
        # Compute second derivative of partition function via quadrature:
        # d²Zₙ/dmₙ² = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #           ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] pᵃ(yₙ|xᵢ√(2vₙ) + mₙ)
        d2Z = np.sum(
            ((sigma_points - m) ** 2 / v ** 2 - 1.0 / v)
            * weighted_likelihood_eval
        )
        # d²logZₙ/dmₙ² = d[(dZₙ/dmₙ) / Zₙ]/dmₙ
        #              = (d²Zₙ/dmₙ² * Zₙ - (dZₙ/dmₙ)²) / Zₙ²
        #              = d²Zₙ/dmₙ² / Zₙ - (dlogZₙ/dmₙ)²
        d2lZ = -dlZ ** 2 + Zinv * d2Z
        site_mean = m - dlZ / d2lZ  # approx. likelihood (site) mean (see Rasmussen & Williams p75)
        site_var = -power * (v + 1 / d2lZ)  # approx. likelihood (site) variance
        return lZ, site_mean, site_var

    @partial(jit, static_argnums=0)
    def moment_match(self, y, m, v, hyp=None, power=1.0):
        """
        If no custom moment matching method is provided, we use Gauss-Hermite quadrature.
        """
        return self.moment_match_quadrature(y, m, v, hyp, power=power)

    @staticmethod
    def link_fn(latent_mean):
        return latent_mean

    def sample(self, f, rng_key=123):
        lik_expectation, lik_variance = self.conditional_moments(f)
        lik_std = cholesky(np.diag(np.expand_dims(lik_variance, 0)))
        return lik_expectation + lik_std * random.normal(random.PRNGKey(rng_key), shape=f.shape)

    @partial(jit, static_argnums=0)
    def statistical_linear_regression_quadrature(self, m, v, hyp=None, num_quad_points=20):
        """
        Perform statistical linear regression (SLR) using Gauss-Hermite quadrature.
        We aim to find a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ]).
        """
        x, w = hermgauss(num_quad_points)  # Gauss-Hermite sigma points and weights
        w = w / np.sqrt(pi)  # scale weights by 1/√π
        sigma_points = np.sqrt(2) * np.sqrt(v) * x + m  # scale locations according to cavity dist.
        lik_expectation, _ = self.conditional_moments(sigma_points, hyp)
        # Compute zₙ via quadrature:
        # zₙ = ∫ E[yₙ|fₙ] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ E[yₙ|xᵢ√(2vₙ) + mₙ]
        z = np.sum(
            w * lik_expectation
        )
        # Compute variance S via quadrature:
        # S = ∫ (E[yₙ|fₙ]-zₙ) (E[yₙ|fₙ]-zₙ)' 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ (E[yₙ|xᵢ√(2vₙ) + mₙ]-zₙ) (E[yₙ|xᵢ√(2vₙ) + mₙ]-zₙ)'
        S = np.sum(
            w * (lik_expectation - z) * (lik_expectation - z)
        )
        # Compute cross covariance C via quadrature:
        # C = ∫ (fₙ-mₙ) (E[yₙ|fₙ]-zₙ)' 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ (fₙ-mₙ) (E[yₙ|xᵢ√(2vₙ) + mₙ]-zₙ)'
        C = np.sum(
            w * (sigma_points - m) * (lik_expectation - z)
        )
        # compute likelihood approximation 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ])
        A = C * v**-1  # the scale
        b = z - A * m  # the offset
        omega = S - A * v * A  # the linearisation error
        return A, b, omega

    @partial(jit, static_argnums=0)
    def statistical_linear_regression(self, m, v, hyp=None):
        """
        If no custom SLR method is provided, we use Gauss-Hermite quadrature.
        """
        return self.statistical_linear_regression_quadrature(m, v, hyp)

    @partial(jit, static_argnums=0)
    def observation_model(self, f, r, hyp=None):
        """
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Var[yₙ|fₙ] rₙ
        """
        conditional_expectation, conditional_variance = self.conditional_moments(f, hyp)
        obs_model = conditional_expectation + cholesky(conditional_variance) * r
        return np.squeeze(obs_model)

    @partial(jit, static_argnums=0)
    def analytical_linearisation(self, m, hyp=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term rₙ.
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Var[yₙ|fₙ] rₙ
        The Jacobians are evaluated at the means, fₙ=m, rₙ=0, to be used during
        extended Kalman filtering and extended Kalman EP.
        """
        Jf, Jr = jacrev(self.observation_model, argnums=(0, 1))(m, 0.0, hyp)
        return Jf, Jr

    @partial(jit, static_argnums=0)
    def variational_expectation_quadrature(self, y, m, v, hyp=None, num_quad_points=20):
        """
        Computes the "variational expectation" via Gauss-Hermite quadrature, i.e. the
        expected log-likelihood, and its derivatives w.r.t. the posterior mean
            E[log p(yₙ|fₙ)] = log ∫ p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        with EP power a.
        :param y: observed data (yₙ) [scalar]
        :param m: posterior mean (mₙ) [scalar]
        :param v: posterior variance (vₙ) [scalar]
        :param hyp: likelihood hyperparameter [scalar]
        :param num_quad_points: the number of Gauss-Hermite sigma points to use during quadrature [scalar]
        :return:
            exp_log_lik: the expected log likelihood, E[log p(yₙ|fₙ)]  [scalar]
            dE: first derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
            d2E: second derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
        """
        x, w = hermgauss(num_quad_points)  # Gauss-Hermite sigma points and weights
        w = w / np.sqrt(pi)  # scale weights by 1/√π
        sigma_points = np.sqrt(2) * np.sqrt(v) * x + m  # scale locations according to cavity dist.
        # pre-compute wᵢ log p(yₙ|xᵢ√(2vₙ) + mₙ)
        weighted_log_likelihood_eval = w * self.evaluate_log_likelihood(y, sigma_points, hyp)
        # Compute expected log likelihood via quadrature:
        # E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                 ≈ ∑ᵢ wᵢ pᵃ(yₙ|xᵢ√(2vₙ) + mₙ)
        exp_log_lik = np.sum(
            weighted_log_likelihood_eval
        )
        # Compute first derivative via quadrature:
        # dE[log p(yₙ|fₙ)]/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                      ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ log p(yₙ|xᵢ√(2vₙ) + mₙ)
        dE = np.sum(
            (sigma_points - m) / v
            * weighted_log_likelihood_eval
        )
        # Compute second derivative via quadrature:
        # d²E[log p(yₙ|fₙ)]/dmₙ² = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                        ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] log p(yₙ|xᵢ√(2vₙ) + mₙ)
        d2E = np.sum(
            (0.5 * (v ** -2) * (sigma_points - m) ** 2 - 0.5 * v ** -1)
            * weighted_log_likelihood_eval
        )
        return exp_log_lik, dE, d2E

    @partial(jit, static_argnums=0)
    def variational_expectation(self, y, m, v, hyp=None):
        """
        If no custom variational expectation method is provided, we use Gauss-Hermite quadrature.
        """
        return self.variational_expectation_quadrature(y, m, v, hyp)


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
            self.hyp = 0.1
        self.name = 'Gaussian'

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        """
        Evaluate the Gaussian function 𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q quadrature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :param hyp: likelihood variance σ² [scalar]
        :return:
            𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [Q, 1]
        """
        hyp = softplus(self.hyp) if hyp is None else hyp
        return (2 * pi * hyp) ** -0.5 * np.exp(-0.5 * (y - f) ** 2 / hyp)

    @partial(jit, static_argnums=0)
    def evaluate_log_likelihood(self, y, f, hyp=None):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q quadrature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :param hyp: likelihood variance σ² [scalar]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [Q, 1]
        """
        hyp = softplus(self.hyp) if hyp is None else hyp
        return -0.5 * np.log(2 * pi * hyp) - 0.5 * (y - f) ** 2 / hyp

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        """
        The first two conditional moments of a Gaussian are the mean and variance:
            E[y|f] = f
            Var[y|f] = σ²
        """
        hyp = softplus(self.hyp) if hyp is None else hyp
        return f, hyp

    @partial(jit, static_argnums=0)
    def moment_match(self, y, m, v, hyp=None, power=1.0):
        """
        Closed form Gaussian moment matching.
        Calculates the log partition function of the EP tilted distribution:
            logZₙ = log ∫ 𝓝ᵃ(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
        and its derivatives w.r.t. mₙ, which are required for moment matching.
        :param y: observed data (yₙ) [scalar]
        :param m: cavity mean (mₙ) [scalar]
        :param v: cavity variance (vₙ) [scalar]
        :param hyp: observation noise variance (σ²) [scalar]
        :param power: EP power / fraction (a) - this is never required for the Gaussian likelihood [scalar]
        :return:
            lZ: the log partition function, logZₙ [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        """
        hyp = softplus(self.hyp) if hyp is None else hyp
        return gaussian_moment_match(y, m, v, hyp)


class Probit(Likelihood):
    """
    The Probit Binary Classification likelihood, i.e. the Error Function Likelihood,
    i.e. the Gaussian (Normal) cumulative density function:
        p(yₙ|fₙ) = Φ(yₙfₙ)
                 = ∫ 𝓝(x|0,1) dx, where the integral is over (-∞, fₙyₙ],
    and where we force the data to be +/-1: yₙ ϵ {-1, +1}.
    The Normal CDF is calulcated using the error function:
        Φ(yₙfₙ) = (1 + erf(yₙfₙ / √2)) / 2
    for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    """
    def __init__(self, hyp):
        """
        :param hyp: None. This likelihood model has no hyperparameters.
        """
        super().__init__(hyp=hyp)
        self.name = 'Probit'

    @staticmethod
    @jit
    def link_fn(latent_mean):
        return erfc(-latent_mean / np.sqrt(2.0)) - 1.0

    @partial(jit, static_argnums=0)
    def eval(self, mu, var):
        """
        ported from GPML toolbox - not used.
        """
        lp, _, _ = self.moment_match(1, mu, var)
        p = np.exp(lp)
        ymu = 2 * p - 1
        yvar = 4 * p * (1 - p)
        return lp, ymu, yvar

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        """
        Evaluate the Gaussian CDF likelihood model,
            Φ(yₙfₙ) = (1 + erf(yₙfₙ / √2)) / 2
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
        Can be used to evaluate Q quadrature points when performing moment matching.
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ [Q, 1]
        :param hyp: dummy input, Probit has no hyperparameters
        :return:
            Φ(yₙfₙ) [Q, 1]
        """
        return (1.0 + erf(y * f / np.sqrt(2.0))) / 2.0  # Φ(z)

    @partial(jit, static_argnums=0)
    def evaluate_log_likelihood(self, y, f, hyp=None):
        """
        Evaluate the Gaussian CDF log-likelihood,
            log Φ(yₙfₙ) = log[(1 + erf(yₙfₙ / √2)) / 2]
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z].
        Can be used to evaluate Q quadrature points when performing moment matching.
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ [Q, 1]
        :param hyp: dummy input, Probit has no hyperparameters
        :return:
            log Φ(yₙfₙ) [Q, 1]
        """
        return np.log(1.0 + erf(y * f / np.sqrt(2.0)) + 1e-10) - np.log(2)  # logΦ(z)

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        """
        The first two conditional moments of a Probit likelihood are:
            E[yₙ|fₙ] = Φ(fₙ)
            Var[yₙ|fₙ] = Φ(fₙ) (1 - Φ(fₙ))
            where Φ(fₙ) = (1 + erf(fₙ / √2)) / 2
        """
        # TODO: not working
        # phi = (1.0 + erf(f / np.sqrt(2.0))) / 2.0
        # phi = self.link_fn(f)
        # phi = erfc(f / np.sqrt(2.0)) - 1.0
        phi = self.evaluate_likelihood(1.0, f)
        return phi, phi * (1.0 - phi)

    @partial(jit, static_argnums=(0, 5))
    def moment_match(self, y, m, v, hyp=None, power=1.0):
        """
        Probit likelihood moment matching.
        Calculates the log partition function of the EP tilted distribution:
            logZₙ = log ∫ Φᵃ(yₙfₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        and its derivatives w.r.t. mₙ, which are required for moment matching.
        If the EP fraction a = 1, we get
                  = log Φ(yₙzₙ), where zₙ = mₙ / √(1 + vₙ)   [see Rasmussen & Williams p74]
        otherwise we must use quadrature to compute the log partition and its derivatives.
        Note: we enforce yₙ ϵ {-1, +1}.
        :param y: observed data (yₙ) [scalar]
        :param m: cavity mean (mₙ) [scalar]
        :param v: cavity variance (vₙ) [scalar]
        :param hyp: dummy variable (Probit has no hyperparameters)
        :param power: EP power / fraction (a) [scalar]
        :return:
            lZ: the log partition function, logZₙ [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        """
        y = np.sign(y)  # only allow values of +/-1
        # y[np.where(y == 0)] = -1  # set zeros to -1
        y = np.sign(y - 0.01)  # set zeros to -1
        if power == 1:  # if a = 1, we can calculate the moments in closed form
            z = m / np.sqrt(1.0 + v)
            z = z * y  # zₙ = yₙmₙ / √(1 + vₙ)
            # logZₙ = log ∫ Φ(yₙfₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
            #       = log Φ(yₙmₙ/√(1 + vₙ))  [see Rasmussen & Williams p74]
            lZ, dlp = logphi(z)
            # dlogZₙ/dmₙ = yₙ dlogΦ(zₙ)/dmₙ / √(1 + vₙ)
            dlZ = y * dlp / np.sqrt(1.0 + v)  # first derivative w.r.t mₙ
            # d²logZₙ/dmₙ² = -dlogΦ(zₙ)/dmₙ (zₙ + dlogΦ(zₙ)/dmₙ) / √(1 + vₙ)
            d2lZ = -dlp * (z + dlp) / (1.0 + v)  # second derivative w.r.t mₙ
            site_mean = m - dlZ / d2lZ  # approx. likelihood (site) mean (see Rasmussen & Williams p75)
            site_var = - (v + 1 / d2lZ)  # approx. likelihood (site) variance
            return lZ, site_mean, site_var
        else:
            # if a is not 1, we can calculate the moments via quadrature
            return self.moment_match_quadrature(y, m, v, None, power)


class Erf(Probit):
    pass


class Poisson(Likelihood):
    """
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using quadrature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self, hyp=None, link='exp'):
        """
        :param hyp: None. This likelihood model has no hyperparameters
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(hyp=hyp)
        if link is 'exp':
            self.link_fn = lambda mu: np.exp(mu)
        elif link is 'logistic':
            self.link_fn = lambda mu: np.log(1.0 + np.exp(mu))
        else:
            raise NotImplementedError('link function not implemented')
        self.name = 'Poisson'

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        """
        Evaluate the Poisson likelihood:
            p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q quadrature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :param hyp: dummy variable (Poisson has no hyperparameters)
        :return:
            Poisson(fₙ) = μʸ exp(-μ) / yₙ! [Q, 1]
        """
        mu = self.link_fn(f)
        return mu**y * np.exp(-mu) / np.exp(gammaln(y + 1))

    @partial(jit, static_argnums=0)
    def evaluate_log_likelihood(self, y, f, hyp=None):
        """
        Evaluate the Poisson log-likelihood:
            log p(yₙ|fₙ) = log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!)
        for μ = g(fₙ), where g() is the link function (exponential or logistic).
        We use the gamma function to evaluate yₙ! = gamma(yₙ + 1).
        Can be used to evaluate Q quadrature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :param hyp: dummy variable (Poisson has no hyperparameters)
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = self.link_fn(f)
        return y * np.log(mu) - mu - gammaln(y + 1)

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        return self.link_fn(f), self.link_fn(f)


class Beta(Likelihood):
    """
    likBeta(f) = 1/Z * y^(mu*phi-1) * (1-y)^((1-mu)*phi-1) with
    mean=mu and variance=mu*(1-mu)/(1+phi) where mu = g(f) is the Beta intensity,
    f is a Gaussian process, y is the interval data and
    Z = Gamma(phi)/Gamma(phi*mu)/Gamma(phi*(1-mu)).
    Hence, we have
    llik(f) = log(likBeta(f)) = -lam*(y-mu)^2/(2*mu^2*y) - log(Zy).
    """
    def __init__(self, hyp=None, p1=0.5, p2=0.5):
        """
        :param hyp: None
        """
        self.p1 = p1
        self.p2 = p2
        super().__init__(hyp=hyp)
        self.name = 'Beta'

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        return beta()


class SumOfGaussians(Likelihood):
    """
    A sum of two Gaussians
        p(yₙ|fₙ) = (𝓝(yₙ|fₙ-ω,σ₁²) + 𝓝(yₙ|fₙ+ω,σ₂²)) / 2
    """
    # def __init__(self, hyp=None, omega=2., var1=0.5, var2=2.):
    # def __init__(self, hyp=None, omega=1., var1=0.1, var2=0.7):
    def __init__(self, hyp=None, omega=0.8, var1=0.3, var2=0.5):
        """
        :param hyp: None
        """
        self.omega = omega
        self.var1 = var1
        self.var2 = var2
        super().__init__(hyp=hyp)
        self.name = 'sum of Gaussians'

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        return (npdf(y, f+self.omega, self.var1) + npdf(y, f-self.omega, self.var2)) / 2.

    @partial(jit, static_argnums=0)
    def evaluate_log_likelihood(self, y, f, hyp=None):
        return np.log(self.evaluate_likelihood(y, f, hyp))

    def sample(self, f, rng_key=123):
        samp1 = random.normal(random.PRNGKey(rng_key), shape=f.shape)
        samp2 = random.normal(random.PRNGKey(2*rng_key), shape=f.shape)
        w = binomial(1, .5, f.shape)
        # print(w)
        # print(1.0-w)
        gauss1 = f - self.omega + np.sqrt(self.var1) * samp1
        gauss2 = f + self.omega + np.sqrt(self.var2) * samp2
        return w * gauss1 + (1-w) * gauss2
        # print(gauss2.shape)
        # return (1.0-w) * gauss2

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        return f, (self.var1 + self.var2) / 2

    # @partial(jit, static_argnums=0)
    # def moment_match(self, y, m, v, hyp=None, power=1.0):
    #     # log partition function, lZ:
    #     # logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
    #     #       = log 𝓝(yₙ|mₙ,σ²+vₙ)
    #     lZ = (
    #             - (y - m) ** 2 / (hyp + v) / 2
    #             - np.log(np.maximum(2 * pi * (hyp + v), 1e-10)) / 2
    #     )
    #     # 𝓝(yₙ|fₙ,σ²) = 𝓝(fₙ|yₙ,σ²)
    #     site_mean = y
    #     site_var = hyp
    #     return lZ, site_mean, site_var


class Threshold(Likelihood):
    """
    The threshold likelihood resulting from the spike and slab prior
        p(yₙ|fₙ) = 𝓝(yₙ|h(fₙ),σ²)
    """
    def __init__(self, hyp, rho=1.2, p=0.2):
        """
        :param hyp: the noise variance σ² [scalar]
        """
        self.rho = rho
        self.p = p
        super().__init__(hyp=hyp)
        if self.hyp is None:
            print('using default likelihood parameter since none was supplied')
            self.hyp = 0.1
        self.name = 'Threshold'

    @partial(jit, static_argnums=0)
    def link_fn(self, latent_mean):
        return (1 - self.rho) * latent_mean + self.rho * threshold_func(latent_mean, self.p)

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        return npdf(y, f, hyp)

    @partial(jit, static_argnums=0)
    def evaluate_log_likelihood(self, y, f, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        return log_npdf(y, f, hyp)

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        hyp = self.hyp if hyp is None else hyp
        lik_expectation = self.link_fn(f)
        return lik_expectation, hyp


def npdf(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * pi * v)


def log_npdf(x, m, v):
    return -(x - m) ** 2 / (2 * v) - 0.5 * np.log(2 * pi * v)


def threshold_func(x, p):
    return x * p * npdf(x, 0, 11) / ((1 - p) * npdf(x, 0, 1) + p * npdf(x, 0, 11))
