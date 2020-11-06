import jax.numpy as np
from jax.scipy.special import erf, gammaln
from jax import jit, partial, jacrev, random, vmap, grad
from jax.scipy.linalg import cholesky, cho_factor, cho_solve
from utils import inv, softplus, sigmoid, logphi, gaussian_moment_match, softplus_inv, gauss_hermite, \
    ensure_positive_precision
pi = 3.141592653589793


def gaussian_first_derivative_wrt_mean(f, m, C, w):
    invC = inv(C)
    return invC @ (f - m) * w


def gaussian_second_derivative_wrt_mean(f, m, C, w):
    invC = inv(C)
    return (invC @ (f - m) @ (f - m).T @ invC - invC) * w


class Likelihood(object):
    """
    The likelihood model class, p(yₙ|fₙ). Each likelihood implements its own parameter update methods:
        Moment matching is used for EP
        Statistical linearisation is used for SLEP / UKS / GHKS
        Ananlytical linearisation is used for EEP / EKS
        Variational expectation is used for VI
    If no custom parameter update method is provided, cubature is used (Gauss-Hermite by default).
    The requirement for all inference methods to work is the implementation of the following methods:
        evaluate_likelihood(), which simply evaluates the likelihood given the latent function
        evaluate_log_likelihood()
        conditional_moments(), which return E[y|f] and Cov[y|f]
    """
    def __init__(self, hyp=None):
        """
        :param hyp: (hyper)parameters of the likelihood model
        """
        hyp = [] if hyp is None else hyp
        self.hyp = softplus_inv(np.array(hyp))

    def evaluate_likelihood(self, y, f, hyp=None):
        raise NotImplementedError('direct evaluation of this likelihood is not implemented')

    def evaluate_log_likelihood(self, y, f, hyp=None):
        raise NotImplementedError('direct evaluation of this log-likelihood is not implemented')

    def conditional_moments(self, f, hyp=None):
        raise NotImplementedError('conditional moments of this likelihood are not implemented')

    @partial(jit, static_argnums=(0, 6))
    def moment_match_cubature(self, y, cav_mean, cav_cov, hyp=None, power=1.0, cubature_func=None):
        """
        TODO: N.B. THIS VERSION IS SUPERCEDED BY THE FUNCTION BELOW. HOWEVER THIS ONE MAY BE MORE STABLE.
        Perform moment matching via cubature.
        Moment matching invloves computing the log partition function, logZₙ, and its derivatives w.r.t. the cavity mean
            logZₙ = log ∫ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        with EP power a.
        :param y: observed data (yₙ) [scalar]
        :param cav_mean: cavity mean (mₙ) [scalar]
        :param cav_cov: cavity covariance (cₙ) [scalar]
        :param hyp: likelihood hyperparameter [scalar]
        :param power: EP power / fraction (a) [scalar]
        :param cubature_func: the function to compute sigma points and weights to use during cubature
        :return:
            lZ: the log partition function, logZₙ  [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
        """
        if cubature_func is None:
            x, w = gauss_hermite(cav_mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(cav_mean.shape[0])
        cav_cho, low = cho_factor(cav_cov)
        # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity dist.
        sigma_points = cav_cho @ np.atleast_2d(x) + cav_mean
        # pre-compute wᵢ pᵃ(yₙ|xᵢ√(2vₙ) + mₙ)
        weighted_likelihood_eval = w * self.evaluate_likelihood(y, sigma_points, hyp) ** power

        # a different approach, based on the log-likelihood, which can be more stable:
        # ll = self.evaluate_log_likelihood(y, sigma_points)
        # lmax = np.max(ll)
        # weighted_likelihood_eval = np.exp(lmax * power) * w * np.exp(power * (ll - lmax))

        # Compute partition function via cubature:
        # Zₙ = ∫ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ pᵃ(yₙ|fsigᵢ)
        Z = np.sum(
            weighted_likelihood_eval, axis=-1
        )
        lZ = np.log(Z)
        Zinv = 1.0 / Z

        # Compute derivative of partition function via cubature:
        # dZₙ/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #         ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ pᵃ(yₙ|fsigᵢ)
        covinv_f_m = cho_solve((cav_cho, low), sigma_points - cav_mean)
        dZ = np.sum(
            # (sigma_points - cav_mean) / cav_cov
            covinv_f_m
            * weighted_likelihood_eval,
            axis=-1
        )
        # dlogZₙ/dmₙ = (dZₙ/dmₙ) / Zₙ
        dlZ = Zinv * dZ

        # Compute second derivative of partition function via cubature:
        # d²Zₙ/dmₙ² = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #           ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] pᵃ(yₙ|fsigᵢ)
        d2Z = np.sum(
            ((sigma_points - cav_mean) ** 2 / cav_cov ** 2 - 1.0 / cav_cov)
            * weighted_likelihood_eval
        )

        # d²logZₙ/dmₙ² = d[(dZₙ/dmₙ) / Zₙ]/dmₙ
        #              = (d²Zₙ/dmₙ² * Zₙ - (dZₙ/dmₙ)²) / Zₙ²
        #              = d²Zₙ/dmₙ² / Zₙ - (dlogZₙ/dmₙ)²
        d2lZ = -dlZ @ dlZ.T + Zinv * d2Z
        id2lZ = inv(ensure_positive_precision(-d2lZ) - 1e-10 * np.eye(d2lZ.shape[0]))
        site_mean = cav_mean + id2lZ @ dlZ  # approx. likelihood (site) mean (see Rasmussen & Williams p75)
        site_cov = power * (-cav_cov + id2lZ)  # approx. likelihood (site) variance
        return lZ, site_mean, site_cov

    @partial(jit, static_argnums=(0, 6))
    def moment_match_cubature(self, y, cav_mean, cav_cov, hyp=None, power=1.0, cubature_func=None):
        """
        TODO: N.B. THIS VERSION ALLOWS MULTI-DIMENSIONAL MOMENT MATCHING, BUT CAN BE UNSTABLE
        Perform moment matching via cubature.
        Moment matching invloves computing the log partition function, logZₙ, and its derivatives w.r.t. the cavity mean
            logZₙ = log ∫ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        with EP power a.
        :param y: observed data (yₙ) [scalar]
        :param cav_mean: cavity mean (mₙ) [scalar]
        :param cav_cov: cavity covariance (cₙ) [scalar]
        :param hyp: likelihood hyperparameter [scalar]
        :param power: EP power / fraction (a) [scalar]
        :param cubature_func: the function to compute sigma points and weights to use during cubature
        :return:
            lZ: the log partition function, logZₙ  [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True)  [scalar]
        """
        if cubature_func is None:
            x, w = gauss_hermite(cav_mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(cav_mean.shape[0])
        cav_cho, low = cho_factor(cav_cov)
        # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity dist.
        sigma_points = cav_cho @ np.atleast_2d(x) + cav_mean
        # pre-compute wᵢ pᵃ(yₙ|xᵢ√(2vₙ) + mₙ)
        weighted_likelihood_eval = w * self.evaluate_likelihood(y, sigma_points, hyp) ** power

        # Compute partition function via cubature:
        # Zₙ = ∫ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ pᵃ(yₙ|fsigᵢ)
        Z = np.sum(
            weighted_likelihood_eval, axis=-1
        )
        lZ = np.log(np.maximum(Z, 1e-8))
        Zinv = 1.0 / np.maximum(Z, 1e-8)

        # Compute derivative of partition function via cubature:
        # dZₙ/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #         ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ pᵃ(yₙ|fsigᵢ)
        d1 = vmap(
            gaussian_first_derivative_wrt_mean, (1, None, None, 1)
        )(sigma_points[..., None], cav_mean, cav_cov, weighted_likelihood_eval)
        dZ = np.sum(d1, axis=0)
        # dlogZₙ/dmₙ = (dZₙ/dmₙ) / Zₙ
        dlZ = Zinv * dZ

        # Compute second derivative of partition function via cubature:
        # d²Zₙ/dmₙ² = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] pᵃ(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #           ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹] pᵃ(yₙ|fsigᵢ)
        d2 = vmap(
            gaussian_second_derivative_wrt_mean, (1, None, None, 1)
        )(sigma_points[..., None], cav_mean, cav_cov, weighted_likelihood_eval)
        d2Z = np.sum(d2, axis=0)

        # d²logZₙ/dmₙ² = d[(dZₙ/dmₙ) / Zₙ]/dmₙ
        #              = (d²Zₙ/dmₙ² * Zₙ - (dZₙ/dmₙ)²) / Zₙ²
        #              = d²Zₙ/dmₙ² / Zₙ - (dlogZₙ/dmₙ)²
        d2lZ = -dlZ @ dlZ.T + Zinv * d2Z
        id2lZ = inv(ensure_positive_precision(-d2lZ) - 1e-10 * np.eye(d2lZ.shape[0]))
        site_mean = cav_mean + id2lZ @ dlZ  # approx. likelihood (site) mean (see Rasmussen & Williams p75)
        site_cov = power * (-cav_cov + id2lZ)  # approx. likelihood (site) variance
        return lZ, site_mean, site_cov

    @partial(jit, static_argnums=(0, 6))
    def moment_match(self, y, m, v, hyp=None, power=1.0, cubature_func=None):
        """
        If no custom moment matching method is provided, we use cubature.
        """
        return self.moment_match_cubature(y, m, v, hyp, power, cubature_func)

    @staticmethod
    def link_fn(latent_mean):
        return latent_mean

    def sample(self, f, rng_key=123):
        lik_expectation, lik_variance = self.conditional_moments(f)
        lik_std = cholesky(np.diag(np.expand_dims(lik_variance, 0)))
        return lik_expectation + lik_std * random.normal(random.PRNGKey(rng_key), shape=f.shape)

    @partial(jit, static_argnums=(0, 4))
    def statistical_linear_regression_cubature(self, cav_mean, cav_cov, hyp=None, cubature_func=None):
        """
        Perform statistical linear regression (SLR) using cubature.
        We aim to find a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ]).
        TODO: this currently assumes an additive noise model (ok for our current applications), make more general
        """
        if cubature_func is None:
            x, w = gauss_hermite(cav_mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(cav_mean.shape[0])
        # fsigᵢ=xᵢ√(vₙ) + mₙ: scale locations according to cavity dist.
        sigma_points = cholesky(cav_cov) @ np.atleast_2d(x) + cav_mean
        lik_expectation, lik_covariance = self.conditional_moments(sigma_points, hyp)
        # Compute zₙ via cubature:
        # zₙ = ∫ E[yₙ|fₙ] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ]
        mu = np.sum(
            w * lik_expectation, axis=-1
        )[:, None]
        # Compute variance S via cubature:
        # S = ∫ [(E[yₙ|fₙ]-zₙ) (E[yₙ|fₙ]-zₙ)' + Cov[yₙ|fₙ]] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ [(E[yₙ|fsigᵢ]-zₙ) (E[yₙ|fsigᵢ]-zₙ)' + Cov[yₙ|fₙ]]
        # TODO: allow for multi-dim cubature
        S = np.sum(
            w * ((lik_expectation - mu) * (lik_expectation - mu) + lik_covariance), axis=-1
        )[:, None]
        # Compute cross covariance C via cubature:
        # C = ∫ (fₙ-mₙ) (E[yₙ|fₙ]-zₙ)' 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ (fsigᵢ -mₙ) (E[yₙ|fsigᵢ]-zₙ)'
        C = np.sum(
            w * (sigma_points - cav_mean) * (lik_expectation - mu), axis=-1
        )[:, None]
        # Compute derivative of z via cubature:
        # omega = ∫ E[yₙ|fₙ] vₙ⁻¹ (fₙ-mₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #       ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ] vₙ⁻¹ (fsigᵢ-mₙ)
        omega = np.sum(
            w * lik_expectation * (inv(cav_cov) @ (sigma_points - cav_mean)), axis=-1
        )[None, :]
        return mu, S, C, omega

    @partial(jit, static_argnums=(0, 4))
    def statistical_linear_regression(self, m, v, hyp=None, cubature_func=None):
        """
        If no custom SLR method is provided, we use cubature.
        """
        return self.statistical_linear_regression_cubature(m, v, hyp, cubature_func)

    @partial(jit, static_argnums=0)
    def observation_model(self, f, sigma, hyp=None):
        """
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        """
        conditional_expectation, conditional_covariance = self.conditional_moments(f, hyp)
        obs_model = conditional_expectation + cholesky(conditional_covariance) @ sigma
        return np.squeeze(obs_model)

    @partial(jit, static_argnums=0)
    def analytical_linearisation(self, m, sigma=None, hyp=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        The Jacobians are evaluated at the means, fₙ=m, σₙ=0, to be used during
        Extended Kalman filtering and Extended EP.
        """
        sigma = np.array([[0.0]]) if sigma is None else sigma
        Jf, Jsigma = jacrev(self.observation_model, argnums=(0, 1))(m, sigma, hyp)
        return np.atleast_2d(np.squeeze(Jf)), np.atleast_2d(np.squeeze(Jsigma))

    @partial(jit, static_argnums=(0, 5))
    def variational_expectation_cubature(self, y, post_mean, post_cov, hyp=None, cubature_func=None):
        """
        Computes the "variational expectation" via cubature, i.e. the
        expected log-likelihood, and its derivatives w.r.t. the posterior mean
            E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        :param y: observed data (yₙ) [scalar]
        :param post_mean: posterior mean (mₙ) [scalar]
        :param post_cov: posterior variance (vₙ) [scalar]
        :param hyp: likelihood hyperparameter [scalar]
        :param cubature_func: the function to compute sigma points and weights to use during cubature
        :return:
            exp_log_lik: the expected log likelihood, E[log p(yₙ|fₙ)]  [scalar]
            dE_dm: derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
            dE_dv: derivative of E[log p(yₙ|fₙ)] w.r.t. vₙ  [scalar]
        """
        if cubature_func is None:
            x, w = gauss_hermite(post_mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(post_mean.shape[0])
        # fsigᵢ=xᵢ√(vₙ) + mₙ: scale locations according to cavity dist.
        sigma_points = cholesky(post_cov) @ np.atleast_2d(x) + post_mean
        # pre-compute wᵢ log p(yₙ|xᵢ√(2vₙ) + mₙ)
        weighted_log_likelihood_eval = w * self.evaluate_log_likelihood(y, sigma_points, hyp)
        # Compute expected log likelihood via cubature:
        # E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                 ≈ ∑ᵢ wᵢ p(yₙ|fsigᵢ)
        exp_log_lik = np.sum(
            weighted_log_likelihood_eval
        )
        # Compute first derivative via cubature:
        # dE[log p(yₙ|fₙ)]/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                      ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ log p(yₙ|fsigᵢ)
        invv = np.diag(post_cov)[:, None] ** -1
        dE_dm = np.sum(
            invv * (sigma_points - post_mean)
            * weighted_log_likelihood_eval, axis=-1
        )[:, None]
        # Compute second derivative via cubature (deriv. w.r.t. var = 0.5 * 2nd deriv. w.r.t. mean):
        # dE[log p(yₙ|fₙ)]/dvₙ = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹]/2 log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                        ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹]/2 log p(yₙ|fsigᵢ)
        dE_dv = np.sum(
            (0.5 * (invv ** 2 * (sigma_points - post_mean) ** 2) - 0.5 * invv)
            * weighted_log_likelihood_eval, axis=-1
        )
        dE_dv = np.diag(dE_dv)
        return exp_log_lik, dE_dm, dE_dv

    @partial(jit, static_argnums=(0, 5))
    def variational_expectation(self, y, m, v, hyp=None, cubature_func=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return self.variational_expectation_cubature(y, m, v, hyp, cubature_func)


class Gaussian(Likelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """
    def __init__(self, variance=0.1):
        """
        :param variance: The observation noise variance, σ²
        """
        super().__init__(hyp=variance)
        self.name = 'Gaussian'

    @property
    def variance(self):
        return softplus(self.hyp)

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        """
        Evaluate the Gaussian function 𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
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
        Can be used to evaluate Q cubature points.
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
        return f, hyp.reshape(-1, 1)

    @partial(jit, static_argnums=(0, 6))
    def moment_match(self, y, cav_mean, cav_cov, hyp=None, power=1.0, cubature_func=None):
        """
        Closed form Gaussian moment matching.
        Calculates the log partition function of the EP tilted distribution:
            logZₙ = log ∫ 𝓝ᵃ(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
        and its derivatives w.r.t. mₙ, which are required for moment matching.
        :param y: observed data (yₙ) [scalar]
        :param cav_mean: cavity mean (mₙ) [scalar]
        :param cav_cov: cavity variance (vₙ) [scalar]
        :param hyp: observation noise variance (σ²) [scalar]
        :param power: EP power / fraction (a) - this is never required for the Gaussian likelihood [scalar]
        :param cubature_func: not used
        :return:
            lZ: the log partition function, logZₙ [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        """
        hyp = softplus(self.hyp) if hyp is None else hyp
        return gaussian_moment_match(y, cav_mean, cav_cov, hyp)


class Bernoulli(Likelihood):
    """
    Bernoulli likelihood is p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾, where P = E[yₙ=1|fₙ].
    Link function maps latent GP to [0,1].
    The Probit link function, i.e. the Error Function Likelihood:
        i.e. the Gaussian (Normal) cumulative density function:
        P = E[yₙ=1|fₙ] = Φ(fₙ)
                       = ∫ 𝓝(x|0,1) dx, where the integral is over (-∞, fₙ],
        The Normal CDF is calulcated using the error function:
                       = (1 + erf(fₙ / √2)) / 2
        for erf(z) = (2/√π) ∫ exp(-x²) dx, where the integral is over [0, z]
    The logit link function:
        P = E[yₙ=1|fₙ] = 1 / 1 + exp(-fₙ)
    """
    def __init__(self, link):
        super().__init__(hyp=None)
        if link == 'logit':
            self.link_fn = lambda f: 1 / (1 + np.exp(-f))
            self.dlink_fn = lambda f: np.exp(f) / (1 + np.exp(f)) ** 2
            self.link = link
        elif link == 'probit':
            jitter = 1e-10
            self.link_fn = lambda f: 0.5 * (1.0 + erf(f / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter
            self.dlink_fn = lambda f: grad(self.link_fn)(np.squeeze(f)).reshape(-1, 1)
            self.link = link
        else:
            raise NotImplementedError('link function not implemented')
        self.name = 'Bernoulli'

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :param hyp: dummy input, Probit/Logit has no hyperparameters
        :return:
            p(yₙ|fₙ) = Pʸ(1-P)⁽¹⁻ʸ⁾
        """
        return np.where(np.equal(y, 1), self.link_fn(f), 1 - self.link_fn(f))

    @partial(jit, static_argnums=0)
    def evaluate_log_likelihood(self, y, f, hyp=None):
        """
        :param y: observed data yₙ ϵ {-1, +1} [scalar]
        :param f: latent function value fₙ ϵ ℝ
        :param hyp: dummy input, Probit has no hyperparameters
        :return:
            log p(yₙ|fₙ)
        """
        return np.log(self.evaluate_likelihood(y, f))

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        """
        The first two conditional moments of a Probit likelihood are:
            E[yₙ|fₙ] = Φ(fₙ)
            Var[yₙ|fₙ] = Φ(fₙ) (1 - Φ(fₙ))
        """
        return self.link_fn(f), self.link_fn(f)-(self.link_fn(f)**2)

    @partial(jit, static_argnums=(0, 5, 6))
    def moment_match(self, y, m, v, hyp=None, power=1.0, cubature_func=None):
        """
        Probit likelihood moment matching.
        Calculates the log partition function of the EP tilted distribution:
            logZₙ = log ∫ Φᵃ(yₙfₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        and its derivatives w.r.t. mₙ, which are required for moment matching.
        If the EP fraction a = 1, we get
                  = log Φ(yₙzₙ), where zₙ = mₙ / √(1 + vₙ)   [see Rasmussen & Williams p74]
        otherwise we must use cubature to compute the log partition and its derivatives.
        :param y: observed data (yₙ) [scalar]
        :param m: cavity mean (mₙ) [scalar]
        :param v: cavity variance (vₙ) [scalar]
        :param hyp: dummy variable (Probit has no hyperparameters)
        :param power: EP power / fraction (a) [scalar]
        :param cubature_func: function returning the sigma points and weights for cubature
        :return:
            lZ: the log partition function, logZₙ [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        """
        y = np.sign(y)  # only allow values of {0, 1}
        if power == 1 and self.link == 'probit':  # if a = 1, we can calculate the moments in closed form
            y = np.sign(y - 0.01)  # set zeros to -1 for closed form probit calc
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
            # if a is not 1, we can calculate the moments via cubature
            return self.moment_match_cubature(y, m, v, None, power, cubature_func)

    @partial(jit, static_argnums=0)
    def analytical_linearisation(self, m, sigma=None, hyp=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        """
        Jf = self.dlink_fn(m) + (
            0.5 * (self.link_fn(m) * (1 - self.link_fn(m))) ** -0.5
            * self.dlink_fn(m) * (1 - 2 * self.link_fn(m))
        ) * sigma
        Jsigma = (self.link_fn(m) * (1 - self.link_fn(m))) ** 0.5
        return Jf, Jsigma


class Probit(Bernoulli):
    """
    The probit likelihood = Bernoulli likelihood with probit link.
    """
    def __init__(self):
        super().__init__(link='probit')


class Erf(Probit):
    """
    The error function likelihood = probit = Bernoulli likelihood with probit link.
    """
    pass


class Logit(Bernoulli):
    """
    The logit likelihood = Bernoulli likelihood with logit link.
    """
    def __init__(self):
        super().__init__(link='logit')


class Logistic(Logit):
    """
    The logistic likelihood = logit = Bernoulli likelihood with logit link.
    """
    pass


class Poisson(Likelihood):
    """
    The Poisson likelihood:
        p(yₙ|fₙ) = Poisson(fₙ) = μʸ exp(-μ) / yₙ!
    where μ = g(fₙ) = mean = variance is the Poisson intensity.
    yₙ is non-negative integer count data.
    No closed form moment matching is available, se we default to using cubature.

    Letting Zy = gamma(yₙ+1) = yₙ!, we get log p(yₙ|fₙ) = log(g(fₙ))yₙ - g(fₙ) - log(Zy)
    The larger the intensity μ, the stronger the likelihood resembles a Gaussian
    since skewness = 1/sqrt(μ) and kurtosis = 1/μ.
    Two possible link functions:
    'exp':      link(fₙ) = exp(fₙ),         we have p(yₙ|fₙ) = exp(fₙyₙ-exp(fₙ))           / Zy.
    'logistic': link(fₙ) = log(1+exp(fₙ))), we have p(yₙ|fₙ) = logʸ(1+exp(fₙ)))(1+exp(fₙ)) / Zy.
    """
    def __init__(self, link='exp'):
        """
        :param link: link function, either 'exp' or 'logistic'
        """
        super().__init__(hyp=None)
        if link == 'exp':
            self.link_fn = lambda mu: np.exp(mu)
            self.dlink_fn = lambda mu: np.exp(mu)
        elif link == 'logistic':
            self.link_fn = lambda mu: softplus(mu)
            self.dlink_fn = lambda mu: sigmoid(mu)
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
        Can be used to evaluate Q cubature points when performing moment matching.
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
        Can be used to evaluate Q cubature points when performing moment matching.
        :param y: observed data (yₙ) [scalar]
        :param f: latent function value (fₙ) [Q, 1]
        :param hyp: dummy variable (Poisson has no hyperparameters)
        :return:
            log Poisson(fₙ) = log(μʸ exp(-μ) / yₙ!) [Q, 1]
        """
        mu = self.link_fn(f)
        return y * np.log(mu) - mu - gammaln(y + 1)

    @partial(jit, static_argnums=0)
    def observation_model(self, f, sigma, hyp=None):
        """
        TODO: sort out broadcasting so we don't need this additional function (only difference is the transpose)
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        """
        conditional_expectation, conditional_covariance = self.conditional_moments(f, hyp)
        obs_model = conditional_expectation + cholesky(conditional_covariance.T) @ sigma
        return np.squeeze(obs_model)

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        """
        The first two conditional moments of a Poisson distribution are equal to the intensity:
            E[yₙ|fₙ] = link(fₙ)
            Var[yₙ|fₙ] = link(fₙ)
        """
        # return self.link_fn(f), self.link_fn(f)
        return self.link_fn(f), vmap(np.diag, 1, 2)(self.link_fn(f))

    @partial(jit, static_argnums=0)
    def analytical_linearisation(self, m, sigma=None, hyp=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        """
        Jf = np.diag(np.squeeze(self.link_fn(m) + 0.5 * self.link_fn(m) ** -0.5 * self.dlink_fn(m) * sigma, axis=-1))
        Jsigma = np.diag(np.squeeze(self.link_fn(m) ** 0.5, axis=-1))
        return Jf, Jsigma


class HeteroscedasticNoise(Likelihood):
    """
    The Heteroscedastic Noise likelihood:
        p(y|f1,f2) = N(y|f1,link(f2)^2)
    """
    def __init__(self, link='softplus'):
        """
        :param link: link function, either 'exp' or 'softplus' (note that the link is modified with an offset)
        """
        super().__init__(hyp=None)
        if link == 'exp':
            self.link_fn = lambda mu: np.exp(mu - 0.5)
            self.dlink_fn = lambda mu: np.exp(mu - 0.5)
        elif link == 'softplus':
            self.link_fn = lambda mu: softplus(mu - 0.5) + 1e-10
            self.dlink_fn = lambda mu: sigmoid(mu - 0.5)
        else:
            raise NotImplementedError('link function not implemented')
        self.name = 'Heteroscedastic Noise'

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        """
        Evaluate the likelihood
        """
        mu, var = self.conditional_moments(f)
        return (2 * pi * var) ** -0.5 * np.exp(-0.5 * (y - mu) ** 2 / var)

    @partial(jit, static_argnums=0)
    def evaluate_log_likelihood(self, y, f, hyp=None):
        """
        Evaluate the log-likelihood
        """
        mu, var = self.conditional_moments(f)
        return -0.5 * np.log(2 * pi * var) - 0.5 * (y - mu) ** 2 / var

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        """
        """
        return f[0][None, ...], self.link_fn(f[1][None, ...]) ** 2

    @partial(jit, static_argnums=(0, 6))
    def moment_match(self, y, cav_mean, cav_cov, hyp=None, power=1.0, cubature_func=None):
        """
        """
        if cubature_func is None:
            x, w = gauss_hermite(1, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(1)
        # sigma_points = np.sqrt(2) * np.sqrt(v) * x + m  # scale locations according to cavity dist.
        sigma_points = np.sqrt(cav_cov[1, 1]) * x + cav_mean[1]  # fsigᵢ=xᵢ√cₙ + mₙ: scale locations according to cavity

        f2 = self.link_fn(sigma_points) ** 2. / power
        obs_var = f2 + cav_cov[0, 0]
        const = power ** -0.5 * (2 * pi * self.link_fn(sigma_points) ** 2.) ** (0.5 - 0.5 * power)
        normpdf = const * (2 * pi * obs_var) ** -0.5 * np.exp(-0.5 * (y - cav_mean[0, 0]) ** 2 / obs_var)
        Z = np.sum(w * normpdf)
        Zinv = 1. / np.maximum(Z, 1e-8)
        lZ = np.log(np.maximum(Z, 1e-8))

        dZ_integrand1 = (y - cav_mean[0, 0]) / obs_var * normpdf
        dlZ1 = Zinv * np.sum(w * dZ_integrand1)

        dZ_integrand2 = (sigma_points - cav_mean[1, 0]) / cav_cov[1, 1] * normpdf
        dlZ2 = Zinv * np.sum(w * dZ_integrand2)

        d2Z_integrand1 = (-(f2 + cav_cov[0, 0]) ** -1 + ((y - cav_mean[0, 0]) / obs_var) ** 2) * normpdf
        d2lZ1 = -dlZ1 ** 2 + Zinv * np.sum(w * d2Z_integrand1)

        d2Z_integrand2 = (-cav_cov[1, 1] ** -1 + ((sigma_points - cav_mean[1, 0]) / cav_cov[1, 1]) ** 2) * normpdf
        d2lZ2 = -dlZ2 ** 2 + Zinv * np.sum(w * d2Z_integrand2)

        dlZ = np.block([[dlZ1],
                        [dlZ2]])
        d2lZ = np.block([[d2lZ1, 0],
                         [0., d2lZ2]])
        id2lZ = inv(ensure_positive_precision(-d2lZ) - 1e-10 * np.eye(d2lZ.shape[0]))
        site_mean = cav_mean + id2lZ @ dlZ  # approx. likelihood (site) mean (see Rasmussen & Williams p75)
        site_cov = power * (-cav_cov + id2lZ)  # approx. likelihood (site) variance
        return lZ, site_mean, site_cov

    @partial(jit, static_argnums=0)
    def log_expected_likelihood(self, y, x, w, cav_mean, cav_var, power):
        sigma_points = np.sqrt(cav_var[1]) * x + cav_mean[1]
        f2 = self.link_fn(sigma_points) ** 2. / power
        obs_var = f2 + cav_var[0]
        const = power ** -0.5 * (2 * pi * self.link_fn(sigma_points) ** 2.) ** (0.5 - 0.5 * power)
        normpdf = const * (2 * pi * obs_var) ** -0.5 * np.exp(-0.5 * (y - cav_mean[0]) ** 2 / obs_var)
        Z = np.sum(w * normpdf)
        lZ = np.log(Z + 1e-8)
        return lZ

    @partial(jit, static_argnums=0)
    def dlZ_dm(self, y, x, w, cav_mean, cav_var, power):
        return jacrev(self.log_expected_likelihood, argnums=3)(y, x, w, cav_mean, cav_var, power)

    @partial(jit, static_argnums=(0, 6))
    def moment_match_unstable(self, y, cav_mean, cav_cov, hyp=None, power=1.0, cubature_func=None):
        """
        TODO: Attempt to compute full site covariance, including cross terms. However, this makes things unstable.
        """
        if cubature_func is None:
            x, w = gauss_hermite(1, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(1)
        lZ = self.log_expected_likelihood(y, x, w, np.squeeze(cav_mean), np.squeeze(np.diag(cav_cov)), power)
        dlZ = self.dlZ_dm(y, x, w, np.squeeze(cav_mean), np.squeeze(np.diag(cav_cov)), power)[:, None]
        d2lZ = jacrev(self.dlZ_dm, argnums=3)(y, x, w, np.squeeze(cav_mean), np.squeeze(np.diag(cav_cov)), power)
        # d2lZ = np.diag(np.diag(d2lZ))  # discard cross terms
        id2lZ = inv(ensure_positive_precision(-d2lZ) - 1e-10 * np.eye(d2lZ.shape[0]))
        site_mean = cav_mean + id2lZ @ dlZ  # approx. likelihood (site) mean (see Rasmussen & Williams p75)
        site_cov = power * (-cav_cov + id2lZ)  # approx. likelihood (site) variance
        return lZ, site_mean, site_cov

    @partial(jit, static_argnums=(0, 5))
    def variational_expectation(self, y, m, v, hyp=None, cubature_func=None):
        """
        """
        if cubature_func is None:
            x, w = gauss_hermite(1, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(1)
        m0, m1, v0, v1 = m[0, 0], m[1, 0], v[0, 0], v[1, 1]
        sigma_points = np.sqrt(v1) * x + m1  # fsigᵢ=xᵢ√(2vₙ) + mₙ: scale locations according to cavity dist.
        # pre-compute wᵢ log p(yₙ|xᵢ√(2vₙ) + mₙ)
        var = self.link_fn(sigma_points) ** 2
        log_lik = np.log(var) + var ** -1 * ((y - m0) ** 2 + v0)
        weighted_log_likelihood_eval = w * log_lik
        # Compute expected log likelihood via cubature:
        # E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                 ≈ ∑ᵢ wᵢ p(yₙ|fsigᵢ)
        exp_log_lik = -0.5 * np.log(2 * pi) - 0.5 * np.sum(
            weighted_log_likelihood_eval
        )
        # Compute first derivative via cubature:
        dE_dm1 = np.sum(
            (var ** -1 * (y - m0 + v0)) * w
        )
        # dE[log p(yₙ|fₙ)]/dmₙ = ∫ (fₙ-mₙ) vₙ⁻¹ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                      ≈ ∑ᵢ wᵢ (fₙ-mₙ) vₙ⁻¹ log p(yₙ|fsigᵢ)
        dE_dm2 = - 0.5 * np.sum(
            weighted_log_likelihood_eval * v1 ** -1 * (sigma_points - m1)
        )
        # Compute derivative w.r.t. variance:
        dE_dv1 = -0.5 * np.sum(
            var ** -1 * w
        )
        # dE[log p(yₙ|fₙ)]/dvₙ = ∫ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹]/2 log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #                        ≈ ∑ᵢ wᵢ [(fₙ-mₙ)² vₙ⁻² - vₙ⁻¹]/2 log p(yₙ|fsigᵢ)
        dE_dv2 = -0.25 * np.sum(
            (v1 ** -2 * (sigma_points - m1) ** 2 - v1 ** -1)
            * weighted_log_likelihood_eval
        )
        dE_dm = np.block([[dE_dm1],
                          [dE_dm2]])
        dE_dv = np.block([[dE_dv1, 0],
                          [0., dE_dv2]])
        return exp_log_lik, dE_dm, dE_dv

    @partial(jit, static_argnums=(0, 4))
    def statistical_linear_regression(self, cav_mean, cav_cov, hyp=None, cubature_func=None):
        """
        Perform statistical linear regression (SLR) using cubature.
        We aim to find a likelihood approximation p(yₙ|fₙ) ≈ 𝓝(yₙ|Afₙ+b,Ω+Var[yₙ|fₙ]).
        """
        if cubature_func is None:
            x, w = gauss_hermite(cav_mean.shape[0], 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(cav_mean.shape[0])
        m0, m1, v0, v1 = cav_mean[0, 0], cav_mean[1, 0], cav_cov[0, 0], cav_cov[1, 1]
        # fsigᵢ=xᵢ√(vₙ) + mₙ: scale locations according to cavity dist.
        sigma_points = cholesky(cav_cov) @ x + cav_mean
        var = self.link_fn(sigma_points[1]) ** 2
        # Compute zₙ via cubature:
        # zₙ = ∫ E[yₙ|fₙ] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ]
        mu = m0.reshape(1, 1)
        # Compute variance S via cubature:
        # S = ∫ [(E[yₙ|fₙ]-zₙ) (E[yₙ|fₙ]-zₙ)' + Cov[yₙ|fₙ]] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ [(E[yₙ|fsigᵢ]-zₙ) (E[yₙ|fsigᵢ]-zₙ)' + Cov[yₙ|fₙ]]
        S = v0 + np.sum(
            w * var
        )
        S = S.reshape(1, 1)
        # Compute cross covariance C via cubature:
        # C = ∫ (fₙ-mₙ) (E[yₙ|fₙ]-zₙ)' 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ (fsigᵢ -mₙ) (E[yₙ|fsigᵢ]-zₙ)'
        C = np.sum(
            w * (sigma_points - cav_mean) * (sigma_points[0] - m0), axis=-1
        ).reshape(2, 1)
        # Compute derivative of z via cubature:
        # omega = ∫ E[yₙ|fₙ] vₙ⁻¹ (fₙ-mₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #       ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ] vₙ⁻¹ (fsigᵢ-mₙ)
        omega = np.block([[1., 0.]])
        return mu, S, C, omega

    @partial(jit, static_argnums=0)
    def analytical_linearisation(self, m, sigma=None, hyp=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        """
        return np.block([[np.array(1.0), self.dlink_fn(m[1]) * sigma]]), self.link_fn(np.array([m[1]]))


class AudioAmplitudeDemodulation(Likelihood):
    """
    The Audio Amplitude Demodulation likelihood
    """
    def __init__(self, variance=0.1):
        """
        param hyp: observation noise
        """
        super().__init__(hyp=variance)
        self.name = 'Audio Amplitude Demodulation'
        self.link_fn = lambda f: softplus(f)
        self.dlink_fn = lambda f: sigmoid(f)  # derivative of the link function

    @property
    def variance(self):
        return softplus(self.hyp)

    @partial(jit, static_argnums=0)
    def evaluate_likelihood(self, y, f, hyp=None):
        """
        Evaluate the likelihood
        """
        mu, var = self.conditional_moments(f, hyp)
        return (2 * pi * var) ** -0.5 * np.exp(-0.5 * (y - mu) ** 2 / var)

    @partial(jit, static_argnums=0)
    def evaluate_log_likelihood(self, y, f, hyp=None):
        """
        Evaluate the log-likelihood
        """
        mu, var = self.conditional_moments(f, hyp)
        return -0.5 * np.log(2 * pi * var) - 0.5 * (y - mu) ** 2 / var

    @partial(jit, static_argnums=0)
    def conditional_moments(self, f, hyp=None):
        """
        """
        obs_noise_var = hyp if hyp is not None else self.hyp
        num_components = int(f.shape[0] / 2)
        subbands, modulators = f[:num_components], self.link_fn(f[num_components:])
        return np.atleast_2d(np.sum(subbands * modulators, axis=0)), np.atleast_2d(obs_noise_var)
        # return np.atleast_2d(modulators.T @ subbands),  np.atleast_2d(obs_noise_var)

    @partial(jit, static_argnums=(0, 6))
    def moment_match(self, y, cav_mean, cav_cov, hyp=None, power=1.0, cubature_func=None):
        """
        """
        num_components = int(cav_mean.shape[0] / 2)
        if cubature_func is None:
            x, w = gauss_hermite(num_components, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(num_components)

        subband_mean, modulator_mean = cav_mean[:num_components], self.link_fn(cav_mean[num_components:])
        subband_cov, modulator_cov = cav_cov[:num_components, :num_components], cav_cov[num_components:, num_components:]
        sigma_points = cholesky(modulator_cov) @ x + modulator_mean
        const = power ** -0.5 * (2 * pi * hyp) ** (0.5 - 0.5 * power)
        mu = (self.link_fn(sigma_points).T @ subband_mean)[:, 0]
        var = hyp / power + (self.link_fn(sigma_points).T ** 2 @ np.diag(subband_cov)[..., None])[:, 0]
        normpdf = const * (2 * pi * var) ** -0.5 * np.exp(-0.5 * (y - mu) ** 2 / var)
        Z = np.sum(w * normpdf)
        Zinv = 1. / (Z + 1e-8)
        lZ = np.log(Z + 1e-8)

        dZ1 = np.sum(w * self.link_fn(sigma_points) * (y - mu) / var * normpdf, axis=-1)
        dZ2 = np.sum(w * (sigma_points - modulator_mean) * np.diag(modulator_cov)[..., None] ** -1 * normpdf, axis=-1)
        dlZ = Zinv * np.block([dZ1, dZ2])

        d2Z1 = np.sum(w * self.link_fn(sigma_points) ** 2 * (
            ((y - mu) / var) ** 2
            - var ** -1
        ) * normpdf, axis=-1)
        d2Z2 = np.sum(w * (
            ((sigma_points - modulator_mean) * np.diag(modulator_cov)[..., None] ** -1) ** 2
            - np.diag(modulator_cov)[..., None] ** -1
        ) * normpdf, axis=-1)
        d2lZ = np.diag(-dlZ ** 2 + Zinv * np.block([d2Z1, d2Z2]))
        id2lZ = inv(ensure_positive_precision(-d2lZ) - 1e-10 * np.eye(d2lZ.shape[0]))
        site_mean = cav_mean + id2lZ @ dlZ[..., None]  # approx. likelihood (site) mean (see Rasmussen & Williams p75)
        site_cov = power * (-cav_cov + id2lZ)  # approx. likelihood (site) variance
        return lZ, site_mean, site_cov

    @partial(jit, static_argnums=0)
    def analytical_linearisation(self, m, sigma=None, hyp=None):
        """
        """
        obs_noise_var = hyp if hyp is not None else self.hyp
        num_components = int(m.shape[0] / 2)
        subbands, modulators = m[:num_components], self.link_fn(m[num_components:])
        Jf = np.block([[modulators], [subbands * self.dlink_fn(m[num_components:])]])
        Jsigma = np.array([[np.sqrt(obs_noise_var)]])
        return np.atleast_2d(Jf).T, np.atleast_2d(Jsigma).T

    @partial(jit, static_argnums=(0, 4))
    def statistical_linear_regression(self, cav_mean, cav_cov, hyp=None, cubature_func=None):
        """
        This gives the same result as above - delete
        """
        num_components = int(cav_mean.shape[0] / 2)
        if cubature_func is None:
            x, w = gauss_hermite(num_components, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(num_components)

        subband_mean, modulator_mean = cav_mean[:num_components], self.link_fn(cav_mean[num_components:])
        subband_cov, modulator_cov = cav_cov[:num_components, :num_components], cav_cov[num_components:,
                                                                                        num_components:]
        sigma_points = cholesky(modulator_cov) @ x + modulator_mean
        lik_expectation, lik_covariance = (self.link_fn(sigma_points).T @ subband_mean).T, hyp
        # Compute zₙ via cubature:
        # muₙ = ∫ E[yₙ|fₙ] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #    ≈ ∑ᵢ wᵢ E[yₙ|fsigᵢ]
        mu = np.sum(
            w * lik_expectation, axis=-1
        )[:, None]
        # Compute variance S via cubature:
        # S = ∫ [(E[yₙ|fₙ]-zₙ) (E[yₙ|fₙ]-zₙ)' + Cov[yₙ|fₙ]] 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ [(E[yₙ|fsigᵢ]-zₙ) (E[yₙ|fsigᵢ]-zₙ)' + Cov[yₙ|fₙ]]
        S = np.sum(
            w * ((lik_expectation - mu) * (lik_expectation - mu) + lik_covariance), axis=-1
        )[:, None]
        # Compute cross covariance C via cubature:
        # C = ∫ (fₙ-mₙ) (E[yₙ|fₙ]-zₙ)' 𝓝(fₙ|mₙ,vₙ) dfₙ
        #   ≈ ∑ᵢ wᵢ (fsigᵢ -mₙ) (E[yₙ|fsigᵢ]-zₙ)'
        C = np.sum(
            w * np.block([[self.link_fn(sigma_points) * np.diag(subband_cov)[..., None]],
                          [sigma_points - modulator_mean]]) * (lik_expectation - mu), axis=-1
        )[:, None]
        # Compute derivative of mu via cubature:
        omega = np.sum(
            w * np.block([[self.link_fn(sigma_points)],
                          [np.diag(modulator_cov)[..., None] ** -1 * (sigma_points - modulator_mean) * lik_expectation]]), axis=-1
        )[None, :]
        return mu, S, C, omega

    @partial(jit, static_argnums=(0, 5))
    def variational_expectation(self, y, post_mean, post_cov, hyp=None, cubature_func=None):
        """
        """
        num_components = int(post_mean.shape[0] / 2)
        if cubature_func is None:
            x, w = gauss_hermite(num_components, 20)  # Gauss-Hermite sigma points and weights
        else:
            x, w = cubature_func(num_components)

        subband_mean, modulator_mean = post_mean[:num_components], self.link_fn(post_mean[num_components:])
        subband_cov, modulator_cov = post_cov[:num_components, :num_components], post_cov[num_components:,
                                                                                 num_components:]
        sigma_points = cholesky(modulator_cov) @ x + modulator_mean

        modulator_var = np.diag(subband_cov)[..., None]
        mu = (self.link_fn(sigma_points).T @ subband_mean)[:, 0]
        lognormpdf = -0.5 * np.log(2 * pi * hyp) - 0.5 * (y - mu) ** 2 / hyp
        const = -0.5 / hyp * (self.link_fn(sigma_points).T ** 2 @ modulator_var)[:, 0]
        exp_log_lik = np.sum(w * (lognormpdf + const))

        dE1 = np.sum(w * self.link_fn(sigma_points) * (y - mu) / hyp, axis=-1)
        dE2 = np.sum(w * (sigma_points - modulator_mean) * modulator_var ** -1
                     * (lognormpdf + const), axis=-1)
        dE_dm = np.block([dE1, dE2])[..., None]

        d2E1 = np.sum(w * - 0.5 * self.link_fn(sigma_points) ** 2 / hyp, axis=-1)
        d2E2 = np.sum(w * 0.5 * (
                ((sigma_points - modulator_mean) * modulator_var ** -1) ** 2
                - modulator_var ** -1
        ) * (lognormpdf + const), axis=-1)
        dE_dv = np.diag(np.block([d2E1, d2E2]))
        return exp_log_lik, dE_dm, dE_dv

    @partial(jit, static_argnums=0)
    def analytical_linearisation(self, m, sigma=None, hyp=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        """
        num_components = int(m.shape[0] / 2)
        Jf = np.block([[self.link_fn(m[num_components:])], [m[:num_components] * self.dlink_fn(m[num_components:])]]).T
        Jsigma = np.array([[hyp ** 0.5]])
        return Jf, Jsigma
