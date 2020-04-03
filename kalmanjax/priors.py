import jax.numpy as np
from jax import jit, partial
from jax.nn import softplus


class Prior(object):
    """
    The GP Kernel / prior class.
    Implements methods for converting GP priors,
        f(t) ~ GP(0,k(t,t'))
    into state space models.
    Constructs a linear time-invariant (LTI) stochastic differential
    equation (SDE) of the following form:
        dx(t)/dt = F x(t) + L w(t)
              yₙ ~ p(yₙ | f(tₙ)=H x(tₙ))
    where w(t) is a white noise process and where the state x(t) is
    Gaussian distributed with initial state distribution x(t)~𝓝(0,Pinf).
    F      - Feedback matrix
    L      - Noise effect matrix
    Qc     - Spectral density of white noise process w(t)
    H      - Observation model matrix
    Pinf   - Covariance of the stationary process
    """
    def __init__(self, hyp=None):
        self.hyp = hyp


class Exponential(Prior):
    """
    Exponential, i.e. Matern-1/2 kernel in SDE form
    Hyperparameters:
        variance, σ²
        lengthscale, l
    The associated continuous-time state space model matrices are:
    F      = -1/l
    L      = 1
    Qc     = 2σ²/l
    H      = 1
    Pinf   = σ²
    """
    def __init__(self, hyp=None):
        super().__init__(hyp=hyp)
        if self.hyp is None:
            print('using default kernel parameters since none were supplied')
            self.hyp = [0.55, 0.55]  # softplus(0.55) ~= 1
        self.name = 'Exponential'

    @partial(jit, static_argnums=0)
    def cf_to_ss(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        if hyperparams is None:
            hyperparams = softplus(self.hyp)
        var, ell = hyperparams[0], hyperparams[1]
        F = np.array([[-1.0 / ell]])
        L = np.array([[1.0]])
        Qc = np.array([[2.0 * var / ell]])
        H = np.array([[1.0]])
        Pinf = np.array([[var]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def expm(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the exponential prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [1, 1]
        """
        if hyperparams is None:
            hyperparams = softplus(self.hyp)
        ell = hyperparams[1]
        A = np.broadcast_to(np.exp(-dt / ell), [1, 1])
        return A


class Matern12(Exponential):
    pass


class Matern32(Prior):
    """
    Matern-3/2 kernel in SDE form
    Hyperparameters:
        variance, σ²
        lengthscale, l
    The associated continuous-time state space model matrices are:
    letting λ = √3/l
    F      = ( 0   1
              -λ² -2λ)
    L      = (0
              1)
    Qc     = 4λ³σ²
    H      = (1  0)
    Pinf   = (σ²  0
              0   λ²σ²)
    """
    def __init__(self, hyp=None):
        super().__init__(hyp=hyp)
        if self.hyp is None:
            print('using default kernel parameters since none were supplied')
            self.hyp = [0.55, 0.55]  # softplus(0.55) ~= 1
        self.name = 'Matern-3/2'

    @partial(jit, static_argnums=0)
    def cf_to_ss(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        if hyperparams is None:
            hyperparams = softplus(self.hyp)
        var, ell = hyperparams[0], hyperparams[1]
        lam = 3.0 ** 0.5 / ell
        F = np.array([[0.0,       1.0],
                      [-lam ** 2, -2 * lam]])
        L = np.array([[0],
                      [1]])
        Qc = np.array(12.0 * 3.0 ** 0.5 / ell ** 3.0 * var)
        H = np.array([[1.0, 0.0]])
        Pinf = np.array([[var, 0.0],
                         [0.0, 3.0 * var / ell ** 2.0]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def expm(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [2, 2]
        """
        if hyperparams is None:
            hyperparams = softplus(self.hyp)
        ell = hyperparams[1]
        lam = np.sqrt(3.0) / ell
        A = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
        return A


class Matern52(Prior):
    """
    Matern-5/2 kernel in SDE form
    Hyperparameters:
        variance, σ²
        lengthscale, l
    The associated continuous-time state space model matrices are:
    letting λ = √5/l
    F      = ( 0    1    0
               0    0    1
              -λ³ -3λ² -3λ)
    L      = (0
              0
              1)
    Qc     = 16λ⁵σ²/3
    H      = (1  0  0)
    letting κ = λ²σ²/3,
    Pinf   = ( σ²  0  -κ
               0   κ   0
              -κ   0   λ⁴σ²)
    """
    def __init__(self, hyp=None):
        super().__init__(hyp=hyp)
        if self.hyp is None:
            print('using default kernel parameters since none were supplied')
            self.hyp = [0.55, 0.55]  # softplus(0.55) ~= 1
        self.name = 'Matern-5/2'

    def set_hyperparams(self, hyp):
        self.hyp = hyp

    @partial(jit, static_argnums=0)
    def cf_to_ss(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        if hyperparams is None:
            hyperparams = softplus(self.hyp)
        var, ell = hyperparams[0], hyperparams[1]
        # lam = tf.constant(5.0**0.5 / ell, dtype=floattype)
        lam = 5.0**0.5 / ell
        F = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [-lam**3.0, -3.0*lam**2.0, -3.0*lam]])
        L = np.array([[0.0],
                      [0.0],
                      [1.0]])
        Qc = np.array(var * 400.0 * 5.0 ** 0.5 / 3.0 / ell ** 5.0)
        H = np.array([[1.0, 0.0, 0.0]])
        kappa = 5.0 / 3.0 * var / ell**2.0
        Pinf = np.array([[var,    0.0,   -kappa],
                         [0.0,    kappa, 0.0],
                         [-kappa, 0.0,   25.0*var / ell**4.0]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def expm(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-5/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [3, 3]
        """
        if hyperparams is None:
            hyperparams = softplus(self.hyp)
        ell = hyperparams[1]
        lam = np.sqrt(5.0) / ell
        dtlam = dt * lam
        A = np.exp(-dtlam) \
            * (dt * np.array([[lam * (0.5 * dtlam + 1.0),      dtlam + 1.0,            0.5 * dt],
                              [-0.5 * dtlam * lam ** 2,        lam * (1.0 - dtlam),    1.0 - 0.5 * dtlam],
                              [lam ** 3 * (0.5 * dtlam - 1.0), lam ** 2 * (dtlam - 3), lam * (0.5 * dtlam - 2.0)]])
               + np.eye(3))
        return A


class Matern72(Prior):
    """
    Matern-7/2 kernel in SDE form
    Hyperparameters:
        variance, σ²
        lengthscale, l
    The associated continuous-time state space model matrices are:
    letting λ = √7/l
    F      = ( 0    1    0    0
               0    0    1    0
               0    0    0    1
              -λ⁴ -4λ³ -6λ²  -4λ)
    L      = (0
              0
              0
              1)
    Qc     = 10976σ²√7/(5l⁷)
    H      = (1  0  0  0)
    letting κ = λ²σ²/5,
    and    κ₂ = 72σ²/l⁴
    Pinf   = ( σ²  0  -κ   0
               0   κ   0  -κ₂
               0  -κ₂  0   343σ²/l⁶)
    """
    def __init__(self, hyp=None):
        super().__init__(hyp=hyp)
        if self.hyp is None:
            print('using default kernel parameters since none were supplied')
            self.hyp = [0.55, 0.55]  # softplus(0.55) ~= 1
        self.name = 'Matern-7/2'

    @partial(jit, static_argnums=0)
    def cf_to_ss(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        if hyperparams is None:
            hyperparams = softplus(self.hyp)
        var, ell = hyperparams[0], hyperparams[1]
        lam = 7.0**0.5 / ell
        F = np.array([[0.0,       1.0,           0.0,           0.0],
                      [0.0,       0.0,           1.0,           0.0],
                      [0.0,       0.0,           0.0,           1.0],
                      [-lam**4.0, -4.0*lam**3.0, -6.0*lam**2.0, -4.0*lam]])
        L = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [1.0]])
        Qc = np.array(var * 10976.0 * 7.0 ** 0.5 / 5.0 / ell ** 7.0)
        H = np.array([[1, 0, 0, 0]])
        kappa = 7.0 / 5.0 * var / ell**2.0
        kappa2 = 9.8 * var / ell**4.0
        Pinf = np.array([[var,    0.0,     -kappa, 0.0],
                         [0.0,    kappa,   0.0,    -kappa2],
                         [-kappa, 0.0,     kappa2, 0.0],
                         [0.0,    -kappa2, 0.0,    343.0*var / ell**6.0]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def expm(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-7/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [4, 4]
        """
        if hyperparams is None:
            hyperparams = softplus(self.hyp)
        ell = hyperparams[1]
        lam = np.sqrt(7.0) / ell
        lam2 = lam * lam
        lam3 = lam2 * lam
        dtlam = dt * lam
        dtlam2 = dtlam ** 2
        A = np.exp(-dtlam) \
            * (dt * np.array([[lam * (1.0 + 0.5 * dtlam + dtlam * lam / 6.0), 1.0 + dtlam + 0.5 * dtlam2,
                              0.5 * dt * (1.0 + dtlam),                       dt ** 2 / 2],
                              [-dtlam2 * lam ** 2.0 / 6.0,                    lam * (1.0 + 0.5 * dtlam - 0.5 * dtlam2),
                              1.0 + dtlam - 0.5 * dtlam2,                     dt * (0.5 - dtlam / 6.0)],
                              [lam3 * dtlam * (dtlam / 6.0 - 0.5),            dtlam * lam2 * (0.5 * dtlam - 2.0),
                              lam * (1.0 - 2.5 * dtlam + 0.5 * dtlam2),       1.0 - dtlam + dtlam2 / 6.0],
                              [lam2 ** 2 * (dtlam - 1.0 - dtlam2 / 6.0),      lam3 * (3.5 * dtlam - 4.0 - 0.5 * dtlam2),
                              lam2 * (4.0 * dtlam - 6.0 - 0.5 * dtlam2),      lam * (1.5 * dtlam - 3.0 - dtlam2 / 6.0)]])
               + np.eye(4))
        return A
