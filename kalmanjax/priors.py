import jax.numpy as np
from jax import jit, partial
from jax.scipy.linalg import expm
from utils import softplus, softplus_inv, softplus_list, rotation_matrix, solve
from kernels import Matern12Kernel, Matern32Kernel, Matern52Kernel


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
        self.hyp = softplus_inv(np.array(hyp))

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        raise NotImplementedError('kernel to state space mapping not implemented for this prior')

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        raise NotImplementedError('measurement model not implemented for this prior')

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt).
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters [array]
        :return: state transition matrix A [sd, sd]
        """
        F, _, _, _, _ = self.kernel_to_state_space(hyperparams)
        A = expm(F * dt)
        return A


class Exponential(Prior):
    """
    Exponential, i.e. Matern-1/2 kernel in SDE form.
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
    def __init__(self, variance=1.0, lengthscale=1.0):
        hyp = [variance, lengthscale]
        super().__init__(hyp=hyp)
        self.name = 'Exponential'

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell = hyperparams[0], hyperparams[1]
        F = np.array([[-1.0 / ell]])
        L = np.array([[1.0]])
        Qc = np.array([[2.0 * var / ell]])
        H = np.array([[1.0]])
        Pinf = np.array([[var]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H = np.array([[1.0]])
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the exponential prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [1, 1]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams[1]
        A = np.broadcast_to(np.exp(-dt / ell), [1, 1])
        return A


class Matern12(Exponential):
    pass


class Matern32(Prior):
    """
    Matern-3/2 kernel in SDE form.
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

    def __init__(self, variance=1.0, lengthscale=1.0):
        hyp = [variance, lengthscale]
        super().__init__(hyp=hyp)
        self.name = 'Matern-3/2'

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell = hyperparams[0], hyperparams[1]
        lam = 3.0 ** 0.5 / ell
        F = np.array([[0.0,       1.0],
                      [-lam ** 2, -2 * lam]])
        L = np.array([[0],
                      [1]])
        Qc = np.array([[12.0 * 3.0 ** 0.5 / ell ** 3.0 * var]])
        H = np.array([[1.0, 0.0]])
        Pinf = np.array([[var, 0.0],
                         [0.0, 3.0 * var / ell ** 2.0]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H = np.array([[1.0, 0.0]])
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [2, 2]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams[1]
        lam = np.sqrt(3.0) / ell
        A = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
        return A


class Matern52(Prior):
    """
    Matern-5/2 kernel in SDE form.
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

    def __init__(self, variance=1.0, lengthscale=1.0):
        hyp = [variance, lengthscale]
        super().__init__(hyp=hyp)
        self.name = 'Matern-5/2'

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell = hyperparams[0], hyperparams[1]
        lam = 5.0**0.5 / ell
        F = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [-lam**3.0, -3.0*lam**2.0, -3.0*lam]])
        L = np.array([[0.0],
                      [0.0],
                      [1.0]])
        Qc = np.array([[var * 400.0 * 5.0 ** 0.5 / 3.0 / ell ** 5.0]])
        H = np.array([[1.0, 0.0, 0.0]])
        kappa = 5.0 / 3.0 * var / ell**2.0
        Pinf = np.array([[var,    0.0,   -kappa],
                         [0.0,    kappa, 0.0],
                         [-kappa, 0.0,   25.0*var / ell**4.0]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H = np.array([[1.0, 0.0, 0.0]])
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [3, 3]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
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
    Matern-7/2 kernel in SDE form.
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

    def __init__(self, variance=1.0, lengthscale=1.0):
        hyp = [variance, lengthscale]
        super().__init__(hyp=hyp)
        self.name = 'Matern-7/2'

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
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
        Qc = np.array([[var * 10976.0 * 7.0 ** 0.5 / 5.0 / ell ** 7.0]])
        H = np.array([[1, 0, 0, 0]])
        kappa = 7.0 / 5.0 * var / ell**2.0
        kappa2 = 9.8 * var / ell**4.0
        Pinf = np.array([[var,    0.0,     -kappa, 0.0],
                         [0.0,    kappa,   0.0,    -kappa2],
                         [-kappa, 0.0,     kappa2, 0.0],
                         [0.0,    -kappa2, 0.0,    343.0*var / ell**6.0]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H = np.array([[1, 0, 0, 0]])
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-7/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [4, 4]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams[1]
        lam = np.sqrt(7.0) / ell
        lam2 = lam * lam
        lam3 = lam2 * lam
        dtlam = dt * lam
        dtlam2 = dtlam ** 2
        A = np.exp(-dtlam) \
            * (dt * np.array([[lam * (1.0 + 0.5 * dtlam + dtlam2 / 6.0),      1.0 + dtlam + 0.5 * dtlam2,
                              0.5 * dt * (1.0 + dtlam),                       dt ** 2 / 6],
                              [-dtlam2 * lam ** 2.0 / 6.0,                    lam * (1.0 + 0.5 * dtlam - 0.5 * dtlam2),
                              1.0 + dtlam - 0.5 * dtlam2,                     dt * (0.5 - dtlam / 6.0)],
                              [lam3 * dtlam * (dtlam / 6.0 - 0.5),            dtlam * lam2 * (0.5 * dtlam - 2.0),
                              lam * (1.0 - 2.5 * dtlam + 0.5 * dtlam2),       1.0 - dtlam + dtlam2 / 6.0],
                              [lam2 ** 2 * (dtlam - 1.0 - dtlam2 / 6.0),      lam3 * (3.5 * dtlam - 4.0 - 0.5 * dtlam2),
                              lam2 * (4.0 * dtlam - 6.0 - 0.5 * dtlam2),      lam * (1.5 * dtlam - 3.0 - dtlam2 / 6.0)]])
               + np.eye(4))
        return A


class Cosine(Prior):
    """
    Cosine kernel in SDE form.
    Hyperparameters:
        radial frequency, ω
    The associated continuous-time state space model matrices are:
    F      = ( 0   -ω
               ω    0 )
    L      = N/A
    Qc     = N/A
    H      = ( 1  0 )
    Pinf   = ( 1  0
               0  1 )
    and the discrete-time transition matrix is (for step size Δt),
    A      = ( cos(ωΔt)   -sin(ωΔt)
               sin(ωΔt)    cos(ωΔt) )
    """
    def __init__(self, frequency=1.0):
        super().__init__(hyp=frequency)
        self.name = 'Cosine'
        self.F, self.L, self.Qc, self.H, self.Pinf = self.kernel_to_state_space(self.hyp)

    @property
    def frequency(self):
        return softplus(self.hyp)

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        omega = hyperparams[0]
        F = np.array([[0.0,   -omega],
                      [omega, 0.0]])
        H = np.array([[1.0, 0.0]])
        L = []
        Qc = []
        Pinf = np.eye(2)
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H = np.array([[1.0, 0.0]])
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Cosine prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [M+1, 1]
        :param hyperparams: hyperparameters of the prior: frequency [1, 1]
        :return: state transition matrix A [M+1, D, D]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        omega = hyperparams[0]
        state_transitions = rotation_matrix(dt, omega)  # [2, 2]
        return state_transitions


class SubbandMatern12(Prior):
    """
    Subband Matern-1/2 (i.e. Exponential) kernel in SDE form (product of Cosine and Matern-1/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the exponential and cosine components:
    F      = F_exp ⊕ F_cos  =  ( -1/l  -ω
                                 ω     -1/l )
    L      = L_exp ⊗ I      =  ( 1      0
                                 0      1 )
    Qc     = I ⊗ Qc_exp     =  ( 2σ²/l  0
                                 0      2σ²/l )
    H      = H_exp ⊗ H_cos  =  ( 1      0 )
    Pinf   = Pinf_exp ⊗ I   =  ( σ²     0
                                 0      σ² )
    and the discrete-time transition matrix is (for step size Δt),
    A      = exp(-Δt/l) ( cos(ωΔt)   -sin(ωΔt)
                          sin(ωΔt)    cos(ωΔt) )
    """
    def __init__(self, variance=1.0, lengthscale=1.0, radial_frequency=1.0):
        hyp = [variance, lengthscale, radial_frequency]
        super().__init__(hyp=hyp)
        self.name = 'Subband Matern-1/2'
        self.F, self.L, self.Qc, self.H, self.Pinf = self.kernel_to_state_space(self.hyp)

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @property
    def radial_frequency(self):
        return softplus(self.hyp[2])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell, omega = hyperparams
        F_mat = np.array([[-1.0 / ell]])
        L_mat = np.array([[1.0]])
        Qc_mat = np.array([[2.0 * var / ell]])
        H_mat = np.array([[1.0]])
        Pinf_mat = np.array([[var]])
        F_cos = np.array([[0.0, -omega],
                          [omega, 0.0]])
        H_cos = np.array([[1.0, 0.0]])
        # F = (-1/l -ω
        #      ω    -1/l)
        F = np.kron(F_mat, np.eye(2)) + F_cos
        L = np.kron(L_mat, np.eye(2))
        Qc = np.kron(np.eye(2), Qc_mat)
        H = np.kron(H_mat, H_cos)
        Pinf = np.kron(Pinf_mat, np.eye(2))
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H_mat = np.array([[1.0]])
        H_cos = np.array([[1.0, 0.0]])
        H = np.kron(H_mat, H_cos)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-1/2 prior:
        A = exp(-Δt/l) ( cos(ωΔt)   -sin(ωΔt)
                         sin(ωΔt)    cos(ωΔt) )
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: variance, lengthscale, frequency [3, 1]
        :return: state transition matrix A [2, 2]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell, omega = hyperparams[1], hyperparams[2]
        R = rotation_matrix(dt, omega)
        A = np.exp(-dt / ell) * R  # [2, 2]
        return A


class SubbandExponential(SubbandMatern12):
    pass


class SubbandMatern32(Prior):
    """
    Subband Matern-3/2 kernel in SDE form (product of Cosine and Matern-3/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the Matern3/2 and cosine components:
    letting λ = √3 / l
    F      = F_mat3/2 ⊕ F_cos  =  ( 0     -ω     1     0
                                    ω      0     0     1
                                   -λ²     0    -2λ   -ω
                                    0     -λ²    ω    -2λ )
    L      = L_mat3/2 ⊗ I      =  ( 0      0
                                    0      0
                                    1      0
                                    0      1 )
    Qc     = I ⊗ Qc_mat3/2     =  ( 4λ³σ²  0
                                    0      4λ³σ² )
    H      = H_mat3/2 ⊗ H_cos  =  ( 1      0     0      0 )
    Pinf   = Pinf_mat3/2 ⊗ I   =  ( σ²     0     0      0
                                    0      σ²    0      0
                                    0      0     3σ²/l² 0
                                    0      0     0      3σ²/l²)
    and the discrete-time transition matrix is (for step size Δt),
    R = ( cos(ωΔt)   -sin(ωΔt)
          sin(ωΔt)    cos(ωΔt) )
    A = exp(-Δt/l) ( (1+Δtλ)R   ΔtR
                     -Δtλ²R    (1-Δtλ)R )
    """

    def __init__(self, variance=1.0, lengthscale=1.0, radial_frequency=1.0):
        hyp = [variance, lengthscale, radial_frequency]
        super().__init__(hyp=hyp)
        self.name = 'Subband Matern-3/2'
        self.F, self.L, self.Qc, self.H, self.Pinf = self.kernel_to_state_space(self.hyp)

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @property
    def radial_frequency(self):
        return softplus(self.hyp[2])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell, omega = hyperparams
        lam = 3.0 ** 0.5 / ell
        F_mat = np.array([[0.0, 1.0],
                          [-lam ** 2, -2 * lam]])
        L_mat = np.array([[0],
                          [1]])
        Qc_mat = np.array([[12.0 * 3.0 ** 0.5 / ell ** 3.0 * var]])
        H_mat = np.array([[1.0, 0.0]])
        Pinf_mat = np.array([[var, 0.0],
                             [0.0, 3.0 * var / ell ** 2.0]])
        F_cos = np.array([[0.0, -omega],
                          [omega, 0.0]])
        H_cos = np.array([[1.0, 0.0]])
        # F = (0   -ω   1   0
        #      ω    0   0   1
        #      -λ²  0  -2λ -ω
        #      0   -λ²  ω  -2λ)
        F = np.kron(F_mat, np.eye(2)) + np.kron(np.eye(2), F_cos)
        L = np.kron(L_mat, np.eye(2))
        Qc = np.kron(np.eye(2), Qc_mat)
        H = np.kron(H_mat, H_cos)
        Pinf = np.kron(Pinf_mat, np.eye(2))
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H_mat = np.array([[1.0, 0.0]])
        H_cos = np.array([[1.0, 0.0]])
        H = np.kron(H_mat, H_cos)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: variance, lengthscale, frequency [3, 1]
        :return: state transition matrix A [4, 4]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell, omega = hyperparams[1], hyperparams[2]
        lam = np.sqrt(3.0) / ell
        R = rotation_matrix(dt, omega)
        A = np.exp(-dt * lam) * np.block([
            [(1. + dt * lam) * R, dt * R],
            [-dt * lam ** 2 * R, (1. - dt * lam) * R]
        ])
        return A


class SubbandMatern52(Prior):
    """
    Subband Matern-5/2 kernel in SDE form (product of Cosine and Matern-5/2).
    Hyperparameters:
        variance, σ²
        lengthscale, l
        radial frequency, ω
    The associated continuous-time state space model matrices are constructed via
    kronecker sums and products of the Matern5/2 and cosine components:
    letting λ = √5/l
    F      = F_mat5/2 ⊕ F_cos  =  ( 0    -ω     1     0     0     0
                                    ω     0     0     1     0     0
                                    0     0     0    -ω     1     0
                                    0     0     ω     0     0     1
                                   -λ³    0    -3λ²   0    -3λ   -ω
                                    0    -λ³    0     3λ²   w    -3λ )
    L      = L_mat5/2 ⊗ I      =  ( 0     0     0
                                    0     0     0
                                    0     0     0
                                    1     0     0
                                    0     1     0
                                    0     0     1 )
    Qc     = I ⊗ Qc_mat5/2     =  ( 16λ⁵σ²/3  0
                                    0         16λ⁵σ²/3 )
    H      = H_mat5/2 ⊗ H_cos  =  ( 1     0     0      0     0    0 )
    letting κ = λ²σ²/3
    Pinf   = Pinf_mat5/2 ⊗ I   =  ( σ²    0     0      0    -κ     0
                                    0     σ²    0      0     0    -κ
                                    0     0     κ      0     0     0
                                    0     0     0      κ     0     0
                                   -κ     0     0      0     λ⁴σ²  0
                                    0    -κ     0      0     0     λ⁴σ² )
    and the discrete-time transition matrix is (for step size Δt),
    R = ( cos(ωΔt)   -sin(ωΔt)
          sin(ωΔt)    cos(ωΔt) )
    A = exp(-Δt/l) ( 1/2(2+Δtλ(2+Δtλ))R   Δt(1+Δtλ)R         1/2Δt²R
                    -1/2Δt²λ³R           (1+Δtλ(1-Δtλ))R    -1/2Δt(-2+Δtλ)R
                     1/2Δtλ³(-2+Δtλ)R     Δt²(-3+Δtλ)R       1/2(2+Δtλ(-4+Δtλ))R )
    """

    def __init__(self, variance=1.0, lengthscale=1.0, radial_frequency=1.0):
        hyp = [variance, lengthscale, radial_frequency]
        super().__init__(hyp=hyp)
        self.name = 'Subband Matern-5/2'
        self.F, self.L, self.Qc, self.H, self.Pinf = self.kernel_to_state_space(self.hyp)

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @property
    def radial_frequency(self):
        return softplus(self.hyp[2])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell, omega = hyperparams
        lam = 5.0 ** 0.5 / ell
        F_mat = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [-lam ** 3.0, -3.0 * lam ** 2.0, -3.0 * lam]])
        L_mat = np.array([[0.0],
                          [0.0],
                          [1.0]])
        Qc_mat = np.array([[var * 400.0 * 5.0 ** 0.5 / 3.0 / ell ** 5.0]])
        H_mat = np.array([[1.0, 0.0, 0.0]])
        kappa = 5.0 / 3.0 * var / ell ** 2.0
        Pinf_mat = np.array([[var, 0.0, -kappa],
                             [0.0, kappa, 0.0],
                             [-kappa, 0.0, 25.0 * var / ell ** 4.0]])
        F_cos = np.array([[0.0, -omega],
                          [omega, 0.0]])
        H_cos = np.array([[1.0, 0.0]])
        # F = (0   -ω   1    0    0   0
        #      ω    0   0    1    0   0
        #      0    0   0   -ω    1   0
        #      0    0   ω    0    0   1
        #      -λ³  0  -3λ²  0   -3λ -ω
        #      0   -λ³  0   -3λ²  ω  -3λ )
        F = np.kron(F_mat, np.eye(2)) + np.kron(np.eye(3), F_cos)
        L = np.kron(L_mat, np.eye(2))
        Qc = np.kron(np.eye(2), Qc_mat)
        H = np.kron(H_mat, H_cos)
        Pinf = np.kron(Pinf_mat, np.eye(2))
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H_mat = np.array([[1.0, 0.0, 0.0]])
        H_cos = np.array([[1.0, 0.0]])
        H = np.kron(H_mat, H_cos)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Subband Matern-5/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: variance, lengthscale, frequency [3, 1]
        :return: state transition matrix A [6, 6]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell, omega = hyperparams[1], hyperparams[2]
        lam = 5.0 ** 0.5 / ell
        R = rotation_matrix(dt, omega)
        A = np.exp(-dt * lam) * np.block([
            [0.5*(2. + dt*lam*(2. + dt*lam)) * R, dt * (1. + dt * lam) * R,        0.5 * dt**2 * R],
            [-0.5*dt ** 2 * lam**3 * R,           (1. + dt*lam*(1. - dt*lam)) * R, -0.5 * dt * (-2. + dt * lam) * R],
            [0.5*dt*lam**3 * (-2. + dt*lam) * R,  dt * lam**2*(-3. + dt*lam) * R,  0.5*(2. + dt*lam*(-4. + dt*lam)) * R]
        ])
        return A


class Periodic(Prior):
    """
    Periodic kernel in SDE form.
    Hyperparameters:
        variance, σ²
        lengthscale, l
        period, p
    The associated continuous-time state space model matrices are constructed via
    a sum of cosines.
    """
    def __init__(self, variance=1.0, lengthscale=1.0, period=1.0, order=6):
        hyp = [variance, lengthscale, period]
        super().__init__(hyp=hyp)
        self.name = 'Periodic'
        self.order = order
        self.K = np.meshgrid(np.arange(self.order + 1), np.arange(self.order + 1))[1]
        factorial_mesh_K = np.array([[1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1.],
                                     [2., 2., 2., 2., 2., 2., 2.],
                                     [6., 6., 6., 6., 6., 6., 6.],
                                     [24., 24., 24., 24., 24., 24., 24.],
                                     [120., 120., 120., 120., 120., 120., 120.],
                                     [720., 720., 720., 720., 720., 720., 720.]])
        b = np.array([[1., 0., 0., 0., 0., 0., 0.],
                      [0., 2., 0., 0., 0., 0., 0.],
                      [2., 0., 2., 0., 0., 0., 0.],
                      [0., 6., 0., 2., 0., 0., 0.],
                      [6., 0., 8., 0., 2., 0., 0.],
                      [0., 20., 0., 10., 0., 2., 0.],
                      [20., 0., 30., 0., 12., 0., 2.]])
        self.b_fmK_2K = b * (1. / factorial_mesh_K) * (2. ** -self.K)
        self.F, self.L, self.Qc, self.H, self.Pinf = self.kernel_to_state_space(self.hyp)

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @property
    def period(self):
        return softplus(self.hyp[2])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell, period = hyperparams
        a = self.b_fmK_2K * ell ** (-2. * self.K) * np.exp(-1. / ell ** 2.) * var
        q2 = np.sum(a, axis=0)
        # The angular frequency
        omega = 2 * np.pi / period
        # The model
        F = np.kron(np.diag(np.arange(self.order + 1)), np.array([[0., -omega], [omega, 0.]]))
        L = np.eye(2 * (self.order + 1))
        Qc = np.zeros(2 * (self.order + 1))
        Pinf = np.kron(np.diag(q2), np.eye(2))
        H = np.kron(np.ones([1, self.order + 1]), np.array([1., 0.]))
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H = np.kron(np.ones([1, self.order + 1]), np.array([1., 0.]))
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Periodic prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: variance, lengthscale, period [3, 1]
        :return: state transition matrix A [2(N+1), 2(N+1)]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell, period = hyperparams[1], hyperparams[2]
        # The angular frequency
        omega = 2 * np.pi / period
        harmonics = np.arange(self.order + 1) * omega
        R0 = rotation_matrix(dt, harmonics[0])
        R1 = rotation_matrix(dt, harmonics[1])
        R2 = rotation_matrix(dt, harmonics[2])
        R3 = rotation_matrix(dt, harmonics[3])
        R4 = rotation_matrix(dt, harmonics[4])
        R5 = rotation_matrix(dt, harmonics[5])
        R6 = rotation_matrix(dt, harmonics[6])
        A = np.block([
            [R0, np.zeros([2, 12])],
            [np.zeros([2, 2]),  R1, np.zeros([2, 10])],
            [np.zeros([2, 4]),  R2, np.zeros([2, 8])],
            [np.zeros([2, 6]),  R3, np.zeros([2, 6])],
            [np.zeros([2, 8]),  R4, np.zeros([2, 4])],
            [np.zeros([2, 10]), R5, np.zeros([2, 2])],
            [np.zeros([2, 12]), R6]
        ])
        return A


class QuasiPeriodicMatern12(Prior):
    """
    Quasi-periodic kernel in SDE form (product of Periodic and Matern-1/2).
    Hyperparameters:
        variance, σ²
        lengthscale of Periodic, l_p
        period, p
        lengthscale of Matern, l_m
    The associated continuous-time state space model matrices are constructed via
    a sum of cosines times a Matern-1/2.
    """
    def __init__(self, variance=1.0, lengthscale_periodic=1.0, period=1.0, lengthscale_matern=1.0, order=6):
        hyp = [variance, lengthscale_periodic, period, lengthscale_matern]
        super().__init__(hyp=hyp)
        self.name = 'Quasi-Periodic Exponential'
        self.order = order
        self.K = np.meshgrid(np.arange(self.order + 1), np.arange(self.order + 1))[1]
        factorial_mesh_K = np.array([[1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1.],
                                     [2., 2., 2., 2., 2., 2., 2.],
                                     [6., 6., 6., 6., 6., 6., 6.],
                                     [24., 24., 24., 24., 24., 24., 24.],
                                     [120., 120., 120., 120., 120., 120., 120.],
                                     [720., 720., 720., 720., 720., 720., 720.]])
        b = np.array([[1., 0., 0., 0., 0., 0., 0.],
                      [0., 2., 0., 0., 0., 0., 0.],
                      [2., 0., 2., 0., 0., 0., 0.],
                      [0., 6., 0., 2., 0., 0., 0.],
                      [6., 0., 8., 0., 2., 0., 0.],
                      [0., 20., 0., 10., 0., 2., 0.],
                      [20., 0., 30., 0., 12., 0., 2.]])
        factorial_mesh_K = factorial_mesh_K[:self.order + 1, :self.order + 1]
        b = b[:self.order + 1, :self.order + 1]
        self.b_fmK_2K = b * (1. / factorial_mesh_K) * (2. ** -self.K)
        self.F, self.L, self.Qc, self.H, self.Pinf = self.kernel_to_state_space(self.hyp)

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale_periodic(self):
        return softplus(self.hyp[1])

    @property
    def period(self):
        return softplus(self.hyp[2])

    @property
    def lengthscale_matern(self):
        return softplus(self.hyp[3])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell_p, period, ell_m = hyperparams
        var_p = 1.
        a = self.b_fmK_2K * ell_p ** (-2. * self.K) * np.exp(-1. / ell_p ** 2.) * var_p
        q2 = np.sum(a, axis=0)
        # The angular frequency
        omega = 2 * np.pi / period
        # The model
        F_p = np.kron(np.diag(np.arange(self.order + 1)), np.array([[0., -omega], [omega, 0.]]))
        L_p = np.eye(2 * (self.order + 1))
        # Qc_p = np.zeros(2 * (self.N + 1))
        Pinf_p = np.kron(np.diag(q2), np.eye(2))
        H_p = np.kron(np.ones([1, self.order + 1]), np.array([1., 0.]))
        F_m = np.array([[-1.0 / ell_m]])
        L_m = np.array([[1.0]])
        Qc_m = np.array([[2.0 * var / ell_m]])
        H_m = np.array([[1.0]])
        Pinf_m = np.array([[var]])
        F = np.kron(F_m, np.eye(2 * (self.order + 1))) + F_p
        L = np.kron(L_m, L_p)
        Qc = np.kron(Pinf_p, Qc_m)
        H = np.kron(H_m, H_p)
        Pinf = np.kron(Pinf_m, Pinf_p)
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H_p = np.kron(np.ones([1, self.order + 1]), np.array([1., 0.]))
        H_m = np.array([[1.0]])
        H = np.kron(H_m, H_p)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Quasi-Periodic Exponential prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: variance, lengthscale, period [3, 1]
        :return: state transition matrix A [2*(N+1), 2*(N+1)]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        period, ell_m = hyperparams[2], hyperparams[3]
        # The angular frequency
        omega = 2 * np.pi / period
        harmonics = np.arange(self.order + 1) * omega
        R0 = rotation_matrix(dt, harmonics[0])
        R1 = rotation_matrix(dt, harmonics[1])
        R2 = rotation_matrix(dt, harmonics[2])
        R3 = rotation_matrix(dt, harmonics[3])
        R4 = rotation_matrix(dt, harmonics[4])
        R5 = rotation_matrix(dt, harmonics[5])
        R6 = rotation_matrix(dt, harmonics[6])
        A = np.exp(-dt / ell_m) * np.block([
            [R0, np.zeros([2, 12])],
            [np.zeros([2, 2]),  R1, np.zeros([2, 10])],
            [np.zeros([2, 4]),  R2, np.zeros([2, 8])],
            [np.zeros([2, 6]),  R3, np.zeros([2, 6])],
            [np.zeros([2, 8]),  R4, np.zeros([2, 4])],
            [np.zeros([2, 10]), R5, np.zeros([2, 2])],
            [np.zeros([2, 12]), R6]
        ])
        return A


class QuasiPeriodicExponential(QuasiPeriodicMatern12):
    pass


class QuasiPeriodicMatern32(Prior):
    """
    Quasi-periodic kernel in SDE form (product of Periodic and Matern-3/2).
    Hyperparameters:
        variance, σ²
        lengthscale of Periodic, l_p
        period, p
        lengthscale of Matern, l_m
    The associated continuous-time state space model matrices are constructed via
    a sum of cosines times a Matern-3/2.
    """
    def __init__(self, variance=1.0, lengthscale_periodic=1.0, period=1.0, lengthscale_matern=1.0, order=6):
        hyp = [variance, lengthscale_periodic, period, lengthscale_matern]
        super().__init__(hyp=hyp)
        self.name = 'Periodic'
        self.order = order
        self.K = np.meshgrid(np.arange(self.order + 1), np.arange(self.order + 1))[1]
        factorial_mesh_K = np.array([[1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1.],
                                     [2., 2., 2., 2., 2., 2., 2.],
                                     [6., 6., 6., 6., 6., 6., 6.],
                                     [24., 24., 24., 24., 24., 24., 24.],
                                     [120., 120., 120., 120., 120., 120., 120.],
                                     [720., 720., 720., 720., 720., 720., 720.]])
        b = np.array([[1., 0., 0., 0., 0., 0., 0.],
                      [0., 2., 0., 0., 0., 0., 0.],
                      [2., 0., 2., 0., 0., 0., 0.],
                      [0., 6., 0., 2., 0., 0., 0.],
                      [6., 0., 8., 0., 2., 0., 0.],
                      [0., 20., 0., 10., 0., 2., 0.],
                      [20., 0., 30., 0., 12., 0., 2.]])
        self.b_fmK_2K = b * (1. / factorial_mesh_K) * (2. ** -self.K)
        self.F, self.L, self.Qc, self.H, self.Pinf = self.kernel_to_state_space(self.hyp)

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale_periodic(self):
        return softplus(self.hyp[1])

    @property
    def period(self):
        return softplus(self.hyp[2])

    @property
    def lengthscale_matern(self):
        return softplus(self.hyp[3])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell_p, period, ell_m = hyperparams
        var_p = 1.
        a = self.b_fmK_2K * ell_p ** (-2. * self.K) * np.exp(-1. / ell_p ** 2.) * var_p
        q2 = np.sum(a, axis=0)
        # The angular frequency
        omega = 2 * np.pi / period
        # The model
        F_p = np.kron(np.diag(np.arange(self.order + 1)), np.array([[0., -omega], [omega, 0.]]))
        L_p = np.eye(2 * (self.order + 1))
        # Qc_p = np.zeros(2 * (self.N + 1))
        Pinf_p = np.kron(np.diag(q2), np.eye(2))
        H_p = np.kron(np.ones([1, self.order + 1]), np.array([1., 0.]))
        lam = 3.0 ** 0.5 / ell_m
        F_m = np.array([[0.0, 1.0],
                        [-lam ** 2, -2 * lam]])
        L_m = np.array([[0],
                        [1]])
        Qc_m = np.array([[12.0 * 3.0 ** 0.5 / ell_m ** 3.0 * var]])
        H_m = np.array([[1.0, 0.0]])
        Pinf_m = np.array([[var, 0.0],
                           [0.0, 3.0 * var / ell_m ** 2.0]])
        # F = np.kron(F_p, np.eye(2)) + np.kron(np.eye(14), F_m)
        F = np.kron(F_m, np.eye(2 * (self.order + 1))) + np.kron(np.eye(2), F_p)
        L = np.kron(L_m, L_p)
        Qc = np.kron(Qc_m, Pinf_p)
        H = np.kron(H_m, H_p)
        # Pinf = np.kron(Pinf_m, Pinf_p)
        Pinf = np.block([
            [np.kron(Pinf_m, q2[0] * np.eye(2)), np.zeros([4, 24])],
            [np.zeros([4, 4]), np.kron(Pinf_m, q2[1] * np.eye(2)), np.zeros([4, 20])],
            [np.zeros([4, 8]), np.kron(Pinf_m, q2[2] * np.eye(2)), np.zeros([4, 16])],
            [np.zeros([4, 12]), np.kron(Pinf_m, q2[3] * np.eye(2)), np.zeros([4, 12])],
            [np.zeros([4, 16]), np.kron(Pinf_m, q2[4] * np.eye(2)), np.zeros([4, 8])],
            [np.zeros([4, 20]), np.kron(Pinf_m, q2[5] * np.eye(2)), np.zeros([4, 4])],
            [np.zeros([4, 24]), np.kron(Pinf_m, q2[6] * np.eye(2))]
        ])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H_p = np.kron(np.ones([1, self.order + 1]), np.array([1., 0.]))
        H_m = np.array([[1.0, 0.0]])
        H = np.kron(H_m, H_p)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the closed form discrete-time state
        transition matrix A = expm(FΔt) for the Quasi-Periodic Matern-3/2 prior
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [M+1, 1]
        :param hyperparams: hyperparameters of the prior: variance, lengthscale, period [3, 1]
        :return: state transition matrix A [M+1, D, D]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        period, ell_m = hyperparams[2], hyperparams[3]
        lam = np.sqrt(3.0) / ell_m
        # The angular frequency
        omega = 2 * np.pi / period
        harmonics = np.arange(self.order + 1) * omega
        R0 = self.subband_mat32(dt, lam, harmonics[0])
        R1 = self.subband_mat32(dt, lam, harmonics[1])
        R2 = self.subband_mat32(dt, lam, harmonics[2])
        R3 = self.subband_mat32(dt, lam, harmonics[3])
        R4 = self.subband_mat32(dt, lam, harmonics[4])
        R5 = self.subband_mat32(dt, lam, harmonics[5])
        R6 = self.subband_mat32(dt, lam, harmonics[6])
        A = np.exp(-dt * lam) * np.block([
            [R0, np.zeros([4, 24])],
            [np.zeros([4, 4]),  R1, np.zeros([4, 20])],
            [np.zeros([4, 8]),  R2, np.zeros([4, 16])],
            [np.zeros([4, 12]), R3, np.zeros([4, 12])],
            [np.zeros([4, 16]), R4, np.zeros([4, 8])],
            [np.zeros([4, 20]), R5, np.zeros([4, 4])],
            [np.zeros([4, 24]), R6]
        ])
        return A

    @staticmethod
    def subband_mat32(dt, lam, omega):
        R = rotation_matrix(dt, omega)
        Ri = np.block([
            [(1. + dt * lam) * R, dt * R],
            [-dt * lam ** 2 * R,  (1. - dt * lam) * R]
        ])
        return Ri


class SpatioTemporalMatern52(Prior):
    """
    Spatio-Temporal Matern-5/2 kernel in SDE form.
    Hyperparameters:
        variance, σ²
        temporal lengthscale, lt
        spatial lengthscale, ls
    """
    def __init__(self, variance=1.0, lengthscale_time=1.0, lengthscale_space=1.0, spatial_dims=1, z=None, fixed_grid=False):
        hyp = [variance, lengthscale_time, lengthscale_space]
        super().__init__(hyp=hyp)
        self.spatial_dims = spatial_dims
        if z is None:
            z = np.linspace(-3., 3., num=15)
            if self.spatial_dims > 1:  # Quick, inelegant fix for |r| = 2
                z = np.linspace(-3., 3., num=5)
                zA, zB = np.meshgrid(z,z)  # Adding additional dimension to inducing points grid
                z = np.hstack((zA.reshape(-1,1), zB.reshape(-1,1)))  # Flattening grid for use in kernel functions
            elif self.spatial_dims > 2:
                raise NotImplementedError('initialisation of z with more than 2 dimensions not implemented, please provide a custom initialisation') 
        self.z = z.reshape(-1, self.spatial_dims)
        self.M = self.z.shape[0]
        self.fixed_grid = fixed_grid
        self.spatial_kernel = Matern52Kernel()
        self.name = 'Spatio-Temporal Matern-5/2'

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale_time(self):
        return softplus(self.hyp[1])

    @property
    def lengthscale_space(self):
        return softplus(self.hyp[2])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell_time, ell_space = hyperparams[0], hyperparams[1], hyperparams[2]
        Kmm = self.spatial_kernel.K(self.z, self.z, ell_space)
        lam = 5.0**0.5 / ell_time
        F_time = np.array([[0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [-lam**3.0, -3.0*lam**2.0, -3.0*lam]])
        F = np.kron(np.eye(self.M), F_time)
        L = np.array([[0.0],
                      [0.0],
                      [1.0]])
        Qc = np.array([[var * 400.0 * 5.0 ** 0.5 / 3.0 / ell_time ** 5.0]])
        H = None
        kappa = 5.0 / 3.0 * var / ell_time**2.0
        Pinf_time = np.array([[var,    0.0,   -kappa],
                              [0.0,    kappa, 0.0],
                              [-kappa, 0.0,   25.0*var / ell_time**4.0]])
        Pinf = np.kron(Kmm, Pinf_time)
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell_space = hyperparams[2]
        H_time = np.array([[1.0, 0.0, 0.0]])
        if self.fixed_grid:
            Kx = np.eye(self.z.shape[0])
        else:
            Kzz = self.spatial_kernel.K(self.z, self.z, ell_space)
            Kxz = self.spatial_kernel.K(r.reshape(-1, 1), self.z, ell_space)
            Kx = solve(Kzz, Kxz.T).T  # Kxz / Kzz
        H = np.kron(Kx, H_time)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the spatio-temporal Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [3M, 3M]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams[1]
        lam = np.sqrt(5.0) / ell
        dtlam = dt * lam
        A_time = np.exp(-dtlam) \
            * (dt * np.array([[lam * (0.5 * dtlam + 1.0),      dtlam + 1.0,            0.5 * dt],
                              [-0.5 * dtlam * lam ** 2,        lam * (1.0 - dtlam),    1.0 - 0.5 * dtlam],
                              [lam ** 3 * (0.5 * dtlam - 1.0), lam ** 2 * (dtlam - 3), lam * (0.5 * dtlam - 2.0)]])
               + np.eye(3))
        A = np.kron(np.eye(self.M), A_time)
        return A


class SpatialMatern52(Prior):
    """
    Spatial Matern-5/2 kernel in SDE form. Similar to the spatio-temporal kernel but the
    lengthscale is shared across dimensions.
    Hyperparameters:
        variance, σ²
        lengthscale, l
    """
    def __init__(self, variance=1.0, lengthscale=1.0, z=None, fixed_grid=False):
        hyp = [variance, lengthscale]
        super().__init__(hyp=hyp)
        if z is None:
            z = np.linspace(-3., 3., num=15)
        self.z = z.reshape(-1, 1)
        self.M = self.z.shape[0]
        self.fixed_grid = fixed_grid
        self.spatial_kernel = Matern52Kernel()
        self.name = 'Spatial Matern-5/2'

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell = hyperparams[0], hyperparams[1]
        Kmm = self.spatial_kernel.K(self.z, self.z, ell)
        lam = 5.0**0.5 / ell
        F_time = np.array([[0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [-lam**3.0, -3.0*lam**2.0, -3.0*lam]])
        F = np.kron(np.eye(self.M), F_time)
        L = np.array([[0.0],
                      [0.0],
                      [1.0]])
        Qc = np.array([[var * 400.0 * 5.0 ** 0.5 / 3.0 / ell ** 5.0]])
        H = None
        kappa = 5.0 / 3.0 * var / ell**2.0
        Pinf_time = np.array([[var,    0.0,   -kappa],
                              [0.0,    kappa, 0.0],
                              [-kappa, 0.0,   25.0*var / ell**4.0]])
        Pinf = np.kron(Kmm, Pinf_time)
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams[1]
        H_time = np.array([[1.0, 0.0, 0.0]])
        if self.fixed_grid:
            Kx = np.eye(self.z.shape[0])
        else:
            Kzz = self.spatial_kernel.K(self.z, self.z, ell)
            Kxz = self.spatial_kernel.K(r.reshape(-1, 1), self.z, ell)
            Kx = solve(Kzz, Kxz.T).T  # Kxz / Kzz
        H = np.kron(Kx, H_time)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the spatial Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [3M, 3M]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams[1]
        lam = np.sqrt(5.0) / ell
        dtlam = dt * lam
        A_time = np.exp(-dtlam) \
            * (dt * np.array([[lam * (0.5 * dtlam + 1.0),      dtlam + 1.0,            0.5 * dt],
                              [-0.5 * dtlam * lam ** 2,        lam * (1.0 - dtlam),    1.0 - 0.5 * dtlam],
                              [lam ** 3 * (0.5 * dtlam - 1.0), lam ** 2 * (dtlam - 3), lam * (0.5 * dtlam - 2.0)]])
               + np.eye(3))
        A = np.kron(np.eye(self.M), A_time)
        return A


class SpatialMatern32(Prior):
    """
    Spatial Matern-3/2 kernel in SDE form. Similar to the spatio-temporal kernel but the
    lengthscale is shared across dimensions.
    Hyperparameters:
        variance, σ²
        lengthscale, l
    """
    def __init__(self, variance=1.0, lengthscale=1.0, z=None, fixed_grid=False):
        hyp = [variance, lengthscale]
        super().__init__(hyp=hyp)
        if z is None:
            z = np.linspace(-3., 3., num=15)
        self.z = z.reshape(-1, 1)
        self.M = self.z.shape[0]
        self.fixed_grid = fixed_grid
        self.spatial_kernel = Matern32Kernel()
        self.name = 'Spatial Matern-3/2'

    @property
    def variance(self):
        return softplus(self.hyp[0])

    @property
    def lengthscale(self):
        return softplus(self.hyp[1])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        var, ell = hyperparams[0], hyperparams[1]
        Kmm = self.spatial_kernel.K(self.z, self.z, ell)
        lam = 3.0 ** 0.5 / ell
        F_time = np.array([[0.0, 1.0],
                           [-lam ** 2, -2 * lam]])
        F = np.kron(np.eye(self.M), F_time)
        L = np.array([[0.0],
                      [1.0]])
        Qc = np.array([[12.0 * 3.0 ** 0.5 / ell ** 3.0 * var]])
        H = None
        Pinf_time = np.array([[var, 0.0],
                              [0.0, 3.0 * var / ell ** 2.0]])
        Pinf = np.kron(Kmm, Pinf_time)
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams[1]
        H_time = np.array([[1.0, 0.0]])
        if self.fixed_grid:
            Kx = np.eye(self.z.shape[0])
        else:
            Kzz = self.spatial_kernel.K(self.z, self.z, ell)
            Kxz = self.spatial_kernel.K(r.reshape(-1, 1), self.z, ell)
            Kx = solve(Kzz, Kxz.T).T  # Kxz / Kzz
        H = np.kron(Kx, H_time)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the spatial Matern-3/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :param hyperparams: the kernel hyperparameters, lengthscale is in index 1 [2]
        :return: state transition matrix A [2M, 2M]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams[1]
        lam = np.sqrt(3.0) / ell
        A_time = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
        A = np.kron(np.eye(self.M), A_time)
        return A


class Sum(object):
    """
    A sum of GP priors. 'components' is a list of GP kernels, and this class stacks
    the state space models to produce their sum.
    """
    def __init__(self, components):
        hyp = [components[0].hyp]
        for i in range(1, len(components)):
            hyp = hyp + [components[i].hyp]
        self.components = components
        self.hyp = hyp
        self.name = 'Sum'

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus_list(self.hyp) if hyperparams is None else hyperparams
        F, L, Qc, H, Pinf = self.components[0].kernel_to_state_space(hyperparams[0])
        for i in range(1, len(self.components)):
            F_, L_, Qc_, H_, Pinf_ = self.components[i].kernel_to_state_space(hyperparams[i])
            F = np.block([
                [F, np.zeros([F.shape[0], F_.shape[1]])],
                [np.zeros([F_.shape[0],   F.shape[1]]), F_]
            ])
            L = np.block([
                [L, np.zeros([L.shape[0], L_.shape[1]])],
                [np.zeros([L_.shape[0],   L.shape[1]]), L_]
            ])
            Qc = np.block([
                [Qc,                     np.zeros([Qc.shape[0], Qc_.shape[1]])],
                [np.zeros([Qc_.shape[0], Qc.shape[1]]), Qc_]
            ])
            H = np.block([
                H, H_
            ])
            Pinf = np.block([
                [Pinf, np.zeros([Pinf.shape[0],             Pinf_.shape[1]])],
                [np.zeros([Pinf_.shape[0], Pinf.shape[1]]), Pinf_]
            ])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        hyperparams = softplus_list(self.hyp) if hyperparams is None else hyperparams
        H = self.components[0].measurement_model(r, hyperparams[0])
        for i in range(1, len(self.components)):
            H_ = self.components[i].measurement_model(r, hyperparams[i])
            H = np.block([
                H, H_
            ])
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a sum of GPs
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: [array]
        :return: state transition matrix A [D, D]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        A = self.components[0].state_transition(dt, hyperparams[0])
        for i in range(1, len(self.components)):
            A_ = self.components[i].state_transition(dt, hyperparams[i])
            A = np.block([
                [A, np.zeros([A.shape[0], A_.shape[0]])],
                [np.zeros([A_.shape[0], A.shape[0]]), A_]
            ])
        return A


class Independent(object):
    """
    A stack of independent GP priors. 'components' is a list of GP kernels, and this class stacks
    the state space models such that each component is fed to the likelihood.
    This class differs from Sum only in the measurement model.
    """
    def __init__(self, components):
        hyp = [components[0].hyp]
        for i in range(1, len(components)):
            hyp = hyp + [components[i].hyp]
        self.components = components
        self.hyp = hyp
        self.name = 'Independent'

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus_list(self.hyp) if hyperparams is None else hyperparams
        F, L, Qc, H, Pinf = self.components[0].kernel_to_state_space(hyperparams[0])
        for i in range(1, len(self.components)):
            F_, L_, Qc_, H_, Pinf_ = self.components[i].kernel_to_state_space(hyperparams[i])
            F = np.block([
                [F, np.zeros([F.shape[0], F_.shape[1]])],
                [np.zeros([F_.shape[0],   F.shape[1]]), F_]
            ])
            L = np.block([
                [L, np.zeros([L.shape[0], L_.shape[1]])],
                [np.zeros([L_.shape[0],   L.shape[1]]), L_]
            ])
            Qc = np.block([
                [Qc,                     np.zeros([Qc.shape[0], Qc_.shape[1]])],
                [np.zeros([Qc_.shape[0], Qc.shape[1]]), Qc_]
            ])
            H = np.block([
                [H, np.zeros([H.shape[0], H_.shape[1]])],
                [np.zeros([H_.shape[0],   H.shape[1]]), H_]
            ])
            Pinf = np.block([
                [Pinf, np.zeros([Pinf.shape[0],             Pinf_.shape[1]])],
                [np.zeros([Pinf_.shape[0], Pinf.shape[1]]), Pinf_]
            ])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        hyperparams = softplus_list(self.hyp) if hyperparams is None else hyperparams
        H = self.components[0].measurement_model(r, hyperparams[0])
        for i in range(1, len(self.components)):
            H_ = self.components[i].measurement_model(r, hyperparams[i])
            H = np.block([
                [H, np.zeros([H.shape[0], H_.shape[1]])],
                [np.zeros([H_.shape[0], H.shape[1]]), H_]
            ])
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for a sum of GPs
        :param dt: step size(s), Δt = tₙ - tₙ₋₁ [1]
        :param hyperparams: hyperparameters of the prior: [array]
        :return: state transition matrix A [D, D]
        """
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        A = self.components[0].state_transition(dt, hyperparams[0])
        for i in range(1, len(self.components)):
            A_ = self.components[i].state_transition(dt, hyperparams[i])
            A = np.block([
                [A, np.zeros([A.shape[0], A_.shape[0]])],
                [np.zeros([A_.shape[0], A.shape[0]]), A_]
            ])
        return A


class Separate(Independent):
    pass


class Stack(Independent):
    pass


class SubbandExponentialFixedVar(Prior):
    def __init__(self, variance=1.0, lengthscale=1.0, radial_frequency=1.0):
        hyp = [lengthscale, radial_frequency]
        self.variance = variance
        super().__init__(hyp=hyp)
        self.name = 'Subband Matern-1/2'
        self.F, self.L, self.Qc, self.H, self.Pinf = self.kernel_to_state_space(self.hyp)

    @property
    def lengthscale(self):
        return softplus(self.hyp[0])

    @property
    def radial_frequency(self):
        return softplus(self.hyp[1])

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell, omega = hyperparams
        var = self.variance
        F_mat = np.array([[-1.0 / ell]])
        L_mat = np.array([[1.0]])
        Qc_mat = np.array([[2.0 * var / ell]])
        H_mat = np.array([[1.0]])
        Pinf_mat = np.array([[var]])
        F_cos = np.array([[0.0, -omega],
                           [omega, 0.0]])
        H_cos = np.array([[1.0, 0.0]])
        # F = (-1/l -ω
        #      ω    -1/l)
        F = np.kron(F_mat, np.eye(2)) + F_cos
        L = np.kron(L_mat, np.eye(2))
        Qc = np.kron(np.eye(2), Qc_mat)
        H = np.kron(H_mat, H_cos)
        Pinf = np.kron(Pinf_mat, np.eye(2))
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H_mat = np.array([[1.0]])
        H_cos = np.array([[1.0, 0.0]])
        H = np.kron(H_mat, H_cos)
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell, omega = hyperparams[0], hyperparams[1]
        R = rotation_matrix(dt, omega)
        A = np.exp(-dt / ell) * R  # [2, 2]
        return A


class Matern52FixedVar(Prior):
    def __init__(self, variance=1.0, lengthscale=1.0):
        hyp = lengthscale
        self.variance = variance
        super().__init__(hyp=hyp)
        self.name = 'Matern-5/2'

    @property
    def lengthscale(self):
        return softplus(self.hyp)

    @partial(jit, static_argnums=0)
    def kernel_to_state_space(self, hyperparams=None):
        # uses variance and lengthscale hyperparameters to construct the state space model
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams
        var = self.variance
        lam = 5.0**0.5 / ell
        F = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [-lam**3.0, -3.0*lam**2.0, -3.0*lam]])
        L = np.array([[0.0],
                       [0.0],
                       [1.0]])
        Qc = np.array([[var * 400.0 * 5.0 ** 0.5 / 3.0 / ell ** 5.0]])
        H = np.array([[1.0, 0.0, 0.0]])
        kappa = 5.0 / 3.0 * var / ell**2.0
        Pinf = np.array([[var,    0.0,   -kappa],
                          [0.0,    kappa, 0.0],
                          [-kappa, 0.0,   25.0*var / ell**4.0]])
        return F, L, Qc, H, Pinf

    @partial(jit, static_argnums=0)
    def measurement_model(self, r=None, hyperparams=None):
        H = np.array([[1.0, 0.0, 0.0]])
        return H

    @partial(jit, static_argnums=0)
    def state_transition(self, dt, hyperparams=None):
        hyperparams = softplus(self.hyp) if hyperparams is None else hyperparams
        ell = hyperparams
        lam = np.sqrt(5.0) / ell
        dtlam = dt * lam
        A = np.exp(-dtlam) \
            * (dt * np.array([[lam * (0.5 * dtlam + 1.0),      dtlam + 1.0,            0.5 * dt],
                               [-0.5 * dtlam * lam ** 2,        lam * (1.0 - dtlam),    1.0 - 0.5 * dtlam],
                               [lam ** 3 * (0.5 * dtlam - 1.0), lam ** 2 * (dtlam - 3), lam * (0.5 * dtlam - 2.0)]])
               + np.eye(3))
        return A
