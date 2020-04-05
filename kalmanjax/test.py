from jax import jit, partial, value_and_grad
from jax.scipy.linalg import cho_factor, cho_solve
from jax.ops import index, index_update
from jax.experimental import optimizers
import jax.numpy as np
from jax.nn import softplus
from jax.experimental import loops
pi = 3.141592653589793


class MyObject(object):
    def __init__(self):
        self.param = 1.0
        self.some_other_parameter = 1
        self.model_fixed_param = 0.5
        self.dt = np.linspace(1, 10, num=10)
        self.y = np.ones(10)
        self.update_model(self.param)
        self.minf = np.zeros([1, 1])
        self.site_params = None

    def update_model(self, param):
        # uses variance and lengthscale hyperparameters to construct the state space model
        self.F = np.array([[-1.0 / param]])
        self.H = np.array([[1.0]])
        self.Pinf = np.array([[param]])

    @partial(jit, static_argnums=0)
    def kalman_filter(self, params, site_params=None):
        self.update_model(softplus(params))
        N = self.dt.shape[0]
        if site_params is not None:
            site_mean, site_var = site_params
        with loops.Scope() as s:
            s.neg_log_marg_lik = 0.0  # negative log-marginal likelihood
            s.m, s.P = self.minf, self.Pinf
            s.filtered_mean = np.zeros([N, 1, 1])
            s.filtered_cov = np.zeros([N, 1, 1])
            s.site_mean = np.zeros([N, 1])
            s.site_var = np.zeros([N, 1])
            for k in s.range(N):
                y_k = self.y[k]
                A = np.exp(self.F)
                m_ = A @ s.m
                P_ = A @ (s.P - self.Pinf) @ A.T + self.Pinf
                mu = self.H @ m_
                var = self.H @ P_ @ self.H.T
                log_marg_lik_k, d1, d2 = self.moment_match(y_k, mu, var, softplus(params))
                if site_params is not None:
                    m_site, P_site = site_mean[k], site_var[k]
                else:
                    m_site = mu - d1 / d2  # approximate likelihood (site) mean (see Rasmussen & Williams p75)
                    P_site = -var - 1 / d2  # approximate likelihood (site) variance
                S = var + P_site
                L, low = cho_factor(S)
                K = (cho_solve((L, low), self.H @ P_)).T
                s.m = m_ + K @ (m_site - mu)
                s.P = P_ - K @ S @ K.T
                s.neg_log_marg_lik -= np.sum(log_marg_lik_k)
        return s.neg_log_marg_lik

    @partial(jit, static_argnums=0)
    def smoothing(self, params):
        self.update_model(softplus(params))
        N = self.dt.shape[0]
        with loops.Scope() as s:
            s.neg_log_marg_lik = 0.0  # negative log-marginal likelihood
            s.m, s.P = self.minf, self.Pinf
            s.filtered_mean = np.zeros([N, 1, 1])
            s.filtered_cov = np.zeros([N, 1, 1])
            s.site_mean = np.zeros([N, 1])
            s.site_var = np.zeros([N, 1])
            for k in s.range(N):
                y_k = self.y[k]
                A = np.exp(self.F)
                m_ = A @ s.m
                P_ = A @ (s.P - self.Pinf) @ A.T + self.Pinf
                mu = self.H @ m_
                var = self.H @ P_ @ self.H.T
                log_marg_lik_k, d1, d2 = self.moment_match(y_k, mu, var, softplus(params))
                m_site = mu - d1 / d2  # approximate likelihood (site) mean (see Rasmussen & Williams p75)
                P_site = -var - 1 / d2  # approximate likelihood (site) variance
                S = var + P_site
                L, low = cho_factor(S)
                K = (cho_solve((L, low), self.H @ P_)).T
                s.m = m_ + K @ (m_site - mu)
                s.P = P_ - K @ S @ K.T
                s.neg_log_marg_lik -= np.sum(log_marg_lik_k)
                s.filtered_mean = index_update(s.filtered_mean, index[k, ...], s.m)
                s.filtered_cov = index_update(s.filtered_cov, index[k, ...], s.P)
                s.site_mean = index_update(s.site_mean, index[k, ...], np.squeeze(m_site.T))
                s.site_var = index_update(s.site_var, index[k, ...], np.squeeze(P_site.T))
            s.smoothed_mean = np.zeros([N, 1])
            s.smoothed_var = np.zeros([N, 1])
            for k in s.range(N - 1, -1, -1):
                A = np.exp(self.F)
                m_predicted = A @ s.filtered_mean[k, ...]
                tmp_gain_cov = A @ s.filtered_cov[k, ...]
                P_predicted = A @ (s.filtered_cov[k, ...] - self.Pinf) @ A.T + self.Pinf
                P_predicted_chol, low = cho_factor(P_predicted)
                G_transpose = cho_solve((P_predicted_chol, low), tmp_gain_cov)
                s.m = s.filtered_mean[k, ...] + G_transpose.T @ (s.m - m_predicted)
                s.P = s.filtered_cov[k, ...] + G_transpose.T @ (s.P - P_predicted) @ G_transpose
                s.smoothed_mean = index_update(s.smoothed_mean, index[k, ...], np.squeeze((self.H @ s.m).T))
                s.smoothed_var = index_update(s.smoothed_var, index[k, ...],
                                              np.squeeze(np.diag(self.H @ s.P @ self.H.T)))
        return s.smoothed_mean, s.smoothed_var, (s.site_mean, s.site_var)

    @staticmethod
    def moment_match(y, m, v, hyp=None):
        lZ = (
                - (y - m) ** 2 / (hyp + v) / 2
                - np.log(np.maximum(2 * pi * (hyp + v), 1e-10)) / 2
        )
        dlZ = (y - m) / (hyp + v)  # 1st derivative w.r.t. mean
        d2lZ = -1 / (hyp + v)  # 2nd derivative w.r.t. mean
        return lZ, dlZ, d2lZ

    # def update_model(self, param):
    #     self.some_other_parameter = self.model_fixed_param + param

    @partial(jit, static_argnums=0)
    def loss(self, param):
        # self.update_model(param)
        # p1, p2 = self.some_func(param, 1.0)  # self.some_other_parameter)
        p1 = self.kalman_filter(param)  # self.some_other_parameter)
        return p1

    @partial(jit, static_argnums=0)
    def some_func(self, param, some_input):
        some_input = some_input + 1 + self.some_other_parameter
        self.some_other_parameter += 1.0
        return param ** 2, some_input


my_obj = MyObject()

opt_init, opt_update, get_params = optimizers.adam(step_size=3e-1)
opt_state = opt_init(5.0)


def gradient_step(i, state):
    param = get_params(state)
    # l, dl = value_and_grad(my_obj.loss, argnums=0)(param)
    l, dl = value_and_grad(my_obj.kalman_filter, argnums=0)(param, my_obj.site_params)
    _, _, my_obj.site_params = my_obj.smoothing(param)
    print(l)
    return opt_update(i, dl, state)


# l, dl = value_and_grad(my_obj.kalman_filter, argnums=0)(5.0)
for j in range(5):
    opt_state = gradient_step(j, opt_state)

# print(l)
# print(dl)
# print(my_obj.some_parameter)
