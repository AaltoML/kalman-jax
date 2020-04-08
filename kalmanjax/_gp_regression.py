import numpy as np
import jax.numpy as jnp
from jax.nn import softplus
from utils import softplus_inv
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
from sde_gp import SDEGP
from approximate_inference import EP, PL, CL, IKS
import priors
import likelihoods
pi = 3.141592653589793

prior = priors.Matern52
lik = likelihoods.Gaussian


def wiggly_time_series(x_):
    noise_var = 0.15  # true observation noise
    return (np.cos(0.04*x_+0.33*pi) * np.sin(0.2*x_) +
            np.math.sqrt(noise_var) * np.random.normal(0, 1, x_.shape))


print('generating some data ...')
np.random.seed(12345)
N = 1000
# x = np.linspace(-25.0, 75.0, num=N)  # evenly spaced
x = np.random.permutation(np.linspace(-25.0, 150.0, num=N) + 0.5*np.random.randn(N))  # unevenly spaced
y = wiggly_time_series(x)
x_test = np.linspace(np.min(x)-15.0, np.max(x)+15.0, num=500)

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale
var_y = 0.5  # observation noise

theta_prior = jnp.array([var_f, len_f])
theta_lik = jnp.array(var_y)

prior_ = prior(theta_prior)
lik_ = lik(theta_lik)
# approx_inf_ = EP(power=0.5)
# approx_inf_ = PL()
approx_inf_ = CL(power=0.5)
# approx_inf_ = IKS()

sde_gp_model = SDEGP(prior=prior_, likelihood=lik_, x=x, y=y, x_test=x_test, approx_inf=approx_inf_)

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init(softplus_inv([theta_prior, theta_lik]))


def gradient_step(i, state, model):
    params = get_params(state)
    sde_gp_model.prior.hyp = params[0]
    sde_gp_model.likelihood.hyp = params[1]
    neg_log_marg_lik, gradients = model.run_model()
    print('iter %2d: var_f=%1.2f len_f=%1.2f var_y=%1.2f, nlml=%2.2f' %
          (i, softplus(params[0][0]), softplus(params[0][1]), softplus(params[1]), neg_log_marg_lik))
    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(20):
    opt_state = gradient_step(j, opt_state, sde_gp_model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var, _ = sde_gp_model.predict()
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

lb = posterior_mean[:, 0] - 1.96 * posterior_var[:, 0]**0.5
ub = posterior_mean[:, 0] + 1.96 * posterior_var[:, 0]**0.5
x_pred = sde_gp_model.t_all
test_id = sde_gp_model.test_id

print('sampling from the posterior ...')
t0 = time.time()
posterior_samp = sde_gp_model.posterior_sample(20)
t1 = time.time()
print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'k.', label='observations')
plt.plot(x_pred, posterior_mean, 'b', label='posterior mean')
plt.fill_between(x_pred, lb, ub, color='b', alpha=0.05, label='95% confidence')
plt.plot(sde_gp_model.t_test, posterior_samp[test_id, 0, :], 'b', alpha=0.15)
plt.xlim([sde_gp_model.t_test[0], sde_gp_model.t_test[-1]])
plt.legend()
plt.title('GP regression via Kalman smoothing')
plt.xlabel('time - $t$')
plt.show()
