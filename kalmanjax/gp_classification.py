import numpy as np
import jax.numpy as jnp
from jax.nn import softplus
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
from sde_gp import SDEGP
import priors
import likelihoods
pi = 3.141592653589793

kern = priors.Matern52
lik = likelihoods.Probit

print('generating some data ...')
np.random.seed(99)
N = 10000  # number of training points
x = 100 * np.random.rand(N)
f = 6 * np.sin(pi * x / 10.0) / (pi * x / 10.0 + 1)
y_ = f + np.math.sqrt(0.05)*np.random.randn(x.shape[0])
y = np.sign(y_)

x_test = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)


def softplus_inv(x_):
    return jnp.log(jnp.exp(x_) - 1)


var_f = softplus_inv(1.0)  # GP variance
len_f = softplus_inv(5.0)  # GP lengthscale

theta_prior = jnp.array([var_f, len_f])
theta_lik = jnp.array([])

sde_gp_model = SDEGP(prior=kern, likelihood=lik, x=x, y=y, theta_prior=theta_prior, x_test=x_test)

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-1)
opt_state = opt_init([theta_prior, theta_lik])  # parameters should be a 2-element list [param_prior, param_likelihood]


def gradient_step(i, state):
    params = get_params(state)
    sde_gp_model.prior.hyp = params[0]
    sde_gp_model.likelihood.hyp = params[1]
    neg_log_marg_lik, gradients = sde_gp_model.neg_log_marg_lik()
    print('iter %2d: var_f=%1.2f len_f=%1.2f, nlml=%2.2f' %
          (i, softplus(params[0][0]), softplus(params[0][1]), neg_log_marg_lik))
    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(20):
    opt_state = gradient_step(j, opt_state)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# t0 = time.time()
# nlml, [dtheta_prior, dtheta_lik] = sde_gp_model.neg_log_marg_lik()
# t1 = time.time()
# print('NLML: %0.2f' % nlml)
# print('gradients:', dtheta_prior, dtheta_lik)
# print('gradient step time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var, _, _ = sde_gp_model.predict()
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

lb = posterior_mean[:, 0] - 1.96 * posterior_var[:, 0]**0.5
ub = posterior_mean[:, 0] + 1.96 * posterior_var[:, 0]**0.5
x_pred = sde_gp_model.t_all
test_id = sde_gp_model.test_id
link_fn = sde_gp_model.likelihood.link_fn

print('sampling from the posterior ...')
t0 = time.time()
posterior_samp = sde_gp_model.posterior_sample(20)
t1 = time.time()
print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'b+', label='observations')
plt.plot(x_pred, link_fn(posterior_mean), 'm', label='posterior mean')
plt.fill_between(x_pred, link_fn(lb), link_fn(ub), color='m', alpha=0.05, label='95% confidence')
plt.plot(x_test, link_fn(posterior_samp[test_id, 0, :]), 'm', alpha=0.15)
plt.xlim(x_test[0], x_test[-1])
plt.legend()
plt.title('GP classification via Kalman smoothing')
plt.xlabel('time - $t$')
plt.show()
