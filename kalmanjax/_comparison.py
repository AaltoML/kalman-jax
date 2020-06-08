import numpy as np
from jax.nn import softplus
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
from sde_gp import SDEGP
from approximate_inference import EP, PL
import priors
import likelihoods
pi = 3.141592653589793

prior = priors.Matern52
lik = likelihoods.Probit

print('generating some data ...')
np.random.seed(99)
N = 1000  # number of training points
x = 100 * np.random.rand(N)
f = 6 * np.sin(pi * x / 10.0) / (pi * x / 10.0 + 1)
y_ = f + np.math.sqrt(0.05)*np.random.randn(x.shape[0])
y = np.sign(y_)

x_test = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale

theta_prior = [var_f, len_f]

prior_ = prior(theta_prior)
lik_ = lik()
approx_inf_1 = EP(power=0.1)
approx_inf_2 = EP(power=0.9)

sde_gp_model_1 = SDEGP(prior=prior_, likelihood=lik_, x=x, y=y, x_test=x_test, approx_inf=approx_inf_1)
sde_gp_model_2 = SDEGP(prior=prior_, likelihood=lik_, x=x, y=y, x_test=x_test, approx_inf=approx_inf_2)

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([sde_gp_model_1.prior.hyp, sde_gp_model_1.likelihood.hyp])


def gradient_step(i, state, model):
    params = get_params(state)
    model.prior.hyp = params[0]
    model.likelihood.hyp = params[1]
    neg_log_marg_lik, gradients = model.run_model()
    print('iter %2d: var_f=%1.2f len_f=%1.2f, nlml=%2.2f' %
          (i, softplus(params[0][0]), softplus(params[0][1]), neg_log_marg_lik))
    return opt_update(i, gradients, state)


# print('optimising the hyperparameters ...')
# t0 = time.time()
# for j in range(20):
#     opt_state = gradient_step(j, opt_state, sde_gp_model_1)
# t1 = time.time()
# print('optimisation time: %2.2f secs' % (t1-t0))

for i in range(5):
    sde_gp_model_1.run_model()
    sde_gp_model_2.run_model()

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean_1, posterior_var_1, _ = sde_gp_model_1.predict()
posterior_mean_2, posterior_var_2, _ = sde_gp_model_2.predict()
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print(sde_gp_model_1.sites.site_params[0][100] - sde_gp_model_2.sites.site_params[0][100])
print(posterior_mean_1 - posterior_mean_2)

lb_1 = posterior_mean_1[:, 0] - 1.96 * posterior_var_1[:, 0]**0.5
ub_1 = posterior_mean_1[:, 0] + 1.96 * posterior_var_1[:, 0]**0.5
lb_2 = posterior_mean_2[:, 0] - 1.96 * posterior_var_2[:, 0]**0.5
ub_2 = posterior_mean_2[:, 0] + 1.96 * posterior_var_2[:, 0]**0.5
x_pred = sde_gp_model_1.t_all
test_id = sde_gp_model_1.test_id
link_fn = sde_gp_model_1.likelihood.link_fn

# print('sampling from the posterior ...')
# t0 = time.time()
# posterior_samp = sde_gp_model_1.posterior_sample(20)
# t1 = time.time()
# print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'b+', label='observations')
plt.plot(x_pred, link_fn(posterior_mean_1), 'm', label='posterior mean')
plt.plot(x_pred, link_fn(posterior_mean_2), 'g', label='posterior mean')
# plt.fill_between(x_pred, link_fn(lb_1), link_fn(ub_1), color='m', alpha=0.05, label='95% confidence')
plt.plot(x_pred, link_fn(lb_1), color='m', alpha=0.3)
plt.plot(x_pred, link_fn(ub_1), color='m', alpha=0.3)
plt.plot(x_pred, link_fn(lb_2), color='g', alpha=0.3)
plt.plot(x_pred, link_fn(ub_2), color='g', alpha=0.3)
# plt.plot(sde_gp_model_1.t_test, link_fn(posterior_samp[test_id, 0, :]), 'm', alpha=0.15)
plt.xlim(sde_gp_model_1.t_test[0], sde_gp_model_1.t_test[-1])
plt.legend()
plt.title('GP classification via Kalman smoothing')
plt.xlabel('time - $t$')
plt.show()
