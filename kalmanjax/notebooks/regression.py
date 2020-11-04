import sys
sys.path.insert(0, '../')
import numpy as np
from jax.nn import softplus
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list, plot
pi = 3.141592653589793


def wiggly_time_series(x_):
    noise_var = 0.15  # true observation noise
    return (np.cos(0.04*x_+0.33*pi) * np.sin(0.2*x_) +
            np.math.sqrt(noise_var) * np.random.normal(0, 1, x_.shape))


plot_intermediate = False

print('generating some data ...')
np.random.seed(12345)
N = 1000
# x = np.linspace(-25.0, 75.0, num=N)  # evenly spaced
x = np.random.permutation(np.linspace(-25.0, 150.0, num=N) + 0.5*np.random.randn(N))  # unevenly spaced
y = wiggly_time_series(x)
x_test = np.linspace(np.min(x)-15.0, np.max(x)+15.0, num=500)
y_test = wiggly_time_series(x_test)

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale
var_y = 0.5  # observation noise

prior = priors.Matern52(variance=var_f, lengthscale=len_f)
# prior_ = priors.QuasiPeriodicMatern32([var_f, len_f, 20., 50.])
lik = likelihoods.Gaussian(variance=var_y)
inf_method = approx_inf.EP(power=0.5)
# inf_method = approx_inf.EKS()
# inf_method = approx_inf.EEP()
# inf_method = approx_inf.VI()

model = SDEGP(prior=prior, likelihood=lik, t=x, y=y, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()
    # neg_log_marg_lik, gradients = mod.run_two_stage()  # <-- less elegant but reduces compile time

    prior_params, lik_param = softplus_list(params[0]), softplus(params[1])
    print('iter %2d: var_f=%1.2f len_f=%1.2f var_y=%1.2f, nlml=%2.2f' %
          (i, prior_params[0], prior_params[1], lik_param, neg_log_marg_lik))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(100):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

x_plot = np.linspace(np.min(x)-20.0, np.max(x)+20.0, 200)
# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(t=x_test, y=y_test)
posterior_mean, posterior_cov = model.predict(t=x_plot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('test NLPD: %1.2f' % nlpd)

lb = posterior_mean - 1.96 * posterior_cov ** 0.5
ub = posterior_mean + 1.96 * posterior_cov ** 0.5

print('sampling from the posterior ...')
t0 = time.time()
posterior_samp = model.posterior_sample(20, t=x_plot)
t1 = time.time()
print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'k.', label='training observations')
plt.plot(x_test, y_test, 'r.', alpha=0.4, label='test observations')
plt.plot(x_plot, posterior_mean, 'b', label='posterior mean')
plt.fill_between(x_plot, lb, ub, color='b', alpha=0.05, label='95% confidence')
plt.plot(x_plot, posterior_samp, 'b', alpha=0.15)
plt.xlim([x_plot[0], x_plot[-1]])
plt.legend()
plt.title('GP regression via Kalman smoothing. Test NLPD: %1.2f' % nlpd)
plt.xlabel('time - $t$')
plt.show()
