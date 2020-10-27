import sys
sys.path.insert(0, '../')
import numpy as np
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
import pandas as pd
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list, plot

plot_intermediate = False

print('loading coal data ...')
disaster_timings = pd.read_csv('../../data/coal.txt', header=None).values[:, 0]

# Discretization
num_time_bins = 200
# Discretize the data
x = np.linspace(min(disaster_timings), max(disaster_timings), num_time_bins).T
y = np.histogram(disaster_timings, np.concatenate([[-1e10], x[:-1] + np.diff(x)/2, [1e10]]))[0][:, None]
# Test points
x_test = x

meanval = np.log(len(disaster_timings)/num_time_bins)  # TODO: incorporate mean

var_f = 1.0  # GP variance
len_f = 1.0  # GP lengthscale

prior = priors.Matern52(variance=var_f, lengthscale=len_f)
lik = likelihoods.Poisson()
# inf_method = approx_inf.EP(power=0.5)
# inf_method = approx_inf.SLEP()
inf_method = approx_inf.EKS()
# inf_method = approx_inf.EEP()
# inf_method = approx_inf.VI()

model = SDEGP(prior=prior, likelihood=lik, t=x, y=y, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()
    # neg_log_marg_lik, gradients = mod.run_two_stage()  # <-- less elegant but reduces compile time

    prior_params = softplus_list(params[0])
    print('iter %2d: var_f=%1.2f len_f=%1.2f, nlml=%2.2f' %
          (i, prior_params[0], prior_params[1], neg_log_marg_lik))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(200):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

x_plot = np.linspace(np.min(x_test)-5, np.max(x_test)+5, 200)
# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_cov = model.predict(t=x_plot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
# print('NLPD: %1.2f' % nlpd)

link_fn = model.likelihood.link_fn
scale = num_time_bins / (max(x) - min(x))
post_mean_lgcp = link_fn(posterior_mean + posterior_cov / 2) * scale
lb_lgcp = link_fn(posterior_mean - np.sqrt(posterior_cov) * 1.645) * scale
ub_lgcp = link_fn(posterior_mean + np.sqrt(posterior_cov) * 1.645) * scale

print('sampling from the posterior ...')
t0 = time.time()
posterior_samp = model.posterior_sample(20, t=x_plot)
post_samp_lgcp = link_fn(posterior_samp) * scale
t1 = time.time()
print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(disaster_timings, 0*disaster_timings, 'k+', label='observations', clip_on=False)
plt.plot(x_plot, post_mean_lgcp, 'g', label='posterior mean')
plt.fill_between(x_plot, lb_lgcp, ub_lgcp, color='g', alpha=0.05, label='95% confidence')
plt.plot(x_plot, post_samp_lgcp, 'g', alpha=0.15)
plt.xlim(x_plot[0], x_plot[-1])
plt.ylim(0.0)
plt.legend()
plt.title('log-Gaussian Cox process via Kalman smoothing (coal mining disasters)')
plt.xlabel('year')
plt.ylabel('accident intensity')
plt.show()
