import sys
sys.path.insert(0, '../')
import numpy as np
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list, discretegrid

plot_intermediate = False

print('loading rainforest data ...')
data = np.loadtxt('../../data/TRI2TU-data.csv', delimiter=',')

nr = 100  # spatial grid point (y-aixs)
nt = 200  # temporal grid points (x-axis)
scale = 1000 / nt

t, r, Y = discretegrid(data, [0, 1000, 0, 500], [nt, nr])

np.random.seed(99)
N = nr * nt  # number of data points

var_f = 1  # GP variance
len_f = 10  # lengthscale

prior = priors.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], fixed_grid=True)
lik = likelihoods.Poisson()
inf_method = approx_inf.ExtendedKalmanSmoother(damping=0.5)
# inf_method = approx_inf.ExtendedEP()

model = SDEGP(prior=prior, likelihood=lik, t=t, y=Y, r=r, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=2e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod, plot_num_, mu_prev_):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()
    # neg_log_marg_lik, gradients = mod.run_two_stage()  # <-- less elegant but reduces compile time

    prior_params = softplus_list(params[0])
    print('iter %2d: var=%1.2f len=%1.2f, nlml=%2.2f' %
          (i, prior_params[0], prior_params[1], neg_log_marg_lik))

    return opt_update(i, gradients, state), plot_num_, mu_prev_


plot_num = 0
mu_prev = None
print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(50):
    opt_state, plot_num, mu_prev = gradient_step(j, opt_state, model, plot_num, mu_prev)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
# nlpd = model.negative_log_predictive_density(t=t, r=r, y=Y)
mu, var = model.predict(t=t, r=r)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
# print('test NLPD: %1.2f' % nlpd)

link_fn = model.likelihood.link_fn

print('plotting ...')
plt.figure(1, figsize=(10, 5))
# im = plt.imshow(mu.T, extent=[0, 1000, 0, 500], origin='lower')
im = plt.imshow(link_fn(mu).T / scale, extent=[0, 1000, 0, 500], origin='lower')
plt.colorbar(im, fraction=0.0235, pad=0.04)
plt.xlim(0, 1000)
plt.ylim(0, 500)
# plt.title('2D log-Gaussian Cox process (rainforest tree data). Log-intensity shown.')
plt.title('2D log-Gaussian Cox process (rainforest tree data). Tree intensity per $m^2$.')
plt.xlabel('first spatial dimension, $t$ (metres)')
plt.ylabel('second spatial dimension, $r$ (metres)')
plt.show()
