import numpy as np
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list, plot
pi = 3.141592653589793

plot_intermediate = False

print('generating some data ...')
np.random.seed(99)
N = 100000  # number of training points
x = 100 * np.random.rand(N)
f = 6 * np.sin(pi * x / 10.0) / (pi * x / 10.0 + 1)
y_ = f + np.math.sqrt(0.05)*np.random.randn(x.shape[0])
y = np.sign(y_)

x_test = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale

theta_prior = [var_f, len_f]

prior = priors.Matern52(theta_prior)
lik = likelihoods.Probit()
inf_method = approx_inf.EP(power=0.5)
# inf_method = approx_inf.PL()
# inf_method = approx_inf.EKEP()  <-- not working
# inf_method = approx_inf.VI()

model = SDEGP(prior=prior, likelihood=lik, x=x, y=y, x_test=x_test, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()

    prior_params = softplus_list(params[0])
    print('iter %2d: var_f=%1.2f len_f=%1.2f, nlml=%2.2f' %
          (i, prior_params[0], prior_params[1], neg_log_marg_lik))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(20):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var, _ = model.predict()
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

lb = posterior_mean[:, 0] - 1.96 * posterior_var[:, 0]**0.5
ub = posterior_mean[:, 0] + 1.96 * posterior_var[:, 0]**0.5
x_pred = model.t_all
test_id = model.test_id
link_fn = model.likelihood.link_fn

print('sampling from the posterior ...')
t0 = time.time()
posterior_samp = model.posterior_sample(20)
t1 = time.time()
print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'b+', label='observations')
plt.plot(x_pred, link_fn(posterior_mean), 'm', label='posterior mean')
plt.fill_between(x_pred, link_fn(lb), link_fn(ub), color='m', alpha=0.05, label='95% confidence')
plt.plot(model.t_test, link_fn(posterior_samp[test_id, 0, :]), 'm', alpha=0.15)
plt.xlim(model.t_test[0], model.t_test[-1])
plt.legend()
plt.title('GP classification via Kalman smoothing')
plt.xlabel('time - $t$')
plt.show()
