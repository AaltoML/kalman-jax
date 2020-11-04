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
from utils import softplus_list, plot
pi = 3.141592653589793

plot_intermediate = False

print('generating some data ...')
np.random.seed(99)
N = 1000  # number of training points
x = 100 * np.random.rand(N)
f = lambda x_: 6 * np.sin(pi * x_ / 10.0) / (pi * x_ / 10.0 + 1)
y_ = f(x) + np.math.sqrt(0.05)*np.random.randn(x.shape[0])
y = np.sign(y_)
y[y == -1] = 0
x_test = np.linspace(np.min(x)-5.0, np.max(x)+5.0, num=500)
y_test = np.sign(f(x_test) + np.math.sqrt(0.05)*np.random.randn(x_test.shape[0]))

y_test[y_test == -1] = 0

var_f = 1.  # GP variance
len_f = 5.0  # GP lengthscale

prior = priors.Matern52(variance=var_f, lengthscale=len_f)

lik = likelihoods.Bernoulli(link='logit')
inf_method = approx_inf.ExpectationPropagation(power=0.9, intmethod='UT')
# inf_method = approx_inf.VariationalInference(intmethod='GH')
# inf_method = approx_inf.VariationalInference(intmethod='UT')
# inf_method = approx_inf.ExtendedEP(power=0)
# inf_method = approx_inf.ExtendedKalmanSmoother()
# inf_method = approx_inf.GaussHermiteKalmanSmoother()
# inf_method = approx_inf.StatisticallyLinearisedEP(intmethod='UT')
# inf_method = approx_inf.UnscentedKalmanSmoother()

model = SDEGP(prior=prior, likelihood=lik, t=x, y=y, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=2e-1)
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
for j in range(100):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

x_plot = np.linspace(np.min(x)-10.0, np.max(x)+10.0, num=500)
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
link_fn = model.likelihood.link_fn

print('sampling from the posterior ...')
t0 = time.time()
posterior_samp = model.posterior_sample(20, t=x_plot)
t1 = time.time()
print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'b+', label='training observations')
plt.plot(x_test, y_test, 'r+', alpha=0.4, label='test observations')
plt.plot(x_plot, link_fn(posterior_mean), 'm', label='posterior mean')
plt.fill_between(x_plot, link_fn(lb), link_fn(ub), color='m', alpha=0.05, label='95% confidence')
plt.plot(x_plot, link_fn(posterior_samp), 'm', alpha=0.15)
plt.xlim(x_plot[0], x_plot[-1])
plt.legend()
plt.title('GP classification via Kalman smoothing. Test NLPD: %1.2f' % nlpd)
plt.xlabel('time - $t$')
plt.show()
