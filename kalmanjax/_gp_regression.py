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
y_test = wiggly_time_series(x_test) + 0.5*np.random.randn(x_test.shape[0])

var_f = 1.0  # GP variance
len_f = 5.0  # GP lengthscale
var_y = 0.5  # observation noise

theta_prior = [var_f, len_f]
theta_lik = var_y

prior = priors.Matern52(theta_prior)
# prior_ = priors.QuasiPeriodicMatern32([var_f, len_f, 20., 50.])
lik = likelihoods.Gaussian(theta_lik)
inf_method = approx_inf.EP(power=0.5)
# inf_method = approx_inf.PL()
# inf_method = approx_inf.CL(power=0.5)
# inf_method = approx_inf.IKS()
# inf_method = approx_inf.EKS()
# inf_method = approx_inf.EKEP()
# inf_method = approx_inf.VI()

model = SDEGP(prior=prior, likelihood=lik, x=x, y=y, x_test=x_test, y_test=y_test, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()

    prior_params, lik_param = softplus_list(params[0]), softplus(params[1])
    print('iter %2d: var_f=%1.2f len_f=%1.2f var_y=%1.2f, nlml=%2.2f' %
          (i, prior_params[0], prior_params[1], lik_param, neg_log_marg_lik))

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
posterior_mean, posterior_var, _, nlpd = model.predict()
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('test NLPD: %1.2f' % nlpd)

lb = posterior_mean[:, 0] - 1.96 * posterior_var[:, 0]**0.5
ub = posterior_mean[:, 0] + 1.96 * posterior_var[:, 0]**0.5
x_pred = model.t_all[:, 0]
test_id = model.test_id

print('sampling from the posterior ...')
t0 = time.time()
posterior_samp = model.posterior_sample(20)
t1 = time.time()
print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'k.', label='training observations')
plt.plot(x_test, y_test, 'r.', alpha=0.4, label='test observations')
plt.plot(x_pred, posterior_mean, 'b', label='posterior mean')
plt.fill_between(x_pred, lb, ub, color='b', alpha=0.05, label='95% confidence')
plt.plot(model.t_test, posterior_samp[test_id, 0, :], 'b', alpha=0.15)
plt.xlim([model.t_test[0], model.t_test[-1]])
plt.legend()
plt.title('GP regression via Kalman smoothing. Test NLPD: %1.2f' % nlpd)
plt.xlabel('time - $t$')
plt.show()
