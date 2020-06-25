import sys
sys.path.insert(0, '../../')
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

plot_final = False
plot_intermediate = False

print('loading coal data ...')
if plot_final:
    disaster_timings = pd.read_csv('../../../data/coal.txt', header=None).values[:, 0]
cvind = np.loadtxt('cvind.csv').astype(int)
# 10-fold cross-validation
nt = np.floor(cvind.shape[0]/10).astype(int)
cvind = np.reshape(cvind[:10*nt], (10, nt))

D = np.loadtxt('binned.csv')
x = D[:, 0:1]
y = D[:, 1:]
N = D.shape[0]

np.random.seed(123)
# meanval = np.log(len(disaster_timings)/num_time_bins)  # TODO: incorporate mean

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    method = 0
    fold = 0

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = cvind[fold, :]
ind_train = np.setdiff1d(cvind, ind_test)

x_train = x[ind_train, ...]  # 90/10 train/test split
x_test = x[ind_test, ...]
y_train = y[ind_train, ...]
y_test = y[ind_test, ...]

var_f = 1.0  # GP variance
len_f = 1.0  # GP lengthscale

theta_prior = [var_f, len_f]

prior = priors.Matern52(theta_prior)
lik = likelihoods.Poisson()

if method == 0:
    inf_method = approx_inf.EEP(power=1)
elif method == 1:
    inf_method = approx_inf.EEP(power=0.5)
elif method == 2:
    inf_method = approx_inf.EKS()

elif method == 3:
    inf_method = approx_inf.UEP(power=1)
elif method == 4:
    inf_method = approx_inf.UEP(power=0.5)
elif method == 5:
    inf_method = approx_inf.UKS()

elif method == 6:
    inf_method = approx_inf.GHEP(power=1)
elif method == 7:
    inf_method = approx_inf.GHEP(power=0.5)
elif method == 8:
    inf_method = approx_inf.GHKS()

elif method == 9:
    inf_method = approx_inf.EP(power=1, intmethod='UT')
elif method == 10:
    inf_method = approx_inf.EP(power=0.5, intmethod='UT')
elif method == 11:
    inf_method = approx_inf.EP(power=0.01, intmethod='UT')

elif method == 12:
    inf_method = approx_inf.EP(power=1, intmethod='GH')
elif method == 13:
    inf_method = approx_inf.EP(power=0.5, intmethod='GH')
elif method == 14:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH')

elif method == 15:
    inf_method = approx_inf.VI(intmethod='UT')
elif method == 16:
    inf_method = approx_inf.VI(intmethod='GH')

model = SDEGP(prior=prior, likelihood=lik, x=x_train, y=y_train, x_test=x_test, y_test=y_test, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=2.5e-1)
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
for j in range(300):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var, _, nlpd = model.predict()
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('NLPD: %1.2f' % nlpd)

if plot_final:
    x_pred = model.t_all[:, 0]
    link_fn = model.likelihood.link_fn
    scale = N / (max(x_pred) - min(x_pred))
    post_mean_lgcp = link_fn(posterior_mean[:, 0] + posterior_var[:, 0] / 2) * scale
    lb_lgcp = link_fn(posterior_mean[:, 0] - np.sqrt(posterior_var[:, 0]) * 1.645) * scale
    ub_lgcp = link_fn(posterior_mean[:, 0] + np.sqrt(posterior_var[:, 0]) * 1.645) * scale
    test_id = model.test_id

    # print('sampling from the posterior ...')
    # t0 = time.time()
    # posterior_samp = model.posterior_sample(20)
    # post_samp_lgcp = link_fn(posterior_samp[test_id, 0, :]) * scale
    # t1 = time.time()
    # print('sampling time: %2.2f secs' % (t1-t0))

    print('plotting ...')
    plt.figure(1, figsize=(12, 5))
    plt.clf()
    plt.plot(disaster_timings, 0*disaster_timings, 'k+', label='observations', clip_on=False)
    plt.plot(x_pred, post_mean_lgcp, 'g', label='posterior mean')
    plt.fill_between(x_pred, lb_lgcp, ub_lgcp, color='g', alpha=0.05, label='95% confidence')
    # plt.plot(model.t_test, post_samp_lgcp, 'g', alpha=0.15)
    plt.xlim(model.t_test[0], model.t_test[-1])
    plt.ylim(0.0)
    plt.legend()
    plt.title('log-Gaussian Cox process via Kalman smoothing (coal mining disasters)')
    plt.xlabel('year')
    plt.ylabel('accident intensity')
    plt.show()
