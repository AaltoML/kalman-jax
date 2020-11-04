import sys
sys.path.insert(0, '../../')
import numpy as np
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import plot
import pickle
import pandas as pd
pi = 3.141592653589793

plot_intermediate = False

print('loading data ...')
np.random.seed(99)
N = 52 * 10080  # 10080 = one week, 2049280 total points
electricity_data = pd.read_csv('electricity.csv', sep='  ', header=None, engine='python').values[:N, :]
x = electricity_data[:, 0][:, None]
y = electricity_data[:, 1][:, None]
print('N =', N)

ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

if len(sys.argv) > 1:
    plot_final = False
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    plot_final = True
    method = 0
    fold = 0

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])

x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]

var_y = .1
var_f = 1.  # GP variance
len_f = 1.  # GP lengthscale
period = 1.  # period of quasi-periodic component
len_p = 5.  # lengthscale of quasi-periodic component
var_f_mat = 1.
len_f_mat = 1.

prior1 = priors.Matern32(variance=var_f_mat, lengthscale=len_f_mat)
prior2 = priors.QuasiPeriodicMatern12(variance=var_f, lengthscale_periodic=len_p,
                                      period=period, lengthscale_matern=len_f)
prior = priors.Sum([prior1, prior2])

lik = likelihoods.Gaussian(variance=var_y)

if method == 0:
    inf_method = approx_inf.EKS(damping=.1)
elif method == 1:
    inf_method = approx_inf.UKS(damping=.1)
elif method == 2:
    inf_method = approx_inf.GHKS(damping=.1)
elif method == 3:
    inf_method = approx_inf.EP(power=1, intmethod='GH', damping=.1)
elif method == 4:
    inf_method = approx_inf.EP(power=0.5, intmethod='GH', damping=.1)
elif method == 5:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH', damping=.1)
elif method == 6:
    inf_method = approx_inf.VI(intmethod='GH', damping=.1)

model = SDEGP(prior=prior, likelihood=lik, t=x_train, y=y_train, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()

    print('iter %2d: nlml=%2.2f' %
          (i, neg_log_marg_lik))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
num_iters = 250
for j in range(num_iters):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

x_plot = np.linspace(np.min(x), np.max(x), N)
# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(t=x_test, y=y_test)
posterior_mean, posterior_cov = model.predict(t=x_plot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('test NLPD: %1.2f' % nlpd)

with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpd, fp)

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#     nlpd_show = pickle.load(fp)
# print(nlpd_show)

if plot_final:
    lb = posterior_mean - 1.96 * posterior_cov**0.5
    ub = posterior_mean + 1.96 * posterior_cov**0.5

    print('plotting ...')
    plt.figure(1, figsize=(12, 5))
    plt.clf()
    plt.plot(x, y, 'b.', label='training observations', markersize=4)
    plt.plot(x_test, y_test, 'r.', alpha=0.5, label='test observations', markersize=4)
    plt.plot(x_plot, posterior_mean, 'g', label='posterior mean')
    plt.fill_between(x_plot, lb, ub, color='g', alpha=0.05, label='95% confidence')
    plt.xlim(x_plot[0], x_plot[-1])
    plt.legend()
    plt.title('GP regression via Kalman smoothing. Test NLPD: %1.2f' % nlpd)
    plt.xlabel('time, $t$')
    plt.show()
