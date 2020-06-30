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
from utils import softplus, softplus_list, plot
import pickle
from numpy import pi
from scipy.io import loadmat

plot_final = True
plot_intermediate = False

print('loading data ...')
y = loadmat('speech_female')['y']
fs = 44100  # desired sampling rate

normaliser = 0.5 * np.sqrt(np.var(y))
yTrain = y / normaliser  # rescale the input to unit variance

N = y.shape[0]
x = np.linspace(0., N, num=N) / 100.  # arbitrary evenly spaced inputs inputs

np.random.seed(123)
# 10-fold cross-validation setup
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
    plot_final = False
else:
    method = 11
    fold = 6

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]

sub1 = priors.SubbandExponential([.1, 15., 4 * pi])  # omega = 2pi / freq
sub2 = priors.SubbandExponential([.1, 15., 2 * pi])
sub3 = priors.SubbandExponential([.1, 15., 1 * pi])
mod1 = priors.Matern52([3., 20.])
mod2 = priors.Matern52([3., 20.])
mod3 = priors.Matern52([3., 20.])

# sub1 = priors.SubbandExponential([.1, 20., 10 * pi])  # omega = 2pi / freq
# sub2 = priors.SubbandExponential([.1, 20., 5 * pi])
# sub3 = priors.SubbandExponential([.1, 20., 2.5 * pi])
# mod1 = priors.Matern52([2.5, 40.])
# mod2 = priors.Matern52([2.5, 40.])
# mod3 = priors.Matern52([2.5, 40.])

prior = priors.Independent([sub1, sub2, sub3, mod1, mod2, mod3])

lik = likelihoods.AudioAmplitudeDemodulation(hyp=0.3)

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
    inf_method = approx_inf.EP(power=1, intmethod='UT', damping=0.25)
elif method == 10:
    inf_method = approx_inf.EP(power=0.5, intmethod='UT', damping=0.25)
elif method == 11:
    inf_method = approx_inf.EP(power=0.01, intmethod='UT', damping=0.25)

elif method == 12:
    inf_method = approx_inf.EP(power=1, intmethod='GH', damping=0.25)
elif method == 13:
    inf_method = approx_inf.EP(power=0.5, intmethod='GH', damping=0.25)
elif method == 14:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH', damping=0.25)

elif method == 15:
    inf_method = approx_inf.VI(intmethod='UT', damping=0.25)
elif method == 16:
    inf_method = approx_inf.VI(intmethod='GH', damping=0.1, num_cub_pts=3)

model = SDEGP(prior=prior, likelihood=lik, x=x_train, y=y_train, x_test=x_test, y_test=y_test, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run_two_stage()

    prior_params = softplus_list(params[0])
    print('iter %2d: var1=%1.2f len1=%1.2f om1=%1.2f var2=%1.2f len2=%1.2f om2=%1.2f var3=%1.2f len3=%1.2f om3=%1.2f '
          'vary=%1.2f, nlml=%2.2f' %
          (i, prior_params[0][0], prior_params[0][1], prior_params[0][2],
           prior_params[1][0], prior_params[1][1], prior_params[1][2],
           prior_params[2][0], prior_params[2][1], prior_params[2][2],
           softplus(params[1]), neg_log_marg_lik))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(100):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var, _, nlpd = model.predict()
t1 = time.time()
print('NLPD: %1.2f' % nlpd)
print('prediction time: %2.2f secs' % (t1-t0))

with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpd, fp)

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#     nlpd_show = pickle.load(fp)
# print(nlpd_show)

if plot_final:
    x_pred = model.t_all[:, 0]
    # link = model.likelihood.link_fn
    # lb = posterior_mean[:, 0, 0] - np.sqrt(posterior_var[:, 0, 0]) * 1.96
    # ub = posterior_mean[:, 0, 0] + np.sqrt(posterior_var[:, 0, 0]) * 1.96
    test_id = model.test_id

    posterior_mean_subbands = posterior_mean[:, :3, 0]
    posterior_mean_modulators = softplus(posterior_mean[:, 3:, 0])
    posterior_mean_sig = np.sum(posterior_mean_subbands * posterior_mean_modulators, axis=-1)
    posterior_var_subbands = posterior_var[:, :3, 0]
    posterior_var_modulators = softplus(posterior_var[:, 3:, 0])

    print('plotting ...')
    plt.figure(1, figsize=(12, 5))
    plt.clf()
    plt.plot(x, y, 'k', label='signal', linewidth=0.6)
    plt.plot(x_test, y_test, 'g.', label='test', markersize=4)
    plt.plot(x_pred, posterior_mean_sig, 'r', label='posterior mean', linewidth=0.6)
    # plt.fill_between(x_pred, lb, ub, color='r', alpha=0.05, label='95% confidence')
    plt.xlim(model.t_all[0], model.t_all[-1])
    plt.legend()
    plt.title('Audio Signal Processing via Kalman smoothing (human speech signal)')
    plt.xlabel('time')

    plt.figure(2, figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x_pred, posterior_mean_subbands, linewidth=0.6)
    plt.xlim(model.t_all[0], model.t_all[-1])
    plt.subplot(2, 1, 2)
    plt.plot(x_pred, posterior_mean_modulators, linewidth=0.6)
    plt.xlim(model.t_all[0], model.t_all[-1])
    plt.show()
