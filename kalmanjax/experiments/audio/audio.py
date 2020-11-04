import sys
sys.path.insert(0, '../../')
import numpy as np
from jax.experimental import optimizers
from jax import vmap
import jax.numpy as jnp
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
fs = 44100  # sampling rate (Hz)
scale = 1000  # convert to milliseconds

normaliser = 0.5 * np.sqrt(np.var(y))
yTrain = y / normaliser  # rescale the data

N = y.shape[0]
x = np.linspace(0., N, num=N) / fs * scale  # arbitrary evenly spaced inputs inputs

np.random.seed(123)
# 10-fold cross-validation setup
ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
    plot_final = False
    save_result = True
    num_iters = 250
else:
    method = 9
    fold = 0
    save_result = False
    num_iters = 50

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])
x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]

fundamental_freq = 220  # Hz
radial_freq = 2 * pi * fundamental_freq / scale  # radial freq = 2pi * f / scale
sub1 = priors.SubbandExponentialFixedVar(variance=.1, lengthscale=75., radial_frequency=radial_freq)
sub2 = priors.SubbandExponentialFixedVar(variance=.1, lengthscale=75., radial_frequency=2 * radial_freq)  # 1st harmonic
sub3 = priors.SubbandExponentialFixedVar(variance=.1, lengthscale=75., radial_frequency=3 * radial_freq)  # 2nd harmonic
mod1 = priors.Matern52FixedVar(variance=.5, lengthscale=10.)
mod2 = priors.Matern52FixedVar(variance=.5, lengthscale=10.)
mod3 = priors.Matern52FixedVar(variance=.5, lengthscale=10.)

prior = priors.Independent([sub1, sub2, sub3, mod1, mod2, mod3])

lik = likelihoods.AudioAmplitudeDemodulation(variance=0.3)

if method == 0:
    inf_method = approx_inf.EEP(power=1, damping=0.05)
elif method == 1:
    inf_method = approx_inf.EEP(power=0.5, damping=0.05)
elif method == 2:
    inf_method = approx_inf.EKS(damping=0.05)

elif method == 3:
    inf_method = approx_inf.UEP(power=1, damping=0.05)
elif method == 4:
    inf_method = approx_inf.UEP(power=0.5, damping=0.05)
elif method == 5:
    inf_method = approx_inf.UKS(damping=0.05)

elif method == 6:
    inf_method = approx_inf.GHEP(power=1, damping=0.05)
elif method == 7:
    inf_method = approx_inf.GHEP(power=0.5, damping=0.05)
elif method == 8:
    inf_method = approx_inf.GHKS(damping=0.05)

elif method == 9:
    inf_method = approx_inf.EP(power=1, intmethod='UT', damping=0.05)
elif method == 10:
    inf_method = approx_inf.EP(power=0.5, intmethod='UT', damping=0.05)
elif method == 11:
    inf_method = approx_inf.EP(power=0.01, intmethod='UT', damping=0.05)

elif method == 12:
    inf_method = approx_inf.EP(power=1, intmethod='GH', damping=0.05)
elif method == 13:
    inf_method = approx_inf.EP(power=0.5, intmethod='GH', damping=0.05)
elif method == 14:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH', damping=0.05)

elif method == 15:
    inf_method = approx_inf.VI(intmethod='UT', damping=0.05)
elif method == 16:
    inf_method = approx_inf.VI(intmethod='GH', damping=0.05)

model = SDEGP(prior=prior, likelihood=lik, t=x_train, y=y_train, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-2)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    # neg_log_marg_lik, gradients = mod.run()
    neg_log_marg_lik, gradients = mod.run_two_stage()

    prior_params = softplus_list(params[0])
    # print('iter %2d: var1=%1.2f len1=%1.2f om1=%1.2f var2=%1.2f len2=%1.2f om2=%1.2f var3=%1.2f len3=%1.2f om3=%1.2f '
    #       'var4=%1.2f len4=%1.2f var5=%1.2f len5=%1.2f var6=%1.2f len6=%1.2f '
    #       'vary=%1.2f, nlml=%2.2f' %
    #       (i, prior_params[0][0], prior_params[0][1], prior_params[0][2],
    #        prior_params[1][0], prior_params[1][1], prior_params[1][2],
    #        prior_params[2][0], prior_params[2][1], prior_params[2][2],
    #        prior_params[3][0], prior_params[3][1],
    #        prior_params[4][0], prior_params[4][1],
    #        prior_params[5][0], prior_params[5][1],
    #        softplus(params[1]), neg_log_marg_lik))
    # print('iter %2d: len1=%1.2f om1=%1.2f len2=%1.2f om2=%1.2f len3=%1.2f om3=%1.2f '
    #       'var4=%1.2f len4=%1.2f var5=%1.2f len5=%1.2f var6=%1.2f len6=%1.2f '
    #       'vary=%1.2f, nlml=%2.2f' %
    #       (i, prior_params[0][0], prior_params[0][1],
    #        prior_params[1][0], prior_params[1][1],
    #        prior_params[2][0], prior_params[2][1],
    #        prior_params[3][0], prior_params[3][1],
    #        prior_params[4][0], prior_params[4][1],
    #        prior_params[5][0], prior_params[5][1],
    #        softplus(params[1]), neg_log_marg_lik))
    print('iter %2d: len1=%1.2f om1=%1.2f len2=%1.2f om2=%1.2f len3=%1.2f om3=%1.2f '
          'len4=%1.2f len5=%1.2f len6=%1.2f '
          'vary=%1.2f, nlml=%2.2f' %
          (i, prior_params[0][0], prior_params[0][1],
           prior_params[1][0], prior_params[1][1],
           prior_params[2][0], prior_params[2][1],
           prior_params[3],
           prior_params[4],
           prior_params[5],
           softplus(params[1]), neg_log_marg_lik))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(num_iters):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

x_plot = np.linspace(np.min(x), np.max(x), 20000)
# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var = model.predict(t=x_plot)
nlpd = model.negative_log_predictive_density(t=x_test, y=y_test)
t1 = time.time()
print('NLPD: %1.2f' % nlpd)
print('prediction time: %2.2f secs' % (t1-t0))

if save_result:
    with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)

    # with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
    #     nlpd_show = pickle.load(fp)
    # print(nlpd_show)

if plot_final:

    def diag(Q):
        vectorised_diag = vmap(jnp.diag, 0)
        return vectorised_diag(Q)

    posterior_mean_subbands = posterior_mean[:, :3]
    posterior_mean_modulators = softplus(posterior_mean[:, 3:])
    posterior_mean_sig = np.sum(posterior_mean_subbands * posterior_mean_modulators, axis=-1)
    posterior_var_subbands = diag(posterior_var[:, :3, :3])
    posterior_var_modulators = softplus(diag(posterior_var[:, 3:, 3:]))
    lb_subbands = posterior_mean_subbands - np.sqrt(posterior_var_subbands) * 1.96
    ub_subbands = posterior_mean_subbands + np.sqrt(posterior_var_subbands) * 1.96
    lb_modulators = softplus(posterior_mean_modulators - np.sqrt(posterior_var_modulators) * 1.96)
    ub_modulators = softplus(posterior_mean_modulators + np.sqrt(posterior_var_modulators) * 1.96)

    color1 = [0.2667, 0.4471, 0.7098]  # blue
    color2 = [0.1647, 0.6706, 0.3804]  # green
    color3 = [0.8275, 0.2627, 0.3059]  # red
    color4 = [0.5216, 0.4392, 0.7176]  # purple
    color5 = [0.8118, 0.7255, 0.4118]  # yellow
    color6 = [0.2745, 0.7176, 0.8157]  # light blue
    colors = [color1, color2, color3, color4, color5, color6]

    print('plotting ...')
    plt.figure(1, figsize=(12, 5))
    plt.clf()
    plt.plot(x, y, 'k', label='signal', linewidth=0.6)
    plt.plot(x_test, y_test, 'g.', label='test', markersize=4)
    plt.plot(x_plot, posterior_mean_sig, 'r', label='posterior mean', linewidth=0.6)
    # plt.fill_between(x_pred, lb, ub, color='r', alpha=0.05, label='95% confidence')
    plt.xlim(x_plot[0], x_plot[-1])
    plt.legend()
    plt.title('Audio Signal Processing via Kalman smoothing (human speech signal)')
    plt.xlabel('time (milliseconds)')

    plt.figure(2, figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(x_plot, posterior_mean_subbands[:, i], color=colors[i], linewidth=0.6)
        plt.fill_between(x_plot, lb_subbands[:, i], ub_subbands[:, i], color=colors[i], alpha=0.05, label='95% confidence')
    plt.xlim(x_plot[0], x_plot[-1])
    plt.title('subbands')
    plt.subplot(2, 1, 2)
    plt.plot(x_plot, posterior_mean_modulators, linewidth=0.6)
    for i in range(3):
        plt.plot(x_plot, posterior_mean_modulators[:, i], color=colors[i], linewidth=0.6)
        plt.fill_between(x_plot, lb_modulators[:, i], ub_modulators[:, i], color=colors[i], alpha=0.05, label='95% confidence')
    plt.xlim(x_plot[0], x_plot[-1])
    plt.xlabel('time (milliseconds)')
    plt.title('amplitude modulators')
    plt.show()
