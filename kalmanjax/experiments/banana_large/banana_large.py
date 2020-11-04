import sys
sys.path.insert(0, '../../')
import numpy as np
from jax.experimental import optimizers
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, ListedColormap
from scipy.interpolate import interp1d
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list, plot_2d_classification, plot_2d_classification_filtering
import pickle
pi = 3.141592653589793

plot_intermediate = False

print('loading banana data ...')
np.random.seed(99)
# inputs = np.loadtxt('banana_X_train', delimiter=',')
# Yall = np.loadtxt('banana_Y_train')[:, None]
inputs = np.loadtxt('banana_large.csv', delimiter=',', skiprows=1)
scale_inputs = 10
Xall = scale_inputs * inputs[:, :1]  # temporal inputs (x-axis)
Rall = scale_inputs * inputs[:, 1:2]  # spatial inputs (y-axis)
Yall = np.maximum(inputs[:, 2:], 0)  # observations / labels
N = Xall.shape[0]  # number of training points
N_batch = 50
M = 15
Z = scale_inputs * np.linspace(-3., 3., M)  # inducing points

ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

# Test points
Xtest, Rtest = scale_inputs * np.mgrid[-3.2:3.2:100j, -3.2:3.2:100j]

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
else:
    method = 3
    fold = 0

print('method number', method)
print('batch number', fold)

# Get training and test indices
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])

# Set training and test data
X = Xall[ind_train]
R = Rall[ind_train]
Y = Yall[ind_train]
XT = Xall[ind_test]
RT = Rall[ind_test]
YT = Yall[ind_test]

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

var_f = 1.  # GP variance
len_time = 1. * scale_inputs  # temporal lengthscale
len_space = 1. * scale_inputs  # spacial lengthscale

prior = priors.SpatioTemporalMatern52(variance=var_f, lengthscale_time=len_time, lengthscale_space=len_space)

lik = likelihoods.Bernoulli(link='logit')

model = SDEGP(prior=prior, likelihood=lik, t=X, y=Y, r=R, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=2e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod, plot_num_, mu_prev_):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()
    # neg_log_marg_lik, gradients = mod.run_two_stage()

    prior_params = softplus_list(params[0])
    print('iter %2d: var=%1.2f len_time=%1.2f len_space=%1.2f, nlml=%2.2f' %
          (i, prior_params[0], prior_params[1], prior_params[2], neg_log_marg_lik))

    if plot_intermediate:
        plot_2d_classification(mod, i)
        # plot_num_, mu_prev_ = plot_2d_classification_filtering(mod, i, plot_num_, mu_prev_)

    return opt_update(i, gradients, state), plot_num_, mu_prev_


plot_num = 0
mu_prev = None
print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(500):
    opt_state, plot_num, mu_prev = gradient_step(j, opt_state, model, plot_num, mu_prev)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(t=XT, y=YT, r=RT)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('test NLPD: %1.2f' % nlpd)

with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
    pickle.dump(nlpd, fp)
