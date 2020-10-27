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
inputs = np.loadtxt('banana_X_train', delimiter=',')
Yall = np.loadtxt('banana_Y_train')[:, None]
cvind = np.loadtxt('cvind.csv').astype(int)
# 10-fold cross-validation
cvind = np.reshape(cvind, (10, 40))

# Test points
Xtest, Ytest = np.mgrid[-2.8:2.8:100j, -2.8:2.8:100j]

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
    save_result = True
else:
    method = 12
    fold = 0
    save_result = False

print('method number', method)
print('batch number', fold)

# Get training and test indices
test = cvind[fold, :]
train = np.setdiff1d(cvind, test)

# Set training and test data
X = inputs[train, :1]
R = inputs[train, 1:]
Y = Yall[train, :]
XT = inputs[test, :1]
RT = inputs[test, 1:]
YT = Yall[test, :]

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

# plot_2d_classification(None, 0)

np.random.seed(99)
N = X.shape[0]  # number of training points

var_f = 1.  # GP variance
len_time = 1.  # temporal lengthscale
len_space = 1.  # spacial lengthscale

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
for j in range(250):
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

if save_result:
    with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)
