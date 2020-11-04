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
from utils import softplus_list, plot
import pickle
from sklearn.preprocessing import StandardScaler

plot_intermediate = False

print('loading data ...')
D = np.loadtxt('mcycle.csv', delimiter=',')
X = D[:, 1:2]
Y = D[:, 2:]
N = X.shape[0]

# Standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(Y)
Xall = X_scaler.transform(X)
Yall = y_scaler.transform(Y)

# Load cross-validation indices
cvind = np.loadtxt('cvind.csv').astype(int)

# 10-fold cross-validation setup
nt = np.floor(cvind.shape[0]/10).astype(int)
cvind = np.reshape(cvind[:10*nt], (10, nt))

np.random.seed(123)

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = int(sys.argv[2])
    plot_final = False
    save_result = True
else:
    method = 15
    fold = 2
    plot_final = True
    save_result = False

print('method number', method)
print('batch number', fold)

# Get training and test indices
test = cvind[fold, :]
train = np.setdiff1d(cvind, test)

# Set training and test data
X = Xall[train, :]
Y = Yall[train, :]
XT = Xall[test, :]
YT = Yall[test, :]

var_f1 = 3.  # GP variance
len_f1 = 1.  # GP lengthscale
var_f2 = 3.  # GP variance
len_f2 = 1.  # GP lengthscale

prior1 = priors.Matern32(variance=var_f1, lengthscale=len_f1)
prior2 = priors.Matern32(variance=var_f2, lengthscale=len_f2)
prior = priors.Independent([prior1, prior2])
lik = likelihoods.HeteroscedasticNoise()

step_size = 5e-2

if method == 0:
    inf_method = approx_inf.EEP(power=1, damping=0.5)
elif method == 1:
    inf_method = approx_inf.EEP(power=0.5, damping=0.5)
elif method == 2:
    inf_method = approx_inf.EKS(damping=0.5)

elif method == 3:
    inf_method = approx_inf.UEP(power=1, damping=0.5)
elif method == 4:
    inf_method = approx_inf.UEP(power=0.5, damping=0.5)
elif method == 5:
    inf_method = approx_inf.UKS(damping=0.5)

elif method == 6:
    inf_method = approx_inf.GHEP(power=1, damping=0.5)
elif method == 7:
    inf_method = approx_inf.GHEP(power=0.5, damping=0.5)
elif method == 8:
    inf_method = approx_inf.GHKS(damping=0.5)

elif method == 9:
    step_size = 1e-2
    inf_method = approx_inf.EP(power=1, intmethod='UT', damping=0.05)  # this method requies more damping
elif method == 10:
    if fold in [1]:
        inf_method = approx_inf.EP(power=0.5, intmethod='UT', damping=0.5)
    else:
        inf_method = approx_inf.EP(power=0.5, intmethod='UT', damping=0.5)
elif method == 11:
    inf_method = approx_inf.EP(power=0.01, intmethod='UT', damping=0.5)

elif method == 12:
    if fold in [3, 4, 6, 7, 9]:
        inf_method = approx_inf.EP(power=1, intmethod='GH', damping=0.1)
    else:
        inf_method = approx_inf.EP(power=1, intmethod='GH', damping=0.5)
elif method == 13:
    inf_method = approx_inf.EP(power=0.5, intmethod='GH', damping=0.5)
elif method == 14:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH', damping=0.5)

elif method == 15:
    inf_method = approx_inf.VI(intmethod='UT', damping=0.5)
elif method == 16:
    inf_method = approx_inf.VI(intmethod='GH', damping=0.5)

model = SDEGP(prior=prior, likelihood=lik, t=X, y=Y, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()
    # neg_log_marg_lik, gradients = mod.run_two_stage()

    prior_params = softplus_list(params[0])
    print('iter %2d: var_f1=%1.2f len_f1=%1.2f var_f2=%1.2f len_f2=%1.2f, nlml=%2.2f' %
          (i, prior_params[0][0], prior_params[0][1], prior_params[1][0], prior_params[1][1], neg_log_marg_lik))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
for j in range(250):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

x_plot = np.linspace(np.min(Xall)-0.2, np.max(Xall)+0.2, 200)
# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(t=XT, y=YT)
posterior_mean, posterior_cov = model.predict(t=x_plot)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('NLPD: %1.2f' % nlpd)

if save_result:
    with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#     nlpd_show = pickle.load(fp)
# print(nlpd_show)

if plot_final:
    link = model.likelihood.link_fn
    lb = posterior_mean[:, 0] - np.sqrt(posterior_cov[:, 0, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96
    ub = posterior_mean[:, 0] + np.sqrt(posterior_cov[:, 0, 0] + link(posterior_mean[:, 1]) ** 2) * 1.96

    print('plotting ...')
    plt.figure(1, figsize=(12, 5))
    plt.clf()
    plt.plot(X, Y, 'k.', label='train')
    plt.plot(XT, YT, 'r.', label='test')
    plt.plot(x_plot, posterior_mean[:, 0], 'c', label='posterior mean')
    plt.fill_between(x_plot, lb, ub, color='c', alpha=0.05, label='95% confidence')
    plt.xlim(x_plot[0], x_plot[-1])
    plt.legend()
    plt.title('Heteroscedastic Noise Model via Kalman smoothing (motorcycle crash data)')
    plt.xlabel('time')
    plt.ylabel('acceleration')
    plt.show()
