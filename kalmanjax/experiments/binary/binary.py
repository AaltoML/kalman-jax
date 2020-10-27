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
pi = 3.141592653589793

plot_intermediate = False

print('generating some data ...')
np.random.seed(99)
N = 10000  # number of training points
x = np.sort(70 * np.random.rand(N))
sn = 0.01
f = lambda x_: 12. * np.sin(4 * pi * x_) / (0.25 * pi * x_ + 1)
y_ = f(x) + np.math.sqrt(sn)*np.random.randn(x.shape[0])
y = np.sign(y_)
y[y == -1] = 0

ind_shuffled = np.random.permutation(N)
ind_split = np.stack(np.split(ind_shuffled, 10))  # 10 random batches of data indices

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
ind_test = ind_split[fold]  # np.sort(ind_shuffled[:N//10])
ind_train = np.concatenate(ind_split[np.arange(10) != fold])

x *= 100

x_train = x[ind_train]  # 90/10 train/test split
x_test = x[ind_test]
y_train = y[ind_train]
y_test = y[ind_test]

var_f = 1.  # GP variance
len_f = 25.  # GP lengthscale

prior = priors.Matern72(variance=var_f, lengthscale=len_f)

lik = likelihoods.Bernoulli(link='logit')

damping = .5

if method == 0:
    inf_method = approx_inf.EEP(power=1, damping=damping)
elif method == 1:
    inf_method = approx_inf.EEP(power=0.5, damping=damping)
elif method == 2:
    inf_method = approx_inf.EKS(damping=damping)

elif method == 3:
    inf_method = approx_inf.UEP(power=1, damping=damping)
elif method == 4:
    inf_method = approx_inf.UEP(power=0.5, damping=damping)
elif method == 5:
    inf_method = approx_inf.UKS(damping=damping)

elif method == 6:
    inf_method = approx_inf.GHEP(power=1, damping=damping)
elif method == 7:
    inf_method = approx_inf.GHEP(power=0.5, damping=damping)
elif method == 8:
    inf_method = approx_inf.GHKS(damping=damping)

elif method == 9:
    inf_method = approx_inf.EP(power=1, intmethod='UT', damping=damping)
elif method == 10:
    inf_method = approx_inf.EP(power=0.5, intmethod='UT', damping=damping)
elif method == 11:
    inf_method = approx_inf.EP(power=0.01, intmethod='UT', damping=damping)

elif method == 12:
    inf_method = approx_inf.EP(power=1, intmethod='GH', damping=damping)
elif method == 13:
    inf_method = approx_inf.EP(power=0.5, intmethod='GH', damping=damping)
elif method == 14:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH', damping=damping)

elif method == 15:
    inf_method = approx_inf.VI(intmethod='UT', damping=damping)
elif method == 16:
    inf_method = approx_inf.VI(intmethod='GH', damping=damping)

model = SDEGP(prior=prior, likelihood=lik, t=x_train, y=y_train, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-1)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    if ~np.any(np.isnan(params[0])):
        mod.prior.hyp = params[0]
        mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()
    # neg_log_marg_lik, gradients = mod.run_two_stage()

    prior_params = softplus_list(params[0])
    print('iter %2d: var_f=%1.2f len_f=%1.2f, nlml=%2.2f' %
          (i, prior_params[0], prior_params[1], neg_log_marg_lik))

    if plot_intermediate:
        plot(mod, i)

    return opt_update(i, gradients, state)


print('optimising the hyperparameters ...')
t0 = time.time()
num_iters = 500
for j in range(num_iters):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
nlpd = model.negative_log_predictive_density(t=x_test, y=y_test)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('test NLPD: %1.2f' % nlpd)

if save_result:
    with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "wb") as fp:
        pickle.dump(nlpd, fp)

# with open("output/" + str(method) + "_" + str(fold) + "_nlpd.txt", "rb") as fp:
#     nlpd_show = pickle.load(fp)
# print(nlpd_show)

# lb = posterior_mean[:, 0] - 1.96 * posterior_var[:, 0]**0.5
# ub = posterior_mean[:, 0] + 1.96 * posterior_var[:, 0]**0.5
# x_pred = model.t_all[:, 0]
# test_id = model.test_id
# link_fn = model.likelihood.link_fn
#
# print('plotting ...')
# plt.figure(1, figsize=(12, 5))
# plt.clf()
# plt.plot(x, y, 'b+', label='training observations')
# plt.plot(x_test, y_test, 'r+', alpha=0.4, label='test observations')
# plt.plot(x_pred, link_fn(posterior_mean), 'm', label='posterior mean')
# plt.fill_between(x_pred, link_fn(lb), link_fn(ub), color='m', alpha=0.05, label='95% confidence')
# plt.xlim(model.t_test[0], model.t_test[-1])
# plt.legend()
# plt.title('GP classification via Kalman smoothing. Test NLPD: %1.2f' % nlpd)
# plt.xlabel('time - $t$')
# plt.show()
