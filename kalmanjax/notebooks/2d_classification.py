import sys
sys.path.insert(0, '../')
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

plot_intermediate = False

print('loading banana data ...')
inputs = np.loadtxt('../../data/banana_X_train', delimiter=',')
X = inputs[:, :1]  # temporal inputs (x-axis)
R = inputs[:, 1:]  # spatial inputs (y-axis)
Y = np.loadtxt('../../data/banana_Y_train')[:, None]  # observations / labels

# Test points
Xtest, Rtest = np.mgrid[-2.8:2.8:100j, -2.8:2.8:100j]
# Xtest = np.vstack((Xtest.flatten(), Ytest.flatten())).T
# X0test, X1test = np.linspace(-3., 3., num=100), np.linspace(-3., 3., num=100)

# plot_2d_classification(None, 0)

np.random.seed(99)
N = X.shape[0]  # number of training points

var_f = 0.3  # GP variance
len_time = 0.3  # temporal lengthscale
len_space = 0.3  # spacial lengthscale

prior = priors.SpatioTemporalMatern52(variance=var_f, lengthscale_time=len_time, lengthscale_space=len_space)
lik = likelihoods.Probit()
inf_method = approx_inf.ExpectationPropagation(power=0.5)
# inf_method = approx_inf.StatisticallyLinearisedEP()
# inf_method = approx_inf.ExtendedKalmanSmoother()
# inf_method = approx_inf.VariationalInference()

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
    # neg_log_marg_lik, gradients = mod.run_two_stage()  # <-- less elegant but reduces compile time

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
for j in range(50):
    opt_state, plot_num, mu_prev = gradient_step(j, opt_state, model, plot_num, mu_prev)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
mu, var = model.predict(t=Xtest, r=Rtest)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))

link_fn = model.likelihood.link_fn

print('plotting ...')
plt.figure(1)
for label, mark in [[1, 'o'], [0, 'o']]:
    ind = Y[:, 0] == label
    # ax.plot(X[ind, 0], X[ind, 1], mark)
    plt.scatter(X[ind], R[ind], s=50, alpha=.5)
# ax.imshow(mu.T)
plt.contour(Xtest, Rtest, mu, levels=[.0], colors='k', linewidths=4.)
# plt.axis('equal')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
# ax.axis('off')
lim = 2.8
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
# plt.savefig('output/data.png')

# x1 = np.linspace(-lim, lim, num=100)
# x2 = np.linspace(-lim, lim, num=100)
cmap_ = [[1, 0.498039215686275, 0.0549019607843137], [0.12156862745098, 0.466666666666667, 0.705882352941177]]
cmap = hsv_to_rgb(interp1d([0., 1.], rgb_to_hsv(cmap_), axis=0)(link_fn(np.linspace(-3.5, 3.5, num=64))))
newcmp = ListedColormap(cmap)

plt.figure(2)
im = plt.imshow(link_fn(mu).T, cmap=newcmp, extent=[-lim, lim, -lim, lim], origin='lower')
cb = plt.colorbar(im)
cb.set_ticks([cb.vmin, 0, cb.vmax])
cb.set_ticklabels([-1, 0, 1])
plt.contour(Xtest, Rtest, mu, levels=[.0], colors='k', linewidths=1.5)
# plt.axis('equal')
for label in [1, 0]:
    ind = Y[:, 0] == label
    plt.scatter(X[ind], R[ind], s=50, alpha=.5, edgecolor='k')
# plt.title('Iteration: %02d' % (j + 1), loc='right', fontweight='bold')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
# plt.savefig('output/output_%04d.png' % 1600)
plt.show()
