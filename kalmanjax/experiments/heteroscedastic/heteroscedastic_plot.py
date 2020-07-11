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
from sklearn.preprocessing import StandardScaler
import tikzplotlib

plot_intermediate = False
save_tikz = False

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
xshift = np.min(X) - np.min(Xall)
xrange = (np.max(X) - np.min(X)) / (np.max(Xall) - np.min(Xall))

# Set test data
XT = np.linspace(np.min(Xall), np.max(Xall), num=200)

var_f1 = 3.  # GP variance
len_f1 = 1.  # GP lengthscale
var_f2 = 3.  # GP variance
len_f2 = 1.  # GP lengthscale

prior1 = priors.Matern32(variance=var_f1, lengthscale=len_f1)
prior2 = priors.Matern32(variance=var_f2, lengthscale=len_f2)
prior = priors.Independent([prior1, prior2])
lik = likelihoods.HeteroscedasticNoise()

# inf_method = approx_inf.ExpectationPropagation(power=0.9, intmethod='UT', damping=0.1)
inf_method = approx_inf.ExpectationPropagation(power=0.1, intmethod='GH', damping=0.5)
# inf_method = approx_inf.VariationalInference(intmethod='GH', damping=0.5)
# inf_method = approx_inf.VariationalInference(intmethod='UT', damping=0.5)
# inf_method = approx_inf.ExtendedEP(power=0, damping=0.5)
# inf_method = approx_inf.ExtendedKalmanSmoother(damping=0.5)
# inf_method = approx_inf.GaussHermiteKalmanSmoother(damping=0.5)
# inf_method = approx_inf.StatisticallyLinearisedEP(intmethod='UT', damping=0.5)
# inf_method = approx_inf.UnscentedKalmanSmoother(damping=0.5)

model = SDEGP(prior=prior, likelihood=lik, t=Xall, y=Yall, t_test=XT, approx_inf=inf_method)

opt_init, opt_update, get_params = optimizers.adam(step_size=5e-2)
# parameters should be a 2-element list [param_prior, param_likelihood]
opt_state = opt_init([model.prior.hyp, model.likelihood.hyp])


def gradient_step(i, state, mod):
    params = get_params(state)
    mod.prior.hyp = params[0]
    mod.likelihood.hyp = params[1]

    # grad(Filter) + Smoother:
    neg_log_marg_lik, gradients = mod.run()
    # neg_log_marg_lik, gradients = mod.run_two_stage()  # <-- less elegant but reduces compile time

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

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var, _, nlpd = model.predict(compute_nlpd=False)
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))


# x_pred = xrange * (model.t_all[:, 0] + xshift)
x_pred = X_scaler.inverse_transform(model.t_all[:, 0])
# X_rescale = xrange * (X + xshift)
link = model.likelihood.link_fn
lb = posterior_mean[:, 0, 0] - np.sqrt(posterior_var[:, 0, 0] + link(posterior_mean[:, 1, 0]) ** 2) * 1.96
ub = posterior_mean[:, 0, 0] + np.sqrt(posterior_var[:, 0, 0] + link(posterior_mean[:, 1, 0]) ** 2) * 1.96
post_mean = y_scaler.inverse_transform(posterior_mean[:, 0, 0])
lb = y_scaler.inverse_transform(lb)
ub = y_scaler.inverse_transform(ub)

print('plotting ...')
fig, ax = plt.subplots()
plt.plot(X, Y, 'k.', markersize=0.8)
plt.plot(x_pred, post_mean, 'k', linewidth=0.8)
plt.fill_between(x_pred, lb, ub, color='k', alpha=0.1, edgecolor=(0, 0, 0, 0.12))
plt.xlim(x_pred[0], x_pred[-1])
plt.xlabel('Time (milliseconds)')
plt.ylabel('Accelerometer reading')
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
ax.tick_params(top=True, right=True)
if save_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/linearised-inference/paper/fig/mcycle.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')


lb_1 = posterior_mean[:, 0, 0] - np.sqrt(posterior_var[:, 0, 0]) * 1.96
ub_1 = posterior_mean[:, 0, 0] + np.sqrt(posterior_var[:, 0, 0]) * 1.96
lb_1 = y_scaler.inverse_transform(lb_1)
ub_1 = y_scaler.inverse_transform(ub_1)

fig2, ax2 = plt.subplots()
plt.plot(x_pred, post_mean, 'k', linewidth=0.8, label='posterior mean')
plt.fill_between(x_pred, lb_1, ub_1, color='k', alpha=0.1, edgecolor=(0, 0, 0, 0.12), label='95\% confidence')
plt.plot(X, Y, 'k.', markersize=0.8, label='observations')
plt.xlim(x_pred[0], x_pred[-1])
plt.title('EP: $f^{(1)}$ (mean component)')
# plt.xlabel('Time (milliseconds)')
# plt.ylabel('Accelerometer reading')
# plt.legend(['observations', 'posterior mean', '95% confidence'])
plt.legend(loc=4)
ax2.tick_params(axis="y", direction="in")
ax2.tick_params(axis="x", direction="in")
ax2.tick_params(top=True, right=True)
if save_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/linearised-inference/paper/fig/mcycle_f1.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

mean_f2 = link(posterior_mean[:, 1, 0])
mean_f2 = y_scaler.scale_ * mean_f2
lb_2 = posterior_mean[:, 1, 0] - np.sqrt(posterior_var[:, 1, 1]) * 1.96
ub_2 = posterior_mean[:, 1, 0] + np.sqrt(posterior_var[:, 1, 1]) * 1.96
lb_2 = y_scaler.scale_ * link(lb_2)
ub_2 = y_scaler.scale_ * link(ub_2)

fig3, ax3 = plt.subplots()
# plt.plot(X, Y, 'k.', markersize=0.8)
plt.plot(x_pred, mean_f2, 'k', linewidth=0.8)
plt.fill_between(x_pred, lb_2, ub_2, color='k', alpha=0.1, edgecolor=(0, 0, 0, 0.12))
plt.xlim(x_pred[0], x_pred[-1])
plt.title('EP: $\phi(f^{(2)})$ (noise standard deviation component)')
# plt.xlabel('Time (milliseconds)')
# plt.ylabel('Accelerometer reading')
ax3.tick_params(axis="y", direction="in")
ax3.tick_params(axis="x", direction="in")
ax3.tick_params(top=True, right=True)
if save_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/linearised-inference/paper/fig/mcycle_f2.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

fig4, ax4 = plt.subplots()
plt.plot(X, Y, 'k.', markersize=0.8)
plt.plot(x_pred, post_mean, 'k', linewidth=0.8)
plt.fill_between(x_pred, lb, ub, color='k', alpha=0.1, edgecolor=(0, 0, 0, 0.12))
plt.fill_between(x_pred, lb_1, ub_1, color='white', alpha=0.85, edgecolor=(0, 0, 0, 0.12))
plt.xlim(x_pred[0], x_pred[-1])
plt.title('EP: full posterior')
plt.xlabel('Time (milliseconds)')
# plt.ylabel('Accelerometer reading')
ax4.tick_params(axis="y", direction="in")
ax4.tick_params(axis="x", direction="in")
ax4.tick_params(top=True, right=True)
if save_tikz:
    tikzplotlib.save('/Users/wilkinw1/postdoc/inprogress/linearised-inference/paper/fig/mcycle_combined.tex',
                     axis_width='\\figurewidth',
                     axis_height='\\figureheight',
                     tex_relative_path_to_data='./fig/')

plt.show()
