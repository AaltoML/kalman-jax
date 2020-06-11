import numpy as np
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import softplus_list
pi = 3.141592653589793

plot_intermediate = True

print('loading banana data ...')
X = np.loadtxt('../data/banana_X_train', delimiter=',')
Y = np.loadtxt('../data/banana_Y_train')[:, None]

# Test points
Xtest, Ytest = np.mgrid[-3.:3.:100j, -3.:3.:100j]
Xtest = np.vstack((Xtest.flatten(), Ytest.flatten())).T


# Set up plotting
def plot(m, it_num):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # xtest, ytest = np.mgrid[-2.8:2.8:100j, -2.8:2.8:100j]
    # Xtest = np.vstack((xtest.flatten(), ytest.flatten())).T
    for i, mark in [[1, 'o'], [0, 'o']]:
        ind = Y[:, 0] == i
        # ax.plot(X[ind, 0], X[ind, 1], mark)
        ax.scatter(X[ind, 0], X[ind, 1], s=100, alpha=.5)
    # mu, var = m.predict_y(Xtest)
    # ax.contour(xtest, ytest, mu.numpy().reshape(100, 100), levels=[.5],
    #            colors='k', linewidths=4.)
    ax.axis('equal')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    # ax.axis('off')
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    plt.savefig('output/test_%d.png' % it_num)
    plt.close()


plot(None, 0)

np.random.seed(99)
N = X.shape[0]  # number of training points

var_f = 1.0  # GP variance
len_f = 1.0  # GP lengthscale

theta_prior = [var_f, len_f]

prior = priors.Matern52(theta_prior)
lik = likelihoods.Probit()
inf_method = approx_inf.EP(power=0.5)
# inf_method = approx_inf.PL()
# inf_method = approx_inf.EKS()
# inf_method = approx_inf.EKEP()  # <-- not working
# inf_method = approx_inf.VI()

model = SDEGP(prior=prior, likelihood=lik, x=X, y=Y, x_test=None, y_test=None, approx_inf=inf_method)

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
for j in range(20):
    opt_state = gradient_step(j, opt_state, model)
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))

# calculate posterior predictive distribution via filtering and smoothing at train & test locations:
print('calculating the posterior predictive distribution ...')
t0 = time.time()
posterior_mean, posterior_var, _, nlpd = model.predict()
t1 = time.time()
print('prediction time: %2.2f secs' % (t1-t0))
print('test NLPD: %1.2f' % nlpd)

lb = posterior_mean[:, 0] - 1.96 * posterior_var[:, 0]**0.5
ub = posterior_mean[:, 0] + 1.96 * posterior_var[:, 0]**0.5
x_pred = model.t_all
test_id = model.test_id
link_fn = model.likelihood.link_fn

print('sampling from the posterior ...')
t0 = time.time()
posterior_samp = model.posterior_sample(20)
t1 = time.time()
print('sampling time: %2.2f secs' % (t1-t0))

print('plotting ...')
plt.figure(1, figsize=(12, 5))
plt.clf()
plt.plot(x, y, 'b+', label='training observations')
plt.plot(x_test, y_test, 'r+', alpha=0.4, label='test observations')
plt.plot(x_pred, link_fn(posterior_mean), 'm', label='posterior mean')
plt.fill_between(x_pred, link_fn(lb), link_fn(ub), color='m', alpha=0.05, label='95% confidence')
plt.plot(model.t_test, link_fn(posterior_samp[test_id, 0, :]), 'm', alpha=0.15)
plt.xlim(model.t_test[0], model.t_test[-1])
plt.legend()
plt.title('GP classification via Kalman smoothing. Test NLPD: %1.2f' % nlpd)
plt.xlabel('time - $t$')
plt.show()
