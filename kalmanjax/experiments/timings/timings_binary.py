import sys
sys.path.insert(0, '../../')
import numpy as np
from jax.experimental import optimizers
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
sn = 0.25
f = lambda x_: 12. * np.sin(4 * pi * x_) / (0.25 * pi * x_ + 1)
y_ = f(x) + np.math.sqrt(sn)*np.random.randn(x.shape[0])
y = np.sign(y_)
y[y == -1] = 0

if len(sys.argv) > 1:
    method = int(sys.argv[1])
else:
    method = 0

print('method number', method)

x_train = x
x_test = x
y_train = y
y_test = y

var_f = 1.  # GP variance
len_f = 0.25  # GP lengthscale

prior = priors.Matern72(variance=var_f, lengthscale=len_f)

lik = likelihoods.Bernoulli(link='logit')

if method == 0:
    inf_method = approx_inf.EEP(power=1)
elif method == 1:
    inf_method = approx_inf.EKS()

elif method == 2:
    inf_method = approx_inf.UEP(power=0.5)
elif method == 3:
    inf_method = approx_inf.UKS()

elif method == 4:
    inf_method = approx_inf.GHEP(power=0.5)
elif method == 5:
    inf_method = approx_inf.GHKS()

elif method == 6:
    inf_method = approx_inf.EP(power=0.01, intmethod='UT')
elif method == 7:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH')

elif method == 8:
    inf_method = approx_inf.VI(intmethod='UT')
elif method == 9:
    inf_method = approx_inf.VI(intmethod='GH')

model = SDEGP(prior=prior, likelihood=lik, t=x_train, y=y_train, t_test=x_test, y_test=y_test, approx_inf=inf_method)

neg_log_marg_lik, gradients = model.run()
print(gradients)
neg_log_marg_lik, gradients = model.run()
print(gradients)
neg_log_marg_lik, gradients = model.run()
print(gradients)

print('optimising the hyperparameters ...')
time_taken = np.zeros([10, 1])
for j in range(10):
    t0 = time.time()
    neg_log_marg_lik, gradients = model.run()
    print(gradients)
    t1 = time.time()
    time_taken[j] = t1-t0
    print('optimisation time: %2.2f secs' % (t1-t0))

time_taken = np.mean(time_taken)

with open("output/binary_" + str(method) + ".txt", "wb") as fp:
    pickle.dump(time_taken, fp)

# with open("output/binary_" + str(method) + ".txt", "rb") as fp:
#     time_taken = pickle.load(fp)
# print(time_taken)
