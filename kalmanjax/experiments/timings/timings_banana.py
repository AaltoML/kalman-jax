import sys
sys.path.insert(0, '../../')
import numpy as np
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
import pickle
pi = 3.141592653589793

plot_intermediate = False

print('loading banana data ...')
inputs = np.loadtxt('../banana/banana_X_train', delimiter=',')
X = inputs[:, :1]
R = inputs[:, 1:]
Y = np.loadtxt('../banana/banana_Y_train')[:, None]

# Test points
Xtest, Rtest = np.mgrid[-2.8:2.8:100j, -2.8:2.8:100j]

if len(sys.argv) > 1:
    method = int(sys.argv[1])
else:
    method = 0

print('method number', method)

if method == 0:
    inf_method = approx_inf.EEP(power=1)
elif method == 1:
    inf_method = approx_inf.EKS()

elif method == 2:
    inf_method = approx_inf.UEP(power=1)
elif method == 3:
    inf_method = approx_inf.UKS()

elif method == 4:
    inf_method = approx_inf.GHEP(power=1)
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

# plot_2d_classification(None, 0)

np.random.seed(99)
N = X.shape[0]  # number of training points

var_f = 1.  # GP variance
len_time = 1.  # temporal lengthscale
len_space = 1.  # spacial lengthscale

prior = priors.SpatioTemporalMatern52(variance=var_f, lengthscale_time=len_time, lengthscale_space=len_space)

lik = likelihoods.Bernoulli(link='logit')

model = SDEGP(prior=prior, likelihood=lik, t=X, y=Y, r=R, t_test=Xtest, r_test=Rtest, approx_inf=inf_method)

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

with open("output/banana_" + str(method) + ".txt", "wb") as fp:
    pickle.dump(time_taken, fp)

with open("output/banana_" + str(method) + ".txt", "rb") as fp:
    time_taken = pickle.load(fp)
print(time_taken)
