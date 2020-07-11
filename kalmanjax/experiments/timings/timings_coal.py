import sys
sys.path.insert(0, '../../')
import numpy as np
import time
import pandas as pd
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
import pickle

plot_final = False
plot_intermediate = False

print('loading coal data ...')
if plot_final:
    disaster_timings = pd.read_csv('../../../data/coal.txt', header=None).values[:, 0]

D = np.loadtxt('../coal/binned.csv')
x = D[:, 0:1]
y = D[:, 1:]
N = D.shape[0]

np.random.seed(123)
# meanval = np.log(len(disaster_timings)/num_time_bins)  # TODO: incorporate mean

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = 0
else:
    method = 0
    fold = 0

print('method number', method)
print('batch number', fold)

x_train = x
x_test = x
y_train = y
y_test = y

var_f = 1.0  # GP variance
len_f = 1.0  # GP lengthscale

prior = priors.Matern52(variance=var_f, lengthscale=len_f)
lik = likelihoods.Poisson()

if method == 0:
    inf_method = approx_inf.EEP(power=1, damping=0.5)
elif method == 1:
    inf_method = approx_inf.EKS(damping=0.5)

elif method == 2:
    inf_method = approx_inf.UEP(power=1, damping=0.5)
elif method == 3:
    inf_method = approx_inf.UKS(damping=0.5)

elif method == 4:
    inf_method = approx_inf.GHEP(power=1, damping=0.5)
elif method == 5:
    inf_method = approx_inf.GHKS(damping=0.5)

elif method == 6:
    inf_method = approx_inf.EP(power=0.01, intmethod='UT', damping=0.5)
elif method == 7:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH', damping=0.5)

elif method == 8:
    inf_method = approx_inf.VI(intmethod='UT', damping=0.5)
elif method == 9:
    inf_method = approx_inf.VI(intmethod='GH', damping=0.5)

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

with open("output/coal_" + str(method) + ".txt", "wb") as fp:
    pickle.dump(time_taken, fp)

with open("output/coal_" + str(method) + ".txt", "rb") as fp:
    time_taken = pickle.load(fp)
print(time_taken)
