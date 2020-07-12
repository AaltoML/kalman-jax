import sys
sys.path.insert(0, '../../')
import numpy as np
import time
import pandas as pd
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from datetime import date
import pickle

plot_final = False
plot_intermediate = False

print('loading data ...')
aircraft_accidents = pd.read_csv('../aircraft/aircraft_accidents.txt', sep='-', header=None).values

num_data = aircraft_accidents.shape[0]
xx = np.zeros([num_data, 1])
for j in range(num_data):
    xx[j] = date.toordinal(date(aircraft_accidents[j, 0], aircraft_accidents[j, 1], aircraft_accidents[j, 2])) + 366

BIN_WIDTH = 1
# Discretize the data
x_min = np.floor(np.min(xx))
x_max = np.ceil(np.max(xx))
x_max_int = x_max-np.mod(x_max-x_min, BIN_WIDTH)
x = np.linspace(x_min, x_max_int, num=int((x_max_int-x_min)/BIN_WIDTH+1))
x = np.concatenate([np.min(x)-np.linspace(61, 1, num=61), x])  # pad with zeros to reduce strange edge effects
y, _ = np.histogram(xx, np.concatenate([[-1e10], x[1:]-np.diff(x)/2, [1e10]]))
N = y.shape[0]

np.random.seed(123)
# meanval = np.log(len(disaster_timings)/num_time_bins)  # TODO: incorporate mean

if len(sys.argv) > 1:
    method = int(sys.argv[1])
else:
    method = 0

print('method number', method)

x_train = x
x_test = x
y_train = y
y_test = y

prior_1 = priors.Matern52(variance=2., lengthscale=5.5e4)
prior_2 = priors.QuasiPeriodicMatern32(variance=1., lengthscale_periodic=2., period=365., lengthscale_matern=1.5e4)
prior_3 = priors.QuasiPeriodicMatern32(variance=1., lengthscale_periodic=2., period=7., lengthscale_matern=30*365.)

prior = priors.Sum([prior_1, prior_2, prior_3])
lik = likelihoods.Poisson()

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

with open("output/aircraft_" + str(method) + ".txt", "wb") as fp:
    pickle.dump(time_taken, fp)

# with open("output/aircraft_" + str(method) + ".txt", "rb") as fp:
#     time_taken = pickle.load(fp)
# print(time_taken)
