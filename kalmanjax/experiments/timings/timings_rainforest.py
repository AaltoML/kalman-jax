import sys
sys.path.insert(0, '../../')
import numpy as np
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from utils import discretegrid
import pickle
pi = 3.141592653589793

plot_intermediate = False

print('loading rainforest data ...')
data = np.loadtxt('../rainforest/TRI2TU-data.csv', delimiter=',')

nr = 250  # spatial grid point (y-aixs)
nt = 500  # temporal grid points (x-axis)
scale = 1000 / nt

t, r, Y = discretegrid(data, [0, 1000, 0, 500], [nt, nr])

np.random.seed(99)
N = nr * nt  # number of data points

var_f = 1  # GP variance
len_f = 10  # lengthscale

if len(sys.argv) > 1:
    method = int(sys.argv[1])
else:
    method = 0

if method == 0:
    inf_method = approx_inf.EEP(power=1)
elif method == 1:
    inf_method = approx_inf.EKS()

prior = priors.SpatialMatern32(variance=var_f, lengthscale=len_f, z=r[0, ...], fixed_grid=True)
lik = likelihoods.Poisson()
inf_method = approx_inf.ExtendedKalmanSmoother(damping=1.)
# inf_method = approx_inf.ExtendedEP()

# t_spacetime = np.block([t[..., 0][..., None], r])

model = SDEGP(prior=prior, likelihood=lik, t=t, y=Y, r=r, t_test=t, y_test=Y, r_test=r, approx_inf=inf_method)

neg_log_marg_lik, gradients = model.run_two_stage()
print(gradients)
neg_log_marg_lik, gradients = model.run_two_stage()
print(gradients)
neg_log_marg_lik, gradients = model.run_two_stage()
print(gradients)

print('optimising the hyperparameters ...')
time_taken = np.zeros([10, 1])
for j in range(10):
    t0 = time.time()
    neg_log_marg_lik, gradients = model.run_two_stage()
    print(gradients)
    t1 = time.time()
    time_taken[j] = t1-t0
    print('optimisation time: %2.2f secs' % (t1-t0))

time_taken = np.mean(time_taken)

with open("output/rainforest_" + str(method) + ".txt", "wb") as fp:
    pickle.dump(time_taken, fp)

# with open("output/rainforest_" + str(method) + ".txt", "rb") as fp:
#     time_taken = pickle.load(fp)
# print(time_taken)
