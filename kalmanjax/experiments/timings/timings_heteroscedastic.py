import sys
sys.path.insert(0, '../../')
import numpy as np
import time
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
import pickle
from sklearn.preprocessing import StandardScaler

plot_intermediate = False

print('loading data ...')
D = np.loadtxt('../heteroscedastic/mcycle.csv', delimiter=',')
X = D[:, 1:2]
Y = D[:, 2:]
N = X.shape[0]

# Standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(Y)
Xall = X_scaler.transform(X)
Yall = y_scaler.transform(Y)

np.random.seed(123)

if len(sys.argv) > 1:
    method = int(sys.argv[1])
    fold = 0
else:
    method = 0
    fold = 0

print('method number', method)
print('batch number', fold)

# Set training and test data
X = Xall
Y = Yall
XT = Xall
YT = Yall

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

model = SDEGP(prior=prior, likelihood=lik, t=X, y=Y, t_test=XT, y_test=YT, approx_inf=inf_method)

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

with open("output/heteroscedastic_" + str(method) + ".txt", "wb") as fp:
    pickle.dump(time_taken, fp)

# with open("output/heteroscedastic_" + str(method) + ".txt", "rb") as fp:
#     time_taken = pickle.load(fp)
# print(time_taken)
