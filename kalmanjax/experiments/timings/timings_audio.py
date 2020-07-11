import sys
sys.path.insert(0, '../../')
import numpy as np
from sde_gp import SDEGP
import approximate_inference as approx_inf
import priors
import likelihoods
from numpy import pi
from scipy.io import loadmat
import time
import pickle

print('loading data ...')
y = loadmat('../audio/speech_female')['y']
fs = 44100  # sampling rate (Hz)
scale = 1000  # convert to milliseconds

normaliser = 0.5 * np.sqrt(np.var(y))
yTrain = y / normaliser  # rescale the data

N = y.shape[0]
x = np.linspace(0., N, num=N) / fs * scale  # arbitrary evenly spaced inputs inputs

if len(sys.argv) > 1:
    method = int(sys.argv[1])
else:
    method = 0

print('method number', method)

x_train = x
x_test = x
y_train = y
y_test = y

fundamental_freq = 220  # Hz
radial_freq = 2 * pi * fundamental_freq / scale  # radial freq = 2pi * f / scale
sub1 = priors.SubbandExponentialFixedVar(variance=.1, lengthscale=75., radial_frequency=radial_freq)
sub2 = priors.SubbandExponentialFixedVar(variance=.1, lengthscale=75., radial_frequency=2 * radial_freq)  # 1st harmonic
sub3 = priors.SubbandExponentialFixedVar(variance=.1, lengthscale=75., radial_frequency=3 * radial_freq)  # 2nd harmonic
mod1 = priors.Matern52FixedVar(variance=.5, lengthscale=10.)
mod2 = priors.Matern52FixedVar(variance=.5, lengthscale=10.)
mod3 = priors.Matern52FixedVar(variance=.5, lengthscale=10.)

prior = priors.Independent([sub1, sub2, sub3, mod1, mod2, mod3])

lik = likelihoods.AudioAmplitudeDemodulation(variance=0.3)

if method == 0:
    inf_method = approx_inf.EEP(power=1, damping=0.05)
elif method == 1:
    inf_method = approx_inf.EKS(damping=0.05)

elif method == 2:
    inf_method = approx_inf.UEP(power=1, damping=0.05)
elif method == 3:
    inf_method = approx_inf.UKS(damping=0.05)

elif method == 4:
    inf_method = approx_inf.GHEP(power=1, damping=0.05)
elif method == 5:
    inf_method = approx_inf.GHKS(damping=0.05)

elif method == 6:
    inf_method = approx_inf.EP(power=0.01, intmethod='UT', damping=0.05)
elif method == 7:
    inf_method = approx_inf.EP(power=0.01, intmethod='GH', damping=0.05)

elif method == 8:
    inf_method = approx_inf.VI(intmethod='UT', damping=0.05)
elif method == 9:
    inf_method = approx_inf.VI(intmethod='GH', damping=0.05)

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

with open("output/audio_" + str(method) + ".txt", "wb") as fp:
    pickle.dump(time_taken, fp)

# with open("output/audio_" + str(method) + ".txt", "rb") as fp:
#     time_taken = pickle.load(fp)
# print(time_taken)
