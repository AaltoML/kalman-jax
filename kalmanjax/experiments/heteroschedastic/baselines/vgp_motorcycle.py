from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.config import default_float
from gpflow.utilities import print_summary
from gpflow.utilities import set_trainable

from gpflow.likelihoods import Likelihood

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Pretty much based on this example given on Stack Overflow (with the change of having f and g independent):
# https://stackoverflow.com/questions/57597349/likelihoods-combining-multiple-latent-gps-in-gpflow


class MultiLatentLikelihood(Likelihood):
    def __init__(self, num_latent=1, **kwargs):
        super().__init__(**kwargs)
        self.num_latent = num_latent

    def _transform(self, F):
        return [F[:, i] for i in range(self.num_latent)]

    def predict_mean_and_var(self, Fmu, Fvar):
        return super().predict_mean_and_var(self._transform(Fmu), self._transform(Fvar))

    def predict_density(self, Fmu, Fvar, Y):
        return super().predict_density(self._transform(Fmu), self._transform(Fvar), Y)

    def variational_expectations(self, Fmu, Fvar, Y):
        return super().variational_expectations(self._transform(Fmu), self._transform(Fvar), Y)


class HeteroscedasticGaussian(MultiLatentLikelihood):
    r"""
    When using this class, num_latent must be 2.
    It does not support multi-output (num_output will be 1)
    """
    def __init__(self, transform=tf.nn.softplus, **kwargs):
        super().__init__(num_latent=2, **kwargs)
        self.transform = transform

    def Y_given_F(self, F, G):
        mu = tf.squeeze(F)
        sigma = self.transform(tf.squeeze(G))
        Y_given_F = tfd.Normal(mu, sigma)
        return Y_given_F

    def log_prob(self, F, G, Y):
        return self.Y_given_F(F, G).log_prob(Y) #was log_prob

    def conditional_mean(self, F, G):
        return self.Y_given_F(F, G).mean()

    def conditional_variance(self, F, G):
        return self.Y_given_F(F, G).variance()


# Load the data
D = np.loadtxt('mcycle.csv', delimiter=',')
X = D[:, 1:2]
Y = D[:, 2:]
N = X.shape[0]

# Store original data
x0 = X.copy()
y0 = Y.copy()

# Force into Nx1 format
# X = x[:, None].reshape(N,1)
# y = y[:, None].reshape(N,1)

# standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(Y)
X = X_scaler.transform(X)
Y = y_scaler.transform(Y)

# Kernel
kern_list = [gpflow.kernels.Matern32(variance=3., lengthscale=1.),
             gpflow.kernels.Matern32(variance=3., lengthscale=1.)]
kernel = gpflow.kernels.SeparateIndependent(kern_list)

# Inducing points (we hack this through SVGP, because the SpearateIndependent support in plain
# VGP was broken). We simply put an inducing point at every data point.
Xu = X.copy()
inducing_variables = gpflow.inducing_variables.mo_inducing_variables.\
    SharedIndependentInducingVariables(gpflow.inducing_variables.InducingPoints(Xu))

# The model
model = gpflow.models.SVGP(kernel=kernel, likelihood=HeteroscedasticGaussian(),
                           inducing_variable=inducing_variables, num_latent=2)

# Set trainable (everything except the 'inducing' points, because we want the full model).
set_trainable(model, True)
set_trainable(model.inducing_variable.inducing_variable_shared.Z,False)

# Print model
gpflow.utilities.print_summary(model)

# Optimize parameters
o = gpflow.optimizers.Scipy()
@tf.function(autograph=False)
def objective():
    return -model.elbo((X,Y))
o.minimize(objective, variables=model.trainable_variables)


# Set up plotting
def plot(m, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    mu, var = m.predict_y(X)
    ax.plot(X, mu)
    ax.plot(X, mu + tf.sqrt(var))
    ax.plot(X, mu - tf.sqrt(var))
    ax.plot(X, Y, '+')


# Plot model
plot(model)

# Print model
gpflow.utilities.print_summary(model)

# NLPD on training data
-tf.reduce_mean(model.predict_log_density((X, Y)))

# Load data
D = np.loadtxt('mcycle.csv', delimiter=',')
X = D[:, 1:2]
Y = D[:, 2:]
N = X.shape[0]

# Standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(Y)
Xall = X_scaler.transform(X)
Yall = y_scaler.transform(Y)

# Load cross-validation indices
cvind = np.loadtxt('cvind.csv').astype(int)

# 10-fold cross-validation setup
nt = np.floor(cvind.shape[0] / 10).astype(int)
cvind = np.reshape(cvind[:10 * nt], (10, nt))


# The model
def run_fold(X, Y, XT, YT):
    # Kernel
    kern_list = [gpflow.kernels.Matern32(variance=3., lengthscale=1.),
                 gpflow.kernels.Matern32(variance=3., lengthscale=1.)]
    kernel = gpflow.kernels.SeparateIndependent(kern_list)

    # Inducing points (we hack this through SVGP, because the SpearateIndependent support in plain
    # VGP was broken). We simply put an inducing point at every data point.
    Xu = X.copy()
    inducing_variables = gpflow.inducing_variables.mo_inducing_variables. \
        SharedIndependentInducingVariables(gpflow.inducing_variables.InducingPoints(Xu))

    # The model
    model = gpflow.models.SVGP(kernel=kernel, likelihood=HeteroscedasticGaussian(),
                               inducing_variable=inducing_variables, num_latent=2)

    # Set trainable (everything except the 'inducing' points, because we want the full model).
    set_trainable(model, True)
    set_trainable(model.inducing_variable.inducing_variable_shared.Z, False)

    # Optimize parameters
    o = gpflow.optimizers.Scipy()

    @tf.function(autograph=False)
    def objective():
        return -model.elbo((X, Y))

    o.minimize(objective, variables=model.trainable_variables)

    # Plot model
    plot(model)

    # Print model
    gpflow.utilities.print_summary(model, fmt='notebook')

    # Return NLPD
    return -tf.reduce_mean(model.predict_log_density((XT, YT))).numpy()


# Array of NLPDs
nlpd = []

# Run for each fold
for fold in np.arange(10):
    # Get training and test indices
    test = cvind[fold, :]
    train = np.setdiff1d(cvind, test)

    # Set training and test data
    X = Xall[train, :]
    Y = Yall[train, :]
    XT = Xall[test, :]
    YT = Yall[test, :]

    # . Run results
    res = run_fold(X, Y, XT, YT)
    nlpd.append(res)
    print(res)


print(np.mean(nlpd))
print(np.std(nlpd))
