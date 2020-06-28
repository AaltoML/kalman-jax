import pylab as plt
from autograd import numpy as np
from autograd import grad
import time
import numpy as numpy
import pylab as plt
import seaborn as snb
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

#####################################################################################################
# Plot util
#####################################################################################################
def plot_summary(x, s, interval=95, num_samples=100, sample_color='k', sample_alpha=0.4, interval_alpha=0.25, color='r',
                 legend=True, title="", plot_mean=True, plot_median=False, label=""):
    b = 0.5 * (100 - interval)

    lower = np.percentile(s, b, axis=0).T
    upper = np.percentile(s, 100 - b, axis=0).T

    if plot_median:
        median = np.percentile(s, [50], axis=0).T
        lab = 'Median'
        if len(label) > 0:
            lab += " %s" % label
        plt.plot(x.ravel(), median, label=lab, color=color, linewidth=4)

    if plot_mean:
        mean = np.mean(s, axis=0).T
        lab = 'Mean'
        if len(label) > 0:
            lab += " %s" % label
        plt.plot(x.ravel(), mean, '--', label=lab, color=color, linewidth=4)
    plt.fill_between(x.ravel(), lower.ravel(), upper.ravel(), color=color, alpha=interval_alpha,
                     label='%d%% Interval' % interval)

    if num_samples > 0:
        idx_samples = np.random.choice(range(len(s)), size=num_samples, replace=False)
        plt.plot(x, s[idx_samples, :].T, color=sample_color, alpha=sample_alpha);

    if legend:
        plt.legend(loc='best')

    if len(title) > 0:
        plt.title(title, fontweight='bold')


np.set_printoptions(precision=3)

#####################################################################################################
# Load data
#####################################################################################################
data = pd.read_csv('./mcycle.csv')
x, y = data['times '], data['accel']
N = len(x)

# store original data
x0 = x.copy()
y0 = y.copy()

# force into Nx1 format
X = x[:, None]
y = y[:, None]

# standardize
X_scaler = StandardScaler().fit(X)
y_scaler = StandardScaler().fit(y)
Xall = X_scaler.transform(X)
yall = y_scaler.transform(y)


np.random.seed(0)

#####################################################################################################
# Kernel
#####################################################################################################

matern32 = lambda d, rho: (1 + np.sqrt(3)*d/rho)*np.exp(-np.sqrt(3)*d/rho) 

def Matern32(X, X2, scale):

    # compute pairwise dists using code from GPy
    X1sq = np.sum(np.square(X),1)
    X2sq = np.sum(np.square(X2),1)
    r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
    dists = np.sqrt(r2)

    return matern32(dists, scale)



#####################################################################################################
# Numerical integration
#####################################################################################################

def integrate(W, A, B):

    # do a 1-D integral over every row
    I = np.zeros(len(B))
    for i in range(len(B)):
        I[i] = np.trapz(W[i, :], A)

    # then an integral over the result
    return np.trapz(I, B)


def compute_moments_old(log_fun, m1_cav, v1_cav, m2_cav, v2_cav, num_points=250):

    # determine limits for integration
    l1 = m1_cav - 10 * np.sqrt(v1_cav)
    u1 = m1_cav + 10 * np.sqrt(v1_cav)
    l2 = m2_cav - 10 * np.sqrt(v2_cav)
    u2 = m2_cav + 10 * np.sqrt(v2_cav)

    # create grid
    A, B = np.linspace(l1, u1, num_points), np.linspace(l2, u2, num_points)
    AA, BB = np.meshgrid(A, B)

    # eval integrand
    W = np.exp(log_fun(AA, BB))
    W11 = AA*W
    W12 = AA**2*W
    W21 = BB*W
    W22 = BB**2*W

    # integrate
    Zn = integrate(W, A, B)
    m11n = integrate(W11, A, B) / Zn
    m12n = integrate(W12, A, B) / Zn
    m21n = integrate(W21, A, B) / Zn
    m22n = integrate(W22, A, B) / Zn

    # compute variance from moments
    v1 = m12n - m11n ** 2
    v2 = m22n - m21n ** 2

    return Zn, m11n, v1, m21n, v2



def compute_moments(log_fun, m1_cav, v1_cav, m2_cav, v2_cav, num_points=20):

    # Gauss-Hermite
    x, w = numpy.polynomial.hermite.hermgauss(num_points)

    x1 = np.sqrt(2*v1_cav)*x + m1_cav
    x2 = np.sqrt(2*v2_cav)*x + m2_cav

    # Compute as nested integrals
    def gauss_hermite_2d(integrand_fun, x1, x2):
        return np.sum(w*np.sum(w*integrand_fun(x1, x2[:, None]), axis=1))/np.pi
        
    fun = lambda x1, x2: np.exp(log_fun(x1, x2))

    Zn = gauss_hermite_2d(fun, x1, x2) 
    m11n = gauss_hermite_2d(lambda x, y: x*fun(x, y), x1, x2)/Zn
    m12n = gauss_hermite_2d(lambda x, y: x**2*fun(x, y), x1, x2)/Zn
    m21n = gauss_hermite_2d(lambda x, y: y*fun(x, y), x1, x2)/Zn
    m22n = gauss_hermite_2d(lambda x, y: y**2*fun(x, y), x1, x2)/Zn

    # compute variance from moments
    v1 = m12n - m11n ** 2
    v2 = m22n - m21n ** 2

    return Zn, m11n, v1, m21n, v2


def compute_moments_hsced(log_fun, yn, m1_cav, v1_cav, m2_cav, v2_cav, num_points=20):
    # Gauss-Hermite
    x, w = numpy.polynomial.hermite.hermgauss(num_points)

    link = lambda x: 1e-10 + np.log(1 + np.exp(x))

    x2 = np.sqrt(2 * v2_cav) * x + m2_cav

    Ny = npdf(yn, m1_cav, link(x2) + v1_cav)

    Zn = np.sum(w * Ny)/np.sqrt(np.pi)

    m11_integrand = (link(x2)**-1 + v1_cav**-1)**-1 * (link(x2)**-1 * yn + v1_cav**-1 * m1_cav)
    m11n = np.sum(w * Ny * m11_integrand)/Zn/np.sqrt(np.pi)

    m12_integrand = (link(x2) ** -1 + v1_cav ** -1) ** -1 + m11_integrand**2
    m12n = np.sum(w * Ny * m12_integrand)/Zn/np.sqrt(np.pi)

    m21n = np.sum(w * x2 * Ny)/Zn/np.sqrt(np.pi)
    m22n = np.sum(w * x2**2 * Ny)/Zn/np.sqrt(np.pi)

    # compute variance from moments
    v1 = m12n - m11n ** 2
    v2 = m22n - m21n ** 2

    return Zn, m11n, v1, m21n, v2

#####################################################################################################
# Hyperparameters and settings
#####################################################################################################

var1 = 3.
var2 = 3.
scale1 = 1.
scale2 = 1.

# EP settings
damping = 0.8
max_itt = 50
tol = 1e-6

# link and site fun
#link = lambda x: np.exp(x)
link = lambda x: 1e-10 + np.log(1 + np.exp(x))
log_site_fun = lambda yn, f1, f2: log_npdf(yn, f1, link(f2))

# number of points for numerical integration
num_points = 50

# verbose
verbose = False


#####################################################################################################
# EP
#####################################################################################################


def logZ_fun(m, V, jitter=1e-8):
    n = len(m)

    if V.ndim == 2:
        L = np.linalg.cholesky(V + 1e-8*np.identity(n))
        h = np.linalg.solve(L, m)
        diag_L = np.diag(L)
    elif V.ndim == 1:
        diag_L = np.sqrt(V)
        h = m/diag_L

    constant = 0.5*n*np.log(2*np.pi)
    quad = 0.5*h.T@h
    det = 0.5*np.sum(np.log(diag_L**2))

    return constant + quad + det



# compute posterior
def update_posterior(nu, tau, K):
    s_sqrt = np.diag(np.sqrt(tau))
    B = np.identity(len(nu)) + s_sqrt @ K @ s_sqrt
    L = np.linalg.cholesky(B)
    V = np.linalg.solve(L, s_sqrt @ K)
    Sigma = K - V.T @ V
    mu = Sigma @ nu
    return mu, Sigma, L


def get_cavity(mu, Sigma, nu, tau, n=None):
    if n is None:
        nu_cav = mu / np.diag(Sigma) - nu
        tau_cav = 1. / np.diag(Sigma) - tau
    else:
        nu_cav = mu[n] / np.diag(Sigma)[n] - nu[n]
        tau_cav = 1. / np.diag(Sigma)[n] - tau[n]
    return nu_cav, tau_cav


# auxilary functions
npdf = lambda x, m, v: np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
log_npdf = lambda x, m, v: -0.5 * (x - m) ** 2 / v - 0.5 * np.log(2 * np.pi * v)



def run_ep(X, y, K1, K2, max_itt=20, verbose=False, return_full=False):

    N = len(y)


    # init site approximations
    nu1, nu2 = np.zeros(N), np.zeros(N)
    tau1, tau2 = 1e-10*np.ones(N), 1e-10*np.ones(N)

    # init global approximations
    mu1, Sigma1, L1 = update_posterior(nu1, tau1, K1)
    mu2, Sigma2, L2 = update_posterior(nu2, tau2, K2)

    # for now only check convergence based on means
    mu_joint = np.hstack((mu1, mu2))

    # Z for each site
    Zn = np.ones(N)


    # Iterate
    for itt in range(max_itt):

        mu_old = mu_joint.copy()

        for n in np.random.permutation(N):

            # compute cavity
            nu1_cav, tau1_cav = get_cavity(mu1, Sigma1, nu1, tau1, n)
            nu2_cav, tau2_cav = get_cavity(mu2, Sigma2, nu2, tau2, n)

            # transform to mean parameters
            m1_cav = nu1_cav / tau1_cav
            m2_cav = nu2_cav / tau2_cav
            v1_cav = 1. / tau1_cav
            v2_cav = 1. / tau2_cav

            # define site function
            log_tilde = lambda f1, f2: log_site_fun(y[n], f1, f2)

            # compute moments
            #Zn[n], m11, v1, m21, v2 = compute_moments(log_tilde, m1_cav, v1_cav, m2_cav, v2_cav, num_points=num_points)
            Zn[n], m11, v1, m21, v2 = compute_moments_hsced(log_tilde, y[n], m1_cav, v1_cav, m2_cav, v2_cav, num_points=num_points)

            # update sites
            if 1. / v1 - tau1_cav > 0 and 1. / v2 - tau2_cav > 0:

                tau1[n] = (1 - damping) * tau1[n] + damping * (1. / v1 - tau1_cav)
                nu1[n] = (1 - damping) * nu1[n] + damping * (m11 / v1 - nu1_cav)

                tau2[n] = (1 - damping) * tau2[n] + damping * (1. / v2 - tau2_cav)
                nu2[n] = (1 - damping) * nu2[n] + damping * (m21 / v2 - nu2_cav)

            else:
                if verbose:
                    print('\tSkipping update for n = %d due to negative variance' % (n))
                nu1[n], tau1[n] = 0, 1e-10
                nu2[n], tau2[n] = 0, 1e-10
                Zn[n] = 1.

        # update posteriors
        mu1, Sigma1, L1 = update_posterior(nu1, tau1, K1)
        mu2, Sigma2, L2 = update_posterior(nu2, tau2, K2)
        mu_joint = np.hstack((mu1, mu2))

        diff = np.max((mu_joint - mu_old) ** 2)
        if diff < tol:
            print('Converged in %d iterations with %3.2e' % (itt + 1, diff))
            break
        else:
            if verbose:
                print('Itt %-3d: diff = %3.2e' % (itt + 1, diff))
    print('Stopped after %d iterations with %3.2e' % (itt + 1, diff))

    

    if return_full:
        return mu1, Sigma1, mu2, Sigma2, Zn, nu1, tau1, nu2, tau2
    else:
        return mu1, Sigma1, mu2, Sigma2



def ep_log_marginal_likelihood(X, y, theta):

    # get parameters
    var1, var2, scale1, scale2 = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2]), np.exp(theta[3])

    # prep kernels
    K1 = var1*Matern32(X, X, scale1)
    K2 = var2*Matern32(X, X, scale2)

    # run EP
    mu1, Sigma1, mu2, Sigma2, Zn, nu1, tau1, nu2, tau2 = run_ep(X, y, K1, K2, max_itt=max_itt, return_full=True)

    # compute cavity
    nu1_cav, tau1_cav = get_cavity(mu1, Sigma1, nu1, tau1)
    nu2_cav, tau2_cav = get_cavity(mu2, Sigma2, nu2, tau2)

    # transform to mean parameters
    m1_cav = nu1_cav / tau1_cav
    m2_cav = nu2_cav / tau2_cav
    v1_cav = 1. / tau1_cav
    v2_cav = 1. / tau2_cav

    m1 = nu1/tau1
    m2 = nu2/tau2
    v1 = 1./tau1
    v2 = 1./tau2

    logZ_sites = np.sum(np.log(Zn))

    def compute_ml(theta):

        var1, var2, scale1, scale2 = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2]), np.exp(theta[3])

        # re-compute kernels and posteriors for autograd to capture dependecies on the hyperparams.
        K1 = var1*Matern32(X, X, scale1)
        K2 = var2*Matern32(X, X, scale2)
        mu1, Sigma1, L1 = update_posterior(nu1, tau1, K1)
        mu2, Sigma2, L2 = update_posterior(nu2, tau2, K2)

        # marginal likelihood
        logZ = logZ_fun(mu1, Sigma1) + logZ_fun(mu2, Sigma2) 
        logZ += logZ_fun(m1_cav, v1_cav) - logZ_fun(m1, v1)
        logZ += logZ_fun(m2_cav, v2_cav) - logZ_fun(m2, v2)
        logZ += -logZ_fun(np.zeros(len(mu1)), K1) - logZ_fun(np.zeros(len(mu2)), K2) 
        logZ += logZ_sites

        return logZ

    logZ = compute_ml(theta)
    grad_fun = grad(compute_ml)
    grad_vec = grad_fun(theta)

    return logZ, grad_vec





#####################################################################################################
# Predict
#####################################################################################################



def optimimize_hyper(X, y, max_em_itt=10, learning_rate=1e-3, verbose=True):

    # Use same initial valeus as in the VGP notebook
    var1 = 3
    scale1 = 1
    var2 = 3
    scale2 = 1

    theta_unconstrained = np.array([var1, var2, scale1, scale2])
    theta = np.log(theta_unconstrained)

    # prep objective function
    def objective(theta):
        val, grad = ep_log_marginal_likelihood(X, y, theta)
        return -val, -grad

    if verbose:
        print(100*'-')
        print('Optimizing hyperparameters')
        print(100*'-')

    t0 = time.time()
    for itt in range(max_em_itt):

        # compute logZ and gradients
        neg_logZ, grad_vec = objective(theta)

        # take step
        theta = theta - learning_rate*grad_vec

        print('%-3d: log Z = %3.2f, |grad| = %3.2f' % (itt+1, -neg_logZ, np.linalg.norm(grad_vec)))
        print(np.exp(theta))
        print('\n')

    t1 = time.time()
    if verbose:
        print('Optimization done in %3.2fs' % (t1-t0))


    # build new kernel
    var1, var2, scale1, scale2 = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2]), np.exp(theta[3])
    return var1, var2, scale1, scale2


############################################################################################################################
# Pipeline
############################################################################################################################
def run_fold(X, y, Xt, yt):

    var1, var2, scale1, scale2 = optimimize_hyper(X, y, max_em_itt=20, learning_rate=1e-3, verbose=True)

    K1 = var1*Matern32(X, X, scale1)
    K2 = var2*Matern32(X, X, scale2)

    print('\n')
    print('Var1: %3.2f' % var1)
    print('Var2: %3.2f' % var2)
    print('Scale1: %3.2f' % scale1)
    print('Scale2: %3.2f' % scale2)
    print('\n')

    print(100*'-')
    print('Predicting...')
    print(100*'-')

    mu1, Sigma1, mu2, Sigma2 = run_ep(X, y, K1, K2, max_itt=max_itt)

    Ktf1 = var1*Matern32(Xt, X, scale1)
    Ktf2 = var2*Matern32(Xt, X, scale2)
    Ktt1 = var1*Matern32(Xt, Xt, scale1)
    Ktt2 = var2*Matern32(Xt, Xt, scale2)

    L1 = np.linalg.cholesky(K1 + 1e-8*np.identity(len(X)))
    L2 = np.linalg.cholesky(K2 + 1e-8*np.identity(len(X)))

    g1 = np.linalg.solve(L1, Ktf1.T)
    g2 = np.linalg.solve(L2, Ktf2.T)
    h1 = np.linalg.solve(L1.T, g1)
    h2 = np.linalg.solve(L2.T, g2)

    ft1_mean = g1.T@np.linalg.solve(L1, mu1)
    ft2_mean = g2.T@np.linalg.solve(L2, mu2)
    ft1_cov = Ktt1 - g1.T@g1 + h1.T@Sigma1@h1
    ft2_cov = Ktt2 - g2.T@g2 + h2.T@Sigma2@h2


    print(100*'-')
    print('Computing NLPD')
    print(100*'-')

    # compute log predictive density for ytest set
    nlpds = []
    for i in range(len(Xt)):
        Zn, m11, v1, m21, v2 = compute_moments_hsced([], yt[i], ft1_mean[i], ft1_cov[i, i], ft2_mean[i], ft2_cov[i, i], num_points=num_points)
        nlpds.append(-np.log(Zn))

    return np.mean(nlpds)






############################################################################################################################
# Run cross-validation
############################################################################################################################

# Load cross-validation indices
cvind = np.loadtxt('cvind.csv').astype(int)

# 10-fold cross-validation setup
nt = np.floor(cvind.shape[0]/10).astype(int)
cvind = np.reshape(cvind[:10*nt],(10,nt))


# Array of NLPDs
nlpd = []

# Run for each fold
for fold in np.arange(10):

    print(100*'-')
    print('Fold %d' % fold)
    print(100*'-')
    
    # Get training and test indices
    test = cvind[fold,:]
    train = np.setdiff1d(cvind,test)

    # Set training and test data
    X = Xall[train,:]
    y = yall[train,:]
    Xt = Xall[test,:]
    yt = yall[test,:]
    
    #. Run results
    res = run_fold(X,y,Xt,yt)
    nlpd.append(res)
    print(res)


print(np.mean(nlpd), np.std(nlpd))


