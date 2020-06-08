# kalman-jax
Approximate inference for Markov Gaussian processes using iterated Kalman filtering and smoothing.

This project aims to implement an XLA JIT compilable framework for inference in (non-conjugate) Markov Gaussian processes, with autodiff using [JAX](https://github.com/google/jax).

### We combine two recent advances in the field of probabilistic machine learning:
 - Development of state space methods for linear-time approximate inference in Gaussian processes
 - The ability to JIT compile and autodiff through loops efficiently with JAX

### Code structure
Each approximate inference algorithm will call the same underlying Kalman filter and smoother methods, and will be distinguished by the way in which the approximate likelihood terms are computed.

### Approximate inference algorithms
 - [x] ADF - assumed density filtering (single-sweep EP)
 - [x] PEP - power expectation propagation
 - [x] EKF - extended Kalman filtering
 - [ ] UKF - unscented Kalman filtering
 - [x] GHKF - Gauss-Hermite Kalman filtering
 - [x] SLF - statistical linearisation filter
 - [x] EKS - extended Kalman smoothing
 - [ ] UKS - unscented Kalman smoothing
 - [x] GHKS - Gauss-Hermite Kalman smoothing
 - [x] EK-EP - Extended Kalman EP
 - [ ] UK-EP - Unscented Kalman EP
 - [x] GHK-EP - Gauss-Hermite Kalman EP
 - [x] PL - posterior linearisation
 - [x] CL - cavity linearisation
 - [x] CVI - conjugate-computation variational inference
 - [ ] STVI - spatio-temporal variational inference
 - [ ] STEP - spatio-temporal expectation propagation

### Likelihoods
- [x] Gaussian
- [x] Probit
- [x] Poisson
- [ ] Logit

### Priors
- [x] Matern class
- [x] RBF
- [x] Cosine
- [x] Periodic
- [x] Quasi-periodic
- [x] Subband
- [x] Sum
- [ ] Product
- [ ] Latent force models (linear)

### TODO:
- come up with difficult (bimodal) example to compare the different algorithms
- allow user to feed in test locations during prediction
- extend to multi-output/multi-dimensional case
- make predict() method automatically return test NLPD
- make it so that when a filtering method is chosen, run() only runs the forward filter
- implement getters and setters for parameters that involve softplus mapping
- PL marginal likelihood approximation
- sort out intialisation of likelihood model
