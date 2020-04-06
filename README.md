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
 - [ ] EKF - extended Kalman filtering
 - [ ] UKF - unscented Kalman filtering
 - [ ] GHKF - Gauss-Hermite Kalman filtering
 - [ ] SLF - statistical linearisation filter
 - [ ] EKS - extended Kalman smoothing
 - [ ] UKS - unscented Kalman smoothing
 - [ ] GHKS - Gauss-Hermite Kalman smoothing
 - [ ] EK-EP - Extended Kalman EP
 - [ ] UK-EP - Unscented Kalman EP
 - [x] GHK-EP - Gauss-Hermite Kalman EP
 - [ ] PL - posterior linearisation
 - [ ] CL - cavity linearisation (new)
 - [ ] VI - variational inference
 - [ ] CVI - conjugate-computation variational inference
 - [ ] S2VI - doubly-sparse variational inference
 - [ ] S2EP - doubly-sparse expectation propagation (new)
 - [ ] STVI - spatio-temporal variational inference (new)
 - [ ] STEP - spatio-temporal expectation propagation (new)
