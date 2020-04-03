# kalman-jax
Approximate inference in Markov Gaussian processes using iterated Kalman filtering and smoothing.

This project aims to implement an XLA JIT compilable framework for inference in (non-conjugate) Markov Gaussian processes, with autodiff using [JAX](https://github.com/google/jax).

We aim to implement a full suite of approximate inference algorithms, all of which will call the same underlying Kalman filter and smoother methods. Inference methods will be distinguished by the way in which the approximate likelihood terms are computed.

# TODO
 - [x] ADF - assumed density filtering (single-sweep EP)
 - [ ] PEP - power expectation propagation
 - [ ] EKF - extended Kalman filtering
 - [ ] UKF - unscented Kalman filtering
 - [ ] GHKF - Gauss-Hermite Kalman filtering
 - [ ] SLF - statistical linearisation filter
 - [ ] EKS - extended Kalman smoothing
 - [ ] UKS - unscented Kalman smoothing
 - [ ] GHKS - Gauss-Hermite Kalman smoothing
 - [ ] PL - posterior linearisation
 - [ ] CL - cavity linearisation (new)
 - [ ] VI - variational inference
 - [ ] CVI - conjugate-computation variational inference
 - [ ] S2VI - doubly-sparse variational inference
 - [ ] S2EP - doubly-sparse expectation propagation
 - [ ] STVI - spatio-temproal variational inference
 - [ ] STEP - spatio-temporal expectatiomn propagation
