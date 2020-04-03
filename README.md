# kalman-jax
Approximate inference in Markov Gaussian processes using iterated Kalman filtering and smoothing, in JAX 

This project aims to implement an XLA JIT compilable framework for inference in (non-conjugate) Markov Gaussian processes, with autodiff using [JAX](https://github.com/google/jax)

We aim to implement a full suite of approximate inference algorithms, all of which will call the same Kalman filter and smoother methods. Inference methods will be distinguished by the way in which the approximate likelihood terms are computed.

# TODO
 - [x] ADF
 - [ ] PEP
 - [ ] EKF
 - [ ] UKF
 - [ ] GHKF
 - [ ] EKS
 - [ ] UKS
 - [ ] GHKS
 - [ ] PL
 - [ ] CL
 - [ ] VI
 - [ ] CVI
 - [ ] S2VI
 - [ ] S2EP
 
