from jax import value_and_grad, grad
import time


def func1(x, model_parameter):
    model_parameter += x
    y = model_parameter * x**2
    return y, (model_parameter, model_parameter+1.0)


def func2(x, model_parameter):
    model_parameter += x
    y = model_parameter * x**2
    return y


param = 1.0

t0 = time.time()
f_val2, f_grad2 = value_and_grad(func2, argnums=0)(10., param)
print(f_val2)
print(f_grad2)
t1 = time.time()
print(t1-t0)

t0 = time.time()
f_val2, f_grad2 = value_and_grad(func2, argnums=0)(10., param)
print(f_val2)
print(f_grad2)
t1 = time.time()
print(t1-t0)

t0 = time.time()
# (f_val1, (param1, param2)), f_grad1 = value_and_grad(func1, argnums=0, has_aux=True)(10., param)
A, (B, C) = grad(func1, argnums=0, has_aux=True)(10., param)
print(A)
print(B)
print(C)
# print(f_val1)
# print(f_grad1)
t1 = time.time()
print(t1-t0)


