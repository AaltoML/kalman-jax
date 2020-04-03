from typing import Any, NamedTuple
import jax.numpy as jnp
from jax import jit, partial, value_and_grad


class MyJaxObject(NamedTuple):
    parameters: Any = [jnp.ones(1), ]


class MyModel(object):
    def __init__(self):
        self.foo = 1
        self.param_obj = MyJaxObject()
        print(self.param_obj)
        print(self.param_obj.parameters)
    # parameters: Any = [jnp.ones(1), ]

    @partial(jit, static_argnums=0)
    def loss(self):
        return self.param_obj.parameters[0][0] ** 2


# myobj = MyJaxObject()
myobj = MyModel()


print(myobj)

loss_value = myobj.loss()
print(loss_value)

loss_and_grads_fn = value_and_grad(myobj.loss)
print(loss_and_grads_fn(myobj))
