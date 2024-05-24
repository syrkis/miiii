# tasks.py
#  miiii tasks
# by: Noah Syrkis

# imports
from jax import random


# functions
def task_fn(data):
    pass


def conrad(data):
    pass


if __name__ == "__main__":
    from data import conrad_fn

    rng = random.PRNGKey(0)
    data, c2i, i2c = conrad_fn(rng, 32)
    print(data)
