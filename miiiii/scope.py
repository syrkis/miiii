# # scope.py
# #   neuralnetscopy codebase for visualizing neural networks
# # by: Noah Syrkis

# # imports
# import jax
# from jax import random, grad, jit, value_and_grad, tree_util
# import optax
# import pickle


# # functions
# def scope_fn(params):
#     flat_params, tree = tree_util.tree_flatten(params)
#     print(tree)


# # main
# if __name__ == "__main__":
#     from utils import load_params

#     params = load_params("output/model.pkl")
#     scope_fn(params)
