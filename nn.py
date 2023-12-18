#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 13/12/2023
"""

import jax
import jax.numpy as jnp
from featurize import featurize
from jax import vmap, jit, grad
from projection import project_force
from functools import partial

def energy(params, feat, nonlinearity=jax.nn.tanh):
    """Function to perform a forward pass through MLP"""
    hidden_layers = params[:-1]

    activation = feat
    for w, b in hidden_layers:
        activation = nonlinearity(jnp.dot(w, activation) + b)

    w_last, b_last = params[-1]
    return jnp.dot(w_last, activation) # + b_last


def predict_energy(params, x):
    """Function that predicts energy using forward pass through NN with features as input"""
    ## Featurize coordinates
    feat = featurize(x)
    ## Forward pass through network
    energy_cg = energy(params, feat)[0]
    ## Return CG energy
    return energy_cg

def predict_force(params, x, f_proj, div):
    ## Calculate force as a (minus) derivative of energy
    f_cg = - grad(predict_energy, argnums=1)(params, x)
    ## Project force
    f_cv = project_force(f_cg, f_proj, div)
    ## Return projected force
    return f_cv

def init_MLP(layer_widths, parent_key, scale=0.1):
    """Function to initialize MLP given layer widths"""
    params = []
    keys = jax.random.split(parent_key, num=len(layer_widths)-1)

    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = jax.random.split(key)
        params.append([
                       scale*jax.random.normal(weight_key, shape=(out_width, in_width)),
                       scale*jax.random.normal(bias_key, shape=(out_width,))
                       ]
        )

    return params

def l2_normsq(params, x):
    leaves, _ = jax.tree_util.tree_flatten(params)
    return sum([4 * jnp.sum(leaf ** 2) * x[0] for leaf in leaves])

batched_predict_force = vmap(predict_force, in_axes=(None, 0, 0, 0))

def loss_fn(params, x, f, f_proj, div):
    """Function to calculate MSE loss"""
    f_pred = batched_predict_force(params, x, f_proj, div)
    return jnp.mean((f_pred  - f) ** 2)


def weighted_loss_fn(params, x, f, f_proj, div, wts):
    """Function to calculate MSE loss"""
    f_pred = batched_predict_force(params, x, f_proj, div)
    return jnp.mean(wts * (f_pred  - f) ** 2)

@jit
def update(param, x, y, f_proj, div, lr):
    """Function to update parameters of NN based on gradient steps"""
    loss, grad_loss = jax.value_and_grad(loss_fn)(param, x, y, f_proj, div)
    return loss, jax.tree_util.tree_map(lambda x, g: x - g * lr, param, grad_loss), grad_loss

# @partial(jit, static_argnums=(0,))
# @jit
def weighted_update(param, x, y, f_proj, div, lr, wts):
    """Function to update parameters of NN based on gradient steps"""
    loss, grad_loss = jax.value_and_grad(weighted_loss_fn)(param, x, y, f_proj, div, wts)
    return loss, jax.tree_util.tree_map(lambda x, g: x - g * lr, param, grad_loss), grad_loss

## Section with CG forces

def predict_force_with_cg(params, x, f_proj, div):
    ## Calculate force as a (minus) derivative of energy
    f_cg = - grad(predict_energy, argnums=1)(params, x)
    ## Project force
    f_cv = project_force(f_cg, f_proj, div)
    ## Return projected force
    return f_cg, f_cv

batched_predict_force_with_cg = vmap(predict_force_with_cg, in_axes=(None, 0, 0, 0))

def weighted_loss_fn_with_cg(params, x, f_cg, f_cv, f_proj, div, cv_wts, cg_cv_wts):
    """Function to calculate MSE loss"""
    f_cg_pred, f_cv_pred = batched_predict_force_with_cg(params, x, f_proj, div)
    return cg_cv_wts[0] * jnp.mean((f_cg_pred  - f_cg) ** 2) + cg_cv_wts[1] * jnp.mean(cv_wts * (f_cv_pred  - f_cv) ** 2)

def weighted_update_with_cg(param, x, f_cg, y, f_proj, div, lr, wts, cg_cv_wts):
    """Function to update parameters of NN based on gradient steps"""
    loss, grad_loss = jax.value_and_grad(weighted_loss_fn_with_cg)(param, x, f_cg, y, f_proj, div, wts, cg_cv_wts)
    return loss, jax.tree_util.tree_map(lambda x, g: x - g * lr, param, grad_loss), grad_loss