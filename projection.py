#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 13/12/2023
"""

import jax.numpy as jnp
from jax import jacrev, jacfwd
from jax import random
# from featurize import featurize

def calc_force_proj_operator(x, featurize):
    """ Given x, calculate G_{\alpha, \gamma} * \grad{q_\gamma} and |det G| """

    dqdx, Ginv, abs_det_G = calc_inv_G_and_grad_q(x, featurize)
    sum_gamma_G = []
    ## For each alpha
    count = 0
    for alpha in range(dqdx.shape[0]):
        ## Calculate sum over gamma of G_inv * grad(q)
        count += 1
        sum_gamma_G.append(jnp.sum( Ginv[alpha, :, jnp.newaxis, jnp.newaxis] * dqdx[:, :], axis=0 ))
        
    return x, sum_gamma_G, abs_det_G

def calc_G(xx, featurize):
    """Function that takes in a single coordinate vector, featurization function, and calculates dq/dx and G

    Parameters
    ----------
    xx
        Coordinate vector, shape (N_atoms, N_dim) [N_dim usually = 3]
    featurize
        Function that returns a jax.numpy array of features q, shape (N_features,)

    Returns
    -------
        1) Vector containing dq/dx, shape (N_features, N_atoms, N_dim)
        2) G(i,j) = (dq(i)/dx) . (dq(j)/dx), shape (N_features, N_features)
    """
    ## Calculate gradient of featurize function with respect coordinates xx
    curr_jac = jacfwd(featurize)(xx)
    ## Calculate G (this only works for multi-dimensional featurize, be careful with 1D featurization)
    G_matrix = jnp.sum(
        jnp.expand_dims(curr_jac, 0) * jnp.expand_dims(curr_jac, 1), 
        axis=(2, 3)
    )
    
    return curr_jac, G_matrix

def calc_inv_G_and_grad_q(xx, featurize):
    """Function that calculates gradient of features w.r.t. xx, inverse of matrix G, and det(G)

    Parameters
    ----------
    xx
        Coordinate vector, shape (N_atoms, N_dim) [N_dim usually = 3]
    featurize
        Function that returns a jax.numpy array of features q, shape (N_features,)

    Returns
    -------
    dqdx
        Vector containing dq/dx, shape (N_features, N_atoms, N_dim)
    G_inv
        G(i,j) = (dq(i)/dx) . (dq(j)/dx), shape (N_features, N_features)
        G_inv = G^-1
    abs_det_G
        |det G|
    """
    ## Compute gradient of q and matrix G
    grad_q, G_matrix = calc_G(xx, featurize)
    ## Compute inverse of G
    G_inv = jnp.linalg.inv(G_matrix)
    ## Compute absolute value of determinant of G (|det G|)
    abs_det_G = jnp.linalg.det(G_matrix)

    return grad_q, G_inv, abs_det_G

def calc_div_operator(x, featurize, nfeatures):
    """Function to calculate divergence of the sum over gamma of G * q for each
    data point
    
    """
    def func_sum_gamma_G(xx, i, featurize):
        dqdx, Ginv, _ = calc_inv_G_and_grad_q(xx, featurize)
        return jnp.sum( Ginv[i, :, jnp.newaxis, jnp.newaxis] * dqdx[:, :], axis=0 )
    
    div_list = [] 
    
    for i in range(nfeatures):        
        ## Term 2 = div(sum_gamma_G)
        term2 = jacrev(func_sum_gamma_G)(x, i, featurize)
        div_list.append(term2)
        
    return div_list

def project_force(force, proj_operator, div, beta=1.6776):
    ## Return projected force
    return jnp.sum(proj_operator * force, axis=(1,2)) + div * beta**(-1) 

def find_divergence(arr):
    """Function to manually calculate divergence of an [n, m, n, m] matrix"""
    div = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            div += arr[i, j][i, j]
    return div
