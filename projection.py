#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 13/12/2023
"""

import jax.numpy as jnp
from jax import jacrev, jacfwd
from jax import random
from featurize import featurize

def calc_force_proj_operator(x, torsions_cg_idx, dists_cg_idx):
    """ Given x, calculate G_{\alpha, \gamma} * \grad{q_\gamma} """

    dqdx, Ginv = calc_inv_G_and_grad_q(x, torsions_cg_idx, dists_cg_idx)
    sum_gamma_G = []
    ## For each alpha
    count = 0
    for alpha in range(dqdx.shape[0]):
        ## Calculate sum over gamma of G_inv * grad(q)
        count += 1
        sum_gamma_G.append(jnp.sum( Ginv[alpha, :, jnp.newaxis, jnp.newaxis] * dqdx[:, :], axis=0 ))
        
    return x, sum_gamma_G

def calc_G(xx, torsions_cg_idx, dists_cg_idx):
#     grad_feat = lambda xx: jacfwd(ala2_ics_cg(xx))
    curr_jac = jacfwd(featurize)(xx, torsions_cg_idx, dists_cg_idx)
    return curr_jac, jnp.sum(jnp.expand_dims(curr_jac, 0) * jnp.expand_dims(curr_jac, 1), axis=(2, 3))

def calc_inv_G_and_grad_q(xx, torsions_cg_idx, dists_cg_idx):
    out_calc_G = calc_G(xx, torsions_cg_idx, dists_cg_idx)
    return out_calc_G[0], jnp.linalg.inv(out_calc_G[1])

def calc_div_operator(x, torsion_cg_idx, dist_cg_idx, nfeatures=12):
    """Function to calculate divergence of the sum over gamma of G * q for each
    data point
    
    """
    def func_sum_gamma_G(xx, i, torsion_cg_idx, dist_cg_idx):
        dqdx, Ginv = calc_inv_G_and_grad_q(xx, torsion_cg_idx, dist_cg_idx)
        return jnp.sum( Ginv[i, :, jnp.newaxis, jnp.newaxis] * dqdx[:, :], axis=0 )
    
    div_list = [] # jnp.zeros((12, 6, 3, 6, 3))
    
    for i in range(nfeatures):        
        ## Term 2 = div(sum_gamma_G)
        term2 = jacrev(func_sum_gamma_G)(x, i, torsion_cg_idx, dist_cg_idx)
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
