#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 13/12/2023
"""

import numpy as np
import jax.numpy as jnp

def ala2_ics(x, torsions_idx, dists_idx):
    """ Computes all internal coordinates (torsions + distances) differentiably """
    # Torsions
    torsion_atoms = x[torsions_idx, :]
    torsions = all_torsions(torsion_atoms)
    # Distances
    dist_atoms = x[dists_idx, :]
    dists = all_dists(dist_atoms)
    return jnp.concatenate((jnp.array(torsions), jnp.array(dists)))

def featurize(
        x, 
        torsion_cg_idx=np.array([[0,1,2,4], [1,2,4,5]]), 
        dists_cg_idx=np.array([[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [0, 2], [1, 3], [1, 4], [3, 4], [2, 5]])
        ):
    return ala2_ics(x, torsion_cg_idx, dists_cg_idx)

def torsion(p, cossin=False):
#     print(p)
#     print(p.shape)
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1/jnp.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - jnp.dot(b0, b1)*b1
    w = b2 - jnp.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)
    return jnp.arctan2(y, x)

def all_torsions(p):
    tlist = []
    for p_i in p:
        tlist.append(torsion(p_i))
    
    return tlist

def dist(x):
    """Function to calculate distance in JAX given coordinates x1 and x2"""
    d = x[0]-x[1]
    d2 = jnp.sum(d*d)
    return jnp.sqrt(d2)

def all_dists(p):
    dlist = []
    for p_i in p:
        dlist.append(dist(p_i))
        
    return dlist
