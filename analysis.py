#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 18/12/2023
"""

import numpy as np 
import jax
import matplotlib.pyplot as plt
from jax import vmap, jit
from nn import batched_predict_force
from featurize import featurize
from tqdm import tqdm

def make_test_preds(params, testloader):
    ## Predict forces for test set based on coordinates
    f_pred_all = []
    f_label_all = []
    label_feat = []
    calc_feat = []

    ## Create jitted version of predict function
    partial_pred = jax.tree_util.Partial(batched_predict_force, params=params)
    jitted_pred = jit(batched_predict_force)

    ## Loop over all batches of the testloader
    for i, (x, feat, _, f_proj, div, f) in tqdm(enumerate(testloader), total=len(testloader)):
        ## Predict force for entire batch
        f_pred_projected = jitted_pred(params, x, f_proj, div)
        f_pred_all.append(f_pred_projected)
        f_label_all.append(f)
        label_feat.append(feat)
        ## Calculate features for entire batch
        calc_feat.append(vmap(featurize)(x))

    return np.concatenate(f_pred_all), \
            np.concatenate(f_label_all), \
            np.concatenate(label_feat), \
            np.concatenate(calc_feat)


def make_label_pred_plot(ax, feat, labels, preds, plot_stride=1):
    ax.plot(
        feat[::plot_stride], labels[::plot_stride], 
        'o', markersize=1, 
        label="Labels"
        )
    
    ax.plot(
        feat[::plot_stride], preds[::plot_stride],
        'o', markersize=2, 
        label="Predicted"
        )
    
    return ax